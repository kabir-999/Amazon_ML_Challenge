"""version_2: test-density training. Stage 1 = out-of-fold LightGBM probabilities; stage 2 = probability-context re-ranker
(LightGBM + XGBoost + MLP ensemble). All S1 of the train set are used against the FULL train pool (same density as test)."""
import numpy as np, pandas as pd, time, json, sys, torch, torch.nn as nn, lightgbm as lgb, xgboost as xgb
from config import *
from features_v2 import FEATS2, S2CTX, context_v2, assemble2, stage2_context
from common_v2 import *

TAG, K, SAMP = "trainfull", 4, 0.2
t0 = time.time(); rng = np.random.default_rng(SEED)
D = {}; off1 = offp = 0
for c in ("India", "US"):
    d = load_country(TAG, c); d["off1"], d["offp"] = off1, offp; off1 += len(d["s1"]); offp += len(d["pool"]); D[c] = d
    print(c, "S1", len(d["s1"]), "pairs", len(d["cand"]), "recall-ceiling pos", int(d["y"].sum()), flush=True)
TRUE = truth_sets(D, TAG); n1 = off1
fold_of = rng.integers(0, K, n1); split = np.empty(n1, int); perm = rng.permutation(n1)
split[perm[:int(.7 * n1)]] = 0; split[perm[int(.7 * n1):int(.85 * n1)]] = 1; split[perm[int(.85 * n1):]] = 2
for c, d in D.items():
    d["ctx"] = context_v2(d["F"], d["cand"], d["s1"], d["pool"])
    d["g1"] = d["cand"].i1.values + d["off1"]; d["gp"] = d["cand"].ip.values + d["offp"]
    d["fold"] = fold_of[d["g1"]]; d["split"] = split[d["g1"]]
    print(c, "context done", f"{time.time()-t0:.0f}s", flush=True)

# ---------------- stage 1: OOF LightGBM ----------------
parts = []
for c, d in D.items():
    idx = np.flatnonzero(rng.random(len(d["cand"])) < SAMP)
    X = assemble2(d["F"], d["cand"], d["s1"], d["pool"], d["ctx"], idx); X["y"] = d["y"][idx]; X["fold"] = d["fold"][idx]; parts.append(X)
S = pd.concat(parts, ignore_index=True); del parts
print("stage-1 sample", S.shape, f"{time.time()-t0:.0f}s", flush=True)
P1 = dict(n_estimators=700, learning_rate=0.06, num_leaves=127, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, min_child_samples=50, verbose=-1, n_jobs=8)
fold_models = []
for k in range(K):
    m = lgb.LGBMClassifier(**P1).fit(S.loc[S.fold != k, FEATS2], S.y[S.fold != k]); fold_models.append(m)
    m.booster_.save_model(f"{OUT}/s1_fold{k}.txt")
    for c, d in D.items():
        rows = np.flatnonzero(d["fold"] == k)
        if "p1" not in d: d["p1"] = np.zeros(len(d["cand"]), np.float32)
        for lo in range(0, len(rows), 1_000_000):
            r = rows[lo:lo + 1_000_000]; d["p1"][r] = m.predict_proba(assemble2(d["F"], d["cand"], d["s1"], d["pool"], d["ctx"], r))[:, 1]
    print("stage-1 fold", k, f"{time.time()-t0:.0f}s", flush=True)
del S
for c, d in D.items():
    np.save(f"{WORK}/{TAG}/{c}_p1.npy", d["p1"])
    print(c, "stage-1 OOF AP", round(float(__import__('sklearn.metrics', fromlist=['x']).average_precision_score(d["y"], d["p1"])), 4), flush=True)

# ---------------- stage 2 ----------------
KEEP = 0.005
rows2 = []
for c, d in D.items():
    s2 = stage2_context(d["p1"], d["cand"], d["pool"]); idx = np.flatnonzero(d["p1"] >= KEEP)
    X = assemble2(d["F"], d["cand"], d["s1"], d["pool"], d["ctx"], idx)
    for k in S2CTX: X[k] = s2[k][idx]
    X["y"] = d["y"][idx]; X["g1"] = d["g1"][idx]; X["gp"] = d["gp"][idx]; X["split"] = d["split"][idx]; rows2.append(X)
    print(c, "stage-2 rows", len(idx), "of", len(d["cand"]), f"{time.time()-t0:.0f}s", flush=True)
X2 = pd.concat(rows2, ignore_index=True); del rows2
FE = FEATS2 + S2CTX
tr, va, te = (X2[X2.split == k] for k in (0, 1, 2))
print({k: len(v) for k, v in zip(("train", "val", "test"), (tr, va, te))}, "features", len(FE), flush=True)
va_ids, te_ids = np.flatnonzero(split == 1), np.flatnonzero(split == 2)

models = {}
m = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.05, num_leaves=127, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, min_child_samples=50, verbose=-1, n_jobs=8)
m.fit(tr[FE], tr.y, eval_set=[(va[FE], va.y)], callbacks=[lgb.early_stopping(50, verbose=False)])
models["lightgbm"] = (m.predict_proba(va[FE])[:, 1], m.predict_proba(te[FE])[:, 1]); print("lgb", m.best_iteration_, f"{time.time()-t0:.0f}s", flush=True)
m2 = xgb.XGBClassifier(n_estimators=1500, learning_rate=0.06, max_depth=9, subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=8, early_stopping_rounds=50, eval_metric="logloss")
m2.fit(tr[FE], tr.y, eval_set=[(va[FE], va.y)], verbose=False)
models["xgboost"] = (m2.predict_proba(va[FE])[:, 1], m2.predict_proba(te[FE])[:, 1]); print("xgb", f"{time.time()-t0:.0f}s", flush=True)
dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
mu, sd = tr[FE].mean(), tr[FE].std() + 1e-6
def T(df): return torch.tensor(((df[FE] - mu) / sd).values, dtype=torch.float32)
Xt, yt = T(tr).to(dev), torch.tensor(tr.y.values, dtype=torch.float32).to(dev)
net = nn.Sequential(nn.Linear(len(FE), 384), nn.ReLU(), nn.Dropout(.1), nn.Linear(384, 192), nn.ReLU(), nn.Dropout(.1), nn.Linear(192, 64), nn.ReLU(), nn.Linear(64, 1)).to(dev)
EP, BS = 10, 8192
opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4); lossf = nn.BCEWithLogitsLoss()
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-3, total_steps=EP * ((len(Xt) + BS - 1) // BS))
for ep in range(EP):
    net.train(); idx = torch.randperm(len(Xt), device=dev)
    for s in range(0, len(Xt), BS):
        b = idx[s:s + BS]; opt.zero_grad(); lossf(net(Xt[b]).squeeze(1), yt[b]).backward(); opt.step(); sched.step()
net.eval(); del Xt, yt
def mlp_pred(df):
    out = []
    with torch.no_grad():
        for s in range(0, len(df), 500000): out.append(torch.sigmoid(net(T(df.iloc[s:s + 500000]).to(dev)).squeeze(1)).cpu().numpy())
    return np.concatenate(out)
models["mlp"] = (mlp_pred(va), mlp_pred(te)); print("mlp", dev, f"{time.time()-t0:.0f}s", flush=True)
models["ensemble"] = tuple(np.mean([models[k][i] for k in ("lightgbm", "xgboost", "mlp")], axis=0) for i in (0, 1))

report = {}
def best_thr(pv):
    return max((f05(TRUE, decide(va.g1.values, va.gp.values, pv, t), va_ids), t) for t in np.arange(0.3, 0.98, 0.025))
for name, (pv, pt) in models.items():
    fv, thr = best_thr(pv)
    report[name] = {"val_f05": round(fv, 4), "thr": round(float(thr), 3), "test_f05": round(f05(TRUE, decide(te.g1.values, te.gp.values, pt, thr), te_ids), 4)}
    print(name, report[name], flush=True)
# stage-1 only baseline (same S1 split) for comparison
p1_te = te.p1.values; p1_va = va.p1.values
fv, thr = max((f05(TRUE, decide(va.g1.values, va.gp.values, p1_va, t), va_ids), t) for t in np.arange(0.3, 0.98, 0.025))
report["stage1_only"] = {"val_f05": round(fv, 4), "thr": round(float(thr), 3), "test_f05": round(f05(TRUE, decide(te.g1.values, te.gp.values, p1_te, thr), te_ids), 4)}
print("stage1_only", report["stage1_only"], flush=True)
thr = report["ensemble"]["thr"]; pred = decide(te.g1.values, te.gp.values, models["ensemble"][1], thr)
ctry = np.concatenate([[c] * len(D[c]["s1"]) for c in D])
for c in D: report[f"ensemble_{c}"] = round(f05(TRUE, pred, [i for i in te_ids if ctry[i] == c]), 4)
tp = sum(len(TRUE[i] & pred.get(i, set())) for i in te_ids); npred = sum(len(pred.get(i, ())) for i in te_ids); ntrue = sum(len(TRUE[i]) for i in te_ids)
report["ensemble_pair_precision"], report["ensemble_pair_recall"] = round(tp / npred, 4), round(tp / ntrue, 4)
print(report, flush=True)
json.dump(report, open(f"{OUT}/report_v2.json", "w"), indent=1)
m.booster_.save_model(f"{OUT}/s2_lgb.txt"); m2.save_model(f"{OUT}/s2_xgb.json"); torch.save(net.state_dict(), f"{OUT}/s2_mlp.pt")
pd.to_pickle({"mu": mu, "sd": sd, "FE": FE, "thr": thr, "K": K}, f"{OUT}/meta_v2.pkl")
