"""Train LightGBM / XGBoost / MLP (MPS GPU) on dev candidate pairs (India+US); tune F0.5 threshold on val; report held-out test."""
import pandas as pd, numpy as np, time, json, torch, torch.nn as nn, lightgbm as lgb, xgboost as xgb
from sklearn.metrics import average_precision_score
from config import *
from features import FEATS, context_arrays, assemble

D = f"{WORK}/dev"; parts, s1_all, off1, offp = [], [], 0, 0
gt = pd.read_parquet(f"{WORK}/gt.parquet"); TRUE = {}; ctry = []
gmap = dict(zip(gt.source1_entity_id, gt.matched_entity_ids))
for c in ("India", "US"):
    s1 = pd.read_parquet(f"{D}/{c}_s1n.parquet"); pool = pd.read_parquet(f"{D}/{c}_pooln.parquet"); cand = pd.read_parquet(f"{D}/{c}_cand.parquet")
    F = np.load(f"{D}/{c}_F.npy", mmap_mode="r"); y = np.load(f"{D}/{c}_y.npy")
    ctx = context_arrays(np.asarray(F), cand, s1, pool)
    X = assemble(F, cand, s1, pool, ctx, 0, len(cand)); X["y"] = y
    X["g1"] = cand.i1.values + off1; X["gp"] = cand.ip.values + offp; X["country"] = c
    pid = dict(zip(pool.entity_id, range(len(pool))))
    for k, e in enumerate(s1.entity_id): TRUE[k + off1] = {pid[b] + offp for b in gmap[e].split(",") if b}
    ctry += [c] * len(s1); parts.append(X); off1 += len(s1); offp += len(pool); print(c, len(X), flush=True)
X = pd.concat(parts, ignore_index=True); ctry = np.array(ctry); n1 = off1
rng = np.random.default_rng(SEED); perm = rng.permutation(n1); split = np.empty(n1, int)
split[perm[:int(.7 * n1)]] = 0; split[perm[int(.7 * n1):int(.85 * n1)]] = 1; split[perm[int(.85 * n1):]] = 2
sp = split[X.g1.values]; tr, va, te = X[sp == 0], X[sp == 1], X[sp == 2]
print({k: len(v) for k, v in zip(("train", "val", "test"), (tr, va, te))}, "features", len(FEATS), flush=True)

def f05(pred, ids):
    tot = 0.0
    for i in ids:
        t, p = TRUE[i], pred.get(i, set())
        if not t and not p: tot += 1; continue
        if not t or not p: continue
        tp = len(t & p)
        if tp: P, R = tp / len(p), tp / len(t); tot += 1.25 * P * R / (.25 * P + R)
    return tot / len(ids)
def decide(df, prob, thr, assign=True):
    d = df[["g1", "gp"]].copy(); d["p"] = prob; d = d[d.p >= thr]
    if assign: d = d.sort_values("p", ascending=False).drop_duplicates("gp")
    pred = {}
    for a, b in zip(d.g1.values, d.gp.values): pred.setdefault(a, set()).add(b)
    return pred
def best_thr(df, prob, ids): return max((f05(decide(df, prob, t), ids), t) for t in np.arange(0.2, 0.96, 0.05))

t0 = time.time(); models = {}
m = lgb.LGBMClassifier(n_estimators=1500, learning_rate=0.05, num_leaves=127, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=8)
m.fit(tr[FEATS], tr.y, eval_set=[(va[FEATS], va.y)], callbacks=[lgb.early_stopping(50, verbose=False)])
models["lightgbm"] = (m.predict_proba(va[FEATS])[:, 1], m.predict_proba(te[FEATS])[:, 1]); print("lgb", m.best_iteration_, f"{time.time()-t0:.0f}s", flush=True)
m2 = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.06, max_depth=9, subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=8, early_stopping_rounds=50, eval_metric="logloss")
m2.fit(tr[FEATS], tr.y, eval_set=[(va[FEATS], va.y)], verbose=False)
models["xgboost"] = (m2.predict_proba(va[FEATS])[:, 1], m2.predict_proba(te[FEATS])[:, 1]); print("xgb", f"{time.time()-t0:.0f}s", flush=True)

dev = "mps" if torch.backends.mps.is_available() else "cpu"
mu, sd = tr[FEATS].mean(), tr[FEATS].std() + 1e-6
def T(df): return torch.tensor(((df[FEATS] - mu) / sd).values, dtype=torch.float32)
Xt, yt = T(tr).to(dev), torch.tensor(tr.y.values, dtype=torch.float32).to(dev)
net = nn.Sequential(nn.Linear(len(FEATS), 256), nn.ReLU(), nn.Dropout(.1), nn.Linear(256, 128), nn.ReLU(), nn.Dropout(.1), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)).to(dev)
EP, BS = 12, 8192
opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4); lossf = nn.BCEWithLogitsLoss()
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-3, total_steps=EP * ((len(Xt) + BS - 1) // BS))
for ep in range(EP):
    net.train(); idx = torch.randperm(len(Xt), device=dev)
    for s in range(0, len(Xt), BS):
        b = idx[s:s + BS]; opt.zero_grad(); lossf(net(Xt[b]).squeeze(1), yt[b]).backward(); opt.step(); sched.step()
net.eval()
def mlp_pred(df):
    out = []
    with torch.no_grad():
        for s in range(0, len(df), 500000): out.append(torch.sigmoid(net(T(df.iloc[s:s + 500000]).to(dev)).squeeze(1)).cpu().numpy())
    return np.concatenate(out)
models["mlp"] = (mlp_pred(va), mlp_pred(te)); print("mlp", dev, f"{time.time()-t0:.0f}s", flush=True)
models["ensemble"] = tuple(np.mean([models[k][i] for k in ("lightgbm", "xgboost", "mlp")], axis=0) for i in (0, 1))

va_ids, te_ids = np.where(split == 1)[0], np.where(split == 2)[0]; report = {}
for name, (pv, pt) in models.items():
    fv, thr = best_thr(va, pv, va_ids)
    report[name] = {"val_f05": round(fv, 4), "thr": round(float(thr), 2), "test_f05": round(f05(decide(te, pt, thr), te_ids), 4),
                    "test_f05_no_assign": round(f05(decide(te, pt, thr, False), te_ids), 4), "pair_AP_test": round(average_precision_score(te.y, pt), 4)}
    print(name, report[name], flush=True)
pred = decide(te, models["ensemble"][1], report["ensemble"]["thr"])
for c in ("India", "US"):
    ids = [i for i in te_ids if ctry[i] == c]; report[f"ensemble_{c}"] = round(f05(pred, ids), 4); print(c, report[f"ensemble_{c}"])
# recall ceiling and precision/recall at chosen threshold
tp = sum(len(TRUE[i] & pred.get(i, set())) for i in te_ids); npred = sum(len(pred.get(i, ())) for i in te_ids); ntrue = sum(len(TRUE[i]) for i in te_ids)
report["ensemble_pair_precision"] = round(tp / npred, 4); report["ensemble_pair_recall"] = round(tp / ntrue, 4); print(report["ensemble_pair_precision"], report["ensemble_pair_recall"])
json.dump(report, open(f"{OUT}/report_v1_full.json", "w"), indent=1)
m.booster_.save_model(f"{OUT}/lgb.txt"); m2.save_model(f"{OUT}/xgb.json"); torch.save(net.state_dict(), f"{OUT}/mlp.pt")
pd.to_pickle({"mu": mu, "sd": sd, "FEATS": FEATS, "thr": report["ensemble"]["thr"]}, f"{OUT}/meta.pkl")
