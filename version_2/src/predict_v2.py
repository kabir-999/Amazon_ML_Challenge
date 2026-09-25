"""Score test candidates with the v2 two-stage ensemble and write per-country matching files (+ merged file)."""
import sys, numpy as np, pandas as pd, torch, torch.nn as nn, lightgbm as lgb, xgboost as xgb, os, time
from config import *
from features_v2 import S2CTX, context_v2, assemble2, stage2_context
from common_v2 import load_country

meta = pd.read_pickle(f"{OUT}/meta_v2.pkl"); FE, mu, sd, THR, K = meta["FE"], meta["mu"], meta["sd"], meta["thr"], meta["K"]
THR = float(os.environ.get("ER_THR", THR)); KEEP = 0.005
folds = [lgb.Booster(model_file=f"{OUT}/s1_fold{k}.txt") for k in range(K)]
b2 = lgb.Booster(model_file=f"{OUT}/s2_lgb.txt"); bx = xgb.XGBClassifier(); bx.load_model(f"{OUT}/s2_xgb.json")
dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
net = nn.Sequential(nn.Linear(len(FE), 384), nn.ReLU(), nn.Dropout(.1), nn.Linear(384, 192), nn.ReLU(), nn.Dropout(.1), nn.Linear(192, 64), nn.ReLU(), nn.Linear(64, 1))
net.load_state_dict(torch.load(f"{OUT}/s2_mlp.pt")); net.to(dev).eval()
os.makedirs(f"{OUT}/parts", exist_ok=True); stats = {}
for c in sys.argv[1:]:
    t = time.time(); d = load_country("test", c, with_labels=False); s1, pool, cand, F = d["s1"], d["pool"], d["cand"], d["F"]
    ctx = context_v2(F, cand, s1, pool); n = len(cand); p1 = np.zeros(n, np.float32)
    for lo in range(0, n, 1_000_000):
        r = np.arange(lo, min(lo + 1_000_000, n)); X = assemble2(F, cand, s1, pool, ctx, r)
        p1[r] = np.mean([b.predict(X) for b in folds], axis=0)
    print(c, "stage-1 done", f"{time.time()-t:.0f}s", flush=True)
    s2 = stage2_context(p1, cand, pool); idx = np.flatnonzero(p1 >= KEEP); p2 = np.zeros(n, np.float32)
    for lo in range(0, len(idx), 1_000_000):
        r = idx[lo:lo + 1_000_000]; X = assemble2(F, cand, s1, pool, ctx, r)
        for k in S2CTX: X[k] = s2[k][r]
        with torch.no_grad(): pm = torch.sigmoid(net(torch.tensor(((X[FE] - mu) / sd).values, dtype=torch.float32).to(dev)).squeeze(1)).cpu().numpy()
        p2[r] = (b2.predict(X[FE]) + bx.predict_proba(X[FE])[:, 1] + pm) / 3
    dd = pd.DataFrame({"i1": cand.i1.values, "ip": cand.ip.values, "p": p2})
    kept = dd[dd.p >= THR].sort_values("p", ascending=False).drop_duplicates("ip")
    pe, se = pool.entity_id.values, s1.entity_id.values
    mm = kept.groupby("i1").ip.apply(lambda x: ",".join(pe[x.values]))
    m = pd.DataFrame({"source1_entity_id": se, "matched_entity_ids": ""}); m.loc[mm.index.values, "matched_entity_ids"] = mm.values
    m.to_csv(f"{OUT}/parts/matching_{c}.tsv", sep="\t", index=False)
    nm = kept.groupby("i1").size(); stats[c] = {"s1": len(s1), "singleton_rate": round(1 - len(nm) / len(s1), 4), "avg_matches": round(len(kept) / len(s1), 3), "thr": THR}
    np.save(f"{WORK}/test/{c}_p2.npy", p2); print(c, stats[c], f"{time.time()-t:.0f}s", flush=True)
