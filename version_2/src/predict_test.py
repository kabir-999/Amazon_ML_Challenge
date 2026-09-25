"""Score all test candidate pairs with the ensemble, apply threshold + one-S1-per-record assignment, write submission files."""
import sys, pandas as pd, numpy as np, torch, torch.nn as nn, lightgbm as lgb, xgboost as xgb, os, time
from config import *
from features import context_arrays, assemble

meta = pd.read_pickle(f"{OUT}/meta.pkl"); FEATS, mu, sd, THR = meta["FEATS"], meta["mu"], meta["sd"], meta["thr"]
bl = lgb.Booster(model_file=f"{OUT}/lgb.txt"); bx = xgb.XGBClassifier(); bx.load_model(f"{OUT}/xgb.json")
dev = "mps" if torch.backends.mps.is_available() else "cpu"
net = nn.Sequential(nn.Linear(len(FEATS), 256), nn.ReLU(), nn.Dropout(.1), nn.Linear(256, 128), nn.ReLU(), nn.Dropout(.1), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
net.load_state_dict(torch.load(f"{OUT}/mlp.pt")); net.to(dev).eval()
D = f"{WORK}/test"; os.makedirs(f"{OUT}/submission", exist_ok=True)
m_rows, c_rows = [], []; stats = {}
for c in sys.argv[1:]:
    t = time.time()
    s1 = pd.read_parquet(f"{D}/{c}_s1n.parquet"); pool = pd.read_parquet(f"{D}/{c}_pooln.parquet"); cand = pd.read_parquet(f"{D}/{c}_cand.parquet")
    F = np.load(f"{D}/{c}_F.npy", mmap_mode="r"); ctx = context_arrays(np.asarray(F), cand, s1, pool)
    prob = np.empty(len(cand), np.float32); CH = 1_000_000
    for lo in range(0, len(cand), CH):
        hi = min(lo + CH, len(cand)); X = assemble(F, cand, s1, pool, ctx, lo, hi)
        with torch.no_grad(): pm = torch.sigmoid(net(torch.tensor(((X - mu) / sd).values, dtype=torch.float32).to(dev)).squeeze(1)).cpu().numpy()
        prob[lo:hi] = (bl.predict(X) + bx.predict_proba(X)[:, 1] + pm) / 3
    d = pd.DataFrame({"i1": cand.i1.values, "ip": cand.ip.values, "p": prob})
    kept = d[d.p >= THR].sort_values("p", ascending=False).drop_duplicates("ip")
    pe = pool.entity_id.values; se = s1.entity_id.values
    mm = kept.groupby("i1").ip.apply(lambda x: ",".join(pe[x.values]))
    m = pd.DataFrame({"source1_entity_id": se, "matched_entity_ids": ""}); m.loc[mm.index.values, "matched_entity_ids"] = mm.values
    cc = cand.groupby("i1").ip.apply(lambda x: ",".join(pe[x.values]))
    cd = pd.DataFrame({"source1_entity_id": se, "candidate_entity_ids": ""}); cd.loc[cc.index.values, "candidate_entity_ids"] = cc.values
    m_rows.append(m); c_rows.append(cd)
    n_match = kept.groupby("i1").size()
    stats[c] = {"s1": len(s1), "candidates": len(cand), "s1_with_match": int(len(n_match)), "singleton_rate": round(1 - len(n_match) / len(s1), 4), "avg_matches": round(len(kept) / len(s1), 3)}
    print(c, stats[c], f"{time.time()-t:.0f}s", flush=True)
for c, m, cd in zip(sys.argv[1:], m_rows, c_rows):
    m.to_csv(f"{OUT}/submission/matching_{c}.tsv", sep="\t", index=False); cd.to_csv(f"{OUT}/submission/candidates_{c}.tsv", sep="\t", index=False)
    pd.Series(stats[c]).to_json(f"{OUT}/test_stats_{c}.json")
print("written")
