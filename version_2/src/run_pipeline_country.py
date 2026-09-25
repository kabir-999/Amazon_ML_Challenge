"""blocking + fuzzy features for one country of a tag ('dev' or 'test'); artefacts cached under data/<tag>/."""
import pandas as pd, numpy as np, time, sys, os
from config import *
from blocking import block_country
from features import fuzzy_memmap

def run(tag, c, procs=8):
    d = f"{WORK}/{tag}"; t = time.time()
    s1 = pd.read_parquet(f"{d}/{c}_s1n.parquet"); pool = pd.read_parquet(f"{d}/{c}_pooln.parquet")
    print(f"[{c}] S1 {len(s1)} pool {len(pool)}", flush=True)
    cand = block_country(s1, pool, procs=procs)
    cand.to_parquet(f"{d}/{c}_cand.parquet")
    print(f"[{c}] blocked: {len(cand)} pairs ({len(cand)/len(s1):.1f}/S1) {time.time()-t:.0f}s", flush=True)
    if tag in ("dev", "trainfull"):
        gt = pd.read_parquet(f"{WORK}/gt.parquet" if tag == "dev" else f"{d}/gt.parquet")
        sid = dict(zip(s1.entity_id, range(len(s1)))); pid = dict(zip(pool.entity_id, range(len(pool))))
        true = {(sid[a], pid[b]) for a, m in zip(gt.source1_entity_id, gt.matched_entity_ids) if a in sid for b in m.split(",") if b}
        y = np.fromiter(((i, j) in true for i, j in zip(cand.i1.values, cand.ip.values)), bool, len(cand))
        np.save(f"{d}/{c}_y.npy", y); print(f"[{c}] blocking pair recall {y.sum()/len(true):.4f}", flush=True)
    fuzzy_memmap(s1, pool, cand, f"{d}/{c}_F.npy", procs=procs)
    print(f"[{c}] fuzzy features done {time.time()-t:.0f}s", flush=True)

if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
