"""Test-density training data: use ALL S1 train entities and keep the FULL S2/S3 train pool (all orphans included),
so blocking/candidate statistics match the real test run. Writes data/trainfull/<country>_{s1n,pooln}.parquet + gt."""
import pandas as pd, numpy as np, os, time
from config import *
from normalize import normalize_df
from prep import rd

d = os.path.join(WORK, "trainfull"); os.makedirs(d, exist_ok=True)
rng = np.random.default_rng(SEED)
s1 = rd("train/train_source1.tsv"); gt = rd("train/train_ground_truth.tsv")
pick = set(s1.entity_id.values)      # ALL train S1: competition/density features then match the real test run
s1 = s1[s1.entity_id.isin(pick)].reset_index(drop=True)
gt[gt.source1_entity_id.isin(pick)].to_parquet(f"{d}/gt.parquet"); del gt
raw2, raw3 = rd("train/train_source2.tsv"), rd("train/train_source3.tsv")
for c in sorted(s1.country.unique()):
    t = time.time()
    a = normalize_df(s1[s1.country == c].reset_index(drop=True))
    ps = []
    for s, r in ((2, raw2), (3, raw3)):
        p = normalize_df(r[r.country == c].reset_index(drop=True)); p["src"] = s; ps.append(p)
    pool = pd.concat(ps, ignore_index=True)
    a.to_parquet(f"{d}/{c}_s1n.parquet"); pool.to_parquet(f"{d}/{c}_pooln.parquet")
    print(c, len(a), len(pool), f"{time.time()-t:.0f}s", flush=True)
