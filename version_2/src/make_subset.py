"""Build a development subset of the train data.
Samples N S1 entities, keeps ALL of their matched S2/S3 records and a proportional
sample of orphan S2/S3 records, so pool/S1 ratio matches the full data."""
import pandas as pd, numpy as np, os
from config import *

def rd(p): return pd.read_csv(os.path.join(DATA_DIR, p), sep="\t", dtype=str, keep_default_na=False, quoting=3)

rng = np.random.default_rng(SEED)
s1 = rd("train/train_source1.tsv"); gt = rd("train/train_ground_truth.tsv")
frac = N_S1_SUBSET / len(s1)
pick = set(rng.choice(s1.entity_id.values, N_S1_SUBSET, replace=False))
gt = gt[gt.source1_entity_id.isin(pick)].copy()
matched = set(i for x in gt.matched_entity_ids for i in x.split(",") if i)
s1 = s1[s1.entity_id.isin(pick)].reset_index(drop=True)
allm = None
pools = []
full_gt = rd("train/train_ground_truth.tsv")
allm = set(i for x in full_gt.matched_entity_ids for i in x.split(",") if i)
for s in (2, 3):
    df = rd(f"train/train_source{s}.tsv")
    keep = df.entity_id.isin(matched)
    orphan = ~df.entity_id.isin(allm)
    samp = orphan & (rng.random(len(df)) < frac)
    pools.append(df[keep | samp].reset_index(drop=True))
    print(f"S{s}: total {len(df)}, kept matched {keep.sum()}, orphan sample {samp.sum()}")
# does any S2/S3 id occur under >1 S1?
cnt = pd.Series([i for x in full_gt.matched_entity_ids for i in x.split(",") if i]).value_counts()
print("ids matched to >1 S1:", (cnt > 1).sum(), "of", len(cnt))
s1.to_parquet(f"{WORK}/s1.parquet"); pools[0].to_parquet(f"{WORK}/s2.parquet"); pools[1].to_parquet(f"{WORK}/s3.parquet"); gt.to_parquet(f"{WORK}/gt.parquet")
print(len(s1), len(pools[0]), len(pools[1]), len(gt))
