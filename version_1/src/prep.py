"""Normalise raw data per country and cache it. tag='dev' (train subset) or 'test' (real test files)."""
import pandas as pd, numpy as np, os, sys, time
from config import *
from normalize import normalize_df

def rd(p): return pd.read_csv(os.path.join(DATA_DIR, p), sep="\t", dtype=str, keep_default_na=False, quoting=3)

def prep(tag):
    d = os.path.join(WORK, tag); os.makedirs(d, exist_ok=True)
    if tag == "dev":
        raw = [pd.read_parquet(f"{WORK}/s{s}.parquet") for s in (1, 2, 3)]
    else:
        raw = [rd(f"test/test_source{s}.tsv") for s in (1, 2, 3)]
    countries = sorted(raw[0].country.unique())
    for c in countries:
        t = time.time()
        s1 = normalize_df(raw[0][raw[0].country == c].reset_index(drop=True))
        ps = []
        for s in (2, 3):
            p = normalize_df(raw[s - 1][raw[s - 1].country == c].reset_index(drop=True)); p["src"] = s; ps.append(p)
        pool = pd.concat(ps, ignore_index=True)
        s1.to_parquet(f"{d}/{c}_s1n.parquet"); pool.to_parquet(f"{d}/{c}_pooln.parquet")
        print(tag, c, len(s1), len(pool), f"{time.time()-t:.0f}s", flush=True)
    # countries in pool but not in s1 are ignored (cannot be matched)
if __name__ == "__main__":
    prep(sys.argv[1])
