"""Data preparation script for training:
Reads raw source files from dataset/train, normalizes text, and saves parquet files."""
import os
import sys
import time
import pandas as pd
import numpy as np
from config import DATA_DIR, WORK, SEED
from normalize import normalize_df


def read_tsv(rel_path):
    full_path = os.path.join(DATA_DIR, rel_path)
    print(f"Reading {full_path} ...", flush=True)
    return pd.read_csv(full_path, sep="\t", dtype=str, keep_default_na=False, quoting=3)


def prepare_training_data(n_s1_sample=None):
    t0 = time.time()
    tag_dir = os.path.join(WORK, "train_ready")
    os.makedirs(tag_dir, exist_ok=True)

    print("Step 1: Reading reference S1 and ground truth...", flush=True)
    s1 = read_tsv("train/train_source1.tsv")
    gt = read_tsv("train/train_ground_truth.tsv")

    if n_s1_sample is not None and n_s1_sample < len(s1):
        rng = np.random.default_rng(SEED)
        chosen_indices = rng.choice(len(s1), size=n_s1_sample, replace=False)
        s1 = s1.iloc[chosen_indices].reset_index(drop=True)
        print(f"Sampled {n_s1_sample:,} S1 entities for training.", flush=True)

    s1_ids = set(s1.entity_id.values)
    gt_filtered = gt[gt.source1_entity_id.isin(s1_ids)].reset_index(drop=True)
    gt_filtered.to_parquet(os.path.join(tag_dir, "gt.parquet"))
    print(f"Saved gt.parquet with {len(gt_filtered):,} ground truth rows.", flush=True)
    del gt, gt_filtered

    print("Step 2: Reading candidate pool sources (S2 and S3)...", flush=True)
    s2 = read_tsv("train/train_source2.tsv")
    s3 = read_tsv("train/train_source3.tsv")

    countries = sorted(s1.country.unique())
    for c in countries:
        tc = time.time()
        print(f"\n--- Processing Country: {c} ---", flush=True)
        s1_c = s1[s1.country == c].reset_index(drop=True)
        print(f"Normalizing S1 {c} ({len(s1_c):,} records)...", flush=True)
        s1_norm = normalize_df(s1_c)
        s1_norm.to_parquet(os.path.join(tag_dir, f"{c}_s1n.parquet"))

        s2_c = s2[s2.country == c].reset_index(drop=True)
        s3_c = s3[s3.country == c].reset_index(drop=True)

        print(f"Normalizing S2 {c} ({len(s2_c):,} records)...", flush=True)
        p2_norm = normalize_df(s2_c)
        p2_norm["src"] = 2

        print(f"Normalizing S3 {c} ({len(s3_c):,} records)...", flush=True)
        p3_norm = normalize_df(s3_c)
        p3_norm["src"] = 3

        pool = pd.concat([p2_norm, p3_norm], ignore_index=True)
        pool.to_parquet(os.path.join(tag_dir, f"{c}_pooln.parquet"))

        print(f"[{c}] Finished in {time.time()-tc:.1f}s: S1={len(s1_norm):,}, Pool={len(pool):,}", flush=True)

    print(f"\nAll data prepared successfully in {time.time()-t0:.1f}s!", flush=True)


if __name__ == "__main__":
    sample = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    prepare_training_data(sample)
