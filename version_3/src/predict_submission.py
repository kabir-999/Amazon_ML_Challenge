"""Inference pipeline for generating test set predictions:
Normalizes test sources, generates candidates, extracts features, scores with the GPU ensemble,
and produces matching_results.tsv and candidate_pairs.tsv ready for submission."""
import os
import sys
import time
import pandas as pd
import numpy as np
import torch
import lightgbm as lgb
import xgboost as xgb

from config import DATA_DIR, OUT, WORK, SEED
from normalize import normalize_df
from blocking import block_country
from features import compute_fuzzy_array, build_competition_context, assemble_feature_matrix, ALL_FEATURES
from train_gpu import DeepERNet


def read_tsv(rel_path):
    full_path = os.path.join(DATA_DIR, rel_path)
    print(f"Reading {full_path} ...", flush=True)
    return pd.read_csv(full_path, sep="\t", dtype=str, keep_default_na=False, quoting=3)


def run_inference():
    t_start = time.time()
    print("=" * 70, flush=True)
    print("   GENERATING TEST SUBMISSION (MATCHING & CANDIDATES)", flush=True)
    print("=" * 70, flush=True)

    meta_path = os.path.join(OUT, "meta.pkl")
    if not os.path.exists(meta_path):
        print(f"Error: Model metadata not found at {meta_path}. Run train_gpu.py first!", flush=True)
        return

    meta = pd.read_pickle(meta_path)
    FEATS = meta["FEATS"]
    mu = meta["mean"]
    sd = meta["std"]
    threshold = meta["opt_threshold"]
    print(f"Loaded metadata. Decision Threshold: {threshold:.3f}", flush=True)

    # Load models
    print("Loading trained models...", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference device: {device}", flush=True)

    lgb_booster = lgb.Booster(model_file=os.path.join(OUT, "lgb_model.txt"))
    xgb_clf = xgb.XGBClassifier()
    xgb_clf.load_model(os.path.join(OUT, "xgb_model.json"))

    mlp_net = DeepERNet(len(FEATS)).to(device)
    mlp_net.load_state_dict(torch.load(os.path.join(OUT, "mlp_model.pt"), map_location=device))
    mlp_net.eval()

    # Read test sets
    s1_test = read_tsv("test/test_source1.tsv")
    s2_test = read_tsv("test/test_source2.tsv")
    s3_test = read_tsv("test/test_source3.tsv")

    countries = ["US", "India", "France"]
    all_matching_dfs = []
    all_candidate_dfs = []

    for c in countries:
        t_c = time.time()
        print(f"\n--- Processing Test Country: {c} ---", flush=True)

        s1_c = s1_test[s1_test.country == c].reset_index(drop=True)
        if len(s1_c) == 0:
            continue

        s2_c = s2_test[s2_test.country == c].reset_index(drop=True)
        s3_c = s3_test[s3_test.country == c].reset_index(drop=True)

        print(f"Normalizing {c} records...", flush=True)
        s1_norm = normalize_df(s1_c)

        p2_norm = normalize_df(s2_c)
        p2_norm["src"] = 2

        p3_norm = normalize_df(s3_c)
        p3_norm["src"] = 3

        pool = pd.concat([p2_norm, p3_norm], ignore_index=True)
        print(f"[{c}] S1: {len(s1_norm):,}, Pool (S2+S3): {len(pool):,}", flush=True)

        # 1. Blocking
        print("Generating candidates via multi-channel blocking...", flush=True)
        cand = block_country(s1_norm, pool, k=30, max_cands=60)

        # 2. Features
        print("Computing fuzzy similarity features...", flush=True)
        F = compute_fuzzy_array(s1_norm, pool, cand)

        print("Computing competition context...", flush=True)
        ctx = build_competition_context(F, cand, pool)

        # 3. Model Scoring in chunks
        print("Scoring candidates with GPU Ensemble (LightGBM + XGBoost + PyTorch CUDA)...", flush=True)
        n_cands = len(cand)
        probs = np.empty(n_cands, dtype=np.float32)
        CHUNK = 500_000

        for lo in range(0, n_cands, CHUNK):
            hi = min(lo + CHUNK, n_cands)
            X_chunk = assemble_feature_matrix(F, cand, s1_norm, pool, ctx, lo, hi)

            p_lgb = lgb_booster.predict(X_chunk)
            p_xgb = xgb_clf.predict_proba(X_chunk)[:, 1]

            with torch.no_grad():
                X_tensor = torch.tensor(((X_chunk[FEATS] - mu) / sd).values, dtype=torch.float32).to(device)
                p_mlp = torch.sigmoid(mlp_net(X_tensor).squeeze(1)).cpu().numpy()

            probs[lo:hi] = (p_lgb + p_xgb + p_mlp) / 3.0

        # 4. Mutual Exclusivity Assignment
        print("Applying threshold & mutual exclusivity assignment...", flush=True)
        df_scored = pd.DataFrame({"i1": cand.i1.values, "ip": cand.ip.values, "p": probs})
        kept = df_scored[df_scored.p >= threshold].sort_values("p", ascending=False).drop_duplicates("ip")

        pool_eids = pool.entity_id.values
        s1_eids = s1_norm.entity_id.values

        # Build comma-separated matches
        matched_series = kept.groupby("i1").ip.apply(lambda idxs: ",".join(pool_eids[idxs.values]))
        df_match = pd.DataFrame({"source1_entity_id": s1_eids, "matched_entity_ids": ""})
        df_match.loc[matched_series.index.values, "matched_entity_ids"] = matched_series.values

        # Build comma-separated candidates
        cand_series = cand.groupby("i1").ip.apply(lambda idxs: ",".join(pool_eids[idxs.values]))
        df_cand = pd.DataFrame({"source1_entity_id": s1_eids, "candidate_entity_ids": ""})
        df_cand.loc[cand_series.index.values, "candidate_entity_ids"] = cand_series.values

        all_matching_dfs.append(df_match)
        all_candidate_dfs.append(df_cand)

        n_singletons = (df_match.matched_entity_ids == "").sum()
        total_matches = kept.shape[0]
        print(f"[{c}] Finished in {time.time()-t_c:.1f}s | Singletons: {n_singletons:,} ({n_singletons/len(s1_norm)*100:.1f}%) | Total Matches: {total_matches:,} (avg {total_matches/len(s1_norm):.2f}/S1)", flush=True)

    # Concatenate full test outputs
    final_matching = pd.concat(all_matching_dfs, ignore_index=True)
    final_candidates = pd.concat(all_candidate_dfs, ignore_index=True)

    match_out = os.path.join(OUT, "matching_results.tsv")
    cand_out = os.path.join(OUT, "candidate_pairs.tsv")

    print(f"\nSaving final files to:\n  {match_out}\n  {cand_out}", flush=True)
    final_matching.to_csv(match_out, sep="\t", index=False)
    final_candidates.to_csv(cand_out, sep="\t", index=False)

    print(f"Inference complete in {time.time()-t_start:.1f}s!", flush=True)


if __name__ == "__main__":
    run_inference()
