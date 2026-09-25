"""GPU-accelerated ensemble training pipeline (LightGBM + XGBoost + PyTorch CUDA MLP).
Features live terminal progress logging, metric tracking, and persistent model checkpoints."""
import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import average_precision_score

from config import WORK, OUT, SEED
from blocking import block_country
from features import compute_fuzzy_array, build_competition_context, assemble_feature_matrix, ALL_FEATURES


def f05_score(true_map, pred_map, entity_ids):
    """Computes Macro F0.5 score over S1 entities including singletons."""
    total = 0.0
    for i in entity_ids:
        t = true_map.get(i, set())
        p = pred_map.get(i, set())
        if not t and not p:
            total += 1.0  # Correct singleton
            continue
        if not t or not p:
            continue  # False merge on singleton or missed match
        tp = len(t & p)
        if tp > 0:
            prec = tp / len(p)
            rec = tp / len(t)
            total += (1.25 * prec * rec) / (0.25 * prec + rec)
    return total / max(len(entity_ids), 1)


def assign_predictions(g1_arr, gp_arr, probs, threshold):
    """Filters pairs above threshold and assigns each pool record to its highest-scoring S1."""
    mask = probs >= threshold
    if not np.any(mask):
        return {}

    df = pd.DataFrame({"g1": g1_arr[mask], "gp": gp_arr[mask], "p": probs[mask]})
    df = df.sort_values("p", ascending=False).drop_duplicates("gp")

    preds = {}
    for a, b in zip(df.g1.values, df.gp.values):
        if a not in preds:
            preds[a] = set()
        preds[a].add(b)
    return preds


def optimize_threshold(g1_arr, gp_arr, probs, true_map, val_ids):
    """Grid searches for the threshold that maximizes Macro F0.5 on validation data."""
    best_score, best_thr = -1.0, 0.75
    thresholds = np.arange(0.35, 0.95, 0.05)
    for thr in thresholds:
        preds = assign_predictions(g1_arr, gp_arr, probs, thr)
        score = f05_score(true_map, preds, val_ids)
        if score > best_score:
            best_score = score
            best_thr = thr
    return best_score, best_thr


class DeepERNet(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


def run_pipeline():
    start_total = time.time()
    print("=" * 70, flush=True)
    print("   AMAZON ML CHALLENGE — GPU ENTITY RESOLUTION PIPELINE", flush=True)
    print("=" * 70, flush=True)

    data_dir = os.path.join(WORK, "train_ready")
    gt_path = os.path.join(data_dir, "gt.parquet")
    if not os.path.exists(gt_path):
        print(f"Error: Prepared data not found at {data_dir}. Run prep_data.py first!", flush=True)
        return

    gt = pd.read_parquet(gt_path)
    gt_map = dict(zip(gt.source1_entity_id, gt.matched_entity_ids))
    del gt

    # Detect compute device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware compute device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**2):.0f} MiB", flush=True)

    dataset_parts = []
    true_labels = {}
    country_of_s1 = []
    global_s1_offset = 0
    global_pool_offset = 0

    countries = ["India", "US"]
    for c in countries:
        t_country = time.time()
        s1_file = os.path.join(data_dir, f"{c}_s1n.parquet")
        pool_file = os.path.join(data_dir, f"{c}_pooln.parquet")

        if not os.path.exists(s1_file) or not os.path.exists(pool_file):
            print(f"Skipping {c}: Files missing.", flush=True)
            continue

        s1 = pd.read_parquet(s1_file)
        pool = pd.read_parquet(pool_file)
        s1.drop(columns=[c for c in ("nm_raw", "ad_raw") if c in s1.columns], inplace=True, errors="ignore")
        pool.drop(columns=[c for c in ("nm_raw", "ad_raw") if c in pool.columns], inplace=True, errors="ignore")
        print(f"\n>>> [{c}] S1: {len(s1):,}, Pool: {len(pool):,}", flush=True)

        # 1. High-Recall Candidate Generation
        cand_cache = os.path.join(data_dir, f"{c}_cand.parquet")
        if os.path.exists(cand_cache):
            print(f"  Loading cached candidate pairs from {cand_cache}...", flush=True)
            cand = pd.read_parquet(cand_cache)
        else:
            print("  Generating candidate pairs via multi-channel blocking...", flush=True)
            cand = block_country(s1, pool, k=30, max_cands=60)
            cand.to_parquet(cand_cache)

        # 2. Build Ground Truth Labels
        print("  Mapping ground truth labels to candidate pairs...", flush=True)
        s1_to_idx = {eid: idx for idx, eid in enumerate(s1.entity_id)}
        pool_to_idx = {eid: idx for idx, eid in enumerate(pool.entity_id)}

        for k, eid in enumerate(s1.entity_id):
            matched_str = gt_map.get(eid, "")
            true_labels[k + global_s1_offset] = {
                pool_to_idx[mid] + global_pool_offset
                for mid in matched_str.split(",") if mid in pool_to_idx
            }

        true_pair_set = set()
        for k, eid in enumerate(s1.entity_id):
            matched_str = gt_map.get(eid, "")
            for mid in matched_str.split(","):
                if mid in pool_to_idx:
                    true_pair_set.add((k, pool_to_idx[mid]))

        y_labels = np.fromiter(
            ((i, j) in true_pair_set for i, j in zip(cand.i1.values, cand.ip.values)),
            dtype=bool,
            count=len(cand)
        )
        recall_pct = (y_labels.sum() / max(len(true_pair_set), 1)) * 100.0
        print(f"  --> Candidate pairs: {len(cand):,} | Blocking Pair Recall: {recall_pct:.2f}% ({y_labels.sum():,}/{len(true_pair_set):,})", flush=True)

        # 3. Extract Features
        fuzzy_cache = os.path.join(data_dir, f"{c}_F.npy")
        if os.path.exists(fuzzy_cache):
            print(f"  Loading cached fuzzy features from {fuzzy_cache}...", flush=True)
            F = np.load(fuzzy_cache)
        else:
            print("  Extracting RapidFuzz string similarity features...", flush=True)
            F = compute_fuzzy_array(s1, pool, cand)
            np.save(fuzzy_cache, F)

        print("  Computing global competition and frequency context...", flush=True)
        ctx = build_competition_context(F, cand, pool)

        print("  Assembling complete feature matrix...", flush=True)
        X_country = assemble_feature_matrix(F, cand, s1, pool, ctx)
        X_country["y"] = y_labels
        X_country["g1"] = cand.i1.values + global_s1_offset
        X_country["gp"] = cand.ip.values + global_pool_offset
        X_country["country"] = c

        dataset_parts.append(X_country)
        country_of_s1.extend([c] * len(s1))
        global_s1_offset += len(s1)
        global_pool_offset += len(pool)
        print(f"[{c}] Finished in {time.time()-t_country:.1f}s", flush=True)

    if not dataset_parts:
        print("No data processed.", flush=True)
        return

    print("\n" + "=" * 70, flush=True)
    print("Step 4: Splitting Train / Val / Test (Split by S1 Entity)", flush=True)
    X = pd.concat(dataset_parts, ignore_index=True)
    total_s1 = global_s1_offset
    country_of_s1 = np.array(country_of_s1)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(total_s1)
    split_assign = np.empty(total_s1, dtype=int)
    split_assign[perm[:int(0.70 * total_s1)]] = 0  # Train (70%)
    split_assign[perm[int(0.70 * total_s1):int(0.85 * total_s1)]] = 1  # Val (15%)
    split_assign[perm[int(0.85 * total_s1):]] = 2  # Held-out Test (15%)

    pair_splits = split_assign[X.g1.values]
    tr_df = X[pair_splits == 0].reset_index(drop=True)
    va_df = X[pair_splits == 1].reset_index(drop=True)
    te_df = X[pair_splits == 2].reset_index(drop=True)

    val_s1_ids = np.where(split_assign == 1)[0]
    test_s1_ids = np.where(split_assign == 2)[0]

    print(f"Candidate pairs: Train = {len(tr_df):,} | Val = {len(va_df):,} | Test = {len(te_df):,}", flush=True)
    print(f"Reference S1s:   Train = {int(0.7*total_s1):,} | Val = {len(val_s1_ids):,} | Test = {len(test_s1_ids):,}", flush=True)
    print(f"Total features:  {len(ALL_FEATURES)}", flush=True)

    # ----------------------------------------------------
    # Model 1: LightGBM Classifier
    # ----------------------------------------------------
    print("\n" + "=" * 70, flush=True)
    print("Step 5: Training LightGBM Classifier", flush=True)
    t_lgb = time.time()
    lgb_model = lgb.LGBMClassifier(
        n_estimators=1200,
        learning_rate=0.05,
        num_leaves=127,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        min_child_samples=40,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(
        tr_df[ALL_FEATURES], tr_df.y,
        eval_set=[(va_df[ALL_FEATURES], va_df.y)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False), lgb.log_evaluation(period=200)]
    )
    lgb_val_probs = lgb_model.predict_proba(va_df[ALL_FEATURES])[:, 1]
    lgb_test_probs = lgb_model.predict_proba(te_df[ALL_FEATURES])[:, 1]

    lgb_path = os.path.join(OUT, "lgb_model.txt")
    lgb_model.booster_.save_model(lgb_path)
    print(f"--> LightGBM trained in {time.time()-t_lgb:.1f}s. Saved checkpoint to {lgb_path}", flush=True)

    # ----------------------------------------------------
    # Model 2: XGBoost Classifier
    # ----------------------------------------------------
    print("\n" + "=" * 70, flush=True)
    print("Step 6: Training XGBoost Classifier", flush=True)
    t_xgb = time.time()
    xgb_model = xgb.XGBClassifier(
        n_estimators=800,
        learning_rate=0.06,
        max_depth=9,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        early_stopping_rounds=50,
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1
    )
    xgb_model.fit(
        tr_df[ALL_FEATURES], tr_df.y,
        eval_set=[(va_df[ALL_FEATURES], va_df.y)],
        verbose=150
    )
    xgb_val_probs = xgb_model.predict_proba(va_df[ALL_FEATURES])[:, 1]
    xgb_test_probs = xgb_model.predict_proba(te_df[ALL_FEATURES])[:, 1]

    xgb_path = os.path.join(OUT, "xgb_model.json")
    xgb_model.save_model(xgb_path)
    print(f"--> XGBoost trained in {time.time()-t_xgb:.1f}s. Saved checkpoint to {xgb_path}", flush=True)

    # ----------------------------------------------------
    # Model 3: PyTorch Deep MLP on GPU (CUDA)
    # ----------------------------------------------------
    print("\n" + "=" * 70, flush=True)
    print(f"Step 7: Training PyTorch Deep MLP on {device}", flush=True)
    t_mlp = time.time()
    feature_mean = tr_df[ALL_FEATURES].mean()
    feature_std = tr_df[ALL_FEATURES].std() + 1e-6

    def normalize_tensor(df):
        normed = (df[ALL_FEATURES] - feature_mean) / feature_std
        return torch.tensor(normed.values, dtype=torch.float32)

    X_train_t = normalize_tensor(tr_df)
    y_train_t = torch.tensor(tr_df.y.values, dtype=torch.float32)
    X_val_t = normalize_tensor(va_df)
    y_val_t = torch.tensor(va_df.y.values, dtype=torch.float32)

    mlp_net = DeepERNet(len(ALL_FEATURES)).to(device)
    optimizer = torch.optim.AdamW(mlp_net.parameters(), lr=3e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    EPOCHS = 10
    BATCH_SIZE = 8192
    num_batches = (len(X_train_t) + BATCH_SIZE - 1) // BATCH_SIZE
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=4e-3, total_steps=EPOCHS * num_batches)

    best_val_loss = float("inf")
    mlp_path = os.path.join(OUT, "mlp_model.pt")

    for epoch in range(1, EPOCHS + 1):
        mlp_net.train()
        permutation = torch.randperm(len(X_train_t))
        epoch_loss = 0.0

        for b in range(num_batches):
            indices = permutation[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            batch_x = X_train_t[indices].to(device)
            batch_y = y_train_t[indices].to(device)

            optimizer.zero_grad()
            outputs = mlp_net(batch_x).squeeze(1)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item() * len(batch_x)

        epoch_loss /= len(X_train_t)

        # Validation Loss
        mlp_net.eval()
        with torch.no_grad():
            val_out = mlp_net(X_val_t.to(device)).squeeze(1)
            val_loss = loss_fn(val_out, y_val_t.to(device)).item()

        print(f"  [Epoch {epoch:2d}/{EPOCHS}] Train BCE: {epoch_loss:.4f} | Val BCE: {val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(mlp_net.state_dict(), mlp_path)

    # Load best checkpoint
    mlp_net.load_state_dict(torch.load(mlp_path))
    mlp_net.eval()

    def predict_mlp_probs(df):
        results = []
        with torch.no_grad():
            for s in range(0, len(df), 100000):
                e = min(s + 100000, len(df))
                sub_t = normalize_tensor(df.iloc[s:e]).to(device)
                p = torch.sigmoid(mlp_net(sub_t).squeeze(1)).cpu().numpy()
                results.append(p)
        return np.concatenate(results)

    mlp_val_probs = predict_mlp_probs(va_df)
    mlp_test_probs = predict_mlp_probs(te_df)
    print(f"--> PyTorch MLP trained in {time.time()-t_mlp:.1f}s. Saved checkpoint to {mlp_path}", flush=True)

    # ----------------------------------------------------
    # Model 4: Triplet Ensemble (LightGBM + XGBoost + MLP)
    # ----------------------------------------------------
    print("\n" + "=" * 70, flush=True)
    print("Step 8: Threshold Optimization & Evaluation on Held-Out Test Split", flush=True)
    models_dict = {
        "LightGBM": (lgb_val_probs, lgb_test_probs),
        "XGBoost": (xgb_val_probs, xgb_test_probs),
        "PyTorch_MLP": (mlp_val_probs, mlp_test_probs),
        "Ensemble_3x": (
            (lgb_val_probs + xgb_val_probs + mlp_val_probs) / 3.0,
            (lgb_test_probs + xgb_test_probs + mlp_test_probs) / 3.0
        )
    }

    report = {}
    for name, (val_p, test_p) in models_dict.items():
        val_f05, best_t = optimize_threshold(va_df.g1.values, va_df.gp.values, val_p, true_labels, val_s1_ids)
        test_preds = assign_predictions(te_df.g1.values, te_df.gp.values, test_p, best_t)
        test_f05 = f05_score(true_labels, test_preds, test_s1_ids)
        test_ap = average_precision_score(te_df.y, test_p)

        report[name] = {
            "val_F0.5": round(float(val_f05), 4),
            "opt_threshold": round(float(best_t), 3),
            "test_F0.5": round(float(test_f05), 4),
            "test_AP": round(float(test_ap), 4)
        }
        print(f"  [{name:12s}] Val F0.5: {val_f05:.4f} (thr={best_t:.2f}) | Held-Out Test F0.5: {test_f05:.4f} | AP: {test_ap:.4f}", flush=True)

    # Detailed metrics for Ensemble on Test
    ens_test_preds = assign_predictions(te_df.g1.values, te_df.gp.values, models_dict["Ensemble_3x"][1], report["Ensemble_3x"]["opt_threshold"])
    for c in countries:
        c_ids = [i for i in test_s1_ids if country_of_s1[i] == c]
        c_score = f05_score(true_labels, ens_test_preds, c_ids)
        report[f"Ensemble_test_F0.5_{c}"] = round(float(c_score), 4)
        print(f"    --> Test F0.5 ({c}): {c_score:.4f}", flush=True)

    tp_pairs = sum(len(true_labels[i] & ens_test_preds.get(i, set())) for i in test_s1_ids)
    pred_pairs = sum(len(ens_test_preds.get(i, set())) for i in test_s1_ids)
    true_pairs = sum(len(true_labels[i]) for i in test_s1_ids)

    pair_prec = tp_pairs / max(pred_pairs, 1)
    pair_rec = tp_pairs / max(true_pairs, 1)
    report["Ensemble_pair_precision"] = round(float(pair_prec), 4)
    report["Ensemble_pair_recall"] = round(float(pair_rec), 4)
    print(f"    --> Pair Precision: {pair_prec:.4f} | Pair Recall: {pair_rec:.4f}", flush=True)

    # Save metadata & report
    meta_path = os.path.join(OUT, "meta.pkl")
    pd.to_pickle({
        "FEATS": ALL_FEATURES,
        "mean": feature_mean,
        "std": feature_std,
        "opt_threshold": report["Ensemble_3x"]["opt_threshold"]
    }, meta_path)

    report_path = os.path.join(OUT, "training_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70, flush=True)
    print(f"PIPELINE COMPLETE in {time.time()-start_total:.1f}s! All models and metadata saved to {OUT}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    run_pipeline()
