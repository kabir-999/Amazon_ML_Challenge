"""Master runner script for version_3:
Prepares training data, runs GPU ensemble training, and saves model checkpoints."""
import sys
import os
import time

from prep_data import prepare_training_data
from train_gpu import run_pipeline

if __name__ == "__main__":
    t0 = time.time()
    s1_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 150_000

    print("=" * 70, flush=True)
    print(f"STARTING VERSION_3 PIPELINE WITH {s1_samples:,} S1 REFERENCE ENTITIES", flush=True)
    print("=" * 70, flush=True)

    # Step 1: Prepare data if not already prepared
    from config import WORK
    train_ready_dir = os.path.join(WORK, "train_ready")
    gt_file = os.path.join(train_ready_dir, "gt.parquet")

    if not os.path.exists(gt_file):
        print(f"\n[PHASE 1] Preparing normalized training data ({s1_samples:,} S1s against full pool)...", flush=True)
        prepare_training_data(s1_samples)
    else:
        print(f"\n[PHASE 1] Found existing prepared data at {train_ready_dir}.", flush=True)

    # Step 2: Run GPU Training Pipeline
    print("\n[PHASE 2] Starting GPU Model Training & Checkpointing...", flush=True)
    run_pipeline()

    print(f"\nALL TASKS COMPLETED IN {time.time()-t0:.1f}s!", flush=True)
