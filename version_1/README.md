# Version 1 — blocking + pair classifier (dev subset)
Run from `src/`:
1. `python make_subset.py`   – 60k train S1 entities + their matches + proportional orphans
2. `python run_blocking.py`  – normalise, TF-IDF blocking (addr / name-char / name-word), prints recall
3. `python run_features.py`  – 47 pair features (rapidfuzz, multiprocessing)
4. `python train_eval.py`    – LightGBM, XGBoost, MLP (MPS GPU), ensemble; F0.5 threshold tuned on val; report in `out/report.json`
Split is by S1 entity: 70% train / 15% val (threshold) / 15% test.
