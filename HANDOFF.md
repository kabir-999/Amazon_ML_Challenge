# Business Entity Resolution — Handoff (Team "Broke Code")

Read this first if you are a new Claude/person picking up the project. Problem statement: `PS.txt` (also `student_resource/README.md`).

## 1. Task in one paragraph
Link records of Source 1 (clean reference, S1) to matching records in noisy Source 2 / Source 3 (S2/S3). Output per S1 entity: comma-separated S2/S3 ids (empty = singleton). Metric: **macro F0.5 over S1 entities**, singletons count (empty correct = 1.0, any wrong match on a singleton = 0). Train has US + India with labels; **test adds France (unseen, no labels)**. Rules: no external data/APIs, final model MIT/Apache and <= 8B params. Files are TSV.

Data sizes (records S1 / S2 / S3): train 2.2M / 5.0M / 5.3M; test 1.7M / 4.9M / 5.1M. Test S1: India 810k, US 663k, France 259k.

## 2. Key data facts (see `eda/eda_report.html`)
- S1 is clean. S2: ALL-CAPS addresses (66%), 15% non-ASCII names (Devanagari/Tamil/Bengali…), URL-style names ("kéystonebiomedical.com"), phone-number/symbol junk. S3: mixed-case addresses, truncated names, full state names.
- Ground truth: 5.6% singletons, avg 3.46 matches/S1 (1.67 from S2, 1.79 from S3). **Each S2/S3 record matches at most one S1.** ~27% of S2/S3 rows are orphans (match nothing).
- 38% of S1 names repeat (chains) so name alone is not an identifier; **address (house number + street + city) is the strongest key**. Some true matches have a totally different name (only the address matches).
- India: name similarity of true pairs is low (transliteration), address similarity high. ~5% of S2/S3 addresses empty.

## 3. What we built: version_1 (submitted)
Code: `version_1/src/` (run order in `version_1/README.md`). Pipeline per country:
1. `normalize.py` — lowercase, `unidecode` (also transliterates Indic scripts, crude), legal-suffix map (Pvt/Private, Ltd/Limited, Inc/Incorporated…), domain stripping, phone-junk removal, street abbreviations, US state name -> abbr, Indian state abbr map, house-number/ZIP/PIN extraction, landmark stop-words.
2. `blocking.py` — sparse TF-IDF top-15 per channel: address tokens, name char 3-4-grams (no spaces), name words; union, ranked by `addr + 0.7*nchar + 0.5*nword`, max 40 candidates/S1. Common tokens pruned with df cap `clip(frac*pool, floor, cap)`. Runs per country, parallel via fork.
3. `features.py` — 28 rapidfuzz name/address features + context = **43 features**. Country is deliberately NOT a feature. Competition features: rank/gap of a pair among S1's candidates and among S1s competing for the same S2/S3 record.
4. `train_final.py` — LightGBM + XGBoost + MLP (torch MPS), mean-probability ensemble. Threshold tuned for F0.5 on val = 0.75. Decision: keep p >= thr, then **each S2/S3 record goes only to its highest-scoring S1**.
5. `predict_test.py` — scores test candidates, writes per-country files (merged later).

Dev data: 300k random train S1 entities + all their matches + proportional orphan sample (pool ~1.4M, i.e. **~6x sparser than the real test pool**). Split by S1: 70/15/15.

### Results (dev set, held-out 15%)
| Model | F0.5 |
|---|---|
| LightGBM | 0.9696 |
| XGBoost | 0.9713 |
| MLP | 0.9679 |
| **Ensemble (submitted)** | **0.9707** |
US 0.978, India 0.960; pair precision 0.991, recall 0.940. Blocking pair recall: US 98.7%, India 95.9%.

### Real leaderboard score: **0.88** (vs 0.971 estimated) — the gap is the problem to solve.
Test prediction stats: singleton rate US 3.9% / India 6.6% / France 7.5% (train truth 5.6%); avg matches/S1 3.37 / 3.18 / 3.03 (train 3.46). Test candidates: 61.3M pairs, 35.4/S1.

Submission files: `version_1/upload`-equivalent = `upload/matching_results.tsv` and `upload/code.zip`; full package `Broke_Code_submission.zip` (includes `candidate_pairs.tsv`, docs). Validator passes (`student_resource/utils/validate_submission.py`, also with `--check-ids`).

## 4. Why we think 0.88 < 0.97 (hypotheses, unverified)
1. **Density shift**: dev pool 6x smaller than test -> far fewer look-alike distractors; threshold 0.75 tuned on it is probably too loose; competition features (S1s per S2/S3 record) were computed with only 300k S1s, test has all 1.7M.
2. **France**: unseen, ~15% of test S1; only 85% of France S1 had a strong top address match (US 94%); French suffixes/address terms only partly handled.
3. **India blocking recall** may be lower at full scale.
Which one dominates is unknown — the leaderboard only gives one number.

## 5. Next iteration — version_2 plan (do this on the bigger machine)
Folder convention: every new training run gets its own `version_<n>/` with `src/ data/ out/ upload/`; the final `matching_results.tsv` + `code.zip` for that iteration go in `version_<n>/upload/`. `version_2/src/` already holds a copy of v1 code plus `prep_trainfull.py` (nothing trained yet, `version_2/data` empty).

**Step 1 — test-density training data (most important).** Use ALL 2.2M train S1 against the FULL train S2/S3 pool (`prep_trainfull.py` does the normalisation into `data/trainfull/<country>_{s1n,pooln}.parquet` + `gt.parquet`; then `run_pipeline_country.py trainfull US|India`, which computes candidates, labels `y`, and the fuzzy-feature memmap). This makes candidate counts, chains and competition identical in nature to the test run. Expect ~79M pairs; memmap features ~9 GB disk; ~1 h+ on the M4 Air, much less with more RAM/cores. Train on pairs of a subset of S1 (e.g. 40–50%) but compute competition context over all of them; assemble features by chunk and keep only chosen S1 rows to bound memory.
**Sanity check first:** score the existing v1 models (`version_1/out/{lgb.txt,xgb.json,mlp.pt,meta.pkl}`) on this full-density set. If F0.5 ≈ 0.88 the density hypothesis is confirmed (France then is fine); if it is still ~0.97, France is the main culprit.

**Step 2 — retrain + retune** on full-density data: same 3 models (bigger GPU: larger MLP, or a small char/transformer cross-encoder on the borderline pairs only, MIT/Apache, <= 8B, no external data), re-tune the threshold at full density (expect a higher threshold; each false merge costs ~2x a miss). Report per-country/per-source F0.5.

**Step 3 — extra features / stage 2 (expected +1–2 pts):**
- Ambiguity features per S1: number of candidates with name >= 90 / address >= 90 / both; pool-side name and address frequency (chain-ness; avoid S1-side counts unless all S1 are present).
- Stage-2 re-ranker: features = stage-1 probability, its rank among the S1's candidates (per source), gap to top, best competing S1's probability for the same record, number of candidates with p > 0.5; train on out-of-fold stage-1 probabilities (K-fold by S1 entity).
- Cross-source consistency: an S3 candidate that agrees with an already-confident S2 match of the same S1.
- IDF-weighted name overlap; better transliteration handling for Indic scripts (name is unreliable there, rely on address); handle the case where the S2/S3 address is missing (only accept on very strong name evidence).

**Step 4 — France.** No labels, so make normalisation generic: French suffixes (SARL, SAS, SA, SCI, EURL…) already in the legal list; add address terms (rue, avenue, boulevard, chemin, place; "R." -> rue; "Bd"), postcode (5 digits) handling, region/department names. Sanity-check France output: singleton rate and avg matches per S1 should look like train (5.6%, 3.46); a large deviation means the model is under/over-merging there. Optionally use a slightly stricter threshold for countries with less blocking confidence.

**Step 5 — blocking recall.** India 95.9% / US 98.7% at dev density; retest at full density and, if low, add channels (e.g. house number + first street token key, phonetic name key) or raise K for India.

**Quick win to try before all this (15 min):** raise the threshold (0.85 / 0.9) on the existing test features (`version_1/data/test/<country>_F.npy` + `_cand.parquet`) and resubmit to see if the leaderboard moves.

## 6. Practical notes / gotchas
- Machine used: MacBook Air M4, 16 GB. Memory is the bottleneck: process per country, never build a 60M-row × 43 float64 frame.
- macOS uses `spawn` for multiprocessing → guard with `if __name__ == "__main__"`; blocking uses `fork` context on purpose. macOS `sed -i` needs a suffix (`-i.bak`); prefer Python for edits.
- Cached data lives under `version_1/data/` (~10 GB, regenerable in ~2.5 h). `config.py` paths derive from the script location; `ER_DATA_DIR` env var overrides dataset path.
- Do not use country as a feature (France is unseen). No external data or pretrained models (fair-play rule) — `unidecode` and `rapidfuzz` are local libraries only.
- Portal uploads: `matching_results.tsv` (leaderboard) + code zip; final package zip `<team>_submission.zip` needs `output/`, `code/business_entity_resolution/{src,README.md,requirements.txt}`, `Documentation_template.md` (filled: team "Broke Code", members Kabir Mathur, Aagnya Mistry, Aishwarya Abhijit Deshmukh, Aayush Chaudhari, DJSCE Mumbai).

## 7. Repo map
```
PS.txt                       problem statement
eda/eda_report.html          EDA report (+ compute_stats.py)
version_1/src, out, data     submitted pipeline, models (out/lgb.txt xgb.json mlp.pt meta.pkl), cached data
version_2/src                v1 code + prep_trainfull.py, ready for the next iteration (nothing trained)
upload/                      files uploaded for v1: matching_results.tsv, code.zip
Broke_Code_submission.zip    full final package for v1
```
