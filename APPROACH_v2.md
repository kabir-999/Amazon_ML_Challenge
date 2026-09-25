# Business Entity Resolution — Finalized Approach (v2)

Team "Broke Code" · Amazon ML Challenge · plan finalized 2026-09-25

**Constraints:** MacBook Air M4 (16 GB), < 3 days to deadline, MIT/Apache-2.0 models ≤ 8B params, no external data lookup.
**Starting point:** v1 leaderboard = **0.88** (dev estimate was 0.971). Leaderboard top ≈ 0.98. **Target: ≥ 0.96.**

---

## 1. Why v1 underperformed (recap)

| Cause | Evidence | Fix in v2 |
|---|---|---|
| Blocking recall collapses at real scale | India pair recall 95.9% (dev) → **88.2%** (full density) | Bigger top-K, relaxed df caps, exact-key channels (§4) |
| Dev set was ~6× sparser than test | Threshold + competition features tuned on an "easy" pool | Train, validate and tune at **full density** (§5, §6) |
| Too few matches on test | 56% of test S2/S3 records linked vs ~74% in train | Both fixes above; sanity checks in §8 |

Precision was fine; **recall is the problem**, and most of it is lost in blocking, before any model runs.

---

## 2. Pipeline overview

```
TSV ──► Parquet ──► Normalize ──► Blocking (multi-channel union) ──► Features ──► GBDT ensemble
                                                                                     │
                                  optional: BERT re-score of uncertain pairs ◄───────┤
                                                                                     ▼
                                              Threshold + "each S2/S3 → best S1" assignment
                                                                                     ▼
                                              matching_results.tsv + candidate_pairs.tsv
```

---

## 3. Data format & batching

### 3.1 Storage
- Convert the raw `.tsv` files **once** to **Parquet** (columnar, compressed, 3–5× smaller, much faster to load). Do **not** convert to CSV: same size as TSV, and addresses contain commas.
- Always read the TSVs with `sep="\t"`. Write the final submission files back as TSV.

### 3.2 Batching rule
> Anything that **searches for candidates** or **compares competing S1s** must see the **whole pool of that country**. Only S1 queries and per-pair work may be chunked.

```
for country in sorted(unique country labels in data):     # open set, never hard-code {US, India}
    index ← ALL S2 + S3 records of this country             # never split
    for S1_chunk in chunks(S1 of this country, ~50k):       # safe to split
        candidates → features → model scores → save to disk
    assignment step over the full country                   # needs all chunks done
```

- **Verify once on train:** true matches never cross countries (if any do, drop the country split for blocking).
- Process one country at a time; never build a full 60M-row float64 frame (use float32 memmaps).

---

## 4. Blocking (candidate generation)

**Goal: pair recall ≥ 97% for India and US, measured at full density, with ~60–100 candidates per S1.**

Union of the following channels, merged and ranked, then capped at `max_cands`:

| # | Channel | Type | Change vs v1 |
|---|---|---|---|
| 1 | Address tokens | TF-IDF sparse top-K | top-K 15 → **40**; relax df cap so big-city / common-street tokens stay indexed |
| 2 | Name char 3–4-grams (no spaces) | TF-IDF sparse top-K | top-K 15 → **40**; relax df cap |
| 3 | Name words | TF-IDF sparse top-K | top-K 15 → **40** |
| 4 | `PIN/ZIP + house number` | exact-key hash join | **new** |
| 5 | `house number + first street token + city token` | exact-key hash join | **new** — handles reordered addresses (`"OH, Columbus, 5559 Orville Avenue"`) |
| 6 | `name no-space prefix (8 chars) + state` | exact-key hash join | **new** — handles thin / empty addresses |

- Merged ranking: keep v1's weighted score (`addr + 0.7·nchar + 0.5·nword`) plus a bonus for exact-key hits; `max_cands` 40 → **80–100**.
- Exact-key channels: skip keys shared by too many pool records (e.g. > 200), otherwise they explode.
- **MinHash-LSH: not used.** It approximates the same token overlap the TF-IDF channels already cover, without IDF weighting, and common tokens create giant buckets. Not worth the time within 3 days.
- **Embedding/FAISS channel: not used** (compute budget on M4 Air).

### 4.1 Tuning procedure (do this first)
1. Sample **30k India S1** from `trainfull`; keep the **full 4.1M India pool**.
2. Measure pair recall and recall@rank (40 / 60 / 100) for each change above, one at a time.
3. Inspect ~15 missed true pairs to find the remaining failure types (Indic-script names with thin addresses, truncated addresses, house-number digit drops).
4. Fix the configuration, then repeat once on a 30k **US** sample.

---

## 5. Features & training data

- **Training data at full density:** all train S1 of a country vs. the full train pool of that country (`prep_trainfull.py` → `run_pipeline_country.py trainfull <country>`).
- To fit 16 GB and the time budget: **train on the pairs of a ~20–30% sample of S1**, but compute competition / context features over **all** S1 so their distributions match test.
- Features: v1's 43 (rapidfuzz name/address similarities, number/PIN agreement, rank/gap among candidates, competition among S1s for the same S2/S3 record) plus the cheap v2 additions from `features_v2.py`:
  - count of candidates per S1 with name ≥ 90 / address ≥ 90 / both (ambiguity)
  - pool-side frequency of the name and the address (chain-ness)
  - flag: S2/S3 address empty → model learns to require strong name evidence
- **Country is not a feature** (France is unseen).
- Split **by S1 entity**: train / val (threshold) / holdout, with every split at full density.

---

## 6. Matching model

### 6.1 Stage 1 — GBDT ensemble (required)
- LightGBM + XGBoost (drop the MLP if time is short; it added nothing over XGB in v1).
- Mean probability; ~700 trees, early stopping on val.
- Stage-2 out-of-fold re-ranker from `train_v2.py`: **only if** Stage 1 finishes by the end of Day 2.

### 6.2 Stage 2 — BERT re-score of uncertain pairs (optional, Day 3)
- **Allowed?** The rules require MIT/Apache-2.0 and ≤ 8B params and ban external *data lookup*. Pretrained weights appear allowed. **Confirm on the challenge forum before submitting** and record the model licence in the documentation.
- Model: a small encoder, e.g. `sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0, 22M) or `paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0, 118M; better for Indic scripts, slower). Verify the licence on each model card.
- Ditto-style cross-encoder input: `name | address [SEP] name | address` → match / no-match. Fine-tune on ~200k train pairs drawn from the uncertain band, on MPS.
- Apply **only** to pairs with Stage-1 `0.2 < p < 0.9`; final score = blend of GBDT and BERT probabilities, weight tuned on val.
- **Go / no-go:** keep it only if val F0.5 improves by ≥ 0.003 **and** test inference on the uncertain band fits in the remaining time. Otherwise submit Stage 1.

---

## 7. Decision step

1. Keep pairs with `p ≥ threshold`. **Tune the threshold for macro F0.5 on the full-density val split** (expect higher than v1's 0.75, since a false merge costs ~2× a miss).
2. **Each S2/S3 record goes to at most one S1**: the highest-scoring one (true in all of train). Run this per country after all S1 chunks are scored.
3. S1 with no surviving pair → empty list (singleton; worth 1.0 if correct).
4. Optionally a slightly stricter threshold for France (lower blocking confidence, no labels).

---

## 8. Validation & sanity checks

| Check | Expected |
|---|---|
| Blocking pair recall (full density), India / US | ≥ 97% |
| Val macro F0.5 (full density), per country | report US and India separately |
| Test: fraction of S2/S3 linked to some S1 | ~70%+ (v1: 56%) |
| Test: avg matches per S1 | ~3.5–4.3 |
| Test: singleton rate, per country (incl. France) | ~5–7% |
| `validate_submission.py --check-ids` | PASS |
| Every matched ID also appears in `candidate_pairs.tsv` | yes |
| `candidate_pairs.tsv` = the **final** candidate list the model scored | yes |

---

## 9. Schedule (< 3 days, M4 Air)

| When | Task | Output |
|---|---|---|
| **Day 1 AM** | Parquet conversion; cross-country check; blocking-recall experiments on 30k India sample (§4.1) | chosen blocking config |
| **Day 1 PM** | Full-density blocking + features for train India & US (run overnight if needed) | `trainfull/<country>_{cand,F,y}` |
| **Day 2 AM** | Train LGB + XGB on the S1 sample; tune threshold; per-country val F0.5 | models + threshold |
| **Day 2 PM** | Test blocking + features + prediction for US, India, France; checks in §8; **submit** (safe checkpoint) | first v2 leaderboard score |
| **Day 3** | Optional: BERT Stage 2 (§6.2) **or** stage-2 re-ranker; resubmit only if val improves | improved submission |
| **Day 3 end** | Final package: `output/`, `code/business_entity_resolution/{src,README.md,requirements.txt}`, filled `Documentation_template.md` | `Broke_Code_submission.zip` |

**Rule:** always keep one validated submission in hand. Never overwrite it with an experiment that hasn't been scored on val.

---

## 10. Risks & fallbacks

| Risk | Fallback |
|---|---|
| Full-density features don't fit in 16 GB | Smaller S1 chunks; float32 memmap; train on a 20% S1 sample |
| Bigger blocking makes test feature time too long | `max_cands` 100 → 70; drop the weakest channel |
| BERT deemed not allowed / too slow | Skip it; Stage 1 alone is the submission |
| France output looks off (singleton rate far from ~6%) | Stricter France threshold; check French address normalisation (rue / av / bd, 5-digit postcode) |
