# Machine Learning Solution Approach: Business Entity Resolution
**Amazon ML Challenge 2026**  
**Repository Branch:** `Aayush`  
**Team / Implementation:** Version 3 Pipeline (`version_3/`)

---

## 1. Executive Summary & Problem Formulation

### 1.1 Objective
The task is to perform large-scale **Entity Resolution (ER)** across three disparate data sources:
- **Source 1 ($S_1$):** Deduplicated reference ground truth records.
- **Source 2 ($S_2$) and Source 3 ($S_3$):** Noisy, unstructured records with missing values, abbreviations, and multi-script representations.

For every single reference entity in $S_1$, the system must identify all matching records in $S_2$ and $S_3$ that refer to the exact same real-world business.

```
                  ┌───────────────────────────────┐
                  │    Source 1 (Reference S1)    │
                  └──────────────┬────────────────┘
                                 │ (1-to-many mapping)
                     ┌───────────┴───────────┐
                     ▼                       ▼
           ┌──────────────────┐    ┌──────────────────┐
           │   Source 2 (S2)  │    │   Source 3 (S3)  │
           │  (Noisy Records) │    │  (Noisy Records) │
           └──────────────────┘    └──────────────────┘
```

### 1.2 Mathematical & Topological Constraints
1. **1-to-Many Mapping ($S_1 \to S_2, S_3$):** A single $S_1$ business can link to zero, one, or several records across $S_2$ and $S_3$ (average is ~3.46 matches in ground truth).
2. **Mutual Exclusivity ($S_2, S_3 \to S_1$):** Each record in $S_2$ and $S_3$ can correspond to **at most one** $S_1$ business.
3. **Singletons:** ~5.6% of $S_1$ entities have **zero matches** in $S_2$ and $S_3$.
4. **Orphans:** ~25%–27% of $S_2$ and $S_3$ records belong to no reference $S_1$ entity.
5. **Evaluation Metric (Macro $F_{0.5}$):**
   $$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
   - Computed as an unweighted macro-average over **all individual $S_1$ entities**.
   - Precision is weighted $2\times$ more heavily than recall (false merges heavily penalize the score).
   - Singletons are strictly scored: predicting an empty list gives $1.0$, while predicting any false match yields $0.0$.

---

## 2. Dataset Overview & Key Challenges

### 2.1 Scale of Data
| Dataset Split | Source 1 ($S_1$) | Source 2 ($S_2$) | Source 3 ($S_3$) | Total Pool ($S_2+S_3$) |
|---|---|---|---|---|
| **Training** | 2,206,821 | 5,034,616 | 5,285,603 | **10,320,219** |
| **Test** | 1,732,544 | 4,887,273 | 5,082,316 | **9,969,589** |

Naive pair comparison between test $S_1$ and the test pool requires:
$$1.73 \times 10^6 \times 9.97 \times 10^6 \approx 1.7 \times 10^{13} \text{ pairs (Computationally Impossible)}$$

### 2.2 Core Data Characteristics & Domain Noise
1. **The "Chain" Ambiguity (38.3% duplicate names):**
   Over 38% of $S_1$ businesses share names with other businesses (e.g., franchises, bank branches, retail chains). Name similarity alone is insufficient; **address (house/plot number, street, city, postal code) is the primary discriminator**.
2. **Multi-Script & Transliteration (India):**
   ~15% of $S_2$ and $S_3$ records in India use Indic scripts (Devanagari, Tamil, Bengali, Telugu) while $S_1$ uses English/Latin transliterations.
3. **Unseen Country Zero-Shot Generalization (France):**
   The training set only contains `US` and `India`. The test set introduces **`France`** (~15% of test data, 259,452 entities). All models and features must remain country-agnostic.
4. **Missing Addresses:**
   ~3.3% of $S_2$ and $S_3$ records have completely empty address fields.

---

## 3. Root Cause Analysis of Previous Attempt (v1 Score: 0.88)

In the initial pipeline (`version_1`), the local dev validation score was **0.971**, but the leaderboard score dropped to **0.88**. Our diagnosis identified three primary causes:

1. **The Blocking Recall Collapse (Primary Cap):**
   - The blocking stage used rigid document-frequency caps (`max_df` capped at 20k documents). In a 4.1M pool, this eliminated legitimate city and street names.
   - Blocking recall dropped to **88.2% on India** at full scale. Because a classifier cannot predict matches outside its candidate set, **blocking recall forms a hard mathematical ceiling on $F_{0.5}$**.
2. **Pool Distractor Density Shift:**
   - Dev was trained on a downsampled pool of 1.4M records (~6× sparser than the real 10M test pool).
   - In a 6× denser pool, the number of competing look-alike distractors skyrocketed, throwing off the competition features (`comb_gap_top`, `comb_rank_ip`) and causing the decision threshold (0.75) to be overly conservative.
3. **Severe Under-Linking on Test:**
   - In training ground truth, ~74% of pool records are matched.
   - The v1 submission linked only ~56% of test pool records, leaving nearly a quarter of true matches unpredicted.

---

## 4. The Complete Version 3 Pipeline Architecture (`version_3/`)

The enhanced solution implements a modular, high-recall, GPU-accelerated pipeline:

```
[Raw TSV Data] ──► 1. Multilingual Normalization (normalize.py)
                          │
                          ▼
                   2. Query-Projected Multi-Channel Blocking (blocking.py)
                          │  (TF-IDF + Exact Hash Indices)
                          ▼
                   3. 47-Dimensional Feature Extraction (features.py)
                          │  (Fuzzy string + House number + Competition context)
                          ▼
                   4. Triplet GPU Ensemble Training (train_gpu.py)
                          │  (LightGBM + XGBoost + PyTorch CUDA Deep MLP)
                          ▼
                   5. Threshold Optimization & Mutual Exclusivity Assignment
                          │
                          ▼
                   [matching_results.tsv & candidate_pairs.tsv]
```

---

### Step 1: Multilingual Normalization (`version_3/src/normalize.py`)
Prepares raw names and addresses for both fuzzy and exact indexing:
- **Legal Form Canonicalization:** Normalizes legal entity suffixes across English, Indian, and French jurisdictions:
  - English/US: `incorporated` $\to$ `inc`, `corporation` $\to$ `corp`, `limited` $\to$ `ltd`, `private` $\to$ `pvt`, `company` $\to$ `co`, `llc` $\to$ `llc`.
  - French: `sarl` $\to$ `sarl`, `societe anonyme` $\to$ `sa`, `sas` $\to$ `sas`, `eurl` $\to$ `eurl`, `sci` $\to$ `sci`.
- **Street & Address Standardizations:**
  - Standardizes road types: `rd` $\to$ `road`, `st` $\to$ `street`, `blvd`/`bd` $\to$ `boulevard`, `r`/`r.` $\to$ `rue`, `av` $\to$ `avenue`, `ch` $\to$ `chemin`.
  - Indian geographic terms: `nagar`, `marg`, `colony`, `chowk`, `sector`, `phase`.
- **State Dictionaries:** Bidirectional mapping for all 50 US states and all 28 Indian states/UTs.
- **Entity Extractions:**
  - PIN / ZIP code extractor (5-digit US, 6-digit India, 5-digit France).
  - Alphanumeric house/plot number extractor (`401-402`, `Plot No. 23`, `19404C`).
- **Explicit Missing Address Flag:** Computes `ad_has = 0/1` so downstream models recognize missing addresses instead of treating them as zero similarity.

---

### Step 2: High-Recall Multi-Channel Blocking (`version_3/src/blocking.py`)
To ensure high recall without exceeding memory limits on a 16 GB RAM machine:

#### A. Query-Vocabulary Projection
Instead of fitting large $n$-gram vocabularies across 6.2M pool documents (which allocates >20 GB in Python lists), we project onto the query vocabulary:
1. Fit vectorizer vocabulary on $S_1$ reference queries (bounded to ~20k–70k relevant terms).
2. Transform the 6M+ candidate pool in memory-bounded batches (1,000,000 records).
3. Compute top-$K$ cosine matches in query chunks of 250 queries.

#### B. Channels Used
1. **Address Channel (`addr`):** Word-level TF-IDF on standardized address tokens (`sublinear_tf=True`).
2. **Name Substring Channel (`nchar`):** Character 3-4 grams without spaces (`nm_nospace`) to capture typos, stem variations, and transliteration noise.
3. **Core Name Channel (`nword`):** Word-level TF-IDF on core business names (excluding legal suffixes).
4. **Deterministic Hash Indices:** Inverted hash tables for exact joins:
   - `(PIN/ZIP + first house number)`
   - `(state + house number)`
   - `(first 8 chars of name + state)`

#### C. Candidate Merging & Ranking
For each $S_1$ entity, top-$K$ candidates ($K=30$) from each channel are unified and scored:
$$\text{blk\_score} = 1.2 \times \text{addr} + 0.8 \times \text{nchar} + 0.5 \times \text{nword} + 0.4 \times \text{hash\_key}$$
The top **60 candidates per $S_1$** are preserved.

**Recall Achieved against Full 10.3M Candidate Pool:**
- **US:** **96.56% pair recall** (10,010 / 10,367 true matches retrieved)
- **India:** **92.85% pair recall** (6,438 / 6,934 true matches retrieved)

---

### Step 3: Feature Engineering (`version_3/src/features.py`)
Constructs **47 dense numerical features** for every candidate pair $(S_1, S_{\text{pool}})$:

1. **String Distance Metrics (RapidFuzz):**
   - Token Set Ratio, Token Sort Ratio, Partial Ratio, Ratio.
   - Jaro-Winkler Similarity & Normalized Levenshtein Distance.
   - Word Jaccard Similarity and Word Containment.
   - Exact string match flag, First token equality, Substring containment.
   - Length ratio between reference name and candidate name.
   - Legal suffix equality (`nm_legal_eq`).
2. **Address & Geographic Features:**
   - Full address token Jaccard & Containment.
   - Alphanumeric-only token Jaccard & Containment.
   - House number Jaccard, First house number match, Any house number overlap.
   - Postal code agreement (`ad_zipeq`), Postal code disagreement (`ad_zipne`).
   - State code agreement (`ad_steq`), State code disagreement (`ad_stne`).
   - RapidFuzz address token set & token sort ratios.
3. **Missingness & Noise Indicators:**
   - Candidate address missing flag (`ad_missing_c`).
   - Reference has house number but candidate has none (`ad_num_missing_c`).
   - Non-Latin script indicator (`c_nonlatin`).
   - Domain-name-derived entity name indicator (`c_dom`).
4. **Global Competition & Ambiguity Context:**
   - Composite similarity score:
     $$\text{comb} = 0.5 \times \frac{\text{nm\_tset}}{100} + 0.5 \times \frac{\text{ad\_tset}}{100} + 0.2 \times \text{ad\_numany}$$
   - `comb_rank_s1`: Rank of this candidate among all candidates for this $S_1$.
   - `comb_gap_top`: Difference in score between this candidate and the #1 candidate of this $S_1$.
   - `comb_rank_ip`: Rank of this $S_1$ among all reference entities competing for this same pool record.
   - `comb_gap_ip`: Score gap between this $S_1$ and the highest competing $S_1$ for this pool record.
   - `n_strong_ip`: Number of reference entities showing strong similarity ($\ge 0.85$) to this pool record.
   - `pool_name_freq` & `pool_addr_freq`: Log-frequency of the pool record's name and address (captures chains).

---

### Step 4: Triplet GPU Ensemble Training (`version_3/src/train_gpu.py`)

#### 1. Split Strategy
- Group split strictly by **$S_1$ entity ID** to prevent data leakage:
  - **Train (70%):** 209,921 pairs
  - **Validation (15%):** 44,988 pairs
  - **Held-Out Test (15%):** 44,990 pairs

#### 2. Models in Ensemble
1. **LightGBM Classifier:**
   - 1200 trees, learning rate 0.05, 127 leaves, feature subsampling 0.8, early stopping (50 rounds).
2. **XGBoost Classifier:**
   - 800 trees, learning rate 0.06, max depth 9, histogram tree method (`tree_method='hist'`), early stopping (50 rounds).
3. **PyTorch Deep ER Neural Network (GPU Accelerated on CUDA):**
   - Hardware: **NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM)**.
   - Architecture:
     $$\text{Input (47)} \to \text{Linear}(256) \to \text{BatchNorm} \to \text{ReLU} \to \text{Dropout}(0.15) \to \text{Linear}(128) \to \text{BatchNorm} \to \text{ReLU} \to \text{Dropout}(0.15) \to \text{Linear}(64) \to \text{BatchNorm} \to \text{ReLU} \to \text{Linear}(1)$$
   - Optimization: AdamW optimizer, BCEWithLogitsLoss, OneCycleLR scheduler (max LR $4 \times 10^{-3}$), batch size 8192, 10 epochs.
   - Live terminal logging prints per-epoch train BCE, validation BCE, learning rate, and elapsed time.

#### 3. Ensembling
Predictions are combined via soft probability voting:
$$P_{\text{ensemble}} = \frac{P_{\text{LightGBM}} + P_{\text{XGBoost}} + P_{\text{PyTorch\_MLP}}}{3}$$

---

### Step 5: Post-Processing & Mutual Exclusivity Assignment
To maximize Macro $F_{0.5}$ and respect the ground truth topology:
1. **Threshold Optimization:** Grid search $\text{thr} \in [0.35, 0.95]$ on the validation set, directly evaluating the official Macro $F_{0.5}$ metric (including singletons).
2. **Greedy Mutual Exclusivity Assignment:**
   - Pairs with $P \ge \text{thr}$ are sorted in descending order of confidence.
   - Each pool record ($S_2$ or $S_3$) is assigned strictly to its **highest-scoring $S_1$ entity**. Duplicate assignments of the same pool record are discarded.
3. **Singleton Emission:** Reference entities with zero predicted pairs above threshold are emitted as singletons (empty string in output), scoring a full $1.0$ if truly singletons.

---

## 5. Experimental Results & Verification

Evaluated on the held-out test split (unseen during training and threshold tuning):

| Model | Val Macro $F_{0.5}$ | Optimal Threshold | Held-Out Test Macro $F_{0.5}$ | Average Precision (AP) |
|---|---|---|---|---|
| **LightGBM** | 0.9292 | 0.65 | **0.9313** | **0.9895** |
| **XGBoost** | 0.9282 | 0.70 | **0.9296** | **0.9898** |
| **PyTorch Deep MLP (CUDA)** | 0.9109 | 0.75 | **0.9145** | **0.9814** |
| **Ensemble (3-way)** | 0.9255 | 0.70 | **0.9292** | **0.9886** |

### Per-Country & Quality Metrics:
- **US Test Macro $F_{0.5}$:** **0.9436**
- **India Test Macro $F_{0.5}$:** **0.9087**
- **Pair Precision:** **97.86%**
- **Pair Recall:** **85.92%**

All model checkpoints and metadata are saved under `version_3/out/`:
- `lgb_model.txt`
- `xgb_model.json`
- `mlp_model.pt`
- `meta.pkl`
- `training_report.json`

---

## 6. How to Run & Reproduce

### 1. Environment Setup
```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
# (or install torch with CUDA support from pytorch.org)
```

### 2. Prepare Data & Train
```powershell
# Prepares S1 sample against the full 10.3M pool and runs GPU training
.\.venv\Scripts\python.exe version_3\src\run_all.py 100000
```

### 3. Generate Submission Files
```powershell
# Runs inference on test_source1/2/3.tsv and creates output TSVs
.\.venv\Scripts\python.exe version_3\src\predict_submission.py
```

### 4. Validate Submission
```powershell
python student_resource\utils\validate_submission.py `
    --matching version_3\out\matching_results.tsv `
    --candidate version_3\out\candidate_pairs.tsv `
    --test-dir student_resource\dataset\test
```
