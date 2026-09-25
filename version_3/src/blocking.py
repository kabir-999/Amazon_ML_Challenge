"""High-recall candidate generation (blocking) across multilingual sources.
Combines sparse TF-IDF cosine retrieval with exact-key indexing.
Uses sampled-vocabulary vectorization and chunked query retrieval for zero MemoryErrors on 10M+ pools."""
import gc
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

TFIDF_CHANNELS = [
    ("addr", "ad_toks", dict(analyzer="word", token_pattern=r"\S+", sublinear_tf=True), 0.02),
    ("nchar", "nm_nospace", dict(analyzer="char", ngram_range=(3, 4), sublinear_tf=True), 0.01),
    ("nword", "nm_core", dict(analyzer="word", token_pattern=r"\S+", sublinear_tf=True), 0.01),
]


def build_hash_indices(pool):
    """Build fast inverted index lookups for high-precision exact keys."""
    zip_num_idx = defaultdict(list)
    st_num_idx = defaultdict(list)
    nm_st_idx = defaultdict(list)

    zips = pool.ad_zip.values
    nums = pool.ad_nums.values
    states = pool.ad_state.values
    nms = pool.nm_nospace.values

    for idx in range(len(pool)):
        z = zips[idx]
        nu = nums[idx].split()[0] if nums[idx] else ""
        st = states[idx]
        nm8 = nms[idx][:8] if len(nms[idx]) >= 6 else ""

        if z and nu:
            zip_num_idx[(z, nu)].append(idx)
        if st and nu:
            st_num_idx[(st, nu)].append(idx)
        if nm8 and st:
            nm_st_idx[(nm8, st)].append(idx)

    max_bucket = 50
    zip_num_idx = {k: v for k, v in zip_num_idx.items() if len(v) <= max_bucket}
    st_num_idx = {k: v for k, v in st_num_idx.items() if len(v) <= max_bucket}
    nm_st_idx = {k: v for k, v in nm_st_idx.items() if len(v) <= max_bucket}

    return zip_num_idx, st_num_idx, nm_st_idx


def block_country(g1, gp, k=30, max_cands=60, chunk=250, verbose=True):
    """Generates candidate pairs for reference g1 against candidate pool gp.
    Returns DataFrame: [i1, ip, addr, nchar, nword, hash_key, blk_score, blk_rank]"""
    n_s1 = len(g1)
    n_pool = len(gp)
    t0 = time.time()

    # Step 1: Compute TF-IDF sparse similarity matrices per channel
    channel_sims = {}
    for name, col, kw, frac in TFIDF_CHANNELS:
        sample_size = min(400_000, n_pool)
        sample_docs = gp[col].sample(sample_size, random_state=42)
        sample_max_df = max(10, int(frac * sample_size))

        v = TfidfVectorizer(dtype=np.float32, max_df=sample_max_df, min_df=5, **kw)
        v.fit(sample_docs)
        del sample_docs
        gc.collect()

        Q = v.transform(g1[col])
        if verbose:
            print(f"  [{name}] vocab: {len(v.vocabulary_):,}, max_df: {sample_max_df:,}", flush=True)

        CHUNK_TRANSFORM = 1_000_000
        P_chunks = []
        for s in range(0, n_pool, CHUNK_TRANSFORM):
            e = min(s + CHUNK_TRANSFORM, n_pool)
            P_chunks.append(v.transform(gp[col].iloc[s:e]))

        PT = sp.vstack(P_chunks).T.tocsr()
        del P_chunks, v
        gc.collect()

        if verbose:
            print(f"  [{name}] pool nnz: {PT.nnz:,}", flush=True)

        # Retrieve top-K in small chunks of 250 queries to keep RAM bounded
        rows_list, cols_list, vals_list = [], [], []
        for start in range(0, n_s1, chunk):
            end = min(start + chunk, n_s1)
            M = (Q[start:end] @ PT).tocsr()
            for r in range(M.shape[0]):
                a, b = M.indptr[r], M.indptr[r + 1]
                if a == b:
                    continue
                d, c = M.data[a:b], M.indices[a:b]
                if len(d) > k:
                    top_idx = np.argpartition(-d, k)[:k]
                    d, c = d[top_idx], c[top_idx]
                rows_list.append(np.full(len(d), start + r, dtype=np.int32))
                cols_list.append(c.astype(np.int32))
                vals_list.append(d)

        del Q, PT
        gc.collect()

        if rows_list:
            channel_sims[name] = (np.concatenate(rows_list), np.concatenate(cols_list), np.concatenate(vals_list))
        else:
            channel_sims[name] = (np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.float32))

    # Step 2: Retrieve exact-key candidates
    if verbose:
        print("  Building exact hash indices...", flush=True)
    zip_num_idx, st_num_idx, nm_st_idx = build_hash_indices(gp)

    zips1 = g1.ad_zip.values
    nums1 = g1.ad_nums.values
    states1 = g1.ad_state.values
    nms1 = g1.nm_nospace.values

    hash_i1, hash_ip = [], []
    for i in range(n_s1):
        z = zips1[i]
        nu = nums1[i].split()[0] if nums1[i] else ""
        st = states1[i]
        nm8 = nms1[i][:8] if len(nms1[i]) >= 6 else ""

        matched_p = []
        if z and nu and (z, nu) in zip_num_idx:
            matched_p.extend(zip_num_idx[(z, nu)])
        if st and nu and (st, nu) in st_num_idx:
            matched_p.extend(st_num_idx[(st, nu)][:15])
        if nm8 and st and (nm8, st) in nm_st_idx:
            matched_p.extend(nm_st_idx[(nm8, st)][:15])

        if matched_p:
            uniq_p = set(matched_p)
            hash_i1.append(np.full(len(uniq_p), i, dtype=np.int32))
            hash_ip.append(np.array(list(uniq_p), dtype=np.int32))

    del zip_num_idx, st_num_idx, nm_st_idx
    gc.collect()

    if hash_i1:
        hash_r = np.concatenate(hash_i1)
        hash_c = np.concatenate(hash_ip)
    else:
        hash_r = np.zeros(0, np.int32)
        hash_c = np.zeros(0, np.int32)

    # Step 3: Combine all candidate pairs via flat index
    npool_64 = np.int64(n_pool)
    all_keys = []
    for name in ("addr", "nchar", "nword"):
        r, c, _ = channel_sims[name]
        all_keys.append(r.astype(np.int64) * npool_64 + c.astype(np.int64))
    if len(hash_r) > 0:
        all_keys.append(hash_r.astype(np.int64) * npool_64 + hash_c.astype(np.int64))

    all_keys = np.concatenate(all_keys)
    unique_keys, inv = np.unique(all_keys, return_inverse=True)
    del all_keys
    gc.collect()

    n_unique = len(unique_keys)
    sim_matrix = np.zeros((n_unique, 4), dtype=np.float32)  # [addr, nchar, nword, hash_key]

    offset = 0
    for col_idx, name in enumerate(("addr", "nchar", "nword")):
        _, _, v = channel_sims[name]
        sim_matrix[inv[offset:offset + len(v)], col_idx] = v
        offset += len(v)

    if len(hash_r) > 0:
        sim_matrix[inv[offset:offset + len(hash_r)], 3] = 1.0

    del channel_sims, hash_r, hash_c, inv
    gc.collect()

    # Step 4: Construct candidates DataFrame and rank
    out = pd.DataFrame({
        "i1": (unique_keys // npool_64).astype(np.int32),
        "ip": (unique_keys % npool_64).astype(np.int32),
        "addr": sim_matrix[:, 0],
        "nchar": sim_matrix[:, 1],
        "nword": sim_matrix[:, 2],
        "hash_key": sim_matrix[:, 3]
    })
    del unique_keys, sim_matrix
    gc.collect()

    # Composite ranking score
    out["blk_score"] = (
        1.2 * out.addr +
        0.8 * out.nchar +
        0.5 * out.nword +
        0.4 * out.hash_key
    )

    out = out.sort_values(["i1", "blk_score"], ascending=[True, False], kind="stable")
    out["blk_rank"] = out.groupby("i1").cumcount().astype(np.int16)

    filtered = out[out.blk_rank < max_cands].reset_index(drop=True)
    if verbose:
        print(f"  Blocking done: {len(filtered):,} candidates ({len(filtered)/max(n_s1, 1):.1f}/S1) in {time.time()-t0:.1f}s", flush=True)

    return filtered
