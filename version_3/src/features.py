"""Feature engineering module for candidate pairs.
Extracts fuzzy string, numerical, missingness, and global competition features.
Optimized for batch vectorization and memory efficiency."""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

FUZZY = [
    "nm_ratio", "nm_tset", "nm_tsort", "nm_partial", "nm_jw", "nm_lev", "nm_jac", "nm_cont", "nm_exact",
    "nm_first", "nm_substr", "nm_lenratio", "nm_legal_eq",
    "ad_jac", "ad_cont", "ad_aljac", "ad_alcont", "ad_numjac", "ad_num0eq", "ad_numany", "ad_zipeq", "ad_zipne",
    "ad_steq", "ad_stne", "ad_tset", "ad_tsort", "ad_missing_c", "ad_num_missing_c"
]

COLS = ["nm_core", "nm_nospace", "ad_toks", "ad_nums", "ad_zip", "ad_state", "nm_legal"]


def _jac(a, b):
    if not a or not b:
        return 0.0
    i = len(a & b)
    return i / (len(a) + len(b) - i)


def _cont(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _pair_features(nc1, ns1, a1, nu1, z1, s1_, l1, nc2, ns2, a2, nu2, z2, s2_, l2):
    t1, t2 = set(nc1.split()), set(nc2.split())
    at1, at2 = set(a1.split()), set(a2.split())
    n1, n2 = set(nu1.split()), set(nu2.split())
    al1 = {t for t in at1 if not t.isdigit()}
    al2 = {t for t in at2 if not t.isdigit()}

    return [
        fuzz.ratio(nc1, nc2),
        fuzz.token_set_ratio(nc1, nc2),
        fuzz.token_sort_ratio(nc1, nc2),
        fuzz.partial_ratio(ns1, ns2) if ns1 and ns2 else 0.0,
        JaroWinkler.similarity(nc1, nc2),
        Levenshtein.normalized_similarity(ns1, ns2),
        _jac(t1, t2),
        _cont(t1, t2),
        float(nc1 == nc2),
        float(nc1.split()[:1] == nc2.split()[:1]) if nc1 and nc2 else 0.0,
        float(ns1 in ns2 or ns2 in ns1) if ns1 and ns2 else 0.0,
        len(ns2) / max(len(ns1), 1),
        float(l1 == l2) if l1 and l2 else 0.0,
        _jac(at1, at2),
        _cont(at1, at2),
        _jac(al1, al2),
        _cont(al1, al2),
        _jac(n1, n2),
        float(bool(n1 and n2) and nu1.split()[0] == nu2.split()[0]),
        float(bool(n1 & n2)),
        float(bool(z1) and z1 == z2),
        float(bool(z1 and z2) and z1 != z2),
        float(bool(s1_ and s2_) and s1_ == s2_),
        float(bool(s1_ and s2_) and s1_ != s2_),
        fuzz.token_set_ratio(a1, a2) if a1 and a2 else 0.0,
        fuzz.token_sort_ratio(a1, a2) if a1 and a2 else 0.0,
        float(not a2),
        float(bool(n1) and not n2)
    ]


def compute_fuzzy_array(s1, pool, cand, chunk_size=50000):
    """Computes rapidfuzz features for candidate pairs in vectorized chunks."""
    n = len(cand)
    out = np.empty((n, len(FUZZY)), dtype=np.float32)

    s1_vals = {c: s1[c].values for c in COLS}
    pool_vals = {c: pool[c].values for c in COLS}

    i1_arr = cand.i1.values
    ip_arr = cand.ip.values

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sub_i1 = i1_arr[start:end]
        sub_ip = ip_arr[start:end]

        chunk_rows = []
        for r in range(len(sub_i1)):
            i, j = sub_i1[r], sub_ip[r]
            row = _pair_features(
                s1_vals["nm_core"][i], s1_vals["nm_nospace"][i], s1_vals["ad_toks"][i],
                s1_vals["ad_nums"][i], s1_vals["ad_zip"][i], s1_vals["ad_state"][i], s1_vals["nm_legal"][i],
                pool_vals["nm_core"][j], pool_vals["nm_nospace"][j], pool_vals["ad_toks"][j],
                pool_vals["ad_nums"][j], pool_vals["ad_zip"][j], pool_vals["ad_state"][j], pool_vals["nm_legal"][j]
            )
            chunk_rows.append(row)
        out[start:end] = chunk_rows

    return out


CONTEXT = [
    "src", "c_nonlatin", "c_dom", "nm_len1", "nm_len2",
    "addr", "nchar", "nword", "hash_key", "blk_score", "blk_rank",
    "comb", "comb_rank_s1", "comb_gap_top", "comb_rank_ip", "comb_gap_ip",
    "n_strong_ip", "pool_name_freq", "pool_addr_freq"
]

ALL_FEATURES = FUZZY + CONTEXT


def build_competition_context(F, cand, pool):
    """Calculates global competition and frequency context across candidate pairs."""
    fi = {c: k for k, c in enumerate(FUZZY)}
    comb = (
        0.5 * (F[:, fi["nm_tset"]] / 100.0) +
        0.5 * (F[:, fi["ad_tset"]] / 100.0) +
        0.2 * F[:, fi["ad_numany"]]
    ).astype(np.float32)

    df_comb = pd.DataFrame({
        "i1": cand.i1.values,
        "ip": cand.ip.values,
        "comb": comb
    })

    # Competition within S1's candidates
    g_s1 = df_comb.groupby("i1").comb
    comb_rank_s1 = g_s1.rank(ascending=False, method="min").values.astype(np.float32)
    comb_gap_top = (g_s1.transform("max").values - comb).astype(np.float32)

    # Competition across S1s for the same pool record (S2/S3)
    g_ip = df_comb.groupby("ip").comb
    comb_rank_ip = g_ip.rank(ascending=False, method="min").values.astype(np.float32)
    comb_gap_ip = (g_ip.transform("max").values - comb).astype(np.float32)
    n_strong_ip = g_ip.transform(lambda s: (s >= 0.85).sum()).values.astype(np.float32)

    # Frequency of names/addresses in the pool
    nm_freq = np.log1p(pool.nm_core.map(pool.nm_core.value_counts()).fillna(1).values.astype(np.float32))
    ad_freq = np.log1p(pool.ad_toks.map(pool.ad_toks.value_counts()).fillna(1).values.astype(np.float32))

    return {
        "comb": comb,
        "comb_rank_s1": comb_rank_s1,
        "comb_gap_top": comb_gap_top,
        "comb_rank_ip": comb_rank_ip,
        "comb_gap_ip": comb_gap_ip,
        "n_strong_ip": n_strong_ip,
        "pool_name_freq": nm_freq[cand.ip.values],
        "pool_addr_freq": ad_freq[cand.ip.values]
    }


def assemble_feature_matrix(F, cand, s1, pool, ctx, lo=None, hi=None):
    """Combines fuzzy features and context into a complete DataFrame for modeling."""
    if lo is None:
        lo, hi = 0, len(cand)

    X = pd.DataFrame(F[lo:hi], columns=FUZZY)
    sub_c = cand.iloc[lo:hi]
    ip = sub_c.ip.values
    i1 = sub_c.i1.values

    X["src"] = pool.src.values[ip].astype(np.float32)
    X["c_nonlatin"] = pool.nm_nonlatin.values[ip].astype(np.float32)
    X["c_dom"] = pool.nm_dom.values[ip].astype(np.float32)
    X["nm_len1"] = s1.nm_nospace.str.len().values[i1].astype(np.float32)
    X["nm_len2"] = pool.nm_nospace.str.len().values[ip].astype(np.float32)

    for k in ("addr", "nchar", "nword", "hash_key", "blk_score", "blk_rank"):
        X[k] = sub_c[k].values.astype(np.float32)

    for k in ("comb", "comb_rank_s1", "comb_gap_top", "comb_rank_ip", "comb_gap_ip", "n_strong_ip", "pool_name_freq", "pool_addr_freq"):
        X[k] = ctx[k][lo:hi]

    return X[ALL_FEATURES]
