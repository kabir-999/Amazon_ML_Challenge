"""v2 features: v1 fuzzy features + richer competition/ambiguity context + stage-2 (probability-context) features."""
import numpy as np, pandas as pd
from features import FUZZY, COLS, fuzzy_memmap   # noqa: F401  (stage-1 fuzzy features unchanged)

FI = {c: k for k, c in enumerate(FUZZY)}

def group_top2(keys, vals):
    """per row: (max of group, second max of group (0 if singleton group)) for group `keys` (int array)."""
    order = np.lexsort((-vals, keys)); ks, vs = keys[order], vals[order]
    start = np.r_[True, ks[1:] != ks[:-1]]
    gid = np.cumsum(start) - 1; first = np.flatnonzero(start)
    t1 = vs[first]; sec_idx = np.minimum(first + 1, len(vs) - 1)
    t2 = np.where((first + 1 < len(vs)) & (ks[sec_idx] == ks[first]), vs[sec_idx], 0.0)
    out1, out2 = np.empty(len(vals), np.float32), np.empty(len(vals), np.float32)
    out1[order], out2[order] = t1[gid], t2[gid]
    return out1, out2

def grp_rank(keys, vals):
    """rank (1 = highest, method='min') of vals inside each key group."""
    return pd.Series(vals).groupby(keys).rank(ascending=False, method="min").values.astype(np.float32)

CONTEXT2 = ["src", "c_nonlatin", "c_dom", "nm_len1", "nm_len2", "addr", "nchar", "nword", "blk_score", "blk_rank",
            "comb", "comb_rank_s1", "comb_gap_top", "comb_rank_ip", "comb_gap_ip",
            "comb_top2_ip", "n_strong_ip", "n_nm90_s1", "n_ad90_s1", "n_strong_s1", "pool_name_freq", "pool_addr_freq"]
FEATS2 = FUZZY + CONTEXT2

def context_v2(F, cand, s1, pool):
    """Whole-country context arrays (needs all candidates of the country)."""
    F = np.asarray(F)
    i1, ip = cand.i1.values.astype(np.int64), cand.ip.values.astype(np.int64)
    src = pool.src.values[ip]
    comb = (0.5 * F[:, FI["nm_tset"]] / 100 + 0.5 * F[:, FI["ad_tset"]] / 100 + 0.2 * F[:, FI["ad_numany"]]).astype(np.float32)
    k_s = i1 * 2 + (src - 2)                         # (S1, source) group
    top1_s, _ = group_top2(k_s, comb)
    out = {"comb": comb, "comb_rank_s1": grp_rank(k_s, comb), "comb_gap_top": top1_s - comb}
    top1_p, top2_p = group_top2(ip, comb)
    out["comb_rank_ip"] = grp_rank(ip, comb); out["comb_gap_ip"] = top1_p - comb
    out["comb_top2_ip"] = np.where(comb >= top1_p, top2_p, top1_p).astype(np.float32)   # best competing S1 for this record
    strong = (comb >= 0.85).astype(np.float32)
    out["n_strong_ip"] = pd.Series(strong).groupby(ip).transform("sum").values.astype(np.float32)
    out["n_strong_s1"] = pd.Series(strong).groupby(k_s).transform("sum").values.astype(np.float32)
    out["n_nm90_s1"] = pd.Series((F[:, FI["nm_tset"]] >= 90).astype(np.float32)).groupby(k_s).transform("sum").values.astype(np.float32)
    out["n_ad90_s1"] = pd.Series((F[:, FI["ad_tset"]] >= 90).astype(np.float32)).groupby(k_s).transform("sum").values.astype(np.float32)
    nf = np.log1p(pool.nm_core.map(pool.nm_core.value_counts()).values.astype(np.float32))
    af = np.log1p(pool.ad_toks.map(pool.ad_toks.value_counts()).values.astype(np.float32))
    out["pool_name_freq"], out["pool_addr_freq"] = nf[ip], af[ip]
    return out

def assemble2(F, cand, s1, pool, ctx, rows):
    """feature DataFrame for candidate rows `rows` (index array or slice)."""
    X = pd.DataFrame(np.asarray(F[rows]), columns=FUZZY)
    c = cand.iloc[rows] if not isinstance(rows, slice) else cand.iloc[rows]
    ip, i1 = c.ip.values, c.i1.values
    X["src"] = pool.src.values[ip].astype(np.float32)
    X["c_nonlatin"] = pool.nm_nonlatin.values[ip]; X["c_dom"] = pool.nm_dom.values[ip]
    X["nm_len1"] = s1.nm_nospace.str.len().values[i1]; X["nm_len2"] = pool.nm_nospace.str.len().values[ip]
    for k in ("addr", "nchar", "nword", "blk_score", "blk_rank"): X[k] = c[k].values
    for k, v in ctx.items(): X[k] = v[rows]
    return X[FEATS2]

S2CTX = ["p1", "p1_rank_s1", "p1_gap_top_s1", "p1_second_s1", "n_p50_s1", "n_p50_src", "p1_other_src_max",
         "p1_rank_ip", "p1_top_other_ip", "n_ip_p30", "p1_sum_s1"]

def stage2_context(p1, cand, pool):
    """Probability-context features from stage-1 probabilities over the whole country."""
    i1, ip = cand.i1.values.astype(np.int64), cand.ip.values.astype(np.int64)
    src = pool.src.values[ip]; k_s = i1 * 2 + (src - 2)
    top1_s, top2_s = group_top2(k_s, p1)
    out = {"p1": p1, "p1_rank_s1": grp_rank(k_s, p1), "p1_gap_top_s1": top1_s - p1,
           "p1_second_s1": np.where(p1 >= top1_s, top2_s, top1_s).astype(np.float32)}
    hi = (p1 > 0.5).astype(np.float32)
    out["n_p50_s1"] = pd.Series(hi).groupby(i1).transform("sum").values.astype(np.float32)
    out["n_p50_src"] = pd.Series(hi).groupby(k_s).transform("sum").values.astype(np.float32)
    # best probability among the OTHER source's candidates of the same S1
    other = (i1 * 2 + (1 - (src - 2)))
    tab = pd.Series(top1_s, index=k_s).groupby(level=0).first()
    out["p1_other_src_max"] = tab.reindex(other).fillna(0).values.astype(np.float32)
    t1p, t2p = group_top2(ip, p1)
    out["p1_rank_ip"] = grp_rank(ip, p1); out["p1_top_other_ip"] = np.where(p1 >= t1p, t2p, t1p).astype(np.float32)
    out["n_ip_p30"] = pd.Series((p1 > 0.3).astype(np.float32)).groupby(ip).transform("sum").values.astype(np.float32)
    out["p1_sum_s1"] = pd.Series(p1).groupby(i1).transform("sum").values.astype(np.float32)
    return out
