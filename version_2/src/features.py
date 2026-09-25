"""Pair features. Stage 1 (fuzzy string features) is computed in parallel per S1-block and stored in a memmap;
stage 2 (candidate-side + competition context) is assembled vectorised, chunk by chunk, so 60M-pair jobs fit in 16 GB."""
import numpy as np, pandas as pd
from multiprocessing import Pool
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

def _jac(a, b):
    if not a or not b: return 0.0
    i = len(a & b); return i / (len(a) + len(b) - i)
def _cont(a, b):
    if not a or not b: return 0.0
    return len(a & b) / min(len(a), len(b))

FUZZY = ["nm_ratio", "nm_tset", "nm_tsort", "nm_partial", "nm_jw", "nm_lev", "nm_jac", "nm_cont", "nm_exact",
         "nm_first", "nm_substr", "nm_lenratio", "nm_legal_eq",
         "ad_jac", "ad_cont", "ad_aljac", "ad_alcont", "ad_numjac", "ad_num0eq", "ad_numany", "ad_zipeq", "ad_zipne",
         "ad_steq", "ad_stne", "ad_tset", "ad_tsort", "ad_missing_c", "ad_num_missing_c"]
COLS = ["nm_core", "nm_nospace", "ad_toks", "ad_nums", "ad_zip", "ad_state", "nm_legal"]

def _pair(nc1, ns1, a1, nu1, z1, s1_, l1, nc2, ns2, a2, nu2, z2, s2_, l2):
    t1, t2 = set(nc1.split()), set(nc2.split())
    at1, at2 = set(a1.split()), set(a2.split())
    n1, n2 = set(nu1.split()), set(nu2.split())
    al1, al2 = {t for t in at1 if not t.isdigit()}, {t for t in at2 if not t.isdigit()}
    return [
        fuzz.ratio(nc1, nc2), fuzz.token_set_ratio(nc1, nc2), fuzz.token_sort_ratio(nc1, nc2),
        fuzz.partial_ratio(ns1, ns2) if ns1 and ns2 else 0.0,
        JaroWinkler.similarity(nc1, nc2), Levenshtein.normalized_similarity(ns1, ns2),
        _jac(t1, t2), _cont(t1, t2), float(nc1 == nc2),
        float(nc1.split()[:1] == nc2.split()[:1]) if nc1 and nc2 else 0.0,
        float(ns1 in ns2 or ns2 in ns1) if ns1 and ns2 else 0.0,
        len(ns2) / max(len(ns1), 1),
        float(l1 == l2) if l1 and l2 else 0.0,
        _jac(at1, at2), _cont(at1, at2), _jac(al1, al2), _cont(al1, al2), _jac(n1, n2),
        float(bool(n1 and n2) and nu1.split()[0] == nu2.split()[0]),
        float(bool(n1 & n2)), float(bool(z1) and z1 == z2), float(bool(z1 and z2) and z1 != z2),
        float(bool(s1_ and s2_) and s1_ == s2_), float(bool(s1_ and s2_) and s1_ != s2_),
        fuzz.token_set_ratio(a1, a2) if a1 and a2 else 0.0, fuzz.token_sort_ratio(a1, a2) if a1 and a2 else 0.0,
        float(not a2), float(bool(n1) and not n2)]

def _task(args):
    s1c, pc, li, lj = args      # per-block string columns (lists) and local pair indices
    out = np.empty((len(li), len(FUZZY)), np.float32)
    for r in range(len(li)):
        i, j = li[r], lj[r]
        out[r] = _pair(*[c[i] for c in s1c], *[c[j] for c in pc])
    return out

def fuzzy_memmap(s1, pool, cand, path, procs=8, block=1500):
    """Fuzzy features for all candidate pairs (cand sorted by i1) -> float32 memmap [n_pairs, len(FUZZY)]."""
    n = len(cand)
    mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(n, len(FUZZY)))
    s1a = [s1[c].values for c in COLS]; pa = [pool[c].values for c in COLS]
    i1, ip = cand.i1.values, cand.ip.values
    starts = np.searchsorted(i1, np.arange(0, i1.max() + 1, block))        # pair ranges per S1 block
    bounds = list(starts) + [n]
    def gen():
        for b in range(len(starts)):
            lo, hi = bounds[b], bounds[b + 1]
            if hi <= lo: continue
            u1, li = np.unique(i1[lo:hi], return_inverse=True)
            up, lj = np.unique(ip[lo:hi], return_inverse=True)
            yield ([a[u1].tolist() for a in s1a], [a[up].tolist() for a in pa], li.tolist(), lj.tolist())
    pos = 0
    with Pool(procs) as p:
        for res in p.imap(_task, gen(), chunksize=1):
            mm[pos:pos + len(res)] = res; pos += len(res)
    assert pos == n, (pos, n)
    mm.flush()
    return mm

CONTEXT = ["src", "c_nonlatin", "c_dom", "nm_len1", "nm_len2", "addr", "nchar", "nword", "blk_score", "blk_rank",
           "comb", "comb_rank_s1", "comb_gap_top", "comb_rank_ip", "comb_gap_ip"]
FEATS = FUZZY + CONTEXT

def context_arrays(F, cand, s1, pool):
    """Global (whole-country) competition context; needs only small per-pair arrays."""
    fi = {c: k for k, c in enumerate(FUZZY)}
    comb = (0.5 * F[:, fi["nm_tset"]] / 100 + 0.5 * F[:, fi["ad_tset"]] / 100 + 0.2 * F[:, fi["ad_numany"]]).astype(np.float32)
    g = pd.DataFrame({"i1": cand.i1.values, "ip": cand.ip.values, "comb": comb, "src": pool.src.values[cand.ip.values]})
    out = {"comb": comb}
    out["comb_rank_s1"] = g.groupby(["i1", "src"]).comb.rank(ascending=False, method="min").values.astype(np.float32)
    out["comb_gap_top"] = (g.groupby(["i1", "src"]).comb.transform("max").values - comb).astype(np.float32)
    gi = g.groupby("ip").comb
    out["comb_rank_ip"] = gi.rank(ascending=False, method="min").values.astype(np.float32)
    out["comb_gap_ip"] = (gi.transform("max").values - comb).astype(np.float32)
    return out

def assemble(F, cand, s1, pool, ctx, lo, hi):
    X = pd.DataFrame(np.asarray(F[lo:hi]), columns=FUZZY)
    c = cand.iloc[lo:hi]; ip, i1 = c.ip.values, c.i1.values
    X["src"] = pool.src.values[ip].astype(np.float32)
    X["c_nonlatin"] = pool.nm_nonlatin.values[ip]; X["c_dom"] = pool.nm_dom.values[ip]
    X["nm_len1"] = s1.nm_nospace.str.len().values[i1]; X["nm_len2"] = pool.nm_nospace.str.len().values[ip]
    for k in ("addr", "nchar", "nword", "blk_score", "blk_rank"): X[k] = c[k].values
    for k, v in ctx.items(): X[k] = v[lo:hi]
    return X[FEATS]
