"""Candidate generation (per country): sparse TF-IDF cosine top-K on address tokens, name char n-grams and name words.
Common tokens are pruned with an absolute-scale document-frequency cap so retrieval cost stays bounded on 5M-row pools.
Chunks of S1 rows are processed in parallel with fork (matrices are shared copy-on-write)."""
import numpy as np, pandas as pd, multiprocessing as mp
from sklearn.feature_extraction.text import TfidfVectorizer

CHANNELS = [  # name, column, vectorizer kwargs, max_df = clip(frac*pool, floor, cap) documents
    ("addr", "ad_toks", dict(analyzer="word", token_pattern=r"\S+", sublinear_tf=True), 0.02, 1500, 20000),
    ("nchar", "nm_nospace", dict(analyzer="char", ngram_range=(3, 4), sublinear_tf=True), 0.01, 1200, 15000),
    ("nword", "nm_core", dict(analyzer="word", token_pattern=r"\S+", sublinear_tf=True), 0.01, 800, 10000),
]
_CTX = {}

def _work(args):
    ci, s, e = args
    Q, PT, k = _CTX["Q"][ci][s:e], _CTX["PT"][ci], _CTX["k"]
    M = (Q @ PT).tocsr()
    rows, cols, vals = [], [], []
    for i in range(M.shape[0]):
        a, b = M.indptr[i], M.indptr[i + 1]
        if a == b: continue
        d, c = M.data[a:b], M.indices[a:b]
        if len(d) > k:
            sel = np.argpartition(-d, k)[:k]; d, c = d[sel], c[sel]
        rows.append(np.full(len(d), s + i, dtype=np.int32)); cols.append(c.astype(np.int32)); vals.append(d)
    if not rows: return ci, np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.float32)
    return ci, np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)

def block_country(g1, gp, k=15, max_cands=40, procs=8, chunk=500, verbose=True):
    """g1, gp: normalized frames with a 0..n-1 RangeIndex. Returns candidate DataFrame (i1, ip, addr, nchar, nword, blk_score, blk_rank)."""
    Qs, PTs = [], []
    for name, col, kw, frac, floor, cap in CHANNELS:
        v = TfidfVectorizer(dtype=np.float32, max_df=min(cap, max(floor, int(frac * len(gp)))), **kw)
        P = v.fit_transform(gp[col]); Q = v.transform(g1[col])
        Qs.append(Q.tocsr()); PTs.append(P.T.tocsr())
        if verbose: print(f"  {name}: vocab {len(v.vocabulary_)}, pool nnz {P.nnz}", flush=True)
    _CTX.update(Q=Qs, PT=PTs, k=k)
    tasks = [(ci, s, min(s + chunk, g1.shape[0])) for ci in range(len(CHANNELS)) for s in range(0, g1.shape[0], chunk)]
    with mp.get_context("fork").Pool(procs) as pool:
        res = list(pool.imap_unordered(_work, tasks, chunksize=1))
    _CTX.clear()
    npool = np.int64(len(gp))
    keys = np.concatenate([r[1].astype(np.int64) * npool + r[2] for r in res])
    uk, inv = np.unique(keys, return_inverse=True)
    sims = np.zeros((len(uk), len(CHANNELS)), np.float32)
    off = 0
    for ci, r, c, v in res:
        sims[inv[off:off + len(r)], ci] = v; off += len(r)
    out = pd.DataFrame({"i1": (uk // npool).astype(np.int32), "ip": (uk % npool).astype(np.int32)})
    for ci, ch in enumerate(CHANNELS): out[ch[0]] = sims[:, ci]
    out["blk_score"] = out.addr + 0.7 * out.nchar + 0.5 * out.nword
    out = out.sort_values(["i1", "blk_score"], ascending=[True, False], kind="stable")
    out["blk_rank"] = out.groupby("i1").cumcount().astype(np.int16)
    return out[out.blk_rank < max_cands].reset_index(drop=True)
