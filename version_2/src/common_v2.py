"""Shared helpers: country data loading, F0.5 evaluation, decision rule."""
import numpy as np, pandas as pd
from config import *
from features_v2 import context_v2

def load_country(tag, c, with_labels=True):
    d = f"{WORK}/{tag}"
    s1 = pd.read_parquet(f"{d}/{c}_s1n.parquet"); pool = pd.read_parquet(f"{d}/{c}_pooln.parquet"); cand = pd.read_parquet(f"{d}/{c}_cand.parquet")
    F = np.load(f"{d}/{c}_F.npy", mmap_mode="r")
    y = np.load(f"{d}/{c}_y.npy") if with_labels else None
    return dict(c=c, s1=s1, pool=pool, cand=cand, F=F, y=y)

def truth_sets(D, tag="trainfull"):
    """global-int truth sets {s1_key: {pool_key}} using per-country offsets stored in D[c]['off1'], ['offp']."""
    gt = pd.read_parquet(f"{WORK}/{tag}/gt.parquet"); gmap = dict(zip(gt.source1_entity_id, gt.matched_entity_ids))
    T = {}
    for d in D.values():
        pid = dict(zip(d["pool"].entity_id, range(len(d["pool"]))))
        for k, e in enumerate(d["s1"].entity_id):
            T[k + d["off1"]] = {pid[b] + d["offp"] for b in gmap[e].split(",") if b}
    return T

def f05(TRUE, pred, ids):
    tot = 0.0
    for i in ids:
        t, p = TRUE[i], pred.get(i, set())
        if not t and not p: tot += 1; continue
        if not t or not p: continue
        tp = len(t & p)
        if tp: P, R = tp / len(p), tp / len(t); tot += 1.25 * P * R / (.25 * P + R)
    return tot / len(ids)

def decide(g1, gp, prob, thr, assign=True):
    d = pd.DataFrame({"g1": g1, "gp": gp, "p": prob}); d = d[d.p >= thr]
    if assign: d = d.sort_values("p", ascending=False).drop_duplicates("gp")
    pred = {}
    for a, b in zip(d.g1.values, d.gp.values): pred.setdefault(a, set()).add(b)
    return pred
