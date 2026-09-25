import pandas as pd, numpy as np, json, re, sys
D="/Users/kabirmathur/Documents/Amazon_hack/student_resource/dataset/"
rng=np.random.default_rng(42)
def rd(p): return pd.read_csv(D+p,sep="\t",dtype=str,keep_default_na=False,quoting=3)
files={}
for sp in ("train","test"):
    for s in (1,2,3): files[(sp,s)]=rd(f"{sp}/{sp}_source{s}.tsv"); print(sp,s,len(files[(sp,s)]),flush=True)
gt=rd("train/train_ground_truth.tsv")
# validation = 10% holdout of train S1 entities
s1=files[("train",1)]
val_ids=set(rng.choice(s1.entity_id.values,size=len(s1)//10,replace=False))
gt["is_val"]=gt.source1_entity_id.isin(val_ids)
gt["ids"]=gt.matched_entity_ids.map(lambda x:[i for i in x.split(",") if i])
gt["n2"]=gt.ids.map(lambda l:sum(i.startswith("S2") for i in l)); gt["n3"]=gt.ids.map(lambda l:sum(i.startswith("S3") for i in l))
gt["n"]=gt.n2+gt.n3
gt=gt.merge(s1[["entity_id","country"]],left_on="source1_entity_id",right_on="entity_id",how="left")
def cnt(x): return {str(k):int(v) for k,v in x.items()}
def hist(x,bins):
    h,_=np.histogram(x,bins=bins); return h.tolist()
LEG=["inc","llc","ltd","limited","corp","corporation","pvt","private","llp","co","company","sarl","sas","gmbh","& co","sa"]
def stats(df):
    o={"rows":len(df),"countries":cnt(df.country.value_counts())}
    n=df.business_name; a=df.business_address
    o["missing_name"]=float((n.str.strip()=="").mean()); o["missing_addr"]=float((a.str.strip()=="").mean())
    o["dup_name_pct"]=float(n.duplicated(keep=False).mean())
    o["name_len_hist"]=hist(n.str.len().clip(0,80),np.arange(0,85,5))
    o["addr_len_hist"]=hist(a.str.len().clip(0,120),np.arange(0,130,10))
    o["non_ascii_name"]=float(n.str.contains(r"[^\x00-\x7F]",regex=True).mean())
    o["devanagari_name"]=float(n.str.contains(r"[ऀ-ॿ]",regex=True).mean())
    o["accent_name"]=float(n.str.contains(r"[À-ÿ]",regex=True).mean())
    o["upper_addr"]=float((a.str.upper()==a)[a!=""].mean())
    o["all_upper_name"]=float((n.str.upper()==n)[n!=""].mean())
    o["url_name"]=float(n.str.contains(r"\.(?:com|in|net|org|fr)\b",case=False,regex=True).mean())
    o["junk_prefix"]=float(n.str.match(r"^[^\wऀ-ॿ]").mean())
    o["pin_addr"]=float(a.str.contains(r"\b\d{6}\b",regex=True).mean())
    o["zip_addr"]=float(a.str.contains(r"\b\d{5}\b",regex=True).mean())
    o["near_landmark"]=float(a.str.contains(r"\bnear\b|\bopp\b|\bbehind\b",case=False,regex=True).mean())
    o["avg_name_tokens"]=float(n.str.split().str.len().mean())
    o["by_country"]={c:{"rows":int(len(g)),"missing_addr":float((g.business_address=="").mean()),
        "avg_name_len":float(g.business_name.str.len().mean()),"avg_addr_len":float(g.business_address.str.len().mean())}
        for c,g in df.groupby("country")}
    low=n.str.lower().str.replace(r"[^\w\s&]"," ",regex=True)
    toks=low.str.split().explode()
    o["legal"]={k:int((toks==k).sum()) for k in LEG}
    o["top_tokens"]=cnt(toks[~toks.isin(["",None])].value_counts().head(20))
    # state/last-part of address
    last=a.str.split(",").str[-1].str.strip()
    o["top_regions"]=cnt(last[last!=""].value_counts().head(12))
    return o
out={"stats":{},"samples":{}}
for (sp,s),df in files.items():
    out["stats"][f"{sp}_s{s}"]=stats(df)
    out["samples"][f"{sp}_s{s}"]=df.sample(6,random_state=1).to_dict("records")
    print("stats",sp,s,flush=True)
# val S1 stats
vs1=s1[s1.entity_id.isin(val_ids)]; out["stats"]["val_s1"]=stats(vs1)
out["stats"]["train_s1_noval"]=None
# ground truth
def gstats(g):
    return {"n_s1":len(g),"singleton_pct":float((g.n==0).mean()),"avg_matches":float(g.n.mean()),
      "avg_s2":float(g.n2.mean()),"avg_s3":float(g.n3.mean()),
      "match_hist":cnt(g.n.clip(0,10).value_counts().sort_index()),
      "has_s2":float((g.n2>0).mean()),"has_s3":float((g.n3>0).mean()),"both":float(((g.n2>0)&(g.n3>0)).mean()),
      "total_matched":int(g.n.sum()),
      "by_country":{c:{"n":int(len(x)),"singleton":float((x.n==0).mean()),"avg":float(x.n.mean())} for c,x in g.groupby("country")}}
out["gt"]={"train":gstats(gt[~gt.is_val]),"val":gstats(gt[gt.is_val]),"all":gstats(gt)}
# orphan coverage: fraction of S2/S3 train records matched to some S1
allm=set(i for l in gt.ids for i in l)
for s in (2,3):
    e=files[("train",s)].entity_id; out["gt"][f"s{s}_matched_frac"]=float(e.isin(allm).mean())
# pair similarity on sample of matched pairs
lk=files[("train",2)].set_index("entity_id"); lk3=files[("train",3)].set_index("entity_id")
s1i=s1.set_index("entity_id")
smp=gt[gt.n>0].sample(60000,random_state=3)
def norm(x): return re.sub(r"[^\w\s]"," ",x.lower()).split()
def jac(a,b):
    a,b=set(a),set(b); return len(a&b)/len(a|b) if a|b else 0
rows=[]
for sid,ids in zip(smp.source1_entity_id,smp.ids):
    r=s1i.loc[sid]
    for i in ids[:2]:
        t=(lk if i.startswith("S2") else lk3)
        if i not in t.index: continue
        m=t.loc[i]
        if isinstance(m,pd.DataFrame): m=m.iloc[0]
        rows.append((i[:2],r.country,jac(norm(r.business_name),norm(m.business_name)),jac(norm(r.business_address),norm(m.business_address)),
                     r.business_name.lower()==m.business_name.lower(), m.business_address==""))
pr=pd.DataFrame(rows,columns=["src","country","nj","aj","exact","noaddr"])
out["pairs"]={"n":len(pr),
 "name_jac_hist":{s:hist(g.nj,np.linspace(0,1.0001,11)) for s,g in pr.groupby("src")},
 "addr_jac_hist":{s:hist(g[~g.noaddr].aj,np.linspace(0,1.0001,11)) for s,g in pr.groupby("src")},
 "exact_name":{s:float(g.exact.mean()) for s,g in pr.groupby("src")},
 "mean_name_jac":{f"{s}|{c}":float(g.nj.mean()) for (s,c),g in pr.groupby(["src","country"])},
 "mean_addr_jac":{f"{s}|{c}":float(g[~g.noaddr].aj.mean()) for (s,c),g in pr.groupby(["src","country"])},
 "noaddr":{s:float(g.noaddr.mean()) for s,g in pr.groupby("src")}}
# examples of matched groups
ex=[]
for sid,ids in gt[(gt.n>=3)].sample(6,random_state=5)[["source1_entity_id","ids"]].values:
    rec=[{"id":sid,**{k:s1i.loc[sid][k] for k in("business_name","business_address","country")}}]
    for i in ids[:4]:
        t=lk if i.startswith("S2") else lk3
        if i in t.index:
            m=t.loc[i]; m=m.iloc[0] if isinstance(m,pd.DataFrame) else m
            rec.append({"id":i,"business_name":m.business_name,"business_address":m.business_address,"country":m.country})
    ex.append(rec)
out["examples"]=ex
json.dump(out,open("/Users/kabirmathur/Documents/Amazon_hack/eda/stats.json","w"),ensure_ascii=False)
print("done")
