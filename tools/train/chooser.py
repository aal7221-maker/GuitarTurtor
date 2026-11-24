"""Learned string chooser: per-candidate GBM, beam decode. Train data = (events with true positions)."""
import numpy as np, json, zipfile, pipe
from sklearn.ensemble import HistGradientBoostingClassifier
OPEN=pipe.OPEN
def cands(p): return [(s,p-OPEN[s]) for s in range(6) if 0<=p-OPEN[s]<=22]
def feats(p, c, hist, dt):
    s,f=c; h=hist+[(0,0,60,0.0)]*(3-len(hist)) if len(hist)<3 else hist
    (s1,f1,p1,_),(s2,f2,p2,_),(s3,f3,p3,_)=h[-1],h[-2],h[-3]
    anc=next(((hs,hf) for hs,hf,_,_ in reversed(hist) if hf>0),(s,f))
    nc=len(cands(p)); rank=(sorted(x[0] for x in cands(p)).index(s) if (s,f) in cands(p) else 0)
    return [p,s,f,f==0,s1,f1,p1,s-s1,f-f1,p-p1,min(dt,2.0),s2,f2,p-p2,s3,f3,s==s1,s==s2,
            f-anc[1],s-anc[0],nc,rank,f-min(x[1] for x in cands(p)) if nc else 0,
            sum(1 for hs,hf,_,_ in hist[-4:] if hf==0)]
def sequences_from_events(evs):  # evs: list of (t,[(pitch,s,f)...])
    return evs
def build(seqs, weight):
    X=[];y=[];w=[]
    for seq in seqs:
        hist=[]; lastt=-9
        for t,notes in seq:
            for p,s,f in sorted(notes):
                for c in cands(p):
                    X.append(feats(p,c,hist,t-lastt)); y.append(int(c==(s,f))); w.append(weight)
                hist=(hist+[(s,f,p,t)])[-6:]
            lastt=t
    return np.array(X,float),np.array(y),np.array(w)
def fit(parts, seed=0):
    Xs,ys,ws=zip(*[build(sq,wt) for sq,wt in parts if sq]); X=np.vstack(Xs);y=np.concatenate(ys);w=np.concatenate(ws)
    m=HistGradientBoostingClassifier(max_iter=150,learning_rate=0.08,max_leaf_nodes=15,l2_regularization=1.0,random_state=seed)
    m.fit(X,y,sample_weight=w); return m
def decode(model, events, beam=3):
    """events: list of {'t','p':[pitches]} -> list of tuple positions per event."""
    beams=[(0.0,[],[])]  # logp, hist, out
    lastt=-9
    for e in events:
        for _ in [0]:
            nb=[]
            for lp,hist,out in beams:
                parts=[(lp,hist,[])]
                for p in sorted(e["p"]):
                    new=[]
                    for lp2,h2,o2 in parts:
                        cs=[c for c in cands(p) if all(c[0]!=x[0] for x in o2)] or [(0,0)]
                        X=np.array([feats(p,c,h2,e["t"]-lastt) for c in cs],float)
                        pr=model.predict_proba(X)[:,1] if len(cs)>1 or cs[0]!=(0,0) else np.ones(1)
                        pr=pr/ max(pr.sum(),1e-9)
                        for c,q in zip(cs,pr): new.append((lp2+np.log(q+1e-6),(h2+[(c[0],c[1],p,e["t"])])[-6:],o2+[c]))
                    parts=sorted(new,key=lambda x:-x[0])[:beam]
                for lp3,h3,o3 in parts: nb.append((lp3,h3,out+[tuple(sorted(o3))]))
            beams=sorted(nb,key=lambda x:-x[0])[:beam]
        lastt=e["t"]
    return beams[0][2] if events else []
def events_from_ref(name, lo=0, hi=1):
    R=[n for n in pipe.ref(name)]
    ts=[n["t"] for n in R]; a0,a1=min(ts),max(ts); T0=a0+lo*(a1-a0); T1=a0+hi*(a1-a0)+(1e-6 if hi<1 else 1)
    out=[]
    for n in R:
        if T0<=n["t"]<T1 and not n["_bad"]:
            out.append((n["t"],[(OPEN[s]+(n.get("hb") or f),s,f) for s,f in n["n"]]))
    return out
def ours_seq(name,lo,hi): return events_from_ref(name,lo,hi)
def load_guitarset(players):
    z=zipfile.ZipFile(__import__('os').path.join(pipe.HERE,'data','guitarset','annotation.zip')); res={}
    for nm in z.namelist():
        if not nm.endswith('.jams') or nm[:2] not in players: continue
        d=json.load(z.open(nm)); notes=[]
        for a in d['annotations']:
            if a['namespace']!='note_midi': continue
            s=5-int(a['annotation_metadata']['data_source'])   # GuitarSet 0=low E -> ours 5
            for x in a['data']:
                p=int(round(x['value'])); f=p-OPEN[s]
                if 0<=f<=22: notes.append((x['time'],p,s,f))
        notes.sort(); ev=[]
        for t,p,s,f in notes:
            if ev and t-ev[-1][0]<0.05: ev[-1][1].append((p,s,f))
            else: ev.append((t,[(p,s,f)]))
        res[nm]=ev
    return res
def exact_acc(model_or_none, seq, params=None):
    """string-accuracy on oracle pitches over a seq of (t,[(p,s,f)])."""
    evs=[{"t":t,"p":[p for p,_,_ in ns]} for t,ns in seq]
    if model_or_none is None:
        p=dict(pipe.DEFAULTS); p.update(params or {}); pos=pipe.assign(evs,p)
    else: pos=decode(model_or_none,evs)
    ok=tot=0
    for (t,ns),ps in zip(seq,pos):
        ok+=sorted((s,f) for _,s,f in ns)==sorted(ps) ; tot+=1
    return ok/max(tot,1)
