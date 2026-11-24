"""Parametrised audio-only pipeline (basic-pitch -> skyline -> merge -> Viterbi strings) + scorer."""
import sys, os, json, itertools, functools
import numpy as np
sys.path.insert(0, r"D:\sharvin1\Projects\Music\pitch-model")
import transcribe as T
HERE = os.path.dirname(os.path.abspath(__file__))
OPEN = T.OPEN
SONGS = {
 "choo-lo": dict(audio=r"D:\sharvin1\Projects\Music\guitar3d\audio\choo-lo.mp3", cache=HERE+r"\work\choo-lo.mp3.npz",
                 ref=r"D:\sharvin1\Projects\Music\guitar3d\data\score-choo-lo.json", t0=0.0, t1=38.7),
 "locked-away": dict(audio=HERE+r"\work\locked.wav", cache=HERE+r"\work\locked.wav.npz",
                 ref=r"D:\sharvin1\Projects\Music\guitar3d\data\score-locked-away.json", t0=0.0, t1=1e9),
}
DEFAULTS = dict(onset=0.5, frame=0.4, min_ms=60.0, high=0.75, group_ms=50.0, merge_ms=70.0, max_poly=1,
                open_cost=0.0, str_change=1.0, str_jump=0.25, shift=0.08, big_shift=0.3, span=4, beam=64,
                drone=0.6, fret_pen=0.01, same_str_bonus=0.0)
SPACE = dict(onset=(0.2,0.8), frame=(0.15,0.6), min_ms=(30,150), high=(0.4,1.0), group_ms=(20,90), merge_ms=(30,150),
             max_poly=[1,2], open_cost=(0,1.0), str_change=(0.2,2.5), str_jump=(0,0.8), shift=(0,0.3), big_shift=(0,1.0),
             span=[3,4,5,6], beam=[16,64], drone=(0,1.5), fret_pen=(0,0.05), same_str_bonus=(0,1.0))

@functools.lru_cache(None)
def _out(name):
    z = np.load(SONGS[name]["cache"]); return {k: z[k] for k in ("note","onset","contour")}
@functools.lru_cache(maxsize=64)
def _ev(name, onset, frame, min_ms):
    return T.note_events(_out(name), onset, frame, min_ms)

def group(ev, p):
    groups, cur = [], []
    for n in ev:
        if cur and n[0]-cur[0][0] > p["group_ms"]/1000: groups.append(cur); cur=[]
        cur.append(n)
    if cur: groups.append(cur)
    res=[]
    for g in groups:
        g = sorted(g, key=lambda n:-n[3]); top = g[0]
        top = max((n for n in g if n is g[0] or n[3] >= p["high"]*g[0][3]), key=lambda n:n[2])
        keep=[top]
        if p["max_poly"]>=2:
            for n in g:
                if n is top or len(keep)>=2: continue
                if abs(n[2]-top[2]) in (7,12) and n[3] >= 0.5*top[3]: keep.append(n)
        t=min(n[0] for n in g)
        res.append({"t":round(t,3),"p":sorted(n[2] for n in keep)})
    out=[]
    for e in res:
        if out and e["t"]-out[-1]["t"] < p["merge_ms"]/1000 and e["p"]==out[-1]["p"]: continue
        out.append(e)
    return out

def positions(pitches, span):
    opts=[[(s,q-OPEN[s]) for s in range(6) if 0<=q-OPEN[s]<=22] for q in pitches]
    res=[]
    for combo in itertools.product(*opts):
        st=[s for s,_ in combo]; fr=[f for _,f in combo if f>0]
        if len(set(st))==len(st) and (not fr or max(fr)-min(fr)<=span): res.append(tuple(sorted(combo)))
    return res

def assign(events, p):
    beams=[(0.0,[],None)]
    for e in events:
        cands = positions(e["p"], p["span"]) or [tuple((0,0) for _ in e["p"])]
        nxt={}
        for cost,path,anchor in beams:
            for pos in cands:
                fb=tuple((s,f) for s,f in pos if f>0); anc=fb or anchor
                if not fb: c=p["open_cost"]
                elif anchor is None: c=0.0
                else:
                    sa=np.mean([s for s,_ in anchor]); fa=np.mean([f for _,f in anchor])
                    sb=np.mean([s for s,_ in fb]); fm=np.mean([f for _,f in fb]); c=0.0
                    if sa!=sb: c+=p["str_change"]+p["str_jump"]*abs(sa-sb)
                    else: c-=p["same_str_bonus"]
                    c+=p["shift"]*abs(fa-fm)+(p["big_shift"] if abs(fa-fm)>4 else 0)
                c+=cost+p["fret_pen"]*sum(f for _,f in pos)
                ro={s for q in path[-2:] for s,f in q if f==0}
                c+=p["drone"]*sum(1 for s,f in fb if s in ro)
                k=(pos,anc)
                if k not in nxt or c<nxt[k][0]: nxt[k]=(c,path+[pos],anc)
        beams=sorted(nxt.values(), key=lambda x:x[0])[:int(p["beam"])]
    return beams[0][1] if events else []

def run(name, params=None, chooser=None):
    p=dict(DEFAULTS); p.update(params or {})
    ev=_ev(name, round(p["onset"],3), round(p["frame"],3), round(p["min_ms"],1))
    events=group(ev,p)
    pos = chooser(events) if chooser else assign(events,p)
    return [{"t":e["t"],"n":[list(x) for x in q]} for e,q in zip(events,pos)]

@functools.lru_cache(None)
def ref(name):
    d=json.load(open(SONGS[name]["ref"],encoding="utf-8"))["notes"]
    s=SONGS[name]; keep=[]
    for n in d:
        if not (s["t0"]<=n["t"]<=s["t1"]): continue
        n=dict(n); n["_bad"]= n.get("conf",1)<0.7 or bool(n.get("flags")); keep.append(n)
    return keep
def _pit(n): return sorted(OPEN[s]+(n.get("hb") or f) for s,f in n["n"])
def score(name, auto, lo=0.0, hi=1.0, tol=0.08):
    R=ref(name); ts=[n["t"] for n in R]; a0,a1=min(ts),max(ts); T0=a0+lo*(a1-a0); T1=a0+hi*(a1-a0)+(1e-6 if hi<1 else 1)
    Rr=[n for n in R if T0<=n["t"]<T1]
    A=[a for a in auto if T0-tol<=a["t"]<T1+tol]; used=set(); m=pp=ex=0; scored=0
    for r in Rr:
        c=[(abs(a["t"]-r["t"]),i) for i,a in enumerate(A) if i not in used and abs(a["t"]-r["t"])<=tol]
        if r["_bad"]:
            if c: used.add(min(c)[1])   # consume but don't score
            continue
        scored+=1
        if not c: continue
        used.add(min(c)[1]); a=A[min(c)[1]]; m+=1
        if _pit(r)==_pit(a): pp+=1
        if sorted(map(tuple,r["n"]))==sorted(map(tuple,a["n"])): ex+=1
    extra=len(A)-len(used); N=max(1,scored)
    return dict(N=scored,recall=m/N,pitch=pp/N,exact=ex/N,extra=extra/N)
def obj(s): return (s["recall"]+s["pitch"]+s["exact"])/3 - 0.1*s["extra"]
