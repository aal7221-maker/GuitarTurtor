import json, time, numpy as np, pickle, pipe, chooser as C
from sklearn.ensemble import HistGradientBoostingClassifier
t0=time.time()
GSTR=C.load_guitarset([f"0{i}" for i in range(4)]); GSTE=C.load_guitarset(["05"])
gs_tr=list(GSTR.values()); gs_te=list(GSTE.values()); print("gs clips",len(gs_tr),len(gs_te),flush=True)
GSX=C.build(gs_tr,1.0); print("gs rows",len(GSX[1]),time.time()-t0,flush=True)
def fitmix(ours_seqs, w, use_gs=True, seed=0):
    parts=[GSX] if use_gs else []
    if ours_seqs: parts.append(C.build(ours_seqs,w))
    X=np.vstack([p[0] for p in parts]);y=np.concatenate([p[1] for p in parts]);sw=np.concatenate([p[2] for p in parts])
    m=HistGradientBoostingClassifier(max_iter=150,learning_rate=0.08,max_leaf_nodes=15,l2_regularization=1.0,random_state=seed)
    m.fit(X,y,sample_weight=sw); return m
tuned=json.load(open("best_params.json")) if __import__('os').path.exists("best_params.json") else {}
def e2e(name,model,lo,hi,params=None):
    p=dict(pipe.DEFAULTS); p.update(params or {})
    a=pipe.run(name,params, chooser=(lambda ev: C.decode(model,ev)) if model else None)
    s=pipe.score(name,a,lo,hi); return round(s["exact"],3)
def oracle(model,name,lo,hi,params=None):
    return round(C.exact_acc(model,C.events_from_ref(name,lo,hi),params),3)
def gs_acc(model,params=None):
    ok=n=0
    for sq in gs_te[:8]: 
        a=C.exact_acc(model,sq,params); ok+=a*len(sq); n+=len(sq)
    return round(ok/n,3)
SONGS=["choo-lo","locked-away"]; R={}
def model_set(ours_by_song_ranges, tag):
    seqs=[C.events_from_ref(n,lo,hi) for n,lo,hi in ours_by_song_ranges]
    return {"gs_only":fitmix([],0,True),"ours_only":fitmix(seqs,1,False),
            "gs+ours_w10":fitmix(seqs,10),"gs+ours_w25":fitmix(seqs,25)}
# split60
ms=model_set([(n,0,.6) for n in SONGS],"split")
R["split60"]={}
for k,m in list(ms.items())+[("viterbi_default",None),("viterbi_tuned",None)]:
    pr=tuned if k=="viterbi_tuned" else None
    if k=="viterbi_tuned" and not tuned: continue
    mm=None if k.startswith("viterbi") else m
    R["split60"][k]={"train_oracle":{n:oracle(mm,n,0,.6,pr) for n in SONGS},"held_oracle":{n:oracle(mm,n,.6,1,pr) for n in SONGS},
       "held_e2e_exact":{n:e2e(n,mm,.6,1,pr if mm is None else None) for n in SONGS},"gs_test":gs_acc(mm,pr)}
    print(k,R["split60"][k],flush=True)
pickle.dump(ms,open("work/models_split.pkl","wb"))
# LOSO
R["loso"]={}
for held in SONGS:
    tr=[n for n in SONGS if n!=held]; ms=model_set([(n,0,1) for n in tr],"loso")
    for k,m in list(ms.items())+[("viterbi_default",None)]:
        mm=None if k.startswith("viterbi") else m
        R["loso"].setdefault(k,{})[held]={"oracle":oracle(mm,held,0,1),"e2e_exact":e2e(held,mm,0,1),"gs_test":gs_acc(mm)}
        print("loso",held,k,R["loso"][k][held],flush=True)
# final: all data
final=model_set([(n,0,1) for n in SONGS],"final"); pickle.dump(final,open("work/models_final.pkl","wb"))
json.dump(R,open("chooser_results.json","w"),indent=1); print("done",time.time()-t0)
