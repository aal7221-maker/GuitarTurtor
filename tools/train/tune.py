import json, optuna, pipe, numpy as np, sys
optuna.logging.set_verbosity(optuna.logging.WARNING)
NT=int(sys.argv[1]) if len(sys.argv)>1 else 250
def sample(t):
    p={}
    for k,v in pipe.SPACE.items():
        p[k]=t.suggest_categorical(k,v) if isinstance(v,list) else t.suggest_float(k,*v)
    return p
def evalset(params, spec):  # spec: list of (song, lo, hi)
    cache={}; res=[]
    for s,lo,hi in spec:
        if s not in cache: cache[s]=pipe.run(s,params)
        res.append(pipe.score(s,cache[s],lo,hi))
    return res
def tune(spec, n=NT, seed=0):
    st=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=seed))
    st.enqueue_trial({k:(v if k in pipe.SPACE else v) for k,v in pipe.DEFAULTS.items()})
    def f(t):
        r=evalset(sample(t),spec); return float(np.mean([pipe.obj(x) for x in r]))
    st.optimize(f,n_trials=n); return st.best_params
out={}
ALL=[("choo-lo",0,.6),("locked-away",0,.6)]; TEST=[("choo-lo",.6,1),("locked-away",.6,1)]
def rep(params):
    return {"train":evalset(params,ALL),"held":evalset(params,TEST),"full":evalset(params,[("choo-lo",0,1),("locked-away",0,1)])}
out["default"]=rep({})
bp=tune(ALL); out["split60"]={"params":bp, **rep(bp)}; print("split done",flush=True)
b1=tune([("choo-lo",0,1)]); out["loso_train_choo"]={"params":b1,"test_locked":evalset(b1,[("locked-away",0,1)]),"train":evalset(b1,[("choo-lo",0,1)])}
b2=tune([("locked-away",0,1)]); out["loso_train_locked"]={"params":b2,"test_choo":evalset(b2,[("choo-lo",0,1)]),"train":evalset(b2,[("locked-away",0,1)])}
json.dump(out,open("tune_results.json","w"),indent=1)
json.dump(bp,open("best_params.json","w"),indent=1)
