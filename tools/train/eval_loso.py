"""LOSO guard for the choosers: chooser trained without the held song (loso_chooser.py), hybrid weight picked on the
OTHER song (whole song), tested on the whole held song. Pitch = basic-pitch (and guitab). Also teacher-forced vs
free-running string accuracy on reference pitches (the notebook's numbers are teacher-forced).
run: tools/train/.venv-ort/Scripts/python.exe tools/train/eval_loso.py  -> eval_loso.json"""
import os, json, numpy as np, onnxruntime as ort
import eval_models as E, chooser as C, pipe
HERE = E.HERE; W = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
def oracle(n, sess, lo=0, hi=1):
    seq = C.events_from_ref(n, lo, hi); notes = [(t, p, s, f) for t, ns in seq for p, s, f in sorted(ns)]
    rows = []; masks = []; hist = []
    for t, p, s, f in notes:
        rows.append(E.feat(p, t, hist)); masks.append([1.0 if 0 <= p - E.OPEN[k] <= 22 else 0 for k in range(6)]); hist.append((s, f, p, t))
    tf = float(np.mean(E.logp_batch(rows, masks, sess).argmax(1) == np.array([x[2] for x in notes])))
    evs = [{"t": t, "p": [p for p, _, _ in ns]} for t, ns in seq]
    ok = lambda pos: float(np.mean([sorted((s, f) for _, s, f in ns) == sorted(q) for (t, ns), q in zip(seq, pos)]))
    return dict(teacher_forced=tf, free_run=ok(E.assign_mix(evs, E.P, 0.0, 1.0, sess)), viterbi=ok(pipe.assign(evs, E.P)))
R = {"oracle_colab": {n: {sp: oracle(n, E.chooser_sess(), *E.SPL[sp]) for sp in ("train", "held")} for n in E.NAMES}}
for eng, rawf in (("basic-pitch", E.raw_bp), ("guitab", E.raw_guitab)):
    for held in E.NAMES:
        other = [n for n in E.NAMES if n != held][0]
        sess = ort.InferenceSession(os.path.join(HERE, "work", f"chooser_loso_{held}.onnx"))
        sc = lambda n, mode, w=1.0: E.sc(n, E.run(n, rawf(n), mode, w, sess), "full")
        wb = max(W, key=lambda w: sc(other, "hybrid", w)["exact"])
        R.setdefault(eng, {})[held] = dict(viterbi=sc(held, "viterbi"), chooser=sc(held, "chooser"), hybrid=sc(held, "hybrid", wb), w=wb)
        if eng == "basic-pitch": R.setdefault("oracle_loso", {})[held] = oracle(held, sess)
        print(eng, held, {k: (E.fmt(v) if isinstance(v, dict) else v) for k, v in R[eng][held].items()}, flush=True)
print("oracle", json.dumps({k: R[k] for k in ("oracle_colab", "oracle_loso")}, indent=0))
json.dump(R, open(os.path.join(HERE, "eval_loso.json"), "w"), indent=1)
