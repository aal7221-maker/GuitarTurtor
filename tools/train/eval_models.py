"""Full-pipeline comparison of pitch engines x string choosers on Choo Lo + Locked Away (same splits as pipe.py / tune.py).
Engines: basic-pitch (pipe.py cache), guitab E25 (work/<song>.guitab.json from guitab_raw.py), pitch_crnn (Colab ONNX).
Choosers: Viterbi (pipe.assign), string_chooser (Colab ONNX, beam), hybrid = Viterbi cost + w * -log p(string).
run: tools/train/.venv-ort/Scripts/python.exe tools/train/eval_models.py   -> eval_models.json
"""
import os, sys, json, subprocess, functools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import pipe
import onnxruntime as ort
OUT = os.path.join(HERE, "out", "colab", "out")
FFMPEG = r"C:/Users/sharv/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-9.1-full_build/bin/ffmpeg.exe"
OPEN = pipe.OPEN; NAMES = ["choo-lo", "locked-away"]
SR, HOP, P0, NP = 22050, 512, 40, 49; FPS = SR / HOP
ort.set_default_logger_severity(3)

# ---------------- pitch engines -> raw [(t0,t1,pitch,amp)]
def raw_bp(name): return list(pipe._ev(name, 0.5, 0.4, 60.0))
def raw_guitab(name): return sorted(tuple(x) for x in json.load(open(os.path.join(HERE, "work", name + ".guitab.json"))))

@functools.lru_cache(None)
def cqt(name):
    c = os.path.join(HERE, "work", name + ".cqt.npy")
    if os.path.exists(c): return np.load(c)
    import librosa
    y = np.frombuffer(subprocess.run([FFMPEG, "-v", "error", "-i", pipe.SONGS[name]["audio"], "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                                     capture_output=True, check=True).stdout, np.float32)
    X = np.log1p(10 * np.abs(librosa.cqt(y, sr=SR, hop_length=HOP, fmin=librosa.note_to_hz("C2"), n_bins=120, bins_per_octave=24))).T.astype("f2")
    np.save(c, X); return X

@functools.lru_cache(None)
def crnn_probs(name, model="pitch_crnn.onnx"):
    s = ort.InferenceSession(os.path.join(OUT, model))
    o, f = s.run(None, {"cqt": cqt(name)[None].astype("f")})
    sg = lambda z: 1 / (1 + np.exp(-z))
    return sg(o[0]), sg(f[0])

def crnn_notes(on, fr, thr=0.5, fthr=0.5, min_gap=2):
    """onset peaks (local max over time, > thr) -> notes; end = first frame where frame prob < fthr (or next onset)."""
    T = len(on); res = []
    pk = (on > thr) & (on >= np.roll(on, 1, 0)) & (on > np.roll(on, -1, 0))
    for k in range(NP):
        ts = np.nonzero(pk[:, k])[0]; last = -99
        for i, t in enumerate(ts):
            if t - last <= min_gap: continue
            last = t; nxt = ts[i + 1] if i + 1 < len(ts) else T
            e = t + 1
            while e < min(T, nxt) and fr[e, k] >= fthr: e += 1
            res.append((t / FPS, max(e, t + 2) / FPS, k + P0, float(on[t, k])))
    return sorted(res)
def raw_crnn(name, thr=0.5, fthr=0.5): return crnn_notes(*crnn_probs(name), thr, fthr)

# ---------------- string chooser (Colab MLP: [p/88] + 4 x (s/5, f/22, p/88, dt/2) -> logits over 6 strings)
H = 4
@functools.lru_cache(None)
def chooser_sess(path=os.path.join(OUT, "string_chooser.onnx")): return ort.InferenceSession(path)
def feat(p, t, hist):
    h = hist[-H:]; h = [(0, 0, 60, -9.0)] * (H - len(h)) + list(h)
    return [p / 88] + [v for hs, hf, hp, ht in h for v in (hs / 5, hf / 22, hp / 88, min(t - ht, 2.0) / 2)]
def logp_batch(rows, masks, sess):
    if not rows: return np.zeros((0, 6))
    lg = sess.run(None, {"x": np.array(rows, "f"), "mask": np.array(masks, "f")})[0].astype(float)
    lg = np.where(np.array(masks) > 0, lg, -1e9); lg -= lg.max(1, keepdims=True)
    return lg - np.log(np.exp(lg).sum(1, keepdims=True))

def assign_mix(events, p, w_vit=1.0, w_ch=1.0, sess=None, beam=64):
    """Beam search: cost = w_vit * Viterbi hand cost (pipe.assign) + w_ch * -log p_chooser(string | last 4 notes)."""
    sess = sess or chooser_sess()
    beams = [(0.0, [], None, ())]   # cost, path, anchor, hist (s,f,p,t)
    for e in events:
        cands = pipe.positions(e["p"], p["span"]) or [tuple((0, 0) for _ in e["p"])]
        # chooser log-probs per beam for each pitch (single-note events: one row per beam; chords: sequential, low->high)
        ps = sorted(e["p"]); rows = []; masks = []
        for _, _, _, hist in beams:
            for j, q in enumerate(ps[:1] if len(ps) == 1 else ps):
                m = [1.0 if 0 <= q - OPEN[s] <= 22 else 0.0 for s in range(6)]
                rows.append(feat(q, e["t"], list(hist))); masks.append(m if any(m) else [1.0] * 6)
        LP = logp_batch(rows, masks, sess) if w_ch else None
        nxt = {}; r = 0
        for bi, (cost, path, anchor, hist) in enumerate(beams):
            lp = LP[r:r + len(ps)] if w_ch else None; r += len(ps)
            for pos in cands:
                fb = tuple((s, f) for s, f in pos if f > 0); anc = fb or anchor
                if not fb: c = p["open_cost"]
                elif anchor is None: c = 0.0
                else:
                    sa = np.mean([s for s, _ in anchor]); fa = np.mean([f for _, f in anchor])
                    sb = np.mean([s for s, _ in fb]); fm = np.mean([f for _, f in fb]); c = 0.0
                    if sa != sb: c += p["str_change"] + p["str_jump"] * abs(sa - sb)
                    c += p["shift"] * abs(fa - fm) + (p["big_shift"] if abs(fa - fm) > 4 else 0)
                c += p["fret_pen"] * sum(f for _, f in pos)
                ro = {s for q in path[-2:] for s, f in q if f == 0}
                c += p["drone"] * sum(1 for s, f in fb if s in ro)
                tot = cost + w_vit * c
                if w_ch:   # chords: approximate - each note scored with the same (pre-event) history, by pitch order
                    srt = sorted(pos, key=lambda x: OPEN[x[0]] + x[1])
                    tot += w_ch * -sum(lp[j][s] for j, (s, f) in enumerate(srt))
                nh = (tuple(hist) + tuple((s, f, OPEN[s] + f, e["t"]) for s, f in sorted(pos, key=lambda x: OPEN[x[0]] + x[1])))[-H:]
                k = (pos, anc, tuple(x[:2] for x in nh))
                if k not in nxt or tot < nxt[k][0]: nxt[k] = (tot, path + [pos], anc, nh)
        beams = sorted(nxt.values(), key=lambda x: x[0])[:beam]
    return beams[0][1] if events else []

# ---------------- pipeline + scoring
P = dict(pipe.DEFAULTS)
def run(name, raw, chooser="viterbi", w=1.0, sess=None):
    events = pipe.group(raw, P)
    if chooser == "viterbi": pos = pipe.assign(events, P)
    elif chooser == "chooser": pos = assign_mix(events, P, 0.0, 1.0, sess)
    else: pos = assign_mix(events, P, 1.0, w, sess)
    return [{"t": e["t"], "n": [list(x) for x in q]} for e, q in zip(events, pos)]
SPL = {"train": (0, .6), "held": (.6, 1), "full": (0, 1)}
def sc(name, auto, split): lo, hi = SPL[split]; return pipe.score(name, auto, lo, hi)
def fmt(s): return f"{s['recall']:.3f} / {s['pitch']:.3f} / {s['exact']:.3f}"
def mean_exact(res, split): return float(np.mean([res[n][split]["exact"] for n in NAMES]))

def evaluate(rawf, chooser="viterbi", w=1.0, sess=None):
    res = {}
    for n in NAMES:
        auto = run(n, rawf(n), chooser, w, sess)
        res[n] = {k: sc(n, auto, k) for k in SPL}
    return res

if __name__ == "__main__":
    R = {}
    R["a_basicpitch_viterbi"] = evaluate(raw_bp)
    R["b_guitab_viterbi"] = evaluate(raw_guitab)
    # pitch_crnn: thresholds fixed at the notebook's 0.5 (its train split = our train split, memorised -> cannot select
    # on it); sweep printed for information only
    for thr in (0.3, 0.4, 0.6, 0.7):
        r = evaluate(lambda n: raw_crnn(n, thr, 0.5))
        print("info crnn thr", thr, {n: fmt(r[n]["held"]) for n in NAMES}, flush=True)
    CRNN_THR = (0.5, 0.5); R["c_crnn_viterbi"] = evaluate(lambda n: raw_crnn(n, *CRNN_THR))
    # best pitch engine by held-out mean exact (Viterbi)
    eng = {"basic-pitch": (raw_bp, "a_basicpitch_viterbi"), "guitab": (raw_guitab, "b_guitab_viterbi"),
           "pitch_crnn": (lambda n: raw_crnn(n, *CRNN_THR), "c_crnn_viterbi")}
    bestp = max(eng, key=lambda k: mean_exact(R[eng[k][1]], "held")); rawf = eng[bestp][0]
    print("best pitch by held-out:", bestp, flush=True)
    R["d_chooser"] = evaluate(rawf, "chooser"); R["d_chooser"]["pitch"] = bestp
    # hybrid weight on the train split only
    hy = {}
    for w in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0):
        hy[w] = evaluate(rawf, "hybrid", w); print("hybrid w", w, "train exact %.3f" % mean_exact(hy[w], "train"),
                                                  "held %.3f" % mean_exact(hy[w], "held"), flush=True)
    wb = max(hy, key=lambda w: mean_exact(hy[w], "train"))
    R["e_hybrid"] = hy[wb]; R["e_hybrid"]["w"] = wb; R["e_hybrid"]["pitch"] = bestp
    R["hybrid_sweep"] = {str(w): {n: hy[w][n] for n in NAMES} for w in hy}
    # same chooser variants on every pitch engine (for the table)
    for k, (f, _) in eng.items():
        if k == bestp: continue
        R[f"d_chooser[{k}]"] = evaluate(f, "chooser"); R[f"e_hybrid[{k}]"] = evaluate(f, "hybrid", wb)
    json.dump(R, open(os.path.join(HERE, "eval_models.json"), "w"), indent=1)
    for k, v in R.items():
        if k == "hybrid_sweep": continue
        print(f"{k:28s}", " | ".join(f"{n} {s}: {fmt(v[n][s])}" for n in NAMES for s in ("train", "held", "full")))
