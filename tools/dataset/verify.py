"""Independent pitch-vs-audio check of read tabs with basic-pitch (Spotify, Apache-2.0).
For every tab pitch (sounding MIDI = tuning + capo + fret) take the max basic-pitch note posterior at that MIDI
(or its octave below/above at half weight) within [t-0.06, t+0.20] s. A page is "agreeing" when most of its
pitches have evidence; misread digits (10 read as 1, 6 as 0) land on pitches with ~no evidence.
Runs in the pitch-model venv (basic-pitch + onnxruntime):
  D:/sharvin1/Projects/Music/pitch-model/venv/Scripts/python.exe tools/dataset/verify.py [ids...]
Writes debug/<id>/bp.json (per-note evidence, per-page agreement). Posteriorgrams cached in cache/bp/."""
import os, sys, json, glob, subprocess
import numpy as np
sys.path.insert(0, r"D:\sharvin1\Projects\Music\pitch-model")
import transcribe as T  # noqa: E402  (loads the System32 msvcp140 first, then basic-pitch)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FF = r"C:/Users/sharv/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-9.1-full_build/bin/ffmpeg.exe"
FPS = 22050 / 256
NOTE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
THR = 0.25      # note posterior counted as evidence


def midi(s):
    n = NOTE[s[0].upper()]; i = 1
    if s[i] in "#b":
        n += 1 if s[i] == "#" else -1; i += 1
    return 12 * (int(s[i:]) + 1) + n


def post(key, audio):
    os.makedirs(os.path.join(HERE, "cache", "bp"), exist_ok=True)
    npz = os.path.join(HERE, "cache", "bp", key + ".npz")
    if os.path.exists(npz):
        return T.posteriorgrams(None, npz)
    wav = os.path.join(HERE, "cache", "bp", key + ".wav")
    subprocess.run([FF, "-v", "error", "-y", "-i", audio, "-vn", "-ac", "1", "-ar", "22050", wav], check=True)
    try:
        return T.posteriorgrams(wav, npz)
    finally:
        os.remove(wav)


def evidence(score, out, tun, capo):
    from basic_pitch.note_creation import model_frames_to_time
    note = out["note"]   # frames x 88 (MIDI 21..108)
    ft = model_frames_to_time(len(note))   # NOT frame/FPS: basic-pitch windows drift ~0.4 s/100 s otherwise
    res = []
    for n in score["notes"]:
        a = int(np.searchsorted(ft, n["t"] - 0.06)); b = int(np.searchsorted(ft, n["t"] + 0.20))
        ev = []
        for s, f in n["n"]:
            k = tun[s] + capo + f - 21
            if not (0 <= k < 88) or a >= b:
                ev.append(0.0); continue
            v = note[a:b, k].max()
            o = max(note[a:b, k - 12].max() if k >= 12 else 0, note[a:b, k + 12].max() if k + 12 < 88 else 0)
            ev.append(round(float(max(v, 0.5 * o)), 3))
        res.append(ev)
    return res


def summarize(score, ev):
    pages = {}
    for n, e in zip(score["notes"], ev):
        p = n["sec"].split("||")[0]
        pages.setdefault(p, []).extend(e)
    flat = [x for e in ev for x in e]
    return dict(agree=round(float(np.mean([x >= THR for x in flat])), 3) if flat else 0.0,
                pages={p: round(float(np.mean([x >= THR for x in v])), 3) for p, v in pages.items()},
                per_note=ev, thr=THR)


def run_item(vid, score_path, audio, tun, capo, key=None):
    score = json.load(open(score_path))
    out = post(key or vid, audio)
    r = summarize(score, evidence(score, out, tun, capo))
    print(vid, "agree", r["agree"], flush=True)
    return r


def main(ids):
    # priority songs (reference tabs) = calibration
    std = [64, 59, 55, 50, 45, 40]
    cal = {"choo-lo": (os.path.join(ROOT, "data", "score-choo-lo.json"), os.path.join(ROOT, "audio", "choo-lo.mp3")),
           "locked-away": (os.path.join(ROOT, "data", "score-locked-away.json"),
                           os.path.join(ROOT, "tools", "tabocr", "cache", "UtL5Ogfxqy4.mp4"))}
    for k, (sp, au) in cal.items():
        if ids and k not in ids:
            continue
        d = os.path.join(HERE, "debug", "_" + k); os.makedirs(d, exist_ok=True)
        if not os.path.exists(os.path.join(d, "bp.json")) and os.path.exists(au):
            json.dump(run_item(k, sp, au, std, 0), open(os.path.join(d, "bp.json"), "w"))
    for mf in sorted(glob.glob(os.path.join(HERE, "debug", "*", "metrics.json"))):
        m = json.load(open(mf)); d = os.path.dirname(mf)
        if (ids and m["id"] not in ids) or m.get("error") or os.path.exists(os.path.join(d, "bp.json")):
            continue
        tun = [midi(x) for x in m["tuning"].split(",")]
        if tun[0] < tun[-1]:
            tun = tun[::-1]
        au = os.path.join(HERE, "cache", "audio", m["id"] + ".m4a")
        if not os.path.exists(au):
            continue
        json.dump(run_item(m["id"], os.path.join(d, "score.json"), au, tun, m["capo"]), open(os.path.join(d, "bp.json"), "w"))


if __name__ == "__main__":
    main(sys.argv[1:])
