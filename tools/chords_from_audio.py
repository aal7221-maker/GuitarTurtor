"""Chord track + strum pattern from a recording (local, offline). Output = the score fields the site reads (docs/CHORD-DATA.md):

  python tools/chords_from_audio.py AUDIO.(mp3|wav|m4a...) -o out.json [--engine auto|btc|chroma] [--no-strum]
  -> {"chords": [{t0,t1,name,conf}], "strum": {...score.strum, omitted when not strummed...}, "chord_engine": "...", "duration": s}

Chords: BTC (chords/detect_btc.py, large vocabulary, needs chords/third_party/BTC + torch, mir_eval, pretty_midi) when available,
else chords/detect_audio.py (librosa harmonic CQT chroma + templates + Viterbi). Strum: the browser's js/strumdetect.js run in
node (tools/strum_cli.mjs), so the site and this tool give identical strum output. Times are seconds into the audio file.
"""
import argparse, json, os, shutil, subprocess, sys, tempfile, pathlib

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "chords"))


def to_wav(src, dst):
    import numpy as np, librosa, soundfile as sf
    y, _ = librosa.load(str(src), sr=22050, mono=True)
    sf.write(str(dst), (np.clip(y, -1, 1) * 32767).astype("int16"), 22050, subtype="PCM_16")
    return len(y) / 22050


def btc_available():
    if not (ROOT / "chords" / "third_party" / "BTC" / "test.py").exists(): return False
    try:
        import torch, mir_eval, pretty_midi, yaml  # noqa: F401
        return True
    except Exception:
        return False


def chords_of(wav, engine):
    if engine in ("auto", "btc") and btc_available():
        import detect_btc
        segs, used = detect_btc.detect(str(wav)), "btc-large-voca"
    elif engine == "btc":
        raise SystemExit("BTC is not available (needs chords/third_party/BTC and torch, mir_eval, pretty_midi, pyyaml)")
    else:
        import detect_audio
        segs, used = detect_audio.detect(str(wav), vocab="basic"), "librosa-cqt-chroma-viterbi"
    out = []
    for s in segs:
        if s["chord"] in ("N", "X"): continue
        c = {"t0": round(s["t0"], 3), "t1": round(s["t1"], 3), "name": s["chord"], "conf": None if s.get("confidence") is None else round(s["confidence"], 3)}
        if out and out[-1]["name"] == c["name"] and abs(out[-1]["t1"] - c["t0"]) < 1e-3: out[-1]["t1"] = c["t1"]
        else: out.append(c)
    return [c for c in out if c["t1"] - c["t0"] >= 0.05], used


def strum_of(wav, chords):
    node = shutil.which("node")
    if not node: return None
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(chords, f); cf = f.name
    try:
        r = subprocess.run([node, str(HERE / "strum_cli.mjs"), str(wav), cf], capture_output=True, text=True, timeout=1800)
        return json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else None
    finally:
        os.unlink(cf)


def analyse(audio, engine="auto", strum=True):
    d = tempfile.mkdtemp()
    try:
        wav = pathlib.Path(d) / "a.wav"
        dur = to_wav(audio, wav)
        chords, used = chords_of(wav, engine)
        res = {"chords": chords, "chord_engine": used, "duration": round(dur, 3)}
        if strum:
            s = strum_of(wav, chords)
            if s: res["strum"] = s
        return res
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("audio"); ap.add_argument("-o", "--out")
    ap.add_argument("--engine", default="auto", choices=["auto", "btc", "chroma"]); ap.add_argument("--no-strum", action="store_true")
    a = ap.parse_args()
    res = analyse(a.audio, a.engine, not a.no_strum)
    txt = json.dumps(res)
    if a.out: pathlib.Path(a.out).write_text(txt, encoding="utf-8")
    s = res.get("strum") or {}
    print(f"{len(res['chords'])} chord segments ({res['chord_engine']}); strum: {(s.get('pattern') or {}).get('id')} "
          f"conf {s.get('confidence')} tempo {s.get('tempo')}" + (f" -> {a.out}" if a.out else ""))
