"""Chord track + strum pattern for songs already on the site, written next to (not into) their score files.

  python tools/add_chords_to_song.py [ID ...] [--engine auto|btc|chroma] [--audio PATH --id ID]
  -> data/chords-<id>.json = {"id", "audio", "chords": [{t0,t1,name,conf}], "strum": {...}, "chord_engine", "offset"}

Songs come from data/songs.json (entries whose audio/<id>.mp3 exists locally). Times are converted to the score's clock:
the site's audio files start at video time 0, so offset = 0 unless the entry has "audioOffset". --audio analyses any file.
"""
import argparse, json, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from chords_from_audio import analyse


def shift(res, off):
    if not off: return res
    f = lambda t: round(t + off, 3)
    res["chords"] = [{**c, "t0": f(c["t0"]), "t1": f(c["t1"])} for c in res["chords"]]
    s = res.get("strum")
    if s:
        s["strokes"] = [{**k, "t": f(k["t"])} for k in s.get("strokes", [])]
        s["sections"] = [{**k, "t0": f(k["t0"]), "t1": f(k["t1"])} for k in s.get("sections", [])]
        if s.get("t_bar0") is not None: s["t_bar0"] = f(s["t_bar0"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*"); ap.add_argument("--engine", default="auto", choices=["auto", "btc", "chroma"])
    ap.add_argument("--audio"); ap.add_argument("--id")
    a = ap.parse_args()
    jobs = []
    if a.audio:
        jobs.append((a.id or pathlib.Path(a.audio).stem, pathlib.Path(a.audio), 0))
    else:
        for s in json.loads((ROOT / "data" / "songs.json").read_text(encoding="utf-8")):
            p = ROOT / (s.get("audio") or f"audio/{s['id']}.mp3")
            if (not a.ids or s["id"] in a.ids) and p.exists(): jobs.append((s["id"], p, s.get("audioOffset", 0)))
            elif not a.ids or s["id"] in a.ids: print(f"{s['id']}: no local audio, skipped")
    for sid, path, off in jobs:
        res = shift(analyse(path, a.engine), off)
        out = {"id": sid, "audio": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name, "offset": off, **res}
        (ROOT / "data" / f"chords-{sid}.json").write_text(json.dumps(out), encoding="utf-8")
        st = res.get("strum") or {}
        print(f"{sid}: {len(res['chords'])} chord segments ({res['chord_engine']}), first: "
              f"{' '.join(c['name'] for c in res['chords'][:8])} | strum {(st.get('pattern') or {}).get('id')} "
              f"conf {st.get('confidence')} tempo {st.get('tempo')} coverage {st.get('chord_coverage')} -> data/chords-{sid}.json")


if __name__ == "__main__":
    main()
