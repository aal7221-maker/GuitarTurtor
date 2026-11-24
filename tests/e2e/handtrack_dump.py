"""Run the handtrack fretting-hand tracker (D:/sharvin1/Projects/Music/handtrack/track.py, unmodified) on a real
performance video and dump PER-FRAME fingertip estimates, which track.py itself only keeps inside its Evaluator.

  <handtrack venv python> handtrack_dump.py --frames-out out.json -- --video clip.mp4 --score score.json [track.py args]

Output: {"fps", "frames": [{"t", "grid", "tips": [[finger 0..3, string_float, fret_float, string, fret], ...]}], "summary"}
(string 0 = high e, fret_float: 0 at the nut, n at fret wire n; same conventions as guitar3d).
"""
import json
import os
import sys

HT = os.environ.get("HANDTRACK_DIR", r"D:\sharvin1\Projects\Music\handtrack")
sys.path.insert(0, HT)

argv = sys.argv[1:]
out = argv[argv.index("--frames-out") + 1]
rest = argv[argv.index("--") + 1:]

import score as S  # noqa: E402  (handtrack/score.py)

FRAMES = []
_orig_add = S.Evaluator.add


def _add(self, t, grid_valid, snap_ok, tips, raw=None):
    FRAMES.append({"t": round(float(t), 4), "grid": bool(grid_valid),
                   "tips": [[int(e[0]), round(float(e[1]), 3), round(float(e[2]), 3), int(e[3]), int(e[4])] for e in tips]})
    return _orig_add(self, t, grid_valid, snap_ok, tips, raw)


S.Evaluator.add = _add
_orig_summary = S.Evaluator.summary
SUMMARY = {}


def _summary(self):
    s, notes = _orig_summary(self)
    SUMMARY["summary"] = s
    SUMMARY["notes"] = notes
    return s, notes


S.Evaluator.summary = _summary

import track  # noqa: E402

sys.argv = ["track.py"] + rest
track.main()
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
with open(out, "w") as f:
    json.dump({"args": rest, "frames": FRAMES, **SUMMARY}, f)
print(f"[handtrack_dump] {len(FRAMES)} frames -> {out}")
