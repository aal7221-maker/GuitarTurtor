"""Strumming-hand motion from a real lesson video (MediaPipe hands via handtrack/hands.py, unmodified).

  <handtrack venv python> strum_track.py VIDEO --start 449 --end 529 --out strum_frames.json

Per frame: the strumming hand (player's right hand = the hand farther from the headstock; in an un-mirrored
front view that is the image-left hand) -> wrist + index-MCP + middle-MCP pixel positions, normalised by the frame
height, plus a scene-cut flag (large grey-level change) so velocities are not taken across camera cuts.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

HT = os.environ.get("HANDTRACK_DIR", r"D:\sharvin1\Projects\Music\handtrack")
sys.path.insert(0, HT)
from hands import MediaPipeHands  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--start", type=float, default=0.0)
ap.add_argument("--end", type=float, default=None)
ap.add_argument("--out", required=True)
ap.add_argument("--max-side", type=int, default=960)
a = ap.parse_args()

cap = cv2.VideoCapture(a.video)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
cap.set(cv2.CAP_PROP_POS_MSEC, a.start * 1000)
det = MediaPipeHands(num_hands=2, mirrored=False)
det.tta = False
frames, prev_small, prev_flow, i = [], None, None, 0
while True:
    ok, fr = cap.read()
    if not ok:
        break
    t = a.start + i / fps
    i += 1
    if a.end is not None and t > a.end:
        break
    h0, w0 = fr.shape[:2]
    s = min(1.0, a.max_side / max(h0, w0))
    if s < 1:
        fr = cv2.resize(fr, (int(w0 * s), int(h0 * s)), interpolation=cv2.INTER_AREA)
    H, W = fr.shape[:2]
    small = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (64, 36)).astype(np.float32)
    cut = bool(prev_small is not None and float(np.mean(np.abs(small - prev_small))) > 18.0)
    prev_small = small
    hands = det(fr, t)
    rec = {"t": round(t, 4), "cut": cut, "n": len(hands)}
    # dense optical flow in the strumming region (left 60 % of the frame, lower 80 %): the strumming hand is the
    # fastest-moving thing there, so the vertical flow of the top-5 % fastest pixels follows the strum even when
    # the hand is too motion-blurred for the landmark model
    g = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (480, 270))
    reg = g[54:, :288]
    if prev_flow is not None and not cut:
        fl = cv2.calcOpticalFlowFarneback(prev_flow, reg, None, 0.5, 3, 21, 3, 5, 1.1, 0)
        mag = np.hypot(fl[..., 0], fl[..., 1])
        thr = np.percentile(mag, 95)
        m = mag >= max(thr, 0.5)
        if m.sum() > 30:
            rec["vy"] = round(float(np.median(fl[..., 1][m])) / 270 * fps, 4)    # frame heights per second, + = downward
            rec["vmag"] = round(float(np.median(mag[m])) / 270 * fps, 4)
    prev_flow = reg
    # strumming hand: labelled 'Right' if MediaPipe says so, else the left-most hand in the image
    cand = [hd for hd in hands if hd.label == "Right"] or hands
    if cand:
        hd = min(cand, key=lambda q: q.pts[0, 0])
        if hd.pts[0, 0] < 0.62 * W:          # the fretting hand sits near the headstock on the right
            p = hd.pts / H
            rec.update({"wrist": p[0].round(4).tolist(), "imcp": p[5].round(4).tolist(), "mmcp": p[9].round(4).tolist(),
                        "itip": p[8].round(4).tolist(), "label": hd.label, "score": round(float(hd.score), 3)})
    frames.append(rec)
cap.release()
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
with open(a.out, "w") as f:
    json.dump({"video": os.path.basename(a.video), "fps": fps, "start": a.start, "frames": frames}, f)
print(f"[strum_track] {len(frames)} frames, strumming hand in {sum('wrist' in r for r in frames)} -> {a.out}")
