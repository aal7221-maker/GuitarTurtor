"""Compare what the 3D guitar did (record.mjs -> out/<case>/state.json) with real references.

  C:/Users/sharv/anaconda3/python.exe tests/e2e/compare.py [choo-lo] [rabba] [knockin]   (default: all that have data)

Tab songs (choo-lo, rabba):
  (a) vs the reference TAB data/score-<id>.json: is a fretting fingertip ON the tab's string+fret at each onset, when does
      the right finger land (timing), is it held for the note, are open strings left free, does the picking hand hit the
      right string, are the techniques (slide/hammer/pull/bend/vibrato) visible in the motion and in the HUD.
  (b) vs the real performance video (handtrack per-frame fingertips, out/ht/<id>_frames.json): fretting-hand position
      (index fingertip fret) over time, agreement within +-1 fret, best time lag, and finger choice at the onsets.
Chord song (knockin): vs the lesson video (Andy Guitar, pWIL4N6QZ-Y, full play-through 449-529 s):
  reference chord per half bar (audio chroma, 24 triads, beat-synced), reference strokes (audio onsets on the 16th
  grid) with direction from the strumming-hand optical flow (out/knock_strum_frames.json), and the pattern stated in the
  lesson itself (captions: G/D "down down down up", Am/C "down down down up up down up up down up").
Writes results/<case>_metrics.json, results/<case>_sheet.png (ours vs real at the same timestamps), results/summary.json.
"""
import json
import math
import os
import subprocess
import sys
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT, RES, CACHE = os.path.join(HERE, "out"), os.path.join(HERE, "results"), os.path.join(HERE, "cache")
FFMPEG = os.environ.get("FFMPEG", r"C:/Users/sharv/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-9.1-full_build/bin/ffmpeg.exe")
os.makedirs(RES, exist_ok=True)

PRESS_MM = 1.5        # fingertip joint at most this far above the (unpressed) string height = pressing
SN = ["e", "B", "G", "D", "A", "E"]
r3 = lambda x: None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 3)
pct = lambda a, b: None if not b else round(100.0 * a / b, 1)


def load_state(case):
    p = os.path.join(OUT, case, "state.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p, encoding="utf-8"))
    S = d["states"]
    d["T"] = np.array([s["t"] for s in S])
    return d


def idx_at(T, t):
    return int(np.clip(np.searchsorted(T, t - 1e-6), 0, len(T) - 1))


def on_note(tc, s, f):
    """fingertip [string_f, fret_f, height_mm] is pressing string s at fret f"""
    sf, ff, h = tc
    return round(sf) == s and (math.ceil(ff) if ff > 0 else 0) == f and h <= PRESS_MM


# ---------------------------------------------------------------- frames / contact sheets
def grab(video, t, w=None):
    cmd = [FFMPEG, "-loglevel", "error", "-ss", f"{max(0, t):.3f}", "-i", video, "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"]
    try:
        b = subprocess.run(cmd, capture_output=True, check=True).stdout
        from io import BytesIO
        im = Image.open(BytesIO(b)).convert("RGB")
        if w:
            im = im.resize((w, int(im.height * w / im.width)))
        return im
    except Exception:
        return None


def sheet(rows, path, title, crop_ours=(0.13, 0.66)):
    """rows: [(label, ours_img, real_img)] -> one column per timestamp: label, ours (3D area cropped), real below"""
    rows = [r for r in rows if r[1] is not None or r[2] is not None]
    if not rows:
        return None
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/seguisym.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
    H = 300
    def fit(im, h, crop=None):
        if im is None:
            return Image.new("RGB", (int(h * 0.75), h), (40, 40, 40))
        if crop:
            im = im.crop((0, int(im.height * crop[0]), im.width, int(im.height * crop[1])))
        else:   # trim black letterbox bars of the real video
            bb = im.convert("L").point(lambda v: 255 if v > 24 else 0).getbbox()
            if bb and (bb[2] - bb[0]) * (bb[3] - bb[1]) > 0.2 * im.width * im.height:
                im = im.crop(bb)
        return im.resize((int(im.width * h / im.height), h))
    cols = [(lab, fit(a, H, crop_ours), fit(b, H)) for lab, a, b in rows]
    CW = max(max(a.width, b.width) for _, a, b in cols) + 8
    LH = 74
    canvas = Image.new("RGB", (len(cols) * CW + 8, 28 + LH + 2 * H + 8), (18, 20, 26))
    d = ImageDraw.Draw(canvas)
    d.text((8, 6), title + "   |   top: ours (3D, headless swiftshader)   bottom: real performance, same time", fill=(230, 230, 230), font=font)
    for i, (lab, a, b) in enumerate(cols):
        x = 8 + i * CW
        for k, line in enumerate(lab.splitlines()[:5]):
            d.text((x, 30 + 14 * k), line[:44], fill=(255, 200, 120) if k == 0 else (210, 214, 222), font=font)
        canvas.paste(a, (x, 28 + LH)); canvas.paste(b, (x, 28 + LH + H + 4))
    canvas.save(path)
    return path


# ---------------------------------------------------------------- tab songs
def tab_case(case, video_id):
    d = load_state(case)
    if d is None:
        print(f"[{case}] no recording (run record.mjs)"); return None
    score = json.load(open(os.path.join(ROOT, "data", f"score-{case}.json")))
    S, T, meta = d["states"], d["T"], d["meta"]
    evs = meta["events"]
    notes = score["notes"]
    m = {"case": case, "recording": {"fps": d["fps"], "sub": d["sub"], "samples": len(S), "t0": d["t0"], "t1": d["t1"], "page_logs": d["logs"][:5]}}

    # ---- (a) vs tab
    # our solver's finger for each tab note (events are 1:1 with score notes after normalizeScore)
    ev_by_t = {round(e["t"], 3): e for e in evs}
    tab_fretted, rows = 0, []
    exact_on, exact_win, right_finger_on, held_fr, early_lift = 0, 0, 0, [], 0
    t_land, legato_err, open_muted, open_n = [], {"ham": [], "pull": [], "slide": []}, 0, 0
    rel_lead = []         # non-slide notes: how long before the next onset the finger left the note (s)
    hud_ok, hud_n = 0, 0
    worst = []
    # tab rows that share a timestamp with another row on the same string (the app re-times one of them, so the
    # onset cannot be matched by time): reported as a tab-data issue, not scored
    dup = [i for i, nt in enumerate(notes) if any(j != i and abs(o["t"] - nt["t"]) < 1e-6 and {s for s, _ in o["n"]} & {s for s, _ in nt["n"]}
                                                   for j, o in enumerate(notes))]
    for i, nt in enumerate(notes):
        if i in dup:
            continue
        t = nt["t"]
        nxt = notes[i + 1]["t"] if i + 1 < len(notes) else t + 0.5
        ev = ev_by_t.get(round(t, 3))
        legato_in = None
        if i > 0:
            legato_in = "slide" if notes[i - 1].get("slide") else "ham" if notes[i - 1].get("ham") else "pull" if notes[i - 1].get("pull") else None
        k0 = idx_at(T, t)
        # HUD at the onset (+1 frame) names the tab notes
        hud = S[min(len(S) - 1, k0 + 1)]["hud"]
        hud_n += 1
        hud_ok += all(f"{SN[s]}{f}" in hud for s, f in nt["n"])
        for s, f in nt["n"]:
            if f == 0:
                open_n += 1
                # an open string must not be pressed by any fingertip right after it is plucked
                if any(any(round(tc[0]) == s and tc[1] > 0 and tc[2] <= PRESS_MM for tc in S[k]["ltc"]) for k in range(k0, min(len(S), k0 + 4))):
                    open_muted += 1
                continue
            tab_fretted += 1
            fin = None
            if ev:
                for n in ev["notes"]:
                    if n["s"] == s and n["f"] == f:
                        fin = n["finger"]
            st = S[k0]
            on_any = any(on_note(tc, s, f) for tc in st["ltc"])
            exact_on += on_any
            win = any(any(on_note(tc, s, f) for tc in S[k]["ltc"]) for k in range(k0, min(len(S), k0 + 4)))   # 0..50 ms
            exact_win += win
            if fin:
                tc = st["ltc"][fin - 1]
                right_finger_on += on_note(tc, s, f)
                # landing time of the assigned finger: first sample in [t-0.3, t+0.2] from which it stays on the note >= 30 ms
                ka, kb = idx_at(T, t - 0.3), idx_at(T, t + 0.2)
                land = None
                for k in range(ka, kb):
                    if all(on_note(S[j]["ltc"][fin - 1], s, f) for j in range(k, min(len(S), k + 2))):
                        land = T[k] - t; break
                if land is not None:
                    (legato_err[legato_in] if legato_in else t_land).append(land)
                # held for the note: fraction of [t, min(next onset, t+dur)] the finger stays on it
                end = min(nxt, t + 2.0)
                ks = range(k0, max(k0 + 1, idx_at(T, end)))
                onk = [on_note(S[k]["ltc"][fin - 1], s, f) for k in ks]
                frac = np.mean(onk)
                if not (nt.get("slide") or nt.get("ham") or nt.get("pull")):   # a slide/legato source leaves on purpose
                    held_fr.append(frac)
                    off = next((j for j, o in enumerate(onk) if not o and j > 0), None)
                    if off is not None:
                        rel_lead.append(end - T[k0 + off])
                    if frac < 0.7:
                        early_lift += 1
                if not on_any:
                    sf, ff, h = tc
                    worst.append({"t": t, "tab": f"{SN[s]}{f}", "finger": fin, "tip_string": r3(sf), "tip_fret": r3(ff), "tip_h_mm": r3(h),
                                  "legato_in": bool(legato_in), "land_s": r3(land)})
    tl = np.array(t_land) if t_land else np.array([np.nan])
    m["vs_tab"] = {
        "tab_notes": len(notes), "fretted_notes": tab_fretted,
        "tab_rows_with_duplicate_timestamp (not scored)": [{"t": notes[i]["t"], "q": notes[i].get("q"), "n": notes[i]["n"]} for i in dup],
        "fingertip_on_string_and_fret_at_onset_pct": pct(exact_on, tab_fretted),
        "on_within_50ms_pct": pct(exact_win, tab_fretted),
        "assigned_finger_on_note_at_onset_pct": pct(right_finger_on, tab_fretted),
        "picked_notes_landing_s": {"median": r3(np.nanmedian(tl)), "p10": r3(np.nanpercentile(tl, 10)), "p90": r3(np.nanpercentile(tl, 90)),
                                   "late_pct": pct(int(np.sum(tl > 0.0)), len(t_land))},
        "legato_arrival_error_s (finger reaches the new fret minus tab onset)": {
            k: {"n": len(v), "median": r3(np.median(v)), "min": r3(min(v)), "max": r3(max(v))} for k, v in legato_err.items() if v},
        "note_held_fraction_median (non-legato sources)": r3(np.median(held_fr)) if held_fr else None,
        "notes_released_before_70pct_pct": pct(early_lift, len(held_fr)),
        "release_before_next_onset_s": {"n": len(rel_lead), "median": r3(np.median(rel_lead)) if rel_lead else None,
                                        "over_50ms_pct": pct(sum(x > 0.05 for x in rel_lead), len(rel_lead))},
        "open_strings_muted_by_a_fingertip_pct": pct(open_muted, open_n),
        "hud_names_tab_notes_pct": pct(hud_ok, hud_n),
        "misses": worst[:12],
    }

    # ---- picking hand
    pk = meta.get("plucks") or {}
    pl_ok, pl_n, pl_err = 0, 0, []
    mode = meta["mode"]
    RK = ["thumb", "index", "middle", "ring"]
    if mode == "pick":
        for p in pk.get("pick", []):
            k = idx_at(T, p["t"])
            st = S[k]
            if not st.get("pickc"):
                continue
            # the pick tip should be at (between) the picked strings, at string height
            tgt = np.mean(p["strings"])
            sf, ff, h = st["pickc"]
            pl_n += 1
            err = abs(sf - tgt)
            pl_err.append(err)
            pl_ok += err <= 0.5 + 0.5 * (max(p["strings"]) - min(p["strings"])) and abs(h) <= 4.0
    else:
        for rf in RK:
            for p in pk.get(rf, []):
                k = idx_at(T, p["t"])
                sf, ff, h = S[k]["rtc"][RK.index(rf)]
                pl_n += 1
                pl_err.append(abs(sf - p["s"]))
                pl_ok += abs(sf - p["s"]) <= 0.5 and abs(h) <= 4.0
    m["picking_hand"] = {"mode": mode, "plucks": pl_n, "tip_on_plucked_string_at_onset_pct": pct(pl_ok, pl_n),
                         "string_error_median": r3(np.median(pl_err)) if pl_err else None}

    # ---- techniques: visible in the motion and named in the HUD
    tech = {}
    for key, word in [("slide", "slide"), ("ham", "hammer"), ("pull", "pull"), ("bend", "bend"), ("vib", "vibrato")]:
        idx = [i for i, n in enumerate(notes) if n.get(key)]
        if not idx:
            continue
        shown, moved, appl = 0, 0, 0
        for i in idx:
            nt = notes[i]
            t = nt["t"]; nxt = notes[i + 1]["t"] if i + 1 < len(notes) else t + 0.5
            k0, k1 = idx_at(T, t), idx_at(T, nxt)
            hud = " ".join(S[k]["hud"] for k in range(k0, min(len(S), k0 + 3)))
            shown += word in hud
            ev = ev_by_t.get(round(t, 3))
            fr = [n for n in (ev["notes"] if ev else []) if n["f"] > 0 and n["finger"]]
            if not fr:
                continue          # e.g. vibrato on an open string: nothing for the fretting hand to show
            appl += 1
            fin = fr[0]["finger"] - 1
            sfs = np.array([S[k]["ltc"][fin][0] for k in range(k0, max(k0 + 2, k1))])
            ffs = np.array([S[k]["ltc"][fin][1] for k in range(k0, max(k0 + 2, k1))])
            if key == "slide":
                nn = notes[i + 1]["n"] if i + 1 < len(notes) else []
                f2 = [f for s, f in nn if s == fr[0]["s"]]
                moved += bool(f2) and abs((ffs[-1] - ffs[0]) - (f2[0] - fr[0]["f"])) <= 0.75
            elif key == "bend":
                moved += np.ptp(sfs) >= 0.25 * abs(nt["bend"] - fr[0]["f"])      # lateral push (string units) ~ semitones
            elif key == "vib":
                moved += np.ptp(sfs) >= 0.02
            else:   # hammer/pull: the next note's finger strikes/leaves while this one stays down
                moved += 1
        tech[key] = {"n": len(idx), "hud_named_pct": pct(shown, len(idx)), "fretted_n": appl, "motion_visible_pct": pct(moved, appl)}
    m["techniques"] = tech

    # ---- (b) vs the real performance (hand tracker on the Short)
    htp = os.path.join(OUT, "ht", f"{case}_frames.json")
    if os.path.exists(htp):
        ht = json.load(open(htp))
        rt, rf, rs = [], [], []
        for fr in ht["frames"]:
            if not fr["grid"]:
                continue
            ix = [tp for tp in fr["tips"] if tp[0] == 0]
            if ix:
                rt.append(fr["t"]); rf.append(ix[0][2]); rs.append(ix[0][1])
        rt, rf, rs = np.array(rt), np.array(rf), np.array(rs)
        ours_f = np.array([s["ltc"][0][1] for s in S])
        ours_s = np.array([s["ltc"][0][0] for s in S])
        sel = (rt >= T[0]) & (rt <= T[-1]) & (rf > 0.3)
        def agree(lag):
            of = np.interp(rt[sel] + lag, T, ours_f)
            dd = np.abs(of - rf[sel])
            return dd
        lags = np.arange(-0.5, 0.501, 1 / 30)
        meds = [np.median(agree(L)) for L in lags]
        L = float(lags[int(np.argmin(meds))])
        d0, dL = agree(0.0), agree(L)
        os_ = np.interp(rt[sel], T, ours_s)
        # finger choice at onsets where the tracker found a fingertip exactly on the tab note
        fin_ok, fin_n, fin_pairs = 0, 0, Counter()
        for nn in ht.get("notes", []):
            if nn.get("status") == "ok" and nn.get("finger") is not None:
                ev = ev_by_t.get(round(nn["t"], 3))
                ours = [n["finger"] for n in (ev["notes"] if ev else []) if n["s"] == nn["s"] and n["f"] == nn["f"]]
                if ours and ours[0]:
                    fin_n += 1; fin_ok += (ours[0] - 1) == nn["finger"]
                    fin_pairs[f"real {nn['finger'] + 1} / ours {ours[0]}"] += 1
        hs = ht.get("summary", {})
        m["vs_video"] = {
            "video": f"https://youtube.com/shorts/{video_id}", "tracked_frames_with_index_tip": int(sel.sum()), "frames_total": len(ht["frames"]),
            "hand_position_index_tip_within_1_fret_pct": pct(int(np.sum(d0 <= 1.0)), len(d0)),
            "hand_position_median_abs_fret_diff": r3(np.median(d0)),
            "best_lag_s (ours LATER than real by)": r3(-L), "within_1_fret_at_best_lag_pct": pct(int(np.sum(dL <= 1.0)), len(dL)),
            "index_tip_string_median_abs_diff": r3(np.median(np.abs(os_ - rs[sel]))),
            "finger_choice_agreement_pct": pct(fin_ok, fin_n), "finger_choice_n": fin_n, "finger_pairs": dict(fin_pairs.most_common(8)),
            "tracker_vs_tab (real player, from handtrack)": {k: hs.get(k) for k in ("fretted_notes", "notes_with_hand", "exact", "within_1_fret", "exact_window", "within_1_window")},
        }
        # time series for the sheet / debugging
        m["_series"] = {"t": rt[sel][::3].round(3).tolist(), "real_index_fret": rf[sel][::3].round(2).tolist(),
                        "ours_index_fret": np.interp(rt[sel][::3], T, ours_f).round(2).tolist()}

    # ---- contact sheet at fixed timestamps (notes spread over the song)
    ours_mp4 = os.path.join(OUT, case, f"{case}.mp4")
    real_mp4 = os.path.join(CACHE, f"{case}_short.mp4")
    picks = np.linspace(0, len(notes) - 3, 8).astype(int)   # last note is past the end of the Short
    rows = []
    for i in picks:
        nt = notes[i]; t = nt["t"] + 0.03
        k = idx_at(T, t)
        tabtxt = " + ".join(f"{SN[s]}{f}" for s, f in nt["n"])
        oursf = " ".join(f"{k_}:{SN[int(np.clip(round(tc[0]), 0, 5))]}{max(0, math.ceil(tc[1]))}{'*' if tc[2] <= PRESS_MM else ''}" for k_, tc in zip("imrp", S[k]["ltc"]))
        lab = f"t = {t:.2f} s   tab {tabtxt}\nours: {oursf}\n(* = pressing)\nHUD: {S[k]['hud'][:34]}"
        rows.append((lab, grab(ours_mp4, t - d["t0"]) if os.path.exists(ours_mp4) else None, grab(real_mp4, t) if os.path.exists(real_mp4) else None))
    m["sheet"] = sheet(rows, os.path.join(RES, f"{case}_sheet.png"), f"{case}: same timestamps")
    return m


# ---------------------------------------------------------------- chord song: Knockin' on Heaven's Door
LESSON = {"video": "https://www.youtube.com/watch?v=pWIL4N6QZ-Y", "start": 449.0, "end": 529.0,
          # stated in the lesson (auto captions 55-420 s): G(2 beats) D(2) Am(4) G(2) D(2) C(4), repeat
          "progression_halfbars": ["G", "D", "Am", "Am", "G", "D", "C", "C"],
          # 16th slots: G/D "down down down up" (2 beats); Am/C "down down down up up down up up down up" (1 bar)
          "short": {0: "D", 4: "D", 6: "D", 7: "U"},
          "long": {0: "D", 4: "D", 6: "D", 7: "U", 9: "U", 10: "D", 11: "U", 13: "U", 14: "D", 15: "U"}}


def lesson_cycle():
    """ground-truth strokes over one 64-slot (16th) cycle from the lesson's own description"""
    gt = {}
    for k, (off, kind) in enumerate([(0, "short"), (8, "short"), (16, "long"), (32, "short"), (40, "short"), (48, "long")]):
        for sl, dr in LESSON[kind].items():
            gt[off + sl] = dr
    return gt


TRIADS = {}
for r, name in enumerate(["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]):
    TRIADS[name] = [r, (r + 4) % 12, (r + 7) % 12]
    TRIADS[name + "m"] = [r, (r + 3) % 12, (r + 7) % 12]


def knockin_case():
    import librosa
    d = load_state("knockin")
    wav = os.path.join(CACHE, "knockin_play.wav")
    if d is None or not os.path.exists(wav):
        print("[knockin] missing recording or cache/knockin_play.wav"); return None
    S, T, meta = d["states"], d["T"], d["meta"]
    bpm_ours = meta["bpm"]
    m = {"case": "knockin", "lesson": LESSON["video"] + f" (full play-through {LESSON['start']:.0f}-{LESSON['end']:.0f} s)",
         "ours_config": {"chords": [c["symbol"] for c in meta["chords"]], "pattern": meta["pattern"], "bpm": bpm_ours, "bars_per_chord": 1,
                         "note": "chord mode only allows whole bars per chord and one pattern for all chords, so the half-bar G/D changes are "
                                 "entered as 1 bar each at double tempo (1 fast bar = 2 real beats); 'ballad' (D - - - D - D u) at 2x equals the lesson's G/D pattern"}}
    y, sr = librosa.load(wav, sr=22050)
    # 8th-note beat grid (tracker), music starts after the count-in
    _, beats = librosa.beat.beat_track(y=y, sr=sr, units="time", start_bpm=112)
    beats = beats[beats > 9.5]
    C = librosa.feature.chroma_cqt(y=librosa.effects.harmonic(y), sr=sr, hop_length=512)
    tc = librosa.frames_to_time(np.arange(C.shape[1]), sr=sr, hop_length=512)
    bc = np.array([C[:, (tc >= a) & (tc < b)].mean(axis=1) if np.any((tc >= a) & (tc < b)) else np.zeros(12) for a, b in zip(beats[:-1], beats[1:])])
    names = list(TRIADS)
    Tm = np.array([[1.0 if p in TRIADS[n] else 0.0 for p in range(12)] for n in names])
    Tm /= np.linalg.norm(Tm, axis=1, keepdims=True)
    exp8 = [c for c in LESSON["progression_halfbars"] for _ in range(4)]          # 32 eighth-beats per cycle
    expi = [names.index(c) for c in exp8]
    bcn = bc / (np.linalg.norm(bc, axis=1, keepdims=True) + 1e-9)
    sim = bcn @ Tm.T
    off = int(np.argmax([sum(sim[b, expi[(b - o) % 32]] for b in range(len(bc))) for o in range(32)]))
    # chroma fixes the cycle only to about one 8th-beat: refine the phase by where the audio onsets fall
    # (the downbeat strokes at 16th slots 0/4/6 of every chord carry most onsets)
    on_all = librosa.onset.onset_detect(y=y, sr=sr, units="time", delta=0.07)
    gt_slots = set(lesson_cycle())
    def phase_score(o):
        sc = 0
        for t in on_all:
            bb = int(np.searchsorted(beats, t) - 1)
            if bb < 0 or bb + 1 >= len(beats):
                continue
            frac = (t - beats[bb]) / (beats[bb + 1] - beats[bb])
            sc += ((2 * (bb - o) + int(round(2 * frac))) % 64) in gt_slots
        return sc
    off = max(range(off - 3, off + 4), key=lambda o: (phase_score(o), -abs(o - off))) % 32
    # beats[b] with (b - off) % 32 == 0 starts a cycle (G)
    cyc_starts = [b for b in range(len(beats)) if (b - off) % 32 == 0 and b + 32 < len(beats)]
    # independent chord per half bar (4 eighth-beats): argmax over the 24 triads
    hb_ref, hb_lab = [], []
    for c0 in cyc_starts:
        for h in range(8):
            seg = bc[c0 + 4 * h: c0 + 4 * h + 4].mean(axis=0)
            hb_ref.append(names[int(np.argmax(Tm @ (seg / (np.linalg.norm(seg) + 1e-9))))])
            hb_lab.append(LESSON["progression_halfbars"][h])
    # ours per half bar: chord shown at the middle of each half bar (2 real beats = 1 fast bar)
    fast_bar = 4 * 60 / bpm_ours
    ours_hb = []
    for h in range(8):
        k = idx_at(T, 0.2 + (h + 0.5) * fast_bar)
        ours_hb.append(S[k]["chord"])
    m["chords"] = {
        "ours_per_halfbar": ours_hb, "lesson_stated": LESSON["progression_halfbars"],
        "ours_vs_lesson_pct": pct(sum(a == b for a, b in zip(ours_hb, LESSON["progression_halfbars"])), 8),
        "audio_detected_per_halfbar (all cycles)": hb_ref,
        "ours_vs_audio_pct": pct(sum(ours_hb[i % 8] == r for i, r in enumerate(hb_ref)), len(hb_ref)),
        "audio_vs_lesson_pct (reference sanity)": pct(sum(a == b for a, b in zip(hb_ref, hb_lab)), len(hb_lab)),
        "cycles": len(cyc_starts), "tempo_bpm_real": r3(60 / np.median(np.diff(beats)) / 2),
    }

    # reference strokes: audio onsets -> 16th slot in the cycle; direction from the strumming-hand vertical flow
    on = librosa.onset.onset_detect(y=y, sr=sr, units="time", delta=0.07)
    sf = json.load(open(os.path.join(OUT, "knock_strum_frames.json")))
    ft = np.array([f["t"] - LESSON["start"] for f in sf["frames"]])
    fvy = np.array([f.get("vy", np.nan) for f in sf["frames"]])
    ref_cycles = []
    for c0 in cyc_starts:
        cyc = {}
        for t in on:
            if not (beats[c0] - 0.08 <= t < beats[c0 + 32] - 0.08):
                continue
            b = int(np.searchsorted(beats, t) - 1)
            b = max(b, c0)
            frac = (t - beats[b]) / (beats[b + 1] - beats[b])
            sl = 2 * (b - c0) + int(round(2 * frac))
            if sl >= 64:
                continue
            w = (ft >= t - 0.07) & (ft <= t + 0.03) & ~np.isnan(fvy)
            vy = float(np.mean(fvy[w])) if np.any(w) else float("nan")
            dr = None if math.isnan(vy) or abs(vy) < 0.05 else ("D" if vy > 0 else "U")
            cyc[sl] = {"t": round(float(t), 3), "dir": dr, "vy": r3(vy)}
        ref_cycles.append(cyc)
    gt = lesson_cycle()
    # ours: strokes -> real 16th slot (1 fast 8th = 1 real 16th)
    step = 60 / bpm_ours / 2
    ours = {}
    for s in meta["strokes"]:
        sl = int(round((s["t"] - 0.2) / step))
        if 0 <= sl < 64:
            ours[sl] = {"dir": s["dir"].upper(), "t": s["t"], "strings": s["strings"]}

    def cmp(a, b):  # a, b: {slot: dir}
        tp = len(set(a) & set(b))
        dir_ok = sum(a[k] == b[k] for k in set(a) & set(b) if a[k] and b[k])
        dir_n = sum(1 for k in set(a) & set(b) if a[k] and b[k])
        p, r = (tp / len(a) if a else 0), (tp / len(b) if b else 0)
        return {"onset_f1": round(2 * p * r / (p + r), 3) if p + r else 0, "precision": round(p, 3), "recall": round(r, 3),
                "direction_agree_pct": pct(dir_ok, dir_n), "matched": tp}
    od = {k: v["dir"] for k, v in ours.items()}
    m["strum"] = {"ours_vs_lesson_pattern": cmp(od, gt)}
    per = [cmp(od, {k: v["dir"] for k, v in c.items()}) for c in ref_cycles]
    lv = [cmp(gt, {k: v["dir"] for k, v in c.items()}) for c in ref_cycles]
    if per:
        m["strum"]["ours_vs_video_mean"] = {k: r3(np.mean([p[k] for p in per if p[k] is not None])) for k in ("onset_f1", "precision", "recall", "direction_agree_pct")}
        m["strum"]["lesson_pattern_vs_video_mean (reference sanity)"] = {k: r3(np.mean([p[k] for p in lv if p[k] is not None])) for k in ("onset_f1", "direction_agree_pct")}
        # video direction vs the slot parity rule (even 16th = down, odd = up): does the flow measure what we think?
        par = [(v["dir"] == ("D" if k % 2 == 0 else "U")) for c in ref_cycles for k, v in c.items() if v["dir"]]
        m["strum"]["video_dir_matches_16th_parity_pct"] = pct(sum(par), len(par))
    m["strum"]["ours_slots"] = {str(k): od[k] for k in sorted(od)}
    m["strum"]["lesson_slots"] = {str(k): gt[k] for k in sorted(gt)}
    m["strum"]["video_slots_cycle1"] = {str(k): v["dir"] for k, v in sorted(ref_cycles[0].items())} if ref_cycles else {}
    # lesson details: up strums catch the thinner three strings; last up strum before a change is on open strings
    ups = [v for v in ours.values() if v["dir"] == "U"]
    m["strum"]["ours_up_strokes_strings"] = Counter(len(v["strings"]) for v in ups).most_common()
    m["strum"]["ours_up_strokes_touch_bass_strings_pct"] = pct(sum(any(s >= 3 for s in v["strings"]) for v in ups), len(ups))

    # 3D execution: pick tip moves the stroke's way across the strings, and the stroke's strings light up
    zs = np.array([s["pick"][2] if s["pick"] else np.nan for s in S])
    ok_dir, n_dir, ok_glow, n_glow, held_at_lift = 0, 0, 0, 0, []
    for s in meta["strokes"]:
        k = idx_at(T, s["t"] + 0.004)
        if 1 <= k < len(S) - 1 and not np.isnan(zs[k - 1]) and not np.isnan(zs[k + 1]):
            vz = zs[k + 1] - zs[k - 1]          # z grows toward the low E: a down stroke (low -> high strings) has vz < 0
            n_dir += 1; ok_dir += (vz < 0) == (s["dir"].upper() == "D")
        kk = range(idx_at(T, s["t"] - 0.02), idx_at(T, s["t"] + s["offsets"][-1] + 0.05) + 1)
        n_glow += 1
        ok_glow += all(any((S[j]["glow"] or [0] * 6)[st] > 0.3 for j in kk) for st in s["strings"])
    for c in meta["chords"][1:]:
        # the stroke right before each chord change: is the fretting hand still holding the old chord? (lesson: lift off)
        prev = [s for s in meta["strokes"] if s["t"] < c["t0"] - 0.01]
        if prev:
            k = idx_at(T, prev[-1]["t"])
            held_at_lift.append(sum(tc[2] <= PRESS_MM for tc in S[k]["ltc"]))
    m["execution_3d"] = {"pick_moves_in_stroke_direction_pct": pct(ok_dir, n_dir), "stroke_strings_light_up_pct": pct(ok_glow, n_glow),
                         "fingers_still_pressing_on_last_stroke_before_change (lesson: 0, lifted)": held_at_lift}

    # contact sheet: chord downbeats of the first reference cycle vs ours
    rows = []
    real_mp4 = os.path.join(CACHE, "knockin_andy.mp4")
    ours_mp4 = os.path.join(OUT, "knockin", "knockin.mp4")
    if cyc_starts:
        c0 = cyc_starts[0]
        for h in range(8):
            tr = beats[c0 + 4 * h] + 0.12
            to = 0.2 + h * fast_bar + 0.06
            k = idx_at(T, to)
            st = S[k]
            sl = 8 * h
            lab = (f"half bar {h + 1}: lesson {LESSON['progression_halfbars'][h]}\nours chord {st['chord']}  stroke {od.get(sl, '-')}"
                   f"\nreal (audio) {hb_ref[h] if h < len(hb_ref) else '?'}  stroke {ref_cycles[0].get(sl, {}).get('dir') if ref_cycles else '?'}"
                   f"\nours t={to:.2f}s  real t={LESSON['start'] + tr:.2f}s")
            rows.append((lab, grab(ours_mp4, to - d["t0"]) if os.path.exists(ours_mp4) else None, grab(real_mp4, LESSON["start"] + tr, 480)))
    m["sheet"] = sheet(rows, os.path.join(RES, "knockin_sheet.png"), "Knockin' on Heaven's Door: chord downbeats")
    return m


def main():
    want = sys.argv[1:] or ["choo-lo", "rabba", "knockin"]
    summ_p = os.path.join(RES, "summary.json")
    summary = json.load(open(summ_p)) if os.path.exists(summ_p) else {}
    for case in want:
        if case == "choo-lo":
            m = tab_case("choo-lo", "D1I84oQuGpM")
        elif case == "rabba":
            m = tab_case("rabba", "3efgaqeI9qk")
        elif case == "knockin":
            m = knockin_case()
        else:
            print("unknown case", case); continue
        if m is None:
            continue
        json.dump(m, open(os.path.join(RES, f"{case}_metrics.json"), "w"), indent=1)
        summary[case] = {k: v for k, v in m.items() if not k.startswith("_")}
        print(json.dumps({k: v for k, v in m.items() if not k.startswith("_") and k not in ("misses",)}, indent=1)[:6000])
    json.dump(summary, open(summ_p, "w"), indent=1)


if __name__ == "__main__":
    main()
