"""On-screen tab reader: read the tab shown in a lesson/cover video and align it to the audio.

usage: python tools/tabocr/read_tabs.py VIDEO_URL_OR_FILE -o score.json [--start S --end S]
           [--audio OTHER_RECORDING] [--capo N] [--tuning E2,A2,D3,G3,B3,E4] [--fps 2] [--debug DIR]

Output: site score JSON {bpm, grid, notes:[{t,q,d,n:[[string,fret]],ham|pull|slide|bend|rel|vib, sec}]}
string 0 = high e. t = onset time in the recording, q = beat position.
--audio aligns the tab read from the video to a different recording of the same arrangement
(the page-on-screen time constraint is then not used).
"""
import os, sys, json, copy, argparse, pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import video, detect, ocr, tabseq, align, sanity  # noqa: E402

NOTE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def note_midi(s):
    s = s.strip(); n = NOTE[s[0].upper()]; i = 1
    if i < len(s) and s[i] in "#b":
        n += 1 if s[i] == "#" else -1; i += 1
    return 12 * (int(s[i:]) + 1) + n


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def read_video_tabs(path, start, end, fps, debug=None):
    cache_key = None
    if debug:
        os.makedirs(debug, exist_ok=True)
        cache_key = os.path.join(debug, "pages.pkl")
    if cache_key and os.path.exists(cache_key):
        pages, intervals = pickle.load(open(cache_key, "rb"))
        log(f"loaded {len(pages)} pages from {cache_key}")
    else:
        log("tracking tab pages ...")
        pages, intervals = detect.track_pages(video.frames(path, fps=fps, start=start, end=end), log=log)
        for p in pages:      # keep only the median frame (small cache, same pixels the OCR uses)
            p["crops"] = [ocr.median_frame(p["crops"])]
        if cache_key:
            pickle.dump((pages, intervals), open(cache_key, "wb"))
    log(f"{len(pages)} distinct tab pages, {len(intervals)} visible intervals")
    systems = ocr.read_pages(pages, log=log, debug=debug)
    return pages, intervals, systems


def page_event_lists(systems):
    out = []
    for sy in systems:
        evs, nb = tabseq.page_events(sy)
        # rhythm prior: x position within the bar (bar = 4 beats); the aligner treats it as a soft guide
        for e in evs:
            e["q"] = e["bar"] * 4 + 4 * e["pos"]
        out.append((evs, nb))
    return out


def prep(evs, capo, tuning):
    return tabseq.finalize(copy.deepcopy(evs), capo, tuning)


def interp_times(events, assign, aud, T):
    """Times for every event: matched onsets, others interpolated on q."""
    qs = np.array([e["q"] for e in events], float)
    m = [i for i, j in enumerate(assign) if j >= 0]
    t = np.zeros(len(events))
    if not m:
        return None
    mq = qs[m]; mt = aud.times[[assign[i] for i in m]]
    for i in range(len(events)):
        if assign[i] >= 0:
            t[i] = aud.times[assign[i]]
        elif mq[0] <= qs[i] <= mq[-1] and len(m) > 1:
            t[i] = np.interp(qs[i], mq, mt)
        else:
            ref = 0 if qs[i] < mq[0] else -1
            t[i] = mt[ref] + (qs[i] - mq[ref]) * (T or 0.5)
    return t


def passes_for_interval(evs, aud, lo, hi, capo, tuning, min_frac, min_sal, vis=None, first=None, depth=0):
    """Align one page's events within [lo, hi]; then look for repeated plays (lessons often play a page
    slowly, then at speed) in what is left of the time the page is visible (vis)."""
    ev = prep(evs, capo, tuning)
    if not ev or hi - lo < 0.3:
        return []
    assign, score, T = align.align_events(ev, aud, lo, hi)
    frac, sal = align.quality(ev, assign, aud)
    if first is not None:      # a repeat must look as convincing as the main play
        min_frac = max(min_frac, 0.8); min_sal = max(min_sal, 0.8 * first["sal"])
    if frac < min_frac or sal < min_sal:
        return []
    t = interp_times(ev, assign, aud, T)
    mt = aud.times[[j for j in assign if j >= 0]]
    t = np.clip(t, min(lo, mt.min()), max(hi, mt.max()))
    for i in range(1, len(t)):
        t[i] = max(t[i], t[i - 1] + 0.01)
    sanity.annotate_audio(ev, assign, aud)
    res = [dict(events=ev, assign=assign, t=t, T=T, frac=frac, sal=sal)]
    if depth < 4 and vis is not None:
        a, b = mt.min(), mt.max()
        dur = b - a
        top = first or res[0]
        L, R = max(lo, vis[0] - 0.5), min(hi, vis[1] + 0.5)
        if a - L > 0.6 * dur + 0.5:
            res += passes_for_interval(evs, aud, L, a - 0.05, capo, tuning, min_frac, min_sal, vis, top, depth + 1)
        if R - b > 0.6 * dur + 0.5:
            res += passes_for_interval(evs, aud, b + 0.05, R, capo, tuning, min_frac, min_sal, vis, top, depth + 1)
    return res


def build_score(passes, label):
    """passes: list of dict(events, t, T, page). Returns site score dict."""
    passes = sorted(passes, key=lambda p: p["t"].min())
    notes, grid_pts = [], []
    qbase = 0
    prev_end_t, prev_T = None, None
    for p in passes:
        ev, t, T = p["events"], p["t"], p["T"] or 0.5
        # keep strictly increasing times
        for i in range(1, len(t)):
            if t[i] <= t[i - 1]:
                t[i] = t[i - 1] + 0.01
        if prev_end_t is not None:
            gap_beats = max(0.0, (t[0] - prev_end_t) / T)
            qbase = int(np.ceil((qbase_end + gap_beats - 0.5) / 4.0)) * 4
            qbase = max(qbase, int(np.ceil(qbase_end / 4.0)) * 4)
        bars = sorted(set(e["bar"] for e in ev))
        b0 = bars[0]
        # bar start times from each bar's first event
        A = {}
        for b in bars:
            k = next(i for i, e in enumerate(ev) if e["bar"] == b)
            A[b] = t[k] - (ev[k]["q"] - b * 4) * T
        Al = [A[b] for b in bars]
        for i, b in enumerate(bars):
            if i and Al[i] <= Al[i - 1] + 0.5 * T:
                Al[i] = Al[i - 1] + 4 * T
        A = dict(zip(bars, Al))
        last_q = -1
        for i, e in enumerate(ev):
            b = e["bar"]; a = A[b]
            nb = bars.index(b) + 1
            z = A[bars[nb]] if nb < len(bars) else a + 4 * T
            loc = 4 * (t[i] - a) / max(1e-3, z - a)
            loc = float(np.clip(round(loc * 4) / 4, 0, 3.75))
            q = qbase + (b - b0) * 4 + loc
            if q <= last_q:
                q = last_q + 0.25
            last_q = q
            nt = {"t": round(float(t[i]), 3), "q": q,
                  "n": [[n["s"], n["fret"]] for n in sorted(e["notes"], key=lambda n: n["s"])]}
            for n in e["notes"]:
                for k in ("slide", "ham", "pull", "bend", "rel", "vib", "pm"):
                    if n.get(k):
                        nt[k] = n[k] if k == "bend" else 1
            ns = sorted(e["notes"], key=lambda n: n["s"])
            nt["conf"] = round(float(min(n.get("conf", 1.0) for n in ns)), 2)
            nt["cn"] = [round(float(n.get("conf", 1.0)), 2) for n in ns]
            fl = sorted({f for n in ns for f in n.get("flags", [])})
            if fl:
                nt["flags"] = fl
            nt["sec"] = f"{label} p{p['page'] + 1}||3"
            notes.append(nt)
        for i, b in enumerate(bars):
            z = A[bars[i + 1]] if i + 1 < len(bars) else A[b] + 4 * T
            for k in range(4):
                grid_pts.append((qbase + (b - b0) * 4 + k, A[b] + k * (z - A[b]) / 4))
        qbase_end = max(last_q + 0.25, qbase + (bars[-1] - b0 + 1) * 4)
        prev_end_t = A[bars[-1]] + 4 * T
    for i, n in enumerate(notes):
        nq = notes[i + 1]["q"] if i + 1 < len(notes) else n["q"] + 1
        n["d"] = float(min(16, (nq - n["q"]) * 4))
    grid_pts.sort()
    gq = np.array([g[0] for g in grid_pts]); gt = np.array([g[1] for g in grid_pts])
    nb = int(np.ceil((notes[-1]["q"] + 1) / 4)) * 4 + 2 if notes else 0
    grid = []
    for k in range(nb):
        if len(gq) and gq.min() <= k <= gq.max():
            grid.append(float(np.interp(k, gq, gt)))
        elif len(gq):
            T = np.median(np.diff(gt)) if len(gt) > 1 else 0.5
            ref = 0 if k < gq.min() else -1
            grid.append(float(gt[ref] + (k - gq[ref]) * T))
    for i in range(1, len(grid)):
        if grid[i] <= grid[i - 1]:
            grid[i] = grid[i - 1] + 0.01
    beat = np.median(np.diff(gt)) if len(gt) > 1 else 0.5
    return {"bpm": round(60 / beat, 1), "grid": [round(g, 3) for g in grid], "notes": notes}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--start", type=float); ap.add_argument("--end", type=float)
    ap.add_argument("--audio", help="align to this recording instead of the video's own audio")
    ap.add_argument("--capo", type=int, default=0)
    ap.add_argument("--tuning", default="E4,B3,G3,D3,A2,E2", help="high to low, or low to high (auto)")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--lead", type=float, default=2.5, help="seconds a page may be played before/after it shows")
    ap.add_argument("--min-frac", type=float, default=0.6); ap.add_argument("--min-sal", type=float, default=0.3)
    ap.add_argument("--label", default="Tab")
    ap.add_argument("--debug", help="dir for page images / cached pages / read tab text")
    a = ap.parse_args()

    tun = [note_midi(x) for x in a.tuning.split(",")]
    if tun[0] < tun[-1]:
        tun = tun[::-1]
    path = video.fetch(a.src)
    log("video:", path)
    pages, intervals, systems = read_video_tabs(path, a.start, a.end, a.fps, a.debug)
    pev = page_event_lists(systems)
    sanity.check_pages(pev, tun, a.capo)
    if a.debug:
        from PIL import Image
        with open(os.path.join(a.debug, "tab.txt"), "w", encoding="utf-8") as f:
            for k, (evs, nb) in enumerate(pev):
                f.write(f"page {k} ({nb} bars): {tabseq.to_text(evs)}\n")
            f.write("intervals: " + " ".join(f"p{p}@{t0:.1f}-{t1:.1f}" for p, t0, t1 in intervals) + "\n")
        for k, p in enumerate(pages):
            y0, y1, x0, x1 = detect.region(p["staves"], p["shape"])
            Image.fromarray(ocr.median_frame(p["crops"])[y0:y1, x0:x1]).save(os.path.join(a.debug, f"page{k:02d}.png"))
    for k, (evs, nb) in enumerate(pev):
        log(f"page {k}: {tabseq.to_text(evs)}")

    aud_src = a.audio or path
    y, sr = video.audio(aud_src)
    aud = align.Audio(y, sr)
    log(f"audio {len(y)/sr:.1f}s, {len(aud.times)} onsets")
    passes = []
    if a.audio:
        # one global alignment of the play order (intervals) to the other recording
        seq, qoff = [], 0
        for pid, t0, t1 in intervals:
            evs, nb = pev[pid]
            for e in evs:
                e2 = dict(e); e2["q"] = e["q"] + qoff; e2["bar"] = e["bar"] + qoff // 4; e2["page"] = pid
                seq.append(e2)
            qoff += 4 * nb
        ev = prep(seq, a.capo, tun)
        assign, score, T = align.align_events(ev, aud)
        frac, sal = align.quality(ev, assign, aud)
        log(f"global alignment: matched {frac:.0%}, pitch evidence {sal:.2f}, beat {T}")
        t = interp_times(ev, assign, aud, T)
        sanity.annotate_audio(ev, assign, aud)
        passes.append(dict(events=ev, assign=assign, t=t, T=T, page=intervals[0][0] if intervals else 0))
    else:
        prev_end = a.start or 0.0
        for k, (pid, t0, t1) in enumerate(intervals):
            evs, nb = pev[pid]
            if not evs:
                continue
            nxt = intervals[k + 1][1] if k + 1 < len(intervals) else t1 + a.lead
            lo = max(t0 - a.lead, prev_end); hi = min(t1 + a.lead, nxt + a.lead)
            ps = passes_for_interval(evs, aud, lo, hi, a.capo, tun, a.min_frac, a.min_sal, vis=(t0, t1))
            for p in ps:
                p["page"] = pid
                log(f"  page {pid} [{t0:.1f}-{t1:.1f}s]: played {p['t'].min():.2f}-{p['t'].max():.2f}s, "
                    f"matched {p['frac']:.0%}, pitch {p['sal']:.2f}")
            if not ps:
                log(f"  page {pid} [{t0:.1f}-{t1:.1f}s]: no confident play found")
            passes += ps
            if ps:
                prev_end = max(aud.times[max(p["assign"])] for p in ps) + 0.03
    if not passes:
        log("nothing aligned"); sys.exit(1)
    score = build_score(passes, a.label)
    n_on = sum(sum(1 for j in p["assign"] if j >= 0) for p in passes)
    n_all = sum(len(p["events"]) for p in passes)
    log(f"notes {len(score['notes'])}, on detected onsets {n_on}/{n_all} ({100*n_on/max(1,n_all):.1f}%), bpm {score['bpm']}")
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(score, f)
    log("wrote", a.out)


if __name__ == "__main__":
    main()
