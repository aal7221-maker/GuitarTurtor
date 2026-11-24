"""Build the (audio, exact tab) lesson dataset: download candidates, run tools/tabocr/read_tabs.py, gate, write manifest.
PRIVATE RESEARCH ONLY: media stays in cache/ (gitignored); labels/ holds derived tab JSON + video ids only.

usage: python -I build.py run [--jobs 3] [--limit N] [--ids a,b] [--after K (skip first K candidates)]   # download + read + measure (resumable)
       python -I build.py manifest                                 # gate all results -> manifest.jsonl + labels/
Per item: cache/<id>.mp4, cache/audio/<id>.m4a, debug/<id>/{log.txt,score.json,tab.txt,page*.png,metrics.json}
"""
import os, re, sys, json, glob, subprocess, statistics as st
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PY = sys.executable
READER = os.path.join(ROOT, "tools", "tabocr", "read_tabs.py")
YTDLP = r"C:/Users/sharv/anaconda3/Scripts/yt-dlp.exe"
FFDIR = r"C:/Users/sharv/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-9.1-full_build/bin"
CACHE, DEBUG, LABELS = (os.path.join(HERE, d) for d in ("cache", "debug", "labels"))
STD = ["E4", "B3", "G3", "D3", "A2", "E2"]

# quality gate. Item-level (whole video): onset %, share of pages aligned. Then pages that look misread are dropped
# from the label (time spans listed in "exclude" so training masks them): pitch-vs-audio agreement from an independent
# transcriber (verify.py: basic-pitch posterior at each tab pitch, debug/<id>/bp.json) below PAGE_GATE, or low digit
# conf. The remaining label must pass the note-level gate.
# Calibration: A_HYDoMi45g (spot-checked exact) agrees 0.99, Choo Lo reference 0.97, Locked Away reference 0.75
# (band mix), 1WIMZScl0lk (font misread, by eye) 0.80, PquIcZ0jL_M (10 read as 1) 0.59. The reader's own per-note
# 'audio' flag is too noisy to gate on (an exact riff page had 5/13 flagged) and is only reported.
GATE = dict(min_onset=80.0, min_page_frac=0.6, min_notes=40, min_kept=0.5, min_conf_mean=0.88, max_flag_rate=0.30,
            min_agree=0.90, min_cover=0.25)
PAGE_GATE = dict(min_agree=0.85, min_conf=0.85, max_flags=0.5)

SKIP_TITLE = re.compile(r"how to add|tutorial on adding|capcut|top \d+|\bvs\b|reaction|review|shorts?\b", re.I)


def tuning_from_title(t):
    t = t.lower(); capo = 0
    m = re.search(r"capo\s*(?:on\s*)?(\d{1,2})", t)
    if m:
        capo = int(m.group(1))
    tun = list(STD); name = "standard"
    if re.search(r"half[ -]?step down|e\s?b tuning|e flat|eb standard|\beb\b", t):
        tun = ["D#4", "A#3", "F#3", "C#3", "G#2", "D#2"]; name = "Eb"
    if re.search(r"drop\s*d\b", t):
        tun[5] = "D2"; name = "dropD"
    if re.search(r"drop\s*c#|drop\s*db", t):
        tun = ["D#4", "A#3", "F#3", "C#3", "G#2", "C#2"]; name = "dropC#"
    if re.search(r"drop\s*c\b", t):
        tun = ["D4", "A3", "F3", "C3", "G2", "C2"]; name = "dropC"
    if re.search(r"d standard|whole step down", t):
        tun = ["D4", "A3", "F3", "C3", "G2", "D2"]; name = "D"
    return ",".join(tun), name, capo


def sh(cmd, log=None, timeout=None):
    with open(log, "w", encoding="utf-8") if log else open(os.devnull, "w") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout).returncode


def download(vid):
    hit = [p for p in glob.glob(os.path.join(CACHE, vid + ".*")) if p.endswith(".mp4")]
    if hit:
        return hit[0]
    out = os.path.join(CACHE, "%(id)s.%(ext)s")
    sh([YTDLP, "-q", "--no-playlist", "-f", "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080]",
        "--merge-output-format", "mp4", "--ffmpeg-location", FFDIR, "-o", out,
        f"https://www.youtube.com/watch?v={vid}"], timeout=600)
    p = os.path.join(CACHE, vid + ".mp4")
    return p if os.path.exists(p) else None


def extract_audio(vid, mp4):
    os.makedirs(os.path.join(CACHE, "audio"), exist_ok=True)
    a = os.path.join(CACHE, "audio", vid + ".m4a")
    if not os.path.exists(a):
        sh([os.path.join(FFDIR, "ffmpeg.exe"), "-v", "error", "-y", "-i", mp4, "-vn", "-c:a", "copy", a])
    return a if os.path.exists(a) else None


def parse_log(txt):
    m = {}
    g = re.search(r"(\d+) distinct tab pages, (\d+) visible intervals", txt)
    m["pages"], m["intervals"] = (int(g.group(1)), int(g.group(2))) if g else (0, 0)
    played = re.findall(r"page (\d+) \[[\d.]+-[\d.]+s\]: played ([\d.]+)-([\d.]+)s, matched (\d+)%, pitch ([\d.]+)", txt)
    nope = re.findall(r"page (\d+) \[[\d.]+-[\d.]+s\]: no confident play found", txt)
    m["passes"] = len(played); m["pages_unplayed"] = len(nope)
    ivs = re.search(r"intervals: (.*)", txt)
    aligned = {p[0] for p in played}
    m["pages_aligned"] = len(aligned)
    m["page_frac"] = round(len(aligned) / max(1, len(aligned | {n for n in nope})), 3)
    m["pitch_mean"] = round(st.mean(float(p[4]) for p in played), 3) if played else 0.0
    m["matched_mean"] = round(st.mean(int(p[3]) for p in played), 1) if played else 0.0
    g = re.search(r"on detected onsets (\d+)/(\d+) \(([\d.]+)%\)", txt)
    m["onset_pct"] = float(g.group(3)) if g else 0.0
    g = re.search(r"audio ([\d.]+)s, (\d+) onsets", txt)
    m["audio_s"] = float(g.group(1)) if g else 0.0
    return m


def score_stats(score, dur):
    ns = score["notes"]
    m = {"n_notes": len(ns), "n_pitches": sum(len(n["n"]) for n in ns), "bpm": score.get("bpm")}
    if not ns:
        return m
    conf = [c for n in ns for c in n.get("cn", [n.get("conf", 1.0)])]
    flags = {}
    for n in ns:
        for f in n.get("flags", []):
            flags[f] = flags.get(f, 0) + 1
    m.update(conf_mean=round(st.mean(conf), 3), conf_min=round(min(conf), 3),
             conf_p10=round(sorted(conf)[len(conf) // 10], 3),
             conf_ge07=round(sum(c >= 0.7 for c in conf) / len(conf), 3),
             flag_rate=round(sum(1 for n in ns if n.get("flags")) / len(ns), 3),
             audio_flag=round(flags.get("audio", 0) / len(ns), 3), flags=flags,
             t0=ns[0]["t"], t1=ns[-1]["t"], cover=round((ns[-1]["t"] - ns[0]["t"]) / max(1, dur), 3),
             chords=round(sum(len(n["n"]) > 1 for n in ns) / len(ns), 3),
             techniques={k: sum(1 for n in ns if n.get(k)) for k in ("ham", "pull", "slide", "bend", "vib", "pm")},
             max_fret=max(f for n in ns for s, f in n["n"]))
    return m


def gate(m):
    r = []
    if m.get("onset_pct", 0) < GATE["min_onset"]: r.append(f"onset%<{GATE['min_onset']:.0f}")
    if m.get("page_frac", 0) < GATE["min_page_frac"]: r.append("pages unaligned")
    if m.get("bp_agree") is None: r.append("not verified")
    if m.get("n_notes", 0) < GATE["min_notes"]: r.append(f"notes<{GATE['min_notes']}")
    if m.get("kept_frac", 1) < GATE["min_kept"]: r.append("too many misread pages")
    if m.get("conf_mean", 0) < GATE["min_conf_mean"]: r.append("low digit conf")
    if m.get("flag_rate", 1) > GATE["max_flag_rate"]: r.append("many sanity flags")
    if m.get("bp_agree") is not None and (m.get("bp_agree_kept") or 0) < GATE["min_agree"]: r.append("pitch!=audio")
    if m.get("cover", 0) < GATE["min_cover"]: r.append("tab covers little of video")
    return r


def trim_pages(score, bp=None):
    """Drop page groups (note 'sec') that look misread; add per-note audio evidence 'ev' (basic-pitch).
    Returns (trimmed score, dropped page labels, exclude spans, agreement of kept notes)."""
    ns = [dict(n) for n in score["notes"]]
    if bp:
        for n, e in zip(ns, bp["per_note"]):
            n["ev"] = e
    groups = {}
    for n in ns:
        groups.setdefault(n["sec"].split("||")[0], []).append(n)
    bad = set()
    for p, g in groups.items():
        conf = st.mean(c for n in g for c in n.get("cn", [n.get("conf", 1.0)]))
        fl = sum(bool(n.get("flags")) for n in g) / len(g)
        ag = bp["pages"].get(p, 0) if bp else 0
        if ag < PAGE_GATE["min_agree"] or conf < PAGE_GATE["min_conf"] or fl > PAGE_GATE["max_flags"]:
            bad.add(p)
    keep = [n for n in ns if n["sec"].split("||")[0] not in bad]
    spans, cur = [], None
    for n in ns:
        if n["sec"].split("||")[0] in bad:
            cur = [n["t"] - 0.3, n["t"] + 0.3] if cur is None else [cur[0], n["t"] + 0.3]
        elif cur is not None:
            spans.append([round(cur[0], 2), round(cur[1], 2)]); cur = None
    if cur is not None:
        spans.append([round(cur[0], 2), round(cur[1], 2)])
    ev = [x for n in keep for x in n.get("ev", [])]
    agree = round(sum(x >= bp["thr"] for x in ev) / len(ev), 3) if bp and ev else None
    out = dict(score); out["notes"] = keep
    return out, sorted(bad), spans, agree


def process(c, timeout=1800):
    vid = c["id"]; d = os.path.join(DEBUG, vid); os.makedirs(d, exist_ok=True)
    mf = os.path.join(d, "metrics.json")
    if os.path.exists(mf):
        return json.load(open(mf))
    m = dict(c)
    tun, tname, capo = tuning_from_title(c["title"] or "")
    m.update(tuning=tun, tuning_name=tname, capo=capo)
    mp4 = download(vid)
    if not mp4:
        m["error"] = "download failed"
    else:
        m["audio_file"] = os.path.relpath(extract_audio(vid, mp4) or "", HERE)
        out = os.path.join(d, "score.json")
        try:
            rc = sh([PY, "-I", READER, mp4, "-o", out, "--debug", d, "--tuning", tun, "--capo", str(capo),
                     "--label", "Tab"], log=os.path.join(d, "log.txt"), timeout=timeout)
        except subprocess.TimeoutExpired:
            rc = "timeout"
        txt = open(os.path.join(d, "log.txt"), encoding="utf-8", errors="replace").read()
        m.update(parse_log(txt))
        if rc == 0 and os.path.exists(out):
            m.update(score_stats(json.load(open(out)), c.get("duration") or m["audio_s"]))
        else:
            m["error"] = f"reader rc={rc}: " + (txt.strip().splitlines() or [""])[-1][:160]
    json.dump(m, open(mf, "w"), indent=1)
    print(vid, m.get("n_notes"), m.get("onset_pct"), m.get("error", ""), flush=True)
    return m


def candidates(after=0):
    cs = [json.loads(l) for l in open(os.path.join(CACHE, "candidates.jsonl"), encoding="utf-8")][after:]
    return [c for c in cs if "tab" in (c["title"] or "").lower() and not SKIP_TITLE.search(c["title"] or "")]


def run(args):
    cs = candidates(int(args[args.index("--after") + 1]) if "--after" in args else 0)
    if "--ids" in args:
        ids = args[args.index("--ids") + 1].split(","); cs = [c for c in cs if c["id"] in ids]
    if "--limit" in args:
        cs = cs[:int(args[args.index("--limit") + 1])]
    by = {}
    for c in cs:
        by.setdefault(c["style"], []).append(c)
    cs = [l[i] for i in range(max(map(len, by.values()))) for l in by.values() if i < len(l)]  # round-robin styles
    jobs = int(args[args.index("--jobs") + 1]) if "--jobs" in args else 3
    print(len(cs), "candidates to process", flush=True)
    with ThreadPoolExecutor(jobs) as ex:
        list(ex.map(process, cs))


PRIORITY = [
    dict(id="choo-lo", title="Choo Lo (priority, hand-corrected)", channel="", url="https://www.youtube.com/watch?v=D1I84oQuGpM", priority=True,
         label=os.path.join(ROOT, "data", "score-choo-lo.json"), audio=os.path.join(ROOT, "audio", "choo-lo.mp3"),
         until=38.7, style="fingerstyle"),
    dict(id="locked-away", title="Locked Away (priority)", channel="", url="https://www.youtube.com/watch?v=UtL5Ogfxqy4",
         priority=True, label=os.path.join(ROOT, "data", "score-locked-away.json"),
         audio=os.path.join(ROOT, "tools", "tabocr", "cache", "UtL5Ogfxqy4.mp4"), style="electric"),
]


def manifest():
    os.makedirs(LABELS, exist_ok=True)
    rows = []
    for p in PRIORITY:
        sc = json.load(open(p["label"]))
        m = {k: v for k, v in p.items() if k not in ("label", "audio")}
        bpf = os.path.join(DEBUG, "_" + p["id"], "bp.json")
        if os.path.exists(bpf):
            m["bp_agree"] = json.load(open(bpf))["agree"]
        m.update(score_stats(sc, sc["notes"][-1]["t"]), tuning=",".join(STD), tuning_name="standard", capo=0,
                 status="pass", reason="priority (reference tab)", label=f"labels/{p['id']}.json",
                 audio_local=os.path.relpath(p["audio"], ROOT).replace("\\", "/"))
        json.dump(sc, open(os.path.join(LABELS, p["id"] + ".json"), "w"))
        rows.append(m)
    for mf in sorted(glob.glob(os.path.join(DEBUG, "*", "metrics.json"))):
        m = json.load(open(mf))
        lab = os.path.join(LABELS, m["id"] + ".json")
        sc = None
        if not m.get("error"):
            full = json.load(open(os.path.join(DEBUG, m["id"], "score.json")))
            bpf = os.path.join(DEBUG, m["id"], "bp.json")
            bp = json.load(open(bpf)) if os.path.exists(bpf) else None
            sc, bad, spans, m["bp_agree_kept"] = trim_pages(full, bp)
            m["bp_agree"] = bp["agree"] if bp else None
            m["n_notes_read"] = len(full["notes"])
            m.update(score_stats(sc, m.get("duration") or m.get("audio_s") or 1))
            m["kept_frac"] = round(len(sc["notes"]) / max(1, len(full["notes"])), 3)
            m["dropped_pages"] = bad; m["exclude"] = spans
        r = [m["error"]] if m.get("error") else gate(m)
        m["status"] = "fail" if r else "pass"; m["reason"] = "; ".join(r) or "ok"
        m["priority"] = False
        if not r:
            sc["source"] = dict(video_id=m["id"], url=m["url"], tuning=m["tuning"], capo=m["capo"],
                                exclude=m["exclude"], dropped_pages=m["dropped_pages"])
            json.dump(sc, open(lab, "w")); m["label"] = f"labels/{m['id']}.json"
            m["audio_local"] = f"tools/dataset/cache/audio/{m['id']}.m4a"
        elif os.path.exists(lab):
            os.remove(lab)
        sp = os.path.join(DEBUG, m["id"], "spotcheck.json")
        if os.path.exists(sp):
            m["spotcheck"] = json.load(open(sp))
        m.pop("audio_file", None); m.pop("query", None)
        rows.append(m)
    with open(os.path.join(HERE, "manifest.jsonl"), "w", encoding="utf-8") as f:
        for m in rows:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    k = [m for m in rows if m["status"] == "pass"]
    print(f"{len(rows)} items, {len(k)} pass ({sum(m['priority'] for m in k)} priority), "
          f"{sum(m.get('n_notes', 0) for m in k)} notes")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    run(sys.argv[2:]) if cmd == "run" else manifest()
