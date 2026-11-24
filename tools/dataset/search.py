"""Find candidate tab-on-screen guitar videos via yt-dlp ytsearch. Writes cache/candidates.jsonl (metadata only).
usage: python -I search.py [--per 12] [--add]   (--add: round-2 queries, appended)"""
import os, sys, json, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
YTDLP = r"C:/Users/sharv/anaconda3/Scripts/yt-dlp.exe"
Q = [  # (query, style tag)
 ("guitar pro tab playthrough", "electric"), ("guitar tab playthrough with tabs on screen", "electric"),
 ("fingerstyle guitar tab", "fingerstyle"), ("fingerstyle tutorial with tabs", "fingerstyle"),
 ("acoustic guitar riff tab", "acoustic"), ("easy guitar riff tab lesson", "riff"),
 ("guitar solo tab playthrough", "solo"), ("metal riff guitar tab", "distorted"),
 ("blues guitar lick tab", "blues"), ("classical guitar tab", "classical"),
 ("rock riff guitar tab", "distorted"), ("guitar arpeggio chords tab lesson", "chords"),
 ("Smoke on the Water guitar tab", "riff"), ("Seven Nation Army guitar tab", "riff"),
 ("Stairway to Heaven intro guitar tab", "fingerstyle"), ("Nothing Else Matters intro guitar tab", "fingerstyle"),
 ("Wish You Were Here intro guitar tab", "acoustic"), ("Sweet Child O Mine intro guitar tab", "solo"),
 ("Hotel California solo guitar tab", "solo"), ("Romanza guitar tab", "classical"),
 ("Fur Elise guitar tab", "fingerstyle"), ("Canon in D fingerstyle tab", "fingerstyle"),
 ("Enter Sandman riff guitar tab", "distorted"), ("Come As You Are guitar tab", "riff"),
 ("Blackbird guitar tab", "fingerstyle"), ("Tears in Heaven guitar tab", "fingerstyle"),
 ("Snow Hey Oh guitar tab", "electric"), ("Under the Bridge intro guitar tab", "electric"),
 ("bollywood song guitar tab", "acoustic"), ("indie guitar riff tab", "electric"),
 ("songsterr playthrough", "electric"), ("guitar tab cover with tabs", "electric"),
]
EXTRA = [  # round 2: channels/formats whose tabs passed the gate in round 1 (clean Guitar Pro style rendering)
 ("yunus guitarist fingerstyle tutorial TAB", "fingerstyle"), ("Kevin Balke guitar tab", "riff"),
 ("Kevin Balke guitar lesson tab", "electric"), ("fingerstyle guitar tutorial + TAB & Lyrics", "fingerstyle"),
 ("guitar cover with tabs on screen rock", "distorted"), ("acoustic guitar lesson tab on screen", "acoustic"),
 ("guitar solo lesson with tab on screen", "solo"), ("blues guitar solo tab on screen", "blues"),
]
if "--add" in sys.argv:
    Q = EXTRA
per = int(sys.argv[sys.argv.index("--per") + 1]) if "--per" in sys.argv else 12
os.makedirs(os.path.join(HERE, "cache"), exist_ok=True)
seen, out = set(), []
CF = os.path.join(HERE, "cache", "candidates.jsonl")
if "--add" in sys.argv and os.path.exists(CF):
    out = [json.loads(l) for l in open(CF, encoding="utf-8")]; seen = {o["id"] for o in out}
for q, tag in Q:
    r = subprocess.run([YTDLP, "--flat-playlist", "-J", f"ytsearch{per}:{q}"], capture_output=True, text=True, encoding="utf-8")
    try:
        ents = json.loads(r.stdout).get("entries", [])
    except Exception:
        continue
    for e in ents:
        vid, d = e.get("id"), e.get("duration") or 0
        if not vid or vid in seen or not (60 <= d <= 360):
            continue
        seen.add(vid)
        out.append(dict(id=vid, title=e.get("title"), channel=e.get("channel") or e.get("uploader"),
                        url=f"https://www.youtube.com/watch?v={vid}", duration=d, query=q, style=tag))
with open(os.path.join(HERE, "cache", "candidates.jsonl"), "w", encoding="utf-8") as f:
    for o in out:
        f.write(json.dumps(o, ensure_ascii=False) + "\n")
print(len(out), "candidates")
