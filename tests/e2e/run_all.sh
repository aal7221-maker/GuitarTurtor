#!/usr/bin/env bash
# E2E accuracy suite: record what the 3D guitar does and compare it with real references.
#   bash tests/e2e/run_all.sh            # re-record our playback (3 cases) + compare (reuses cached reference data)
#   bash tests/e2e/run_all.sh --refs     # also (re)download the reference videos and re-run the trackers on them
#   bash tests/e2e/run_all.sh --state    # state only (no frames/mp4, ~3x faster); sheets then show only the real side
# Outputs: tests/e2e/results/*_metrics.json, *_sheet.png, summary.json   (REPORT.md is written by hand from these)
# Downloads, tracker output and our renders stay in tests/e2e/cache and tests/e2e/out (gitignored; never commit them).
set -e
cd "$(dirname "$0")"
E2E=$(pwd); ROOT=$(cd ../.. && pwd)
PY=${PY:-C:/Users/sharv/anaconda3/python.exe}
HTPY=${HTPY:-D:/sharvin1/Projects/Music/handtrack/.venv/Scripts/python.exe}
YTDLP=${YTDLP:-C:/Users/sharv/anaconda3/Scripts/yt-dlp.exe}
FFDIR=${FFDIR:-C:/Users/sharv/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-9.1-full_build/bin}
REFS=0; NOFR=""
for a in "$@"; do case $a in --refs) REFS=1 ;; --state) NOFR="--no-frames" ;; esac; done
mkdir -p cache out/ht results

dl() {  # id file [format]
  [ -f "cache/$2" ] && [ $REFS = 0 ] && return 0
  "$YTDLP" -f "${3:-bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b}" --merge-output-format mp4 --ffmpeg-location "$FFDIR" -o "cache/$2" "https://www.youtube.com/watch?v=$1"
}
# ---- reference media (private, local only)
dl D1I84oQuGpM choo-lo_short.mp4
dl 3efgaqeI9qk rabba_short.mp4
dl pWIL4N6QZ-Y knockin_andy.mp4 "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720]"
[ -f cache/knockin_play.wav ] || "$FFDIR/ffmpeg.exe" -y -loglevel error -ss 449 -to 529 -i cache/knockin_andy.mp4 -vn -ac 1 -ar 22050 cache/knockin_play.wav

# ---- trackers on the real videos (slow: a few minutes each)
if [ $REFS = 1 ] || [ ! -f out/ht/choo-lo_frames.json ]; then
  "$HTPY" handtrack_dump.py --frames-out out/ht/choo-lo_frames.json -- --video cache/choo-lo_short.mp4 --score "$ROOT/data/score-choo-lo.json" > out/ht/choo-lo.log 2>&1
fi
if [ $REFS = 1 ] || [ ! -f out/ht/rabba_frames.json ]; then
  "$HTPY" handtrack_dump.py --frames-out out/ht/rabba_frames.json -- --video cache/rabba_short.mp4 --score "$ROOT/data/score-rabba.json" --frets 24 --detector hybrid > out/ht/rabba.log 2>&1
fi
if [ $REFS = 1 ] || [ ! -f out/knock_strum_frames.json ]; then
  "$HTPY" strum_track.py cache/knockin_andy.mp4 --start 449 --end 529 --out out/knock_strum_frames.json > out/strum_track.log 2>&1
fi

# ---- our playback (headless Chrome, deterministic 1/15 s steps, 4 animation frames per step)
cd "$ROOT"
node tests/e2e/record.mjs --song choo-lo $NOFR > tests/e2e/out/rec_choo-lo.log 2>&1
node tests/e2e/record.mjs --song rabba $NOFR > tests/e2e/out/rec_rabba.log 2>&1
# Knockin' on Heaven's Door as the lesson plays it: G(2 beats) D(2) Am(4) G(2) D(2) C(4) at ~56 bpm. Chord mode only
# does whole bars per chord, so each half bar is one bar at double tempo; 'ballad' at 2x = the lesson's G/D pattern.
node tests/e2e/record.mjs --chords "G D Am Am G D C C" --pattern ballad --bpm 112 --bars 1 --hand pick --name knockin $NOFR > tests/e2e/out/rec_knockin.log 2>&1

# ---- compare
"$PY" tests/e2e/compare.py choo-lo rabba knockin > tests/e2e/out/compare.log 2>&1
"$PY" - <<'EOF'
import json
s = json.load(open("tests/e2e/results/summary.json"))
for c in ("choo-lo", "rabba"):
    if c in s:
        v, w = s[c]["vs_tab"], s[c].get("vs_video", {})
        print(f"{c:8s} on-note@onset {v['fingertip_on_string_and_fret_at_onset_pct']}%  released>50ms-early {v['release_before_next_onset_s']['over_50ms_pct']}%  "
              f"hand+-1fret vs video {w.get('hand_position_index_tip_within_1_fret_pct')}%  finger agree {w.get('finger_choice_agreement_pct')}%")
if "knockin" in s:
    k = s["knockin"]
    print(f"knockin  chords vs lesson {k['chords']['ours_vs_lesson_pct']}%  strum F1 vs video {k['strum']['ours_vs_video_mean']['onset_f1']}  "
          f"direction {k['strum']['ours_vs_video_mean']['direction_agree_pct']}%")
EOF
