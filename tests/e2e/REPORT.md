# End-to-end verification: our 3D playback vs real performances

Run with `tests/e2e/run_all.sh` (about 20 min; `--refs` re-downloads videos and reruns the trackers, `--state` skips frames). Results in `tests/e2e/results/` (`*_sheet.png`: ours on top, real video below, same timestamps; `*_metrics.json`, `summary.json`).

## Metrics

| case | fingertip on tab string+fret at onset | lands before pluck (median) | released >50 ms before next onset | picking hand on right string | hand position vs real video (index tip within ±1 fret) | finger choice vs real | tracker vs tab (reference quality) |
|---|---|---|---|---|---|---|---|
| Choo Lo (80 fretted notes) | **98.8 %** (100 % within 50 ms) | −53 ms | 7 of 7, median 126 ms early | 100 % (160 plucks) | **59.9 %**, median 0.84 fret off; ours 0.23 s later | **13 %** (player uses middle, we use index on 28 of 38) | 47.5 % exact / 57.5 % ±1 |
| Rabba (75 fretted notes) | **97.3 %** | −55 ms | **97 %** of 67, median 100 ms early | 98.6 % (73 strokes) | 26 % (not conclusive: tracker has a constant +2-fret offset) | only 5 notes | 6.3 % exact / 12.7 % ±1 |

- **Techniques:** HUD names them 100 % of the time; the motion is visible for every slide, hammer-on, bend, vibrato and pull-off. Hammer-ons land +11 ms after the tab onset. Slides reach the new fret 39–137 ms early (median 65 ms).
- **Knockin' on Heaven's Door (chord mode):** setup `G D Am Am G D C C`, 1 bar per chord, 112 bpm (lesson is 56.2 bpm; chord mode allows only whole bars, so tempo was doubled), pattern `ballad`, pick.
  - Chords per half bar: 100 % vs the lesson's stated progression and vs chords detected from its audio.
  - Strokes vs the lesson's spoken pattern: F1 0.82, direction 100 % (G/D bars exact; Am/C bars F1 0.67).
  - Strokes vs the video: F1 0.76, direction 92 % (the lesson pattern itself matches the video at F1 0.88 / 94 %).
  - 3D hand: pick moves in the stroke direction 100 %, right strings light 100 %; up strokes hit 4 strings where the lesson says 3, and 2–3 fingers stay pressed on the last strum before every chord change, where the lesson lifts off.

## Method
- **Ours:** virtual page clock (fake `Audio`, replaced `requestAnimationFrame` and `performance.now`); each 1/15 s step runs 4 frames at dt 1/60 s, so results don't depend on rendering speed. State logged at 60 Hz (fingertip string/fret/height from `tipG` with `guitar.js` geometry, targets, HUD, chord name, pick, glow). Frames to mp4 at 15 fps.
- **References:** tabs `data/score-*.json`; Shorts via unmodified `handtrack/track.py` (per-frame output from `handtrack_dump.py`); Knockin' lesson (YouTube pWIL4N6QZ-Y, play-through 449–529 s): chords from audio chroma per half bar, strokes from audio onsets on the 16th grid, stroke direction from strumming-hand optical flow (`strum_track.py`), pattern from the lesson's own captions (G/D "down down down up", Am/C "down down down up up down up up down up").

## Top mismatches and fixes, ranked
1. **Notes cut short: fingers lift about 100–126 ms before the next onset** (97 % of Rabba notes). `js/timeline.js` `sample()` switches at `next.t - next.lead` (lead up to 110 ms); the legato-hold branch is skipped when the next note is on the same string or after `pj.t + pj.dur + 0.04`. Fix: keep the previous note pressed until about `next.t - 0.012` unless the same finger must travel; separate the new finger's approach lead from the old finger's release.
2. **Slides arrive 39–137 ms early.** `timeline.js` `press()` glides over `[e.t + 0.3d, e.t + 0.8d]` plus spring lag. Fix: glide over `[next.t - min(0.12, 0.6·gap), next.t - 0.01]`.
3. **Choo Lo finger choice:** the player slides G9→11→13→14→16 with the middle finger (27 of 28 identified notes); we use the index, which explains most of the 0.84-fret hand-position offset. Fix in `js/fingering.js`: penalise re-fingering the same string/fret within ~1 s and let a slide chain keep the finger that starts it.
4. **Chord mode can't express this lesson** (no half-bar chords, one pattern for every chord). Fix: per-chord beats (`G:2 D:2 Am:4`) and per-chord pattern override in `chords-mode.js` `parseChords`, `chordsong.js` `buildChordSong`, `chords/strum_model.js` `progressionTimeline`; add the 16th pattern `D - - - D - D U - U D U - U D U` to `chords/strum_patterns.json`. The app suggests `down8` for this song, which is far off.
5. **Strum details:** limit up strokes to e/B/G in `strum_model.js`; release the chord shape at the last stroke before a change and strum open strings (`FRET_LEAD` 0.15 s in `chordsong.js` is shorter than the 0.27 s from the last up-strum to the change at this tempo).
6. **Position shifts about 0.23 s later than the player** (Choo Lo, correlation 0.74; partly the tracker's smoothing). Try raising `leadH` in `prepareTimeline` toward 0.3 s for shifts over 2 frets.
7. **Pinky reach misses:** Rabba B10 at 3.91 s and e9 at 27.32 s (tip 2.0–2.2 mm above the string, about 0.2 string off). Refit the pinky chain in `hand.js` or add a pinky stretch cost in `fingering.js`.
8. **Tab data bug:** `score-rabba.json` has same-string note pairs with identical `t` at 11.436 s (q 34.5/35) and 19.598 s (q 58.5/59); excluded from scoring.

## Limits
- Timing is the app's logic on a virtual clock, not real playback; audio latency and the YouTube-sync path aren't tested.
- Rendering is swiftshader at 540×960, so sheets show pose, not final visual quality.
- "On the note" = IK tip on the rounded string, fret = `ceil(fret)`, tip ≤ 1.5 mm above the string.
- Video references are only as good as the trackers: Choo Lo's is usable, Rabba's isn't (Rabba numbers rest on the tab).
- One chord song only (Wonderwall needs a capo, which chord mode lacks). No Ultimate Guitar or Songsterr data used.
