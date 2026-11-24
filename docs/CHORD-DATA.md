# Chord + strum data in song JSON

A score (`data/score-<id>.json`, a browser-made song's `score`, or a songforge score) may carry two optional fields next to
`bpm`, `grid` and `notes`. Both are omitted when detection fails or does not apply. Times are seconds on the score's clock.
For browser recordings, `js/song-audio.js` `shiftScore` moves chords and strum together with the notes.

```jsonc
{
  "bpm": 90, "grid": [...], "notes": [...],

  // Chord track: from js/audio-chords.js (browser: 24 major/minor templates, 0.5 s windows, reconciled with chord notes by
  // js/chord-agreement.js), or BTC / chords/detect_audio.py (tools/chords_from_audio.py, larger vocabulary: 7, maj7, m7, sus, ...).
  // Names use chord_library.json spelling (C Db D Eb E F F# G Ab A Bb B + suffix). Gaps = no clear chord (the detector abstains).
  "chords": [ { "t0": 0.5, "t1": 2.5, "name": "G", "conf": 0.71 } ],

  // Strum pattern as played in the recording (js/strumdetect.js strumForScore; tools/strum_cli.mjs runs the same code).
  // Only present when the audio is strummed: >= 70% of hits add >= 2 pitch classes at once, the per-bar pattern repeats
  // (confidence >= 0.4) and, with chords, >= 30% of the time is chord-labelled. Picked / fingerstyle songs get no "strum".
  "strum": {
    "pattern": { /* a full chords/strum_patterns.json entry (id, grid, steps_per_bar, beats_per_bar, step_beats, swing, ...)
                    or the same schema with "id": "custom" when no library pattern is within distance 0.25 */ },
    "id": "basic",                 // pattern.id
    "grid": "D - D U - U D> U",    // the detected majority pattern, strum_patterns.json token syntax (D U d u > x -)
    "steps_per_bar": 8, "beats_per_bar": 4, "step_beats": 0.5,
    "subdivision": "8th",          // 8th | 16th | triplet
    "swing": 0,                    // as strum_patterns.json: off-beat 8ths delayed by swing * step
    "tempo": 90,                   // bpm of the strum's own beat tracker (may differ from score.bpm, e.g. half time)
    "t_bar0": 0.41,                // time of the first downbeat of bar 0 (bar lines: t_bar0 + k * beats_per_bar * 60 / tempo)
    "confidence": 0.82,            // (1 - library distance) x bar-to-bar consistency, 0..1
    "match": { "id": "basic", "dist": 0 },
    "sections": [ { "t0": 0.41, "t1": 21.7, "id": "basic", "grid": "D - D U - U D U" } ],   // per 8 bars ("custom" if none)
    "strokes": [ { "t": 0.41, "dir": "D", "step": 0, "bar": 0, "vel": 0.6, "accent": true, "muted": true } ]
                                   // every detected hit; dir follows the matched pattern step; accent / muted only when true
  }
}
```

Existing site songs keep their score files unchanged; `python tools/add_chords_to_song.py [ids]` writes
`data/chords-<id>.json` = `{ id, audio, offset, chords, strum?, chord_engine, duration }` (same field formats).

## Playing it ("as in the video")

`strumProgression(score.chords, score.strum, chordShapes)` (js/strumdetect.js) turns the two fields into the input of
`js/chordsong.js buildChordSong(prog, pattern, bpm)`: bars on the strum grid, the chord covering most of each bar, one
`chord_shapes.json` shape per symbol. Shift the built song by `startT - LEAD_IN` to line it up with the recording.
Name, diagram, fretting hand, strummed strings and synth notes then all come from that one shape;
`node tools/chordcheck.mjs` verifies this for library progressions and for songs detected from audio.

In the app every song is ONE timeline (`js/songchords.js buildMixedTimeline`, no Notes / Chords switch): bars where the tab
stacks >= 3 notes are strummed from the tab (exact voicing), bars with >= 2 detected `strum.strokes` (or a chord entry flagged
`"strum": true`) are strummed with the chord song below, all other bars play the tab's single notes. `buildSongChords` builds that chord song: it
time-warps the nominal pattern clock onto the recording (knots from `strum.strokes`, else `score.grid`), uses the detected
stroke times and directions when present, and without a strum picks a pattern by tempo / meter / style
(`chordsong.suggestPattern`). Up strokes hit G B e only; a stroke inside the tempo-dependent fretting lead
(`chordsong.fretLead(bpm)`) before a shape change sounds the open strings.

## Tools

| command | what |
| --- | --- |
| `python tools/chords_from_audio.py AUDIO -o out.json [--engine auto\|btc\|chroma]` | chords (BTC if `chords/third_party/BTC` + torch/mir_eval/pretty_midi, else librosa) + strum |
| `python tools/add_chords_to_song.py [ids]` | the above for data/songs.json entries with local audio -> data/chords-<id>.json |
| `node tools/strum_cli.mjs AUDIO.wav [chords.json] [--full]` | strum only (16-bit 22.05 kHz wav) |
| `node tools/eval_synth.mjs [chords\|strum\|all]` | accuracy on synthetic Karplus-Strong strummed progressions (tools/synth_guitar.mjs) |
| `node tools/chordcheck.mjs` | name / diagram / fretting hand / strummed strings / synth notes agree |
