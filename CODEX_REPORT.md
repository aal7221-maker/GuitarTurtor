# Verified guitar tutor improvements

## Round 4, task 1: captured Basic Pitch sustain artifacts

The supplied `tools/fixtures/basic-pitch-strum-real.json` is the evidence for this change; no fresh model inference is claimed. Extended the earlier weak-fragment rule with a separate continuity rule, applied only after recognizing a ringing chord. Fewer than three onset pitch classes can be removed when every tone belongs to that chord and is a split sustain (same pitch offset/onset within 15 ms) or a late tone contained in its measured lifetime. Split continuations may extend beyond the original offset. Overlapping same-pitch notes, nearby pitch transitions in either direction, and three-class re-strums remain protected. The original weak harmonic rule still handles the synthetic partials. Inputs are not mutated.

Before: 12 events, 8 chords, 4 unwanted singles. After: **8 events, 8 correct chords**, C/C/G/G/Am/Am/F/F, at **.522, 1.532, 2.497, 3.496, 4.496, 5.529, 6.494, 7.504 s**. These are measured model onsets (all within 40 ms of the supplied acoustic truth), not fabricated .5-second timestamps. The 4.7394 group contains both late A3 and split E3; the 8.2018 group contains two tones, which explains why a singleton-only rule failed.

Validation at this checkpoint: **32/32 regressions**, original monophonic JSON byte-identical, soft-melody/hammer fixtures and all eight repeated strums retained, selfcheck **0.10/0.49 and 0.51/3.09 mm**. Ran stamp. Limits: raw notes cannot distinguish every deliberate isolated chord-tone re-pluck from a sustain artifact. This is a context heuristic, not an onset-confidence model; the earlier Basic Pitch decoder sources still apply. No rule runs without an accepted, still-ringing chord.

Completed 2026-11-21. Read README.md, PLAN.md, RESEARCH.md, all requested modules, and tools/selfcheck.mjs before implementation. Ran the baseline before any edits. Chose three bounded changes: remove IK scratch allocation, repair missing-beat/numeric normalization, and make empty timelines release into valid rest poses. No anatomical model or fingering-cost redesign is claimed.

**1. Reuse FK/DLS scratch storage — js/hand.js.** Each chain now owns cumulative quaternions and its DLS vectors, matrix, Jacobian, coefficient array, and perturbed-angle array. Every working value is overwritten before use. The finite-difference step, damping, multiplication order, convergence thresholds, seed search, joint limits, and output objects are unchanged. Scratch tip storage remains separate from FK output because each Jacobian probe overwrites that output.

The pinned [r160 Quaternion implementation](https://github.com/mrdoob/three.js/blob/r160/src/math/Quaternion.js) confirms that clone allocates while copy/identity mutate existing storage. The [r160 SkinnedMesh implementation](https://github.com/mrdoob/three.js/blob/r160/src/objects/SkinnedMesh.js) also uses reusable math scratch objects. These primary sources support storage reuse without changing the mathematics.

Added tools/measure-motion.mjs, which simulates the same two songs at 60 fps, counts Vector3/Quaternion clones inside FK/DLS, and hashes the Float64 representations of both hands' placement matrices, joint angles, chain errors, positions, and quaternions at every frame. Hands are reused between songs, matching selfcheck. Results:

| Song | Frames | FK/DLS clone calls before → after | FK calls before = after | DLS calls before = after |
| --- | ---: | ---: | ---: | ---: |
| Choo Lo | 2,393 | 834,708 → 0 | 192,321 | 25,685 |
| Rabba | 1,716 | 2,250,676 → 0 | 555,040 | 16,152 |

The full-frame fingerprints are identical before and after all changes:

```text
choo-lo 61856229e636a00972810e10640d9986651bc2b702c3eb1bdb6d77c2817d2f29
rabba   d91d4f7ae7a0fee7345de63b11c1b9d6cf3ea083a03f265a9efd6e396eca29fe
```

This removes 3,085,384 measured clones, plus uncounted direct constructors/arrays in these loops. It does not eliminate allocations elsewhere. One instrumented Node run went from 250/333 ms to 203/255 ms (Choo Lo/Rabba); this is indicative only, not a statistically established browser speedup or GPU benchmark.

**2. Normalize absent beats and invalid tempo/duration — js/fingering.js.** Missing or nonfinite q now uses elapsed time from the first onset times bpm/60, matching the existing tab-strip convention. Finite authored q values remain intact. Nonfinite/nonpositive bpm falls back to 120; invalid d falls back to two sixteenths. Normalization still copies source records, and the tests verify input immutability.

Previously an eight-note, 120-bpm eighth-note melody with no q produced eight downstrokes and zero harmony segments. It now produces [down, up, down, up, down, up, down, up], beats [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5], and one estimated C segment. This improves picking motion for user songs and removes the missing-data exclusion from harmony; it does not establish improved chord-estimation accuracy. The existing estimator labels this fixture's key Em, illustrating why its output remains an estimate.

The bugs were established from executable repo fixtures. [Number.isFinite documentation](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Number/isFinite) verifies rejection of NaN/Infinity without coercion. [MySong's primary publication page](https://www.microsoft.com/en-us/research/publication/mysong-automatic-accompaniment-generation-for-vocal-melodies/) describes learned HMM accompaniment and subjective evaluation; it supports treating melody harmonization as inference, not ground truth. No new chord weights were taken from it. The tempo fallback assumes constant tempo when beat annotations are absent; it does not recover an unknown beat grid. Malformed strings, frets, or onset times are outside this patch.

**3. Rest safely on an empty song — js/timeline.js and js/main.js.** Empty samples now supply four nonpressing finger targets and resting right-hand targets. Unprepared empty arrays also work. The main update no longer exits solely because the event list is empty, allowing strings, markers, HUD, and the playback display to update instead of retaining the previous frame.

Before: loading an empty timeline into a real Performer after a populated one threw `Cannot read properties of undefined (reading 'strike')`. After: prepared and raw empty lists, both picking modes, and repeated seeks produce finite, joint-limited poses, six null pressed-string entries, and zero sounding events. This is a repo-specific lifecycle fix, not a biomechanical claim. The r160 math source above informed retaining normal pose computation rather than substituting invalid coordinates.

**Verification.** Added tools/regression.mjs with four tests: absent beats/picking/harmony/input preservation; invalid tempo/duration; empty/reset timelines using actual GLB rigs; forward/backward seeks and rate changes. Before the fixes, three failed and one passed; after, all four pass. Seek/rate coverage checks finite poses and joint limits at 0.5×, 0.75×, 1×, and 1.5×; it does not claim audio-device synchronization accuracy.

Selfcheck was rerun after each change batch, including the diagnostic additions. Every run retained the baseline results:

| Measurement (mean / max) | Choo Lo before → after | Rabba before → after |
| --- | --- | --- |
| Fretting error, mm | 0.10 / 0.49 → 0.10 / 0.49 | 0.51 / 3.09 → 0.51 / 3.09 |
| Picking error, mm | 2.03 / 2.04 → 2.03 / 2.04 | 2.03 / 8.73 → 2.03 / 8.73 |
| MCP flexion, degrees | 26.81 / 27.98 → unchanged | 29.55 / 51.02 → unchanged |
| PIP flexion, degrees | 65.59 / 70.67 → unchanged | 55.49 / 76.08 → unchanged |
| Tip tilt, degrees | 25.10 / 31.75 → unchanged | 30.77 / 57.45 → unchanged |

Fretting samples: 80/79; picking samples: 160/73. There are still zero/one fretting errors over 3 mm. No rendered browser/GPU or audio playback evaluation was performed; numerical rig verification is the evidence for preserved hand quality.

Commands: `npm run selfcheck`, `node --test tools/regression.mjs`, `node tools/measure-motion.mjs`, `node --check` on every edited JS/MJS file, and `git diff --check`. Ran `npm run stamp` after every js/ edit batch; only index.html's generated script hashes changed. ES modules, no-build operation, and the jsDelivr three@0.160.0 pin remain intact. No dependencies, commits, pushes, or git configuration changes. The pre-existing untracked lessons/ directory and all protected paths were untouched, including tabs.html.

**Ideas evaluated but rejected/deferred.**

- Time-aware/minimax fingering: [Hori and Sagayama, ISMIR 2016](https://archives.ismir.net/ismir2016/paper/000285.pdf) models position-transition difficulty using onset intervals and a minimax objective. It also selects strings, whereas this project's tablature already fixes them. It supports the current DP approach and suggests future time-aware costs, but supplies no calibrated weights for this rig. No measured improvement or labeled fingering benchmark justified altering the current solution.
- Pad contact, nail clearance, barres: [Xu and Wang's guitar-playing research](https://arxiv.org/abs/2409.16629) uses physically simulated control and reference motion. The existing selfcheck measures tip joints, not skinned-pad contact, collision, or adjacent-string muting. A tip-only improvement would not verify those features. Deferred until those measurements exist.
- Changing DIP coupling: the [cadaveric index-finger study](https://pmc.ncbi.nlm.nih.gov/articles/PMC2483967/) reports a DIP/PIP regression slope of 0.692 ± 0.358 under FDP loading, with explicit limitations in generalizing to live hands. This supports leaving the existing 0.66 approximation alone, not replacing it with another apparently precise universal value.
- Wrist-rotation strokes and palm mute: deferred because pick-tip onset distance alone does not verify wrist motion, pick orientation, or palm/bridge contact. No borrowed motion data or runtime physics added.
- Enabling hand frustum culling: r160's SkinnedMesh bounding-sphere computation visits skinned vertices. A stale sphere can clip animation, while recomputing it for subdivided hands adds CPU work. No verified conservative envelope or GPU measurement justified changing the current setting.
- Chord-model reweighting: MySong uses training data and evaluation; the repo has no labeled chord benchmark. Kept the estimator and its uncertainty messaging, fixing only beat availability.
- Broader performer/marker/tab-strip allocation work: the verified FK/DLS change removes the large repeated allocation source. Further caching adds invalidation/aliasing risk without a measured browser bottleneck, so it was left out.

## Hear a song -> 3D lesson

Implemented and checked in the requested order on 2026-11-21. Baseline: the original four regression tests passed, with fretting error Choo Lo 0.10/0.49 mm and Rabba 0.51/3.09 mm (mean/max).

**1. Recording to lesson.** `create.html` now transcribes automatically after microphone Stop or tab-capture Stop, including Chrome ending the shared video track. Uploaded files still wait for **Write the tab**, which also allows rerunning the current audio with another model. Controls prevent overlapping recording/transcription operations. The checked-by-default **Open the 3D lesson when done** option navigates in the same tab after saving; unticking it leaves the result and download links visible. Failed localStorage saves do not navigate to an unavailable song.

`js/song-audio.js` stores the original Blob in IndexedDB by song id. Open, upgrade, transaction, request, abort, quota, blocked-storage and close failures are caught; stalled operations time out, and a late database open is closed. Writes resolve only when their transaction commits. Delete in the saved-song list also requests deletion of that recording. Unavailable audio storage leaves the score usable with video/synth fallback and a status message; deletion failure is reported rather than claimed successful.

`js/main.js` loads a user song's stored Blob before hosted audio, YouTube or synth. Scores retain their video-time format; `audioOffset` shifts notes, grid and chord segments to the recording's clock without modifying the saved score. Recording-relative transcription is computed before this shift, preserving leading silence and notes around a negative capture offset. The decoded `audioDuration` supplies a finite scrub range even when MediaRecorder WebM metadata reports Infinity. Old songs without this metadata retain their existing behavior. Blob URLs are revoked when replaced, and stale asynchronous loads cannot overwrite a newer selection. The lesson opens paused at zero with Play ready; automatic navigation does not require browser autoplay permission. `js/user-songs.js` needed no change: its existing JSON persistence preserves the additional metadata and `score.chords`.

Verification before step 2: seven tests passed, including page-controller tests for recording completion, automatic navigation, opt-out, uploads/reruns, failed score saves, audio deletion and recording-time translation.

**2. Preserve chord events.** `js/songmaker.js` retains the existing 50 ms onset grouping and melody beam search. Within a group, overlapping notes that meet both the existing relative loudness threshold (default 75% of the loudest onset) and amplitude 0.3 qualify for chord assignment when they contain at least three distinct pitch classes. Duplicate MIDI pitches collapse; octave doublings remain. Repeated chord strums are retained even when their top note repeats within 130 ms.

Assignment first searches matching pitch-class entries in `chords/chord_shapes.json`, translating its low-E-first order into score string 0 = high e. It uses only the detected pitches present in a shape, without adding unheard notes or changing octaves. If no library shape fits, an exact-pitch string search prefers lower frets. Every accepted assignment has distinct strings, frets 0–22 and a maximum four-fret span excluding open strings. Unplayable groups, dyads, quiet notes and non-overlapping groups retain the melody path. Library loading failure still permits string search. Newly transcribed songs containing chord events use the existing pick mode to sweep their strings; monophonic songs retain fingerstyle mode. No rig or timeline changes were needed.

Before editing the converter, captured its monophonic score JSON in `tools/fixtures/monophonic.json` at offsets 0, +3.25 and -0.5 seconds. All three outputs remain identical as serialized JSON, with and without the chord library. Eleven tests passed before starting step 3, including grouping, repeated strums, fallback, pitch preservation and assignment validity.

**3. Chords from the recording.** New `js/audio-chords.js` analyzes the decoded mono 22,050 Hz Float32Array in half-second windows. It computes Hann-windowed FFT peaks, interpolates their frequencies, folds pitches into chroma, aggregates frames by median, and cosine-matches 24 major/minor templates. Root weighting, scaled emissions, sticky Viterbi smoothing and similarity/margin confidence follow the essential approach in `chords/detect_audio.py`. Silence/unclear windows can remain unlabelled; adjacent identical results merge with duration-weighted confidence. Processing yields periodically to keep progress updates responsive.

Creation saves `score.chords = [{t0, t1, name, conf}]`, including in downloaded JSON. Names use the library spelling: C, Db, D, Eb, E, F, F#, G, Ab, A, Bb, B. `js/chordpanel.js` prefers these segments and labels them **from the audio**. Scores without that array retain `js/harmony.js` with **estimated from the melody**. An explicitly empty audio result stays empty instead of silently substituting melody estimates. Fourteen tests passed after this stage.

**Research and sources.** Read the specified repository files before implementation. Used the connected Firecrawl tools because the local Firecrawl CLI was unavailable. Product pages describe workflows and vendor claims, not independently measured accuracy:

- Songscription, audio-to-tab workflow and transcription review: https://www.songscription.ai/guitar-tab-generator and https://www.songscription.ai/blog/audio-to-guitar-tabs . These support making the generated result immediately available for listening and review; no proprietary model or accuracy claim was adopted.
- Note2Tabs, playable tabs, string/fret optimization, alternative chord fingerings and practice playback: https://www.note2tabs.com/ . This informed retaining guitar constraints after pitch detection.
- Tabtify, recording/upload to editable draft, playback and source comparison: https://tabtify.com/ . This supports preserving the heard recording alongside the tab. No audio was sent to these services.
- Spotify Basic Pitch, polyphonic transcription and best results with one instrument at a time: https://github.com/spotify/basic-pitch . Browser API, mono input and 22,050 Hz processing: https://github.com/spotify/basic-pitch-ts . Existing pinned models and dependencies were retained.
- AudioLabs FMP, chroma and cosine template matching: https://www.audiolabs-erlangen.de/resources/MIR/FMP/C5/C5S2_ChordRec_Templates.html . Temporal context and Viterbi postfiltering: https://www.audiolabs-erlangen.de/resources/MIR/FMP/C5/C5S3_ChordRec_HMM.html . These corroborate the local Python detector's approach and its simplified browser port.

**4. Final verification.** `node --test tools/regression.mjs`: **18/18 passed**, comprising the original four and fourteen added tests. Fixtures under `tools/fixtures/` cover original monophonic output, generated sine/harmonic-sum audio, and browser API doubles. Additional final checks exercise the actual main.js loading/clock functions: exact Blob identity and priority, ready-to-play state, offset alignment, video/synth fallback, URL replacement and stale reads. IndexedDB tests cover round-trip storage/deletion and eight failure stages, including request success followed by transaction abort. The create controller is run from the real HTML; capture testing ends the shared video track.

Synthetic audio correctly names C, G, Am and F individually with both pure sine tones and five harmonics. A C–G–Am–F progression produces the expected four segments and boundaries, with leading/trailing silence unlabelled. These are deterministic DSP checks, not a real-recording accuracy benchmark.

| Fretting error, mm (mean/max) | Before | After |
| --- | --- | --- |
| Choo Lo | 0.10 / 0.49 | 0.10 / 0.49 |
| Rabba | 0.51 / 3.09 | 0.51 / 3.09 |

`npm run selfcheck` passed before and after. `node --check` passed for every changed/new JS and MJS file and the extracted create.html inline module. `npm run stamp` updated only index.html and create.html; `git diff --check` passed. Protected tracked paths have no changes, including tabs.html. The pre-existing untracked lessons/ directory was left alone. No runtime dependencies, CDN libraries, commits, pushes or git-configuration changes were made; three remains jsDelivr 0.160.0. All audio decoding, transcription, chord analysis and recording storage remain in the browser.

**Known limits and skipped work.** No live microphone/tab-share permission session, real ONNX/TF.js inference run, WebGL visual review or audio-device latency measurement was performed. Controller and storage tests use API doubles; real browser codecs, quota, eviction and permissions still vary. A recording that cannot be stored falls back to video/synth; failed audio deletion may require retrying when storage is accessible or clearing site data. Saved audio playback was added only to the requested 3D view; tabs.html remains untouched.

The browser detector intentionally substitutes an FFT chromagram for Python's harmonic separation/CQT and uses the permitted fixed 0.5 s windows. Beat-synchronous analysis, tuning estimation and optional seventh-chord templates were skipped to keep this dependency-free port small and verifiable. Dense mixes, percussion, strong harmonics, single-note overtones, fast changes and non-standard tuning can confuse it; confidence is a heuristic, not a calibrated probability. Grouping can miss slow strums beyond 50 ms or quieter chord tones. Chord names do not infer missing tab pitches. String assignment enforces the requested geometric constraints, but the existing rig still lacks a true barre pose, and the monophonic beam search does not optimize transitions between whole chord shapes. Those animation/solver extensions were outside this change and would require new motion evidence before altering the protected hand paths.

## Refinement round: strums, chord lifetimes, detector and caching

Continued from `9999ed2` on 2026-11-21. Read both existing report sections, the requested modules, controller, measurement tool and all fixtures before editing. Baseline was 21 passing tests, with Choo Lo 0.10/0.49 mm and Rabba 0.51/3.09 mm fretting error (mean/max). Implemented and verified A, B, C, then D; inspected E without deleting anything. This section supersedes the earlier grouping/detector/cache limitations where addressed below.

**A. Complete strums before the melody filter.** `js/songmaker.js` now considers the complete overlapping, playable strum before applying the melody's relative-amplitude skyline. The earlier extension required every string to reach 75% of the strongest note, then applied that filter again when assigning the chord. Unequal model strengths could therefore split a sweep or remove its identifying tones even within 50 ms. Strum recognition retains the absolute 0.3 floor, distinct pitches, overlap, six-string/four-fret playability, 90 ms general span and 100 ms monotonic span. It preserves pitches and places the chord at the earliest onset. Quiet/unplayable groups still take the original melody path. The serialized monophonic fixture remains byte-identical at all three offsets, with and without the shape library.

Expanded the existing test to 8, 12 and 15 ms/string, up/down, library/no-library, with unequal strengths `[.42, .85, .56, .71, .38, .63]`. These are deterministic note-event fixtures using the KS chord pitches and timing, **not newly transcribed Basic Pitch output**. Paired runs against the converter in HEAD gave the same results in both directions:

| Per-string spread | HEAD score events / chord events | Revised score events / chord events |
| --- | ---: | ---: |
| 8 ms | 8 / 0 | 8 / 8 |
| 12 ms | 12 / 0 | 8 / 8 |
| 15 ms | 16 / 0 | 8 / 8 |

Each revised strum produces exactly one chord event at .5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5 or 7.5 seconds. The older equal-amplitude 12 ms fixture already passed at HEAD; it did not expose this failure.

**B. Chord lifetimes and panel segments.** Chord events additionally retain a relative `sustain` duration from the detected note ends; monophonic records stay unchanged. `js/chord-agreement.js` constructs note overrides independently of audio-window boundaries. A melody/dyad fragment no longer cancels a chord. Overrides begin at the first onset even when the detector's first labelled window starts later, and stop at the next named chord, measured sustain end or explicit rest. An explicit rest can end at a subsequent melody onset. Older scores lacking measured sustain use the existing audio horizon rather than guessing silence from quantized beat lengths.

The shared cleanup rounds boundaries consistently, excludes invalid/empty intervals, merges identical adjacent name/source pairs with minimum confidence, then removes segments shorter than 50 ms. New creation already calls this reconciliation before saving/downloading; runtime applies it to older scores too. The panel uses original score records when available so normalization cannot discard `sustain`. Its countdown skips every repeat of the current chord, uses exact segment ends and rounds a positive countdown upward to avoid displaying `0.0 s` for a future onset.

A paired held-chord/rest fixture, with the same audio windows and notes on both versions, changed from four segments (`Em 1–1.5`, `C 1.5–3`, `F 3–3.5`, `Em 3.5–5`) to two (`C .5–2.5`, `F 3–3.5`), correctly preserving the intervening/final rests. Confidence is still audio evidence, not a recalibrated probability; a note-only extension without audio evidence conservatively contributes confidence zero.

The actual create controller, with the unequal-strength eight-strum model double and deliberately wrong Em/A audio labels, now saves exactly four segments: C .5–2.5, G 2.5–4.5, Am 4.5–6.5, F 6.5–8.5. Runtime reconciliation gives the identical array, including after time shifts and repeated reconciliation. At .5/1.7/2.7/4.7/6.7/7.7 s the labels are C/C/G/Am/F/F. Dedicated panel-method checks cover repeated names, zero/sub-50 ms intervals, seeking backward and segment ends.

**C. Detector measurements and the WAV discrepancy.** Kept only the tested combination of a gentle `1/sqrt(frequency)` spectral rolloff and square-root compression of aggregated chroma strengths. The first reduces upper-partial dominance; the second balances unequal string strengths. FFT size, templates, tuning tolerance, half-second windows and Viterbi transition costs remain unchanged. Rejected rolloff-only and some per-peak compression combinations because they reduced accuracy on existing fixtures. Final detector output also excludes sub-50 ms terminal segments.

`tools/fixtures/audio-chords-baseline.mjs` is byte-identical to HEAD's detector after LF normalization. `node tools/measure-chords.mjs .qa/strum_C_G_Am_F.wav` compares that baseline with the revised implementation. Percentages measure duration with the correct label over the annotated intervals; they are not note-transcription accuracy, and do not penalize labels outside those intervals.

| Fixture | Before | After |
| --- | ---: | ---: |
| Sine C G Am F | 100% | 100% |
| Five-harmonic C G Am F | 100% | 100% |
| KS C G Am F | 100% | 100% |
| KS C G Am F, +35 cents | 100% | 100% |
| KS C G Am F, −35 cents | 100% | 100% |
| KS D A Bm G | 87.5% | 87.5% |
| KS D A Bm G, +35 cents | 81.25% | 100% |
| KS D A Bm G, −35 cents | 93.75% | 100% |
| KS Em C G D | 93.75% | 100% |
| KS Em C G D, +35 cents | 100% | 100% |
| KS Em C G D, −35 cents | 93.75% | 100% |
| Supplied PCM WAV, C G Am F | 62.5% | 100% |

Across the original 80 annotated seconds, accuracy rises from 95% to 98.75%; completely correct chord intervals rise from 39/44 to 43/44. No fixture loses accuracy. The remaining D–A–Bm–G failure is one unlabelled second at the start of G; the detector still abstains there.

Added a small dependency-free diagnostic RIFF decoder in `tools/fixtures/wav.mjs`, with tests for signed PCM values, stereo averaging, padded chunks and truncation. It reads the supplied file directly: 22,050 Hz, mono, signed 16-bit PCM, 209,475 samples / 9.5 s. SHA-256: `c233199211b4ff98822d0969bafb64ed83569b3ead40b4e123d06ac785d6eefd`. No resampling or browser decode is needed for this format. The WAV remains untracked in `.qa/`.

The old detector on those samples reproduces Em at .5, C at 1.5, G at 2.5, Em at 4.5, A at 5 and F at 6.5. Thus the detector failure exists in the waveform analysis itself, independently of browser caching or score reconciliation. The JS KS fixture and this WAV have different spectral balances: in the first C window, normalized C/E/G/B chroma is approximately .52/.63/.43/.22 for the JS fixture, versus .38/.66/.34/.43 for the WAV. The latter favors Em. In the WAV's Am window at 5 s, Db is .23 versus C .19, while E dominates at .76, favoring A major. Upper harmonics can create these competing classes (E's third partial is near B; A's fifth is near Db). The analysis demonstrates excessive weighting of these partials and strong strings; the precise excitation/seed used to generate the WAV is not available. A PCM16-quantized version of the JS fixture still scores 100% before the change, so PCM quantization alone does not explain the discrepancy.

After the change the WAV yields C .5–2.5, G 2.5–4.5, Am 4.5–6.5 and F 6.5–9.0. The final F extends 0.5 s beyond the annotated interval; the 100% figure excludes that tail. Measured note sustain supplies the more precise endpoint when creating a score.

**D. Content hashing.** `tools/stamp.mjs` now creates a missing import map before module loads, traverses re-exports and parent-directory dependencies as well as imports, and generates matching preloads. The five local modules in create's graph all resolve with `?v=`: `js/songmaker.js`, `js/chord-agreement.js`, `chords/chordnamer.js`, `js/song-audio.js`, `js/audio-chords.js`. Before, only the three direct entry imports were hashed; both transitive modules were unversioned. Inline entry imports now use the map, avoiding separate URLs for the same module. Existing CDN versions remain pinned; remote model/library downloads are outside the local content-hash graph.

Regression checks resolve each edge relative to a hashed parent URL and verify its exact SHA-256 content hash and preload. They also verify map ordering, re-export/literal dynamic import discovery, CRLF-to-LF normalization and byte-for-byte idempotence. Two consecutive `npm run stamp` invocations leave the second with `already up to date`. Generated index/create pages are LF-only, and classic-only `tabs.html` remains byte-identical. Live network-request inspection was unavailable; the import-resolution check is a static graph verification.

**E. Temporary songs.** The tracked catalog contains only Rabba, Choo Lo and Falling in Love with You. No temporary QA song was found in repo catalog/score files, so none was removed. Existing localStorage songs and the pre-existing untracked `lessons/` directory were left alone.

**Sources and validation.** The evidence is the paired HEAD implementations, [converter/panel/controller regression cases](tools/regression.mjs), [deterministic audio fixtures and accuracy definition](tools/fixtures/audio.mjs), [frozen detector baseline](tools/fixtures/audio-chords-baseline.mjs), [measurement command](tools/measure-chords.mjs), [PCM decoder](tools/fixtures/wav.mjs), and the supplied WAV and Chrome QA observations. Pitch naming continues to use the existing [chord namer](chords/chordnamer.js). The earlier research links remain background; no new external accuracy benchmark is claimed.

Final checks: **28/28 regression tests pass**. `npm run selfcheck` preserves Choo Lo **0.10/0.49 mm** and Rabba **0.51/3.09 mm**, with the other reported selfcheck metrics unchanged. `node --check` passes for every touched JS/MJS file and create's extracted inline module; `npm run stamp` and `git diff --check` pass. Protected paths have no tracked changes. No runtime dependencies, new CDN libraries, commits, pushes, git configuration changes or uploads.

**Remaining validation limit.** The user-supplied Chrome/real Basic Pitch baseline was 13 score events, 5/8 chord strums and ten saved chord segments, with a missing initial chord and Am at 6.7 s. That end-to-end run could not be repeated here: computer-use inventory returned `The browser access host is unavailable`, and direct model/package downloads from this restricted shell failed. The attempted isolated install was cancelled and installed no dependencies. Therefore the new eight-strum counts and four saved segments are verified on deterministic converter/controller inputs, not on freshly inferred notes from this WAV. Actual Chrome Basic Pitch producing 8/8 chord events remains unverified. Older saved scores cannot recover pitches already discarded by their original conversion, and without measured sustain or explicit rests their end times remain limited by existing audio evidence. Dense mixes, missing/quiet notes, unplayable groups and strums beyond 100 ms remain outside the demonstrated cases.

## Phantom-note filter

Continued from `76dd208` on 2026-11-21. The supplied Chrome observation for this round is **12 score events, including 8 correct strums**. Read the requested modules, measurement tool and fixtures before editing. Baseline: 28/28 regression tests passed; selfcheck matched the required values.

**What and why.** `js/songmaker.js` now rejects likely residual single-note activations after recognizing chords, before beat estimation and fingering. This prevents a rejected fragment from influencing hand positions, the HUD, tempo or beat durations. It only considers an onset group containing exactly one raw note; accepted chords and ambiguous multi-note groups are never filtered.

All of these conditions must hold: positive duration at most 180 ms; activation below 0.4 and below half the median activation of the still-ringing chord notes; onset at least 120 ms after the last string of the latest accepted chord; at least three distinct chord pitch classes whose measured offsets cover the entire fragment; and either a matching pitch class or a third/fifth partial above a sounding note (19/28 semitones). Octave partials already match pitch class. A new chord replaces old evidence. A nearby pitch, 1-3 semitones away, ending within 60 ms of the onset protects plausible hammer-ons/pull-offs, using both the chord constituents and the previous retained group. No duration/activation gate is applied to monophonic input without chord context. Raw inputs are not modified.

These numerical thresholds are conservative local heuristics, not published or calibrated Basic Pitch recommendations. The median prevents one exceptionally strong string from setting the reference. Requiring the whole fragment to fit inside measured ringing evidence avoids suppressing a note that continues into a rest. Global transcription thresholds and model dependencies remain unchanged.

**Research and sources.**

- [Spotify's deployed Basic Pitch 1.0.1 decoder](https://cdn.jsdelivr.net/npm/@spotify/basic-pitch@1.0.1/esm/toMidi.js): onset/frame thresholds control onset selection and note continuation; short candidates are discarded. This repo already passes 0.6/0.4 and a minimum of 9 frames, with inferred onsets and the remaining-energy pass enabled. With a 256-sample hop at 22,050 Hz, nine frames are approximately 104 ms; the decoder rejects lengths less than or equal to that frame count. Its 11-frame tolerance permits temporary subthreshold gaps. Inferred onsets come from frame-activation increases; the remaining-energy pass can recover notes without passing the onset threshold. Amplitude is the mean note-frame activation, **not acoustic amplitude, instantaneous decay level, or onset confidence**. Thus raising only the onset threshold need not eliminate residuals, while raising length/frame thresholds globally risks losing genuine quiet/short notes. The inspected decoder suppresses neighboring semitone bins but has no ringing-chord-relative or harmonic-relation veto.
- [Spotify's Python note decoder](https://github.com/spotify/basic-pitch/blob/main/basic_pitch/note_creation.py) corroborates this two-pass design, mean-activation calculation and configurable duration filtering. Its current defaults are not substituted for the pinned browser version.
- [Julius O. Smith, Karplus-Strong algorithm](https://ccrma.stanford.edu/~jos/pasp/Karplus_Strong_Algorithm.html) describes noise excitation and frequency-dependent string losses. This supports modeling decaying string partials. It does not prove that a particular Basic Pitch event is an overtone. The causal explanation for the supplied Chrome artifacts remains an inference.

**Fixture construction and results.** `tools/fixtures/phantom-notes.mjs` generates raw model-like `{t,e,p,a}` records using the existing KS fixture's C x32010, G 320003, Am x02210 and F 133211 pitches: two strums per chord, starting at .5 s, one second apart, with 12 ms/string. Notes end at the next strum/change, except explicitly fragmented detections. This is a deterministic model-level reproduction, **not captured neural inference or a replay of the original Chrome raw output**.

Four residuals are derived from those fundamentals/partials: E3 (MIDI 52) at 4.82 s, A3 (57) at 5.18 s, B4 (71) at 5.84 s and Db5 (73) at 6.16 s. Earlier E3/A3 detections end before their fragments; octave-related chord tones remain sounding. The residual activation islands last 12-15 nominal frames (139-174 ms), with .44 every fourth frame and .29 between, yielding means around .33. They model frame peaks exceeding .4, separated by gaps shorter than the energy tolerance, and survive the existing minimum length. These activation patterns and times are illustrative assumptions, not measured audio amplitudes. Separate counterexamples include a soft D4 passing note over Am, a longer soft E3, a short E3-to-G3 hammer over C, and a soft C4 passing note over G.

`tools/fixtures/songmaker-before-phantoms.mjs` freezes the notes-to-score portion of `76dd208` for repeatable paired measurements. Run `node tools/measure-phantoms.mjs`:

| Fixture | Events before | Events after | Chords before / after |
| --- | ---: | ---: | ---: |
| Clean KS strums | 8 | 8 | 8 / 8 |
| KS residual fundamentals and partials | 12 | 8 | 8 / 8 |
| Soft passing melody and sustained chord tone | 10 | 10 | 8 / 8 |
| Soft chord-tone hammer-on | 9 | 9 | 8 / 8 |
| Phantoms plus real melody | 13 | 9 | 8 / 8 |

Every resulting score is byte-identical as serialized JSON to the frozen converter applied to that fixture's intended retained notes. Regression checks repeat this with/without the shape library and at offsets 0, +3.25 and -4.6, including chord context before a clipped recording offset. All eight strums retain their original onset, pitches, assignment and sustain. Removing extra events can legitimately change beat-grid estimation and adjacent durations relative to the contaminated score. The original monophonic snapshots remain byte-identical; the existing 8/12/15 ms up/down strum cases now also assert full serialized equality against the frozen converter. Additional cases protect stronger/longer notes, dyads, late strings near the strum, expired/incomplete chord evidence, wrong-octave harmonic lookalikes, a single-note hammer transition and missing duration. The create-controller harness saves eight events and four C/G/Am/F segments; runtime reconciliation agrees.

**Verification.** `node --test tools/regression.mjs`: **31/31 pass**, including all original 28. `npm run selfcheck`: Choo Lo **0.10/0.49 mm**, Rabba **0.51/3.09 mm**, with the other metrics unchanged. `node --check` passes for all five touched/new JS/MJS files and extracted inline modules from the two stamped HTML pages. `npm run stamp` updates only generated songmaker hashes in index.html/create.html. The existing `node tools/measure-chords.mjs` detector results remain unchanged. `git diff --check` passes. Protected paths, including the pre-existing untracked lessons/ directory, remain untouched. No runtime dependencies, CDN libraries, git commits/pushes/config changes, or uploads of local content.

**Limits.** A real short, weak chord-tone re-pluck or harmonic melody note with no detectable pitch transition can have exactly the same `{t,e,p,a}` as a residual and can be removed. Conversely, longer/stronger artifacts, multi-note artifacts and fragments without three sustained pitch classes are deliberately retained. Offsets and model activations can be inaccurate; comparing mean activations does not measure the chord's current acoustic decay. Guitab uses onset-head strength rather than Basic Pitch's mean-frame strength, so the shared converter's thresholds are not cross-model calibration. No new Chrome inference, microphone recording or audio-quality benchmark was performed. The synthetic 12-to-8 result demonstrates this bounded post-processing rule, not guaranteed cleanup of the uncaptured original raw output. Existing saved songs need re-transcription to benefit.
