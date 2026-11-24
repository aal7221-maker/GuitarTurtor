# Audio -> tab: training / tuning results (Choo Lo + Locked Away)

Pipeline = basic-pitch -> skyline/merge -> beam-Viterbi string choice (`pipe.py`, a parametrised copy of pitch-model/transcribe.py). Audio only, no on-screen tab.
Score: fractions of reference notes (match <=80 ms); Locked Away notes with conf<0.7 or flags excluded (N=676 scored; Choo Lo N=174, until 38.7 s). Train = first 60% of each song, held-out = last 40%. "extra" = unmatched auto notes / N.
Not run: guitab chain (sweep_guitab etc.) - not needed for the conclusions below. Quantisation not tuned (scoring is on onset time, not grid).

## 1. Baseline (defaults) vs tuned params (17 params, Optuna TPE 150 trials, objective = mean(recall,pitch,exact) - 0.1*extra)
Format recall / pitch / string+fret exact.

| Split | Song | Baseline | Tuned |
|---|---|---|---|
| train (first 60%, tuned on both songs) | Choo Lo | .935 / .841 / .729 | .944 / .860 / .766 |
| train | Locked Away | .743 / .514 / .229 | .713 / .514 / .370 |
| **held-out (last 40%)** | Choo Lo | .940 / .478 / .388 | .910 / .478 / .328 |
| **held-out** | Locked Away | .778 / .534 / .280 | .753 / .509 / .283 |
| full song | Choo Lo | .937 / .701 / .598 | .931 / .713 / .598 |
| full song | Locked Away | .757 / .522 / .250 | .729 / .512 / .334 (partly train) |
| LOSO: tune on Choo Lo -> test Locked Away | Locked Away | .757 / .522 / .250 | .660 / .499 / .283 |
| LOSO: tune on Locked Away -> test Choo Lo | Choo Lo | .937 / .701 / .598 | .943 / .672 / .529 |

Held-out does not improve (exact: -0.06 Choo, +0.003 Locked; LOSO: +0.03 / -0.07; recall drops). Gains are on train only = overfit. **Defaults kept** in `best_params.json`; the tuned set is in `tuned_params_split60.json` for reference. (The existing "default" preset had already been tuned on Choo Lo.)
Biggest pipeline weakness is Locked Away: pitch 0.52 (distorted/band mix, 58% extra notes), not the params.

## 2. Learned string chooser (HistGradientBoosting per candidate string + beam decode; features = pitch, candidate string/fret, previous 3 notes' string/fret/pitch, dt, hand anchor)
Pretrain GuitarSet (players 00-03, 240 clips, 166k rows) ; "priority" = our songs weighted x10 / x25 in the same fit (the local venv's torch is broken; GBM weighting = the priority fine-tune; the notebook does a real two-stage torch version). GuitarSet test = player 05, 8 clips, string accuracy on true pitches.
Cells: string accuracy on true (reference) pitches / end-to-end exact (audio pitches + chooser).

| Model | Held-out last 40% Choo Lo | Held-out last 40% Locked | LOSO -> Choo Lo | LOSO -> Locked | GuitarSet test |
|---|---|---|---|---|---|
| Viterbi costs (default) | .866 / .388 | .401 / .280 | .868 / .598 | .392 / .250 | .429 |
| GuitarSet only | .149 / .090 | .649 / .251 | .155 / .178 | .666 / .294 | .688 |
| GuitarSet + our songs x10 | .149 / .090 | .892 / .348 | .155 / .178 | .675 / .312 | .694 |
| GuitarSet + our songs x25 | .149 / .090 | .871 / .319 | .155 / .178 | .661 / .308 | .698 |
| our songs only | .567 / .284 | .918 / .394 | .109 / .075 | .478 / .246 | .494 |

(our-songs-only reaches 1.00 on its own train = memorises.) Train oracle for GS+ours x25: .28 Choo / .95 Locked.
Reading: learned chooser beats Viterbi on Locked Away (string acc .40 -> .65-.92, e2e exact .28 -> .31-.39) and on GuitarSet (.43 -> .69), but collapses on Choo Lo (.87 -> .15): that song is an unusual G-string melody at frets 8-9 where "stay on the previous string" beats the usual low-fret prior, and 107 notes at x25 weight are not enough to teach it. The 2 songs have opposite string idioms, so one learned model does not fit both; Viterbi is the better *average* (held-out e2e exact mean .334 Viterbi vs .219-.341 learned, LOSO mean .424 vs .245-.255).
**Pick:** Viterbi default for general use. Best single learned candidate (GS + ours x10; saved `string_chooser_gbm.pkl`, trained on all data, sklearn pickle) is worth using only for band/chord-heavy material like Locked Away; a hybrid (learned log-prob + Viterbi hand-position/same-string costs, or per-song idiom flag) is the obvious next step.

## 3. Verdict
- Can generalise: onset detection (recall .94 Choo, .76 Locked, params barely matter), general string/fret priors from GuitarSet (+26 pts on its test split vs the hand costs).
- Cannot (with 2 songs): param tuning (train-only gains), a learned chooser that satisfies both songs, pitch accuracy on Locked Away (needs a better pitch model / stem separation, i.e. real fine-tuning with more audio+tab pairs: GAPS, SynthTab, GuitarSet audio, more of the user's lesson tabs). The GuitarSet mono-mic audio (656 MB) was downloaded to `data/guitarset/` but not used for pitch fine-tuning locally (CPU/4 GB GPU, torch broken in the venv); see notebook.
- Numbers are small-sample (Choo Lo held-out = 67 notes); differences of a few points are noise.

## 4. Colab models in the full pipeline (2026-11-24; `eval_models.py`, `eval_loso.py`, `loso_chooser.py`, `guitab_raw.py`)
Same splits/scorer as above (pipe.score = eval_tabs.py metrics + conf/flag exclusion). Grouping/params = `best_params.json` defaults.
Engines: basic-pitch (pipe.py cache), guitab E25 onset 0.3 (as the browser), `pitch_crnn.onnx` (notebook CQT: 22.05 kHz, hop 512,
C2, 120 bins, 24/oct, log1p(10|C|); onset peaks > 0.5, note end where frame < 0.5 - the notebook's threshold, fixed because the
train split was in its training set). Chooser = `string_chooser.onnx` beam decode; hybrid = Viterbi cost + w * -log p(string),
w picked on the train split (first 60%, mean exact of both songs) from {.1,.25,.5,1,2,4} -> w = 0.25.
LOSO: no training for (a)/(b) = whole-song score; choosers retrained locally without the held song (`loso_chooser.py`, notebook
Part A recipe on CPU, selected on GuitarSet 04 + held-out lessons, never the test song), hybrid w picked on the other song.
pitch_crnn has no LOSO (Colab fine-tuned it on the first 60% of both songs).
Cells: held-out (last 40%) recall / pitch / **string+fret exact**; LOSO = string+fret exact on the whole held song.

| | Choo Lo held-out | Locked Away held-out | **held mean exact** | LOSO Choo / Locked (mean) |
|---|---|---|---|---|
| (a) basic-pitch + Viterbi (default) | .940 / .478 / .388 | .778 / .534 / .280 | **.334** | .598 / .250 (.424) |
| (b) guitab + Viterbi | .925 / .522 / .433 | .785 / .470 / .219 | .326 | .632 / .204 (.418) |
| (c) pitch_crnn + Viterbi | .851 / .478 / .388 | .541 / .466 / .208 | .298 | n/a (trained on both songs) |
| (d) basic-pitch + string_chooser | .940 / .478 / .090 | .778 / .534 / .147 | .119 | .178 / .167 (.172) |
| (e) basic-pitch + hybrid (w .25) | .940 / .478 / .299 | .778 / .534 / .272 | .286 | .420 / .336 (.378) |

"best pitch" for (d)/(e) = basic-pitch (best Viterbi held-out mean). Other combinations (held exact Choo / Locked):
guitab+chooser .119 / .269, guitab+hybrid .328 / .315 (mean .322), crnn+chooser .104 / .197, crnn+hybrid .358 / .330 (mean .344,
not a protocol cell: w chosen for basic-pitch, no LOSO possible, Choo recall .85, +1 pt = noise). Hybrid sweep held mean
(basic-pitch): w .1 .318, .25 .285, .5 .308, 1 .273, 2 .260, 4 .233 - never above Viterbi .334.
pitch_crnn onset threshold (held, info only): 0.3 gives .433 / .244 exact (recall Locked .63); the default 0.5 loses recall on
Locked Away (.54 vs basic-pitch .78). Train split is memorised (pitch 1.00 / .92), so it cannot be tuned there.

**Why the chooser fails end to end:** the notebook's string accuracies are teacher-forced (true previous strings as input).
Free-running (its own previous choices, reference pitches) it collapses:

| string acc on reference pitches | teacher-forced | free-running | Viterbi |
|---|---|---|---|
| Choo Lo held-out (Colab model) | .716 | .149 | .866 |
| Locked Away held-out (Colab model) | .935 | .556 | .401 |
| Choo Lo LOSO (local) | .201 | .241 | .868 |
| Locked Away LOSO (local) | .891 | .510 | .392 |

It mostly learned "same string as the last note", so one wrong choice propagates; with audio pitches (extra/missing notes) the
history is worse still. It does carry real signal on Locked Away (free-run .51-.56 vs Viterbi .39; hybrid LOSO .336 vs .250),
but costs Choo Lo far more (.598 -> .420).

**Verdict: nothing beats the default on held-out; basic-pitch + Viterbi stays** (local helper) and nothing was integrated into
`js/songmaker.js` / `make_song.py`. Browser check (create.html upload of audio/choo-lo.mp3, headless Chrome, `--until 38.7`):
guitab E25 (the page's default option) 93.1 / 73.0 / 63.2 % (recall / pitch / exact), basic-pitch 90.8 / 70.7 / 60.3 %.
Next for the chooser: train with scheduled sampling / noisy history (or drop history and keep only pitch + hand anchor), and
score free-running in the notebook; for pitch_crnn: more audio+tab pairs, lower onset threshold, LOSO run on Colab.
Basic-pitch timing: pipe.py uses basic_pitch's own `model_output_to_notes` (model_frames_to_time), not frame/FPS, so the
~0.4 s/100 s drift seen elsewhere does not affect these numbers.

## Files
`pipe.py` (pipeline+scorer), `tune.py`, `chooser.py`, `exp_chooser.py`, `baseline.py`, results JSON (`tune_results.json`, `chooser_results.json`), `best_params.json`, `tuned_params_split60.json`, `string_chooser_gbm.pkl`, `package_data.py` (-> `out/GuitarTurtor_train.zip`, audio only with `--with-audio`), `colab_finetune.ipynb` (built by `mk_nb.py`), `eval_models.py` / `eval_loso.py` / `loso_chooser.py` / `guitab_raw.py` (section 4; run with `.venv-ort` = anaconda + onnxruntime 1.20.1 + basic-pitch 0.4.0 --no-deps; results `eval_models.json`, `eval_loso.json`), `.gitignore` (audio/data/work).
Data: GuitarSet (Xi et al., ISMIR 2018), Zenodo 3371780, CC BY 4.0 - attribution required. Our audio and GuitarSet stay local, never commit.
