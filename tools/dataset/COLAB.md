# Training on Colab with the lesson dataset

Private personal research only. The zip with `--with-audio` contains copyrighted recordings: keep it in your own
Drive, never share the folder, never commit it. Labels (`tools/dataset/labels/*.json`) hold derived tab data and video ids only.

## 0. What is in the dataset
- `manifest.jsonl`: one row per video (id, title, channel, url, duration, tuning/capo, n_notes, onset %, conf stats,
  sanity flags, pitch evidence, pass/fail + reason, spot-check result). `priority: true` = Choo Lo, Locked Away.
- `labels/<id>.json`: site score format `{bpm, grid, notes:[{t, q, d, n:[[string, fret]], conf, cn, flags, sec}]}`,
  string 0 = high e, `t` = onset seconds in the video's audio, frets relative to capo, tuning/capo in the manifest.
- `audio/<id>.m4a|mp3` (only `--with-audio`): the video's own audio track, same timeline as `t`.
- Choo Lo labels are valid only up to `until` = 38.7 s (rest unreviewed).
- Lesson labels: pages that failed the gate were dropped; their time spans are in `exclude` (manifest and
  `label.source.exclude`) and the notebook masks them. Each note has `ev` = basic-pitch evidence per pitch.
- Known reader error modes left in kept labels (by-eye spot checks, ~95% of events right): 6<->8 digit
  confusions, a few missed notes, tied "(n)" notes read as new attacks, rare wrong string. Treat as
  slightly noisy labels (the notebook drops conf < 0.7 / flagged notes; consider also `min(ev) < 0.25`).

## 1. Rebuild the dataset / zip (local, Windows)
```
cd D:\sharvin1\Projects\Music\guitar3d
set PY=C:/Users/sharv/anaconda3/python.exe
%PY% -I tools/dataset/search.py --per 10 [--add]       # yt-dlp search -> cache/candidates.jsonl
%PY% -I tools/dataset/build.py run --jobs 4 [--after K] # download + read_tabs.py --debug (resumable, ~2 CPU-min/video)
D:/sharvin1/Projects/Music/pitch-model/venv/Scripts/python.exe tools/dataset/verify.py   # basic-pitch agreement
%PY% -I tools/dataset/build.py manifest                 # gate -> manifest.jsonl + labels/
%PY% -I tools/dataset/spot.py <id>                      # 2 random kept pages to check by eye (--record "...")
%PY% -I tools/dataset/package.py --with-audio --guitarset
```
-> `tools/dataset/out/GuitarTurtor_train.zip` (labels + manifest + audio + GuitarSet annotation.zip, ~40 MB + audio).
Without `--with-audio` you get a labels-only zip (enough for Part A, the string chooser).
`--guitarset-audio` also packs GuitarSet mono-mic audio (656 MB); otherwise the notebook downloads it from Zenodo.

## 2. Upload
Upload `GuitarTurtor_train.zip` to the root of Google Drive (`MyDrive/GuitarTurtor_train.zip`). No need to unzip:
the notebook extracts it to the Colab VM's local disk (faster than reading from Drive).

## 3a. Run the notebook (simplest)
1. Open https://colab.research.google.com -> File -> Upload notebook -> `tools/train/colab_finetune.ipynb`.
2. Runtime -> Change runtime type -> GPU (T4 is enough).
3. Run all. Allow the Drive mount when asked.
   - Part A (string/fret chooser, ~5 min): GuitarSet + lesson train videos pretrain -> priority fine-tune
     (Choo Lo + Locked Away first 60%, weight x20, replay) -> best epoch by held-out
     (mean of: Choo Lo last 40%, Locked Away last 40%, held-out lesson videos) -> leave-one-song-out check.
   - Part B (audio -> onset/frame pitch CRNN, ~30-60 min on T4, needs `--with-audio`): same recipe, selected by
     note-level onset F1 on the same held-out sets; GuitarSet player 05 = sanity test.
4. Outputs in `MyDrive/GuitarTurtor/out/`: `string_chooser.pt/.onnx`, `pitch_crnn.pt/.onnx`, `results_*.json`.
   Download into `tools/train/out/` locally.

Held-out lesson videos are picked by md5(id) (~15%, min 3) so the split is the same every run. Labels with
conf < 0.7 or any sanity flag are excluded from training and scoring.

## 3b. Run Claude Code on Colab (to iterate on the recipe)
In a Colab notebook cell (GPU runtime):
```
!curl -fsSL https://deb.nodesource.com/setup_20.x | bash - >/dev/null && apt-get -qq install -y nodejs
!npm i -g @anthropic-ai/claude-code
from google.colab import drive; drive.mount('/content/drive')
!mkdir -p /content/GT && cd /content/GT && unzip -q -o /content/drive/MyDrive/GuitarTurtor_train.zip
!cp /content/drive/MyDrive/colab_finetune.ipynb /content/GT/   # optional: upload the notebook to Drive first
```
Then open Colab's Terminal (left bar -> Terminal; Colab Pro, or use the `colab-xterm` extension:
`!pip install colab-xterm` then `%load_ext colabxterm` and `%xterm`), and run:
```
cd /content/GT && claude
```
Log in with your Anthropic account when prompted (or `export ANTHROPIC_API_KEY=...` first). Suggested prompt:
> Data is in /content/GT/GuitarTurtor/train (manifest.jsonl, labels/, audio/, guitarset/). Follow the recipe in
> colab_finetune.ipynb: pretrain on GuitarSet + lesson videos, priority fine-tune on choo-lo + locked-away
> (first 60%), select by held-out (last 40% of both songs + held-out lesson videos), export weights to
> /content/drive/MyDrive/GuitarTurtor/out. Report held-out numbers before/after fine-tune.

Copy results out to Drive before the VM disconnects (`/content` is wiped).

## 4. Licences / attribution
GuitarSet: Xi et al., ISMIR 2018, CC BY 4.0 (Zenodo 3371780). Lesson videos: rights belong to their creators;
used here for private research only; the manifest keeps ids/urls only.
