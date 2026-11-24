"""guitab E25 raw notes (onset 0.3, as the browser's runGuitab) -> work/<name>.guitab.json [[t0,t1,pitch,amp]].
run: anysong-research/venv311/Scripts/python.exe guitab_raw.py AUDIO OUT.json"""
import sys, json, numpy as np
from pathlib import Path
from guitab.audio import compute_cqt, load_audio
from guitab.inference import extract_notes, load_model, predict
from guitab.paths import MODELS, fetch
feat = compute_cqt(load_audio(Path(sys.argv[1]).resolve()))
model = load_model(str(fetch(MODELS["E25"][0])), "cpu")
tab, on = predict(model, feat, "cpu")
notes = extract_notes(tab, on, {**dict(model.decoding), "onset": 0.3})
F = 22050 / 256
def peak(n):
    f = int(round(n["onset"] * F)); c = 6 - n["string"]
    return float(on[max(0, f - 2):f + 3, c].max())
json.dump([[n["onset"], n["offset"], int(n["pitch"]), peak(n)] for n in notes], open(sys.argv[2], "w"))
print(len(notes), "notes")
