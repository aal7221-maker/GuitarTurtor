"""(Superseded by tools/dataset/package.py: manifest + lesson-video labels + priority songs.) Build GuitarTurtor/train/ folder + zip for Google Drive. LOCAL ONLY - audio is the user's private data, never commit.
usage: python package_data.py [--with-audio] [--guitarset]
  out/GuitarTurtor/train/pairs/<song>/{tab.json,meta.json[,audio]}; guitarset/annotation.zip (CC BY 4.0) if --guitarset
Audio is included only with --with-audio AND only if the file already exists locally."""
import os, sys, json, shutil, zipfile
HERE = os.path.dirname(os.path.abspath(__file__))
SONGS = {"choo-lo": dict(tab=r"D:\sharvin1\Projects\Music\guitar3d\data\score-choo-lo.json", audio=r"D:\sharvin1\Projects\Music\guitar3d\audio\choo-lo.mp3", until=38.7),
         "locked-away": dict(tab=r"D:\sharvin1\Projects\Music\guitar3d\data\score-locked-away.json", audio=os.path.join(HERE, "work", "locked.wav"), until=1e9)}
base = os.path.join(HERE, "out"); root = os.path.join(base, "GuitarTurtor", "train")
shutil.rmtree(base, ignore_errors=True)
for n, s in SONGS.items():
    d = os.path.join(root, "pairs", n); os.makedirs(d)
    shutil.copy(s["tab"], os.path.join(d, "tab.json"))
    json.dump({"until": s["until"], "exclude": "conf<0.7 or flags"}, open(os.path.join(d, "meta.json"), "w"))
    if "--with-audio" in sys.argv and os.path.exists(s["audio"]):
        shutil.copy(s["audio"], os.path.join(d, os.path.basename(s["audio"])))
if "--guitarset" in sys.argv:
    g = os.path.join(HERE, "data", "guitarset", "annotation.zip")
    if os.path.exists(g):
        os.makedirs(os.path.join(root, "guitarset")); shutil.copy(g, os.path.join(root, "guitarset"))
zp = os.path.join(base, "GuitarTurtor_train.zip")
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for dp, _, fs in os.walk(os.path.join(base, "GuitarTurtor")):
        for f in fs:
            p = os.path.join(dp, f); z.write(p, os.path.relpath(p, base))
print("wrote", zp, "- unzip into Drive MyDrive/ so you get GuitarTurtor/train/")
