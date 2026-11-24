"""Build a Colab-ready zip of the lesson dataset. LOCAL/PRIVATE: audio only with --with-audio, never commit or share the zip.

usage: python -I package.py [--with-audio] [--guitarset] [--guitarset-audio] [--all]
  -> tools/dataset/out/GuitarTurtor_train.zip containing
     GuitarTurtor/train/manifest.jsonl          (pass items only unless --all)
     GuitarTurtor/train/labels/<id>.json        (site score JSON: t,q,d,n:[[string,fret]],conf,cn,flags)
     GuitarTurtor/train/audio/<id>.<ext>        (only with --with-audio; m4a/mp3/wav, Locked Away as m4a from its mp4)
     GuitarTurtor/train/guitarset/POINTER.txt   (+ annotation.zip with --guitarset, + audio_mono-mic.zip with --guitarset-audio)
Unzip into Google Drive MyDrive/ so the notebook finds MyDrive/GuitarTurtor/train/.
"""
import os, sys, json, shutil, zipfile, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
GS = os.path.join(ROOT, "tools", "train", "data", "guitarset")
FF = r"C:/Users/sharv/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-9.1-full_build/bin/ffmpeg.exe"


def main(a):
    out = os.path.join(HERE, "out"); base = os.path.join(out, "GuitarTurtor", "train")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(os.path.join(base, "labels"))
    rows = [json.loads(l) for l in open(os.path.join(HERE, "manifest.jsonl"), encoding="utf-8")]
    if "--all" not in a:
        rows = [r for r in rows if r["status"] == "pass"]
    keep = []
    for r in rows:
        if r.get("label"):
            shutil.copy(os.path.join(HERE, r["label"]), os.path.join(base, "labels"))
        r = {k: v for k, v in r.items() if k != "audio_local"}
        if "--with-audio" in a and r["status"] == "pass":
            src = os.path.join(ROOT, next(x for x in rows if x["id"] == r["id"])["audio_local"])
            os.makedirs(os.path.join(base, "audio"), exist_ok=True)
            if src.endswith(".mp4"):     # Locked Away: strip video, keep audio stream
                dst = os.path.join(base, "audio", r["id"] + ".m4a")
                subprocess.run([FF, "-v", "error", "-y", "-i", src, "-vn", "-c:a", "copy", dst], check=True)
            elif os.path.exists(src):
                dst = os.path.join(base, "audio", r["id"] + os.path.splitext(src)[1]); shutil.copy(src, dst)
            else:
                dst = None
            r["audio"] = f"audio/{os.path.basename(dst)}" if dst else None
        keep.append(r)
    with open(os.path.join(base, "manifest.jsonl"), "w", encoding="utf-8") as f:
        for r in keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    g = os.path.join(base, "guitarset"); os.makedirs(g)
    open(os.path.join(g, "POINTER.txt"), "w").write(
        "GuitarSet (Xi et al., ISMIR 2018), CC BY 4.0 - attribution required. Zenodo record 3371780.\n"
        "Local copy: tools/train/data/guitarset/{annotation.zip, audio_mono-mic.zip}.\n"
        "If annotation.zip is not next to this file, the notebook downloads it from\n"
        "https://zenodo.org/records/3371780/files/annotation.zip (and audio_mono-mic.zip for the pitch stage).\n")
    for flag, fn in (("--guitarset", "annotation.zip"), ("--guitarset-audio", "audio_mono-mic.zip")):
        if flag in a and os.path.exists(os.path.join(GS, fn)):
            shutil.copy(os.path.join(GS, fn), g)
    zp = os.path.join(out, "GuitarTurtor_train.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for dp, _, fs in os.walk(os.path.join(out, "GuitarTurtor")):
            for fn in fs:
                p = os.path.join(dp, fn); z.write(p, os.path.relpath(p, out))
    print(f"wrote {zp} ({os.path.getsize(zp)/1e6:.1f} MB): {len(keep)} items, "
          f"{sum(r.get('n_notes', 0) for r in keep)} notes, audio={'--with-audio' in a}")


if __name__ == "__main__":
    main(sys.argv[1:])
