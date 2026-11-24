"""Video helpers: download (yt-dlp), audio extraction and gray frame streaming (ffmpeg)."""
import os, re, glob, shutil, subprocess, hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")

_FF_DIRS = [
    r"C:/Users/sharv/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-9.1-full_build/bin",
]
_YTDLP = [r"C:/Users/sharv/anaconda3/Scripts/yt-dlp.exe"]


def _find(name, dirs):
    p = shutil.which(name)
    if p:
        return p
    for d in dirs:
        for ext in ("", ".exe"):
            c = os.path.join(d, name + ext)
            if os.path.exists(c):
                return c
    raise FileNotFoundError(name)


def ffmpeg():
    return os.environ.get("FFMPEG") or _find("ffmpeg", _FF_DIRS)


def ffprobe():
    return os.environ.get("FFPROBE") or _find("ffprobe", _FF_DIRS)


def ytdlp():
    p = os.environ.get("YTDLP")
    if p:
        return p
    for c in _YTDLP:
        if os.path.exists(c):
            return c
    return _find("yt-dlp", [])


def is_url(s):
    return bool(re.match(r"^(https?://|ytsearch\d*:)", s))


def fetch(src, max_height=1080):
    """Return a local video path for a file or URL (downloads into cache/)."""
    if not is_url(src):
        return os.path.abspath(src)
    os.makedirs(CACHE, exist_ok=True)
    m = re.search(r"(?:v=|youtu\.be/|shorts/)([\w-]{11})", src)
    key = m.group(1) if m else hashlib.md5(src.encode()).hexdigest()[:11]
    hit = [p for p in glob.glob(os.path.join(CACHE, key + ".*")) if p.endswith((".mp4", ".mkv", ".webm"))]
    if hit:
        return hit[0]
    out = os.path.join(CACHE, key + ".%(ext)s")
    fmt = f"bv*[height<={max_height}][ext=mp4]+ba[ext=m4a]/bv*[height<={max_height}]+ba/b[height<={max_height}]/b"
    cmd = [ytdlp(), "-q", "--no-playlist", "-f", fmt, "--merge-output-format", "mp4",
           "--ffmpeg-location", os.path.dirname(ffmpeg()), "-o", out, src]
    subprocess.run(cmd, check=True)
    hit = [p for p in glob.glob(os.path.join(CACHE, key + ".*")) if p.endswith((".mp4", ".mkv", ".webm"))]
    if not hit:
        raise RuntimeError("download failed: " + src)
    return hit[0]


def probe(path):
    r = subprocess.run([ffprobe(), "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height:format=duration", "-of", "default=nw=1", path],
                       capture_output=True, text=True, check=True).stdout
    d = dict(l.split("=", 1) for l in r.split() if "=" in l)
    return int(d["width"]), int(d["height"]), float(d["duration"])


def frames(path, fps=2.0, start=None, end=None, width=None):
    """Yield (t, gray uint8 HxW) frames sampled at fps."""
    W, H, dur = probe(path)
    if width and width < W:
        H = int(round(H * width / W / 2) * 2); W = width
    t0 = start or 0.0
    cmd = [ffmpeg(), "-v", "error"]
    if start:
        cmd += ["-ss", str(start)]
    cmd += ["-i", path]
    if end:
        cmd += ["-t", str(end - t0)]
    cmd += ["-vf", f"fps={fps},scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=W * H * 4)
    n = 0
    try:
        while True:
            buf = p.stdout.read(W * H)
            if len(buf) < W * H:
                break
            yield t0 + n / fps, np.frombuffer(buf, np.uint8).reshape(H, W)
            n += 1
    finally:
        p.stdout.close(); p.wait()


def audio(path, sr=22050):
    """Decode mono float32 audio via ffmpeg."""
    cmd = [ffmpeg(), "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy(), sr
