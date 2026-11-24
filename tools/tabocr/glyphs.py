"""Character classifier for tab glyphs: synthetic font templates + Tesseract votes + per-video clustering.

Identical glyphs rendered in one video look (nearly) identical, so components are clustered and each
cluster gets the label with the most evidence summed over its members (Tesseract reading of each member +
nearest synthetic templates rendered from system fonts)."""
import os, glob
import numpy as np
from scipy import ndimage as ndi
from PIL import Image, ImageDraw, ImageFont

CHARS = "0123456789hpbrx/\\~()"
HERE = os.path.dirname(os.path.abspath(__file__))
FONT_NAMES = ["arial.ttf", "arialbd.ttf", "ARIALN.TTF", "consola.ttf", "consolab.ttf", "cour.ttf", "courbd.ttf",
              "verdana.ttf", "verdanab.ttf", "tahoma.ttf", "tahomabd.ttf", "segoeui.ttf", "segoeuib.ttf",
              "calibri.ttf", "calibrib.ttf", "times.ttf", "timesbd.ttf", "lucon.ttf", "LTYPE.TTF", "LTYPEB.TTF",
              "cambriab.ttf", "georgia.ttf", "trebuc.ttf", "trebucbd.ttf", "micross.ttf", "FRAMDCN.TTF",
              "DejaVuSans.ttf", "DejaVuSansMono.ttf", "DejaVuSans-Bold.ttf", "LiberationMono-Regular.ttf",
              "LiberationSans-Regular.ttf", "seguisb.ttf", "corbel.ttf", "candara.ttf", "constan.ttf", "malgun.ttf"]
FONT_DIRS = [r"C:/Windows/Fonts", "/usr/share/fonts", "/Library/Fonts", "/System/Library/Fonts"]
S = 24


def feature(mask):
    """Shape feature of one glyph mask (bool HxW, tight or not)."""
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return None
    m = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = m.shape
    side = max(h, w)
    sq = np.zeros((side, side), np.float32)
    sq[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = m
    img = Image.fromarray((sq * 255).astype(np.uint8)).resize((S, S), Image.BILINEAR)
    f = ndi.gaussian_filter(np.asarray(img, np.float32) / 255, 1.0).ravel()
    f = np.concatenate([f, [3.0 * min(2.0, w / h)]])
    f = f - f.mean()
    return f / (np.linalg.norm(f) + 1e-6)


def _font_paths():
    out = []
    for n in FONT_NAMES:
        for d in FONT_DIRS:
            p = os.path.join(d, n)
            if os.path.exists(p):
                out.append(p); break
    return out


def synth_set():
    cache = os.path.join(HERE, "cache", "glyph_templates.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return z["F"], list(z["L"])
    F, L = [], []
    for fp in _font_paths():
        for size in (14, 20, 44):
            try:
                font = ImageFont.truetype(fp, size)
            except Exception:
                continue
            for ch in CHARS:
                im = Image.new("L", (size * 3, size * 3), 0)
                ImageDraw.Draw(im).text((size, size // 2), ch, fill=255, font=font)
                a = np.asarray(im)
                if size < 40:   # small renders, upsampled like a video frame would be
                    a = np.asarray(Image.fromarray(a).resize((a.shape[1] * 3, a.shape[0] * 3), Image.BILINEAR))
                for thr in (100, 160):
                    m = a > thr
                    lab, n = ndi.label(m, structure=np.ones((3, 3), bool))
                    if n == 0:
                        continue
                    f = feature(m)
                    if f is not None:
                        F.append(f); L.append(ch)
    F = np.stack(F).astype(np.float32)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez_compressed(cache, F=F, L=np.array(L))
    return F, L


def classify(comps, tess_labels, log=None, k=9):
    """comps: list of features; tess_labels: list of char or None. Returns list of labels."""
    if not comps:
        return []
    SF, SL = synth_set()
    X = np.stack(comps).astype(np.float32)
    idx = {c: i for i, c in enumerate(CHARS)}
    E = np.zeros((len(X), len(CHARS)), np.float32)
    # synthetic templates: similarity-weighted top-k vote
    sims = X @ SF.T
    top = np.argsort(-sims, axis=1)[:, :k]
    for i in range(len(X)):
        w = np.maximum(sims[i, top[i]], 0) ** 4
        for j, ww in zip(top[i], w):
            E[i, idx[SL[j]]] += ww / (w.sum() + 1e-9)
    # tesseract
    for i, t in enumerate(tess_labels):
        if t in idx:
            E[i, idx[t]] += 0.9
    # cluster near-identical glyphs of this video (union-find on high similarity)
    V = X @ X.T
    parent = list(range(len(X)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    ii, jj = np.nonzero(np.triu(V > 0.93, 1))
    for a, b in zip(ii, jj):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    groups = {}
    for i in range(len(X)):
        groups.setdefault(find(i), []).append(i)
    out = [None] * len(X)
    changed = 0
    for g in groups.values():
        lab = CHARS[int(np.argmax(E[g].sum(0)))]
        for i in g:
            out[i] = lab
            if tess_labels[i] is not None and tess_labels[i] != lab:
                changed += 1
    if log:
        log(f"  glyphs: {len(X)} characters in {len(groups)} clusters; {changed} Tesseract readings corrected")
    return out
