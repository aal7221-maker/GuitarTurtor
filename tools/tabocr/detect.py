"""Find guitar-tab staves (6 parallel horizontal lines) in frames and track which tab page is visible when."""
import numpy as np
from scipy import ndimage as ndi


def ink_maps(gray):
    """Return (dark_ink, light_ink) boolean maps relative to a local background."""
    g = gray.astype(np.float32)
    bg = ndi.uniform_filter(g, size=31)
    return g < bg - 28, g > bg + 28


def _runs(mask):
    """Start/end indices of True runs in a 1-D bool array."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def find_lines(ink, min_frac=0.12):
    H, W = ink.shape
    gap = max(3, W // 300)
    closed = ndi.binary_closing(ink, structure=np.ones((1, gap), bool))
    cov = closed.mean(1)
    k = max(3, H // 270)
    lo = np.minimum(np.roll(cov, k), np.roll(cov, -k))
    lo2 = np.minimum(np.roll(cov, k + 2), np.roll(cov, -k - 2))
    cand = (cov >= min_frac) & (cov >= 2.0 * np.maximum(lo, lo2 * 0.8) + 0.02)
    cand[:k + 2] = cand[-k - 2:] = False
    lines = []
    s, e = _runs(cand)
    big = max(gap * 8, W // 40)
    for a, b in zip(s, e):
        if b - a > max(6, H // 120):
            continue
        rows = closed[a:b].any(0)
        rows = ndi.binary_closing(rows, structure=np.ones(big, bool))
        rs, re_ = _runs(rows)
        if not len(rs):
            continue
        i = np.argmax(re_ - rs)
        x0, x1 = int(rs[i]), int(re_[i])
        if x1 - x0 < min_frac * W:
            continue
        w = cov[a:b]
        y = float((np.arange(a, b) * w).sum() / w.sum())
        lines.append(dict(y=y, y0=a, y1=b, x0=x0, x1=x1, cov=float(cov[a:b].max())))
    return lines


def group_staves(lines, H):
    """Groups of 6 equally spaced, x-overlapping lines."""
    out = []
    n = len(lines)
    ys = np.array([l["y"] for l in lines])
    for i in range(n):
        for j in range(i + 1, n):
            d = (ys[j] - ys[i]) / 5.0
            if d < 5 or d > H / 7:
                continue
            idx = [i]
            ok = True
            for k in range(1, 6):
                t = ys[i] + k * d
                c = np.flatnonzero(np.abs(ys - t) < 0.15 * d)
                if not len(c):
                    ok = False; break
                idx.append(int(c[np.argmin(np.abs(ys[c] - t))]))
            if not ok:
                continue
            L = [lines[q] for q in idx]
            x0 = max(l["x0"] for l in L); x1 = min(l["x1"] for l in L)
            wmax = max(l["x1"] - l["x0"] for l in L)
            if x1 - x0 < 0.7 * wmax:
                continue
            yy = np.array([l["y"] for l in L])
            sp = np.diff(yy)
            if sp.std() > 0.12 * sp.mean():
                continue
            # a 7th line at the same spacing means this is not a tab staff
            if any(np.abs(ys - (yy[0] - sp.mean())) < 0.15 * sp.mean()) and \
               any(np.abs(ys - (yy[-1] + sp.mean())) < 0.15 * sp.mean()):
                continue
            pred = yy[0] + np.arange(6) * (yy[5] - yy[0]) / 5
            rms = float(np.sqrt(np.mean((yy - pred) ** 2)))
            cv = np.array([l["cov"] for l in L])
            score = (x1 - x0) * cv.mean() * np.exp(-(rms / max(1.0, 0.05 * d)) ** 2) * np.exp(-3 * cv.std() / cv.mean())
            out.append(dict(ys=[float(v) for v in yy], sp=float(sp.mean()), x0=int(np.median([l["x0"] for l in L])),
                            x1=int(np.median([l["x1"] for l in L])), score=float(score),
                            th=float(np.median([l["y1"] - l["y0"] for l in L]))))
    out.sort(key=lambda s: -s["score"])
    keep = []
    for s in out:
        if all(s["ys"][5] < k["ys"][0] - 0.5 * k["sp"] or s["ys"][0] > k["ys"][5] + 0.5 * k["sp"] for k in keep):
            keep.append(s)
    return keep


def find_staves(gray):
    """Return (staves, polarity). Picks the main tab: widest staves (multi-system pages kept), top to bottom."""
    H, W = gray.shape
    ds = 2 if W >= 1600 else 1
    best, pol = [], None
    for p, ink in zip(("dark", "light"), ink_maps(gray)):
        if ds > 1:   # max-pool so 1-px staff lines survive
            h2, w2 = H // ds, W // ds
            ink = ink[:h2 * ds, :w2 * ds].reshape(h2, ds, w2, ds).any((1, 3))
        st = group_staves(find_lines(ink), ink.shape[0])
        for s in st:
            s["ys"] = [y * ds + (ds - 1) / 2 for y in s["ys"]]; s["sp"] *= ds; s["th"] *= ds
            s["x0"] *= ds; s["x1"] *= ds
        if st and (not best or max(s["score"] for s in st) > max(s["score"] for s in best)):
            best, pol = st, p
    if not best:
        return [], None
    main = max(best, key=lambda s: s["x1"] - s["x0"])
    wm = main["x1"] - main["x0"]
    # other systems of the same page share the left edge and width (a second, different tab elsewhere is ignored)
    best = [s for s in best if abs(s["x0"] - main["x0"]) < 0.05 * W and s["x1"] - s["x0"] >= 0.85 * wm
            and abs(s["sp"] - main["sp"]) < 0.15 * main["sp"] and wm >= 0.25 * W]
    best.sort(key=lambda s: s["ys"][0])
    return best, pol


def region(staves, shape):
    H, W = shape
    sp = max(s["sp"] for s in staves)
    y0 = int(max(0, min(s["ys"][0] for s in staves) - 1.0 * sp))
    y1 = int(min(H, max(s["ys"][5] for s in staves) + 1.0 * sp))
    x0 = int(max(0, min(s["x0"] for s in staves) - 1.5 * sp))
    x1 = int(min(W, max(s["x1"] for s in staves) + 0.5 * sp))
    return y0, y1, x0, x1


def signature(gray, staves, pol):
    out = []
    for s in staves:
        y0 = int(max(0, s["ys"][0] - 0.5 * s["sp"])); y1 = int(min(gray.shape[0], s["ys"][5] + 0.5 * s["sp"]))
        crop = gray[y0:y1, s["x0"]:s["x1"]].astype(np.float32)
        bg = ndi.uniform_filter(crop, 31)
        ink = (crop < bg - 28) if pol == "dark" else (crop > bg + 28)
        h, w = ink.shape
        gh, gw = 32, 384
        ys = (np.arange(gh + 1) * h / gh).astype(int); xs = (np.arange(gw + 1) * w / gw).astype(int)
        c = np.add.reduceat(np.add.reduceat(ink.astype(np.float32), ys[:-1], 0), xs[:-1], 1)
        out.append(c / np.outer(np.diff(ys), np.diff(xs)).clip(1))
    return np.concatenate(out, 0)


def same_geom(a, b, H):
    if len(a) != len(b):
        return False
    return all(abs(p["ys"][0] - q["ys"][0]) < 0.02 * H and abs(p["sp"] - q["sp"]) < 0.1 * p["sp"]
               and abs(p["x0"] - q["x0"]) < 0.05 * (p["x1"] - p["x0"]) for p, q in zip(a, b))


def sig_diff(a, b):
    if a.shape != b.shape:
        return 1.0
    return float(np.abs(a - b).sum() / max(1e-6, 0.5 * (a.sum() + b.sum())))


def track_pages(frame_iter, diff_thr=0.18, keep=7, log=None):
    """Return (pages, intervals).
    pages: list of dict(staves, pol, crops=[full gray frames], ts, shape)
    intervals: list of (page_id, t_start, t_end) in time order."""
    pages, intervals = [], []
    cand = None          # (obs, frame, t) of an unconfirmed new page
    dt = 0.5; last_t = None
    rng = np.random.default_rng(0)

    def mark(pid, t0, t1):
        if intervals and intervals[-1][0] == pid and t0 - intervals[-1][2] <= 3.01 * dt:
            intervals[-1][2] = t1
        else:
            intervals.append([pid, t0, t1])
            if log: log(f"  t={t0:7.1f}s page {pid}")

    def add_crop(p, g, t):
        if len(p["crops"]) < keep:
            p["crops"].append(g); p["ts"].append(t)
        elif rng.random() < 0.3:
            i = int(rng.integers(0, keep)); p["crops"][i] = g; p["ts"][i] = t

    for t, g in frame_iter:
        if last_t is not None: dt = t - last_t
        last_t = t
        st, pol = find_staves(g)
        if not st:
            cand = None; continue
        sig = signature(g, st, pol)
        pid = None
        cache = {}
        for k, p in enumerate(pages):
            if p["pol"] == pol and same_geom(p["staves"], st, g.shape[0]):
                key = id(p["staves"])
                if key not in cache:   # crop this frame with the page's own geometry so the grids line up
                    cache[key] = signature(g, p["staves"], pol)
                if sig_diff(p["sig"], cache[key]) < diff_thr:
                    pid = k; break
        if pid is None:
            # require 2 consecutive agreeing frames before accepting a new page (skips crossfades)
            if cand is not None and same_geom(cand[0][0], st, g.shape[0]) and                sig_diff(cand[0][2], signature(g, cand[0][0], pol)) < diff_thr:
                pages.append(dict(staves=st, pol=pol, sig=sig, crops=[cand[1], g], ts=[cand[2], t], shape=g.shape))
                pid = len(pages) - 1
                mark(pid, cand[2], t)
                cand = None
            else:
                cand = ((st, pol, sig), g, t)
            continue
        cand = None
        add_crop(pages[pid], g, t)
        mark(pid, t, t)
    for p in pages:
        p.pop("sig", None)
    return pages, [tuple(iv) for iv in intervals]
