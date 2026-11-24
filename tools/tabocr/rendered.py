"""Rendered (Guitar Pro / Songsterr style) tab reader.

Fret digits are found as connected components of a strict ink mask (digits are much darker than the thin
staff lines) that are centred ON a string line and have digit height; clef, time signature, rests, stems,
beams, bar numbers, bend arrows and labels fail the position/size tests.  Components of the whole video are
clustered by shape, each cluster is labelled once by evidence (Tesseract on the cluster's members + synthetic
font templates), then digits are joined into frets.  Slur arcs become hammer/pull/tie, diagonal marks slides."""
import numpy as np
from scipy import ndimage as ndi
import glyphs


def _runs(mask):
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def strict_ink(gray, pol, ys, x0, x1, sp):
    """Bool ink map of the staff crop + (y0, x0) offset: dark glyph pixels, not the thin lighter staff lines."""
    g = gray.astype(np.float32)
    if pol == "light":
        g = 255 - g
    y0 = int(max(0, ys[0] - 1.35 * sp)); y1 = int(min(g.shape[0], ys[5] + 1.35 * sp)) + 1
    xa = int(max(0, x0 - 2 * sp)); xb = int(min(g.shape[1], x1 + 2))
    c = g[y0:y1, xa:xb]
    ln = float(np.median([np.percentile(g[int(y) - 1:int(y) + 2, x0:x1], 30) for y in ys]))
    inner = g[int(max(0, ys[0] - 0.4 * sp)):int(ys[5] + 0.4 * sp) + 1, int(x0):int(x1)]   # not the video's black bars
    bg = float(np.percentile(inner, 60))
    dark = float(np.percentile(inner, 0.5))
    if ln - dark > 0.25 * (bg - dark) and ln < bg - 12:      # lines clearly lighter than text
        thr = dark + 0.8 * (ln - dark)
    else:                          # lines as dark as text: plain threshold, lines are removed afterwards
        thr = dark + 0.7 * (bg - dark)
    return c < thr, y0, xa, c, bg


def _digit_like(d):
    return d["h"] >= 0.5 and d["w"] <= 0.75 and d["fill"] >= 0.3


def _find_line(m, sp):
    """Longest straight thin stroke (a slide line) in mask m: returns (pixel mask of the stroke, slope) or None."""
    rr, cc = np.nonzero(m)
    if len(rr) < 12:
        return None
    best = None
    for deg in list(range(-62, -5, 2)) + list(range(6, 64, 2)):
        t = np.deg2rad(deg)
        # direction (cos t, sin t) in (x, y); normal distance
        rho = -cc * np.sin(t) + rr * np.cos(t)
        along = cc * np.cos(t) + rr * np.sin(t)
        bins = np.round(rho / 1.5).astype(int)
        for b in np.unique(bins):
            sel = np.abs(rho - b * 1.5) <= 1.6
            if sel.sum() < 0.9 * sp:
                continue
            a = np.sort(along[sel])
            # longest run with no gap > 2.5 px
            br = np.flatnonzero(np.diff(a) > 2.5)
            starts = np.concatenate([[0], br + 1]); ends = np.concatenate([br, [len(a) - 1]])
            L = a[ends] - a[starts]
            i = int(np.argmax(L))
            if L[i] >= 1.1 * sp and (best is None or L[i] > best[0]):
                best = (L[i], deg, rho, along, a[starts[i]], a[ends[i]], b * 1.5)
    if best is None:
        return None
    L, deg, rho, along, lo, hi, r0 = best
    sel = (np.abs(rho - r0) <= 1.7) & (along >= lo - 1) & (along <= hi + 1)
    stroke = np.zeros_like(m)
    stroke[rr[sel], cc[sel]] = True
    return stroke, (1 if deg < 0 else -1)


def _straight(m):
    """Angle (deg, + = rising to the right) if mask m is a thin straight stroke, else None."""
    rr, cc = np.nonzero(m)
    if len(rr) < 6:
        return None
    X = np.stack([cc - cc.mean(), -(rr - rr.mean())]).astype(float)
    ev, vec = np.linalg.eigh(X @ X.T / len(rr))
    length = 4 * np.sqrt(ev[1]); thick = 4 * np.sqrt(max(ev[0], 0))
    if thick > 0.2 * length or thick > 4.5:
        return None
    return float(np.degrees(np.arctan2(vec[1, 1], vec[0, 1])) if vec[0, 1] > 0 else np.degrees(np.arctan2(-vec[1, 1], -vec[0, 1])))


def staff_components(gray, staff, pol, ys, bars, ascii_mode=False, minx=None):
    sp = staff["sp"]; x0, x1 = staff["x0"], staff["x1"]
    ink, oy, ox, gc, bg = strict_ink(gray, pol, ys, x0, x1, sp)
    for b in bars:                              # bar lines
        c = int(b) - ox
        ink[:, max(0, c - 2):c + 3] = False
    ysa = np.array(ys, float)
    orig = ink.copy()
    erased = np.zeros_like(ink)
    for y in ys:                                # staff lines that are as dark as text on some pixel rows
        rc = int(round(y - oy))
        for r in range(rc - 3, rc + 4):
            if 0 <= r < ink.shape[0]:
                for a, b in _runs(orig[r]):
                    if b - a > 1.3 * sp:
                        ink[r, a:b] = False; erased[r, a:b] = True
        ra, rb = rc - 4, rc + 4
        if ra - 3 >= 0 and rb + 3 < ink.shape[0]:   # give a digit back its pixels where it has ink above and below
            up = orig[ra - 3:ra + 1].any(0); dn = orig[rb:rb + 4].any(0)
            ok = ndi.binary_dilation(up, structure=np.ones(5, bool)) & ndi.binary_dilation(dn, structure=np.ones(5, bool))
            for r in range(rc - 3, rc + 4):
                ink[r] |= erased[r] & ok
    comps, arcs = [], []
    glue = ndi.binary_dilation(ink, structure=np.ones((3, 3), bool), iterations=1 if ascii_mode else 2) if sp >= 40 else ink   # thin fonts break strokes
    lab, n = ndi.label(glue, structure=np.ones((3, 3), bool))
    work = []
    for k, sl in enumerate(ndi.find_objects(lab)):
        work.append(((sl[0].start, sl[1].start), (lab[sl] == k + 1) & ink[sl], 0))
    while work:
        (r0, c0), m, depth = work.pop()
        rows = np.flatnonzero(m.any(1)); cols = np.flatnonzero(m.any(0))
        if not len(rows):
            continue
        m = m[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
        r0 += rows[0]; c0 += cols[0]
        h, w = m.shape
        area = int(m.sum())
        if area < max(5, 0.008 * sp * sp) or (h < 3 and w < 3):
            continue
        cy = oy + r0 + (h - 1) / 2; cx = ox + c0 + (w - 1) / 2
        left = (minx if minx is not None else x0) if ascii_mode else x0 + min(2.6 * sp, 70)
        if cx < left or cx > x1 + 0.5 * sp:     # left: TAB clef / time signature / string labels
            continue
        s = int(np.argmin(np.abs(ysa - cy)))
        d = dict(x0=ox + c0, x1=ox + c0 + w, y0=oy + r0, y1=oy + r0 + h, cx=cx, cy=cy, h=h / sp, w=w / sp, s=s,
                 dy=(cy - ysa[s]) / sp, mask=m, area=area, fill=area / float(w * h))
        d["core"] = float(gc[r0:r0 + h, c0:c0 + w][m].min())
        inside = ysa[0] - 0.55 * sp <= cy <= ysa[5] + 0.55 * sp
        if not ascii_mode and h / sp <= 0.6 and w / sp >= 0.6 and w >= 2.0 * h and d["fill"] < 0.9:   # shallow arc (slur / tie)
            if ysa[0] - 1.4 * sp <= cy <= ysa[5] + 1.4 * sp:
                arcs.append(d)
            continue
        if not inside:
            continue
        # a slide line touching a digit: cut the straight stroke out and treat the rest again
        if not ascii_mode and depth < 2 and d["w"] >= 0.7 and d["h"] <= 1.3 and abs(d["dy"]) <= 0.6:
            fl = _find_line(m, sp)
            if fl is not None and np.ptp(np.nonzero(fl[0])[0]) >= 3:
                stroke, sgn = fl
                sr, sc_ = np.nonzero(stroke)
                sd = dict(d)
                sd.update(mask=stroke[sr.min():sr.max() + 1, sc_.min():sc_.max() + 1], kind="slide",
                          ch="/" if sgn > 0 else "\\", x0=ox + c0 + sc_.min(), x1=ox + c0 + sc_.max() + 1,
                          y0=oy + r0 + sr.min(), y1=oy + r0 + sr.max() + 1)
                sd["cx"] = (sd["x0"] + sd["x1"] - 1) / 2; sd["cy"] = (sd["y0"] + sd["y1"] - 1) / 2
                sd["w"] = (sd["x1"] - sd["x0"]) / sp; sd["h"] = (sd["y1"] - sd["y0"]) / sp
                sd["s"] = int(np.argmin(np.abs(ysa - sd["cy"])))
                comps.append(sd)
                rest = m & ~ndi.binary_dilation(stroke, iterations=1)
                # inside the digit's own columns keep every pixel: the line ends there and must not bite the digit
                ext = np.array([(np.flatnonzero(rest[:, j])[-1] - np.flatnonzero(rest[:, j])[0] + 1) if rest[:, j].any() else 0
                                for j in range(w)])
                if ext.max() > 0:
                    for a_, b_ in _runs(ext >= 0.55 * ext.max()):     # each digit separately
                        rows_ = np.flatnonzero(rest[:, a_:b_].any(1))
                        if len(rows_):
                            r_a, r_b = rows_.min(), rows_.max() + 1
                            rest[r_a:r_b, a_:b_] = m[r_a:r_b, a_:b_]
                l2, n2 = ndi.label(rest, structure=np.ones((3, 3), bool))
                for k2, sl2 in enumerate(ndi.find_objects(l2)):
                    work.append(((r0 + sl2[0].start, c0 + sl2[1].start), l2[sl2] == k2 + 1, depth + 1))
                continue
        if not ascii_mode and depth < 3 and 0.7 <= d["w"] <= 1.4 and 0.6 <= d["h"] <= 1.0 and abs(d["dy"]) <= 0.3:
            # digit with a thin tail (arc end, slide stub): keep the columns that are as tall as the digit
            ext = np.array([ (np.flatnonzero(m[:, j])[-1] - np.flatnonzero(m[:, j])[0] + 1) if m[:, j].any() else 0 for j in range(w)])
            good = ext >= 0.45 * ext.max()
            runs = _runs(good)
            if len(runs) >= 1:
                a, b = max(runs, key=lambda r: r[1] - r[0])
                if (b - a) < 0.8 * w and (b - a) >= 0.25 * sp:
                    sub = np.zeros_like(m); sub[:, a:b] = m[:, a:b]
                    work.append(((r0, c0), sub, 3))
                    continue
        ang = _straight(m)
        if not ascii_mode and ang is not None and d["h"] >= 0.15 and abs(d["dy"]) <= 0.6 and 8 <= abs(ang) <= 72 and                 max(d["w"], d["h"]) >= 0.35:
            d["kind"] = "slide"; d["ch"] = "/" if ang > 0 else "\\"; comps.append(d)
            continue
        if depth < 2 and (d["h"] > 0.97 or d["dy"] > 0.3 or d["dy"] < -0.3) and d["w"] <= 2.2 and 0.97 < d["h"] <= 1.7                 and d["area"] > 0.15 * sp * sp:
            # digit touched by an arc / stem: cut a window around each string line it crosses
            done = False
            for ly in ysa:
                a = int(round(ly - 0.45 * sp - oy - r0)); b = int(round(ly + 0.45 * sp - oy - r0)) + 1
                if a < -2 or b > h + 2 or b - a < 0.6 * sp:
                    continue
                a = max(a, 0); b = min(b, h)
                sub = np.zeros_like(m); sub[a:b] = m[a:b]
                if sub.sum() < 0.1 * sp * sp:
                    continue
                l2, n2 = ndi.label(sub, structure=np.ones((3, 3), bool))
                for k2, sl2 in enumerate(ndi.find_objects(l2)):
                    work.append(((r0 + sl2[0].start, c0 + sl2[1].start), l2[sl2] == k2 + 1, 2))
                done = True
            if done:
                continue
        if ascii_mode:
            if abs(d["dy"]) <= 0.4 and 0.3 <= d["h"] <= 0.97 and d["w"] <= 1.1:
                d["kind"] = "glyph"
                if d["w"] > WIDE_ASCII and d["h"] >= 0.3:
                    d["wide"] = True          # touching characters ("4~", "6h"): cut at the emptiest column
                comps.append(d)
            continue
        if abs(d["dy"]) > 0.3 or d["h"] > 0.97 or d["h"] < 0.42 or d["w"] > 1.1:
            if d["w"] > 1.1 and 0.42 <= d["h"] <= 0.97 and abs(d["dy"]) <= 0.3:
                d["wide"] = True; d["kind"] = "glyph"; comps.append(d)     # merged glyphs
            continue
        if d["h"] > 0.9 and d["w"] > 0.58:       # rests (an eighth rest reads as a 7)
            continue
        d["kind"] = "glyph"
        comps.append(d)
    cores = sorted(c["core"] for c in comps if c["kind"] == "glyph" and "core" in c)
    if cores:
        ref = bg - np.mean(cores[:max(1, int(0.6 * len(cores)))])       # typical (darkest 60%) digit contrast
        for c in comps:
            if "core" in c:
                c["contrast"] = float(np.clip((bg - c["core"]) / max(ref, 1.0), 0, 1.5))
    return comps, arcs


def split_wide(comps, med_w):
    """Split a merged component (e.g. '/6') into pieces at its emptiest columns."""
    out = []
    for c in comps:
        if not c.get("wide"):
            out.append(c); continue
        m = c["mask"]; col = m.sum(0).astype(float)
        w = m.shape[1]
        lo, hi = int(0.25 * w), int(0.75 * w)
        cut = lo + int(np.argmin(col[lo:hi + 1])) if hi > lo else w // 2
        for a, b in ((0, cut), (cut, w)):
            sub = m[:, a:b]
            rows = np.flatnonzero(sub.any(1)); cols = np.flatnonzero(sub.any(0))
            if not len(rows) or len(cols) < 2:
                continue
            sub = sub[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
            d = dict(c); d.pop("wide"); d["kind"] = "glyph"
            d.update(mask=sub, x0=c["x0"] + a + cols[0], x1=c["x0"] + a + cols[-1] + 1,
                     y0=c["y0"] + rows[0], y1=c["y0"] + rows[-1] + 1)
            d["cx"] = (d["x0"] + d["x1"] - 1) / 2; d["cy"] = (d["y0"] + d["y1"] - 1) / 2
            spx = (c["x1"] - c["x0"]) / c["w"]
            d["w"] = (d["x1"] - d["x0"]) / spx; d["h"] = (d["y1"] - d["y0"]) / spx
            d["area"] = int(sub.sum()); d["fill"] = d["area"] / float(sub.shape[0] * sub.shape[1])
            out.append(d)
    return out


# ---------------------------------------------------------------- classification

import os, re, subprocess, tempfile
from PIL import Image

CH = glyphs.CHARS      # 0-9 h p b r x / \ ~ ( )
TESS_WL = "0123456789x/\~()"
TESS_WL_ASCII = "0123456789xhpbr/\~"
WIDE_ASCII = 0.5
ALLOW = np.array([1.0 if (c in TESS_WL and c not in "()") else 0.0 for c in CH], np.float32)   # ghost parens: fragments of broken digits look like them
ALLOW_ASCII = np.array([1.0 if c in TESS_WL_ASCII else 0.0 for c in CH], np.float32)


def _tess_single(img_list, ocr_mod, wl=None):
    """One Tesseract run (psm 10) per image. Returns list of chars ('' if none)."""
    out = []
    exe = ocr_mod.tesseract()
    for im in img_list:
        fd, path = tempfile.mkstemp(suffix=".png"); os.close(fd)
        try:
            Image.fromarray(im).save(path)
            r = subprocess.run([exe, path, "stdout", "--psm", "10", "-c", "tessedit_char_whitelist=" + (wl or TESS_WL)],
                               capture_output=True, text=True).stdout.strip()
        finally:
            os.remove(path)
        out.append(r[:1] if len(r) == 1 or (r and len(set(r)) == 1) else (r[:1] if r else ""))
    return out


def _glyph_img(mask, pad=14, target=56):
    ys, xs = np.nonzero(mask)
    m = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = m.shape
    sc = target / max(h, 1)
    im = Image.fromarray(np.where(m, 0, 255).astype(np.uint8)).resize((max(3, int(w * sc)), target), Image.LANCZOS)
    a = np.full((im.height + 2 * pad, im.width + 2 * pad), 255, np.uint8)
    a[pad:pad + im.height, pad:pad + im.width] = np.array(im)
    return a


def classify_comps(comps, ocr_mod, log=None, sim=0.93, ntess=3, ascii_mode=False):
    """Label each component; returns (labels, conf) with conf in 0..1 from the cluster's evidence margin."""
    n = len(comps)
    if not n:
        return [], [], []   # (labels, conf, groups): was 2 values -> ValueError in ocr.read_pages
    feats = [glyphs.feature(c["mask"]) for c in comps]
    X = np.stack(feats).astype(np.float32)
    # a glyph's size class must match too: a "1" and a tiny mark can look alike after normalisation
    hs = np.array([c["h"] for c in comps]); ws = np.array([c["w"] for c in comps])
    V = X @ X.T
    ok = (np.abs(hs[:, None] - hs[None, :]) < 0.12) & (np.abs(ws[:, None] - ws[None, :]) < 0.12)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    ii, jj = np.nonzero(np.triu((V > sim) & ok, 1))
    for a, b in zip(ii, jj):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    SF, SL = glyphs.synth_set()
    idx = {c: i for i, c in enumerate(CH)}
    labels = [None] * n; conf = [0.0] * n
    # tesseract on a few members of each cluster
    reps = []
    for g in groups.values():
        g2 = sorted(g, key=lambda i: -comps[i]["area"])
        mid = g2[len(g2) // 4: len(g2) // 4 + ntess] or g2[:ntess]
        reps.append((g, mid))
    texts = _tess_single([_glyph_img(comps[i]["mask"]) for _, mid in reps for i in mid], ocr_mod,
                         TESS_WL_ASCII if ascii_mode else TESS_WL)
    pos = 0
    info = {}
    for g, mid in reps:
        E = np.zeros(len(CH), np.float32)
        sims = X[g] @ SF.T
        top = np.argsort(-sims, axis=1)[:, :9]
        for r in range(len(g)):
            w = np.maximum(sims[r, top[r]], 0) ** 4
            for j, ww in zip(top[r], w):
                E[idx[SL[j]]] += ww / (w.sum() + 1e-9)
        E /= len(g)
        T = np.zeros(len(CH), np.float32)
        for i in mid:
            t = texts[pos]; pos += 1
            if t in idx:
                T[idx[t]] += 1.0 / len(mid)
        S = (E + 1.5 * T) * (ALLOW_ASCII if ascii_mode else ALLOW)
        if os.environ.get("TABOCR_DEBUG"):
            print("cluster n=%d h=%.2f E=%s T=%s" % (len(g), comps[g[0]]["h"], {CH[k]: round(float(E[k]), 2) for k in np.argsort(-E)[:3]},
                                                      {CH[k]: round(float(T[k]), 2) for k in range(len(CH)) if T[k]}))
        order = np.argsort(-S)
        best = int(order[0]); second = int(order[1])
        c = float(np.clip((S[best] - S[second]) / max(S[best], 1e-6), 0, 1))
        if len(g) >= 5 and T[best] >= 0.5:      # a frequent shape that Tesseract agrees on is not in doubt
            c = max(c, 0.9)
        for i in g:
            labels[i] = CH[best]; conf[i] = c
        info[find(g[0])] = (CH[best], c, len(g), CH[second])
    # small / unsure clusters: adopt the label of the most similar well-supported cluster of this video
    big = [(g, labels[g[0]]) for g in groups.values() if len(g) >= 5 and conf[g[0]] >= 0.3]
    if big:
        cen = np.stack([X[g].mean(0) / (np.linalg.norm(X[g].mean(0)) + 1e-9) for g, _ in big])
        for g in groups.values():
            if len(g) < 5 and conf[g[0]] < 0.6:
                cm = X[g].mean(0); cm /= (np.linalg.norm(cm) + 1e-9)
                sims = cen @ cm
                k = int(np.argmax(sims))
                if sims[k] > 0.85:
                    for i in g:
                        labels[i] = big[k][1]; conf[i] = min(conf[i], 0.5) if conf[i] else 0.5
    for i, c in enumerate(comps):       # a bare vertical stick is a "1" (its flag may have been cut by a slide line)
        if c["w"] <= 0.3 and c["h"] >= 0.55 and c["fill"] >= 0.45 and labels[i] != "1":
            labels[i] = "1"; conf[i] = min(conf[i], 0.6)
    if log:
        log(f"  rendered glyphs: {n} components in {len(groups)} clusters")
    return labels, conf, groups


def montage(comps, labels, groups, path, per=40):
    """Debug image: one representative of each cluster with its label."""
    from PIL import ImageDraw
    cells = []
    for g in sorted(groups.values(), key=lambda g: -len(g)):
        i = g[0]
        im = Image.fromarray(np.where(comps[i]["mask"], 0, 255).astype(np.uint8))
        im = im.resize((im.width * 3, im.height * 3), Image.NEAREST)
        cell = Image.new("L", (90, 80), 255); cell.paste(im, (4, 4))
        ImageDraw.Draw(cell).text((4, 64), f"{labels[i]} n={len(g)}", fill=0)
        cells.append(cell)
    cols = 12; rows = (len(cells) + cols - 1) // cols
    sheet = Image.new("L", (cols * 90, max(1, rows) * 80), 255)
    for k, c in enumerate(cells):
        sheet.paste(c, ((k % cols) * 90, (k // cols) * 80))
    sheet.save(path)


# ---------------------------------------------------------------- assembly

def assemble(info, comps, labels, confs, arcs):
    """info: segment_staff dict; comps/labels/confs aligned lists for this staff.  Returns the notes dict that
    ocr._notes_of would return (+ per note 'conf')."""
    sp = info["sp"]
    notes = []
    bystr = {}
    slides = []
    for c, l, cf in zip(comps, labels, confs):
        if c["kind"] == "slide":
            slides.append(c); continue
        if c.get("contrast", 1.0) < 0.4:       # ghost digit of a fading page: not read
            continue
        if l == "7" and c.get("fill", 0) > 0.375 and (c["w"] > 0.45 or c["h"] > 0.89):      # an eighth rest is a heavier "7"
            continue
        bystr.setdefault(c["s"], []).append((c, l, cf * min(1.0, 1.4 * c.get("contrast", 1.0))))
    for s, items in bystr.items():
        items.sort(key=lambda t: t[0]["x0"])
        i = 0
        while i < len(items):
            c, l, cf = items[i]
            if l == "x":
                notes.append(dict(s=s, x=c["cx"], fret=None, dead=1, conf=cf, txt="x", via=None, vib=None,
                                  ghost=None, xa=float(c["x0"]), xb=float(c["x1"]), ya=None, yb=None))
                i += 1; continue
            if l not in "0123456789":
                i += 1; continue
            j = i; digs = [items[i]]
            while j + 1 < len(items) and items[j + 1][1] in "0123456789" and len(digs) < 3 and \
                    items[j + 1][0]["x0"] - items[j][0]["x1"] <= 0.45 * sp:
                j += 1; digs.append(items[j])
            ghost = i > 0 and items[i - 1][1] == "(" and c["x0"] - items[i - 1][0]["x1"] <= 0.4 * sp
            fret = int("".join(d[1] for d in digs))
            if fret > 24:
                fret = None                  # not a playable fret: misread
            if fret is not None:
                x0 = digs[0][0]["x0"]; x1 = digs[-1][0]["x1"]
                notes.append(dict(s=s, x=float((x0 + x1 - 1) / 2), fret=fret, via=None, vib=None, dead=None,
                                  ghost=ghost, txt="".join(d[1] for d in digs), conf=min(d[2] for d in digs),
                                  xa=float(x0), xb=float(x1), ya=float(min(d[0]["y0"] for d in digs)),
                                  yb=float(max(d[0]["y1"] for d in digs))))
            i = j + 1
    notes.sort(key=lambda n: n["x"])
    for a in arcs:       # "H" / "P" letters sit right on top of hammer / pull arcs
        mid = (a["x0"] + a["x1"]) / 2
        for n in notes:
            if abs(n["x"] - mid) <= 0.35 * (a["x1"] - a["x0"]) and abs(n["x"] - mid) < 0.6 * sp and                     n["ya"] is not None and a["y0"] - 0.7 * sp <= n["yb"] <= a["y0"] + 4:
                n["drop"] = 1
    _slides(notes, slides, sp)
    _arcs(notes, arcs, sp)
    notes = [n for n in notes if not n.get("drop")]
    return dict(bars=info["bars"], notes=notes, ascii=False, adv=None, x0=info["x0"], x1=info["x1"], sp=sp,
                ys=info["ys"])


def _slides(notes, slides, sp):
    for sd in slides:
        same = [n for n in notes if n["s"] == sd["s"] and not n.get("dead")]
        A = max((n for n in same if n["xb"] <= sd["cx"] + 0.3 * sp), key=lambda n: n["x"], default=None)
        B = min((n for n in same if n["xa"] >= sd["cx"] - 0.3 * sp), key=lambda n: n["x"], default=None)
        if B is not None and B["xa"] - sd["x1"] < 0.8 * sp:
            if A is not None and B["x"] - A["x"] < 8 * sp and sd["x0"] - A["xb"] < 0.8 * sp:
                B["via"] = sd["ch"]                 # arrival of a slide from A
        elif A is not None and sd["x0"] - A["xb"] < 0.8 * sp:
            A["slide_out"] = 1


def _arcs(notes, arcs, sp):
    for a in arcs:
        cand_a = [n for n in notes if not n.get("dead") and abs(n["x"] - a["x0"]) <= 1.0 * sp]
        cand_b = [n for n in notes if not n.get("dead") and abs(n["x"] - a["x1"]) <= 1.0 * sp]
        best = None
        for A in cand_a:
            for B in cand_b:
                if B["x"] <= A["x"] + 0.5 * sp or A["s"] != B["s"]:
                    continue
                cost = abs(A["x"] - a["x0"]) + abs(B["x"] - a["x1"])
                if best is None or cost < best[0]:
                    best = (cost, A, B)
        if best is None:
            continue
        _, A, B = best
        chain = sorted([n for n in notes if n["s"] == A["s"] and not n.get("dead") and A["x"] - 1 <= n["x"] <= B["x"] + 1],
                       key=lambda n: n["x"])
        for P, Q in zip(chain, chain[1:]):         # one arc over several notes: a legato chain
            if Q["fret"] == P["fret"]:
                Q["drop"] = 1                      # tie: sustained, no new attack
            elif Q["fret"] > P["fret"]:
                Q["via"] = "h"
            else:
                Q["via"] = "p"


# ---------------------------------------------------------------- cross-page fusion

def fuse_same_pages(infos, tol=0.012):
    """The same tab page shows up under several page ids (crossfades, different zoom). Systems whose notes sit at
    the same relative positions on the same strings share one reading: per position the most confident wins."""
    n = len(infos)
    R = [i["rnotes"] for i in infos]
    U = [[(x["x"] - r["x0"]) / float(r["x1"] - r["x0"]) for x in r["notes"]] for r in R]

    def match(a, b):
        used, pairs = set(), []
        for i, na in enumerate(R[a]["notes"]):
            best = None
            for j, nb in enumerate(R[b]["notes"]):
                if j in used or na["s"] != nb["s"] or abs(U[a][i] - U[b][j]) > tol:
                    continue
                if best is None or abs(U[a][i] - U[b][j]) < abs(U[a][i] - U[b][best]):
                    best = j
            if best is not None:
                used.add(best); pairs.append((i, best))
        small = min(len(R[a]["notes"]), len(R[b]["notes"]))
        if small < 3 or len(pairs) < 0.6 * small or len(pairs) < 0.6 * max(len(R[a]["notes"]), len(R[b]["notes"])) - 3:
            return None
        return pairs
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for a in range(n):
        for b in range(a + 1, n):
            if find(a) != find(b) and match(a, b) is not None:
                parent[find(b)] = find(a)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    keys = ("fret", "conf", "via", "ghost", "dead", "txt", "flags")
    for g in groups.values():
        if len(g) < 2:
            continue
        ref = max(g, key=lambda k: np.mean([x["conf"] for x in R[k]["notes"]] or [0]))
        merged = [dict(x, u=U[ref][i]) for i, x in enumerate(R[ref]["notes"])]
        for k in g:
            if k == ref:
                continue
            pairs = match(ref, k)
            if pairs is None:
                continue
            seen = set()
            for i, j in pairs:
                seen.add(j)
                nb = R[k]["notes"][j]
                if nb["conf"] > merged[i]["conf"] + 0.05:
                    for kk in keys:
                        if kk in nb:
                            merged[i][kk] = nb[kk]
        for k in g:
            r = R[k]; own = []
            used = set()
            for m in merged:
                hit = None
                for j, nb in enumerate(r["notes"]):
                    if j not in used and nb["s"] == m["s"] and abs(U[k][j] - m["u"]) <= tol:
                        hit = j; break
                if hit is not None:
                    used.add(hit)
                    nb = dict(r["notes"][hit])
                    for kk in keys:
                        if kk in m:
                            nb[kk] = m[kk]
                    own.append(nb)
            for j, nb in enumerate(r["notes"]):      # notes without a counterpart stay as read
                if j not in used:
                    own.append(dict(nb))
            own.sort(key=lambda x: x["x"])
            r = dict(r); r["notes"] = own
            infos[k]["rnotes"] = r


def assemble_ascii(info, comps, labels, confs, parse_token):
    """ASCII (monospace) tab: characters grouped into tokens per string, parsed like the Tesseract path."""
    sp = info["sp"]
    bystr = {}
    for c, l, cf in zip(comps, labels, confs):
        bystr.setdefault(c["s"], []).append((c, l, cf))
    notes = []
    for s, items in bystr.items():
        items.sort(key=lambda t: t[0]["x0"])
        groups = []
        for it in items:
            if groups and it[0]["x0"] - groups[-1][-1][0]["x1"] <= 0.4 * sp:
                groups[-1].append(it)
            else:
                groups.append([it])
        for g in groups:
            if len(g) >= 3 and sum(1 for c, _, _ in g if c["h"] < 0.5) >= 2:
                continue                      # a word ("slide", "let ring"): small letters among tall characters
            txt = "".join(l for _, l, _ in g)
            for it in parse_token(txt):
                a = g[it["pos"]]; b = g[it["pos"] + it["len"] - 1]
                conf = min(x[2] for x in g[it["pos"]:it["pos"] + it["len"]])
                notes.append(dict(s=s, x=float(a[0]["x0"]), fret=it["fret"], via=it.get("via"), vib=it.get("vib"),
                                  dead=it.get("dead"), ghost=it.get("ghost"), txt=txt, conf=conf))
    notes.sort(key=lambda n: n["x"])
    return dict(bars=info["bars"], notes=notes, ascii=True, adv=info["adv"], x0=info["x0"], x1=info["x1"], sp=sp,
                ys=info["ys"])
