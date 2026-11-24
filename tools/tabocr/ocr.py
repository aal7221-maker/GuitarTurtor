"""Read a tab page: per string the fret numbers, their x positions, techniques and bar lines."""
import os, re, shutil, subprocess, tempfile
import numpy as np
from scipy import ndimage as ndi
from PIL import Image
import glyphs

WHITELIST = "0123456789hpbr/\\~x()"


ASCII_NEW = os.environ.get("TABOCR_ASCII", "new") == "new"     # monospace tabs through the glyph-cluster reader


def tesseract():
    p = os.environ.get("TESSERACT") or shutil.which("tesseract")
    if p:
        return p
    for c in (r"C:/Program Files/Tesseract-OCR/tesseract.exe", r"C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"):
        if os.path.exists(c):
            return c
    raise FileNotFoundError("tesseract (winget install UB-Mannheim.TesseractOCR)")


def _runs(mask):
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def median_frame(crops):
    if len(crops) == 1:
        return crops[0]
    return np.median(np.stack(crops[:9]), axis=0).astype(np.uint8)


def _ink(gray, pol):
    g = gray.astype(np.float32)
    if pol == "light":
        g = 255 - g
    bg = ndi.uniform_filter(g, 41)
    bg = np.maximum(bg, ndi.maximum_filter(ndi.uniform_filter(g, 5), 9))   # background = local light level
    ink = g < bg - 40
    return ink, g


def _refine_lines(ink, staff):
    sp = staff["sp"]; ys = []
    for y in staff["ys"]:
        a, b = int(y - 0.2 * sp), int(y + 0.2 * sp) + 1
        cov = ink[a:b, staff["x0"]:staff["x1"]].mean(1)
        ys.append(a + int(np.argmax(cov)))
    return ys


def _period(sig, lo, hi):
    s = sig.astype(np.float32) - sig.mean()
    if s.std() < 1e-3:
        return None, 0.0
    ac = np.correlate(s, s, "full")[len(s) - 1:]
    ac /= ac[0]
    lo, hi = max(2, int(lo)), min(len(ac) - 1, int(hi))
    if hi <= lo:
        return None, 0.0
    k = lo + int(np.argmax(ac[lo:hi]))
    return k, float(ac[k])


def segment_staff(gray, staff, pol):
    """Find bar lines and glyph segments per string. Returns dict."""
    ink, g = _ink(gray, pol)
    sp = staff["sp"]; x0, x1 = staff["x0"], staff["x1"]
    ys = _refine_lines(ink, staff)
    # line half-thickness
    hw = []
    for y in ys:
        prof = ink[y - int(0.3 * sp):y + int(0.3 * sp) + 1, x0:x1].mean(1)
        hw.append(max(1, int(np.sum(prof > 0.5 * prof.max()) // 2) + 1))
    hw = int(np.median(hw))
    # bar lines: ink at every line and at (nearly) every midpoint between lines
    W = gray.shape[1]
    mids = [(ys[i] + ys[i + 1]) // 2 for i in range(5)]
    at_lines = np.mean([ink[y - hw:y + hw + 1].any(0) for y in ys], 0)
    at_mids = np.mean([ink[m - 1:m + 2].any(0) for m in mids], 0)
    barcol = (at_lines >= 5 / 6) & (at_mids >= 0.8)
    barcol[:max(0, x0 - int(sp))] = False; barcol[min(W, x1 + int(sp)):] = False
    bars = [int((a + b) / 2) for a, b in _runs(barcol) if b - a <= max(4, 0.2 * sp)]
    barmask = ndi.binary_dilation(barcol, iterations=2)
    # ASCII tab? dashes give a strong period on the line row
    adv, strength = _period(ink[ys[5], x0:x1] | ink[ys[0], x0:x1], 0.2 * sp, 0.9 * sp)
    ascii_tab = strength > 0.35
    strings = []
    for i, y in enumerate(ys):
        top, bot = int(y - 0.5 * sp), int(y + 0.5 * sp) + 1
        strip = ink[top:bot].copy()
        strip[:, barmask] = False
        rel = np.arange(top, bot) - y
        off = (np.abs(rel) > hw + 1) & (np.abs(rel) < 0.46 * sp)
        glyph = strip[off].any(0)
        glyph[:max(0, x0 - int(0.3 * sp))] = False
        glyph[min(W, x1 + 2):] = False
        gap = 0.75 * adv if ascii_tab else 0.28 * sp
        segs = []
        for a, b in _runs(glyph):
            if segs and a - segs[-1][1] <= gap:
                segs[-1][1] = b
            else:
                segs.append([a, b])
        segs = [s for s in segs if strip[:, s[0]:s[1]].sum() >= 0.04 * sp * sp]
        strings.append(dict(y=y, top=top, bot=bot, segs=segs, glyph=glyph))
    # string-name labels (e B G D A E) or a TAB clef: an aligned column of glyphs at the very left
    firsts = [s["segs"][0][0] for s in strings if s["segs"]]
    if len(firsts) >= 4:
        xs = np.array(firsts)
        left = xs[xs <= x0 + 0.6 * sp]
        if len(left) >= 4 and left.max() - left.min() < 0.5 * sp:
            lab_w = (0.9 * adv if ascii_tab else 0.6 * sp)
            ends = np.array([s["segs"][0][1] for s in strings if s["segs"] and s["segs"][0][0] <= left.max() + 1])
            if ascii_tab and len(ends) >= 4 and ends.max() - ends.min() <= 3:
                for s in strings:           # "e|" "B|": label letters + bar line end at one common x
                    if s["segs"] and s["segs"][0][0] <= left.max() + 1:
                        s["segs"] = s["segs"][1:]
                lab_w = None
            for s in strings:
                segs = []
                for g in s["segs"]:
                    if lab_w is not None and g[0] <= left.max() + 1:
                        g = [int(g[0] + lab_w), g[1]]      # "e/16": keep what follows the label
                        sub = s["glyph"][g[0]:g[1]]
                        if sub.sum() < 0.15 * sp:
                            continue
                        g[0] += int(np.argmax(sub))
                        if ascii_tab and g[1] - g[0] < 0.4 * adv:      # the "|" after the label
                            continue
                    segs.append(g)
                s["segs"] = segs
    return dict(ys=ys, hw=hw, bars=bars, ascii=ascii_tab, adv=adv if ascii_tab else None,
                strings=strings, ink=ink, sp=sp, x0=x0, x1=x1)


def _token_mask(info, s, a, b):
    ink = info["ink"]; hw = info["hw"]
    top, bot, y = s["top"], s["bot"], s["y"]
    a0, b0 = max(0, a - 2), b + 2
    m = ink[top:bot, a0:b0].copy()
    gl = s["glyph"][a0:b0]
    r0, r1 = y - hw - top, y + hw + 1 - top
    m[r0:r1, ~gl] = False          # erase dashes / staff line between glyphs
    return m, a0


def _token_img(m, sp, target_h=44):
    rows = np.flatnonzero(m.any(1))
    if not len(rows):
        return None
    m = m[max(0, rows[0] - 1):rows[-1] + 2]
    scale = target_h / max(8, 0.62 * sp)
    img = Image.fromarray(np.where(m, 0, 255).astype(np.uint8))
    img = img.resize((max(4, int(img.width * scale)), max(4, int(img.height * scale))), Image.LANCZOS)
    return np.array(img)


def _components(m, a0, sp, line_row):
    """Character components of a token mask: list of (x_left, x_right, feature)."""
    lab, n = ndi.label(m, structure=np.ones((3, 3), bool))
    if n == 0:
        return []
    boxes = []
    for k, sl in enumerate(ndi.find_objects(lab)):
        if (lab[sl] == k + 1).sum() < max(3, 0.004 * sp * sp):
            continue
        boxes.append([sl[1].start, sl[1].stop, [k + 1]])
    boxes.sort()
    merged = []
    for bx in boxes:   # parts of one glyph overlap in x
        if merged and min(bx[1], merged[-1][1]) - max(bx[0], merged[-1][0]) > 0.5 * min(bx[1] - bx[0], merged[-1][1] - merged[-1][0]):
            merged[-1][0] = min(merged[-1][0], bx[0]); merged[-1][1] = max(merged[-1][1], bx[1]); merged[-1][2] += bx[2]
        else:
            merged.append(bx)
    out = []
    H = m.shape[0]
    for ca, cb, ids in merged:
        cm = np.isin(lab, ids)
        rows = np.flatnonzero(cm.any(1))
        # fret numbers sit centred on their string line and fit inside the strip; rests, slurs, stems,
        # bend arrows and text between the lines do not
        if rows[0] == 0 or rows[-1] == H - 1:
            continue
        if abs((rows[0] + rows[-1]) / 2 - line_row) > 0.24 * sp or (rows[-1] - rows[0] + 1) > 0.95 * sp:
            continue
        f = glyphs.feature(cm)
        if f is None:
            continue
        out.append((a0 + ca, a0 + cb, f, (rows[-1] - rows[0] + 1) / sp))
    return out


def run_tesseract(tokens, psm=6, wl=None):
    """tokens: list of uint8 images. Returns list of strings (one per token)."""
    if not tokens:
        return []
    pad = 24
    Wd = max(t.shape[1] for t in tokens) + 2 * pad
    rows, spans, y = [], [], pad
    for t in tokens:
        canvas = np.full((t.shape[0] + 2 * pad, Wd), 255, np.uint8)
        canvas[pad:pad + t.shape[0], pad:pad + t.shape[1]] = t
        rows.append(canvas); spans.append((y, y + t.shape[0])); y += canvas.shape[0]
    sheet = np.concatenate(rows, 0)
    fd, path = tempfile.mkstemp(suffix=".png"); os.close(fd)
    try:
        Image.fromarray(sheet).save(path)
        cmd = [tesseract(), path, "stdout", "--psm", str(psm), "-c", "tessedit_char_whitelist=" + (wl or WHITELIST),
               "-c", "load_system_dawg=0", "-c", "load_freq_dawg=0", "tsv"]
        out = subprocess.run(cmd, capture_output=True, text=True).stdout
    finally:
        os.remove(path)
    words = [[] for _ in tokens]
    tops = [p - 0 for p, _ in spans]
    for line in out.splitlines()[1:]:
        f = line.split("\t")
        if len(f) < 12 or not f[11].strip():
            continue
        l, t, w, h, conf = int(f[6]), int(f[7]), int(f[8]), int(f[9]), float(f[10])
        cy = t + h / 2 + pad
        k = int(np.argmin([abs(cy - (a + b) / 2 - pad) for a, b in spans]))
        words[k].append((l, f[11].strip(), conf))
    return ["".join(w for _, w, _ in sorted(ws)) for ws in words], [
        (min(c for _, _, c in ws) if ws else -1) for ws in words]


TOKEN_RE = re.compile(r"\(?\d{1,2}\)?|[hpbr/\\~x]")


def parse_token(text):
    """'8h9' -> [(fret, tech_into_this_note, flags)] items in order with char offsets."""
    items = []
    pending = None
    for m in TOKEN_RE.finditer(text):
        tok = m.group(0)
        if tok in "hpbr/\\":
            pending = tok
        elif tok == "~":
            if items: items[-1]["vib"] = 1
        elif tok == "x":
            items.append(dict(fret=None, dead=1, pos=m.start(), len=1, via=pending)); pending = None
        else:
            ghost = tok.startswith("(")
            f = int(tok.strip("()"))
            if f > 24:
                continue          # not a playable fret: misread
            items.append(dict(fret=f, pos=m.start() + (1 if ghost else 0), len=len(tok.strip("()")),
                              via=pending, ghost=ghost)); pending = None
    return items


def read_pages(pages, log=None, debug=None):
    """OCR every page. pages: list of dict(staves, pol, crops). Returns per page a list of systems:
    dict(bars, notes=[dict(s, x, fret, via, vib, dead, ghost, txt, conf)], ascii, adv, x0, x1, sp).
    ASCII tabs go through Tesseract tokens; rendered (Guitar Pro style) tabs through rendered.py."""
    import rendered
    allsys = []
    rinfos = []          # rendered staves
    for pi, p in enumerate(pages):
        gray = median_frame(p["crops"])
        systems = []
        for staff in p["staves"]:
            info = segment_staff(gray, staff, p["pol"])
            if not info["ascii"]:
                comps, arcs = rendered.staff_components(gray, staff, p["pol"], info["ys"], info["bars"])
                comps = rendered.split_wide(comps, 0)
                info.pop("ink", None)
                for s in info["strings"]:
                    s.pop("glyph", None)
                info["tokens"] = []
                info["rendered"] = (comps, arcs)
                rinfos.append(info)
                systems.append(info)
                continue
            toks = []
            for si, s in enumerate(info["strings"]):
                for a, b in s["segs"]:
                    m, a0 = _token_mask(info, s, a, b)
                    im = _token_img(m, info["sp"])
                    if im is None:
                        continue
                    toks.append(dict(s=si, a=int(a), b=int(b), img=im, comps=_components(m, a0, info["sp"], s["y"] - s["top"])))
            texts, confs = run_tesseract([t["img"] for t in toks])
            for t, txt, cf in zip(toks, texts, confs):
                t["text"], t["conf"] = txt, cf
            for t in toks:   # retry empty/odd ones one by one (single-word mode)
                if not VALID.fullmatch(t["text"] or "") or not re.search(r"\d|x", t["text"]):
                    t2, c2 = run_tesseract([t["img"]], psm=8)
                    t["text"], t["conf"] = t2[0], c2[0]
            # words in the tab strip ("slide", "let ring") are not notes: a letters-allowed reading finds them
            if toks:
                wtxt, _ = run_tesseract([t["img"] for t in toks], wl="abcdefgijklmnoqrstuvwyz0123456789hpbx/\~-")
                for t, w in zip(toks, wtxt):
                    w = re.sub(r"[^a-z0-9]", "", w or "")
                    if len(re.findall(r"[acdefgijklmnoqrstuvwyz]", w)) >= 4 and                             len(re.findall(r"[acdefgijklmnoqrstuvwyz]", w)) >= 0.7 * len(w):
                        t["comps"] = []; t["text"] = ""; t["word"] = True
            toks = [t for t in toks if not t.get("word")]
            if ASCII_NEW:       # second reading: glyph clusters (used if the font allows it, see below)
                firsts = [sg["segs"][0][0] for sg in info["strings"] if sg["segs"]]
                minx = (min(firsts) - 2) if firsts else None
                comps, arcs = rendered.staff_components(gray, staff, p["pol"], info["ys"], info["bars"],
                                                        ascii_mode=True, minx=minx)
                info["rendered"] = (rendered.split_wide(comps, 0), arcs)
                rinfos.append(info)
            info.pop("ink", None)
            for s in info["strings"]:
                s.pop("glyph", None)
            info["tokens"] = toks
            systems.append(info)
        allsys.append(systems)
        if log:
            log(f"  page {pi}: " + ", ".join(
                f"{len(s['tokens'])} tokens/{len(s['bars'])} bars" if s["ascii"] else
                f"{len(s['rendered'][0])} glyphs/{len(s['bars'])} bars" for s in systems))
    import sys as _sys
    for is_ascii in (False, True):
        group = [i for i in rinfos if i["ascii"] == is_ascii]
        if not group:
            continue
        flat = [c for info in group for c in info["rendered"][0] if c["kind"] == "glyph"]
        labels, confs, groups = rendered.classify_comps(flat, _sys.modules[__name__], log=log, ascii_mode=is_ascii)
        if debug and flat:
            rendered.montage(flat, labels, groups, os.path.join(debug, "glyphs.png"))
        if is_ascii and (len(flat) < 20 or len(groups) > 0.3 * len(flat) or float(np.mean(confs)) < 0.9):
            # glyphs do not repeat cleanly (dashes touching digits, odd font): keep the Tesseract token reading
            if log:
                log(f"  ascii: {len(groups)} glyph clusters for {len(flat)} characters - using the Tesseract reading")
            for info in group:
                del info["rendered"]
            continue
        lab = {id(c): (l, cf) for c, l, cf in zip(flat, labels, confs)}
        for info in group:
            comps, arcs = info["rendered"]
            ls = [lab[id(c)][0] if c["kind"] == "glyph" else c["ch"] for c in comps]
            cs = [lab[id(c)][1] if c["kind"] == "glyph" else 1.0 for c in comps]
            if info["ascii"]:
                info["rnotes"] = rendered.assemble_ascii(info, comps, ls, cs, parse_token)
            else:
                info["rnotes"] = rendered.assemble(info, comps, ls, cs, arcs)
            del info["rendered"]
        if not is_ascii:
            rendered.fuse_same_pages(group)
    asc = [[s for s in sy if s["ascii"] and "rnotes" not in s] for sy in allsys]
    if any(asc):
        _self_consistency(asc, log)
    return [[info["rnotes"] if "rnotes" in info else _notes_of(info) for info in systems] for systems in allsys]


VALID = re.compile(r"[0-9()hpbr/\\~x]+")


def _self_consistency(allsys, log=None):
    """Re-read every character with the glyph classifier (font templates + Tesseract + per-video clusters)."""
    feats, tl, where = [], [], []
    for systems in allsys:
        for info in systems:
            for t in info["tokens"]:
                txt = "".join(VALID.findall(t["text"] or ""))
                t["text"] = txt
                ok = len(txt) == len(t["comps"])
                for ci, c in enumerate(t["comps"]):
                    ch = txt[ci] if ok else None
                    if ch == "~" and c[3] >= 0.28:        # a squiggle is small: a tall "~" is a failed reading
                        ch = None
                    feats.append(c[2]); tl.append(ch); where.append((t, ci))
    labels = glyphs.classify(feats, tl, log)
    per = {}
    for (t, ci), l in zip(where, labels):
        h = t["comps"][ci][3]
        if h < 0.28:                      # small marks: vibrato squiggle or noise
            w = (t["comps"][ci][1] - t["comps"][ci][0]) / max(1e-6, h * 1.0)
            l = "~" if l == "~" else ""
        per.setdefault(id(t), (t, {}))[1][ci] = l
    for t, d in per.values():
        keep = [ci for ci in range(len(t["comps"])) if d.get(ci)]
        t["comps"] = [t["comps"][ci] for ci in keep]
        t["text"] = "".join(d[ci] for ci in keep)


def _notes_of(info):
    notes = []
    for t in info["tokens"]:
        txt = t["text"]
        items = parse_token(txt)
        n = max(1, len(txt))
        comps = t["comps"] if len(t["comps"]) == len(txt) else None
        for it in items:
            if comps:
                ca = comps[it["pos"]][0]; cb = comps[it["pos"] + it["len"] - 1][1]
            else:
                ca = t["a"] + it["pos"] * (t["b"] - t["a"]) / n
                cb = t["a"] + (it["pos"] + it["len"]) * (t["b"] - t["a"]) / n
            x = ca if info["ascii"] else (ca + cb) / 2     # ASCII: a number starts at its column
            notes.append(dict(s=t["s"], x=float(x), fret=it["fret"], via=it.get("via"), vib=it.get("vib"),
                              dead=it.get("dead"), ghost=it.get("ghost"), txt=txt))
    return dict(bars=info["bars"], notes=notes, ascii=info["ascii"], adv=info["adv"],
                x0=info["x0"], x1=info["x1"], sp=info["sp"], ys=info["ys"])
