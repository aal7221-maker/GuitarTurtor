"""Music sanity checks on read tabs: flag likely OCR errors and lower per-note confidence.

- fret above the guitar's range (default 22) or above 24
- hand position: within a bar, notes more than 5 frets from the bar's median fretted position
- key: pitch classes outside the best-fitting diatonic key that are rare in the piece
- audio: tab pitch vs the detected audio at the aligned onset (annotate_audio, after alignment)
Each flag multiplies the note's conf; flags are kept in note["flags"]."""
import numpy as np

MAJOR = np.array([0, 2, 4, 5, 7, 9, 11]); MINOR = np.array([0, 2, 3, 5, 7, 8, 10])


def _flag(n, name, mult):
    n.setdefault("flags", [])
    if name not in n["flags"]:
        n["flags"].append(name)
        n["conf"] = n.get("conf", 1.0) * mult


def check_pages(pev, tuning, capo=0, max_fret=22, span=5):
    """pev: list of (events, nbars) as from read_tabs.page_event_lists. Modifies note dicts in place."""
    allnotes = []
    for evs, nb in pev:
        bars = {}
        for e in evs:
            for n in e["notes"]:
                if n.get("fret") is None:
                    continue
                n.setdefault("conf", 1.0)
                if n["fret"] > max_fret:
                    _flag(n, "fret>%d" % max_fret, 0.4)
                bars.setdefault(e["bar"], []).append(n)
                allnotes.append(n)
        for b, ns in bars.items():
            fr = [n["fret"] for n in ns if n["fret"] > 0]
            if len(fr) < 3:
                continue
            med = float(np.median(fr))
            for n in ns:
                if n["fret"] > 0 and abs(n["fret"] - med) > span:
                    _flag(n, "span", 0.6)
    if len(allnotes) < 20:
        return
    pcs = np.zeros(12)
    for n in allnotes:
        pcs[(tuning[n["s"]] + capo + n["fret"]) % 12] += 1
    best = None
    for root in range(12):
        for sc in (MAJOR, MINOR):
            ins = set(((sc + root) % 12).tolist())
            score = sum(pcs[p] for p in ins)
            if best is None or score > best[0]:
                best = (score, ins)
    ins = best[1]
    tot = pcs.sum()
    for n in allnotes:
        p = (tuning[n["s"]] + capo + n["fret"]) % 12
        if p not in ins and pcs[p] < 0.03 * tot:
            _flag(n, "key", 0.7)


def annotate_audio(events, assign, aud, thr=0.2):
    """Per-note audio evidence at the aligned onset. Sets n['sal']; flags notes whose pitch is absent."""
    for e, j in zip(events, assign):
        if j < 0:
            continue
        for n, m in zip(e["notes"], e["midi"]):
            sal = aud.salience(j, [m], e.get("legato", False))
            n["sal"] = round(float(sal), 3)
            if sal < thr:
                _flag(n, "audio", 0.6)
            elif sal > 0.5:                 # the audio has this pitch at this onset: reading confirmed
                n["conf"] = max(n.get("conf", 1.0), 0.85)
