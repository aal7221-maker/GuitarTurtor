"""Turn OCR'd tab systems into ordered note events with relative rhythm (from x spacing and bar lines)."""
import numpy as np

TECH = {"h": "ham", "p": "pull", "/": "slide", "\\": "slide"}


def system_events(sysm):
    """Events of one system: list of dict(x, bar, pos, notes=[dict(s, fret, ...)]), plus number of bars."""
    notes = [n for n in sysm["notes"] if n["fret"] is not None or n.get("dead")]
    if not notes:
        return [], 0
    tol = 0.6 * sysm["adv"] if sysm["ascii"] and sysm["adv"] else 0.35 * sysm["sp"]
    notes.sort(key=lambda n: n["x"])
    clusters = []
    for n in notes:
        if clusters and n["x"] - clusters[-1]["x"] <= tol and n["s"] not in {m["s"] for m in clusters[-1]["notes"]}:
            c = clusters[-1]; c["notes"].append(n); c["x"] = float(np.mean([m["x"] for m in c["notes"]]))
        else:
            clusters.append(dict(x=n["x"], notes=[n]))
    # bars: only those that contain notes (lesson tabs pad pages with empty bars)
    bl = sorted(b for b in sysm["bars"])
    edges = [min(sysm["x0"], clusters[0]["x"] - 1)] + [b for b in bl if clusters[0]["x"] - tol < b < clusters[-1]["x"] + tol]
    nxt = [b for b in bl if b > clusters[-1]["x"]]
    edges.append(nxt[0] if nxt else max(sysm["x1"], clusters[-1]["x"] + 1))
    edges = sorted(set(edges))
    bar_of = np.searchsorted(edges, [c["x"] for c in clusters], side="right") - 1
    used = sorted(set(bar_of.tolist()))
    remap = {b: k for k, b in enumerate(used)}
    for c, b in zip(clusters, bar_of):
        a, z = edges[b], edges[min(b + 1, len(edges) - 1)]
        if b == 0 and len(edges) > 1:
            # first bar on the system may start at the label column: measure from its first note
            a = min(a, c["x"]) if a > 0 else a
        c["bar"] = remap[int(b)]
        c["pos"] = float(np.clip((c["x"] - a) / max(1.0, z - a), 0, 0.999))
    # the first bar of a system usually starts right after the string labels: rescale its positions so
    # its first event sits where the following bars' first events sit (typical lead-in)
    return clusters, len(used)


def page_events(systems):
    evs, nb = [], 0
    for sysm in systems:
        e, n = system_events(sysm)
        for c in e:
            c["bar"] += nb
        evs += e; nb += n
    return evs, nb


def finalize(events, capo=0, tuning=(64, 59, 55, 50, 45, 40)):
    """Attach MIDI pitches, techniques (on the source note, site convention) and legato flags."""
    last = {}   # string -> (event idx, note dict)
    for i, e in enumerate(events):
        e["legato"] = False
        e["glide"] = False      # slide/release arrival: no new attack, timed by interpolation
        out = []
        for n in e["notes"]:
            if n.get("dead"):
                continue
            via = n.get("via")
            prev = last.get(n["s"])
            if via == "b" and prev:
                prev[1]["bend"] = n["fret"]; continue          # bend target is not a new note
            if via in TECH and prev and i - prev[0] <= 3:
                prev[1][TECH[via]] = 1
                e["legato"] = True
                if TECH[via] == "slide" and len(e["notes"]) == 1:
                    e["glide"] = True
            if n.get("slide_out"):
                n["slide"] = 1
            if via == "r" and prev:
                n = dict(n); n["rel"] = 1; e["legato"] = True; e["glide"] = len(e["notes"]) == 1
            out.append(n)
            last[n["s"]] = (i, n)
            if n.get("vib"):
                n["vib"] = 1
        e["notes"] = out
        e["midi"] = [tuning[n["s"]] + capo + n["fret"] for n in out]
    return [e for e in events if e["notes"]]


def to_text(events):
    s = []
    for e in events:
        s.append("+".join(f"{'eBGDAE'[n['s']]}{n['fret']}" for n in sorted(e["notes"], key=lambda n: n["s"])))
    return " ".join(s)
