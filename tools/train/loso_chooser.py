"""Local CPU port of the notebook's Part A (string chooser MLP) for an honest leave-one-song-out guard:
pretrain GuitarSet 00-03 + lesson videos (train split) -> fine-tune on the OTHER priority song only (whole song, x20,
replay) -> select by GuitarSet player 04 + held-out lesson videos (never the test song) -> work/chooser_loso_<held>.onnx.
run: tools/train/.venv-ort/Scripts/python.exe tools/train/loso_chooser.py
"""
import os, json, zipfile, hashlib
import numpy as np, torch, torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__)); DS = os.path.join(HERE, "..", "dataset")
torch.manual_seed(0); np.random.seed(0); dev = "cpu"
STD = [64, 59, 55, 50, 45, 40]; NOTE = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
MAN = [json.loads(l) for l in open(os.path.join(DS, "manifest.jsonl"), encoding="utf-8")]
MAN = [m for m in MAN if m.get("status") == "pass" and m.get("label")]
PRI = [m for m in MAN if m.get("priority")]; LES = [m for m in MAN if not m.get("priority")]
order = sorted(LES, key=lambda m: hashlib.md5(m["id"].encode()).hexdigest())
nh = max(3, round(.15 * len(LES))) if len(LES) >= 6 else 0
LES_HO, LES_TR = order[:nh], order[nh:]
def load_item(m, lo=0., hi=1.):
    d = json.load(open(os.path.join(DS, m["label"]), encoding="utf-8"))["notes"]
    if m.get("until"): d = [n for n in d if n["t"] < m["until"]]
    d = [n for n in d if n.get("conf", 1) >= .7 and not n.get("flags")]
    if not d: return []
    ts = [n["t"] for n in d]; a0, a1 = min(ts), max(ts); T0 = a0 + lo * (a1 - a0); T1 = a0 + hi * (a1 - a0) + (1e-6 if hi < 1 else 1)
    return sorted((n["t"], STD[s] + f, s, f, 0) for n in d if T0 <= n["t"] < T1 for s, f in n["n"] if 0 <= s < 6)
def cands(p): return [(s, p - STD[s]) for s in range(6) if 0 <= p - STD[s] <= 22]
def load_gs(players):
    z = zipfile.ZipFile(os.path.join(HERE, "data", "guitarset", "annotation.zip")); res = []
    for nm in z.namelist():
        if not nm.endswith(".jams") or os.path.basename(nm)[:2] not in players: continue
        d = json.load(z.open(nm)); ns = []
        for a in d["annotations"]:
            if a["namespace"] != "note_midi": continue
            s = 5 - int(a["annotation_metadata"]["data_source"])
            for x in a["data"]:
                p = int(round(x["value"])); f = p - STD[s]
                if 0 <= f <= 22: ns.append((x["time"], p, s, f, p))
        res.append(sorted(ns))
    return res
H = 4
def seq_to_samples(seq):
    X = []; M = []; Y = []; hist = []
    for t, p, s, f, _ in seq:
        h = hist[-H:]; h = [(0, 0, 60, -9)] * (H - len(h)) + h
        X.append([p / 88] + [v for hs, hf, hp, ht in h for v in (hs / 5, hf / 22, hp / 88, min(t - ht, 2.0) / 2)])
        m = np.zeros(6)
        for cs, cf in cands(p): m[cs] = 1
        m[s] = 1; M.append(m); Y.append(s); hist.append((s, f, p, t))
    return np.array(X, 'f'), np.array(M, 'f'), np.array(Y, 'int64')
class Net(nn.Module):
    def __init__(s): super().__init__(); s.f = nn.Sequential(nn.Linear(1 + 4 * H, 128), nn.ReLU(), nn.Dropout(.2), nn.Linear(128, 128), nn.ReLU(), nn.Dropout(.2), nn.Linear(128, 6))
    def forward(s, x, m): return s.f(x).masked_fill(m == 0, -1e9)
def tens(seqs, w):
    parts = [seq_to_samples(q) for q in seqs if len(q) > 1]
    if not parts: return None
    X = np.vstack([p[0] for p in parts]); M = np.vstack([p[1] for p in parts]); Y = np.concatenate([p[2] for p in parts])
    return [torch.tensor(a) for a in (X, M, Y)] + [torch.full((len(Y),), float(w))]
def acc(net, T):
    net.eval()
    with torch.no_grad(): return (net(T[0], T[1]).argmax(1) == T[2]).float().mean().item()
def train(net, parts, score, epochs=40, lr=2e-3, patience=8):
    parts = [p for p in parts if p is not None]
    X, M, Y, W = [torch.cat([p[i] for p in parts]) for i in range(4)]
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2); best = (-1, None); bad = 0
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(Y))
        for i in range(0, len(Y), 512):
            b = perm[i:i + 512]; l = (nn.functional.cross_entropy(net(X[b], M[b]), Y[b], reduction='none') * W[b]).sum() / W[b].sum()
            opt.zero_grad(); l.backward(); opt.step()
        a = score(net)
        if a > best[0]: best = (a, {k: v.clone() for k, v in net.state_dict().items()}); bad = 0
        else:
            bad += 1
            if bad >= patience: break
    net.load_state_dict(best[1]); return best[0]
if __name__ == "__main__":
    G_tr, G_va = tens(load_gs([f"0{i}" for i in range(4)]), 1), tens(load_gs(["04"]), 1)
    L_tr = tens([load_item(m) for m in LES_TR], 1); L_ho = tens([load_item(m) for m in LES_HO], 1)
    sel = lambda n: .5 * acc(n, G_va) + .5 * (acc(n, L_ho) if L_ho else acc(n, G_va))
    pre = os.path.join(HERE, "work", "chooser_pre_local.pt")
    net = Net(); train(net, [G_tr, L_tr], sel); torch.save(net.state_dict(), pre)
    res = {"pretrain": {m["id"]: round(acc(net, tens([load_item(m)], 1)), 3) for m in PRI}}
    for held in PRI:
        other = [m for m in PRI if m is not held]
        n2 = Net(); n2.load_state_dict(torch.load(pre))
        train(n2, [tens([load_item(m) for m in other], 20), L_tr, G_tr], sel, epochs=30, lr=5e-4)
        res[held["id"]] = round(acc(n2, tens([load_item(held)], 1)), 3)
        torch.onnx.export(n2, (torch.zeros(1, 1 + 4 * H), torch.ones(1, 6)), os.path.join(HERE, "work", f"chooser_loso_{held['id']}.onnx"),
                          input_names=["x", "mask"], output_names=["logits"], dynamic_axes={"x": {0: "n"}, "mask": {0: "n"}, "logits": {0: "n"}})
    print("string acc on true pitches (whole held song):", res)
    json.dump(res, open(os.path.join(HERE, "work", "loso_chooser.json"), "w"), indent=1)
