"""Align a tab note sequence to audio onsets with pitch evidence (banded DTW / subsequence DP)."""
import numpy as np
import scipy.signal as ss

HOP = 256
MIDI_LO = 36
import os
NOV_MODE = os.environ.get("TABOCR_NOV", "mix")


class Audio:
    def __init__(self, y, sr, highpass=70.0):
        import librosa
        self.sr = sr
        if highpass:
            y = ss.filtfilt(*ss.butter(4, highpass / (sr / 2), "high"), y)
        self.y = y.astype(np.float32)
        env = librosa.onset.onset_strength(y=self.y, sr=sr, hop_length=HOP)
        fr = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=HOP, backtrack=False,
                                        delta=0.04, wait=2, units="frames")
        C = np.abs(librosa.cqt(self.y, sr=sr, hop_length=HOP, fmin=librosa.midi_to_hz(MIDI_LO),
                               n_bins=72, bins_per_octave=12))
        C = np.log1p(30 * C / (C.max() + 1e-9))
        self.C = C
        # merge double detections (< 50 ms apart): keep the first frame
        keep = []
        for f in fr:
            if keep and (f - keep[-1]) * HOP / sr < 0.05:
                continue
            keep.append(f)
        fr = np.array(keep, int)
        self.times = librosa.frames_to_time(fr, sr=sr, hop_length=HOP)
        st = env[fr]
        self.strength = st / (np.percentile(st, 95) + 1e-9) if len(st) else st
        n = C.shape[1]
        nov, pres = [], []
        Clin = np.expm1(C)
        for f in fr:
            # re-attacks of an already ringing string show as a ratio jump, not an absolute one
            post = Clin[:, min(n - 1, f + 1):min(n, f + 5)].max(1)
            pre = Clin[:, max(0, f - 3):max(1, f)].mean(1)
            v = np.maximum(np.log((post + 1e-3) / (pre + 1e-3)), 0) * np.sqrt(post / (post.max() + 1e-9))
            if NOV_MODE == "mix":
                pc = C[:, min(n - 1, f + 1):min(n, f + 6)].mean(1); pp = C[:, max(0, f - 5):max(1, f - 1)].mean(1)
                b = np.maximum(pc - 0.75 * pp, 0)
                v = v / (v.max() + 1e-9) + b / (b.max() + 1e-9)
            nov.append(v); pres.append(C[:, min(n - 1, f + 1):min(n, f + 6)].mean(1))
        self.nov = np.array(nov).reshape(len(fr), 72)
        self.pres = np.array(pres).reshape(len(fr), 72)

    def salience(self, j, midis, legato=False):
        """0..1 evidence that MIDI pitches `midis` start (or, legato, sound) at onset j."""
        v = self.pres[j] if legato else self.nov[j]
        lo, hi = 40 - MIDI_LO, 90 - MIDI_LO
        ref = v[lo:hi].max() + 1e-6
        sc = []
        for m in midis:
            k = m - MIDI_LO
            if not (0 <= k < 72):
                sc.append(0.0); continue
            a = v[max(0, k - 0):k + 1].max()
            h2 = v[k + 12] if k + 12 < 72 else 0
            sc.append(min(1.0, (a + 0.35 * h2) / ref))
        return float(np.mean(sc)) if sc else 0.0


def dp_align(ev_q, ev_mid, ev_leg, aud, j_lo, j_hi, T=None, free_start=True, free_end=True,
             w_time=1.0, skip_ev=1.0, skip_leg=0.12, skip_on=0.08, max_ev_skip=3, max_on_skip=10):
    """Subsequence alignment of events (beat positions ev_q, pitch lists ev_mid) to onsets j_lo..j_hi-1.
    Returns (assign list event->onset index or -1, score)."""
    N = len(ev_q); J = np.arange(j_lo, j_hi); M = len(J)
    if N == 0 or M == 0:
        return [-1] * N, -1e9
    t = aud.times[J]
    strength = np.clip(aud.strength[J], 0, 2)
    P = np.concatenate([[0], np.cumsum(strength)])
    S = np.zeros((N, M))
    for i in range(N):
        for m in range(M):
            S[i, m] = aud.salience(J[m], ev_mid[i], ev_leg[i]) - (0.30 if not ev_leg[i] else 0.22)
    skipc = np.array([skip_leg if l else skip_ev for l in ev_leg])
    csk = np.concatenate([[0], np.cumsum(skipc)])
    NEG = -1e9
    D = np.full((N, M), NEG); back = np.full((N, M, 2), -1, int)
    for i in range(N):
        base = S[i].copy()
        D[i] = base - csk[i]          # event i is the first matched one (earlier events skipped)
        for di in range(1, max_ev_skip + 2):
            ip = i - di
            if ip < 0:
                break
            ev_pen = csk[i] - csk[ip + 1]
            for k in range(1, max_on_skip + 1):
                if k >= M:
                    break
                prev = D[ip, :-k]
                cand = prev + base[k:] - ev_pen - skip_on * (P[np.arange(k, M)] - P[np.arange(1, M - k + 1)])
                if T is not None:
                    # gap in 16th slots: observed vs. what the tab spacing suggests
                    exp16 = max(0.5, 4 * (ev_q[i] - ev_q[ip]))
                    obs16 = (t[k:] - t[:-k]) / (T / 4)
                    err = (obs16 - exp16) / (0.45 + 0.15 * exp16)
                    cand = cand - w_time * np.minimum(3.0, 0.5 * err * err)
                else:
                    dt = t[k:] - t[:-k]
                    cand = cand - np.where(dt < 0.03 * di, 1.0, 0.0)
                better = cand > D[i, k:]
                if better.any():
                    idx = np.flatnonzero(better) + k
                    D[i, idx] = cand[better]
                    back[i, idx, 0] = ip; back[i, idx, 1] = idx - k
    # end
    best, bi, bm = NEG, -1, -1
    for i in range(N):
        if not free_end and i < N - 1 - max_ev_skip:
            continue
        tail = csk[N] - csk[i + 1]
        m = int(np.argmax(D[i])); v = D[i, m] - tail
        if v > best:
            best, bi, bm = v, i, m
    assign = [-1] * N
    i, m = bi, bm
    while i >= 0 and m >= 0:
        assign[i] = int(J[m])
        i, m = back[i, m]
    return assign, float(best)


def fit_tempo(ev_q, assign, aud):
    pts = [(ev_q[i], aud.times[j]) for i, j in enumerate(assign) if j >= 0]
    if len(pts) < 4:
        return None
    q, t = np.array(pts).T
    if q.max() - q.min() < 1:
        return None
    # robust slope: median of pairwise slopes between points a few apart
    sl = []
    for a in range(len(q)):
        for b in range(a + 1, min(len(q), a + 12)):
            if q[b] - q[a] >= 0.5:
                sl.append((t[b] - t[a]) / (q[b] - q[a]))
    return float(np.median(sl)) if sl else None


def align_events(events, aud, t_lo=None, t_hi=None, free_start=True, free_end=True):
    """Two passes: pitch-only, then with the fitted tempo. Returns (assign, score, T)."""
    times = aud.times
    j_lo = int(np.searchsorted(times, t_lo)) if t_lo is not None else 0
    j_hi = int(np.searchsorted(times, t_hi)) if t_hi is not None else len(times)
    idx = [i for i, e in enumerate(events) if not e.get("glide")]
    sub = [events[i] for i in idx]
    q = [e["q"] for e in sub]; mid = [e["midi"] for e in sub]; leg = [e["legato"] for e in sub]

    def full(a):
        out = [-1] * len(events)
        for i, j in zip(idx, a):
            out[i] = j
        return out
    a1, s1 = dp_align(q, mid, leg, aud, j_lo, j_hi, None, free_start, free_end)
    T = fit_tempo(q, a1, aud)
    if T is None or T <= 0.05 or T > 3:
        return full(a1), s1, T
    a2, s2 = dp_align(q, mid, leg, aud, j_lo, j_hi, T, free_start, free_end)
    T2 = fit_tempo(q, a2, aud) or T
    return full(a2), s2, T2


def quality(events, assign, aud):
    m = [(e, j) for e, j in zip(events, assign) if j >= 0]
    n = sum(1 for e in events if not e.get("glide"))
    if not n:
        return 0, 0
    frac = len(m) / n
    sal = np.mean([aud.salience(j, e["midi"], e["legato"]) for e, j in m]) if m else 0
    return frac, float(sal)
