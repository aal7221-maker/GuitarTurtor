// Frozen notes-to-score implementation from 76dd208, before the phantom filter.
// Used only for paired regression/measurement; no model loading or network.
const OPEN = [64, 59, 55, 50, 45, 40], MAX_FRET = 22;
const playable = notes => {
  const frets = notes.map(([, f]) => f).filter(f => f > 0);
  return new Set(notes.map(([s]) => s)).size === notes.length &&
    notes.every(([s, f]) => Number.isInteger(s) && s >= 0 && s < 6 && Number.isInteger(f) && f >= 0 && f <= MAX_FRET) &&
    (!frets.length || Math.max(...frets) - Math.min(...frets) <= 4);
};

// Keep the heard pitches (including octaves), prefer a library shape, then search the six strings.
export function assignChord(pitches, lib = null) {
  const ps = [...new Set(pitches)].sort((a, b) => a - b), pcs = new Set(ps.map(p => p % 12));
  if (ps.length > 6 || pcs.size < 3) return null;
  const cost = ns => ns.reduce((v, [, f]) => v + f, 0) + Math.max(...ns.map(([, f]) => f)) * 2;
  let best = null, bestCost = Infinity;
  const consider = ns => {
    if (playable(ns) && cost(ns) < bestCost) { best = ns.map(n => n.slice()); bestCost = cost(ns); }
  };
  for (const chord of Object.values(lib?.chords || {})) {
    if (chord.pcs.length !== pcs.size || !chord.pcs.every(p => pcs.has(p))) continue;
    for (const shape of chord.shapes) {
      const ns = shape.frets.flatMap((f, low) => Number.isInteger(f) ? [[5 - low, f]] : []);
      const match = ps.map(p => ns.find(([s, f]) => OPEN[s] + f === p));
      if (match.every(Boolean)) consider(match);
    }
  }
  if (best) return best.sort((a, b) => a[0] - b[0]);
  const visit = (i, ns) => {
    if (i === ps.length) { consider(ns); return; }
    for (let s = 0; s < 6; s++) {
      const next = [...ns, [s, ps[i] - OPEN[s]]];
      if (playable(next)) visit(i + 1, next);
    }
  };
  visit(0, []);
  return best && best.sort((a, b) => a[0] - b[0]);
}

// Preserve the 50 ms melody skyline, but recognize the complete strum BEFORE
// applying its relative loudness filter. Model amplitudes vary between strings;
// that melody filter can otherwise discard the bass/third of a clear strum.
// Require distinct, sustained pitches and an exact playable assignment.
function skyline(raw, high = 0.75, lib = null) {
  const ev = raw.filter(n => n.p >= OPEN[5] && n.p <= OPEN[0] + MAX_FRET).sort((a, b) => a.t - b.t);
  const out = [];
  for (let i = 0; i < ev.length;) {
    let j = i; while (j < ev.length && ev[j].t - ev[i].t <= 0.05) j++;
    let end = j;
    while (end < ev.length && ev[end].t - ev[i].t <= .1 + 1e-9) end++;
    let strum = null;
    for (let k = end; k >= j; k--) {
      const sweep = ev.slice(i, k), span = sweep.at(-1).t - sweep[0].t;
      const monotonic = [1, -1].some(dir => sweep.every((n, x) => !x || dir * (n.p - sweep[x - 1].p) > 0));
      if (span > .09 + 1e-9 && !monotonic) continue;
      // Do not swallow a late quiet note, a repeated pitch, or a non-overlapping melody run.
      if (new Set(sweep.map(n => n.p)).size !== sweep.length ||
          !sweep.every(n => n.a >= .3 && n.e > sweep.at(-1).t)) continue;
      strum = assignChord(sweep.map(n => n.p), lib);
      if (strum) { j = k; break; }
    }
    const g = ev.slice(i, j), loud = Math.max(...g.map(n => n.a));
    const strong = g.filter(n => n.a >= loud - (1 - high) * Math.abs(loud));
    const top = strong.reduce((a, b) => (b.p > a.p ? b : a));
    const sounding = strong.filter(n => n.a >= 0.3 && n.e > strong[strong.length - 1].t);
    const chord = strum || assignChord(sounding.map(n => n.p), lib);
    out.push({ t: ev[i].t, p: top.p, a: top.a,
      ...(chord ? { chord, sustain: Math.max(...(strum ? g : sounding).map(n => n.e)) - ev[i].t } : {}) });
    i = j;
  }
  return out.filter((e, i, arr) => !(i && !e.chord && !arr[i - 1].chord && e.p === arr[i - 1].p && e.t - arr[i - 1].t < 0.13));
}

// String/fret by beam-search Viterbi following the hand (port of pitch-model/transcribe.py: assign/trans_cost):
// changing string costs 1 (+0.25 per string), moving the hand 0.08/fret (+0.3 beyond 4 frets), open strings keep the
// hand where it was, and a string just played open is not fretted next (drone idiom).
function fingering(events, beam = 64) {
  let beams = [{ c: 0, path: [], anc: null }];
  for (const e of events) {
    const cands = OPEN.map((o, s) => [s, e.p - o]).filter(([, f]) => f >= 0 && f <= MAX_FRET);
    const next = new Map();
    for (const b of beams) {
      const recentOpen = new Set(b.path.slice(-2).filter(([, f]) => f === 0).map(([s]) => s));
      for (const pos of cands) {
        const [s, f] = pos;
        let c = b.c + 0.01 * f;
        if (f > 0 && b.anc) {
          const [sa, fa] = b.anc;
          if (sa !== s) c += 1 + 0.25 * Math.abs(sa - s);
          c += 0.08 * Math.abs(fa - f) + (Math.abs(fa - f) > 4 ? 0.3 : 0);
          if (recentOpen.has(s)) c += 0.6;
        }
        const anc = f > 0 ? pos : b.anc, key = pos + '|' + anc;
        if (!next.has(key) || c < next.get(key).c) next.set(key, { c, path: b.path.concat([pos]), anc });
      }
    }
    beams = [...next.values()].sort((a, b) => a.c - b.c).slice(0, beam);
  }
  return beams[0].path;
}

// Tempo from onset autocorrelation at the beat plus half the beat (60–160 bpm, log-Gaussian prior around 100 bpm);
// phase by onset alignment. Gets Choo Lo (73) right and Rabba at half time; 3:2 / 4:3 confusions remain possible.
function beatGrid(times, end) {
  const res = 0.01, n = Math.ceil(end / res) + 1, x = new Float32Array(n);
  for (const t of times) { const k = Math.round(t / res); for (let d = -2; d <= 2; d++) if (k + d >= 0 && k + d < n) x[k + d] += 1 - Math.abs(d) / 3; }
  const ac = lag => { lag = Math.round(lag); let s = 0; for (let i = lag; i < n; i++) s += x[i] * x[i - lag]; return s / Math.max(1, n - lag); };
  let best = 0.6, bestS = -1;
  for (let lag = 38; lag <= 100; lag++) {
    let s = ac(lag) + 0.5 * ac(lag / 2);
    s *= Math.exp(-0.5 * (Math.log2(60 / (lag * res) / 100) / 0.6) ** 2);
    if (s > bestS) { bestS = s; best = lag * res; }
  }
  let phase = 0, bestP = -1;
  for (let ph = 0; ph < best; ph += 0.01) {
    let s = 0; for (const t of times) { const r = ((t - ph) / best) % 1; s += Math.cos(2 * Math.PI * r); }
    if (s > bestP) { bestP = s; phase = ph; }
  }
  const first = Math.min(...times);
  let g0 = phase + Math.floor((first - phase) / best + 1e-6) * best;
  const grid = [];
  for (let t = g0; t <= end + 4 * best; t += best) grid.push(+t.toFixed(3));
  return { period: best, grid };
}

// raw notes [{t,e,p,a}] (seconds into the audio) -> {bpm, grid, notes:[{t,q,d,n:[[string,fret]],sec}]}.
// offset = video time at the audio's start, so note times are video times.
export function toScore(raw, offset = 0, high = 0.75, lib = null) {
  const ev = skyline(raw, high, lib).map(e => ({ ...e, t: e.t + offset })).filter(e => e.t >= 0);
  if (!ev.length) throw new Error('No notes were found in this audio.');
  const pos = fingering(ev);
  const end = ev[ev.length - 1].t;
  const { period, grid } = beatGrid(ev.map(e => e.t), end);
  let prevQ = -1;
  const notes = ev.map((e, i) => {
    let q = Math.round((e.t - grid[0]) / period * 4) / 4;
    if (q <= prevQ) q = prevQ + 0.25;
    prevQ = q;
    return { t: +e.t.toFixed(3), q, n: e.chord || [pos[i].slice()],
      ...(e.chord ? { sustain: +e.sustain.toFixed(3) } : {}) };
  });
  notes.forEach((n, i) => {
    n.d = i + 1 < notes.length ? Math.max(1, Math.min(16, Math.round((notes[i + 1].q - n.q) * 4))) : 4;
    n.sec = 'Part ' + (Math.floor(n.q / 16) + 1);
  });
  return { bpm: Math.round(600 / period) / 10, grid, notes };
}
