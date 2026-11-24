// Strum-pattern inference from audio (ES module, no dependencies, browser + node). Schema: docs/CHORD-DATA.md.
//   import { detectStrum, strumForScore } from './strumdetect.js';
//   const s = detectStrum(y /*mono Float32Array 22.05 kHz*/, { chords /*score.chords from audio-chords.js*/, patterns /*strum_patterns.json*/ });
//   s = { pattern, match:{id,dist}, detected_grid, confidence, consistency, tempo, swing, subdivision, steps_per_bar, beats_per_bar,
//         bar_phase, t_bar0, chord_coverage, sections:[{bar0,bar1,t0,t1,grid,id,dist}], strokes:[{t,bar,step,beat,dir,dir_raw,accent,muted,vel}] }
// 1. Onsets: SuperFlux-style positive spectral flux (512-pt STFT, hop 5.8 ms, 120 Hz-8 kHz, highs weighted: strums are broadband),
//    adaptive threshold, weak bumps (< 0.3 x the 90th-percentile peak) dropped.
// 2. Direction per stroke (strokeFeatures): a down stroke reaches the low strings first, an up stroke the high strings; the
//    onset time of every partial (70 Hz-1 kHz) that gained energy is regressed on log-frequency (ms/octave, > 0 = down), plus
//    the share of new energy below 160 Hz (down strokes hit the bass strings). Combined with the hand's pendulum prior (down on
//    even grid steps, up on odd); finally strokes take the direction of the matched pattern step (raw value kept in dir_raw).
// 3. Strummed-ness: share of hits adding >= 2 pitch classes at once; below 0.7 (picked / fingerstyle) no pattern is given.
//    Muted/percussive: little new pitched energy 70 ms after the hit. Accent: flux > 1.3 x median; light: < 0.6 x median.
// 4. Beat grid (rhythm()) -> subdivision (8th / 16th / triplet) and swing -> steps; bar phase + pattern chosen jointly by
//    matching the per-bar majority pattern against strum_patterns.json (bar phase also voted by chord changes).

/* ---------- DSP: FFT, STFT, onset envelope, tempo + beat tracking (autocorrelation tempo, Ellis DP beats) ---------- */
export const SR = 22050;

const tw = new Map();
function twiddles(n) {
  let t = tw.get(n);
  if (!t) { t = { c: new Float64Array(n / 2), s: new Float64Array(n / 2) }; for (let i = 0; i < n / 2; i++) { t.c[i] = Math.cos(2 * Math.PI * i / n); t.s[i] = -Math.sin(2 * Math.PI * i / n); } tw.set(n, t); }
  return t;
}
export function fft(re, im) {
  const n = re.length, { c, s } = twiddles(n);
  for (let i = 1, j = 0; i < n; i++) {                       // bit reversal
    let bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit;
    if (i < j) { let x = re[i]; re[i] = re[j]; re[j] = x; x = im[i]; im[i] = im[j]; im[j] = x; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1, step = n / len;
    for (let i = 0; i < n; i += len) for (let k = 0; k < half; k++) {
      const wr = c[k * step], wi = s[k * step], a = i + k, b = a + half;
      const xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr;
      re[b] = re[a] - xr; im[b] = im[a] - xi; re[a] += xr; im[a] += xi;
    }
  }
}
const hann = n => Float64Array.from({ length: n }, (_, i) => 0.5 - 0.5 * Math.cos(2 * Math.PI * i / n));
export function stftFrames(y, n, hop, cb) {
  const w = hann(n), re = new Float64Array(n), im = new Float64Array(n), mag = new Float32Array(n / 2 + 1);
  const frames = Math.max(0, Math.floor((y.length - n) / hop) + 1);
  for (let f = 0; f < frames; f++) {
    const o = f * hop;
    for (let i = 0; i < n; i++) { re[i] = y[o + i] * w[i]; im[i] = 0; }
    fft(re, im);
    for (let k = 0; k <= n / 2; k++) mag[k] = Math.hypot(re[k], im[k]);
    cb(mag, f);
  }
  return frames;
}

// time (s) -> beat index (float) on a beat-time grid; extrapolates with the end intervals
export function beatPos(grid, t) {
  const n = grid.length; if (t <= grid[0]) return (t - grid[0]) / (grid[1] - grid[0]);
  if (t >= grid[n - 1]) return n - 1 + (t - grid[n - 1]) / (grid[n - 1] - grid[n - 2]);
  let lo = 0, hi = n - 1; while (hi - lo > 1) { const m = (lo + hi) >> 1; if (grid[m] <= t) lo = m; else hi = m; }
  return lo + (t - grid[lo]) / (grid[hi] - grid[lo]);
}

// Onset-strength envelope (hop 256 = 11.6 ms): positive flux of log-compressed spectrum, 100 Hz–8 kHz.
export function onsetEnvelope(y, sr = SR) {
  const n = 1024, hop = 256, lo = Math.round(100 / (sr / n)), hi = Math.round(8000 / (sr / n));
  let prev = null; const env = [];
  stftFrames(y, n, hop, mag => {
    const L = new Float32Array(hi); for (let k = lo; k < hi; k++) L[k] = Math.log1p(20 * mag[k]);
    let s = 0; if (prev) for (let k = lo; k < hi; k++) s += Math.max(0, L[k] - prev[k]);
    env.push(s); prev = L;
  });
  return { env: Float32Array.from(env), hop: hop / sr };
}

// tempo + beats. opts: {bpm: hint, min:60, max:180, center:100}
export function rhythm(y, sr = SR, opts = {}) {
  const { env, hop } = onsetEnvelope(y, sr), N = env.length;
  const { min = 60, max = 180, center = 100 } = opts;
  if (N < 200) throw new Error('audio too short for beat tracking');
  // normalise: subtract local mean (0.5 s), divide by std
  const loc = new Float32Array(N), W = Math.round(0.5 / hop); let acc = 0;
  for (let i = 0; i < N; i++) { acc += env[i]; if (i >= 2 * W) acc -= env[i - 2 * W]; loc[i] = acc / Math.min(i + 1, 2 * W); }
  const o = env.map((v, i) => Math.max(0, v - loc[Math.max(0, i - W)]));
  const sd = Math.sqrt(o.reduce((a, v) => a + v * v, 0) / N) || 1; for (let i = 0; i < N; i++) o[i] /= sd;
  const ac = lag => { const l = Math.round(lag); let s = 0; for (let i = l; i < N; i++) s += o[i] * o[i - l]; return s / (N - l); };
  let best = 0, bestS = -1, bpm = opts.bpm || 0;
  if (!bpm) {
    for (let b = max; b >= min; b -= 0.5) {
      const P = 60 / b / hop;
      let s = ac(P) + 0.5 * ac(2 * P) + 0.5 * ac(4 * P) + 0.25 * ac(P / 2);
      s *= Math.exp(-0.5 * (Math.log2(b / center) / 0.7) ** 2);
      if (s > bestS) { bestS = s; best = b; }
    }
    bpm = best;
  }
  const P = 60 / bpm / hop, tight = 100, score = new Float64Array(N), back = new Int32Array(N).fill(-1);
  for (let t = 0; t < N; t++) {
    let bs = 0, bi = -1;
    for (let p = t - Math.round(2 * P); p <= t - Math.round(P / 2); p++) {
      if (p < 0) continue;
      const d = Math.log((t - p) / P), v = score[p] - tight * d * d;
      if (bi < 0 || v > bs) { bs = v; bi = p; }
    }
    score[t] = o[t] + (bi >= 0 ? bs : 0); back[t] = bi;
  }
  // start from the best score among the last beat period of frames
  let t = N - 1, tb = N - 1; for (let i = N - 1; i > N - 1 - Math.round(P); i--) if (score[i] > score[tb]) tb = i;
  t = tb; const idx = [];
  while (t >= 0) { idx.push(t); t = back[t]; }
  idx.reverse();
  const beats = idx.map(i => i * hop + 512 / sr);       // + half of the 1024 window = frame centre
  const per = 60 / bpm;
  while (beats[0] - per >= 0) beats.unshift(beats[0] - per);
  const dur = y.length / sr; while (beats[beats.length - 1] + per < dur) beats.push(beats[beats.length - 1] + per);
  return { bpm, period: per, beats, env: o, hop };
}

/* ---------- strum analysis ---------- */

const median = a => { const s = Array.from(a).sort((x, z) => x - z); return s.length ? s[Math.floor(s.length / 2)] : 0; };
// Per-stroke direction evidence around onset time t.
//  lag:   onset-time vs pitch slope (ms per octave). 2048-pt spectra every 6 ms from t-50 to t+110 ms; every spectral peak
//         between 70 Hz and 1 kHz that gained energy gets an onset time (50% point of its rise); weighted regression of
//         onset time on log2(frequency). A down stroke reaches the low strings first -> slope > 0; an up stroke -> < 0.
//  ratio: log of NEW energy (after - before) below 160 Hz (E2/A2 region) over new energy 160-1200 Hz. Down strokes hit
//         the bass strings, up strokes mostly the top 3-4 strings.
export function strokeFeatures(y, sr, t) {
  const n = 2048, df = sr / n, re = new Float64Array(n), im = new Float64Array(n);
  const spec = c => { const o = Math.round(c * sr) - n / 2;
    for (let i = 0; i < n; i++) { const v = y[o + i] || 0; re[i] = v * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / n)); im[i] = 0; } fft(re, im);
    return Float64Array.from({ length: 1300 / df | 0 }, (_, k) => Math.hypot(re[k], im[k])); };
  const T = [], M = [];
  for (let c = t - 0.05; c <= t + 0.11; c += 0.006) { T.push(c); M.push(spec(c)); }
  const B = M[0], A = M[Math.round(0.12 / 0.006)];          // before: centred t-50 ms; after: centred t+70 ms
  let lo = 0, mid = 0, maxR = 0;
  for (let k = Math.round(60 / df); k < M[0].length; k++) { const e = Math.max(0, A[k] ** 2 - B[k] ** 2); if (k * df < 160) lo += e; else mid += e; }
  const rise = k => A[k] - B[k];
  for (let k = Math.round(70 / df); k < 1000 / df; k++) maxR = Math.max(maxR, rise(k));
  const pcR = new Float64Array(12);                  // new energy per pitch class: a strum adds >= 3 pitch classes at once, a picked note 1
  for (let k = Math.round(70 / df); k < 1000 / df; k++) if (rise(k) > 0 && A[k] > A[k - 1] && A[k] >= A[k + 1]) pcR[((Math.round(69 + 12 * Math.log2(k * df / 440)) % 12) + 12) % 12] += rise(k);
  const pcMax = Math.max(...pcR), npc = pcMax > 0 ? pcR.filter(v => v >= 0.25 * pcMax).length : 0;
  let sw = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (let k = Math.round(70 / df); k < 1000 / df; k++) {
    if (!(A[k] > A[k - 1] && A[k] >= A[k + 1]) || rise(k) < 0.15 * maxR) continue;
    const th = B[k] + 0.5 * rise(k); let j = 0; while (j < M.length - 1 && M[j][k] < th) j++;
    const x = Math.log2(k * df), yv = T[j] * 1000, w = rise(k);
    sw += w; sx += w * x; sy += w * yv; sxx += w * x * x; sxy += w * x * yv;
  }
  const den = sw * sxx - sx * sx;
  return { npc, newE: lo + mid, lag: sw && den > 1e-9 ? (sw * sxy - sx * sy) / den : 0, ratio: Math.max(-5, Math.min(3, Math.log((lo + 1e-9) / (mid + 1e-9)))) };
}
function parseTok(t) { const m = /^([DUdux-])(>?)(p?)$/.exec(t) || []; return m[1] === '-' || !m[1] ? null : { dir: m[1] === 'x' ? 'X' : m[1].toUpperCase(), accent: !!m[2] }; }
const gcd = (a, b) => (b ? gcd(b, a % b) : a);

// cost between two step sequences (null = rest, {dir}) at their own resolutions g1, g2 steps/beat
function patDist(A, g1, B, g2) {
  const R = g1 * g2 / gcd(g1, g2), n = A.length * R / g1; let cost = 0, hits = 0;
  for (let i = 0; i < n; i++) {
    const a = i % (R / g1) ? null : A[i / (R / g1)], b = i % (R / g2) ? null : B[i / (R / g2)];
    if (!a && !b) continue; hits++;
    if (!a || !b) cost += 1; else if (a.dir !== b.dir) cost += (a.dir === 'X' || b.dir === 'X') ? 0.4 : 0.7;
  }
  return hits ? cost / hits : 1;
}

export function detectStrum(y, opts = {}) {
  const { sr = SR, beatsPerBar: bpb = 4, patterns = null, chords = null, motionPrior: MP = 1, muteRatio: MUTE = 0.2 } = opts;
  const rh = opts.rhythm || rhythm(y, sr), beats = rh.beats;
  // ---- 1. onsets
  const n = 512, hop = 128, df = sr / n, kLo = Math.round((opts.fluxLo ?? 120) / df), kHi = Math.round(8000 / df);
  // SuperFlux-style: compare with the frame 2 hops back, max-filtered over +-1 bin, so beating partials don't fire
  const flux = [], hist = [];
  stftFrames(y, n, hop, mag => {
    const L = new Float32Array(kHi + 1); let s = 0; const P = hist.length >= 2 ? hist[hist.length - 2] : null;
    for (let k = kLo; k < kHi; k++) { L[k] = Math.log1p(20 * mag[k]); }
    if (P) for (let k = kLo; k < kHi; k++) s += Math.max(0, L[k] - Math.max(P[k - 1], P[k], P[k + 1])) * (1 + k * df / 3000);
    flux.push(s); hist.push(L); if (hist.length > 3) hist.shift();
  });
  const N = flux.length, fsec = f => (f * hop + n / 2) / sr, W = Math.round(0.75 / (hop / sr)), sep = Math.round(0.07 / (hop / sr));
  const ref = Array.from(flux).sort((a, b) => a - b)[Math.floor(0.99 * N)] || 1;
  const gmed = median(flux), TH = opts.onsetThreshold ?? 0.15;
  const cs = new Float64Array(N + 1); for (let i = 0; i < N; i++) cs[i + 1] = cs[i] + flux[i];
  const peaks = [];
  for (let i = 1; i < N - 1; i++) {
    const lo = Math.max(0, i - W), hi = Math.min(N, i + W), loc = (cs[hi] - cs[lo]) / (hi - lo);
    if (flux[i] < loc + TH * (ref - gmed)) continue;
    let mx = true; for (let j = Math.max(0, i - sep); j <= Math.min(N - 1, i + sep); j++) if (flux[j] > flux[i] || (flux[j] === flux[i] && j < i)) { mx = false; break; }
    if (mx) peaks.push(i);
  }
  { const pf = peaks.map(i => flux[i]).sort((a, b) => a - b), p90 = pf[Math.floor(0.9 * pf.length)] || 0;   // drop weak bumps (beating, noise)
    for (let j = peaks.length - 1; j >= 0; j--) if (flux[peaks[j]] < 0.3 * p90) peaks.splice(j, 1); }
  const res = { pattern: null, match: null, detected_grid: '', confidence: 0, consistency: 0, tempo: +rh.bpm.toFixed(1), swing: 0, subdivision: '8th',
                steps_per_bar: 2 * bpb, beats_per_bar: bpb, bar_phase: 0, t_bar0: null, sections: [], strokes: [], method: 'js-superflux-sweepslope' };
  if (peaks.length < 8) return res;
  // ---- 2./3. per-stroke features
  const st = peaks.map(p => {
    const t = fsec(p) - 0.006, sf = strokeFeatures(y, sr, t);
    return { t, npc: sf.npc, newE: sf.newE, lag: sf.lag, ratio: sf.ratio, str: flux[p] };
  });
  const z = key => { const m = st.reduce((a, s) => a + s[key], 0) / st.length, sd = Math.sqrt(st.reduce((a, s) => a + (s[key] - m) ** 2, 0) / st.length) || 1; st.forEach(s => (s['z' + key] = (s[key] - m) / sd)); };
  z('lag'); z('ratio');
  const nE = st.map(s => s.newE).sort((a, b) => a - b), mNew = nE[Math.floor(0.75 * nE.length)], mStr = median(st.map(s => s.str));
  st.forEach(s => { s.feat = s.lag / 5 + 0.5 * s.zratio; s.muted = s.newE < MUTE * mNew; s.accent = s.str > 1.3 * mStr; s.light = s.str < 0.6 * mStr; s.vel = Math.min(1, 0.6 * s.str / mStr); });
  res.direction_method = MP ? 'sweep-slope+motion' : 'sweep-slope';
  // ---- 4. grid: subdivision + swing
  st.forEach(s => { s.pos = beatPos(beats, s.t); s.frac = s.pos - Math.floor(s.pos); });
  const near = (f, c, tol) => Math.abs(f - c) < tol;
  const n16 = st.filter(s => near(s.frac, 0.25, 0.08) || near(s.frac, 0.75, 0.08)).length;
  const off = st.filter(s => s.frac > 0.38 && s.frac < 0.8).map(s => s.frac), mo = off.length >= 4 ? median(off) : 0.5;
  let g = 2, swing = 0;
  if (mo > 0.58 && n16 < 0.25 * st.length) { swing = Math.min(0.5, 2 * (mo - 0.5)); if (swing >= 0.25) g = 3; }
  else if (n16 > 0.15 * st.length) g = 4;
  const unswing = f => (g !== 2 || !swing) ? f : f < 0.5 + swing / 2 ? f * 0.5 / (0.5 + swing / 2) : 0.5 + (f - 0.5 - swing / 2) * 0.5 / (0.5 - swing / 2);
  st.forEach(s => (s.gstep = Math.round((Math.floor(s.pos) + unswing(s.frac)) * g)));
  // direction = spectral evidence + the strumming hand's pendulum (down on even grid steps, up on odd; triplets: 0 D, 2 U)
  st.forEach(s => { const k = ((s.gstep % g) + g) % g, prior = g === 3 ? (k === 0 ? 1 : k === 2 ? -1 : 0) : (s.gstep % 2 === 0 ? 1 : -1);
    s.dir = s.feat + MP * prior >= 0 ? 'D' : 'U'; });
  res.swing = +swing.toFixed(2); res.subdivision = g === 3 ? 'triplet' : g === 4 ? '16th' : '8th';
  const S = bpb * g; res.steps_per_bar = S;
  const lib = patterns ? patterns.strum.filter(p => p.type === 'strum' && p.beats_per_bar === bpb) : [];
  const libSteps = lib.map(p => ({ p, g: Math.round(p.steps_per_bar / bpb), A: p.grid.split(/\s+/).map(parseTok) }));
  // per-bar step arrays for a phase, majority pattern, best library match
  const analyse = (phase, from = -Infinity, to = Infinity) => {
    const bars = new Map();
    for (const s of st) {
      const rel = s.gstep - phase * g, bar = Math.floor(rel / S), step = rel - bar * S;
      if (bar < from || bar >= to) continue;
      if (!bars.has(bar)) bars.set(bar, new Array(S).fill(null));
      const cell = bars.get(bar); if (!cell[step] || cell[step].str < s.str) cell[step] = s;
    }
    const act = [...bars.values()].filter(b => b.filter(Boolean).length >= 2); if (!act.length) return null;
    const cons = [];
    for (let i = 0; i < S; i++) {
      const h = act.map(b => b[i]).filter(Boolean);
      if (h.length < 0.5 * act.length) { cons.push(null); continue; }
      const cnt = d => h.filter(s => (s.muted ? 'X' : s.dir) === d).length, dir = ['D', 'U', 'X'].reduce((a, d) => (cnt(d) > cnt(a) ? d : a), 'D');
      cons.push({ dir, accent: h.filter(s => s.accent).length > h.length / 2, light: h.filter(s => s.light).length > h.length / 2 });
    }
    const consistency = act.reduce((a, b) => a + b.filter((s, i) => !!s === !!cons[i] && (!s || (s.muted ? 'X' : s.dir) === cons[i].dir)).length / S, 0) / act.length;
    let best = null; for (const L of libSteps) { const d = patDist(cons, g, L.A, L.g); if (!best || d < best.dist) best = { id: L.p.id, dist: +d.toFixed(3), p: L.p }; }
    const grid = cons.map(c => !c ? '-' : c.dir === 'X' ? 'x' : (c.light && !c.accent ? c.dir.toLowerCase() : c.dir) + (c.accent ? '>' : '')).join(' ');
    return { cons, grid, consistency, best, bars };
  };
  // bar phase vote from chord changes (chords = score.chords [{t0,t1,name}] from js/audio-chords.js): changes fall on bar lines
  const segs = Array.isArray(chords) ? chords : (chords && chords.segments) || [];
  const votes = new Float64Array(bpb); segs.slice(1).forEach(c => (votes[((Math.round(beatPos(beats, c.t0)) % bpb) + bpb) % bpb] += Math.min(4, c.t1 - c.t0)));
  const chordVote = segs.length > 2 ? votes.indexOf(Math.max(...votes)) : null;
  let best = null;
  for (let ph = 0; ph < bpb; ph++) {
    const a = analyse(ph); if (!a) continue;
    a.phase = ph; a.score = (a.best ? a.best.dist : 0.5) - 0.3 * a.consistency - (ph === chordVote ? 0.06 : 0);
    if (!best || a.score < best.score - 1e-9) best = a;
  }
  if (!best) return res;
  const ph = best.phase, beat0 = beats[0] + 0, firstBar = Math.floor((st[0].gstep - ph * g) / S);
  res.bar_phase = ph; res.detected_grid = best.grid; res.consistency = +best.consistency.toFixed(2);
  const bi = firstBar * bpb + ph; res.t_bar0 = +(bi >= 0 && bi < beats.length ? beats[bi] : beat0 + bi * rh.period).toFixed(3);
  res.match = best.best ? { id: best.best.id, dist: best.best.dist } : null;
  const dist = best.best ? best.best.dist : 1;
  res.confidence = +((1 - dist) * best.consistency).toFixed(2);
  const custom = { id: 'custom', name: 'Detected pattern', time_sig: bpb + '/4', subdivision: res.subdivision === '16th' ? '16th' : '8th', steps_per_bar: S,
    beats_per_bar: bpb, step_beats: 1 / g, grid: best.grid, bpm: [Math.round(rh.bpm), Math.round(rh.bpm)], difficulty: 3, songs: [],
    description: 'Strum pattern detected from the recording' + (res.match ? ` (closest: ${res.match.id})` : ''), type: 'strum', swing: swing || undefined };
  res.pattern = dist <= 0.25 ? { ...best.best.p, swing: swing || best.best.p.swing } : custom;
  // strummed-ness: share of hits that add >= 2 pitch classes at once (fingerpicking / single-note lines stay low)
  res.strummed = +(st.filter(s => s.npc >= 2).length / st.length).toFixed(2);
  if (res.pattern && res.confidence < (opts.minConfidence ?? 0.4)) { res.pattern = null; res.note = 'no consistently repeating strum pattern'; }
  if (res.strummed < (opts.minStrummed ?? 0.7)) { res.pattern = null; res.note = 'hits are mostly single notes (picked / fingerstyle): no strum pattern'; }
  // the chord detector abstains on windows without >= 3 strong pitch classes: little chord coverage = melody / riff, not strumming
  const covered = segs.reduce((a, c) => a + c.t1 - c.t0, 0), span = st[st.length - 1].t - st[0].t + 0.5;
  res.chord_coverage = chords ? +(covered / span).toFixed(2) : null;
  if (chords && covered < 0.3 * span && res.pattern) { res.pattern = null; res.note = 'little chordal audio (single notes / riff): no strum pattern'; }
  // sections of 8 bars (pattern may change between verse and chorus)
  const nbars = Math.floor((st[st.length - 1].gstep - ph * g) / S) + 1;
  for (let b = firstBar; b < nbars; b += 8) {
    const a = analyse(ph, b, b + 8); if (!a) continue;
    const t = k => { const i = k * bpb + ph; return +(i >= 0 && i < beats.length ? beats[i] : beats[0] + i * rh.period).toFixed(3); };
    res.sections.push({ bar0: b - firstBar, bar1: Math.min(b + 8, nbars) - firstBar, t0: t(b), t1: t(Math.min(b + 8, nbars)), grid: a.grid, id: a.best && a.best.dist <= 0.25 ? a.best.id : 'custom', dist: a.best ? a.best.dist : 1 });
  }
  res.strokes = st.map(s => { const rel = s.gstep - ph * g, bar = Math.floor(rel / S);
    const step = rel - bar * S, c = res.pattern && best.cons[step];   // matched pattern step -> its direction (raw kept in dir_raw)
    return { t: +s.t.toFixed(3), bar: bar - firstBar, step, beat: +s.pos.toFixed(2), dir: c && c.dir !== 'X' && dist <= 0.25 ? c.dir : s.dir, dir_raw: s.dir,
             accent: s.accent, muted: s.muted, vel: +s.vel.toFixed(2) }; });
  return res;
}

/* ---------- site integration ---------- */
let patternsP = null;
/** chords/strum_patterns.json (cached) */
export function loadStrumPatterns() {
  return patternsP ||= fetch(new URL('../chords/strum_patterns.json', import.meta.url)).then(r => (r.ok ? r.json() : null)).catch(() => null);
}
/**
 * Compact `score.strum` (docs/CHORD-DATA.md) for a recording, or null when there is no usable strum pattern
 * (not chordal, too few strokes). chords = score.chords ([{t0,t1,name}], recording clock). Times are seconds in the recording.
 */
export function strumForScore(y, chords, patterns, opts = {}) {
  const r = detectStrum(y, { ...opts, chords, patterns });
  if (!r.pattern || !r.strokes.length) return null;
  const p = r.pattern;
  return { pattern: p, id: p.id, grid: r.detected_grid, steps_per_bar: r.steps_per_bar, beats_per_bar: r.beats_per_bar, step_beats: 1 / (r.steps_per_bar / r.beats_per_bar),
    subdivision: r.subdivision, swing: r.swing, tempo: r.tempo, t_bar0: r.t_bar0, confidence: r.confidence, match: r.match,
    sections: r.sections.map(({ t0, t1, id, grid }) => ({ t0, t1, id, grid })),
    strokes: r.strokes.map(({ t, dir, accent, muted, step, bar, vel }) => ({ t, dir, step, bar, vel, ...(accent ? { accent } : {}), ...(muted ? { muted } : {}) })) };
}

/**
 * Detected chords + strum -> input for chordsong.buildChordSong(prog, pattern, bpm) so chord mode can play the song "as recorded".
 * Bars follow the strum's grid (t_bar0, tempo, beats_per_bar); each bar takes the chord that covers most of it; repeated bars
 * merge. Every symbol is resolved to a chord_shapes.json shape, so the name, diagram, fretting hand, strummed strings and synth
 * notes all come from the same shape (tools/chordcheck.mjs verifies this). Bars with no chord (or a symbol the library lacks)
 * keep the previous chord. Returns { prog:[{symbol,shape,bars,t0}], pattern, bpm, startT } or null.
 */
export function strumProgression(chords, strum, lib, { shapeIndex = 0 } = {}) {
  if (!strum || !strum.pattern || !chords || !chords.length) return null;
  const bpb = strum.beats_per_bar || 4, bpm = strum.tempo, barDur = bpb * 60 / bpm, t0 = strum.t_bar0 ?? chords[0].t0;
  const end = Math.max(...chords.map(c => c.t1)), ENH = { 'C#': 'Db', 'D#': 'Eb', 'G#': 'Ab', 'A#': 'Bb', Gb: 'F#' };
  const canon = s => s.replace(/^([A-G][#b]?)/, m => ENH[m] || m);
  const prog = []; let prev = null;
  for (let t = t0; t < end - 0.25 * barDur; t += barDur) {
    const cov = {}; for (const c of chords) { const ov = Math.min(t + barDur, c.t1) - Math.max(t, c.t0); if (ov > 0) cov[canon(c.name)] = (cov[canon(c.name)] || 0) + ov; }
    let sym = Object.keys(cov).sort((a, b) => cov[b] - cov[a]).find(s => lib.chords[s]?.shapes?.length) || prev;
    if (!sym) continue;
    if (prev === sym) prog[prog.length - 1].bars++;
    else prog.push({ symbol: sym, shape: lib.chords[sym].shapes[Math.min(shapeIndex, lib.chords[sym].shapes.length - 1)], bars: 1, t0: +t.toFixed(3) });
    prev = sym;
  }
  return prog.length ? { prog, pattern: strum.pattern, bpm, startT: prog[0].t0 } : null;
}
