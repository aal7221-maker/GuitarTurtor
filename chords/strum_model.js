// Strum / pick timeline generator. ES module, no dependencies (pattern data is passed in).
//   import { strumTimeline, pickTimeline, progressionTimeline } from './strum_model.js';
//   const strokes = strumTimeline(shape, pattern, 90, { bars: 2, patterns });   // pattern = object from strum_patterns.json
// shape: fret per string, low E -> high e, e.g. ['x',3,2,0,1,0]  (x / -1 / null = muted).
// Stroke: { t (s), dir 'D'|'U'|'P'(pluck), strings:[0=low E..5] in play order, offsets:[s per string], frets:[..],
//           velocity 0..1, muted:false|'palm'|'dead', dur (s, hand travel time), step, bar }
// Per-string offsets: ~5-12 ms between adjacent strings (tighter at high tempo / light touch). D goes low->high, U high->low.
// Up strokes hit the top 3 strings (G B e) unless the pattern's stroke_strings set says otherwise (e.g. "up4").
// Performer drives the pick from stroke.t and dur; the synth plays string strings[k] at t + offsets[k].
// opts: bars | beats (length; beats may be fractional, e.g. 2 or 1.5), startT, humanize (0..1, default .15), seed, palmMute,
//       patterns (whole JSON, for its stroke_strings sets).
// A chord's pattern starts at its first step on the chord change (the way a teacher counts "for each chord: D D D U").

const fretOf = (shape, s) => { const f = shape[s]; return (f === 'x' || f === -1 || f == null) ? null : f; };
const DEFAULT_SETS = {
  default: { D: [0, 1, 2, 3, 4, 5], U: [3, 4, 5] },
  up4: { D: [0, 1, 2, 3, 4, 5], U: [2, 3, 4, 5] },
  top3: { D: [3, 4, 5], U: [3, 4, 5] },
  bass_then_top: { D: [0], d: [1, 2, 3, 4, 5], U: [3, 4, 5] },
  bassmix: { D: [0, 3, 4, 5], U: [3, 4, 5] },
};

export function parseToken(tok) {
  const m = /^([DUdux-])(>?)(p?)$/.exec(tok);
  if (!m) throw new Error('bad strum token: ' + tok);
  const [, c, acc, pm] = m;
  if (c === '-') return { rest: true };
  const dead = c === 'x';
  return { rest: false, dir: dead ? 'D' : c.toUpperCase(), light: c === 'd' || c === 'u', accent: !!acc, palm: !!pm, dead, letter: c };
}

// tiny deterministic RNG so tests and replays are repeatable
function rng(seed) { let s = (seed >>> 0) || 1; return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296); }

/** pattern tokens (strum grid or pick steps) */
export const patternTokens = p => (p.type === 'pick' ? p.steps : p.grid.split(/\s+/));
/** number of steps that start inside `beats` beats of pattern p */
const stepCount = (p, beats) => Math.max(0, Math.ceil(beats / p.step_beats - 1e-6));
/** length of a progression entry in beats: c.beats, else c.bars (default 1) bars of its own pattern */
export function chordBeats(c, pattern) {
  if (Number.isFinite(c.beats) && c.beats > 0) return c.beats;
  return (c.bars || 1) * ((c.pattern && c.pattern.beats_per_bar) || pattern.beats_per_bar);
}

/** the stroke_strings set a pattern uses (built-in sets, overridden by the JSON's) */
function stringSet(pattern, patterns) {
  const sets = { ...DEFAULT_SETS, ...((patterns && patterns.stroke_strings) || {}) };
  return sets[pattern.stroke_strings || 'default'] || sets.default;
}

/**
 * One strum stroke on `shape` at time t. tk = parseToken() result (or {dir, letter, accent, light, dead, palm}).
 * ctx: {bpm, stepDur, set | (pattern + patterns), rand, humanize, palmMute, velocity (overrides the token's), jitter, step, bar}.
 */
export function strumStroke(shape, tk, t, ctx) {
  const { bpm, stepDur, rand = rng(1), humanize = 0.15, palmMute = false, pattern = {} } = ctx;
  const set = ctx.set || stringSet(pattern, ctx.patterns);
  const sounding = [0, 1, 2, 3, 4, 5].filter(s => fretOf(shape, s) !== null);
  const wanted = tk.dead ? sounding : (set[tk.letter] || set[tk.dir] || DEFAULT_SETS.default[tk.dir]);
  let strings = wanted.filter(s => sounding.includes(s));
  if (!strings.length) strings = sounding.slice(-2);          // e.g. an upstroke on a shape with only low strings
  let vel = tk.dead ? 0.3 : tk.light ? 0.45 : tk.dir === 'U' ? 0.6 : 0.8;
  if (tk.accent) vel = Math.min(1, vel + 0.2);
  if (Number.isFinite(ctx.velocity)) vel = ctx.velocity;
  vel = Math.min(1, Math.max(0.05, vel * (1 + humanize * (rand() - 0.5) * 0.4)));
  // inter-string gap: 5-12 ms down, 4-9 ms up; tighter at fast tempo and light touch
  const speed = Math.min(1, bpm / 160);
  const base = tk.dir === 'D' ? 0.012 - 0.005 * speed : 0.009 - 0.004 * speed;
  const gap = base * (1.15 - 0.3 * vel);
  const order = strings.slice().sort((a, c) => (tk.dir === 'D' ? a - c : c - a));
  const offsets = order.map((_, k) => k * gap * (1 + humanize * (rand() - 0.5) * 0.2));
  const dur = Math.max(0.05, Math.min(0.16, stepDur * 0.8));
  const jit = humanize * (rand() - 0.5) * 0.012;               // up to ~1 ms timing jitter at default humanize
  if (ctx.jitter !== false) t += jit;
  return { t, dir: tk.dir, strings: order, offsets, frets: order.map(s => fretOf(shape, s)), velocity: vel,
           muted: tk.dead ? 'dead' : (tk.palm || palmMute || pattern.pm) ? 'palm' : false, dur, step: ctx.step ?? 0, bar: ctx.bar ?? 0 };
}

export function strumTimeline(shape, pattern, bpm, opts = {}) {
  const { bars = 1, startT = 0, humanize = 0.15, seed = 1, palmMute = false, patterns } = opts;
  const rand = rng(seed), beat = 60 / bpm;
  const stepDur = pattern.step_beats * beat;
  const set = stringSet(pattern, patterns);
  const toks = pattern.grid.split(/\s+/);
  const n = opts.beats != null ? stepCount(pattern, opts.beats) : bars * toks.length;
  const out = [];
  for (let j = 0; j < n; j++) {
    const i = j % toks.length, b = Math.floor(j / toks.length);
    const tk = parseToken(toks[i]);
    if (tk.rest) continue;
    let t = startT + j * stepDur;
    if (pattern.swing != null && pattern.subdivision === '8th' && i % 2 === 1) t += pattern.swing * stepDur;
    out.push(strumStroke(shape, tk, t, { bpm, stepDur, set, rand, humanize, palmMute, pattern, step: i, bar: b }));
  }
  return out;
}

// Fingerpicking: roles p = lowest sounding bass string, p2 = next bass string, i = G(3), m = B(4), a = e(5); 'a+b' plucks together.
export function pickTimeline(shape, pattern, bpm, opts = {}) {
  const { bars = 1, startT = 0, humanize = 0.1, seed = 1 } = opts;
  const rand = rng(seed), stepDur = pattern.step_beats * 60 / bpm;
  const snd = [0, 1, 2, 3, 4, 5].filter(s => fretOf(shape, s) !== null);
  const bass = snd.filter(s => s <= 3);
  const role = { p: bass[0] ?? snd[0], p2: bass[1] ?? bass[0] ?? snd[0], i: 3, m: 4, a: 5 };
  const N = pattern.steps.length, n = opts.beats != null ? stepCount(pattern, opts.beats) : bars * N;
  const out = [];
  for (let j = 0; j < n; j++) {
    const i = j % N, b = Math.floor(j / N);
    const strings = pattern.steps[i].split('+').map(r => role[r]).filter(s => fretOf(shape, s) !== null);
    if (!strings.length) continue;
    out.push({ t: startT + j * stepDur + humanize * (rand() - 0.5) * 0.01, dir: 'P', strings,
               offsets: strings.map((_, k) => k * 0.004), frets: strings.map(s => fretOf(shape, s)),
               velocity: strings[0] <= 2 ? 0.7 : 0.55, muted: false, dur: 0.06, step: i, bar: b });
  }
  return out;
}

// prog = [{ shape, bars? | beats?, pattern? }]; one concatenated timeline, each stroke tagged with `chord` (index into prog).
// A chord lasts chordBeats(c) beats (half beats allowed) and is played with its own pattern when it has one.
export function progressionTimeline(prog, pattern, bpm, opts = {}) {
  const beat = 60 / bpm;
  let t = opts.startT || 0; const out = [];
  prog.forEach((c, k) => {
    const p = c.pattern || pattern, beats = chordBeats(c, pattern), nb = beats / p.beats_per_bar;
    const f = p.type === 'pick' ? pickTimeline : strumTimeline;
    const len = Math.abs(nb - Math.round(nb)) < 1e-9 ? { bars: Math.round(nb) } : { beats };
    f(c.shape, p, bpm, { ...opts, ...len, startT: t, seed: (opts.seed || 1) + k })
      .forEach(s => out.push({ ...s, chord: k }));
    t += beats * beat;
  });
  return out;
}
