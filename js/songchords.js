// A song played as chords + strumming, "as in the video" (no DOM; main.js shows it, tools can test it).
// buildMixedTimeline (bottom) is what the page plays: ONE timeline per song; strummed bars use the chord song (strum hand +
// chord shapes) or the tab's own stacked chords, the other bars the tab's single notes.
//   const r = buildSongChords({ chords, strum, grid, bpm, mode }, chordShapes, strumPatterns, { patternId });
//   r = { cs /* js/chordsong.js chord song, recording clock */, meta: {patternId, pattern, tempo, detected, synthetic, ...} }
// Input: score.chords / score.strum (browser-made songs) or data/chords-<id>.json (site songs), docs/CHORD-DATA.md.
//  - With a detected strum: bars follow the strum's own grid. The recording's tempo drifts, so a time warp maps the
//    nominal (constant-tempo) pattern clock onto the recording: one knot per bar, taken from the detected strokes. Chords
//    are mapped onto the nominal clock, strumProgression() picks the chord of each bar, buildChordSong() builds the song
//    from the first chord bar (= shifting a LEAD_IN-based song by sp.startT - LEAD_IN) and warps it back; every stroke
//    then sits at its detected time (score.strum.strokes[].t) with the detected direction.
//  - Without one: the score's beat grid (score.grid) gives the knots, the meter (4/4, 3/4 or 6/8) comes from where the chord
//    changes fall, and the pattern is suggested by tempo and style (chordsong.suggestPattern: fingerstyle songs get a
//    fingerpicking pattern, slow songs a ballad pattern).
import { strumProgression, beatPos } from './strumdetect.js';
import { buildChordSong, suggestPattern, shapeNotes, fretLead } from './chordsong.js';
import { playedChordName } from './chord-agreement.js';
import { prepareTimeline } from './timeline.js';
import { buildMixedStrumPath } from './strumhand.js';

const median = a => { const s = [...a].sort((x, y) => x - y); return s.length ? s[Math.floor((s.length - 1) / 2)] : 0; };
const mod = (a, n) => ((a % n) + n) % n;

/** chordal = a strum was detected, or at least a quarter of the played events are chords (js/chord-agreement.js) */
export function isChordal({ strum = null, events = [] } = {}) {
  if (strum && strum.pattern) return true;
  const ev = events.filter(e => (e.notes || e.n || []).length);
  if (ev.length < 8) return false;
  return ev.filter(e => playedChordName(e)).length / ev.length >= 0.25;
}

/**
 * Meter from the chord changes on a beat grid: share of change weight (incoming chord length, max 4 s) that lands on a
 * bar line, per beats-per-bar and phase. 4/4 unless 3/4 fits clearly better; 2 (6/8: two dotted-quarter beats) when
 * the changes fall every second slow beat but not on 4/4 bar lines.
 */
export function chooseMeter(chords, beats, tempo = 90) {
  const ch = [];
  for (let i = 1; i < chords.length; i++) if (chords[i].name !== chords[i - 1].name) ch.push({ b: beatPos(beats, chords[i].t0), w: Math.min(4, chords[i].t1 - chords[i].t0) });
  const fit = bpb => {
    let best = { f: 0, phase: 0 };
    const W = ch.reduce((a, c) => a + c.w, 0) || 1;
    for (let ph = 0; ph < bpb; ph++) {
      const f = ch.reduce((a, c) => a + (Math.abs(c.b - Math.round(c.b)) < 0.3 && mod(Math.round(c.b) - ph, bpb) === 0 ? c.w : 0), 0) / W;
      if (f > best.f + 1e-9) best = { f, phase: ph };
    }
    return best;
  };
  const f4 = fit(4), f3 = fit(3), f2 = fit(2);
  if (f3.f > f4.f + 0.15 && f3.f >= 0.5) return { bpb: 3, ...f3, fits: { f4: f4.f, f3: f3.f, f2: f2.f } };
  if (tempo < 90 && f2.f >= 0.6 && f4.f < 0.55) return { bpb: 2, ...f2, fits: { f4: f4.f, f3: f3.f, f2: f2.f } };
  return { bpb: 4, ...f4, fits: { f4: f4.f, f3: f3.f, f2: f2.f } };
}

/** monotone knots {pos (beats), t (s)} whose local tempo stays within 25 % of the nominal beat */
function cleanKnots(knots, beat) {
  const s = knots.filter(k => Number.isFinite(k.pos) && Number.isFinite(k.t)).sort((a, b) => a.pos - b.pos), out = [];
  for (const k of s) {
    const p = out[out.length - 1];
    if (p && (k.pos <= p.pos || k.t <= p.t)) continue;
    if (p) { const r = (k.t - p.t) / (k.pos - p.pos) / beat; if (r < 0.75 || r > 1.33) continue; }
    out.push(k);
  }
  return out;
}
function interp(K, x, kx, ky, slope) {
  if (x <= K[0][kx]) return K[0][ky] + (x - K[0][kx]) * slope;
  const L = K[K.length - 1]; if (x >= L[kx]) return L[ky] + (x - L[kx]) * slope;
  let lo = 0, hi = K.length - 1; while (hi - lo > 1) { const m = (lo + hi) >> 1; if (K[m][kx] <= x) lo = m; else hi = m; }
  return K[lo][ky] + (x - K[lo][kx]) * (K[hi][ky] - K[lo][ky]) / (K[hi][kx] - K[lo][kx]);
}
/** one knot per bar from detected strokes: bar start = median(stroke time - its offset in the bar) */
function strokeKnots(S, beat) {
  const bpb = S.beats_per_bar || 4, sb = S.step_beats || 0.5, bars = new Map();
  for (const k of S.strokes || []) {
    if (!Number.isFinite(k.bar) || !Number.isFinite(k.step)) continue;
    if (!bars.has(k.bar)) bars.set(k.bar, []);
    bars.get(k.bar).push(k.t - k.step * sb * beat);
  }
  return [...bars].map(([b, ts]) => ({ pos: b * bpb, t: median(ts) }));
}

/**
 * @param {{chords:object[], strum?:object, grid?:number[], bpm?:number, mode?:string}} src  (recording clock)
 * @param {object} lib chords/chord_shapes.json   @param {object} pats chords/strum_patterns.json
 * @param {{patternId?:string}} opts  a library pattern instead of the detected / suggested one
 * @returns {{cs:object, meta:object}|null}
 */
export function buildSongChords(src, lib, pats, { patternId = null } = {}) {
  const chords = (src.chords || []).filter(c => c && c.name && c.name !== 'N' && c.t1 > c.t0).slice().sort((a, b) => a.t0 - b.t0);
  if (!chords.length || !lib || !pats) return null;
  const byId = id => [...pats.strum, ...pats.pick].find(p => p.id === id) || null;
  const S = src.strum && src.strum.pattern && Number.isFinite(src.strum.tempo) ? src.strum : null;
  let pattern, tempo, tBar0, knots, meta;
  if (S) {
    tempo = S.tempo; pattern = S.pattern; tBar0 = S.t_bar0 ?? chords[0].t0;
    knots = strokeKnots(S, 60 / tempo);
    meta = { detected: true, synthetic: false, confidence: S.confidence ?? null, detectedGrid: S.grid || null, patternId: pattern.id };
  } else {
    let beats = Array.isArray(src.grid) && src.grid.length >= 8 ? src.grid.filter(Number.isFinite) : null;
    if (!beats) {
      const b = 60 / (src.bpm || 90), end = Math.max(...chords.map(c => c.t1));
      beats = []; for (let t = chords[0].t0 - 4 * b; t < end + 4 * b; t += b) beats.push(+t.toFixed(3));
    }
    tempo = 60 / median(beats.slice(1).map((t, i) => t - beats[i]));
    const m = chooseMeter(chords, beats, tempo);
    pattern = suggestPattern(pats, { bpm: tempo, beatsPerBar: m.bpb, style: src.mode === 'fingers' ? 'pick' : 'strum' });
    // bar line at or just before the first chord
    const bFirst = Math.round(beatPos(beats, chords[0].t0)), i0 = bFirst - mod(bFirst - m.phase, m.bpb);
    knots = beats.map((t, i) => ({ pos: i - i0, t }));
    tBar0 = interp(cleanKnots(knots, 60 / tempo), 0, 'pos', 't', 60 / tempo);
    meta = { detected: false, synthetic: true, meter: m.bpb === 2 ? '6/8' : m.bpb === 3 ? '3/4' : '4/4', meterFit: m.fits, patternId: pattern.id };
  }
  if (patternId && patternId !== meta.patternId && byId(patternId)) {
    pattern = byId(patternId);
    meta = { ...meta, detected: false, synthetic: false, chosen: true, patternId };
  }
  const beat = 60 / tempo, K = cleanKnots(knots, beat), NOM0 = tBar0;
  const warp = K.length >= 2 ? (tau => interp(K, (tau - NOM0) / beat, 'pos', 't', beat)) : (tau => tau);
  const unwarp = K.length >= 2 ? (t => NOM0 + interp(K, t, 't', 'pos', 1 / beat) * beat) : (t => t);
  const chordsN = chords.map(c => ({ ...c, t0: unwarp(c.t0), t1: unwarp(c.t1) }));
  const sp = strumProgression(chordsN, { pattern, tempo, beats_per_bar: pattern.beats_per_bar, t_bar0: NOM0 }, lib);
  if (!sp) return null;
  const useStrokes = !!(S && pattern === S.pattern && S.strokes && S.strokes.length);
  const cs = buildChordSong(sp.prog, sp.pattern, sp.bpm, {
    patterns: pats, startT: sp.startT, warp, unwarp, strokes: useStrokes ? S.strokes : null,
    strumHand: src.mode === 'pick' ? 'pick' : 'fingers', humanize: 0.1,
  });
  return { cs, meta: { ...meta, pattern, tempo, knots: K.length, exactStrokes: useStrokes } };
}

// ---------------------------------------------------------------------------------------------------------------
// Mixed songs: real songs strum chords in verses / choruses and pick single notes in riffs and solos. The song is cut
// into bars on its beat axis and every bar gets one kind:
//   'tabchord'  the tab itself stacks >= 3 notes on at least half of the bar's events: strummed, exact voicing from the tab
//   'strum'     the recording's strum (score.strum strokes, >= 2 in the bar) or a chord entry flagged {strum: true}:
//               strummed with the chord song (library shapes, detected stroke times), built by buildSongChords
//   'notes'     everything else: the tab's events, picked / plucked as before
// Songs without strummed bars come back unchanged ({mixed: false}): the tab events are used as they are.

const MIN_STACK = 3;
const near = (x, y, tol = 0.12) => Math.abs(x - y) < tol;

/** the song's beat axis: q(t) / t(q) in the tab's beats (score.grid when present, aligned to the tab's q), bars every 4 beats */
export function songBeatAxis(score, events = []) {
  const bpm = Number.isFinite(score.bpm) && score.bpm > 0 ? score.bpm : 120, beat = 60 / bpm;
  const ev = events.filter(e => Number.isFinite(e.t) && Number.isFinite(e.q));
  let K = [];
  const grid = Array.isArray(score.grid) ? score.grid.filter(Number.isFinite) : [];
  if (grid.length >= 2) {
    const G = grid.map((t, i) => ({ t, q: i }));
    const off = ev.length ? Math.round(median(ev.map(e => e.q - interp(G, e.t, 't', 'q', 1 / beat)))) : 0;
    K = G.map(k => ({ t: k.t, q: k.q + off }));
  } else for (const e of ev) {
    const l = K[K.length - 1];
    if (!l || (e.t > l.t + 1e-3 && e.q > l.q + 1e-3)) K.push({ t: e.t, q: e.q });
  }
  if (!K.length) K = [{ t: 0, q: 0 }];
  if (K.length === 1) K.push({ t: K[0].t + beat, q: K[0].q + 1 });
  return { q: t => interp(K, t, 't', 'q', 1 / beat), t: q => interp(K, q, 'q', 't', beat), beat, bpm };
}

/** one entry per 4-beat bar: {m, t0, t1, kind, events, strokes} */
export function classifyBars({ events = [], axis, strokes = [], chords = [] }) {
  const flagged = (chords || []).filter(c => c && c.strum && c.t1 > c.t0);
  const ts = [...events.map(e => e.t), ...strokes.map(s => s.t), ...flagged.map(c => c.t0)];
  const te = [...events.map(e => e.t + (e.dur || 0.1)), ...strokes.map(s => s.t + 0.1), ...flagged.map(c => c.t1)];
  if (!ts.length) return [];
  const m0 = Math.floor(axis.q(Math.min(...ts)) / 4 + 1e-6), m1 = Math.floor(axis.q(Math.max(...te) - 1e-3) / 4 + 1e-6);
  const bars = [];
  for (let m = m0; m <= m1; m++) {
    const t0 = axis.t(4 * m), t1 = axis.t(4 * m + 4);
    const E = events.filter(e => e.q >= 4 * m - 0.01 && e.q < 4 * m + 4 - 0.01);
    const S = strokes.filter(s => s.t >= t0 - 0.03 && s.t < t1 - 0.03);
    const stacked = E.filter(e => e.notes.length >= MIN_STACK).length;
    const flag = flagged.some(c => Math.min(c.t1, t1) - Math.max(c.t0, t0) >= 0.5 * (t1 - t0));
    const kind = E.length && stacked >= Math.max(1, Math.ceil(E.length / 2)) ? 'tabchord' : S.length >= 2 || flag ? 'strum' : 'notes';
    bars.push({ m, t0, t1, kind, events: E, strokes: S });
  }
  return bars;
}

/** a strummed tab event as a stroke (strings low-E-first, exact frets from the tab); 8th grid: D on beats, U on "and" */
function tabStroke(ev, sixteenth, bpm) {
  const fr = ev.q - Math.floor(ev.q);
  const dir = sixteenth ? (near(fr, 0) || near(fr, 1) || near(fr, 0.5) ? 'D' : 'U') : (near(fr, 0.5) ? 'U' : 'D');
  const vel = ev.accent ? 0.95 : dir === 'D' ? 0.8 : 0.62;
  const speed = Math.min(1, bpm / 160), gap = (dir === 'D' ? 0.012 - 0.005 * speed : 0.009 - 0.004 * speed) * (1.15 - 0.3 * vel);
  const notes = [...ev.notes].sort((a, b) => (dir === 'D' ? b.s - a.s : a.s - b.s));   // D: low E first
  return { t: ev.t, dir, strings: notes.map(n => 5 - n.s), offsets: notes.map((_, k) => k * gap), frets: notes.map(n => n.f), velocity: vel,
    muted: ev.pm ? 'palm' : false, dur: Math.max(0.05, Math.min(0.16, 0.8 * (ev.gap || 0.2))), step: 0, bar: 0, fromTab: true };
}
const voicingKey = ev => ev.notes.map(n => n.s + ':' + n.f).sort().join(',');
let shapeIndex = null, shapeLib = null;
/**
 * Fingers for a chord written in the tab (the one-finger-per-fret solver cannot finger a chord): the chord library's own
 * fingering when the exact voicing is a library shape, else chordsong.shapeNotes' assignment (fretted notes in fret order,
 * index first; more than 4 fretted notes = an index barre across the lowest fret). Sets ev.notes[].finger, ev.barre, ev.pos.
 */
function fingerChord(ev, lib) {
  const frets = [0, 1, 2, 3, 4, 5].map(sl => { const n = ev.notes.find(x => x.s === 5 - sl); return n ? n.f : 'x'; });
  if (lib && lib !== shapeLib) {
    shapeLib = lib; shapeIndex = new Map();
    for (const c of Object.values(lib.chords || {})) for (const sh of c.shapes || []) if (!shapeIndex.has(sh.frets.join(','))) shapeIndex.set(sh.frets.join(','), sh);
  }
  let shape = lib && shapeIndex.get(frets.join(','));
  if (!shape) {
    shape = { frets };
    const fretted = frets.map((f, sl) => ({ f, sl })).filter(x => x.f !== 'x' && x.f > 0);
    if (fretted.length > 4) {
      const lo = Math.min(...fretted.map(x => x.f)), on = fretted.filter(x => x.f === lo);
      if (on.length >= 2) shape.barre = { fret: lo, from: on[0].sl, to: Math.max(...fretted.map(x => x.sl)) };
    }
  }
  const sn = shapeNotes(shape);
  ev.notes = sn.notes.map(n => ({ ...ev.notes.find(m => m.s === n.s), ...n }));
  ev.barre = sn.barre;
  const fingered = ev.notes.filter(n => n.finger);
  if (fingered.length) ev.pos = sn.barre ? sn.barre.fret : Math.max(1, median(fingered.map(n => n.f - n.finger + 1)));
}

/**
 * @param {{score:object, events:object[], mode:'pick'|'fingers', track?:{chords:object[], strum?:object}, lib?:object, pats?:object}} a
 *   events: the tab's prepared events (prepareTimeline(solveFingering(normalizeScore(score)))), left untouched
 * @returns {{mixed:boolean, bars, segments, events?, strokes?, spans?, noteOnly?, end?, meta?}}
 *   events: the merged timeline (fresh objects) with events.strum (mixed strum path), events.strokes, events.spans
 *   spans: chord name / diagram data while chords sound [{t0, t1, symbol, notes, muted, barre, source}]
 *   noteOnly: the tab events outside the strummed bars (what the synth plays as notes)
 */
export function buildMixedTimeline({ score, events, mode = 'fingers', track = null, lib = null, pats = null }) {
  const axis = songBeatAxis(score, events);
  const detected = track && track.strum && Array.isArray(track.strum.strokes) ? track.strum.strokes : [];
  const bars = classifyBars({ events, axis, strokes: detected, chords: track ? track.chords : [] });
  let cs = null, meta = null;
  if (bars.some(b => b.kind === 'strum') && track && track.chords && track.chords.length && lib && pats) {
    try {
      const r = buildSongChords({ chords: track.chords, strum: track.strum, grid: score.grid, bpm: score.bpm, mode }, lib, pats);
      if (r) { cs = r.cs; meta = r.meta; }
    } catch (e) { console.warn('chord song', e); }
  }
  for (const b of bars) if (b.kind === 'strum' && !cs) b.kind = 'notes';
  const segments = [];
  for (const b of bars) {
    const l = segments[segments.length - 1];
    if (l && l.kind === b.kind) { l.t1 = b.t1; l.q1 = 4 * b.m + 4; l.bars.push(b); }
    else segments.push({ kind: b.kind, t0: b.t0, t1: b.t1, q0: 4 * b.m, q1: 4 * b.m + 4, bars: [b] });
  }
  if (!segments.some(s => s.kind !== 'notes')) return { mixed: false, bars, segments, axis };

  const bpm = axis.bpm, out = [], strokes = [], spans = [], parts = [];
  const cloneEv = e => ({ ...e, notes: e.notes.map(n => ({ ...n })) });
  const inSeg = (e, s) => e.q >= s.q0 - 0.01 && e.q < s.q1 - 0.01;
  const covered = e => segments.some(s => inSeg(e, s));
  for (const e of events) if (!covered(e)) out.push(cloneEv(e));   // notes outside the bar range (none in practice)
  for (const seg of segments) {
    if (seg.kind === 'notes') { for (const e of events) if (inSeg(e, seg)) out.push(cloneEv(e)); continue; }
    const segStrokes = [];
    if (seg.kind === 'tabchord') {
      const E = events.filter(e => inSeg(e, seg) && e.notes.length);
      const sixteenth = E.some(e => { const f = e.q - Math.floor(e.q); return near(f, 0.25, 0.08) || near(f, 0.75, 0.08); });
      let span = null;
      E.forEach(e => {
        const ev = cloneEv(e);
        ev.strummed = true;
        ev.symbol = e.notes.length >= MIN_STACK ? playedChordName(e) : null;
        if (e.notes.length >= MIN_STACK) fingerChord(ev, lib);
        const key = voicingKey(e);
        if (e.notes.length >= MIN_STACK && (!span || span.key !== key)) {
          if (span) { span.t1 = ev.t; ev.chordLead = fretLead(bpm); }   // a change of shape: the fingers leave a little early
          span = { t0: ev.t, t1: seg.t1, symbol: ev.symbol, notes: ev.notes.map(n => ({ s: n.s, f: n.f, finger: n.finger || 0 })),
            muted: [0, 1, 2, 3, 4, 5].filter(s => !ev.notes.some(n => n.s === s)), barre: ev.barre || null, source: 'tab', key, id: 'tab' + spans.length };
          spans.push(span);
        }
        const st = tabStroke(ev, sixteenth, bpm);
        st.chord = span ? span.id : 'tab-' + ev.t;
        ev.dir = st.dir;
        out.push(ev); segStrokes.push(st);
      });
    } else {
      cs.events.forEach((e, k) => {
        const c = cs.chords[k], end = Math.min(c.t1, seg.t1);
        if (!(c.t0 < seg.t1 - 0.03 && c.t1 > seg.t0 + 0.03)) return;
        const ev = cloneEv(e), t = Math.max(e.t, seg.t0 - 0.03);
        Object.assign(ev, { t, q: axis.q(t), dur: Math.max(0.2, end - t - 0.2), gap: end - t, strummed: true, chordLead: e.lead });
        delete ev.hand;
        out.push(ev);
        spans.push({ t0: t, t1: end, symbol: c.symbol, notes: c.notes, muted: c.muted, barre: c.barre, shape: c.shape, source: 'chords', chord: k });
      });
      for (const st of cs.strokes) if (st.t >= seg.t0 - 0.03 && st.t < seg.t1 - 0.03) segStrokes.push(st);
    }
    segStrokes.sort((a, b) => a.t - b.t);
    if (segStrokes.length) segStrokes[segStrokes.length - 1].ringUntil = seg.t1 + 0.25;
    strokes.push(...segStrokes);
    if (segStrokes.length) parts.push({ seg, strokes: segStrokes });
  }
  out.sort((a, b) => a.t - b.t || (a.strummed ? 1 : 0) - (b.strummed ? 1 : 0));
  out.bpm = events.bpm || bpm;
  prepareTimeline(out, mode);
  // chord-song events keep their fretting lead (tempo dependent) and barre handling (js/chordsong.js); the lead times stay
  // ordered for timeline.sample()'s binary search
  out.forEach((ev, i) => {
    const prev = out[i - 1];
    if (ev.chordLead != null) {
      ev.lead = Math.min(ev.chordLead, prev ? ev.t - prev.t + prev.lead - 1e-3 : 1);
      ev.leadH = Math.min(Math.max(ev.lead, ev.leadH), prev ? ev.t - prev.t + prev.leadH - 1e-3 : 1);
    }
    if (ev.barre && ev.strummed) {
      ev.byFinger[0] = ev.notes.find(n => n.s === ev.barre.tip && n.finger === 1) || ev.byFinger[0];
      ev.notes.forEach(n => { if (n.finger === 1 && n !== ev.byFinger[0]) n.barred = true; });
    }
  });
  // strum windows: approach from the last picked note (at most 0.3 s), follow-through until the next picked note's preparation
  const picked = out.filter(e => !e.strummed);
  for (const p of parts) {
    const first = p.strokes[0].t - 0.03, last = p.strokes[p.strokes.length - 1].t + 0.06;
    const before = picked.filter(e => e.t < first).pop(), after = picked.find(e => e.t > last);
    p.w0 = Math.max(first - 0.3, before ? before.t + 0.06 : -Infinity);
    p.w1 = Math.min(last + 0.15, after ? after.t - 0.13 : Infinity);
  }
  strokes.sort((a, b) => a.t - b.t);
  out.strum = buildMixedStrumPath(parts);
  out.strokes = strokes;
  out.spans = spans.sort((a, b) => a.t0 - b.t0);
  out.mixed = true;
  out.mode = mode;
  const noteOnly = events.filter(e => !segments.some(s => s.kind !== 'notes' && inSeg(e, s)));
  const end = Math.max(...out.map(e => e.t + e.dur), ...strokes.map(s => s.t + 0.5));
  return { mixed: true, bars, segments, axis, events: out, strokes, spans: out.spans, noteOnly, end, meta, cs };
}
