// Chord "song" builder (no DOM, so tools/strumcheck.mjs can use it): a progression + strum pattern + tempo
// becomes (1) timeline events in the format timeline.js / performer.js already understand (one event per chord:
// notes with finger numbers; the fretting hand holds the shape), (2) the stroke list for the strumming hand and
// the synth, and (3) the strum path used by the performer.
//
// Progression entries: { symbol, shape, bars? | beats?, pattern? }. `beats` gives a chord's length in beats (halves allowed,
// e.g. G:2 D:2 Am:4); `pattern` (object or id) overrides the song's pattern for that chord, which restarts on the change.
// Songs played "as recorded" pass opts.warp/unwarp (nominal beat clock <-> recording clock) and opts.strokes (detected
// strokes, recording clock): chord changes follow the recording's beat and every stroke sits at its detected time.
//
// String order: chords/ data is LOW-E-FIRST; the events produced here use the 3D code's order (string 0 = high e).
import { progressionTimeline, strumStroke, chordBeats } from '../chords/strum_model.js';
import { prepareTimeline } from './timeline.js';
import { buildStrumPath } from './strumhand.js';

export const LEAD_IN = 0.2;      // s of silence before the first stroke
/**
 * How long before a chord change the fretting hand leaves the old shape: about 0.3 s at slow tempos, 0.14 s at 100 bpm,
 * 0.08 s when fast. A stroke inside that window (typically the last up-strum) sounds the open strings: the fingers are
 * already lifted and travelling to the next shape, as players do.
 */
export function fretLead(bpm) { return Math.min(0.32, Math.max(0.08, 0.36 * 60 / bpm - 0.076)); }
export const FRET_LEAD = fretLead(100);   // reference value; buildChordSong uses fretLead(bpm)
const isMuted = f => f === 'x' || f === -1 || f == null;
const ENH = { 'C#': 'Db', 'D#': 'Eb', 'G#': 'Ab', 'A#': 'Bb', Gb: 'F#' };

/** 'C#m7' -> 'Dbm7' (the chord library spells roots C Db D Eb E F F# G Ab A Bb B) */
export function canonicalSymbol(sym) {
  const m = /^([A-G][#b]?)(.*)$/.exec(String(sym).trim());
  return m ? (ENH[m[1]] || m[1]) + m[2] : String(sym).trim();
}
export const shapesOf = (lib, sym) => (lib.chords[canonicalSymbol(sym)] || { shapes: [] }).shapes;

/**
 * Fingering for a library shape (frets low->high). Uses the library's finger numbers when they are a clean
 * one-finger-per-note assignment, otherwise fingers go to the fretted notes in fret order (index first).
 * A barre (>= 3 strings at the same fret inside shape.barre) is the index finger lying across them; the
 * fingertip target is its treble-most string, the other strings are pressed by the same finger.
 * @returns {{notes:{s:number,f:number,finger:number}[], barre:null|{fret:number,strings:number[],tip:number}, muted:number[]}}
 */
export function shapeNotes(shape) {
  const fr = shape.frets;
  const fretted = [];
  fr.forEach((f, sl) => { if (!isMuted(f) && f > 0) fretted.push({ sl, f, finger: 0 }); });
  let barreSet = [];
  if (shape.barre) barreSet = fretted.filter(n => n.f === shape.barre.fret && n.sl >= shape.barre.from && n.sl <= shape.barre.to);
  const isBarre = barreSet.length >= 3;
  const rest = fretted.filter(n => !(isBarre && barreSet.includes(n)));
  if (isBarre) barreSet.forEach(n => { n.finger = 1; });
  const given = rest.map(n => (shape.fingers || [])[n.sl]);
  const clean = rest.length && given.every(g => g >= 1 && g <= 4) && new Set(given).size === given.length && !(isBarre && given.includes(1));
  if (clean) rest.forEach((n, i) => { n.finger = given[i]; });
  else {
    let fin = isBarre ? 2 : 1;
    [...rest].sort((a, b) => a.f - b.f || a.sl - b.sl).forEach(n => { n.finger = Math.min(4, fin++); });
  }
  const notes = [];
  const muted = [];
  fr.forEach((f, sl) => {
    const s = 5 - sl;
    if (isMuted(f)) { muted.push(s); return; }
    const n = fretted.find(x => x.sl === sl);
    notes.push({ s, f, finger: n ? n.finger : 0 });
  });
  notes.sort((a, b) => a.s - b.s);
  let barre = null;
  if (isBarre) {
    const strings = barreSet.map(n => 5 - n.sl).sort((a, b) => a - b);
    barre = { fret: shape.barre.fret, strings, tip: strings[0] };
  }
  return { notes, barre, muted };
}

const median = a => { const b = [...a].sort((x, y) => x - y); return b[Math.floor((b.length - 1) / 2)]; };
const sameShape = (a, b) => a.frets.every((f, i) => String(f) === String(b.frets[i]));
const findPattern = (pats, id) => (pats ? [...(pats.strum || []), ...(pats.pick || [])].find(p => p.id === id) : null) || null;

/**
 * Pattern to suggest for a tempo / meter / style (no detected strum): slow songs get ballad patterns, never the driving
 * all-down or rock patterns; 3/4 -> waltz, 6/8 (beatsPerBar 2, dotted-quarter beat) -> 6/8 strum; style 'pick'
 * (fingerstyle songs) -> a fingerpicking pattern of the same meter.
 */
export function suggestPattern(pats, { bpm = 90, beatsPerBar = 4, style = 'strum' } = {}) {
  if (!pats) return null;
  if (style === 'pick') {
    const want = beatsPerBar === 2 ? ['six8_arp'] : beatsPerBar === 3 ? ['waltz_pick'] : bpm < 100 ? ['arp_up', 'pima'] : ['pinch', 'travis'];
    const p = want.map(id => findPattern(pats, id)).find(Boolean) || pats.pick.find(x => x.beats_per_bar === beatsPerBar);
    if (p) return p;
  }
  const SPECIAL = new Set(['reggae', 'funk16', 'funk16b', 'shuffle', 'bossa', 'rock8']);   // style-specific: only when chosen
  let best = null, bs = Infinity;
  for (const p of pats.strum) {
    if (p.beats_per_bar !== beatsPerBar) continue;
    const [lo, hi] = p.bpm || [60, 140], out = Math.max(0, lo - bpm, bpm - hi) / 20;
    const feel = p.feel || '';
    let s = out + 0.1 * Math.abs((p.difficulty || 2) - 2) + (SPECIAL.has(p.id) ? 1 : 0);
    if (bpm < 90 && feel === 'driving') s += 5;
    if (bpm < 80 && feel === 'ballad') s -= 0.3;
    if (s < bs - 1e-9) { bs = s; best = p; }
  }
  return best || findPattern(pats, 'basic');
}

/**
 * @param {{symbol:string, shape:object, bars?:number, beats?:number, pattern?:object|string}[]} prog
 * @param {object} pattern entry of strum_patterns.json (default for chords without their own)
 * @param {number} bpm
 * @param {{patterns?:object, palmMute?:boolean, strumHand?:'pick'|'fingers', humanize?:number, seed?:number, startT?:number,
 *          warp?:(t:number)=>number, unwarp?:(t:number)=>number, strokes?:object[], release?:boolean}} opts
 *   startT: nominal time of the first chord (default LEAD_IN). warp/unwarp: nominal <-> real clock (default identity).
 *   strokes: detected strokes [{t, dir, accent?, muted?, vel?, step?, bar?}] on the real clock; replace the pattern's strokes.
 *   release (default true): strokes within the fretting lead before a change of shape sound the open strings.
 */
export function buildChordSong(prog, pattern, bpm, opts = {}) {
  const { strumHand = 'pick', release = true } = opts;
  const startT = opts.startT ?? LEAD_IN, beat = 60 / bpm, barDur = pattern.beats_per_bar * beat;
  const warp = opts.warp || (t => t), unwarp = opts.unwarp || (t => t);
  const items = prog.map(c => {
    const p = typeof c.pattern === 'string' ? findPattern(opts.patterns, c.pattern) : c.pattern || null;
    return { ...c, pattern: p, beats: chordBeats({ ...c, pattern: p }, pattern) };
  });
  const humanize = opts.humanize ?? 0.15;
  let strokes = progressionTimeline(items.map(c => ({ shape: c.shape.frets, beats: c.beats, pattern: c.pattern || undefined })), pattern, bpm,
    { startT, patterns: opts.patterns, palmMute: !!opts.palmMute, humanize, seed: opts.seed ?? 1 });
  if (opts.warp) strokes.forEach(s => { s.t = warp(s.t); });
  const chords = [];
  let b = 0;
  const events = items.map((c, k) => {
    const { notes, barre, muted } = shapeNotes(c.shape);
    const p = c.pattern || pattern;
    const t0 = warp(startT + b * beat), t1 = warp(startT + (b + c.beats) * beat), dur = t1 - t0;
    const fingered = notes.filter(n => n.finger);
    let pos = fingered.length ? Math.max(1, median(fingered.map(n => n.f - n.finger + 1))) : (k ? chords[k - 1].pos : 1);
    if (barre) pos = barre.fret;
    const ev = {
      t: t0, dur: Math.max(0.2, dur - 0.2), q: b, sec: '', notes, pos,
      slide: false, bend: 0, hb: 0, rel: false, vib: false, ham: false, pull: false, pm: false, gap: dur,
      chord: k, symbol: c.symbol, barre,
    };
    chords.push({ symbol: c.symbol, shape: c.shape, t0, t1, bars: c.beats / p.beats_per_bar, beats: c.beats, beat0: b, pattern: p, pos, barre, muted, notes });
    b += c.beats;
    return ev;
  });
  // detected strokes: exact times from the recording; each takes the shape of the chord it falls in (30 ms early = next chord)
  if (opts.strokes && opts.strokes.length && chords.length) {
    const end = chords[chords.length - 1].t1;
    let k = 0;
    strokes = opts.strokes.filter(s => s.t >= chords[0].t0 - 0.03 && s.t < end).map(s => {
      while (k + 1 < chords.length && s.t >= chords[k + 1].t0 - 0.03) k++;
      const c = chords[k], p = c.pattern.type === 'pick' ? pattern : c.pattern, dir = s.dir === 'U' ? 'U' : 'D';
      const st = strumStroke(c.shape.frets, { dir, letter: dir, accent: !!s.accent, light: false, dead: !!s.muted, palm: false }, s.t,
        { bpm, stepDur: p.step_beats * beat, pattern: p, patterns: opts.patterns, humanize: 0.1, jitter: false,
          velocity: Number.isFinite(s.vel) ? Math.min(1, 0.35 + 0.6 * s.vel) + (s.accent ? 0.1 : 0) : undefined, step: s.step ?? 0, bar: s.bar ?? 0 });
      st.chord = k;
      return st;
    });
  }
  strokes = strokes.filter(s => s.t >= 0);   // a song whose first bar line falls before the recording starts
  // fretting lead (tempo dependent, at most 60 % of the previous chord) and the open-string release stroke before a change
  const lead = fretLead(bpm);
  const leads = chords.map((c, k) => (k ? Math.min(lead, 0.6 * (c.t0 - chords[k - 1].t0)) : lead));
  if (release) for (const st of strokes) {
    const nx = chords[st.chord + 1];
    if (!nx || st.dir === 'P' || st.muted === 'dead' || sameShape(chords[st.chord].shape, nx.shape)) continue;
    if (st.t >= nx.t0 - leads[st.chord + 1]) { st.frets = st.frets.map(() => 0); st.release = true; }
  }
  events.bpm = bpm;
  prepareTimeline(events, 'pick');
  events.forEach((ev, k) => {
    ev.lead = leads[k];
    ev.leadH = Math.max(ev.lead, ev.leadH);   // prepareTimeline: 0.2 s (0.3 s for shifts over 2 frets), capped by the gap
    if (ev.barre) {                         // the index tip sits on the barre's treble-most string
      ev.byFinger[0] = ev.notes.find(n => n.s === ev.barre.tip && n.finger === 1);
      ev.notes.forEach(n => { if (n.finger === 1 && n !== ev.byFinger[0]) n.barred = true; });
    }
  });
  events.mode = strumHand === 'fingers' ? 'fingers' : 'pick';
  const path = buildStrumPath(strokes);
  events.strum = path;
  events.strokes = strokes;
  const t0 = chords.length ? chords[0].t0 : warp(startT), total = warp(startT + b * beat);
  /** beats since the first chord on the nominal (pattern) clock at real time T */
  const posAt = T => (unwarp(T) - startT) / beat;
  return { events, strokes, path, chords, pattern, bpm, barDur, startT: t0, nomStart: startT, total, stepDur: pattern.step_beats * beat, lead, posAt };
}
