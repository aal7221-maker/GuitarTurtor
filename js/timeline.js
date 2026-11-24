// Timeline sampler: given solved events and time T, decide what every finger of both hands is doing.
// This is a pure function of T; the springs in performer.js add the physical smoothing.
//
// Technique semantics follow the tab-viewer (tab-viewer/app.js):
//   slide on note i  -> finger glides from note i to note i+1 just before note i+1 (slideWindow); note i+1 is legato
//   ham / pull on i  -> note i+1 is hammered on / pulled off (legato, not picked)
//   bend: f -> bend fret, ramp in [t+min(.05,.2d), t+min(.18,.6d)];  hb: held at hb fret for the whole note
//   rel: bend released during [t+.45d, t+.6d];  vib: 6 Hz lateral wobble
import { contactX, stringY, stringZ, FRET_H } from './guitar.js';

export const FINGER_KEYS = ['index', 'middle', 'ring', 'pinky'];
export const PRESS_Y = FRET_H + 0.0022;   // fingertip joint height when the string is pushed onto the fret
const HOVER = 0.007, HOVER_COUPLED = 0.0035, REST = 0.018;
const BEND_PER_SEMI = 0.0026;             // lateral string push per semitone (m)
const HOVER_DEFAULT = [3, 2, 2, 1];
// note release: the outgoing finger stays down until just before the next onset (players lift at the next
// pluck); RELEASE_SAME when the next note is on the same string at a lower fret / open (must be clear by then)
const RELEASE = 0.012, RELEASE_SAME = 0.035;
// slide glide: the finger travels over [t1 - min(SLIDE_MAX + SLIDE_PER_FRET*(frets-2), SLIDE_K*gap), t1] with
// t1 = next.t - SLIDE_END, so it enters the new fret just before the next onset and the stiff (strike) spring has
// settled on it at the onset (it used to arrive 39-137 ms early)
const SLIDE_MAX = 0.06, SLIDE_K = 0.6, SLIDE_END = 0.05, SLIDE_PER_FRET = 0.03;
// hand lead: shifts of more than 2 frets start earlier (matches the Choo Lo player's timing)
const LEAD_H = 0.2, LEAD_H_SHIFT = 0.3;
const EMPTY_PLUCKS = { thumb: [], index: [], middle: [], ring: [], pick: [] };

const clamp01 = x => Math.max(0, Math.min(1, x));
const smooth = x => { x = clamp01(x); return x * x * (3 - 2 * x); };

export function prepareTimeline(events, mode = 'fingers') {
  events.forEach((ev, i) => {
    ev.i = i;                      // own index, so sample() never has to search for it
    const prev = events[i - 1];
    const gapPrev = prev ? ev.t - prev.t : 1;
    ev.legato = prev ? (prev.slide ? 'slide' : prev.ham ? 'ham' : prev.pull ? 'pull' : null) : null;
    ev.lead = ev.legato === 'slide' ? 0.02 : ev.legato === 'ham' ? 0.035 : Math.min(0.11, 0.6 * gapPrev);
    // the hand may start its shift during open-string notes (they need no fretting position); keep the lead
    // times ordered so the binary search in sample() stays valid
    const shift = prev ? Math.abs(ev.pos - prev.pos) : 0;
    let jf = i - 1;
    while (jf > 0 && !events[jf].notes.some(n => n.f > 0)) jf--;
    const gapH = jf >= 0 ? ev.t - events[jf].t : 1;
    ev.leadH = Math.min(shift > 2 ? LEAD_H_SHIFT : LEAD_H, 0.8 * gapH, prev ? gapPrev + prev.leadH - 1e-3 : 1);
    ev.byFinger = [null, null, null, null];
    ev.notes.forEach(n => { if (n.finger) ev.byFinger[n.finger - 1] = n; });
    ev.fretted = ev.notes.filter(n => n.f > 0);
  });
  events.forEach((ev, i) => {
    ev.prevUse = FINGER_KEYS.map((_, k) => { for (let j = i - 1; j >= 0; j--) if (events[j].byFinger[k]) return j; return -1; });
    ev.nextUse = FINGER_KEYS.map((_, k) => { for (let j = i + 1; j < events.length; j++) if (events[j].byFinger[k]) return j; return -1; });
  });
  // picking-hand plucks (legato notes are not picked; strummed events of a mixed song belong to the strum path)
  const plucks = { thumb: [], index: [], middle: [], ring: [], pick: [] };
  events.forEach((ev, i) => {
    const gapPrev = i ? ev.t - events[i - 1].t : 1;
    if (ev.legato || ev.strummed) return;
    if (mode === 'pick') {
      // alternate picking: down on the beat grid, up in between. Use the 16th grid when the line is dense,
      // otherwise the 8th grid (down on beats, up on "and"s).
      const q = ev.q ?? 0, sixteenth = ev.dur / Math.max(1, ev.d16 || 1);
      const dense = (ev.gap < 0.75 * (60 / (events.bpm || 120)) / 2) || (gapPrev < 0.75 * (60 / (events.bpm || 120)) / 2);
      const k = dense ? Math.round(q * 4) : Math.round(q * 2);
      void sixteenth;
      plucks.pick.push({ t: ev.t, strings: ev.notes.map(n => n.s), dir: k % 2 === 0 ? 1 : -1, ev });
      return;
    }
    const used = new Set();
    [...ev.notes].sort((a, b) => b.s - a.s).forEach(n => {
      const order = n.s >= 3 ? ['thumb', 'index', 'middle'] : n.s === 2 ? ['index', 'middle', 'thumb'] : n.s === 1 ? ['middle', 'ring', 'index'] : ['ring', 'middle', 'index'];
      const rf = order.find(o => !used.has(o)) || order[0];
      used.add(rf);
      n.pluck = rf;
      plucks[rf].push({ t: ev.t, s: n.s, ev });
    });
  });
  for (const list of Object.values(plucks)) list.forEach((p, i) => { p.gap = i ? p.t - list[i - 1].t : 1; });
  events.plucks = plucks;
  events.mode = mode;
  return events;
}

function lastLE(arr, T, key) {
  let lo = 0, hi = arr.length - 1, ans = -1;
  while (lo <= hi) { const m = (lo + hi) >> 1; if (key(arr[m]) <= T) { ans = m; lo = m + 1; } else hi = m - 1; }
  return ans;
}

/** lateral bend offset (m) of note n in event e at time T */
export function bendOffset(e, n, T) {
  if (!e || !n || n.f <= 0) return 0;
  const age = T - e.t, d = e.dur;
  let semis = 0;
  if (e.hb) semis = e.hb - n.f;
  else if (e.bend) semis = (e.bend - n.f) * smooth((age - Math.min(0.05, 0.2 * d)) / Math.max(0.03, Math.min(0.18, 0.6 * d) - Math.min(0.05, 0.2 * d)));
  if (e.rel) semis *= 1 - smooth((age - 0.45 * d) / (0.15 * d));
  if (semis <= 0) return 0;
  return Math.sign(2.5 - n.s) * BEND_PER_SEMI * semis;
}

/** slide glide window [t0, t1] of event e toward the next event (null if e is not a slide). Longer slides glide
 *  for longer (SLIDE_PER_FRET) so the stiff spring can follow and the finger is on the new fret at the onset. */
function slideWindow(events, e) {
  const nx = events[e.i + 1];
  if (!e.slide || !nx) return null;
  const a = e.fretted[0], b = a && nx.notes.find(m => m.s === a.s && m.f > 0);
  const df = b ? Math.abs(b.f - a.f) : 2;
  const gap = nx.t - e.t, t1 = nx.t - SLIDE_END;
  const w = Math.min(SLIDE_MAX + SLIDE_PER_FRET * Math.max(0, df - 2), SLIDE_K * gap);
  return [Math.min(t1 - 0.01, t1 - w), t1];
}

/**
 * @returns {{pos, focus, targets:{index..pinky:{x,y,z,press,strike,note,ev}}, right:{...}, sounding, upcoming, now}}
 */
export function sample(events, T) {
  const out = { pos: events.length ? events[0].pos : 5, focus: null, targets: {}, right: {}, sounding: [], upcoming: [], now: -1 };
  if (!events.length) {
    // A cleared score still needs valid rest targets so the previous pose can release.
    FINGER_KEYS.forEach((key, k) => {
      const x = contactX(out.pos + k);
      out.targets[key] = { x, y: stringY(x) + REST, z: stringZ(HOVER_DEFAULT[k], x), press: false, note: null };
    });
    out.right = sampleRight(events, T);
    return out;
  }
  const i = lastLE(events, T, e => e.t - e.lead);
  const iH = lastLE(events, T, e => e.t - e.leadH);
  const iNow = lastLE(events, T, e => e.t);
  out.now = iNow;
  const ev = i >= 0 ? events[i] : null;
  const next = events[i + 1];
  const evH = events[Math.max(0, iH)];
  out.pos = evH.pos;
  out.evH = evH;
  // during a slide the hand travels with the finger
  const slideStart = e => slideWindow(events, e)?.[0] ?? e.t;
  if (ev && ev.slide && next && T > slideStart(ev)) out.pos = next.pos;
  if (evH.fretted.length) out.focus = evH.fretted.reduce((a, n) => a + n.s, 0) / evH.fretted.length;
  const pos = ev ? (ev.slide && next && T > slideStart(ev) ? next.pos : ev.pos) : events[0].pos;
  const resting = !ev || (T > ev.t + ev.dur + 0.6 && (!next || next.t - next.lead - T > 0.35));

  const press = (k, n, e, extra = {}) => { // eslint-disable-line prefer-const
    let x = contactX(n.f), f = n.f;
    // slide: glide along the string toward the next note without lifting
    if (e.slide) {
      const nx = events[e.i + 1];
      const nn = nx && nx.notes.find(m => m.s === n.s);
      if (nn && nn.f > 0) {
        const [t0, t1] = slideWindow(events, e);
        const u = smooth((T - t0) / (t1 - t0));
        extra = { ...extra, strike: true };   // finger stays on the string: stiff tracking
        x = contactX(n.f) + (contactX(nn.f) - contactX(n.f)) * u;
        f = u > 0.5 ? nn.f : n.f;
      }
    }
    let z = stringZ(n.s, x) + bendOffset(e, n, T);
    if (e.vib && T >= e.t && T < e.t + e.dur + 0.15) z += 0.0008 * Math.sin(2 * Math.PI * 6 * (T - e.t));
    out.targets[FINGER_KEYS[k]] = { x, y: PRESS_Y, z, press: true, note: n, fret: f, ev: e, ...extra };
  };

  FINGER_KEYS.forEach((key, k) => {
    if (ev && !resting) {
      let n = ev.byFinger[k];
      if (n) return press(k, n, ev);
      // pull-off preparation: the destination finger is already planted
      if (ev.pull && next && next.byFinger[k] && T >= ev.t) return press(k, next.byFinger[k], next);
      // outgoing note: the previous event's finger keeps its note until just before this onset; only a finger
      // this event needs (handled above) travels early with the approach lead
      if (T < ev.t && ev.i > 0) {
        const pe = events[ev.i - 1], pn = pe.byFinger[k];
        if (pn && !pe.slide) {
          const same = ev.notes.some(m => m.s === pn.s && m.f < pn.f);
          if (T < ev.t - (same ? RELEASE_SAME : RELEASE) && pn.f >= pos - 1 && pn.f <= pos + 4) return press(k, pn, pe);
        }
      }
      // legato hold of a previous note still ringing
      const j = ev.prevUse[k];
      if (j >= 0) {
        const pj = events[j], nj = pj.byFinger[k];
        if (T < pj.t + pj.dur + 0.04 && nj.f >= pos - 1 && nj.f <= pos + 4 && !ev.notes.some(m => m.s === nj.s)) return press(k, nj, pj);
      }
    }
  });

  // bend support: lower free fingers back up the bending finger on the same string
  if (ev && !resting) for (const n of ev.fretted) if ((ev.bend || ev.hb) && n.finger > 1) {
    for (let k = 0; k < n.finger - 1; k++) {
      if (out.targets[FINGER_KEYS[k]]) continue;
      const f = n.f - (n.finger - 1 - k);
      if (f < 1) continue;
      const x = contactX(f);
      out.targets[FINGER_KEYS[k]] = { x, y: PRESS_Y + 0.0006, z: stringZ(n.s, x) + bendOffset(ev, n, T), press: false, support: true, note: null };
    }
  }

  // free fingers hover over the next note they will play (anticipation), else over their last string
  FINGER_KEYS.forEach((key, k) => {
    if (out.targets[key]) return;
    let s = HOVER_DEFAULT[k], f = Math.min(22, Math.max(1, pos + k)), strike = false;
    const base = Math.max(i, 0);
    const jn = ev ? ev.nextUse[k] : (events[0].byFinger[k] ? 0 : -1);
    if (jn >= 0 && events[jn].t - T < 0.6) {
      const nn = events[jn].byFinger[k];
      s = nn.s;
      if (Math.abs(nn.f - (out.pos + k)) <= 1) f = nn.f; else f = Math.min(22, Math.max(1, out.pos + k));
      strike = events[jn].legato === 'ham';
    } else if (ev && ev.prevUse[k] >= 0) s = events[ev.prevUse[k]].byFinger[k].s;
    else if (ev && ev.byFinger[k]) s = ev.byFinger[k].s;
    const x = contactX(f);
    let lift = resting ? REST : strike ? 0.010 : HOVER;
    if (!resting && !strike) {
      const nb = k === 2 ? out.targets.pinky : k === 3 ? out.targets.ring : k === 1 ? out.targets.ring : null;
      if (nb && nb.press) lift = HOVER_COUPLED;   // ring/pinky (and middle/ring) tendon coupling
    }
    out.targets[key] = { x, y: stringY(x) + lift, z: stringZ(s, x), press: false, note: null, strike };
    void base;
  });
  // hammered note: strike flag tells the performer to use a fast spring
  if (ev && ev.legato === 'ham') for (const n of ev.fretted) {
    const tg = out.targets[FINGER_KEYS[n.finger - 1]];
    if (tg && tg.note === n) tg.strike = true;
  }

  out.right = sampleRight(events, T);

  for (let j = Math.max(0, iNow - 8); j < events.length; j++) {
    const e = events[j];
    if (e.t > T + 0.6) break;
    if (e.t <= T && T < e.t + Math.max(e.dur, 0.18)) out.sounding.push(e);
    else if (e.t > T) out.upcoming.push(e);
  }
  return out;
}

// ---------------- picking hand ----------------
export const PLUCK_X = 0.545;                 // over the soundhole / between pickups
const RIGHT_HOME = { thumb: 4, index: 2, middle: 1, ring: 0 };
const RIGHT_DX = { thumb: -0.03, index: -0.008, middle: 0.002, ring: 0.012 };

function sampleRight(events, T) {
  const P = events.plucks || EMPTY_PLUCKS, res = {};
  if (events.mode === 'pick') {
    const list = P.pick;
    const CROSS = 0.008;                                 // pick starts crossing slightly before the onset
    const j = lastLE(list, T + 0.12, p => p.t);          // next stroke within prep window
    const jp = lastLE(list, T + CROSS, p => p.t);        // stroke in progress / already played
    let tgt;
    const x = PLUCK_X + 0.015;
    const strokeZ = (p, side) => {
      const zs = p.strings.map(s => stringZ(s, x));
      const zHi = Math.max(...zs), zLo = Math.min(...zs);
      // dir +1 = downstroke (from the low-E side, +z, toward the floor, -z)
      return side < 0 ? (p.dir > 0 ? zHi + 0.006 : zLo - 0.006) : (p.dir > 0 ? zLo - 0.006 : zHi + 0.006);
    };
    if (j >= 0 && j !== jp && list[j].t - T < Math.min(0.12, 0.6 * list[j].gap)) {
      const p = list[j];
      tgt = { x, y: stringY(x) - 0.0004, z: strokeZ(p, -1), stroke: p };
    } else if (jp >= 0 && T - list[jp].t < 0.25) {
      const p = list[jp];
      tgt = { x, y: stringY(x) + (T - p.t < 0.05 ? -0.0004 : 0.004), z: strokeZ(p, 1), stroke: p };
    } else {
      tgt = { x, y: stringY(x) + 0.008, z: stringZ(2.5, x), stroke: null };
    }
    res.pick = tgt;
    return res;
  }
  for (const rf of ['thumb', 'index', 'middle', 'ring']) {
    const list = P[rf];
    const x = PLUCK_X + RIGHT_DX[rf];
    const jp = lastLE(list, T, p => p.t);
    const jn = jp + 1 < list.length ? jp + 1 : -1;
    const dirZ = rf === 'thumb' ? -1 : 1;     // fingers pull toward the palm (+z), thumb pushes toward the floor (-z)
    let tgt;
    const prep = jn >= 0 ? Math.min(0.12, 0.6 * list[jn].gap) : 0;
    if (jp >= 0 && T - list[jp].t < 0.08) {
      const p = list[jp];   // follow-through
      tgt = { x, y: stringY(x) + 0.005, z: stringZ(p.s, x) + dirZ * 0.008, s: p.s, phase: 'through' };
    } else if (jn >= 0 && list[jn].t - T < prep) {
      const p = list[jn];   // planted on the string, on the side opposite to the pluck direction
      tgt = { x, y: stringY(x) + 0.0008, z: stringZ(p.s, x) - dirZ * 0.0022, s: p.s, phase: 'plant', pluck: p };
    } else {
      const s = jn >= 0 && list[jn].t - T < 0.8 ? list[jn].s : RIGHT_HOME[rf];
      tgt = { x, y: stringY(x) + 0.006, z: stringZ(s, x) - dirZ * 0.003, s, phase: 'rest' };
    }
    res[rf] = tgt;
  }
  return res;
}
