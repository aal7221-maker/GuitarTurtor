// Strumming-hand path: a pure function of time T built from a stroke list (chords/strum_model.js).
// Strokes use LOW-E-FIRST string numbers; everything here is converted to the 3D code's numbering
// (string 0 = high e, 5 = low E; z grows toward the low E).
//
// Each stroke is a straight sweep through the strings at their own offset times (the contact point crosses
// string k exactly at stroke.t + offsets[k]), with a short lead-in before the first string and a follow-through
// after the last. Between strokes the hand returns like a pendulum: it lifts over the strings when the next
// stroke goes the same way, and just turns round when it goes the other way.
// Muted strokes are lower and nearer the bridge (palm mute: at the bridge; dead chuck: flat on the strings).
import { stringZ, stringY } from './guitar.js';

export const STRUM = { x: 0.56, xPalm: 0.607, xDead: 0.585, depth: 0.0006, lead: 0.022, trail: 0.026, vDefault: 0.8, vMax: 1.6 };
export const STRUM_Z0 = stringZ(2.5, STRUM.x);

const smooth = u => { u = Math.max(0, Math.min(1, u)); return u * u * (3 - 2 * u); };
const xOf = st => st.muted === 'palm' ? STRUM.xPalm : st.muted ? STRUM.xDead : STRUM.x;
const toString3D = sLow => 5 - sLow;

export function buildStrumPath(strokes) {
  const segs = strokes.map(st => {
    const x = xOf(st);
    const pts = st.strings.map((sl, k) => ({ t: st.t + st.offsets[k], z: stringZ(toString3D(sl), x), s: toString3D(sl) }));
    let sg = st.dir === 'D' ? -1 : st.dir === 'U' ? 1 : (st.strings[0] <= 2 ? -1 : 1);
    const first = pts[0], last = pts[pts.length - 1];
    if (pts.length > 1 && Math.abs(last.z - first.z) > 1e-6) sg = Math.sign(last.z - first.z);
    let v = pts.length > 1 && last.t - first.t > 1e-4 ? (last.z - first.z) / (last.t - first.t) : sg * STRUM.vDefault;
    v = sg * Math.min(STRUM.vMax, Math.max(0.4, Math.abs(v)));
    return { st, x, sg, v, pts, t0: first.t - STRUM.lead, t1: last.t + STRUM.trail };
  });
  // overlapping windows (very fast patterns): split the gap between the last and the next first contact
  for (let i = 1; i < segs.length; i++) {
    const a = segs[i - 1], b = segs[i];
    if (b.t0 < a.t1 + 0.004) {
      const m = 0.5 * (a.pts[a.pts.length - 1].t + b.pts[0].t);
      a.t1 = Math.min(a.t1, m - 0.002); b.t0 = Math.max(b.t0, m + 0.002);
    }
  }
  for (const g of segs) {
    const f = g.pts[0], l = g.pts[g.pts.length - 1];
    g.path = [{ t: g.t0, z: f.z - g.v * (f.t - g.t0) }, ...g.pts.map(p => ({ t: p.t, z: p.z })), { t: g.t1, z: l.z + g.v * (g.t1 - l.t) }];
    g.startState = { t: g.t0, z: g.path[0].z, x: g.x, sg: g.sg };
    g.endState = { t: g.t1, z: g.path[g.path.length - 1].z, x: g.x, sg: g.sg };
  }

  const home = t => ({ t, z: STRUM_Z0, x: STRUM.x, sg: 0 });
  const find = T => { let lo = 0, hi = segs.length - 1, a = -1; while (lo <= hi) { const m = (lo + hi) >> 1; if (segs[m].t0 <= T) { a = m; lo = m + 1; } else hi = m - 1; } return a; };

  function at(T) {
    if (!segs.length) return { x: STRUM.x, y: stringY(STRUM.x) + 0.02, z: STRUM_Z0, contact: false, stroke: null, ph: 0, sg: 0 };
    const i = find(T);
    if (i >= 0 && T <= segs[i].t1) {
      const g = segs[i], p = g.path;
      let k = 0; while (k < p.length - 2 && T >= p[k + 1].t) k++;
      const u = p[k + 1].t > p[k].t ? Math.max(0, Math.min(1, (T - p[k].t) / (p[k + 1].t - p[k].t))) : 0;
      return { x: g.x, y: stringY(g.x) - STRUM.depth, z: p[k].z + (p[k + 1].z - p[k].z) * u, contact: true, stroke: g.st, ph: (T - g.t0) / (g.t1 - g.t0), sg: g.sg };
    }
    const a = i >= 0 ? segs[i].endState : home(segs[0].t0 - 0.5);
    const b = i + 1 < segs.length ? segs[i + 1].startState : home(a.t + 0.5);
    const u = (T - a.t) / Math.max(1e-3, b.t - a.t), s = smooth(u);
    const x = a.x + (b.x - a.x) * s;
    const homeLeg = a.sg === 0 || b.sg === 0;
    const h = homeLeg ? 0.012 : a.sg === b.sg ? Math.max(0.004, Math.min(0.016, 0.003 + 0.1 * (b.t - a.t))) : 0.0035;
    const uu = Math.max(0, Math.min(1, u));
    return { x, y: stringY(x) - STRUM.depth + h * Math.sin(Math.PI * uu), z: a.z + (b.z - a.z) * s, contact: false, stroke: null, ph: 0, sg: 0 };
  }

  const times = strokes.map(s => s.t);
  /** string glow / vibration, 3D string order (0 = high e): each string lights when the hand reaches it */
  function glow(T) {
    const out = [0, 0, 0, 0, 0, 0];
    let lo = 0, hi = times.length - 1, a = -1;
    while (lo <= hi) { const m = (lo + hi) >> 1; if (times[m] <= T + 0.1) { a = m; lo = m + 1; } else hi = m - 1; }
    for (let j = a; j >= 0 && times[j] > T - 1.6; j--) {
      const st = strokes[j];
      const tau = st.muted === 'dead' ? 0.06 : st.muted ? 0.14 : 0.55, amp = st.muted === 'dead' ? 0.5 * st.velocity : st.velocity;
      st.strings.forEach((sl, k) => {
        const ts = st.t + st.offsets[k];
        if (ts > T) return;
        const val = Math.min(1, 0.35 + amp) * Math.exp(-(T - ts) / tau);
        const s3 = toString3D(sl);
        if (val > out[s3]) out[s3] = val;
      });
    }
    return out;
  }

  return { segs, at, glow };
}

/**
 * Strum path of a song that mixes strummed bars with single notes (js/songchords.js buildMixedTimeline).
 * parts: [{strokes, w0, w1}]: one path per strummed segment, active only inside its window [w0, w1] (a short approach
 * before the first stroke and a follow-through after the last). Outside every window at() returns null and the
 * performer picks the tab notes. blend < 1 during the approach: the performer eases the hand in from where it was.
 */
export function buildMixedStrumPath(parts) {
  const P = parts.filter(p => p.strokes && p.strokes.length).map(p => {
    const path = buildStrumPath(p.strokes), first = path.segs[0].t0, last = path.segs[path.segs.length - 1].t1;
    const w0 = Math.min(first, Number.isFinite(p.w0) ? p.w0 : first - 0.3), w1 = Math.max(last, Number.isFinite(p.w1) ? p.w1 : last + 0.1);
    return { path, first, w0, w1 };
  });
  const segs = P.flatMap(p => p.path.segs);
  function at(T) {
    for (const p of P) {
      if (T < p.w0 || T > p.w1) continue;
      const r = p.path.at(T);
      r.blend = T < p.first ? Math.max(0, Math.min(1, (T - p.w0) / Math.max(1e-3, p.first - p.w0))) : 1;
      return r;
    }
    return null;
  }
  function glow(T) {
    const out = [0, 0, 0, 0, 0, 0];
    for (const p of P) {
      if (T < p.w0 - 0.1 || T > p.w1 + 2) continue;
      p.path.glow(T).forEach((v, s) => { if (v > out[s]) out[s] = v; });
    }
    return out;
  }
  return { segs, at, glow, windows: P.map(p => [p.w0, p.w1]), mixed: true };
}
