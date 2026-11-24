// Fingering solver: one-finger-per-fret positions chosen by Viterbi over hand positions.
// Input: score {bpm, notes:[{t,d,n:[[s,f]],...}]}.  Output: events with hand position + finger per note.
// finger: 0 = open string (no finger), 1 = index, 2 = middle, 3 = ring, 4 = pinky.

export const P_MIN = 1, P_MAX = 20;  // pos 20 lets the ring finger take fret 22 (bends)
const REFINGER_T = 1.0, REFINGER_COST = 3, SLIDE_BREAK = 6, PINKY_FRET = 14, PINKY_COST = 1;

export function normalizeScore(score) {
  const bpm = Number.isFinite(score.bpm) && score.bpm > 0 ? score.bpm : 120;
  const sixteenth = 60 / bpm / 4;
  const notes = [...score.notes].map(n => ({ ...n })).sort((a, b) => a.t - b.t || (a.q ?? 0) - (b.q ?? 0));
  // data repair: two onsets on the same string at (almost) the same time are physically impossible.
  // Re-time the earlier one (by beat position q) between its neighbours.
  for (let i = 1; i < notes.length; i++) {
    const a = notes[i - 1], b = notes[i];
    if (Math.abs(b.t - a.t) < 0.02 && a.n.some(([s]) => b.n.some(([s2]) => s2 === s)) && a.q != null && b.q != null && b.q > a.q) {
      const p = notes.slice(0, i - 1).reverse().find(n => n.t < a.t - 0.02 && n.q != null);
      if (p && b.q > p.q) a.t = p.t + (a.q - p.q) / (b.q - p.q) * (b.t - p.t);
      a.retimed = true;
    }
  }
  const out = notes.map((n, i) => {
    const next = notes[i + 1];
    const dur = Math.max(0.08, (Number.isFinite(n.d) && n.d > 0 ? n.d : 2) * sixteenth);
    return {
      // Share the tab strip's tempo fallback with picking and harmony; keep authored beats intact.
      t: n.t, dur, q: Number.isFinite(n.q) ? n.q : (n.t - notes[0].t) * bpm / 60, sec: n.sec || '',
      notes: n.n.map(([s, f]) => ({ s, f })),
      slide: !!n.slide, bend: n.bend || 0, hb: n.hb || 0, rel: !!n.rel, vib: !!n.vib,
      ham: !!n.ham, pull: !!n.pull, pm: !!n.pm,
      gap: next ? next.t - n.t : 1,
    };
  });
  out.bpm = bpm;
  return out;
}

function emission(ev, p) {
  const fr = ev.notes.filter(n => n.f > 0).map(n => n.f);
  if (!fr.length) return 0;
  let cost = 0;
  for (const f of fr) {
    if (f >= p && f <= p + 3) continue;
    const k = f < p ? p - f : f - (p + 3);
    if (k > 1) return Infinity;           // allow a 1-fret stretch only
    cost += 2 + 3 * k;
  }
  // mild preference for index on the lowest fretted note of the event (natural position playing)
  const lo = Math.min(...fr);
  cost += 0.15 * Math.abs(lo - p);
  // the pinky's reach is short where the frets are wide: a pinky note below fret PINKY_FRET costs a little, so a
  // cheap shift that hands the note to the ring finger wins (Rabba B10 at 3.91 s, e9 at 27.32 s)
  if (fr.some(f => f - p === 3 && f < PINKY_FRET)) cost += PINKY_COST;
  // bends are played with the ring finger (backed by middle + index) where possible
  if ((ev.bend || ev.hb) && fr.length === 1 && fr[0] - p + 1 !== 3) cost += 2.5;
  return cost;
}

export function solveFingering(events) {
  const N = events.length, S = P_MAX - P_MIN + 1;
  if (!N) return events;
  const cost = new Float64Array(N * S), back = new Int16Array(N * S);
  for (let j = 0; j < S; j++) cost[j] = emission(events[0], P_MIN + j);
  // slide pairs: the same finger should glide, i.e. the hand shifts by exactly the fret difference
  const slideDelta = i => {
    const a = events[i - 1], b = events[i];
    if (!a.slide) return null;
    const na = a.notes.find(n => n.f > 0), nb = na && b.notes.find(n => n.s === na.s && n.f > 0);
    return nb ? nb.f - na.f : null;
  };
  // re-fingering: when the last fretted event before l (only open strings in between) played the same
  // string/fret less than REFINGER_T earlier, a shift in between would put a different finger on the note the
  // hand just played (players keep the finger, e.g. Choo Lo's middle-finger G9->11->13->14->16 slide chain).
  // reFinger[i] marks the transition (i-1 -> i).
  const reFinger = new Uint8Array(N);
  for (let l = 1; l < N; l++) {
    let j = l - 1;
    while (j >= 0 && !events[j].notes.some(m => m.f > 0)) j--;
    if (j < 0 || events[l].t - events[j].t >= REFINGER_T) continue;
    if (events[l].notes.some(n => n.f > 0 && events[j].notes.some(m => m.s === n.s && m.f === n.f))) for (let m = j + 1; m <= l; m++) reFinger[m] = 1;
  }
  for (let i = 1; i < N; i++) {
    const ev = events[i];
    const sd = slideDelta(i);
    const shiftW = 0.6, shiftFix = 1.5;
    for (let j = 0; j < S; j++) {
      const e = emission(ev, P_MIN + j);
      if (e === Infinity) { cost[i * S + j] = Infinity; continue; }
      let best = Infinity, arg = 0;
      for (let k = 0; k < S; k++) {
        const c0 = cost[(i - 1) * S + k];
        if (c0 === Infinity) continue;
        let c = c0 + (k === j ? 0 : shiftFix + shiftW * Math.abs(k - j) + (reFinger[i] ? REFINGER_COST : 0));
        // a slide keeps the finger that starts it: breaking the chain costs more than the per-note index preference
        if (sd !== null) c = c0 + (j - k === sd ? 0 : SLIDE_BREAK + shiftW * Math.abs(k - j));
        else if ((events[i - 1].ham || events[i - 1].pull) && k !== j) c += 4; // legato needs a fixed hand
        if (c < best) { best = c; arg = k; }
      }
      cost[i * S + j] = best + e;
      back[i * S + j] = arg;
    }
  }
  let j = 0, best = Infinity;
  for (let k = 0; k < S; k++) if (cost[(N - 1) * S + k] < best) { best = cost[(N - 1) * S + k]; j = k; }
  const pos = new Array(N);
  for (let i = N - 1; i >= 0; i--) { pos[i] = P_MIN + j; j = back[i * S + j]; }
  // open-string-only events need no hand position: the hand stays where it last fretted (the previous note keeps
  // ringing under its finger) and shifts on the way to the next fretted note, not at the first open string
  for (let i = 1; i < N; i++) if (!events[i].notes.some(n => n.f > 0)) pos[i] = pos[i - 1];

  events.forEach((ev, i) => {
    const p = pos[i];
    ev.pos = p;
    const used = new Set();
    // assign lowest strings first so collisions resolve predictably
    const fretted = ev.notes.filter(n => n.f > 0).sort((a, b) => a.f - b.f || b.s - a.s);
    for (const n of fretted) {
      let fin = Math.min(4, Math.max(1, n.f - p + 1));
      if (used.has(fin)) {
        const alt = [fin + 1, fin - 1, fin + 2, fin - 2].find(c => c >= 1 && c <= 4 && !used.has(c));
        if (alt) fin = alt;
      }
      used.add(fin);
      n.finger = fin;
    }
    ev.notes.forEach(n => { if (n.f <= 0) n.finger = 0; });
  });
  return events;
}
