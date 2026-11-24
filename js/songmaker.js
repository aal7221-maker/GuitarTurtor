// In-browser transcription for create.html: audio -> notes (guitab E24 or Spotify basic-pitch) -> the site's score JSON.
// Everything runs in the visitor's browser. Libraries come from jsDelivr, pinned. The guitab E25 model (21 MB, from
// huggingface.co/danmor/guitab) is committed in models/ as two parts (< 15 MB each) joined here.
//   guitab (D. Morotti, code MIT, weights CC BY-NC-SA 4.0): https://github.com/DanieleMorotti/guitab
//   basic-pitch (Spotify, Apache-2.0): https://github.com/spotify/basic-pitch
const GT = 'https://cdn.jsdelivr.net/gh/DanieleMorotti/guitab@b2537dcd1ad68fa2c1cde35496a353b0a33081d6/docs/';
const ORT = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
const BP = 'https://cdn.jsdelivr.net/npm/@spotify/basic-pitch@1.0.1/';
const GT_MODEL = ['part0', 'part1'].map(p => new URL('../models/guitab_e25.onnx.' + p, import.meta.url).href);
const SR = 22050;
const OPEN = [64, 59, 55, 50, 45, 40];             // string 0 = high e … 5 = low E
const MAX_FRET = 22;
export { reconcileChords } from './chord-agreement.js';

// Any audio/video file the browser can decode -> mono Float32Array at 22.05 kHz (the rate of both models).
export async function decodeAudio(blob) {
  const ctx = new OfflineAudioContext(1, 1, SR);   // decodeAudioData resamples to the context's rate
  const buf = await ctx.decodeAudioData(await blob.arrayBuffer());
  const out = new Float32Array(buf.length);
  for (let c = 0; c < buf.numberOfChannels; c++) {
    const ch = buf.getChannelData(c);
    for (let i = 0; i < out.length; i++) out[i] += ch[i] / buf.numberOfChannels;
  }
  return out;
}

async function download(url, label, progress, done = 0, all = 1) {
  let cache = null;
  try { cache = await caches.open('songmaker-v1'); } catch { /* no Cache Storage */ }
  const hit = cache && await cache.match(url).catch(() => null);
  if (hit) return new Uint8Array(await hit.arrayBuffer());
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Could not download ${label} (HTTP ${res.status})`);
  const total = +res.headers.get('Content-Length') || 0, reader = res.body.getReader(), parts = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value); got += value.length;
    progress((done + (total ? got / total : 0)) / all, `Downloading ${label}…`);
  }
  const bytes = new Uint8Array(got);
  let o = 0; for (const p of parts) { bytes.set(p, o); o += p.length; }
  if (cache) cache.put(url, new Response(bytes)).catch(() => {});
  return bytes;
}

// guitab E25 through its own browser code (CQT + windowed prediction + note extraction), ONNX Runtime Web (WebAssembly).
async function runGuitab(y, progress, onset) {
  progress(0, 'Loading guitab…');
  const [dspMod, notesMod, ort, dsp] = await Promise.all([
    import(GT + 'js/dsp.js'), import(GT + 'js/notes.js'), import(ORT + 'ort.wasm.min.mjs'),
    fetch(GT + 'assets/dsp.json').then(r => r.json())]);
  ort.env.wasm.wasmPaths = ORT; ort.env.logLevel = 'error';
  const parts = [];
  for (const [i, u] of GT_MODEL.entries()) parts.push(await download(u, 'the guitab model (21 MB, first time only)', progress, i, GT_MODEL.length));
  const bytes = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  parts.reduce((o, p) => (bytes.set(p, o), o + p.length), 0);
  const sess = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] });
  progress(0, 'Computing the spectrogram…');
  await new Promise(r => setTimeout(r, 30));
  const feat = dspMod.cqtFeature(y, dsp);
  const probs = await notesMod.predict(feat.data, feat.frames, dsp.n_bins, async x => {
    const out = await sess.run({ x: new ort.Tensor('float32', x, [1, 432, dsp.n_bins]) });
    return { tab: await out.tab.getData(), onset: await out.onset.getData() };
  }, f => progress(f, 'Transcribing with guitab…'));
  // loudness for the skyline = the onset head's peak at the note's start (how freshly plucked it is)
  const F = 22050 / 256, on = probs.onset;
  const peak = n => { const f = Math.round(n.onset * F), col = 6 - n.string; let v = 0; for (let k = Math.max(0, f - 2); k <= Math.min(probs.frames - 1, f + 2); k++) v = Math.max(v, on[k * 6 + col]); return v; };
  return notesMod.extractNotes(probs, { onset }).map(n => ({ t: n.onset, e: n.offset, p: n.pitch, a: peak(n) }));
}

// basic-pitch (TF.js) with the "clean" settings of pitch-model/transcribe.py.
async function runBasicPitch(y, progress) {
  progress(0, 'Loading basic-pitch…');
  const bp = await import(BP + '+esm');
  const model = new bp.BasicPitch(BP + 'model/model.json');
  const frames = [], onsets = [];
  await model.evaluateModel(y, (f, o) => { frames.push(...f); onsets.push(...o); },
    p => progress(p, 'Transcribing with basic-pitch…'));
  const ev = bp.noteFramesToTime(bp.outputToNotesPoly(frames, onsets, 0.6, 0.4, 9, true, null, null, true));
  return ev.map(n => ({ t: n.startTimeSeconds, e: n.startTimeSeconds + n.durationSeconds, p: n.pitchMidi, a: n.amplitude }));
}

// engine: 'guitab' | 'basic-pitch'. Returns { notes, engine } (falls back to basic-pitch if guitab fails to load or run).
export async function transcribe(y, { engine = 'guitab', onset = 0.3, progress = () => {} } = {}) {
  if (engine === 'guitab') {
    try { return { notes: await runGuitab(y, progress, onset), engine: 'guitab E25' }; }
    catch (e) { console.warn('guitab failed, using basic-pitch:', e); progress(0, 'guitab failed (' + e.message + '), using basic-pitch…'); }
  }
  return { notes: await runBasicPitch(y, progress), engine: 'basic-pitch' };
}

/* ---------- notes -> score ---------- */
export async function loadChordShapes() {
  try {
    const r = await fetch(new URL('../chords/chord_shapes.json', import.meta.url));
    if (!r.ok) return null;
    return await r.json();
  } catch { return null; } // Exact-pitch string search still works offline without the library.
}

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

// A short residual activation can become a note even without a fresh BP onset
// (its remaining-energy pass). Amplitudes are model salience, NOT acoustic dB.
// Only suppress isolated, weak, late fragments while >=3 chord pitch classes
// still ring through the fragment. No global minimum length or melody gate.
function isChordResidual(n, held, previous) {
  if (!held.length || !(n.e > n.t && n.e - n.t <= .18 + 1e-9) ||
      !(n.a >= 0 && n.a < .4) || n.t - held.at(-1).t < .12) return false;
  const ringing = held.filter(s => s.e >= n.e && Number.isFinite(s.a));
  if (new Set(ringing.map(s => s.p % 12)).size < 3) return false;
  const strengths = ringing.map(s => s.a).sort((a, b) => a - b), mid = strengths.length >> 1;
  const median = strengths.length % 2 ? strengths[mid] : (strengths[mid - 1] + strengths[mid]) / 2;
  if (n.a >= median * .5) return false;
  // Preserve plausible hammer-ons/pull-offs: a nearby pitch ends at this onset.
  // Raw notes lack string/onset-head evidence, so err on the side of keeping it.
  if ([...held, ...previous].some(s => Math.abs(s.e - n.t) <= .06 &&
      Math.abs(s.p - n.p) >= 1 && Math.abs(s.p - n.p) <= 3)) return false;
  // Same pitch class, or the 3rd/5th partial ABOVE a still-sounding note.
  // 2nd/4th partials are already covered by pitch class. Do not fold arbitrary
  // fifths/thirds or subharmonics into this test.
  return ringing.some(s => s.p % 12 === n.p % 12 || [19, 28].includes(n.p - s.p));
}

// BP can split a sustain at exactly its offset, or detect a chord tone late.
// These need not be quiet. Require a still-ringing chord, fewer than three
// re-onset classes, and continuity for EVERY member of the group. A fresh
// three-class strum is recognized first. Overlapping same-pitch activations
// are ambiguous re-plucks; preserve those, as well as nearby pitch transitions.
function isSustainArtifact(group, held, previous, future) {
  if (!held.length || new Set(group.map(n => n.p % 12)).size >= 3 ||
      group[0].t - held.at(-1).t < .12) return false;
  const horizon = Math.max(...held.map(n => n.e));
  const pcs = new Set(held.map(n => n.p % 12));
  return group.every(n => n.e > n.t && n.t < horizon &&
    (n.e <= horizon + 1e-6 || held.some(s => s.p === n.p && Math.abs(s.e - n.t) <= .015)) &&
    pcs.has(n.p % 12) && !held.some(s => s.p === n.p && s.e > n.t + .015) &&
    !future.some(s => Math.abs(s.t - n.e) <= .06 && Math.abs(s.p - n.p) >= 1 && Math.abs(s.p - n.p) <= 3) &&
    ![...held, ...previous].some(s => Math.abs(s.e - n.t) <= .06 &&
      Math.abs(s.p - n.p) >= 1 && Math.abs(s.p - n.p) <= 3));
}

// Preserve the 50 ms melody skyline, but recognize the complete strum BEFORE
// applying its relative loudness filter. Model amplitudes vary between strings;
// that melody filter can otherwise discard the bass/third of a clear strum.
// Require distinct, sustained pitches and an exact playable assignment.
function skyline(raw, high = 0.75, lib = null) {
  const ev = raw.filter(n => n.p >= OPEN[5] && n.p <= OPEN[0] + MAX_FRET).sort((a, b) => a.t - b.t);
  const out = [];
  let held = [], previous = [];
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
    if (chord) held = strum ? g : sounding; // A new chord replaces old ringing evidence.
    else if (isSustainArtifact(g, held, previous, ev.slice(j)) || (g.length === 1 && isChordResidual(top, held, previous))) {
      // Retain continuation evidence for another split, without inventing a pluck.
      held = held.map(s => ({ ...s, e: Math.max(s.e, ...g.filter(n => n.p === s.p).map(n => n.e)) }));
      i = j; continue;
    }
    out.push({ t: ev[i].t, p: top.p, a: top.a,
      ...(chord ? { chord, sustain: Math.max(...(strum ? g : sounding).map(n => n.e)) - ev[i].t } : {}) });
    previous = g;
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
