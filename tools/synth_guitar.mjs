// Synthetic strummed guitar for testing chord/strum detection: Karplus-Strong strings (tuning allpass, damped loop),
// strokes from chords/strum_model.js (so timings/directions/velocities are known ground truth), optional drums + noise.
//   import { renderCase, writeWav } from './synth_guitar.mjs';
//   const { y, sr, gt } = renderCase({ pattern:'basic', bpm:90, prog:['G','D','Em','C'], barsPer:1, loops:4, drums:false, noise:0, seed:1 });
// gt = { bpm, bars:[{bar,t0,t1,chord}], strokes:[{t,step,bar,dir,muted,vel}], pattern, steps_per_bar, lead }
import fs from 'node:fs';
import { strumTimeline } from '../chords/strum_model.js';
const J = f => JSON.parse(fs.readFileSync(new URL('../chords/' + f, import.meta.url)));
const shapes = J('chord_shapes.json'), pats = J('strum_patterns.json');
const byId = Object.fromEntries(pats.strum.map(p => [p.id, p]));
const OPEN = [40, 45, 50, 55, 59, 64], SR = 22050;
function rng(seed) { let s = (seed >>> 0) || 1; return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296); }

function pluck(out, t0, freq, amp, loopGain, dur, rand) {
  const L = SR / freq - 0.5, N = Math.floor(L), d = L - N, a = (1 - d) / (1 + d);
  // excitation: plucked-string triangle (pluck point ~1/5 of the string -> ~1/n^2 harmonic spectrum like a real pick) + a little noise
  const r = new Float32Array(N), pp = Math.max(1, Math.round(N * (0.17 + 0.06 * rand())));
  for (let i = 0; i < N; i++) r[i] = (i < pp ? i / pp : (N - i) / (N - pp)) - 0.5 + 0.15 * (rand() * 2 - 1);
  let i = 0, prev = 0, apx = 0, apy = 0; const n = Math.min(out.length - Math.round(t0 * SR), Math.round(dur * SR)), o = Math.round(t0 * SR);
  for (let k = 0; k < n; k++) {
    const cur = r[i], nxt = r[(i + 1) % N], v = loopGain * 0.5 * (cur + nxt);
    const y = a * v + apx - a * apy; apx = v; apy = y; r[i] = y; i = (i + 1) % N;
    out[o + k] += amp * cur;
  }
}

export function renderCase({ pattern = 'basic', bpm = 90, prog = ['G', 'D', 'Em', 'C'], barsPer = 1, loops = 4, drums = false, noise = 0, seed = 1, lead = 0.4 } = {}) {
  const rand = rng(seed), pat = byId[pattern], beat = 60 / bpm, barDur = pat.beats_per_bar * beat;
  const seq = []; for (let l = 0; l < loops; l++) for (const c of prog) seq.push(c);
  const total = lead + seq.length * barsPer * barDur + 2.5, y = new Float32Array(Math.round(total * SR));
  const gt = { bpm, pattern, steps_per_bar: pat.steps_per_bar, beats_per_bar: pat.beats_per_bar, lead, bars: [], strokes: [] };
  let bar = 0;
  for (const chord of seq) for (let b = 0; b < barsPer; b++, bar++) {
    const shape = shapes.chords[chord].shapes[0].frets, t0 = lead + bar * barDur;
    gt.bars.push({ bar, t0, t1: t0 + barDur, chord });
    const st = strumTimeline(shape, pat, bpm, { bars: 1, startT: t0, humanize: 0.3, seed: seed * 100 + bar, patterns: pats });
    for (const s of st) {
      gt.strokes.push({ t: s.t, step: s.step, bar, dir: s.dir, muted: !!s.muted && s.muted === 'dead', vel: s.velocity });
      s.strings.forEach((str, k) => {
        const f = s.frets[k], freq = 440 * 2 ** ((OPEN[str] + f - 69) / 12), dead = s.muted === 'dead', palm = s.muted === 'palm';
        pluck(y, s.t + s.offsets[k], freq, s.velocity * 0.25 * (0.9 + 0.2 * rand()), dead ? 0.9 : palm ? 0.985 : 0.9965, dead ? 0.08 : palm ? 0.5 : 2.2, rand);
      });
    }
  }
  if (drums) {
    for (let t = lead, i = 0; t < total - 2; t += beat / 2, i++) {
      const hit = (dur, f, noiseAmt, amp) => { const o = Math.round(t * SR); for (let k = 0; k < dur * SR && o + k < y.length; k++) { const e = Math.exp(-k / (dur * SR / 4)); y[o + k] += amp * e * ((1 - noiseAmt) * Math.sin(2 * Math.PI * f * k / SR) + noiseAmt * (rand() * 2 - 1)); } };
      if (i % 8 === 0 || i % 8 === 4) hit(0.15, 60, 0.05, 0.5);          // kick on 1, 3
      if (i % 8 === 2 || i % 8 === 6) hit(0.12, 200, 0.8, 0.35);         // snare on 2, 4
      hit(0.03, 8000, 1, 0.08);                                         // hat on every 8th
    }
  }
  let pk = 0; for (const v of y) pk = Math.max(pk, Math.abs(v)); for (let i = 0; i < y.length; i++) y[i] = y[i] / pk * 0.8;
  if (noise) { let e = 0; for (const v of y) e += v * v; const sd = Math.sqrt(e / y.length) * 10 ** (-noise / 20); for (let i = 0; i < y.length; i++) y[i] += sd * (rand() + rand() + rand() - 1.5) * 2; }
  return { y, sr: SR, gt };
}

export function writeWav(path, y, sr = SR) {
  const b = Buffer.alloc(44 + y.length * 2); b.write('RIFF', 0); b.writeUInt32LE(36 + y.length * 2, 4); b.write('WAVEfmt ', 8); b.writeUInt32LE(16, 16);
  b.writeUInt16LE(1, 20); b.writeUInt16LE(1, 22); b.writeUInt32LE(sr, 24); b.writeUInt32LE(sr * 2, 28); b.writeUInt16LE(2, 32); b.writeUInt16LE(16, 34);
  b.write('data', 36); b.writeUInt32LE(y.length * 2, 40); y.forEach((v, i) => b.writeInt16LE(Math.max(-32768, Math.min(32767, Math.round(v * 32767))), 44 + 2 * i));
  fs.writeFileSync(path, b);
}
// minimal 16-bit PCM wav reader -> mono Float32Array (assumes the sample rate you converted to, e.g. ffmpeg -ar 22050 -ac 1)
export function readWav(path) {
  const b = fs.readFileSync(path); let o = 12, ch = 1, sr = SR, data = null;
  while (o < b.length) { const id = b.toString('ascii', o, o + 4), n = b.readUInt32LE(o + 4);
    if (id === 'fmt ') { ch = b.readUInt16LE(o + 10); sr = b.readUInt32LE(o + 12); } if (id === 'data') { data = b.subarray(o + 8, o + 8 + Math.min(n, b.length - o - 8)); break; } o += 8 + n + (n & 1); }
  const frames = Math.floor(data.length / 2 / ch), y = new Float32Array(frames);
  for (let i = 0; i < frames; i++) { let s = 0; for (let c = 0; c < ch; c++) s += data.readInt16LE((i * ch + c) * 2); y[i] = s / ch / 32768; }
  return { y, sr };
}
