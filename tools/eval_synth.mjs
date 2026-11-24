// Evaluate chord + strum detection on synthetic Karplus-Strong strummed progressions with known ground truth.
//   node tools/eval_synth.mjs [chords|strum|all] [--set quick|full]
import { renderCase } from './synth_guitar.mjs';
import { detectAudioChords } from '../js/audio-chords.js';
import { detectStrum, rhythm } from '../js/strumdetect.js';
import fs from 'node:fs';
const PATS = JSON.parse(fs.readFileSync(new URL('../chords/strum_patterns.json', import.meta.url)));
const norm = g => g.split(/\s+/).map(t => t.replace(/[>p]/g, '').toUpperCase()).join(' ');
const GRID = Object.fromEntries(PATS.strum.map(p => [p.id, norm(p.grid)]));
const PR = { A: ['G', 'D', 'Em', 'C'], B: ['Am', 'F', 'C', 'G'], C: ['C', 'G', 'Am', 'F'], D: ['E', 'A', 'E7', 'A'], E: ['Dm', 'Bb', 'F', 'C'],
  F: ['A', 'D', 'E', 'A'], G: ['Em', 'C', 'G', 'D'], H: ['D', 'A', 'Bm', 'G'], I: ['Am', 'G', 'F', 'E'] };
export const CASES = [
  ['basic', 90, 'A', 0, 0], ['down8', 110, 'B', 0, 0], ['folk', 80, 'C', 0, 0], ['pop', 100, 'D', 0, 0], ['ddudu', 120, 'E', 0, 0], ['ballad2', 70, 'F', 0, 0],
  ['basic', 130, 'G', 0, 0], ['folk', 100, 'H', 0, 0], ['down8', 75, 'I', 0, 0], ['funk16', 90, 'A', 0, 0], ['rock_open', 140, 'B', 0, 0], ['quarters', 90, 'C', 0, 0],
  ['basic', 100, 'A', 20, 0], ['folk', 90, 'B', 10, 0], ['pop', 110, 'G', 0, 1], ['basic', 95, 'C', 0, 1], ['down8', 85, 'H', 20, 1], ['ddudu', 100, 'F', 10, 1],
].map(([pattern, bpm, p, noise, drums]) => ({ pattern, bpm, prog: PR[p], noise, drums: !!drums, name: `${pattern}@${bpm} ${PR[p].join('-')}${noise ? ' n' + noise + 'dB' : ''}${drums ? ' drums' : ''}` }));

// strokes matched within 35 ms; pattern counted right when the matched id has the same grid ignoring accents/light/palm
export function scoreStrum(gt, r) {
  let hit = 0, dirOK = 0, mutOK = 0; const used = new Set();
  for (const g of gt.strokes) {
    let bj = -1, bd = 0.035; r.strokes.forEach((s, j) => { const d = Math.abs(s.t - g.t); if (d < bd && !used.has(j)) { bd = d; bj = j; } });
    if (bj < 0) continue; used.add(bj); hit++; const s = r.strokes[bj];
    if (g.muted ? s.muted : s.dir === g.dir) dirOK++; if (s.muted === g.muted) mutOK++;
  }
  const id = r.match && r.match.id;
  return { recall: hit / gt.strokes.length, precision: r.strokes.length ? hit / r.strokes.length : 0, dir: hit ? dirOK / hit : 0, muted: hit ? mutOK / hit : 0,
           patternOK: !!id && (id === gt.pattern || GRID[id] === GRID[gt.pattern]), id };
}
export function scoreChords(gt, r, rootOf = c => c.replace(/(m|7|maj7|sus2|sus4|5|dim|aug).*$/, '')) {
  let n = 0, ex = 0, rt = 0;
  for (const b of gt.bars) for (let t = b.t0 + 0.15; t < b.t1 - 0.05; t += 0.05) {
    const s = r.segments.find(s => s.t0 <= t && t < s.t1); n++;
    if (s) { if (s.chord === b.chord) ex++; if (rootOf(s.chord) === rootOf(b.chord)) rt++; }
  }
  return { exact: ex / n, root: rt / n };
}
if (process.argv[1].endsWith('eval_synth.mjs')) {
  const what = process.argv[2] || 'chords', sel = CASES.filter((_, i) => !process.env.ONLY || process.env.ONLY.split(',').includes(String(i)));
  const M = {}, add = (k, v) => (M[k] = (M[k] || 0) + v); let N = 0;
  for (const c of sel) {
    const { y, gt } = renderCase({ ...c, loops: 4, seed: 3 });
    const rh = rhythm(y), segs = (await detectAudioChords(y)).map(s => ({ ...s, chord: s.name })), r = { segments: segs }; N++;
    let line = c.name.padEnd(40) + ' bpm ' + rh.bpm.toFixed(0).padStart(3);
    if (what !== 'strum') { const s = scoreChords(gt, r); add('exact', s.exact); add('root', s.root);
      line += ` exact ${s.exact.toFixed(2)} root ${s.root.toFixed(2)} ` + segs.slice(0, 6).map(b => b.name).join(' '); }
    if (what !== 'chords') { const st = detectStrum(y, { rhythm: rh, chords: segs, patterns: PATS }), s = scoreStrum(gt, st), raw = scoreStrum(gt, { ...st, strokes: st.strokes.map(k => ({ ...k, dir: k.dir_raw })) });
      for (const k of ['recall', 'precision', 'dir', 'muted']) add(k, s[k]); add('dir_raw', raw.dir); add('pattern', +s.patternOK);
      line += ` | on R${s.recall.toFixed(2)} P${s.precision.toFixed(2)} dir ${s.dir.toFixed(2)} (raw ${raw.dir.toFixed(2)}) mute ${s.muted.toFixed(2)} ${st.direction_method} sw${st.swing} ${st.subdivision} -> ${s.id}(${st.match && st.match.dist}) [${st.detected_grid}] ${s.patternOK ? 'OK' : 'X'}`; }
    console.log(line);
  }
  console.log('MEAN', Object.entries(M).map(([k, v]) => k + ' ' + (v / N).toFixed(3)).join('  '));
}
