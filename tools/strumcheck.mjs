// Offline check for chord strumming (Chords practice mode), same style as selfcheck.mjs: simulates the exact
// performer code at 60 fps and measures
//   1. at each chord onset: distance of every fingertip to the shape's fret position (string, fret contact point);
//   2. for every strummed string: |time the strumming hand's contact point crosses the string - the stroke's string time|.
// Default: G-D-Em-C, 'basic' pattern, 90 BPM, 2 bars per chord, pick and fingers. `node tools/strumcheck.mjs [--all]`
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const imp = p => import(pathToFileURL(path.join(ROOT, p)).href);
const TL = await imp('js/timeline.js'), H = await imp('js/hand.js'), PF = await imp('js/performer.js'), G = await imp('js/guitar.js');
const CS = await imp('js/chordsong.js');
const lib = JSON.parse(fs.readFileSync(path.join(ROOT, 'chords/chord_shapes.json')));
const pats = JSON.parse(fs.readFileSync(path.join(ROOT, 'chords/strum_patterns.json')));
const byId = Object.fromEntries([...pats.strum, ...pats.pick].map(p => [p.id, p]));

async function loadHand(side) {
  const buf = fs.readFileSync(path.join(ROOT, `assets/generic-hand-${side}.glb`));
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  const gltf = await new Promise((res, rej) => new GLTFLoader().parse(ab, '', res, rej));
  return new H.HandRig(gltf.scene, side);
}
const left = await loadHand('left'), right = await loadHand('right');
const stats = (a, d = 2) => a.length ? { n: a.length, mean: +(a.reduce((x, y) => x + y, 0) / a.length).toFixed(d), max: +Math.max(...a).toFixed(d) } : { n: 0 };

function run(symbols, patternId, bpm, hand, { bars = 2, verbose = true } = {}) {
  const prog = symbols.map(s => ({ symbol: s, shape: CS.shapesOf(lib, s)[0], bars }));
  const song = CS.buildChordSong(prog, byId[patternId], bpm, { patterns: pats, strumHand: hand });
  return measure(`${symbols.join('-')} / ${patternId} / ${bpm} bpm / ${hand} hand`, song.events, song.strokes, song.total + 0.5, hand, verbose);
}

function measure(label, events, strokes, end, hand, verbose = true) {
  const perf = new PF.Performer(left, right);
  perf.setEvents(events);
  const dt = 1 / 60;
  const fretErr = [], worst = [], frames = [];
  let ei = 0, first = true;
  for (let T = 0; T < end; T += dt) {
    perf.update(T, dt, first); first = false;
    while (ei < events.length && events[ei].t <= T) {
      const ev = events[ei++];
      for (const n of ev.fretted) {
        if (ev.byFinger[n.finger - 1] !== n) continue;                 // barre: only the tip string is checked
        const k = TL.FINGER_KEYS[n.finger - 1];
        const exact = new THREE.Vector3(G.contactX(n.f), TL.PRESS_Y, G.stringZ(n.s, G.contactX(n.f)));
        const d = left.tipG(k).distanceTo(exact) * 1000;
        fretErr.push(d); worst.push({ d, sym: ev.symbol, s: n.s, f: n.f, k });
      }
    }
    const tip = hand === 'pick' ? perf.pick.tip.clone() : right.tipG('index');
    frames.push({ T, x: tip.x, y: tip.y, z: tip.z });
  }
  // contact crossings
  const crossErr = [], missed = [];
  for (const st of strokes) {
    if (st.muted === 'dead') { /* still swept through the strings */ }
    st.strings.forEach((sl, k) => {
      const s3 = 5 - sl, ts = st.t + st.offsets[k];
      let best = null;
      for (let i = 1; i < frames.length; i++) {
        const a = frames[i - 1], b = frames[i];
        if (b.T < ts - 0.08) continue; if (a.T > ts + 0.08) break;
        const za = G.stringZ(s3, a.x), zb = G.stringZ(s3, b.x);
        const da = a.z - za, db = b.z - zb;
        if (da * db > 0 && Math.abs(db) > 1e-9 && Math.abs(da) > 1e-9) continue;
        const u = da === db ? 0 : da / (da - db), y = a.y + (b.y - a.y) * u, x = a.x + (b.x - a.x) * u;
        if (y > G.stringY(x) + 0.001) continue;                         // passes above the string: no contact
        const tc = a.T + (b.T - a.T) * u;
        if (!best || Math.abs(tc - ts) < Math.abs(best - ts)) best = tc;
      }
      if (best === null) missed.push({ t: ts, s3 }); else crossErr.push(Math.abs(best - ts) * 1000);
    });
  }
  worst.sort((a, b) => b.d - a.d);
  const r = { fret: stats(fretErr), cross: stats(crossErr, 1), within15: crossErr.filter(v => v <= 15).length, missed: missed.length, strings: crossErr.length };
  if (verbose) {
    console.log(`\n== ${label} ==`);
    console.log('fretting fingertip error at onset (mm):', JSON.stringify(r.fret), ' worst:', worst.slice(0, 3).map(w => `${w.sym || "note"} s${w.s}f${w.f} ${w.k} ${w.d.toFixed(1)}`).join(' | '));
    console.log(`strum contact time error (ms): ${JSON.stringify(r.cross)}  within +-15 ms: ${r.within15}/${r.strings}  missed contacts: ${r.missed}`);
  }
  return r;
}

run(['G', 'D', 'Em', 'C'], 'basic', 90, 'pick');
run(['G', 'D', 'Em', 'C'], 'basic', 90, 'fingers');
// a mixed song (tools/fixtures/mixed-song.json): 2 strummed bars (G D, detected strokes), 2 bars of single notes, 2 bars of
// tab chords (exact voicing) on ONE timeline (js/songchords.js buildMixedTimeline): strum crossings in the strummed bars,
// fingertips on every chord and note onset
{
  const F = await imp('js/fingering.js'), SC = await imp('js/songchords.js');
  const fx = JSON.parse(fs.readFileSync(path.join(ROOT, 'tools/fixtures/mixed-song.json')));
  for (const hand of ['pick', 'fingers']) {
    const tab = TL.prepareTimeline(F.solveFingering(F.normalizeScore(fx.score)), hand);
    const mix = SC.buildMixedTimeline({ score: fx.score, events: tab, mode: hand, track: { chords: fx.score.chords, strum: fx.score.strum }, lib, pats });
    measure(`mixed song (strum G D | notes | tab chords G D) / ${hand} hand`, mix.events, mix.strokes, mix.end + 0.5, hand);
  }
}
if (process.argv.includes('--all')) {
  for (const syms of [['C', 'G', 'Am', 'F'], ['Am', 'F', 'C', 'G'], ['E', 'A', 'B7'], ['A7', 'D7', 'E7']]) run(syms, 'pop', 90, 'pick', { bars: 1 });
  run(['G', 'D', 'Em', 'C'], 'funk16', 100, 'pick');
  run(['G', 'D', 'Em', 'C'], 'rock8', 120, 'pick');
}
