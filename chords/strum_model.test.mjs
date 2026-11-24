// node strum_model.test.mjs
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { strumTimeline, pickTimeline, progressionTimeline, parseToken } from './strum_model.js';
const pats = JSON.parse(readFileSync(new URL('./strum_patterns.json', import.meta.url)));
const byId = Object.fromEntries([...pats.strum, ...pats.pick].map(p => [p.id, p]));
const C = ['x', 3, 2, 0, 1, 0], G = [3, 2, 0, 0, 0, 3];
let n = 0; const ok = (name, fn) => { fn(); n++; console.log('ok', name); };

ok('every strum grid length matches steps_per_bar and parses', () => {
  for (const p of pats.strum) { const t = p.grid.split(/\s+/); assert.equal(t.length, p.steps_per_bar, p.id); t.forEach(parseToken); }
});
ok('basic pattern strokes and timing (90 bpm)', () => {
  const s = strumTimeline(C, byId.basic, 90, { humanize: 0 });
  assert.deepEqual(s.map(x => x.dir + x.step), ['D0', 'D2', 'U3', 'U5', 'D6', 'U7']);
  assert.ok(Math.abs(s[1].t - 2 * 0.5 * 60 / 90) < 1e-9);
});
ok('down low->high, up high->low (G B e only), muted string skipped', () => {
  const s = strumTimeline(C, byId.basic, 90, { humanize: 0 });
  assert.deepEqual(s[0].strings, [1, 2, 3, 4, 5]);
  assert.deepEqual(s[2].strings, [5, 4, 3]);
  const u4 = strumTimeline(C, { ...byId.basic, stroke_strings: 'up4' }, 90, { humanize: 0, patterns: pats });
  assert.deepEqual(u4[2].strings, [5, 4, 3, 2]);
  assert.ok(s[0].offsets.every((o, i, a) => i === 0 || o > a[i - 1]));
});
ok('per-string spread within 4-12 ms', () => {
  const s = strumTimeline(G, byId.down8, 100, { humanize: 0 })[0];
  const gaps = s.offsets.slice(1).map((o, i) => (o - s.offsets[i]) * 1000);
  assert.ok(gaps.every(g => g >= 4 && g <= 12), gaps.join(','));
});
ok('palm-mute and dead-strum flags', () => {
  assert.ok(strumTimeline(G, byId.rock8, 120, {}).every(x => x.muted === 'palm'));
  const f = strumTimeline(G, byId.funk16, 90, { humanize: 0 });
  assert.ok(f.some(x => x.muted === 'dead') && f.some(x => !x.muted));
});
ok('waltz: bass only on beat 1, 3 beats per bar', () => {
  const w = strumTimeline(G, byId.waltz, 120, { bars: 2, patterns: pats, humanize: 0 });
  assert.deepEqual(w[0].strings, [0]); assert.equal(w.length, 6);
  assert.ok(Math.abs(w[3].t - 1.5) < 1e-9);
});
ok('shuffle has 12 steps per bar', () => assert.equal(strumTimeline(C, byId.shuffle, 100, { humanize: 0 }).length, 8));
ok('travis alternates bass strings, skips muted low E', () => {
  const p = pickTimeline(C, byId.travis, 100, { humanize: 0 });
  assert.deepEqual(p.map(x => x.strings[0]), [1, 3, 2, 4, 1, 3, 2, 4]);
});
ok('progression offsets bars and tags chord', () => {
  const t = progressionTimeline([{ shape: C, bars: 1 }, { shape: G, bars: 1 }], byId.down8, 120, { humanize: 0 });
  assert.equal(t.length, 16); assert.ok(Math.abs(t[8].t - 2) < 1e-9); assert.equal(t[8].chord, 1);
});
ok('per-chord beats (G:2 D:2 Am:4) and per-chord pattern', () => {
  const D = ['x', 'x', 0, 2, 3, 2], Am = ['x', 0, 2, 2, 1, 0], p = byId.ballad16, bpm = 60;
  const t = progressionTimeline([{ shape: G, beats: 2 }, { shape: D, beats: 2 }, { shape: Am, beats: 4 }], p, bpm, { humanize: 0 });
  // half of the 16th ballad (D - - - D - D U) per 2-beat chord, the full bar on Am
  assert.deepEqual(t.filter(x => x.chord === 0).map(x => x.dir + x.step), ['D0', 'D4', 'D6', 'U7']);
  assert.ok(Math.abs(t.find(x => x.chord === 1).t - 2) < 1e-9 && Math.abs(t.find(x => x.chord === 2).t - 4) < 1e-9);
  assert.equal(t.filter(x => x.chord === 2).map(x => x.dir).join(''), 'DDDUUDUUDU');
  const o = progressionTimeline([{ shape: G, beats: 4 }, { shape: D, beats: 1.5, pattern: byId.quarters }], byId.down8, bpm, { humanize: 0 });
  assert.equal(o.filter(x => x.chord === 1).length, 2);
});
ok('ballad16 grid', () => assert.equal(byId.ballad16.grid, 'D - - - D - D U - U D U - U D U'));
ok('deterministic with seed', () => assert.deepEqual(strumTimeline(C, byId.basic, 90, { seed: 3 }), strumTimeline(C, byId.basic, 90, { seed: 3 })));
console.log(n + ' tests passed');
