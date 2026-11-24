// Edge cases beyond the bundled songs: node --test tools/regression.mjs
import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs';
import crypto from 'node:crypto';
import { stampHTML, localImports } from './stamp.mjs';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { HandRig } from '../js/hand.js';
import { Performer } from '../js/performer.js';
import { normalizeScore, solveFingering } from '../js/fingering.js';
import { prepareTimeline } from '../js/timeline.js';
import { analyzeHarmony } from '../js/harmony.js';
import { createHarness } from './fixtures/create-harness.mjs';
import { loadSongAudio, saveSongAudio, deleteSongAudio, shiftScore } from '../js/song-audio.js';
import { assignChord, toScore, reconcileChords, findTrim } from '../js/songmaker.js';
import { playedChordName, cleanChordSegments } from '../js/chord-agreement.js';
import { describeNotes } from '../js/theory.js';
import { buildMixedTimeline } from '../js/songchords.js';
import { detectAudioChords } from '../js/audio-chords.js';
import { chordSource, ChordPanel } from '../js/chordpanel.js';
import { CHORDS, tone, joinAudio, audioFixtures, chordAccuracy } from './fixtures/audio.mjs';
import { detectAudioChords as baselineChords } from './fixtures/audio-chords-baseline.mjs';
import { decodePCM16Wav } from './fixtures/wav.mjs';
import { playbackHarness } from './fixtures/playback-harness.mjs';
import { fakeIndexedDB } from './fixtures/indexeddb.mjs';
import { toScore as beforePhantoms } from './fixtures/songmaker-before-phantoms.mjs';
import { ksNotes, residual, phantomFixtures } from './fixtures/phantom-notes.mjs';
import { CaptureMachine, fitRect, videoAspect, clockText } from '../js/capture.js';

const prepare = (score, mode = 'pick') => prepareTimeline(solveFingering(normalizeScore(score)), mode);
const melody = () => ({ bpm: 120, notes: Array.from({ length: 8 }, (_, i) => ({ t: 2 + i / 4, n: [[0, [8, 12, 15, 12][i % 4]]] })) });
const library = JSON.parse(fs.readFileSync(new URL('../chords/chord_shapes.json', import.meta.url)));
const rawChord = (ps, t = 0, a = 1) => ps.map((p, i) => ({ t: t + i * .008, e: t + .5, p, a }));
const openMidi = [64, 59, 55, 50, 45, 40];

test('create import map hashes every local entry, re-export and parent-directory dependency', () => {
  const html = fs.readFileSync(new URL('../create.html', import.meta.url), 'utf8');
  const stamped = stampHTML(html), match = stamped.match(/<script type="importmap">([\s\S]*?)<\/script>/);
  const { imports } = JSON.parse(match[1]), base = 'https://example.test/tutor/';
  const resolved = new Map(Object.entries(imports).map(([k, v]) => [new URL(k, base).href, new URL(v, base).href]));
  assert.ok(match.index < stamped.indexOf('<link rel="modulepreload"'));
  const edges = [
    ['create.html', './js/songmaker.js'], ['create.html', './js/song-audio.js'], ['create.html', './js/audio-chords.js'],
    ['js/songmaker.js?v=parent', './chord-agreement.js'], ['js/chord-agreement.js?v=parent', '../chords/chordnamer.js'],
  ];
  for (const [parent, specifier] of edges) {
    const plain = new URL(specifier, base + parent), target = new URL(resolved.get(plain.href));
    const file = plain.pathname.slice('/tutor/'.length);
    const source = fs.readFileSync(new URL('../' + file, import.meta.url), 'utf8').replace(/\r\n?/g, '\n');
    assert.equal(target.searchParams.get('v'), crypto.createHash('sha256').update(source).digest('hex').slice(0, 10), file);
    assert.ok(stamped.includes(`href="${file}${target.search}"`), file + ' preload');
  }
  assert.equal(stampHTML(stamped), stamped);
  assert.equal(stampHTML(stamped.replace(/\n/g, '\r\n')), stamped);
  assert.ok(!stamped.includes('\r'));
  assert.equal(stamped, html, 'run npm run stamp');
  const tabs = fs.readFileSync(new URL('../tabs.html', import.meta.url), 'utf8');
  assert.equal(stampHTML(tabs), tabs, 'classic-only tabs.html stays byte-identical');
  assert.deepEqual(localImports("export { x } from '../chords/x.js';\nimport './side.js';\nconst f = () => import('./lazy.js');\nimport 'three';"),
    ['../chords/x.js', './side.js', './lazy.js']);
});

test('3D loads the exact stored Blob before other sources and is ready for Play', async () => {
  const score = { bpm: 120, grid: [12, 12.5], notes: [{ t: 12.2, n: [[0, 0]] }], chords: [{ t0: 12, t1: 13, name: 'C', conf: .9 }] };
  const song = { score, user: true, audioOffset: 12, audioDuration: 1, video: 'test', audio: 'test.mp3', mode: 'fingers' };
  const blob = new Blob(['exact take']), h = playbackHarness({ test: song }, async () => blob);
  await h.ctx.loadSong('test');
  assert.equal(h.created[0], blob); assert.deepEqual(h.sources, []);
  assert.equal(h.ctx.events[0].t, .2); assert.equal(h.ctx.curScore.chords[0].t0, 0);
  assert.equal(h.ui.play.textContent, 'Play'); assert.equal(h.audio.currentTime, 0);
  assert.equal(h.ui.scrub.max, 1);
  assert.equal(h.media.length, 0);
  h.ctx.setPlaying(true); await Promise.resolve(); assert.deepEqual(h.media, ['play']);
  await h.ctx.loadSong('test'); assert.deepEqual(h.revoked, ['blob:test-1']);
  assert.equal(score.notes[0].t, 12.2);
});

test('missing stored audio falls back to video then synth without shifting video times', async () => {
  const score = { bpm: 120, notes: [{ t: 12.2, n: [[0, 0]] }] };
  const h = playbackHarness({ video: { score, user: true, video: 'test', audioOffset: 12 }, synth: { score, user: true } }, async () => null);
  await h.ctx.loadSong('video'); assert.deepEqual(h.sources, ['video']);
  assert.equal(h.ctx.events[0].t, 12.2);
  await h.ctx.loadSong('synth'); assert.deepEqual(h.sources, ['video', 'synth']);
  assert.equal(h.ui.play.textContent, 'Play');
});

test('late recording reads cannot overwrite a newer lesson, even with the same id', async () => {
  const pending = [], score = { bpm: 120, notes: [{ t: .2, n: [[0, 0]] }] };
  const h = playbackHarness({ test: { score, user: true } }, () => new Promise(resolve => pending.push(resolve)));
  const first = h.ctx.loadSong('test'), second = h.ctx.loadSong('test');
  const latest = new Blob(['latest']); pending[1](latest); await second;
  pending[0](new Blob(['stale'])); await first;
  assert.deepEqual(h.created, [latest]);
});

test('IndexedDB Blob round-trip waits for commit and catches storage failure stages', async () => {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'indexedDB');
  try {
    const db = fakeIndexedDB(); Object.defineProperty(globalThis, 'indexedDB', { configurable: true, value: db });
    const blob = new Blob(['recorded bytes']);
    assert.equal(await saveSongAudio('one', blob), true);
    assert.equal(await (await loadSongAudio('one')).text(), 'recorded bytes');
    assert.equal(await deleteSongAudio('one'), true); assert.equal(await loadSongAudio('one'), null);
    assert.equal(db.closes, 4);
    for (const mode of ['open', 'upgrade', 'transaction', 'request', 'quota', 'abort', 'error', 'blocked']) {
      const broken = fakeIndexedDB(mode);
      Object.defineProperty(globalThis, 'indexedDB', { configurable: true, value: broken });
      assert.equal(await saveSongAudio('one', blob), false, mode);
      assert.equal(await loadSongAudio('one'), null, mode);
      assert.equal(await deleteSongAudio('one'), false, mode);
      assert.equal(broken.data.size, 0, mode);
    }
  } finally {
    if (descriptor) Object.defineProperty(globalThis, 'indexedDB', descriptor); else delete globalThis.indexedDB;
  }
});

test('audio chroma names C, G, Am and F for sine and harmonic-sum chords', async () => {
  for (const harmonics of [1, 5]) for (const [name, ps] of Object.entries(CHORDS)) {
    const y = tone(ps, { harmonics }), segs = await detectAudioChords(y);
    assert.deepEqual(segs.map(s => s.name), [name], name + ' harmonics=' + harmonics);
    assert.equal(segs[0].t0, 0); assert.equal(segs[0].t1, 1);
    assert.ok(segs[0].conf >= 0 && segs[0].conf <= 1);
  }
});

test('audio smoothing follows a progression and leaves silence unlabelled', async () => {
  const silence = new Float32Array(11025);
  const y = joinAudio(silence, ...Object.values(CHORDS).map(ps => tone(ps, { harmonics: 5 })), silence);
  const segs = await detectAudioChords(y);
  assert.deepEqual(segs.map(s => s.name), ['C', 'G', 'Am', 'F']);
  assert.deepEqual(segs.map(s => [s.t0, s.t1]), [[.5, 1.5], [1.5, 2.5], [2.5, 3.5], [3.5, 4.5]]);
  assert.deepEqual(await detectAudioChords(silence), []);
  assert.deepEqual(await detectAudioChords(new Float32Array()), []);
});

test('detector improves the fixture suite without losing accuracy on any fixture', async () => {
  let improvement = 0;
  for (const f of audioFixtures()) {
    const before = chordAccuracy(await baselineChords(f.y), f.truth).percent;
    const after = chordAccuracy(await detectAudioChords(f.y), f.truth).percent;
    assert.ok(after >= before, f.name + ': ' + before + ' -> ' + after);
    improvement += after - before;
  }
  assert.ok(improvement > 0);
});

test('PCM WAV decoding preserves signed samples, channel averaging and chunk padding', () => {
  const b = Buffer.alloc(62);
  b.write('RIFF'); b.writeUInt32LE(54, 4); b.write('WAVE', 8);
  b.write('JUNK', 12); b.writeUInt32LE(1, 16); b[20] = 123;
  b.write('fmt ', 22); b.writeUInt32LE(16, 26); b.writeUInt16LE(1, 30);
  b.writeUInt16LE(2, 32); b.writeUInt32LE(22050, 34); b.writeUInt32LE(88200, 38);
  b.writeUInt16LE(4, 42); b.writeUInt16LE(16, 44); b.write('data', 46); b.writeUInt32LE(8, 50);
  [-32768, 0, 32767, 16384].forEach((v, i) => b.writeInt16LE(v, 54 + 2 * i));
  const decoded = decodePCM16Wav(b);
  assert.equal(decoded.sampleRate, 22050); assert.equal(decoded.channels, 2);
  assert.deepEqual([...decoded.y], [-.5, (32767 + 16384) / 65536]);
  assert.throws(() => decodePCM16Wav(b.subarray(0, 60)), /Truncated/);
});

test('panel prefers audio segments, including an empty result, and labels melody fallback', () => {
  const events = prepare(melody()), chords = [{ t0: 0, t1: 1, name: 'C', conf: .9 }];
  assert.deepEqual(chordSource({ chords }, events), { segs: chords, label: 'from the audio' });
  assert.deepEqual(chordSource({ chords: [] }, events), { segs: [], label: 'from the audio' });
  const fallback = chordSource({}, events);
  assert.equal(fallback.label, 'estimated from the melody'); assert.ok(fallback.segs.length);
});

test('monophonic conversion exactly matches the original converter fixture', () => {
  const f = JSON.parse(fs.readFileSync(new URL('./fixtures/monophonic.json', import.meta.url)));
  for (const c of f.cases) for (const lib of [null, library]) assert.equal(JSON.stringify(toScore(f.raw, c.offset, .75, lib)), JSON.stringify(c.score));
});

test('real Chrome Basic Pitch output retains exactly eight correctly named strums', () => {
  const fixture = JSON.parse(fs.readFileSync(new URL('./fixtures/basic-pitch-strum-real.json', import.meta.url)));
  const raw = fixture.notes.map(([t, e, p, a]) => ({ t, e, p, a })), before = JSON.stringify(raw);
  for (const lib of [null, library]) {
    const score = toScore(raw, 0, .75, lib);
    assert.equal(score.notes.length, 8);
    assert.deepEqual(score.notes.map(n => playedChordName(n)), fixture.truth.map(n => n.chord));
    assert.deepEqual(score.notes.map(n => n.t), [.522, 1.532, 2.497, 3.496, 4.496, 5.529, 6.494, 7.504]);
    score.notes.forEach((n, i) => assert.ok(Math.abs(n.t - fixture.truth[i].t) < .04));
  }
  assert.equal(JSON.stringify(raw), before);
  const chord = rawChord([48, 52, 55], 0).map(n => ({ ...n, e: 1 }));
  const split = { t: 1, e: 1.5, p: 55, a: .8 };
  assert.equal(toScore([...chord, split]).notes.length, 2, 'no filtering after the chord stops');
});

test('phantom fixtures remove only residuals and preserve all eight strums and real soft notes', () => {
  for (const f of phantomFixtures()) for (const lib of [null, library]) for (const offset of [0, 3.25, -4.6]) {
    const input = JSON.stringify(f.raw), score = toScore(f.raw, offset, .75, lib);
    assert.equal(JSON.stringify(score), JSON.stringify(beforePhantoms(f.expected, offset, .75, lib)), f.name);
    assert.equal(JSON.stringify(f.raw), input, 'source notes are immutable');
    if (offset === 0) {
      const old = beforePhantoms(f.raw, 0, .75, lib);
      assert.equal(score.notes.filter(n => n.n.length > 1).length, 8, f.name);
      assert.deepEqual(score.notes.filter(n => n.n.length > 1).map(n => [n.t, n.n, n.sustain]),
        old.notes.filter(n => n.n.length > 1).map(n => [n.t, n.n, n.sustain]), f.name);
    }
  }
});

test('residual filter keeps ambiguous groups, stronger/longer notes, expired chords and pitch transitions', () => {
  const base = ksNotes(), ghost = residual(71, 4.82);
  const cases = [
    [...base, { ...ghost, a: .4 }], // absolute threshold
    [...base.map(n => ({ ...n, a: .5 })), ghost], // not weak relative to chord
    [...base, { ...ghost, e: ghost.t + .181 }],
    [...base, residual(62, 4.82)], // D4 is not a low harmonic of a sounding Am note
    [...base, residual(59, 4.82)], // B3: same class as E3's 3rd partial, wrong octave
    [...base, ghost, residual(74, 4.83)], // ambiguous dyad, not a singleton
    [...base, residual(64, 4.60)], // near the strum: preserve late strings
    [...base, residual(64, 8.6)], // chord expired
    [...base.map(n => ({ ...n, e: n.t >= 4.5 && n.t < 5.5 ? 4.9 : n.e })), ghost], // tail crosses end
    [...base.map(n => ({ ...n, e: n.t >= 4.5 && n.t < 5.5 && n.p % 12 === 0 ? 4.7 : n.e })), ghost], // <3 classes ringing
    [...base, { p: 69, t: 4.7, e: 4.82, a: .5 }, ghost], // single-note hammer A4 -> B4
    [...base, { ...ghost, e: NaN }], // missing duration is not evidence
  ];
  for (const raw of cases) assert.equal(JSON.stringify(toScore(raw)), JSON.stringify(beforePhantoms(raw)));
  // The latest chord must replace prior evidence, even when old offsets overlap.
  const changes = [...rawChord([48, 52, 55, 60, 64], 0).map(n => ({ ...n, e: 3 })),
    ...rawChord([50, 57, 62, 66], 1), residual(64, 1.2)];
  assert.equal(JSON.stringify(toScore(changes)), JSON.stringify(beforePhantoms(changes)));
  // Soft, short monophonic lines have no chord context and remain byte-identical.
  const mono = [52, 64, 71, 76, 55, 67].map((p, i) => residual(p, .2 + i * .25));
  assert.equal(JSON.stringify(toScore(mono)), JSON.stringify(beforePhantoms(mono)));
});

test('creation saves the filtered eight-strum score and four chord labels', async () => {
  const f = phantomFixtures()[1], h = createHarness({ notes: f.raw,
    chords: [{ t0: .5, t1: 8.5, name: 'Em', conf: .8 }] });
  h.get('file').onchange({ target: { files: [new File(['fixture'], 'phantoms.wav')] } });
  await h.get('go').onclick();
  const { score } = [...h.songs.values()][0];
  assert.equal(score.notes.length, 8);
  assert.deepEqual(score.chords.map(s => [s.t0, s.t1, s.name]),
    [[.5, 2.5, 'C'], [2.5, 4.5, 'G'], [4.5, 6.5, 'Am'], [6.5, 8.5, 'F']]);
  assert.deepEqual(chordSource(score, normalizeScore(score)).segs, score.chords);
});

test('played triads/sevenths split audio labels, survive shifts and guard older scores', () => {
  const notes = [
    { t: .5, n: [[2, 0], [3, 2], [4, 3]] }, // C
    { t: 2.5, n: [[3, 0], [4, 2], [5, 3]] }, // G
    { t: 4.5, n: [[1, 1], [3, 2], [4, 0]] }, // Am
    { t: 6.5, n: [[2, 2], [4, 3], [5, 1]] }, // F
  ];
  const chords = [{ t0: 0, t1: 5, name: 'Em', conf: .7 }, { t0: 5, t1: 8.5, name: 'A', conf: .7 }];
  const before = structuredClone({ notes, chords });
  const result = reconcileChords(chords, notes);
  assert.deepEqual(result.map(s => [s.t0, s.t1, s.name]), [[0, .5, 'Em'], [.5, 2.5, 'C'], [2.5, 4.5, 'G'], [4.5, 6.5, 'Am'], [6.5, 8.5, 'F']]);
  const prepared = normalizeScore({ notes });
  prepared.forEach((e, i) => assert.equal(playedChordName(e), describeNotes(e.notes).name));
  assert.deepEqual(chordSource({ chords }, prepared).segs, result);
  assert.match(chordSource({ chords }, prepared).label, /played notes/);
  assert.deepEqual(reconcileChords(result, notes), result);
  assert.deepEqual({ notes, chords }, before);
  const shifted = shiftScore({ chords, notes }, 12);
  assert.deepEqual(reconcileChords(shifted.chords, shifted.notes), shiftScore({ chords: result, notes }, 12).chords);
  assert.equal(playedChordName({ n: [[0, 0], [1, 1], [2, 0], [4, 3]] }), 'C');
  assert.equal(playedChordName({ n: [[0, 0], [1, 1], [2, 3], [4, 3]] }), 'C7');
  assert.equal(playedChordName({ n: [[0, 0], [1, 1]] }), null);
  const held = reconcileChords(chords, [notes[0], { t: 1, n: [[0, 0]] }]);
  assert.deepEqual(held.map(s => [s.t0, s.t1, s.name]), [[0, .5, 'Em'], [.5, 8.5, 'C']]);
  assert.deepEqual(reconcileChords([], [{ ...notes[0], sustain: 1 }]).map(s => [s.t0, s.t1, s.name]), [[.5, 1.5, 'C']]);
  assert.deepEqual(reconcileChords([{ t0: 0, t1: .5, name: 'G' }], [notes[0]]).map(s => s.name), ['G']);
});

test('creation persists note-agreed labels before navigation', async () => {
  const h = createHarness({ notes: rawChord([45, 48, 52], .2), chords: [{ t0: 0, t1: 1, name: 'Em', conf: .8 }] });
  h.get('file').onchange({ target: { files: [new File(['upload'], 'fixture.wav')] } });
  await h.get('go').onclick();
  const song = [...h.songs.values()][0];
  assert.deepEqual(song.score.chords.map(s => [s.t0, s.t1, s.name]), [[0, .2, 'Em'], [.2, .7, 'Am']]);
  assert.equal(h.navigation[0], 'index.html#' + song.id);
});

test('eight strums save four held segments and the runtime panel agrees at the QA timestamps', async () => {
  const pitches = [[48, 52, 55, 60, 64], [43, 47, 50, 55, 59, 67], [45, 52, 57, 60, 64], [41, 48, 53, 57, 60, 65]];
  const notes = pitches.flatMap((ps, c) => [0, 1].flatMap(beat => ps.map((p, s) => ({
    p, t: .5 + 2 * c + beat + .012 * s, e: 2.5 + 2 * c, a: [.42, .85, .56, .71, .38, .63][s],
  }))));
  const h = createHarness({ notes, chords: [{ t0: .5, t1: 5, name: 'Em', conf: .8 }, { t0: 5, t1: 9, name: 'A', conf: .6 }] });
  h.get('file').onchange({ target: { files: [new File(['fixture'], 'strum.wav')] } });
  await h.get('go').onclick();
  const { score } = [...h.songs.values()][0];
  assert.equal(score.notes.length, 8);
  assert.ok(score.notes.every(n => playedChordName(n)));
  assert.deepEqual(score.chords.map(s => [s.t0, s.t1, s.name]), [[.5, 2.5, 'C'], [2.5, 4.5, 'G'], [4.5, 6.5, 'Am'], [6.5, 8.5, 'F']]);
  const runtime = chordSource(score, normalizeScore(score)).segs;
  assert.deepEqual(runtime, score.chords);
  assert.deepEqual([.5, 1.7, 2.7, 4.7, 6.7, 7.7].map(t => runtime.find(s => s.t0 <= t && t < s.t1)?.name),
    ['C', 'C', 'G', 'Am', 'F', 'F']);
});

test('held strums bridge audio windows from t0, ignore fragments, and stop at measured or explicit rests', () => {
  const C = [[0, 0], [1, 1], [2, 0], [4, 3]], F = [[0, 1], [1, 1], [2, 2], [5, 1]];
  const notes = [{ t: .5, n: C, sustain: 2 }, { t: .7, n: [[0, 0]] },
    { t: 1.5, n: C, sustain: 1 }, { t: 3, n: F, sustain: 1 }, { t: 3.5, n: [] }];
  const audio = [{ t0: 1, t1: 1.51, name: 'Em', conf: .8 },
    { t0: 1.51, t1: 2, name: 'A', conf: .6 }, { t0: 2, t1: 5, name: 'Em', conf: .9 }];
  const before = structuredClone({ notes, audio }), segs = reconcileChords(audio, notes);
  assert.deepEqual(segs.map(s => [s.t0, s.t1, s.name]), [[.5, 2.5, 'C'], [3, 3.5, 'F']]);
  assert.deepEqual(reconcileChords(segs, notes), segs);
  assert.deepEqual(chordSource({ chords: audio, notes }, normalizeScore({ notes })).segs, segs);
  assert.deepEqual({ notes, audio }, before);
  const shifted = shiftScore({ chords: audio, notes }, 12);
  assert.deepEqual(reconcileChords(shifted.chords, shifted.notes), shiftScore({ chords: segs, notes }, 12).chords);
  const resumed = reconcileChords(audio, [...notes, { t: 4, n: [[0, 0]] }]);
  assert.deepEqual(resumed.map(s => [s.t0, s.t1, s.name]), [[.5, 2.5, 'C'], [3, 3.5, 'F'], [4, 5, 'Em']]);
});

test('segment cleanup rejects tiny/invalid intervals and keeps minimum confidence across windows', () => {
  const segs = cleanChordSegments([
    { t0: 0, t1: 1, name: 'C', conf: .9, source: 'notes' },
    { t0: 1, t1: 1.02, name: 'C', conf: .4, source: 'notes' },
    { t0: 1.02, t1: 2, name: 'C', conf: .8, source: 'notes' },
    { t0: 2, t1: 2, name: 'Em' }, { t0: 2, t1: 2.049, name: 'Em' },
    { t0: 3, t1: 3.05, name: 'G' }, { t0: NaN, t1: 4, name: 'A' },
  ]);
  assert.deepEqual(segs, [{ t0: 0, t1: 2, name: 'C', conf: .4, source: 'notes' }, { t0: 3, t1: 3.05, name: 'G' }]);
  assert.deepEqual(cleanChordSegments(segs), segs);
});

test('panel countdown skips repeated chords, tiny segments and never rounds a future onset to zero', () => {
  const panel = Object.create(ChordPanel.prototype);
  panel.el = {}; panel.ui = { next: {} }; panel._show = s => { panel.current = s?.name ?? null; };
  panel.setSegments([{ t0: .5, t1: 1.5, name: 'C' }, { t0: 1.5, t1: 2.5, name: 'C', source: 'notes' },
    { t0: 2.5, t1: 2.5, name: 'Em' }, { t0: 2.5, t1: 3.5, name: 'G' }]);
  for (const t of [.5, 1.7, 2.49, .6]) {
    panel.update(t); assert.equal(panel.current, 'C'); assert.match(panel.ui.next.innerHTML, /<b>G<\/b>/);
    assert.doesNotMatch(panel.ui.next.innerHTML, /0\.0 s/);
  }
  panel.update(3.5); assert.equal(panel.current, null); assert.equal(panel.ui.next.innerHTML, '');
});

test('loud overlapping chord notes share one event and repeated strums survive', () => {
  const ps = [48, 52, 55, 60, 64], raw = [...rawChord(ps), ...rawChord(ps, .1)];
  const before = structuredClone(raw), score = toScore(raw, 0, .75, library);
  assert.equal(score.notes.length, 2);
  for (const n of score.notes) {
    assert.equal(n.n.length, 5);
    assert.deepEqual(n.n.map(([s, f]) => openMidi[s] + f).sort((a, b) => a - b), ps);
    assert.deepEqual(n.n, [[0, 0], [1, 1], [2, 0], [3, 2], [4, 3]]);
  }
  assert.deepEqual(raw, before);
});

test('quiet, non-overlapping, two-class and unplayable groups retain the melody path', () => {
  for (const raw of [rawChord([60, 64, 67], 0, .1), rawChord([60, 64]), rawChord([40, 41, 42]),
    rawChord([60, 64, 67]).map((n, i) => ({ ...n, e: n.t + .001 })),
    rawChord([60, 64, 67]).map((n, i) => ({ ...n, a: i === 2 ? 1 : .2 }))]) {
    assert.equal(toScore(raw).notes[0].n.length, 1);
  }
  assert.equal(toScore(rawChord([60, 64, 67]).map((n, i) => ({ ...n, t: i * .2, e: i * .2 + .15 }))).notes.length, 3);
});

test('KS-style 8/12/15 ms per string strums stay whole at their first onset, in both directions', () => {
  const chords = [[48, 52, 55, 60, 64], [43, 47, 50, 55, 59, 67], [45, 52, 57, 60, 64], [41, 48, 53, 57, 60, 65]];
  for (const spread of [.008, .012, .015]) for (const reverse of [false, true]) for (const lib of [null, library]) {
    const raw = chords.flatMap((ps, c) => [0, 1].flatMap(beat => (reverse ? [...ps].reverse() : ps)
      .map((p, s) => ({ p, t: .5 + 2 * c + beat + spread * s, e: 2.5 + 2 * c, a: [.42, .85, .56, .71, .38, .63][s] }))));
    const score = toScore(raw, 0, .75, lib);
    assert.equal(JSON.stringify(score), JSON.stringify(beforePhantoms(raw, 0, .75, lib)));
    assert.deepEqual(score.notes.map(n => n.t), [.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]);
    score.notes.forEach((n, i) => assert.deepEqual(n.n.map(([s, f]) => openMidi[s] + f).sort((a, b) => a - b), chords[i >> 1]));
  }
  for (const ps of [[48, 52, 55, 60, 64], [48, 55, 52, 64, 60]]) {
    const raw = ps.map((p, i) => ({ p, t: i * .022, e: .5, a: 1 }));
    assert.equal(toScore(raw).notes.length, 1); // nonmonotonic 88 ms
  }
  const sweep = [43, 47, 50, 55, 59, 67].map((p, i) => ({ p, t: i * .020, e: .5, a: 1 }));
  assert.equal(toScore(sweep).notes.length, 1); // monotonic 100 ms
  assert.ok(toScore(sweep.map(n => ({ ...n, t: n.t * 2 }))).notes.length > 1);
  assert.ok(toScore(sweep.map(n => ({ ...n, e: n.t + .005 }))).notes.length > 1);
});

test('chord assignments preserve pitches on distinct strings within a four-fret span', () => {
  for (const ps of [[48, 52, 55, 60, 64], [43, 47, 50, 55, 59, 67], [45, 52, 57, 60, 64], [41, 48, 53, 57, 60, 65], [60, 64, 67]]) {
    for (const lib of [null, library]) {
      const ns = assignChord(ps, lib); assert.ok(ns);
      assert.equal(new Set(ns.map(([s]) => s)).size, ns.length);
      assert.ok(ns.every(([s, f]) => s >= 0 && s <= 5 && Number.isInteger(f) && f >= 0 && f <= 22));
      const frets = ns.map(([, f]) => f).filter(f => f > 0);
      assert.ok(Math.max(...frets) - Math.min(...frets) <= 4);
      assert.deepEqual(ns.map(([s, f]) => openMidi[s] + f).sort((a, b) => a - b), [...ps].sort((a, b) => a - b));
    }
  }
  assert.equal(assignChord([40, 41, 42]), null);
});

test('stopping mic or tab capture transcribes, saves audio, and opens the lesson', async () => {
  for (const input of ['mic', 'cap']) {
    const h = createHarness();
    if (input === 'cap') { h.get('url').value = 'abcdefghijk'; await h.get('loadVid').onclick(); }
    await h.get(input).onclick();
    assert.equal(h.calls, 0);
    assert.equal(h.get('go').disabled, true);
    await h.stop(input === 'cap');
    assert.equal(h.calls, 1);
    const song = [...h.songs.values()][0];
    assert.equal(await h.blobs.get(song.id).text(), 'test recording');
    assert.equal(h.navigation[0], 'index.html#' + song.id);
    assert.equal(shiftScore(song.score, -song.audioOffset).notes[0].t, .2);
    assert.equal(h.get('go').disabled, false);
    const row = h.get('mine').children.at(-1);
    await row.children.at(-1).onclick();
    assert.equal(h.songs.size, 0); assert.equal(h.blobs.size, 0);
  }
});

test('uploads wait for the button; opt-out and failed saves never navigate', async () => {
  for (const saved of [true, false]) {
    const h = createHarness({ saved }); h.get('autoOpen').checked = !saved;
    h.get('file').onchange({ target: { files: [new File(['upload'], 'fixture.wav')] } });
    assert.equal(h.calls, 0);
    await h.get('go').onclick();
    assert.equal(h.calls, 1); assert.equal(h.navigation.length, 0);
    assert.equal(h.get('open3d').hidden, !saved);
    await h.get('go').onclick();
    assert.equal(h.calls, 2);
  }
});

test('unavailable IndexedDB is harmless and timeline shifting preserves the source', async () => {
  assert.equal(await saveSongAudio('test', new Blob(['a'])), false);
  assert.equal(await loadSongAudio('test'), null);
  assert.equal(await deleteSongAudio('test'), false);
  const score = { grid: [12, 12.5], notes: [{ t: 12.5, n: [[0, 0]] }], chords: [{ t0: 12, t1: 13, name: 'C', conf: .9 }] };
  const copy = structuredClone(score), shifted = shiftScore(score, -12);
  assert.equal(shifted.notes[0].t, .5); assert.equal(shifted.chords[0].t0, 0);
  assert.deepEqual(shiftScore(shifted, 12), score); assert.deepEqual(score, copy);
});

test('missing beats use tempo relative to the first onset, including picking and harmony', () => {
  const score = melody(), before = JSON.stringify(score), ev = prepare(score);
  assert.deepEqual(ev.map(e => e.q), [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]);
  assert.deepEqual(ev.plucks.pick.map(p => p.dir), [1, -1, 1, -1, 1, -1, 1, -1]);
  assert.ok(analyzeHarmony(ev).length > 0);
  assert.equal(JSON.stringify(score), before);
  score.notes.forEach((n, i) => { n.q = 4 + i / 2; });
  assert.deepEqual(prepare(score).map(e => e.q), score.notes.map(n => n.q));
});

test('invalid tempos and durations cannot produce infinite beats or durations', () => {
  for (const bpm of [0, -120, NaN, Infinity, undefined]) {
    const score = melody(); score.bpm = bpm; score.notes[0].d = Infinity;
    const ev = prepare(score);
    assert.equal(ev.bpm, 120);
    assert.ok(ev.every(e => Number.isFinite(e.q) && Number.isFinite(e.dur) && e.dur > 0));
  }
});

async function hand(side) {
  const b = fs.readFileSync(new URL(`../assets/generic-hand-${side}.glb`, import.meta.url));
  const g = await new Promise((resolve, reject) => new GLTFLoader().parse(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength), '', resolve, reject));
  return new HandRig(g.scene, side);
}
function finitePose(h) {
  assert.ok(h.place.elements.every(Number.isFinite));
  for (const ch of Object.values(h.chains)) {
    assert.ok(ch.theta.every((v, j) => Number.isFinite(v) && v >= ch.lo[j] && v <= ch.hi[j]));
    for (const p of ch.outP) assert.ok([p.x, p.y, p.z].every(Number.isFinite));
  }
}

test('empty songs and clearing a song produce finite resting poses in both modes', async () => {
  const left = await hand('left'), right = await hand('right');
  for (const mode of ['fingers', 'pick']) {
    const perf = new Performer(left, right);
    perf.setEvents(prepare(melody(), mode));
    perf.update(2.5, 1 / 60, true);
    for (const empty of [prepare({ bpm: 120, notes: [] }, mode), []]) {
      perf.setEvents(empty);
      for (const T of [0, 5, 0]) {
        const out = perf.update(T, 1 / 60, true);
        assert.equal(out.st.now, -1);
        assert.deepEqual(out.pressed, [null, null, null, null, null, null]);
        assert.equal(out.st.sounding.length, 0);
        finitePose(left); finitePose(right);
      }
    }
  }
});

test('forward/backward seeks and rate changes keep both rigs finite', async () => {
  const left = await hand('left'), right = await hand('right');
  for (const mode of ['fingers', 'pick']) {
    const perf = new Performer(left, right);
    perf.setEvents(prepare(melody(), mode));
    for (const start of [0, 3.5, 2.1, 8, 0]) {
      perf.update(start, 1 / 60, true);
      let T = start;
      for (const rate of [0.5, 0.75, 1, 1.5]) for (let i = 0; i < 12; i++) {
        T += rate / 60; perf.update(T, 1 / 60, false);
        finitePose(left); finitePose(right);
      }
    }
  }
});

// ---------- one timeline per song: strummed bars + single notes (js/songchords.js buildMixedTimeline) ----------
const mixedFixture = JSON.parse(fs.readFileSync(new URL('./fixtures/mixed-song.json', import.meta.url)));
const strumPats = JSON.parse(fs.readFileSync(new URL('../chords/strum_patterns.json', import.meta.url)));
const buildMixed = (fx = mixedFixture) => {
  const ev = prepare(fx.score, fx.mode);
  return { tab: ev, mix: buildMixedTimeline({ score: fx.score, events: ev, mode: fx.mode, track: { chords: fx.score.chords, strum: fx.score.strum }, lib: library, pats: strumPats }) };
};

test('mixed song: bars split into strum / notes / tab chords, chords only where they sound', () => {
  const { tab, mix } = buildMixed();
  assert.equal(mix.mixed, true);
  assert.deepEqual(mix.bars.map(b => b.kind), ['strum', 'strum', 'notes', 'notes', 'tabchord', 'tabchord']);
  assert.deepEqual(mix.spans.map(s => [s.symbol, s.source]), [['G', 'chords'], ['D', 'chords'], ['G', 'tab'], ['D', 'tab']]);
  const ev = mix.events, [, , n0, n1, t0] = mix.bars;
  // note bars: the tab's own events, plucked by the pick, never strummed and never labelled as a chord
  const notes = ev.filter(e => e.t >= n0.t0 - 0.01 && e.t < n1.t1 - 0.01);
  assert.equal(notes.length, 16);
  assert.ok(notes.every(e => !e.strummed && !e.symbol && e.notes.length === 1));
  assert.equal(ev.plucks.pick.length, 16);
  assert.ok(ev.plucks.pick.every(p => !p.ev.strummed));
  // tab chords keep the tab's exact voicing (G = 3 2 0 0 3 3, not the library's 3 2 0 0 0 3) for hand, strokes and diagram
  const g = mix.spans[2], tabG = tab.find(e => e.t >= t0.t0 - 0.01);
  assert.deepEqual(g.notes.map(n => [n.s, n.f]).sort(), tabG.notes.map(n => [n.s, n.f]).sort());
  for (const st of mix.strokes.filter(s => s.fromTab && s.t < t0.t1 - 0.03)) {
    assert.deepEqual(st.strings.map((sl, k) => [5 - sl, st.frets[k]]).sort(), tabG.notes.map(n => [n.s, n.f]).sort());
  }
  // strum directions follow the 8th grid (D on beats, U on "and"): D - D U - U D U
  assert.equal(mix.strokes.filter(s => s.fromTab && s.t < t0.t1 - 0.03).map(s => s.dir).join(''), 'DDUUDU');
  // the strum path is only active in strummed bars; the pick plays the tab in between
  assert.ok(ev.strum.at(1.0) && ev.strum.at(11.0));
  assert.equal(ev.strum.at(6.0), null);
  // lead times stay ordered for timeline.sample()'s binary search
  for (let i = 1; i < ev.length; i++) {
    assert.ok(ev[i].t - ev[i].lead >= ev[i - 1].t - ev[i - 1].lead - 1e-9, 'lead order at ' + i);
    assert.ok(ev[i].t - ev[i].leadH >= ev[i - 1].t - ev[i - 1].leadH - 1e-9, 'leadH order at ' + i);
  }
  // the synth plays exactly the note bars as notes and every strummed bar as strokes
  assert.equal(mix.noteOnly.length, 16);
  assert.equal(mix.strokes.length, 24);
});

test('songs without strummed bars keep their tab events unchanged', async () => {
  for (const id of ['choo-lo', 'rabba', 'falling-in-love-with-you']) {
    const score = JSON.parse(fs.readFileSync(new URL(`../data/score-${id}.json`, import.meta.url)));
    const track = JSON.parse(fs.readFileSync(new URL(`../data/chords-${id}.json`, import.meta.url)));
    const ev = prepare(score, id === 'rabba' ? 'pick' : 'fingers');
    const r = buildMixedTimeline({ score, events: ev, mode: 'fingers', track: { chords: track.chords, strum: track.strum || null }, lib: library, pats: strumPats });
    assert.equal(r.mixed, false, id);
    assert.ok(r.bars.every(b => b.kind === 'notes'), id);
  }
});

test('mixed song: both hands stay finite and the picking hand moves smoothly into and out of strummed bars', async () => {
  const left = await hand('left'), right = await hand('right');
  for (const mode of ['pick', 'fingers']) {
    const { mix } = buildMixed({ ...mixedFixture, mode });
    const perf = new Performer(left, right);
    perf.setEvents(mix.events);
    const fr = [];
    for (let T = 0, first = true; T < mix.end; T += 1 / 60, first = false) {
      perf.update(T, 1 / 60, first);
      finitePose(left); finitePose(right);
      fr.push({ T, p: (mode === 'pick' ? perf.pick.tip : right.tipG('index')).clone(), contact: !!(perf.strumState && perf.strumState.contact) });
    }
    // where strumming and picking meet the hand travels over several frames, it never teleports: no single frame covers
    // most of the distance moved in the 0.2 s around it
    const edges = mix.events.strum.windows.flat();
    let worst = 0;
    for (let i = 7; i < fr.length - 7; i++) {
      if (fr[i].contact || !edges.some(w => Math.abs(fr[i].T - w) < 0.12)) continue;
      const d = fr[i].p.distanceTo(fr[i - 1].p);
      let path = 0; for (let j = i - 6; j <= i + 6; j++) path += fr[j].p.distanceTo(fr[j - 1].p);
      if (d > 0.004) worst = Math.max(worst, d / path);
    }
    assert.ok(worst < 0.5, `${mode}: one frame covers ${(100 * worst).toFixed(0)} % of the move at a strum window edge`);
  }
});

test('index.html keeps only the learning controls; the rest sits in the More menu', () => {
  const html = fs.readFileSync(new URL('../index.html', import.meta.url), 'utf8');
  for (const gone of ['id="songview"', 'id="sv-notes"', 'id="sv-chords"', 'id="view-front"', 'id="view-back"', 'id="follow"', 'id="info"'])
    assert.ok(!html.includes(gone), gone + ' should be removed');
  for (const kept of ['id="song"', 'id="play"', 'id="scrub"', 'id="rate"', 'id="loop"', 'id="learn-tools"', 'id="more"'])
    assert.ok(html.includes(kept), kept);
  const menu = html.slice(html.indexOf('id="more-menu"'), html.indexOf('<div class="bar" id="bottom">'));
  for (const inMore of ['id="mode-chords"', 'id="loadfile"', 'class="legend"', 'id="status"']) assert.ok(menu.includes(inMore), inMore + ' in More');
});

// ---------- create: auto-trim leading / trailing silence ----------
/** synthetic take: `sil` s of faint noise, plucked tones (G B D G, 0.5 s each) for `tones` s, then `after` s of noise */
function take(sil, tones, after, noise = 2e-4) {
  const sr = 22050, y = new Float32Array(Math.round((sil + tones + after) * sr));
  let seed = 1; const rnd = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296 - 0.5; };
  for (let i = 0; i < y.length; i++) y[i] = noise * rnd();
  const f = [196, 247, 294, 392];
  for (let k = 0; k < tones / 0.5; k++) {
    const i0 = Math.round((sil + k * 0.5) * sr);
    for (let i = 0; i < 0.5 * sr; i++) { const t = i / sr; let v = 0; for (let h = 1; h <= 4; h++) v += Math.sin(2 * Math.PI * f[k % 4] * h * t) / h; y[i0 + i] += 0.3 * Math.exp(-3 * t) * v; }
  }
  return y;
}

test('trim finds the first sustained musical onset after 3 s of silence, and the silent end', () => {
  const t = findTrim(take(3, 4, 2));
  assert.equal(t.trimmed, true);
  assert.ok(t.start >= 2.7 && t.start <= 2.85, 'start ' + t.start);   // onset 3.0 s minus ~0.2 s
  assert.ok(t.end >= 7.0 && t.end <= 7.7, 'end ' + t.end);            // last note ends at 7.0 s, plus its tail
  for (const [y, what] of [[take(0, 4, 0), 'no silence'], [take(0.3, 4, 0.3), 'short silence'], [new Float32Array(22050 * 3), 'digital silence'], [take(5, 0, 0, 0.05), 'noise only']])
    assert.deepEqual([findTrim(y).trimmed, findTrim(y).start], [false, 0], what);
});

test('create trims a take with leading silence, keeps note times on the video clock, and Undo restores the full take', async () => {
  const h = createHarness({ audio: take(3, 4, 2), notes: [{ t: .2, e: .7, p: 64, a: 1 }] });
  h.get('autoOpen').checked = false;
  h.get('file').onchange({ target: { files: [new File(['upload'], 'take.wav')] } });
  h.get('offset').value = '10';                                     // audio starts at video time 10 s
  await h.get('go').onclick();
  const song = [...h.songs.values()][0], cut = +h.get('offset').value - 10;
  assert.ok(cut >= 2.7 && cut <= 2.85, 'cut ' + cut);
  assert.equal(h.get('trimMsg').hidden, false);
  assert.match(h.get('trimTxt').textContent, /^Trimmed 2\.\d s of silence at the start and \d\.\d s at the end\.$/);
  // the note 0.2 s into the trimmed audio sits at video time 10 + cut + 0.2; the stored take is the kept part
  assert.equal(song.audioOffset, +(10 + cut).toFixed(2));
  assert.equal(song.score.notes[0].t, +(10 + cut + .2).toFixed(3));
  assert.equal(shiftScore(song.score, -song.audioOffset).notes[0].t, .2);
  assert.match(await h.blobs.get(song.id).text(), /^trimmed 2\.\d\d-7\.\d\d$/);
  assert.ok(song.audioDuration < 5);
  // Undo: back to the whole take, offset restored, the tab rewritten from it
  await h.get('trimUndo').onclick();
  assert.equal(h.get('trimMsg').hidden, true);
  assert.equal(h.get('offset').value, '10.00');
  assert.equal(h.songs.size, 1);
  const again = [...h.songs.values()][0];
  assert.notEqual(again.id, song.id);
  assert.equal(again.audioOffset, 10);
  assert.equal(await h.blobs.get(again.id).text(), 'upload');
  assert.equal(h.blobs.has(song.id), false);
});

// ---------- create: sound + fretting-hand video fusion (js/handvision.js) ----------
test('hand video picks the string/fret nearest the fingertips; pitch is kept; no hand = plain Viterbi', async () => {
  const { visionCosts, findH, applyH, homographyFromCalibration, fretX, stringY } = await import('../js/handvision.js');
  // D#4 (63) three times: audio-only Viterbi plays it low (B string 4); fingertips at fret ~8 on the G string ask for (2, 8)
  const raw = [0, 1, 2].map(i => ({ t: .5 + i, e: .9 + i, p: 63, a: 1 }));
  const plain = toScore(raw);
  assert.ok(!plain.source && plain.notes.every(n => n.conf === undefined));
  const track = { frames: [] };
  for (let t = 0; t < 3; t += 1 / 12) track.frames.push({ t, tips: t < 2 ? [[2.1, 7.6], [1.2, 9.4]] : null });
  const fused = toScore(raw, 0, .75, null, { vision: visionCosts(track) });
  assert.equal(fused.source, 'audio+video');
  assert.deepEqual(plain.notes.map(n => n.n), [[[1, 4]], [[1, 4]], [[1, 4]]]);
  assert.deepEqual(fused.notes.slice(0, 2).map(n => n.n), [[[2, 8]], [[2, 8]]]);
  assert.ok(fused.notes[0].conf > .5 && fused.notes.every(n => n.conf >= 0 && n.conf <= 1));
  assert.ok(fused.notes.every((n, i) => openMidi[n.n[0][0]] + n.n[0][1] === 63), 'pitch unchanged');
  // geometry: calibration homography maps the clicked corners back
  const pts = [[100, 50], [110, 150], [400, 170], [390, 60]], H = homographyFromCalibration(pts, [0, 12]);
  const [x, y] = applyH(H, fretX(12), stringY(5, fretX(12)));
  assert.ok(Math.hypot(x - 400, y - 170) < 1e-6);
  const src = [[0, 0], [1, 0], [1, 1], [0, 1], [.5, .5], [.2, .7]], dst = src.map(([a, b]) => applyH(H, a, b));
  dst.push([999, 999]); src.push([.3, .3]);
  assert.equal(findH(src, dst, 1).inl.filter(Boolean).length, 6);
});

test('capture state machine: one stop wins, ended only after playing, end time, fit rect', () => {
  const m = new CaptureMachine({ start: 5, end: 20 });
  assert.equal(m.player(0), false); assert.equal(m.tick(30), false);          // idle: nothing stops
  assert.ok(m.arm()); assert.ok(m.begin(1000));
  assert.equal(m.player(0), false, 'stale ended before playing');
  assert.equal(m.tick(19.9), false); assert.equal(m.elapsed(3500), 2.5);
  assert.equal(m.tick(20), true); assert.equal(m.reason, 'end');
  assert.equal(m.stop('esc'), false); assert.equal(m.reason, 'end');
  const e = new CaptureMachine({ end: NaN }); e.arm(); e.begin(0);
  assert.equal(e.end, null); assert.equal(e.tick(1e6), false);
  assert.equal(e.player(1), false); assert.equal(e.player(0), true); assert.equal(e.reason, 'ended');
  assert.equal(new CaptureMachine({ start: 10, end: 4 }).end, null);
  const a = new CaptureMachine(); a.arm(); assert.ok(a.stop('shared')); assert.equal(a.begin(0), false);
  assert.deepEqual(fitRect(1600, 900, 16 / 9), { left: 0, top: 0, width: 1600, height: 900 });
  const p = fitRect(1600, 900, 9 / 16); assert.equal(p.height, 900); assert.equal(p.left, (1600 - 506.25) / 2);
  const t = fitRect(1000, 900, 16 / 9); assert.equal(t.width, 1000); assert.equal(t.top, (900 - 562.5) / 2);
  assert.equal(videoAspect('https://youtube.com/shorts/abcdefghijk'), 9 / 16); assert.equal(videoAspect('https://youtu.be/abcdefghijk'), 16 / 9);
  assert.equal(clockText(75.9), '1:15');
});

test('tab capture fills the window, stops itself (ended / Esc / stop time) and goes straight on to the tab', async () => {
  for (const how of ['ended', 'esc', 'end', 'shared']) {
    const h = createHarness({ videoTime: 12 });
    h.get('url').value = 'https://www.youtube.com/shorts/abcdefghijk'; await h.get('loadVid').onclick();
    h.get('capFrom').value = '3'; h.get('capTo').value = how === 'end' ? '10' : '';
    await h.get('cap').onclick();
    const html = h.document.documentElement, box = h.yt.panel.querySelector('.ytbody');
    assert.ok(html.classList.contains('yt-theatre'), how); assert.ok(h.yt.panel.classList.contains('theatre'));
    assert.equal(box.style.height, '900px'); assert.equal(box.style.width, '506.25px');   // 9:16 pillarbox in 1600x900
    assert.equal(h.document.body.children.at(-1).className, 'capbar');
    assert.deepEqual(h.yt.plays, [[3, 1]]);
    if (how === 'ended') { h.yt.onState(1); h.yt.onState(0); }
    else if (how === 'esc') h.key('Escape');
    else if (how === 'end') await new Promise(r => setTimeout(r, 450));             // 12 s >= stop at 10 s
    else h.tracks[1].onended();
    assert.equal(h.recorder.state, 'inactive', how);
    await h.recorder.done;
    assert.ok(!html.classList.contains('yt-theatre')); assert.ok(!h.yt.panel.classList.contains('theatre'));
    assert.equal(box.style.height, ''); assert.ok(h.document.body.children.at(-1).removed);
    assert.equal(h.calls, 1, how);
    const song = [...h.songs.values()][0];
    assert.equal(song.audioOffset, 3);                                             // audio t = 0 is video t = 3
    assert.equal(h.navigation[0], 'index.html#' + song.id);
    h.key('Escape'); h.yt.onState(0); assert.equal(h.calls, 1);                    // late events after the take are harmless
  }
});
