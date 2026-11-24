// Exercise main.js's actual loading/clock functions without constructing a WebGL renderer.
import fs from 'node:fs';
import vm from 'node:vm';
import { normalizeScore, solveFingering } from '../../js/fingering.js';
import { prepareTimeline } from '../../js/timeline.js';
import { shiftScore } from '../../js/song-audio.js';

export function playbackHarness(songs, loadAudio) {
  const created = [], revoked = [], sources = [], panel = [], media = [];
  const audio = { src: '', currentTime: 0, playbackRate: 1, pause() {}, async play() { media.push('play'); } };
  const ui = { song: {}, scrub: {}, play: { setAttribute() {} }, rate: { value: '1' } };
  const ctx = vm.createContext({
    SONGS: songs, songKey: '', ui, audio, audioOK: false, internalT: 0, lastT: null, playing: false,
    events: [], duration: 1, perf: null, learn: { setEvents() {} }, ytActive: false,
    yt: { async load() { sources.push('video'); }, setRate() {}, seek() {}, time: () => 0, hide() {}, pause() {}, play() {} },
    curScore: null,
    document: { getElementById: () => null }, setStatus() {}, poke() {}, console,
    normalizeScore, solveFingering, prepareTimeline, shiftScore, loadSongAudio: loadAudio,
    renderSong: async () => { sources.push('synth'); return new Blob(['synth']); },
    fetch: async () => { sources.push('hosted'); return { ok: false }; },
    URL: { createObjectURL(blob) { created.push(blob); return 'blob:test-' + created.length; }, revokeObjectURL: u => revoked.push(u) },
  });
  const src = fs.readFileSync(new URL('../../js/main.js', import.meta.url), 'utf8');
  new vm.Script(src.slice(src.indexOf('let sourceReady ='), src.indexOf('// ---------- hands ----------'))).runInContext(ctx);
  return { ctx, audio, ui, sources, created, revoked, panel, media };
}
