import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { buildGuitar, layoutString, contactX, stringY, stringZ } from './guitar.js';
import { normalizeScore, solveFingering } from './fingering.js';
import { prepareTimeline } from './timeline.js';
import { HandRig } from './hand.js';
import { Performer } from './performer.js';
import { HAND_LEFT, HAND_RIGHT } from './hand-assets.js';
import { renderSong, renderMixed } from './synth.js';
import { initChords } from './chords-mode.js';   // Chord practice (More menu): progression, strum pattern, diagrams (js/chords-mode.js)
import { initLearn } from './learn.js';   // tab strip, lesson markers, the current note / chord display, hands toggle, lesson cam (js/learn.js)
import { loadSongAudio, shiftScore } from './song-audio.js';
import { buildMixedTimeline } from './songchords.js';   // one timeline per song: strummed bars + single-note bars

let SONGS = {};   // filled from data/songs.json (written by the tab-viewer build)
const FINGER_COLORS = [0x4cc9f0, 0x80ed99, 0xffd166, 0xef476f];

const $ = id => document.getElementById(id);
const ui = {
  status: $('status'), play: $('play'), scrub: $('scrub'), time: $('time'), song: $('song'),
  rate: $('rate'), loop: $('loop'), loading: $('loading'),
};

// ---------- scene ----------
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
$('stage').appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x14161b);
const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 20);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.minDistance = 0.08;
controls.maxDistance = 2.5;

scene.add(new THREE.HemisphereLight(0xfff6ea, 0x30343c, 1.1));
const sun = new THREE.DirectionalLight(0xffffff, 2.2);
sun.position.set(0.3, 0.8, 1.2);
sun.castShadow = true;
sun.shadow.mapSize.set(1024, 1024);
Object.assign(sun.shadow.camera, { left: -0.6, right: 0.6, top: 0.6, bottom: -0.6, near: 0.1, far: 4 });
sun.shadow.bias = -0.0004;
scene.add(sun);
const fill = new THREE.DirectionalLight(0xbfd4ff, 0.6);
fill.position.set(-0.8, -0.2, 0.6);
scene.add(fill);

// guitar space -> world: x_G -> -X, y_G -> +Z, z_G -> +Y, then tilt the neck up a little
const guitar = buildGuitar();
guitar.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(
  new THREE.Vector3(-1, 0, 0), new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 1, 0)));
guitar.quaternion.premultiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), 0.12));
guitar.position.set(0.25, 0, 0);
scene.add(guitar);
sun.target = guitar;
const strings = guitar.userData.strings;

// note markers (pool)
const markerGeo = new THREE.SphereGeometry(0.0032, 16, 12);
const markers = [];
for (let i = 0; i < 24; i++) {
  const m = new THREE.Mesh(markerGeo, new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.9, depthWrite: false }));
  m.visible = false;
  m.renderOrder = 2;
  guitar.add(m);
  markers.push(m);
}
// guitar pick (triangle)
const pickShape = new THREE.Shape();
pickShape.moveTo(0, -0.011); pickShape.quadraticCurveTo(0.011, 0.004, 0.006, 0.009);
pickShape.quadraticCurveTo(0, 0.012, -0.006, 0.009); pickShape.quadraticCurveTo(-0.011, 0.004, 0, -0.011);
const pickMesh = new THREE.Mesh(new THREE.ExtrudeGeometry(pickShape, { depth: 0.0008, bevelEnabled: false }),
  new THREE.MeshStandardMaterial({ color: 0xff6a3d, roughness: 0.4 }));
pickMesh.visible = false;
guitar.add(pickMesh);
// the camera follows the fretting hand unless the Lesson cam (fixed framing) is on
let followOn = true, followTarget = null;
const learn = initLearn({ guitar, camera, controls, now: () => now(), seek: t => seek(t), hands: () => [left, right], pickMesh, markers, resize: () => resize(),
  setFollow: on => { followOn = on; followTarget = null; } });

const chords = initChords({
  now: () => now(), seek: t => seek(t), isPlaying: () => playing, setPlaying: p => setPlaying(p),
  load: cs => loadChordSong(cs), restore: () => loadSong(songKey), resize: () => resize(),
  leaveSong: () => {},
});

// ---------- state ----------
let events = [], duration = 1, songKey = 'choo-lo', curScore = null;
let lastT = null, internalT = 0, playing = false, audioOK = true;
let left = null, right = null, perf = null;
const audio = new Audio();
audio.preload = 'auto';
audio.addEventListener('error', () => { if (audio.src) { audioOK = false; setStatus('Audio failed to load: animating on internal clock'); } });
audio.addEventListener('loadedmetadata', () => {
  // the previous song's file can report its length while the next song is loading: ignore it then
  if (!sourceReady) return;
  // MediaRecorder WebM may report Infinity; keep the decoded duration in that case.
  if (Number.isFinite(audio.duration)) duration = Math.max(duration, audio.duration);
  ui.scrub.max = duration;
});
audio.addEventListener('ended', () => setPlaying(false));

// the original Short in a small floating player (only where js/yt.js is loaded, i.e. the GitHub Pages build)
let ytActive = false;
const yt = window.YTSync ? new window.YTSync({
  onState: s => {
    if (!ytActive) return;
    if (s === 1 && !playing) setUi(true);
    else if (s === 2 && playing) setUi(false);
    else if (s === 0) setUi(false);
  },
}) : null;

function setStatus(s) { if (ui.status) ui.status.textContent = s; poke(); }

// ---------- render on demand ----------
// While paused and untouched nothing on screen changes, so the scene and the tab strip are only redrawn
// while playing, and for a short settling window (camera damping, hand springs, follow-cam) after anything
// that can change the picture: input, camera moves, seeks, song loads and late-arriving textures.
let wake = 120;
function poke(frames = 60) { wake = Math.max(wake, frames); }
for (const type of ['pointerdown', 'pointermove', 'wheel', 'keydown', 'input', 'change', 'click', 'touchstart'])
  window.addEventListener(type, () => poke(), { passive: true, capture: true });
document.addEventListener('visibilitychange', () => poke());
{ const m = THREE.DefaultLoadingManager, prev = m.onLoad; m.onLoad = (...a) => { poke(); if (prev) prev(...a); }; m.onProgress = () => poke(); }
const setText = (el, s) => { if (el.textContent !== s) el.textContent = s; };
let sourceReady = false, pendingPlay = false, songLoad = 0;
// what is playing: 'recording' | 'file' | 'yt' | 'synth' (from the tab: notes, plus the strummed bars of a mixed song) |
// 'chordsynth' (Chord practice) | null
let audioSource = null, noteEvents = [];
function setUi(p) {
  playing = p;
  poke();
  ui.play.textContent = sourceReady ? (p ? 'Pause' : 'Play') : (p ? 'Starting…' : 'Loading…');
  ui.play.setAttribute('aria-pressed', String(p));
}
function setPlaying(p) {
  if (!sourceReady) { pendingPlay = p; setUi(p); return; }   // pressed while the sound is still loading: start once it's ready
  setUi(p);
  if (ytActive) { if (p) yt.play(yt.time(), parseFloat(ui.rate.value)); else yt.pause(); return; }
  if (audioOK) { if (p) audio.play().catch(() => { audioOK = false; }); else audio.pause(); }
}
function now() { return ytActive ? yt.time() : audioOK ? audio.currentTime : internalT; }
function seek(t) { if (ytActive) yt.seek(t); else if (audioOK) audio.currentTime = t; internalT = t; lastT = null; poke(); }

// Chord practice (More menu): events/strokes/audio come from js/chordsong.js + synth.renderStrokes (via js/chords-mode.js)
let chordLoad = 0;
async function loadChordSong(cs) {
  const my = ++chordLoad, wasPlaying = playing;
  setPlaying(false);
  ytActive = false; if (yt) { yt.pause(); yt.hide(); }
  sourceReady = false; pendingPlay = false; setUi(false);
  events = cs.events; duration = cs.total;
  if (perf) perf.setEvents(events);
  learn.setEvents(events);
  ui.scrub.max = duration;
  setStatus('Generating guitar sound for ' + cs.label + '…');
  let src = null;
  try { src = URL.createObjectURL(await cs.renderAudio()); } catch (e) { console.error(e); }
  if (my !== chordLoad) return;
  setAudio(src, 'chordsynth', cs.label);
  duration = cs.total; ui.scrub.max = duration;
  sourceIsReady();
  if (wasPlaying) setPlaying(true);
}

async function loadSong(key) {
  if (typeof chords !== 'undefined') { chordLoad++; chords.deactivate(); }   // the playback test harness loads main.js without chord mode
  const load = ++songLoad;
  const song = SONGS[key];
  songKey = key;
  ui.song.value = key;
  const tl = document.getElementById('tabslink'); if (tl) tl.href = tl.href.split('#')[0] + '#' + key;
  setPlaying(false);
  sourceReady = false; pendingPlay = false; setUi(false);
  setStatus('Loading ' + song.label + '…');
  let score = song.score || await (await fetch(song.data)).json();   // user songs (js/user-songs.js) carry their score
  const recording = song.user ? await loadSongAudio(key) : null;
  if (load !== songLoad) return;
  if (recording) score = shiftScore(score, -(song.audioOffset || 0));
  curScore = score;
  events = prepareTimeline(solveFingering(normalizeScore(score)), song.mode);
  noteEvents = events;
  // ONE timeline: strummed bars (chord song / the tab's stacked chords) + single-note bars (the tab); songs without strummed
  // bars keep the tab events unchanged (not in the playback test harness)
  let mix = null;
  if (typeof buildMix !== 'undefined') {
    mix = await buildMix(key, song, score, noteEvents);
    if (load !== songLoad) return;
    if (mix && mix.mixed) events = mix.events;
  }
  if (perf) perf.setEvents(events);
  learn.setEvents(events);
  duration = noteEvents.length ? noteEvents[noteEvents.length - 1].t + 2 : 1;
  if (mix && mix.mixed) duration = Math.max(duration, mix.end + 1.5);
  if (recording && Number.isFinite(song.audioDuration) && song.audioDuration > 0) duration = song.audioDuration;
  ui.scrub.max = duration;
  const nStrum = mix && mix.mixed ? mix.bars.filter(b => b.kind !== 'notes').length : 0;
  const summary = `${noteEvents.length} notes${nStrum ? ` · ${nStrum} strummed bar${nStrum > 1 ? 's' : ''}` : ''} · ${Math.round(score.bpm)} bpm · ${song.mode === 'pick' ? 'pick' : 'p-i-m-a'}`;
  // Saved audio first, then hosted audio, YouTube, and finally the guitar synth.
  let src = recording ? URL.createObjectURL(recording) : null, source = recording ? 'file' : 'recording';
  ytActive = false;
  if (!src && song.audio) try { const r = await fetch(song.audio, { method: 'HEAD' }); if (r.ok) src = song.audio; } catch (e) { /* not hosted */ }
  if (load !== songLoad) return;
  if (!src && yt && song.video) {
    setStatus('Loading the original video…');
    try {
      await yt.load(song.video, song.label);
      if (load !== songLoad) return;
      audio.pause(); audioOK = false; ytActive = true; audioSource = 'yt';
      yt.setRate(parseFloat(ui.rate.value)); seek(0);
      setStatus(`${summary} · sound: original video (YouTube)`);
      sourceIsReady();
      return;
    } catch (e) { console.warn(e); }
  }
  if (load !== songLoad) return;
  if (yt && !ytActive) yt.hide();
  if (!src) {
    setStatus('Generating guitar sound from the tab…');
    try {
      // mixed song: the single notes as plucked / picked, the strummed bars as strokes (exactly what the hands play)
      const blob = mix && mix.mixed ? await renderMixed(mix.noteOnly, mix.strokes, song.mode, duration + 1.5)
        : await renderSong(normalizeScore(score), song.mode, duration + 1.5);
      if (load !== songLoad) return;
      src = URL.createObjectURL(blob); source = 'synth';
    } catch (e) { console.error(e); }
  }
  if (load !== songLoad) return;
  setAudio(src, source, summary);
  sourceIsReady();
}
function sourceIsReady() {
  sourceReady = true;
  const go = pendingPlay; pendingPlay = false;
  setUi(false);
  if (go) setPlaying(true);
}

let lastSummary = '';
function setAudio(src, source, summary = lastSummary) {
  lastSummary = summary;
  if (audio.src.startsWith('blob:')) URL.revokeObjectURL(audio.src);
  audioOK = !!src; audioSource = src ? source : null;
  if (src) { audio.src = src; audio.playbackRate = parseFloat(ui.rate.value); }
  seek(0);
  const label = { recording: 'original recording', synth: 'synth guitar from the tab', chordsynth: 'synth guitar strumming the chords', file: 'your recording' }[source] || 'no audio';
  setStatus(`${summary} · sound: ${src ? label : 'no audio'}`);
}

// ---------- hands ----------
const loader = new GLTFLoader();
const b64buf = b => { const s = atob(b), u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return u.buffer; };
const loadHand = side => new Promise((res, rej) => loader.parse(b64buf(side === 'left' ? HAND_LEFT : HAND_RIGHT), '', g => res(new HandRig(g.scene, side)), rej));
Promise.all([loadHand('left'), loadHand('right')]).then(([l, r]) => {
  left = l; right = r;
  guitar.add(left.root, right.root, left.arm, right.arm);
  perf = new Performer(left, right);
  if (events.length) perf.setEvents(events);
  lastT = null;
  ui.loading.hidden = true;
  poke();
  // smooth the hands once the first frames are on screen (~180 ms each, one hand per task so the page
  // stays responsive); until then the original low-poly mesh is shown
  const idle = cb => (window.requestIdleCallback ? requestIdleCallback(cb, { timeout: 600 }) : setTimeout(cb, 120));
  requestAnimationFrame(() => idle(() => { left.refine(); poke(); idle(() => { right.refine(); poke(); }); }));
}).catch(err => { ui.loading.textContent = 'Hand model failed to load'; console.error(err); });

// ---------- one timeline per song: strummed bars + single notes ----------
// js/songchords.js buildMixedTimeline cuts the song into bars: bars where the tab stacks chords are strummed from the tab
// (exact voicing), bars with the recording's strum (score.strum / data/chords-<id>.json) or a chord flagged {strum: true}
// are strummed with the chord song (shapes from chords/chord_shapes.json), the rest are the tab's notes.

/** the song's chord track on the page clock: score.chords/strum, else data/chords-<id>.json (shifted by its offset) */
async function songChordTrack(key, song, score) {
  if (Array.isArray(score.chords) && score.chords.length) return { chords: score.chords, strum: score.strum || null };
  if (song.user || song.fixture) return null;
  try {
    const r = await fetch(`data/chords-${key}.json`);
    if (!r.ok) return null;
    const d = await r.json();
    if (!Array.isArray(d.chords) || !d.chords.length) return null;
    const strum = d.strum && d.strum.pattern && Array.isArray(d.strum.strokes) ? d.strum : null;
    const s = shiftScore({ notes: [], chords: d.chords, ...(strum ? { strum } : {}) }, d.offset || 0);
    return { chords: s.chords, strum: s.strum || null };
  } catch (e) { return null; }
}
async function buildMix(key, song, score, tabEvents) {
  try {
    const track = await songChordTrack(key, song, score);
    const needShapes = track && (track.strum || track.chords.some(c => c.strum));
    const data = needShapes ? await chords.ensureData() : null;
    return buildMixedTimeline({ score, events: tabEvents, mode: song.mode === 'pick' ? 'pick' : 'fingers', track, lib: data && data.lib, pats: data && data.pats });
  } catch (e) { console.warn('mixed timeline', e); return null; }
}

// ---------- A-B loop ----------
// Loop: 1st press marks A (now), 2nd press marks B and loops A-B, 3rd press clears.
const loopAB = { a: null, b: null };
function markLoop() {
  const T = now();
  if (loopAB.a === null) loopAB.a = T;
  else if (loopAB.b === null && T > loopAB.a + 0.3) { loopAB.b = T; seek(loopAB.a); }
  else { loopAB.a = loopAB.b = null; }
  const on = loopAB.b !== null;
  ui.loop.textContent = on ? `Loop ${fmt(loopAB.a)}–${fmt(loopAB.b)}` : loopAB.a !== null ? `Loop from ${fmt(loopAB.a)}… set end` : 'Loop';
  ui.loop.setAttribute('aria-pressed', String(on || loopAB.a !== null));
}
function clearLoop() { loopAB.a = loopAB.b = null; if (ui.loop) { ui.loop.textContent = 'Loop'; ui.loop.setAttribute('aria-pressed', 'false'); } }

// ---------- per-frame ----------
const camOffset = new THREE.Vector3(0.06, 0.1, 0.42);

function update(dt) {
  if (!audioOK && playing) internalT += dt * parseFloat(ui.rate.value);
  let T = now();
  if (loopAB.b !== null && playing && (T >= loopAB.b || T < loopAB.a - 0.5)) { seek(loopAB.a); T = loopAB.a; }
  const jumped = lastT === null || Math.abs(T - lastT) > 0.25;
  lastT = T;
  if (!perf) return;
  const { st, pressed, anchor } = perf.update(T, dt, jumped);

  // strings: pressed geometry from the actual fingertips, glow + vibration while sounding
  const glow = [0, 0, 0, 0, 0, 0];
  if (events.strum) events.strum.glow(T).forEach((v, s) => { glow[s] = v; });   // strummed: each string lights when the hand reaches it
  if (!events.strum || events.mixed) for (const e of st.sounding) if (!e.strummed) for (const n of e.notes) glow[n.s] = Math.max(glow[n.s], Math.exp(-(T - e.t) / 0.45));
  strings.forEach((str, s) => {
    str.mat.emissiveIntensity = 2.0 * glow[s];
    layoutString(str, pressed[s], glow[s] * 0.0005, T);
  });

  // pick
  pickMesh.visible = perf.pick.visible;
  if (pickMesh.visible) {
    const dir = perf.pick.tip.clone().sub(perf.pick.base).normalize();
    pickMesh.position.copy(perf.pick.tip).addScaledVector(dir, -0.009);
    pickMesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, -1, 0), dir);
    pickMesh.rotateY(Math.PI / 2);
  }

  // markers: sounding (solid, finger colour) and upcoming (ghost)
  let mi = 0;
  const place = (n, opacity, scale) => {
    if (mi >= markers.length) return;
    const m = markers[mi++];
    const x = n.f > 0 ? contactX(n.f) + 0.005 : -0.009;
    m.position.set(x, stringY(Math.max(x, 0)) + 0.003, stringZ(n.s, Math.max(x, 0)));
    m.material.color.setHex(n.finger ? FINGER_COLORS[n.finger - 1] : 0xffffff);
    m.material.opacity = opacity;
    m.scale.setScalar(scale);
    m.visible = true;
  };
  for (const e of st.sounding) for (const n of e.notes) place(n, 0.95, 1.1);
  for (const e of st.upcoming) for (const n of e.notes) place(n, 0.28 * (1 - (e.t - T) / 0.6) + 0.08, 0.75);
  for (; mi < markers.length; mi++) markers[mi].visible = false;

  // camera follows the fretting hand (keeps the user's orbit offset)
  if (followOn) {
    const w = guitar.localToWorld(anchor.clone().setY(0).setZ(0));
    if (!followTarget) followTarget = controls.target.clone();
    const prev = followTarget.clone();
    followTarget.lerp(w, jumped ? 1 : 1 - Math.exp(-dt * 3));
    const d = followTarget.clone().sub(prev);
    controls.target.add(d);
    camera.position.add(d);
  } else followTarget = null;

  if (!ui.scrub.matches(':active') && ui.scrub.value !== String(T)) ui.scrub.value = T;
  setText(ui.time, `${fmt(T)} / ${fmt(duration)}`);
}
const fmt = t => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, '0')}`;

function resize() {
  const el = $('stage');
  const w = el.clientWidth, h = el.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / Math.max(1, h);
  camera.fov = camera.aspect < 0.8 ? 55 : 38;
  camera.updateProjectionMatrix();
  poke();
}
window.addEventListener('resize', resize);

let prevFrame = performance.now();
function frame(t) {
  const dt = Math.min(0.1, Math.max(0, (t - prevFrame) / 1000));
  prevFrame = t;
  if (playing || wake > 0) {
    if (!playing) wake--;
    update(dt);
    learn.update(now());
    chords.update(now());
    controls.update();   // dispatches 'change' while damping, which pokes and keeps this going until it settles
    renderer.render(scene, camera);
  }
  requestAnimationFrame(frame);
}
controls.addEventListener('change', () => poke());

// ---------- UI ----------
ui.song.addEventListener('change', () => { clearLoop(); loadSong(ui.song.value); });
ui.play.addEventListener('click', () => setPlaying(!playing));
ui.scrub.addEventListener('input', () => seek(parseFloat(ui.scrub.value)));
ui.rate.addEventListener('change', () => { audio.playbackRate = parseFloat(ui.rate.value); if (ytActive) yt.setRate(parseFloat(ui.rate.value)); });
if (ui.loop) ui.loop.addEventListener('click', () => { markLoop(); ui.loop.blur(); });
$('loadfile').addEventListener('change', e => {
  const f = e.target.files && e.target.files[0];
  if (!f) return;
  setPlaying(false);
  if (ytActive) { yt.pause(); yt.hide(); ytActive = false; }
  setAudio(URL.createObjectURL(f), 'file');
  e.target.value = '';
  closeMore();
});
window.addEventListener('keydown', e => {
  if (e.code === 'Space' && e.target === document.body) { e.preventDefault(); setPlaying(!playing); }
});
// "More" menu: links, Chord practice, Load recording, Hide controls, details
const more = { btn: $('more'), menu: $('more-menu') };
function closeMore() { if (more.menu) { more.menu.hidden = true; more.btn.setAttribute('aria-expanded', 'false'); } }
if (more.btn) {
  more.btn.addEventListener('click', e => { e.stopPropagation(); more.menu.hidden = !more.menu.hidden; more.btn.setAttribute('aria-expanded', String(!more.menu.hidden)); });
  document.addEventListener('click', e => { if (!more.menu.hidden && !more.menu.contains(e.target)) closeMore(); });
  for (const b of more.menu.querySelectorAll('button')) b.addEventListener('click', () => setTimeout(closeMore, 0));
  window.addEventListener('keydown', e => { if (e.key === 'Escape') closeMore(); });
}

/** camera presets for scripted views (tests/e2e/record.mjs); the page itself only offers the Lesson cam */
function setView(v) {
  const offs = { front: [0.06, 0.1, 0.42], player: [0.03, 0.36, 0.1], close: [0.03, -0.1, 0.22], back: [0.05, -0.05, -0.32] };
  if (learn.lessonOn) learn.setLesson(false);
  if (v === 'pick') {
    followOn = false; followTarget = null;
    const tgt = guitar.localToWorld(new THREE.Vector3(0.55, 0, 0));
    controls.target.copy(tgt);
    camera.position.copy(tgt).add(new THREE.Vector3(0.02, -0.2, 0.24));
    return;
  }
  followOn = true;
  const tgt = followTarget ? followTarget.clone() : guitar.localToWorld(new THREE.Vector3(contactX(6), 0, 0));
  controls.target.copy(tgt);
  camera.position.copy(tgt).add(new THREE.Vector3(...(offs[v] || offs.front)));
}

resize();
const start = guitar.localToWorld(new THREE.Vector3(contactX(7), 0, 0));
controls.target.copy(start);
camera.position.copy(start).add(camOffset);
(async () => {
  try {
    const list = await (await fetch('data/songs.json')).json();
    // songs made in create.html and kept in this browser
    try { for (const s of (window.UserSongs ? window.UserSongs.list() : [])) if (s.score && s.score.notes && s.score.notes.length && !list.some(x => x.id === s.id)) list.push({ id: s.id, label: s.title + ' ★ (yours)', score: s.score, user: true, audioOffset: s.audioOffset || 0, audioDuration: s.audioDuration, video: s.video || null, mode: s.mode || 'fingers' }); } catch (e) { console.warn(e); }
    SONGS = Object.fromEntries(list.map(x => [x.id, x]));
    for (const x of list) ui.song.add(new Option(x.label, x.id));
    const want = location.hash.slice(1);
    await loadSong(SONGS[want] ? want : (SONGS['choo-lo'] ? 'choo-lo' : list[0].id));
    if (want === 'chords') chords.activate();
  } catch (e) { setStatus('Failed to load the song list: ' + e.message); console.error(e); }
})();
window.addEventListener('hashchange', () => { const k = location.hash.slice(1); if (SONGS[k] && k !== songKey) { clearLoop(); loadSong(k); } else if (k === 'chords') chords.activate(); });
requestAnimationFrame(frame);

/** add a song at runtime (headless checks: tools/fixtures/mixed-song.json) and load it */
async function addSong(s) {
  SONGS[s.id] = { ...s, fixture: true };
  if (![...ui.song.options].some(o => o.value === s.id)) ui.song.add(new Option(s.label, s.id));
  clearLoop();
  await loadSong(s.id);
}

// exposed for debugging / headless checks
window.__g3d = {
  get events() { return events; }, get hand() { return left; }, get right() { return right; }, get perf() { return perf; },
  get song() { return songKey; }, get score() { return curScore; }, get sourceReady() { return sourceReady; }, get loop() { return { ...loopAB }; },
  chords, learn, loadSong, addSong, setView, seek, setPlaying, now: () => now(), camera, controls,
};
