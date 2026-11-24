// Learning layer for index.html: the scrolling tab strip, lesson markers on the neck, the ONE current note / chord display
// (HUD over the 3D view: the note, or the chord name + diagram while a chord sounds), the Hands on/off toggle, the
// "Lesson cam" preset and a clean screen-recording mode (key H). main.js only creates it and calls setEvents() / update().
import * as THREE from 'three';
import { TabStrip } from './tabstrip.js';
import { LessonMarkers } from './markers.js';
import { noteName, midiOf, STRING_LABELS } from './theory.js';
import { fretX } from './guitar.js';
import { diagramSVG } from './chords-mode.js';

const FINGER_NAMES = ['index', 'middle', 'ring', 'pinky'];
const FINGER_CSS = ['#4cc9f0', '#80ed99', '#ffd166', '#ef476f'];
const HANDS_KEY = 'guitar3d.hands';
const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : v; } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* private mode */ } },
};

const CSS = `
:root { --strip-h: 150px; --bottom-h: 0px; --top-h: 0px; }
#stage { bottom: calc(var(--bottom-h) + var(--strip-h)) !important; }
.learn { position: fixed; left: 0; right: 0; bottom: var(--bottom-h); height: var(--strip-h); z-index: 2;
  background: #0d0f14; border-top: 1px solid rgba(255,255,255,0.12); display: flex; flex-direction: column; color: #e9ecf1; }
.learn-tools { display: flex; gap: 6px; flex: none; }
.learn-tools button { min-height: 36px; }
.learn-tools button[aria-pressed="true"] { border-color: var(--accent, #ffb347); }
.learn canvas { flex: 1 1 auto; width: 100%; min-height: 0; display: block; cursor: pointer; touch-action: pan-y; }
.learn-hud { position: fixed; left: 16px; top: calc(var(--top-h) + 10px); z-index: 1; pointer-events: none; display: flex; gap: 10px; align-items: flex-start;
  text-shadow: 0 1px 3px rgba(0,0,0,0.9), 0 0 12px rgba(0,0,0,0.6); max-width: calc(100vw - 32px); }
.learn-hud .big { font: 800 34px/1.05 system-ui, -apple-system, "Segoe UI", sans-serif; color: #fff; letter-spacing: 0.01em; }
.learn-hud .big.chord { color: #ffb347; }
.learn-hud .small { font: 600 13px/1.35 system-ui, sans-serif; color: #d5dbe5; margin-top: 2px; }
.learn-hud .tech { display: inline-block; margin-top: 4px; font: 700 12px/1 system-ui, sans-serif; color: #14161b; background: #ffb347;
  border-radius: 5px; padding: 3px 6px; text-shadow: none; }
.learn-hud .tech:empty, .learn-hud .dia:empty { display: none; }
.learn-hud .dia { background: rgba(13,15,20,0.78); border: 1px solid rgba(255,255,255,0.14); border-radius: 10px; padding: 2px 4px; color: #cfd5e0; text-shadow: none; }
.learn-hud .dia svg { display: block; width: 92px; height: 116px; }
.learn-toast { position: fixed; left: 50%; top: 14px; transform: translateX(-50%); z-index: 5; background: rgba(13,15,20,0.9); color: #e9ecf1;
  font: 600 13px system-ui, sans-serif; padding: 8px 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.15); transition: opacity .4s; pointer-events: none; }
body.rec #top, body.rec #bottom, body.rec #loading, body.rec .ytpanel { display: none !important; }
body.rec .learn-hud { top: 14px; }
@media (max-width: 600px) {
  :root { --strip-h: 138px; }
  .learn-hud .big { font-size: 24px; }
  .learn-hud .small { font-size: 12px; }
  .learn-hud .dia svg { width: 70px; height: 88px; }
}
`;

/**
 * ctx: { guitar, camera, controls, now(), seek(t), hands() -> [left, right], pickMesh, markers (old sphere pool), resize(),
 *        setFollow(on) (camera follows the fretting hand; off while the lesson cam is on) }
 */
export function initLearn(ctx) {
  const st = document.createElement('style');
  st.id = 'learn-css'; st.textContent = CSS;
  document.head.appendChild(st);

  const wrap = document.createElement('div');
  wrap.className = 'learn';
  wrap.innerHTML = '<canvas id="learn-strip" aria-label="Tab: click a note to jump to it, drag to scrub"></canvas>';
  document.body.appendChild(wrap);
  // Hands + Lesson cam live in the bottom bar (index.html #learn-tools), else in a small row above the strip
  const tools = document.getElementById('learn-tools') || (() => { const d = document.createElement('div'); d.className = 'learn-tools'; d.style.padding = '6px 12px 0'; wrap.prepend(d); return d; })();
  tools.classList.add('learn-tools');
  tools.insertAdjacentHTML('beforeend', `
    <button id="learn-hands" aria-pressed="true" title="Show or hide the hand models (markers keep working)">Hands</button>
    <button id="learn-lesson" aria-pressed="false" title="Fixed camera framing the frets this song uses (off: the camera follows the fretting hand)">Lesson cam</button>`);
  const hud = document.createElement('div');
  hud.className = 'learn-hud';
  hud.innerHTML = '<div class="txt"><div class="big"></div><div class="small"></div><span class="tech"></span></div><div class="dia"></div>';
  document.body.appendChild(hud);
  const $ = id => document.getElementById(id);
  const ui = { hands: $('learn-hands'), lesson: $('learn-lesson'), rec: $('more-rec'),
    big: hud.querySelector('.big'), small: hud.querySelector('.small'), tech: hud.querySelector('.tech'), dia: hud.querySelector('.dia') };

  const strip = new TabStrip($('learn-strip'), { seek: t => ctx.seek(t) });
  const marks = new LessonMarkers(ctx.guitar);
  window.__learnStrip = strip;   // for headless checks
  let events = [];

  // ---- layout: the 3D stage ends above the strip, which sits above the bottom bar ----
  const root = document.documentElement;
  const measure = () => {
    const b = $('bottom'), t = $('top');
    const bh = b && getComputedStyle(b).display !== 'none' ? b.getBoundingClientRect().height : 0;
    const th = t && getComputedStyle(t).display !== 'none' ? t.getBoundingClientRect().height : 0;
    root.style.setProperty('--bottom-h', bh + 'px');
    root.style.setProperty('--top-h', th + 'px');
    const sh = wrap.getBoundingClientRect().height;
    root.style.setProperty('--ytpanel-bottom', Math.round(bh + sh + 10) + 'px');
  };
  const ro = new ResizeObserver(() => { measure(); strip.layout(); ctx.resize && ctx.resize(); });
  for (const id of ['bottom', 'top', 'stage', 'learn-strip']) { const el = $(id); if (el) ro.observe(el); }
  measure();

  // ---- hands on/off ----
  let handsOn = store.get(HANDS_KEY, '1') !== '0';
  const setHands = on => {
    handsOn = on;
    ui.hands.setAttribute('aria-pressed', String(on));
    ui.hands.textContent = on ? 'Hands: on' : 'Hands: off';
    store.set(HANDS_KEY, on ? '1' : '0');
  };
  setHands(handsOn);
  ui.hands.addEventListener('click', () => { setHands(!handsOn); ui.hands.blur(); });

  // ---- lesson cam: fixed camera framing the song's fret range (toggle; off = follow the fretting hand) ----
  let lessonOn = false;
  const setLesson = on => {
    lessonOn = on; ui.lesson.setAttribute('aria-pressed', String(on));
    if (ctx.setFollow) ctx.setFollow(!on);
  };
  function lessonCam() {
    const fr = [];
    for (const e of events) for (const n of e.notes) if (n.f > 0) { fr.push(n.f); if (e.bend || e.hb) fr.push(n.f); }
    const lo = fr.length ? Math.min(...fr) : 1, hi = fr.length ? Math.max(...fr) : 5;
    const xa = fretX(Math.max(0, lo - 1.7)), xb = fretX(Math.min(22.6, hi + 0.9));
    const g = ctx.guitar, cam = ctx.camera;
    g.updateMatrixWorld();
    const target = g.localToWorld(new THREE.Vector3((xa + xb) / 2 + 0.01, 0, -0.022));   // a little toward the hand side + body
    const span = Math.max(0.14, (xb - xa) * 1.32);
    const vf = cam.fov * Math.PI / 180, hf = 2 * Math.atan(Math.tan(vf / 2) * cam.aspect);
    const dist = Math.max((span / 2) / Math.tan(hf / 2), 0.1 / Math.tan(vf / 2));
    const normal = new THREE.Vector3(0, 1, 0).transformDirection(g.matrixWorld);   // out of the fingerboard
    const bass = new THREE.Vector3(0, 0, 1).transformDirection(g.matrixWorld);     // toward the low E (top of the view)
    const dir = normal.multiplyScalar(1).addScaledVector(bass, 0.22).normalize();
    setLesson(true);
    ctx.controls.target.copy(target);
    cam.position.copy(target).addScaledVector(dir, dist);
    ctx.controls.update();
  }
  ui.lesson.addEventListener('click', () => { if (lessonOn) setLesson(false); else lessonCam(); ui.lesson.blur(); });

  // ---- clean recording mode (H, or More > Hide controls) ----
  let toastT = null;
  const toast = msg => {
    let el = document.querySelector('.learn-toast');
    if (!el) { el = document.createElement('div'); el.className = 'learn-toast'; document.body.appendChild(el); }
    el.textContent = msg; el.style.opacity = '1';
    clearTimeout(toastT); toastT = setTimeout(() => { el.style.opacity = '0'; }, 1800);
  };
  const setRec = on => {
    document.body.classList.toggle('rec', on);
    measure();
    if (on) toast('Recording view: press H (or Esc) to show the controls');
  };
  if (ui.rec) ui.rec.addEventListener('click', () => setRec(true));
  window.addEventListener('keydown', e => {
    const tag = (e.target && e.target.tagName) || '';
    if (/INPUT|SELECT|TEXTAREA/.test(tag) || e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key === 'h' || e.key === 'H') { e.preventDefault(); setRec(!document.body.classList.contains('rec')); }
    else if (e.key === 'Escape' && document.body.classList.contains('rec')) setRec(false);
  });

  // ---- the current note / chord (only rebuilt when it changes) ----
  // A chord name + diagram only while a chord sounds (strummed bars: events.spans from js/songchords.js); notes otherwise.
  let shownKey = '';
  const spanAt = T => {
    const sp = events.spans || [];
    for (let i = sp.length - 1; i >= 0; i--) if (sp[i].t0 <= T + 0.005) return T < sp[i].t1 + 0.3 ? i : -1;
    return -1;
  };
  function showNote(i) {
    const e = events[i];
    const notes = [...e.notes].sort((a, b) => b.s - a.s);
    const tech = [e.legato === 'ham' && 'hammer-on', e.legato === 'pull' && 'pull-off', e.legato === 'slide' && 'slide in', e.slide && 'slide',
      e.bend && 'bend', e.hb && 'held bend', e.rel && 'release', e.vib && 'vibrato', e.pm && 'palm mute'].filter(Boolean);
    ui.big.className = 'big';
    if (notes.length === 1) {
      const n = notes[0];
      ui.big.textContent = noteName(midiOf(n.s, n.f));
      ui.small.textContent = `${STRING_LABELS[n.s]} string · fret ${n.f}${n.finger ? ' · ' + FINGER_NAMES[n.finger - 1] + ' finger' : ' · open'}`;
    } else {
      ui.big.textContent = notes.map(n => noteName(midiOf(n.s, n.f))).join(' + ');
      ui.small.innerHTML = notes.map(n => `<b style="color:${n.finger ? FINGER_CSS[n.finger - 1] : '#f4f6fa'}">${STRING_LABELS[n.s]}${n.f}</b>`).join(' + ');
    }
    ui.tech.textContent = tech.join(' · ');
    ui.dia.innerHTML = '';
  }
  function showChord(k, e) {
    const sp = events.spans[k];
    const voicing = [5, 4, 3, 2, 1, 0].map(s => { const n = sp.notes.find(x => x.s === s); return n ? String(n.f) : 'x'; }).join(' ');
    ui.big.className = 'big chord';
    ui.big.textContent = sp.symbol || sp.notes.map(n => noteName(midiOf(n.s, n.f), false)).join(' ');
    ui.small.textContent = `strum${e && e.dir ? (e.dir === 'U' ? ' ↑' : ' ↓') : ''} · ${voicing}${sp.source === 'tab' ? ' (as in the tab)' : ''}`;
    ui.tech.textContent = '';
    ui.dia.innerHTML = diagramSVG({ notes: sp.notes, muted: sp.muted || [], barre: sp.barre || null });
  }

  return {
    strip, markers: marks,
    get handsOn() { return handsOn; }, setHands, lessonCam, setLesson, setRec,
    get lessonOn() { return lessonOn; },
    setEvents(ev) {
      events = ev || [];
      strip.setEvents(events);
      marks.setEvents(events);
      shownKey = '';
      if (lessonOn) requestAnimationFrame(lessonCam);
    },
    update(T) {
      // never break main.js's render loop
      try { this._update(T); } catch (err) { if (!this._warned) { this._warned = true; console.error('learn layer: ' + (err && err.stack || err)); } }
    },
    _update(T) {
      // our markers replace main.js's sphere markers
      if (ctx.markers) for (const m of ctx.markers) m.visible = false;
      const [L, R] = ctx.hands ? ctx.hands() : [];
      for (const h of [L, R]) if (h) { h.root.visible = handsOn; h.arm.visible = handsOn; }
      if (!handsOn && ctx.pickMesh) ctx.pickMesh.visible = false;
      strip.update(T);
      const { cur } = marks.update(T);
      // the display keeps the last note until the next one starts (so a student can pause and read it)
      const e = cur >= 0 ? events[cur] : null;
      const k = e && e.strummed ? spanAt(T) : -1;
      const key = k >= 0 ? 'c' + k + (e.dir || '') : e ? 'n' + cur : '';
      if (key !== shownKey) {
        shownKey = key;
        if (k >= 0) showChord(k, e);
        else if (e && !e.strummed) showNote(cur);
        else { ui.big.textContent = ''; ui.small.textContent = ''; ui.tech.textContent = ''; ui.dia.innerHTML = ''; }
      }
    },
  };
}
