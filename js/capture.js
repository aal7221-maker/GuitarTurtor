// Hands-free tab capture for create.html. While the tab records, the YouTube player fills the window ("theatre", and the
// screen too when the browser allows fullscreen), a small bar shows the elapsed time, and the recording stops by itself:
// when the video ends, at a chosen end time, on Esc / Stop, or when the browser's "Stop sharing" ends the stream.
// No DOM globals at import time (node tests import it); the page passes document/window in.

/** largest rect of `aspect` (w/h) centred in W x H (letterbox / pillarbox) */
export function fitRect(W, H, aspect) {
  const w = Math.min(W, H * aspect), h = w / aspect;
  return { left: (W - w) / 2, top: (H - h) / 2, width: w, height: h };
}
/** Shorts are portrait; other links are assumed 16:9 (the player letterboxes anything else inside its rect). */
export const videoAspect = link => /\/shorts\//.test(link || '') ? 9 / 16 : 16 / 9;
export const clockText = s => { s = Math.max(0, Math.floor(s)); return Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0'); };

// idle -> armed (stream granted) -> recording (recorder started) -> stopped. Every stop source reports here; only the
// first one wins (returns true), so the page stops the recorder exactly once. reason: ended | end | esc | user | fullscreen | shared
export class CaptureMachine {
  constructor({ start = 0, end = null } = {}) {
    this.start = Math.max(0, +start || 0);
    this.end = Number.isFinite(end) && end > this.start ? end : null;
    this.state = 'idle'; this.reason = null; this.played = false; this.t0 = 0;
  }
  arm() { if (this.state === 'idle') this.state = 'armed'; return this.state === 'armed'; }
  begin(now) { if (this.state !== 'armed') return false; this.state = 'recording'; this.t0 = now; return true; }
  /** YouTube player state: 0 (ended) counts only after this take has played (a stale "ended" from before the seek does not) */
  player(s) { if (this.state !== 'recording') return false; if (s === 1) this.played = true; return s === 0 && this.played && this.stop('ended'); }
  /** video time poll: stop at the chosen end */
  tick(videoTime) { return this.state === 'recording' && this.end != null && videoTime >= this.end && this.stop('end'); }
  stop(reason = 'user') {
    if (this.state !== 'recording' && this.state !== 'armed') return false;
    this.state = 'stopped'; this.reason = reason; return true;
  }
  elapsed(now) { return this.state === 'recording' ? Math.max(0, (now - this.t0) / 1000) : 0; }
}

const CSS = `
html.yt-theatre, html.yt-theatre body { overflow: hidden; }
html.yt-theatre main { visibility: hidden; }
.ytpanel.theatre { inset: 0; width: auto; border: 0; border-radius: 0; box-shadow: none; background: #000; z-index: 1000; }
.ytpanel.theatre .ythead { display: none; }
.ytpanel.theatre .ytbody { position: absolute; aspect-ratio: auto; max-height: none; }
.ytpanel.theatre iframe { pointer-events: none; }
.capbar { position: fixed; z-index: 1001; top: 10px; left: 10px; display: flex; gap: 10px; align-items: center; padding: 5px 5px 5px 12px;
  border-radius: 999px; background: rgba(0,0,0,.7); color: #fff; font: 600 13px/1.2 system-ui, sans-serif; }
.capbar i { width: 9px; height: 9px; border-radius: 50%; background: #ef4444; animation: capblink 1s steps(2) infinite; }
.capbar button { all: unset; cursor: pointer; padding: 4px 10px; border-radius: 999px; background: #ef4444; color: #fff; }
.capbar button:focus-visible { outline: 2px solid #3d8bfd; outline-offset: 2px; }
@keyframes capblink { 50% { opacity: .25; } }
@media (prefers-reduced-motion: reduce) { .capbar i { animation: none; } }`;

// The player panel (js/yt.js .ytpanel) fills the window. onExit(reason) = the person asked to stop (Stop button, or
// leaving fullscreen with Esc, which the browser consumes before the page sees it).
export class Theatre {
  constructor(panel, { aspect = 16 / 9, doc = globalThis.document, win = globalThis.window, onExit = null } = {}) {
    Object.assign(this, { panel, aspect, doc, win, onExit, active: false, fs: false, rect: null, bar: null });
  }
  enter() {
    const { doc, win, panel } = this;
    if (this.active || !panel) return;
    if (doc.head && !doc.querySelector?.('#capture-css')) {
      const st = doc.createElement('style'); st.id = 'capture-css'; st.textContent = CSS; doc.head.appendChild(st);
    }
    this.active = true;
    this.box = panel.querySelector('.ytbody');
    panel.classList.add('theatre'); doc.documentElement.classList.add('yt-theatre');
    this.layout = () => {
      const r = this.rect = fitRect(win.innerWidth, win.innerHeight, this.aspect);
      if (this.box) Object.assign(this.box.style, { left: r.left + 'px', top: r.top + 'px', width: r.width + 'px', height: r.height + 'px' });
    };
    this.layout();
    win.addEventListener?.('resize', this.layout);
    // overlay: elapsed time + Stop (focused so Esc / Enter reach the page, not the YouTube iframe)
    const bar = this.bar = doc.createElement('div');
    bar.className = 'capbar'; bar.setAttribute?.('role', 'status');
    bar.innerHTML = '<i aria-hidden="true"></i><span>Recording… press Esc to stop · 0:00</span><button type="button">Stop</button>';
    doc.body.appendChild(bar);
    this.text = bar.querySelector('span');
    const stopBtn = bar.querySelector('button');
    stopBtn.onclick = () => this.onExit && this.onExit('user');
    try { doc.activeElement?.blur?.(); stopBtn.focus?.({ preventScroll: true }); } catch { /* focus is best effort */ }
    this.onFs = () => {
      if (doc.fullscreenElement) this.fs = true;
      else if (this.fs) { this.fs = false; if (this.active && this.onExit) this.onExit('fullscreen'); }
    };
    doc.addEventListener?.('fullscreenchange', this.onFs);
    // fullscreen may need a fresh user gesture (the share picker can use it up): then the full-window layout stays
    try {
      const p = doc.documentElement.requestFullscreen?.({ navigationUI: 'hide' });
      if (p && p.then) p.then(() => { if (!this.active && doc.fullscreenElement) doc.exitFullscreen().catch(() => {}); }, () => {});
    } catch { /* not allowed: keep the CSS theatre */ }
  }
  tick(seconds) { if (this.text) this.text.textContent = 'Recording… press Esc to stop · ' + clockText(seconds); }
  exit() {
    const { doc, win, panel } = this;
    if (!this.active) return;
    this.active = false;
    win.removeEventListener?.('resize', this.layout);
    doc.removeEventListener?.('fullscreenchange', this.onFs);
    if (doc.fullscreenElement && doc.exitFullscreen) doc.exitFullscreen().catch(() => {});
    this.fs = false;
    panel.classList.remove('theatre'); doc.documentElement.classList.remove('yt-theatre');
    if (this.box) Object.assign(this.box.style, { left: '', top: '', width: '', height: '' });
    if (this.bar) { this.bar.remove?.(); this.bar = this.text = null; }
  }
}
