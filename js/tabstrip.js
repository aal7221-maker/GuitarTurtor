// Scrolling tab strip (the tab band of a lesson video): six string lines, fret numbers coloured by the finger
// that plays them, standard technique symbols, a fixed playhead and bar/section labels. Drawn on a 2D canvas
// every frame from the solved events (js/fingering.js + js/timeline.js) and the page's playback clock.
// Clicking a note seeks to it; dragging the strip scrubs.

export const FINGER_CSS = ['#4cc9f0', '#80ed99', '#ffd166', '#ef476f'];
const OPEN_CSS = '#f4f6fa';
const SNAMES = ['e', 'B', 'G', 'D', 'A', 'E'];
const OPEN_MIDI = [64, 59, 55, 50, 45, 40];
const BG = '#0d0f14';
const LINE = 'rgba(200,208,222,0.30)';
const MUTED = '#8a93a3';
const ACCENT = '#ffb347';

const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const bendLabel = semis => semis >= 3 ? '1½' : semis >= 2 ? 'full' : semis >= 1 ? '½' : '¼';

export class TabStrip {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {{seek:(t:number)=>void}} opts
   */
  constructor(canvas, opts) {
    this.cv = canvas;
    this.ctx = canvas.getContext('2d');
    this.seek = opts.seek;
    this.events = [];
    this.bars = []; this.sections = []; this.pmRuns = []; this.minGap = 0.5; this.strokes = []; this.spans = [];
    this.anchors = [[0, 0], [1, 1]];
    this.pxPerBeat = 100;
    this.w = 300; this.h = 140; this.dpr = 1;
    this.beatNow = 0;
    this.cur = -1;
    this.layout();
    this._bindPointer();
  }

  // ---------- model ----------
  setEvents(events) {
    this.events = events || [];
    const ev = this.events;
    const bpm = ev.bpm || 120, bps = bpm / 60;
    const t0 = ev.length ? ev[0].t : 0;
    // beat position of every event (user songs without q get one from the tempo)
    ev.forEach(e => { e._q = Number.isFinite(e.q) ? e.q : (e.t - t0) * bps; });
    // time <-> beat anchors: every note sits on its written beat at the moment it sounds
    const A = [];
    for (const e of ev) {
      const l = A[A.length - 1];
      if (!l || (e.t > l[0] + 1e-3 && e._q > l[1] + 1e-3)) A.push([e.t, e._q]);
    }
    if (!A.length) A.push([0, 0]);
    const f = A[0], l = A[A.length - 1];
    A.unshift([f[0] - 60, f[1] - 60 * bps]);
    A.push([l[0] + 60, l[1] + 60 * bps]);
    this.anchors = A;
    // spacing: the densest common gap must leave room for two-digit numbers + a technique symbol
    const gaps = [];
    for (let i = 1; i < ev.length; i++) { const g = ev[i]._q - ev[i - 1]._q; if (g > 0.05) gaps.push(g); }
    gaps.sort((a, b) => a - b);
    const g10 = gaps.length ? gaps[Math.floor(gaps.length * 0.1)] : 0.5;
    this.minGap = g10;
    // bars (4/4 like the tab view) and section starts
    this.bars = [];
    const lastQ = ev.length ? ev[ev.length - 1]._q + 4 : 4;
    for (let m = 0; m * 4 <= lastQ; m++) this.bars.push(m * 4);
    this.sections = [];
    let prev = null;
    for (const e of ev) {
      if (e.sec && e.sec !== prev) {
        const [name, sub] = e.sec.split('|');
        this.sections.push({ q: e._q, label: sub ? `${name} · ${sub}` : name });
        prev = e.sec;
      }
    }
    // palm-mute runs
    this.pmRuns = [];
    let run = null;
    ev.forEach((e, i) => {
      if (e.pm) { if (run && i === run.b + 1) run.b = i; else { run = { a: i, b: i }; this.pmRuns.push(run); } }
    });
    // strummed bars of a mixed song (js/songchords.js): strum arrows and chord names above the strings
    this.strokes = (ev.strokes || []).map(st => ({ t: st.t, q: this.beatOf(st.t), dir: st.dir, muted: st.muted }));
    this.spans = (ev.spans || []).filter(s => s.symbol).map(s => ({ t0: s.t0, t1: s.t1, q0: this.beatOf(s.t0), q1: this.beatOf(s.t1), symbol: s.symbol }));
    this.layout();
  }

  /** chord names (sticky while the chord is on screen) and one arrow per strum stroke: D down (amber), U up (blue) */
  _drawStrum(T, qL, qR) {
    const c = this.ctx, y0 = this.lineY(0);
    c.font = `800 ${this.font + 1}px system-ui, sans-serif`; c.textAlign = 'left'; c.textBaseline = 'alphabetic';
    for (let i = 0; i < this.spans.length; i++) {
      const sp = this.spans[i];
      if (sp.q1 < qL || sp.q0 > qR) continue;
      const nx = this.spans[i + 1], xEnd = Math.min(this.xOfBeat(sp.q1), nx ? this.xOfBeat(nx.q0) : Infinity) - 6;
      let x = Math.max(this.xOfBeat(sp.q0) - 3, this.gutter + 4);
      const tw = c.measureText(sp.symbol).width;
      if (x + tw > xEnd) x = xEnd - tw;
      if (x + tw < this.gutter + 4) continue;
      const now = T >= sp.t0 && T < sp.t1;
      c.fillStyle = now ? ACCENT : '#cfd6e2';
      c.fillText(sp.symbol, x, y0 - 29);
    }
    let cur = -1;
    for (let i = 0; i < this.strokes.length && this.strokes[i].t <= T + 0.005; i++) cur = i;
    for (let i = 0; i < this.strokes.length; i++) {
      const st = this.strokes[i];
      if (st.q < qL || st.q > qR) continue;
      const x = this.xOfBeat(st.q), dir = st.dir === 'U' ? -1 : 1, isNow = i === cur && T - st.t < 0.2;
      const L = isNow ? 6.5 : 5, cy = y0 - 17, tip = cy + dir * L, tail = cy - dir * L;
      c.globalAlpha = i < cur || (i === cur && !isNow) ? 0.5 : 1;
      c.strokeStyle = c.fillStyle = st.muted === 'dead' ? MUTED : dir > 0 ? ACCENT : '#4cc9f0';
      c.lineWidth = isNow ? 2.4 : 1.7;
      c.beginPath(); c.moveTo(x, tail); c.lineTo(x, tip - dir * 2.5); c.stroke();
      c.beginPath(); c.moveTo(x, tip); c.lineTo(x - 3.2, tip - dir * 4); c.lineTo(x + 3.2, tip - dir * 4); c.closePath(); c.fill();
      c.globalAlpha = 1;
    }
  }

  _interp(x, xi, yi) {
    const A = this.anchors, n = A.length;
    let lo = 0, hi = n - 1;
    if (x <= A[0][xi]) { lo = 0; hi = 1; } else if (x >= A[n - 1][xi]) { lo = n - 2; hi = n - 1; } else {
      while (hi - lo > 1) { const m = (lo + hi) >> 1; if (A[m][xi] <= x) lo = m; else hi = m; }
    }
    const a = A[lo], b = A[hi];
    return a[yi] + (x - a[xi]) * (b[yi] - a[yi]) / ((b[xi] - a[xi]) || 1);
  }
  beatOf(t) { return this._interp(t, 0, 1); }
  timeOf(q) { return this._interp(q, 1, 0); }

  // ---------- layout ----------
  layout() {
    const r = this.cv.getBoundingClientRect();
    this.dpr = Math.min(2, window.devicePixelRatio || 1);
    this.w = Math.max(200, r.width); this.h = Math.max(90, r.height);
    this.cv.width = Math.round(this.w * this.dpr); this.cv.height = Math.round(this.h * this.dpr);
    const small = this.w < 600;
    this.font = small ? 12 : 13;
    this.gutter = small ? 20 : 26;
    this.playX = Math.round(clamp(this.w * 0.28, 70, 340));
    this.topY = (small ? 40 : 46) + (this.strokes && this.strokes.length ? 18 : 0);   // high e line (room for strum arrows)
    this.gap = Math.min(18, (this.h - this.topY - 10) / 5);
    // wider spacing on wide screens, a little tighter on phones
    const perNote = small ? 24 : 30;
    this.pxPerBeat = clamp(perNote / (this.minGap || 0.5), small ? 56 : 70, small ? 150 : 220);
  }
  lineY(s) { return this.topY + s * this.gap; }
  xOfBeat(q) { return this.playX + (q - this.beatNow) * this.pxPerBeat; }

  // ---------- frame ----------
  update(T) {
    const ev = this.events, c = this.ctx;
    this.beatNow = this.beatOf(T);
    // current event: last onset <= T (highlighted while it is still the latest note)
    let lo = 0, hi = ev.length - 1, cur = -1;
    while (lo <= hi) { const m = (lo + hi) >> 1; if (ev[m].t <= T + 0.005) { cur = m; lo = m + 1; } else hi = m - 1; }
    this.cur = cur;
    c.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    c.fillStyle = BG; c.fillRect(0, 0, this.w, this.h);
    const W = this.w, y0 = this.lineY(0), y5 = this.lineY(5);
    const qL = this.beatNow - (this.playX + 40) / this.pxPerBeat, qR = this.beatNow + (W - this.playX + 60) / this.pxPerBeat;

    // playhead band (behind everything)
    c.fillStyle = 'rgba(255,179,71,0.07)';
    c.fillRect(this.playX - 13, y0 - 12, 26, y5 - y0 + 24);

    // strings
    c.strokeStyle = LINE; c.lineWidth = 1;
    for (let s = 0; s < 6; s++) { const y = Math.round(this.lineY(s)) + 0.5; c.beginPath(); c.moveTo(this.gutter, y); c.lineTo(W, y); c.stroke(); }

    // bars + bar numbers
    c.font = `600 ${this.font - 2}px system-ui, sans-serif`;
    c.textBaseline = 'alphabetic'; c.textAlign = 'left';
    this.bars.forEach((q, m) => {
      if (q < qL - 4 || q > qR) return;
      const x = Math.round(this.xOfBeat(q)) + 0.5;
      if (x < this.gutter) return;
      c.strokeStyle = 'rgba(200,208,222,0.45)'; c.beginPath(); c.moveTo(x, y0); c.lineTo(x, y5); c.stroke();
      c.fillStyle = MUTED; c.fillText(String(m + 1), x + 3, y0 - 6);
    });
    // sections
    c.font = `700 ${this.font - 1}px system-ui, sans-serif`;
    for (let i = 0; i < this.sections.length; i++) {
      const sc = this.sections[i], nx = this.sections[i + 1];
      let x = this.xOfBeat(sc.q) - 4;
      const xEnd = nx ? this.xOfBeat(nx.q) - 8 : Infinity;
      if (xEnd < this.gutter + 4 || x > W) continue;
      x = Math.max(x, this.gutter + 4);                  // sticky label while the section is on screen
      const tw = c.measureText(sc.label).width;
      if (x + tw > xEnd) x = xEnd - tw;
      c.fillStyle = ACCENT; c.fillText(sc.label, x, 13);
    }

    if (this.strokes && (this.strokes.length || this.spans.length)) this._drawStrum(T, qL, qR);

    // palm mute
    c.font = `600 ${this.font - 3}px system-ui, sans-serif`;
    for (const r of this.pmRuns) {
      const a = ev[r.a], b = ev[r.b];
      if (b._q < qL || a._q > qR) continue;
      const xa = this.xOfBeat(a._q), xb = this.xOfBeat(b._q) + 10, yy = y0 - 18;
      c.fillStyle = MUTED; c.fillText('P.M.', xa - 10, yy + 3);
      c.strokeStyle = MUTED; c.setLineDash([3, 3]); c.beginPath(); c.moveTo(xa + 18, yy); c.lineTo(xb, yy); c.stroke(); c.setLineDash([]);
      c.beginPath(); c.moveTo(xb, yy - 3); c.lineTo(xb, yy + 4); c.stroke();
    }

    // techniques, then notes on top
    let i0 = this._firstAtBeat(qL - 2);
    for (let i = i0; i < ev.length && ev[i]._q <= qR; i++) this._drawTech(i);
    for (let i = i0; i < ev.length && ev[i]._q <= qR; i++) this._drawNote(i, T);

    // left gutter with string names (covers notes scrolling out)
    const grd = c.createLinearGradient(this.gutter, 0, this.gutter + 36, 0);
    grd.addColorStop(0, BG); grd.addColorStop(1, 'rgba(13,15,20,0)');
    c.fillStyle = grd; c.fillRect(this.gutter, 18, 36, this.h);   // keeps the sticky section name crisp
    c.fillStyle = BG; c.fillRect(0, 0, this.gutter, this.h);
    c.font = `700 ${this.font - 1}px system-ui, sans-serif`; c.textAlign = 'center'; c.textBaseline = 'middle';
    c.fillStyle = MUTED;
    for (let s = 0; s < 6; s++) c.fillText(SNAMES[s], this.gutter / 2, this.lineY(s) + 0.5);

    // playhead
    c.fillStyle = ACCENT;
    c.shadowColor = 'rgba(255,179,71,0.8)'; c.shadowBlur = 6;
    c.fillRect(this.playX - 1, y0 - 14, 2, y5 - y0 + 28);
    c.shadowBlur = 0;
    c.beginPath(); c.moveTo(this.playX - 5, y0 - 16); c.lineTo(this.playX + 5, y0 - 16); c.lineTo(this.playX, y0 - 10); c.closePath(); c.fill();
  }

  _firstAtBeat(q) {
    const ev = this.events;
    let lo = 0, hi = ev.length - 1, ans = ev.length;
    while (lo <= hi) { const m = (lo + hi) >> 1; if (ev[m]._q >= q) { ans = m; hi = m - 1; } else lo = m + 1; }
    return ans;
  }

  _drawNote(i, T) {
    const c = this.ctx, e = this.events[i], x = this.xOfBeat(e._q);
    const isCur = i === this.cur && T < e.t + Math.max(e.dur, 0.25) + 0.6;
    const past = i < this.cur || (i === this.cur && !isCur);
    const upcoming = !past && !isCur && e.t - T < 0.6;
    c.font = `700 ${this.font}px system-ui, sans-serif`;
    c.textAlign = 'center'; c.textBaseline = 'middle';
    for (const n of e.notes) {
      const y = this.lineY(n.s);
      const label = e.hb ? `(${n.f})` : String(n.f);
      const tw = c.measureText(label).width;
      const bw = tw + 6, bh = this.font + 4;
      const col = n.finger ? FINGER_CSS[n.finger - 1] : OPEN_CSS;
      c.fillStyle = BG;
      c.fillRect(x - bw / 2, y - bh / 2, bw, bh);
      if (isCur) {
        const age = T - e.t, pulse = Math.exp(-Math.max(0, age) / 0.25);
        c.fillStyle = hexA(col, 0.22 + 0.25 * pulse);
        rr(c, x - bw / 2 - 2, y - bh / 2 - 1, bw + 4, bh + 2, 3); c.fill();
        c.strokeStyle = col; c.lineWidth = 2;
        rr(c, x - bw / 2 - 2, y - bh / 2 - 1, bw + 4, bh + 2, 3); c.stroke();
      } else if (upcoming) {
        c.strokeStyle = hexA(col, 0.45); c.lineWidth = 1;
        rr(c, x - bw / 2 - 1.5, y - bh / 2 - 0.5, bw + 3, bh + 1, 3); c.stroke();
      }
      c.globalAlpha = past ? 0.5 : 1;
      c.fillStyle = col;
      c.fillText(label, x, y + 0.5);
      c.globalAlpha = 1;
    }
  }

  _sym(txt, x, y, col = '#cfd6e2') {
    const c = this.ctx;
    c.font = `italic 700 ${this.font - 1}px system-ui, sans-serif`;
    c.textAlign = 'center'; c.textBaseline = 'middle';
    const w = Math.max(9, c.measureText(txt).width + 4);
    c.fillStyle = BG; c.fillRect(x - w / 2, y - 7, w, 14);
    c.fillStyle = col; c.fillText(txt, x, y + 0.5);
  }

  _drawTech(i) {
    const c = this.ctx, ev = this.events, e = ev[i], nx = ev[i + 1], pv = ev[i - 1];
    const x = this.xOfBeat(e._q), n0 = e.notes[0];
    if (!n0) return;
    const top = Math.min(...e.notes.map(n => n.s));
    const y = this.lineY(top), yTip = this.lineY(0) - 22;
    c.lineWidth = 1.4; c.strokeStyle = '#cfd6e2'; c.fillStyle = '#cfd6e2';
    if ((e.ham || e.pull) && nx) {
      const x2 = this.xOfBeat(nx._q), mid = (x + x2) / 2;
      c.beginPath(); c.moveTo(x + 2, y - 8); c.quadraticCurveTo(mid, y - 18, x2 - 2, y - 8); c.stroke();
      this._sym(e.ham ? 'h' : 'p', mid, this.lineY(n0.s));
    }
    if (e.slide && nx) {
      const s0 = n0.s, nn = nx.notes.find(n => n.s === s0) || nx.notes[0];
      const up = OPEN_MIDI[nn.s] + nn.f > OPEN_MIDI[s0] + n0.f;
      const x2 = this.xOfBeat(nx._q), y0 = this.lineY(s0);
      // a slanted line between the two numbers
      c.beginPath();
      const a = x + 9, b = x2 - 9, dy = 4.5;
      if (b - a > 4) { c.moveTo(a, up ? y0 + dy : y0 - dy); c.lineTo(b, up ? y0 - dy : y0 + dy); c.lineWidth = 1.8; c.stroke(); }
      else this._sym(up ? '/' : '\\', (x + x2) / 2, y0);
    }
    if (e.bend) {
      const semis = e.bend - n0.f, x0 = x + 8, x1 = x + 18;
      c.beginPath(); c.moveTo(x0, y - 2); c.quadraticCurveTo(x1, y - 2, x1, yTip + 6); c.stroke();
      c.beginPath(); c.moveTo(x1 - 3.5, yTip + 7); c.lineTo(x1, yTip); c.lineTo(x1 + 3.5, yTip + 7); c.closePath(); c.fill();
      c.font = `600 ${this.font - 3}px system-ui, sans-serif`; c.textAlign = 'left'; c.textBaseline = 'alphabetic';
      c.fillText(bendLabel(semis), x1 + 4, yTip + 4);
    }
    if (e.hb) {
      const fromX = pv && (pv.bend || pv.hb) ? this.xOfBeat(pv._q) + (pv.bend ? 18 : 8) : x - 12;
      c.beginPath(); c.moveTo(fromX, yTip); c.lineTo(x + 8, yTip); c.stroke();
      if (!(pv && pv.hb)) { c.font = `600 ${this.font - 3}px system-ui, sans-serif`; c.textAlign = 'left'; c.textBaseline = 'alphabetic'; c.fillText('hold', x - 12, yTip - 4); }
    }
    if (e.rel) {
      const xa = x + 8, xb = x + 20;
      c.beginPath(); c.moveTo(xa, yTip); c.quadraticCurveTo(xb, yTip, xb, y - 9); c.stroke();
      c.beginPath(); c.moveTo(xb - 3.5, y - 15); c.lineTo(xb, y - 8); c.lineTo(xb + 3.5, y - 15); c.closePath(); c.fill();
      this._sym('r', x + 13, this.lineY(n0.s));
    }
    if (e.vib) {
      const len = clamp((e.dur * (this.pxPerBeat * ((ev.bpm || 120) / 60))) * 0.8, 22, 90);
      const yy = this.lineY(0) - 30 + (e.bend || e.hb ? -2 : 8);
      c.beginPath(); c.moveTo(x - 4, yy);
      for (let k = 0; k < len; k += 6) { c.quadraticCurveTo(x - 4 + k + 1.5, yy - 4, x - 4 + k + 3, yy); c.quadraticCurveTo(x - 4 + k + 4.5, yy + 4, x - 4 + k + 6, yy); }
      c.stroke();
    }
  }

  // ---------- interaction ----------
  _bindPointer() {
    const cv = this.cv;
    let down = null;
    cv.addEventListener('pointerdown', e => {
      down = { x: e.clientX, beat: this.beatNow, drag: false };
      try { cv.setPointerCapture(e.pointerId); } catch (_) { /* ignore */ }
    });
    cv.addEventListener('pointermove', e => {
      if (!down) return;
      const dx = e.clientX - down.x;
      if (!down.drag && Math.abs(dx) > 6) down.drag = true;
      if (down.drag) this.seek(Math.max(0, this.timeOf(down.beat - dx / this.pxPerBeat)));
    });
    const up = e => {
      if (!down) return;
      const d = down; down = null;
      if (d.drag) return;
      const r = cv.getBoundingClientRect();
      const x = e.clientX - r.left, y = e.clientY - r.top;
      const hit = this.noteAt(x, y);
      if (hit >= 0) this.seek(Math.max(0, this.events[hit].t - 0.03));
    };
    cv.addEventListener('pointerup', up);
    cv.addEventListener('pointercancel', () => { down = null; });
  }

  /** index of the event under (x, y) in CSS px (nearest note within reach), or -1 */
  noteAt(x, y) {
    const ev = this.events;
    const q = this.beatNow + (x - this.playX) / this.pxPerBeat;
    let best = -1, bd = Infinity;
    for (let i = this._firstAtBeat(q - 2); i < ev.length && ev[i]._q < q + 2; i++) {
      const dx = Math.abs(this.xOfBeat(ev[i]._q) - x);
      const dy = Math.min(...ev[i].notes.map(n => Math.abs(this.lineY(n.s) - y)));
      const d = dx + 0.35 * dy;
      if (dx < 22 && d < bd) { bd = d; best = i; }
    }
    return best;
  }
}

function rr(c, x, y, w, h, r) {
  c.beginPath();
  c.moveTo(x + r, y); c.arcTo(x + w, y, x + w, y + h, r); c.arcTo(x + w, y + h, x, y + h, r);
  c.arcTo(x, y + h, x, y, r); c.arcTo(x, y, x + w, y, r); c.closePath();
}
function hexA(hex, a) {
  const v = parseInt(hex.slice(1), 16);
  return `rgba(${(v >> 16) & 255},${(v >> 8) & 255},${v & 255},${a})`;
}
