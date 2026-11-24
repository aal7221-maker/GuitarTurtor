// Hand + fretboard vision for create.html: which string/fret region the fretting fingertips are on, from the video of a
// YouTube capture (getDisplayMedia) or an uploaded video file. Browser port of ../handtrack (hands.py, fretboard.py,
// autodetect.py, calibrate.py): MediaPipe Tasks HandLandmarker (VIDEO mode) + an OpenCV.js fretboard tracker run in a
// Web Worker (js/handvision-worker.js); this module holds the shared geometry, the main-thread driver with its preview
// canvas and 4-click calibration, and the audio+video fusion cost used by songmaker.toScore.
//
// Plane coordinates (units of the scale length): X = distance from the nut (fret wire n at 1 - 2^(-n/12)), Y across the
// neck; string k (0 = high e … 5 = low E) at Y = (k - 2.5)·sp(X) with a linear taper. H maps plane -> image pixels.

export const SP0 = 0.0108, TAPER = 0.5, EDGE = 0.45, N_FRETS = 22;
export const fretX = n => 1 - Math.pow(2, -n / 12);
export const xToFretFloat = X => -12 * Math.log2(1 - Math.min(X, 0.999));
export const spacing = X => SP0 * (1 + TAPER * X);
export const stringY = (k, X) => (k - 2.5) * spacing(X);
export const planeToString = (X, Y) => Y / spacing(X) + 2.5;

// ---------- 3x3 homographies (row-major arrays of 9) ----------
export const mul3 = (A, B) => { const C = new Array(9); for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) C[i * 3 + j] = A[i * 3] * B[j] + A[i * 3 + 1] * B[3 + j] + A[i * 3 + 2] * B[6 + j]; return C; };
export function inv3(m) {
  const [a, b, c, d, e, f, g, h, i] = m, A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g, det = a * A + b * B + c * C;
  if (!det) return null;
  return [A / det, -(b * i - c * h) / det, (b * f - c * e) / det, B / det, (a * i - c * g) / det, -(a * f - c * d) / det, C / det, -(a * h - b * g) / det, (a * e - b * d) / det];
}
export function applyH(H, x, y) { const w = H[6] * x + H[7] * y + H[8], s = Math.abs(w) < 1e-12 ? 1e-12 : w; return [(H[0] * x + H[1] * y + H[2]) / s, (H[3] * x + H[4] * y + H[5]) / s]; }

function solve(A, b) {               // Gaussian elimination with partial pivoting; A n×n (array of rows)
  const n = b.length;
  for (let c = 0; c < n; c++) {
    let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    if (Math.abs(A[p][c]) < 1e-12) return null;
    [A[c], A[p]] = [A[p], A[c]]; [b[c], b[p]] = [b[p], b[c]];
    for (let r = c + 1; r < n; r++) { const k = A[r][c] / A[c][c]; if (!k) continue; for (let j = c; j < n; j++) A[r][j] -= k * A[c][j]; b[r] -= k * b[c]; }
  }
  const x = new Array(n);
  for (let r = n - 1; r >= 0; r--) { let s = b[r]; for (let j = r + 1; j < n; j++) s -= A[r][j] * x[j]; x[r] = s / A[r][r]; }
  return x;
}
function normT(P) {                  // Hartley conditioning
  let mx = 0, my = 0; for (const [x, y] of P) { mx += x; my += y; } mx /= P.length; my /= P.length;
  let d = 0; for (const [x, y] of P) d += Math.hypot(x - mx, y - my); d = d / P.length || 1;
  const s = Math.SQRT2 / d; return [s, 0, -s * mx, 0, s, -s * my, 0, 0, 1];
}
// least-squares DLT (h33 = 1) on conditioned points
function dlt(src, dst) {
  if (src.length < 4) return null;
  const Ts = normT(src), Td = normT(dst), A = Array.from({ length: 8 }, () => new Array(8).fill(0)), b = new Array(8).fill(0);
  for (let k = 0; k < src.length; k++) {
    const [x, y] = applyH(Ts, ...src[k]), [u, v] = applyH(Td, ...dst[k]);
    for (const [r, t] of [[[x, y, 1, 0, 0, 0, -u * x, -u * y], u], [[0, 0, 0, x, y, 1, -v * x, -v * y], v]]) {
      for (let i = 0; i < 8; i++) { b[i] += r[i] * t; for (let j = 0; j < 8; j++) A[i][j] += r[i] * r[j]; }
    }
  }
  const h = solve(A, b); if (!h) return null;
  const Ti = inv3(Td); if (!Ti) return null;
  return mul3(mul3(Ti, [...h, 1]), Ts);
}
let seed = 12345; const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
/** src -> dst homography; thr > 0 runs RANSAC (reprojection error in dst units). Returns { H, inl } or null. */
export function findH(src, dst, thr = 0, iters = 500) {
  const n = src.length; if (n < 4) return null;
  if (!thr) { const H = dlt(src, dst); return H && { H, inl: new Array(n).fill(true) }; }
  const err = (H, k) => { const [x, y] = applyH(H, ...src[k]); return Math.hypot(x - dst[k][0], y - dst[k][1]); };
  let best = null, bestN = 0;
  for (let it = 0; it < iters; it++) {
    const idx = new Set(); while (idx.size < 4) idx.add(Math.floor(rnd() * n));
    const s = [...idx], H = dlt(s.map(k => src[k]), s.map(k => dst[k])); if (!H) continue;
    let c = 0; for (let k = 0; k < n; k++) if (err(H, k) < thr) c++;
    if (c > bestN) { bestN = c; best = H; if (c === n) break; }
  }
  if (!best || bestN < 4) return null;
  let inl = src.map((_, k) => err(best, k) < thr);
  const H2 = dlt(src.filter((_, k) => inl[k]), dst.filter((_, k) => inl[k]));
  if (H2) { const inl2 = src.map((_, k) => err(H2, k) < thr); if (inl2.filter(Boolean).length >= bestN) { best = H2; inl = inl2; } }
  return { H: best, inl };
}
/** 4 clicks in stored order (fret a @ high e, fret a @ low E, fret b @ low E, fret b @ high e) -> H */
export function homographyFromCalibration(pts, [a, b]) {
  const Xa = fretX(a), Xb = fretX(b);
  const plane = [[Xa, stringY(0, Xa)], [Xa, stringY(5, Xa)], [Xb, stringY(5, Xb)], [Xb, stringY(0, Xb)]];
  return findH(plane, pts)?.H || null;
}
/** Homography of a plane `height` (scale units) above the board towards the camera: fingertip landmarks sit ~1 cm above
 *  the strings, so mapping them through the board H misplaces them under oblique views (fretboard.py lifted_h). */
export function liftedH(H, w, h, height) {
  if (!height) return H;
  const C = [1, 0, -w / 2, 0, 1, -h / 2, 0, 0, 1], Hc = mul3(C, H);
  const a = [Hc[0], Hc[3], Hc[6]], b = [Hc[1], Hc[4], Hc[7]];
  const fs = [];
  for (const [num, den] of [[-(a[0] * b[0] + a[1] * b[1]), a[2] * b[2]], [-(a[0] ** 2 + a[1] ** 2 - b[0] ** 2 - b[1] ** 2), a[2] ** 2 - b[2] ** 2]])
    if (Math.abs(den) > 1e-12 && num / den > 0) fs.push(Math.sqrt(num / den));
  fs.sort((x, y) => x - y);
  const M0 = Math.max(w, h), f = fs.length ? Math.min(4 * M0, Math.max(0.5 * M0, fs.length === 2 ? (fs[0] + fs[1]) / 2 : fs[0])) : 1.2 * M0;
  const M = [Hc[0] / f, Hc[1] / f, Hc[2] / f, Hc[3] / f, Hc[4] / f, Hc[5] / f, Hc[6], Hc[7], Hc[8]];
  const lam = 1 / Math.hypot(M[0], M[3], M[6]);
  let r1 = [M[0], M[3], M[6]].map(v => v * lam), r2 = [M[1], M[4], M[7]].map(v => v * lam), t = [M[2], M[5], M[8]].map(v => v * lam);
  if (t[2] < 0) { r1 = r1.map(v => -v); r2 = r2.map(v => -v); t = t.map(v => -v); }
  let r3 = [r1[1] * r2[2] - r1[2] * r2[1], r1[2] * r2[0] - r1[0] * r2[2], r1[0] * r2[1] - r1[1] * r2[0]];
  if (r3[0] * t[0] + r3[1] * t[1] + r3[2] * t[2] > 0) r3 = r3.map(v => -v);
  const tt = t.map((v, i) => v + height * r3[i]);
  const Hh = [r1[0], r2[0], tt[0], r1[1], r2[1], tt[1], r1[2], r2[2], tt[2]];
  return mul3(mul3(inv3(C), [f, 0, 0, 0, f, 0, 0, 0, 1]), Hh);
}

// ---------- audio + video fusion ----------
// track: { frames: [{ t, tips: [[stringFloat, fretFloat] …] | null }] } in seconds on the audio clock.
// For a note at time t with candidate positions [[s, f] …] (all give the heard pitch) the cost of each candidate is its
// distance to the nearest fretting fingertip (fret units; string differences count less, the fret region is what separates
// same-pitch positions: one string over = 5 frets along), the median over the frames around the onset. Open strings get
// a flat cost (any hand position can play them). Returns null when the hand/grid is not visible near t.
export function visionCosts(track, { before = 0.12, after = 0.08, strW = 0.5, openCost = 'min', cap = 6 } = {}) {
  const fr = (track?.frames || []).filter(f => f.tips && f.tips.length).sort((a, b) => a.t - b.t);
  const ts = fr.map(f => f.t);
  const lower = t => { let lo = 0, hi = ts.length; while (lo < hi) { const m = (lo + hi) >> 1; if (ts[m] < t) lo = m + 1; else hi = m; } return lo; };
  return (t, cands) => {
    const win = [];
    for (let i = lower(t - before); i < fr.length && fr[i].t <= t + after; i++) win.push(fr[i]);
    if (!win.length) return null;
    const c = cands.map(([s, f]) => {
      if (f === 0) return null;
      const d = win.map(w => Math.min(cap, ...w.tips.map(([sf, ff]) => Math.hypot(ff - (f - 0.5), strW * (sf - s))))).sort((a, b) => a - b);
      return d[d.length >> 1];
    });
    // open string: a finger resting on the fretted twin looks the same, so by default ('min') the video stays neutral
    // between open and the best fretted position and the hand-path Viterbi (drone idiom) decides
    const best = Math.min(...c.filter(v => v != null), cap);
    return c.map(v => v != null ? v : openCost === 'min' ? best : openCost);
  };
}

// ---------- main-thread driver ----------
const GRID_COL = 'rgba(243,164,59,.8)', TIP_COL = '#3df5ff';
export class HandVision {
  /** canvas: preview <canvas>; onStatus(text) */
  constructor({ canvas = null, onStatus = () => {}, maxSide = 720 } = {}) {
    this.canvas = canvas; this.onStatus = onStatus; this.maxSide = maxSide;
    this.frames = []; this.worker = null; this.ready = null; this.pending = null; this.last = null; this.calibMode = null;
    this.stats = { frames: 0, hand: 0, grid: 0, tips: 0, ms: 0 };
  }
  init() {
    if (this.ready) return this.ready;
    this.worker = new Worker(new URL('./handvision-worker.js', import.meta.url), { type: 'module' });
    this.ready = new Promise((resolve, reject) => {
      this.worker.onmessage = e => {
        const m = e.data;
        if (m.type === 'ready') resolve();
        else if (m.type === 'error') { (this.pending ? this.pending.reject : reject)(new Error(m.message)); this.pending = null; }
        else if (m.type === 'status') this.onStatus(m.text);
        else if (m.type === 'result' || m.type === 'calibrated') { const p = this.pending; this.pending = null; p && p.resolve(m); }
      };
      this.worker.onerror = e => { const err = new Error('Vision worker failed: ' + (e.message || 'load error')); (this.pending ? this.pending.reject : reject)(err); this.pending = null; };
      this.worker.postMessage({ type: 'init', modelUrl: new URL('../models/hand_landmarker.task', import.meta.url).href });
    });
    return this.ready;
  }
  call(msg, transfer = []) { return new Promise((resolve, reject) => { this.pending = { resolve, reject }; this.worker.postMessage(msg, transfer); }); }
  get busy() { return !!this.pending; }
  // one frame: source = <video>, crop = {x, y, w, h} in source pixels; t = audio-clock seconds
  async push(source, t, crop = null) {
    const sw = crop ? crop.w : source.videoWidth, sh = crop ? crop.h : source.videoHeight;
    if (!sw || !sh) return null;
    const s = Math.min(1, this.maxSide / Math.max(sw, sh)), W = Math.round(sw * s), Hh = Math.round(sh * s);
    const bmp = await createImageBitmap(source, crop ? crop.x : 0, crop ? crop.y : 0, sw, sh, { resizeWidth: W, resizeHeight: Hh, resizeQuality: 'medium' });
    if (this.canvas && !this.calibMode) this.drawFrame(bmp);
    const r = await this.call({ type: 'frame', t, bitmap: bmp }, [bmp]);
    this.stats.frames++; this.stats.ms += r.ms || 0;
    if (r.hand) this.stats.hand++;
    if (r.H) this.stats.grid++;
    if (r.tips && r.tips.length) this.stats.tips++;
    if (r.flipNow != null) {          // string order was just resolved from the wrist side: fix the frames before it
      for (const f of this.frames) if (f.u) { if (r.flipNow && f.tips) f.tips = f.tips.map(([sf, ff]) => [+(5 - sf).toFixed(2), ff]); delete f.u; }
    }
    this.frames.push({ t: +t.toFixed(3), tips: r.tips && r.tips.length ? r.tips.map(p => [+p.sf.toFixed(2), +p.ff.toFixed(2)]) : null, ...(r.unresolved ? { u: 1 } : {}) });
    this.last = r;
    if (this.canvas && !this.calibMode) this.drawOverlay(r);
    return r;
  }
  drawFrame(bmp) {
    const c = this.canvas, g = c.getContext('2d');
    const k = Math.min(c.clientWidth || 240, 360) / bmp.width;
    c.width = Math.round(bmp.width * k); c.height = Math.round(bmp.height * k); this.scale = k; this.fw = bmp.width; this.fh = bmp.height;
    g.drawImage(bmp, 0, 0, c.width, c.height);
  }
  drawOverlay(r) {
    const g = this.canvas.getContext('2d'), k = this.scale || 1, P = (x, y) => { const [u, v] = applyH(r.H, x, y); return [u * k, v * k]; };
    g.lineWidth = 1;
    if (r.H) {
      g.strokeStyle = GRID_COL; g.fillStyle = GRID_COL; g.font = '10px system-ui';
      for (const n of r.frets || []) {
        const X = fretX(n), a = P(X, stringY(-0.3, X)), b = P(X, stringY(5.3, X));
        g.beginPath(); g.moveTo(...a); g.lineTo(...b); g.stroke();
        if ([0, 3, 5, 7, 9, 12, 15, 17].includes(n)) g.fillText(String(n), b[0] + 2, b[1] + 8);
      }
      const f0 = (r.frets || [0])[0], f1 = (r.frets || [12]).at(-1);
      for (let s = 0; s < 6; s++) {
        g.strokeStyle = s === 0 ? 'rgba(255,255,255,.9)' : 'rgba(255,255,255,.45)';
        g.beginPath(); g.moveTo(...P(fretX(Math.max(0, f0 - 1)), stringY(s, fretX(Math.max(0, f0 - 1))))); g.lineTo(...P(fretX(f1), stringY(s, fretX(f1)))); g.stroke();
      }
    }
    if (r.hand) { g.fillStyle = 'rgba(255,255,255,.5)'; for (const [x, y] of r.hand) g.fillRect(x * k - 1, y * k - 1, 2, 2); }
    g.font = 'bold 11px system-ui';
    for (const p of r.tips || []) {
      g.fillStyle = TIP_COL; g.beginPath(); g.arc(p.x * k, p.y * k, 3.5, 0, 7); g.fill();
      g.fillText(`${'eBGDAE'[Math.max(0, Math.min(5, Math.round(p.sf)))]}${Math.max(0, Math.ceil(p.ff))}`, p.x * k + 5, p.y * k - 4);
    }
    g.fillStyle = '#fff'; g.font = '11px system-ui';
    g.fillText(`${r.state || ''} t=${(r.t || 0).toFixed(1)}s`, 4, 12);
  }
  /** 4-click calibration on the frame in the preview: low E @ fret a, high e @ fret a, low E @ fret b, high e @ fret b. */
  calibrate(frets = [0, 12]) {
    if (!this.canvas || !this.fw) return Promise.reject(new Error('No video frame to calibrate on yet.'));
    const c = this.canvas, k = this.scale || 1, pts = [], names = ['LOW E (thickest) string', 'HIGH e (thinnest) string'];
    const label = i => `${i + 1}/4: click ${frets[i >> 1] === 0 ? 'the nut' : 'fret ' + frets[i >> 1]} where it crosses the ${names[i & 1]}`;
    this.calibMode = true; this.onStatus(label(0));
    return new Promise(resolve => {
      const onClick = async ev => {
        const r = c.getBoundingClientRect(), x = (ev.clientX - r.left) * (c.width / r.width) / k, y = (ev.clientY - r.top) * (c.height / r.height) / k;
        pts.push([x, y]);
        const g = c.getContext('2d'); g.fillStyle = '#ff5ad6'; g.beginPath(); g.arc(x * k, y * k, 4, 0, 7); g.fill();
        if (pts.length < 4) { this.onStatus(label(pts.length)); return; }
        c.removeEventListener('click', onClick);
        const std = [pts[1], pts[0], pts[2], pts[3]];        // UI order -> stored order (a@e, a@E, b@E, b@e)
        // the worker may be mid-frame: wait for it
        while (this.pending) await new Promise(r2 => setTimeout(r2, 20));
        const res = await this.call({ type: 'calib', points: std, frets });
        this.calibMode = null;
        this.onStatus(res.ok ? 'Grid calibrated.' : 'Calibration saved (grid not snapped to the frets; check the preview).');
        resolve(res);
      };
      c.addEventListener('click', onClick);
    });
  }
  /** Run on a whole video file: seek-step at `fps`; frames on the file clock (= its audio clock). */
  async processFile(file, { fps = 12, progress = () => {}, crop = null, signal = null } = {}) {
    await this.init();
    const v = document.createElement('video');
    v.muted = true; v.playsInline = true; v.preload = 'auto'; v.src = URL.createObjectURL(file);
    await new Promise((res, rej) => { v.onloadeddata = res; v.onerror = () => rej(new Error('This video file can’t be decoded here.')); });
    const dur = v.duration;
    const seek = t => new Promise(res => { const done = () => { v.removeEventListener('seeked', done); res(); }; v.addEventListener('seeked', done); v.currentTime = t; });
    for (let t = 0; t < dur; t += 1 / fps) {
      if (signal?.aborted) break;
      while (this.calibMode) await new Promise(r => setTimeout(r, 100));
      await seek(t);
      await this.push(v, t, crop);
      progress(t / dur);
    }
    URL.revokeObjectURL(v.src);
    return this.track();
  }
  /** Live: frames of a captured stream; clock() = seconds on the audio clock; crop() = source rect or null. */
  startStream(stream, { clock, crop = () => null, fps = 12 } = {}) {
    const track = stream.getVideoTracks()[0];
    if (!track) return false;
    const v = document.createElement('video'); v.muted = true; v.playsInline = true;
    v.srcObject = new MediaStream([track]); v.play().catch(() => {});
    this.live = { v, stop: false };
    const tick = async () => {
      if (!this.live || this.live.stop) return;
      const t = clock();
      if (t != null && t >= 0 && !this.pending && !this.calibMode && v.videoWidth) {
        try { await this.push(v, t, crop(v)); } catch (e) { console.warn('vision frame failed:', e); }
      }
      if (this.live && !this.live.stop) setTimeout(tick, Math.max(0, 1000 / fps - 5));
    };
    this.init().then(tick, e => this.onStatus('Hand tracking unavailable: ' + e.message));
    return true;
  }
  stopStream() { if (this.live) { this.live.stop = true; this.live.v.srcObject = null; this.live = null; } return this.track(); }
  track() { return { frames: this.frames.slice(), stats: { ...this.stats } }; }
  reset() { this.frames = []; this.stats = { frames: 0, hand: 0, grid: 0, tips: 0, ms: 0 }; if (this.worker) this.worker.postMessage({ type: 'reset' }); }
  dispose() { this.stopStream(); if (this.worker) this.worker.terminate(); this.worker = null; this.ready = null; }
}
