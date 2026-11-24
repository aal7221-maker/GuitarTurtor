// Web Worker for js/handvision.js: per frame, MediaPipe hand landmarks + OpenCV.js fretboard tracking -> fingertip
// (string, fret). Port of ../handtrack (hands.py, fretboard.py, autodetect.py); comments there explain the method.
// Libraries are pinned on jsDelivr; the hand model is committed in models/ (Apache-2.0, MediaPipe).
import { fretX, xToFretFloat, spacing, stringY, planeToString, applyH, inv3, mul3, findH, homographyFromCalibration, liftedH, EDGE, N_FRETS } from './handvision.js';

const CV_URL = 'https://cdn.jsdelivr.net/npm/@techstark/opencv-js@4.10.0-release.1/dist/opencv.js';
const MP = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/';
const MP_MODEL_FALLBACK = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const TIPS = [8, 12, 16, 20], DIPS = [7, 11, 15, 19];
const CONN = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]];
const SINGLE = new Set([3, 5, 7, 9, 15, 17, 19, 21]), DOUBLE = new Set([12, 24]);
const OPT = { alpha: -0.15, tipHeight: 10 / 648, wantLabel: 'Left', autoEvery: 5, maxLost: 20, inlayEvery: 12, regEvery: 12 };

// module workers have no importScripts; MediaPipe's wasm loader and the OpenCV.js UMD build expect it (sync XHR + global eval)
self.importScripts = (...urls) => {
  for (const u of urls) {
    const x = new XMLHttpRequest(); x.open('GET', String(u), false); x.send();
    if (x.status !== 200) throw new Error('Could not load ' + u + ' (HTTP ' + x.status + ')');
    (0, eval)(x.responseText + '\n//# sourceURL=' + u);
  }
};
let cv = null, hands = null, handsImg = null;
const status = text => postMessage({ type: 'status', text });

async function init(modelUrl) {
  status('Loading OpenCV.js (10 MB, first time only)…');
  self.importScripts(CV_URL);
  // the build's Module is a self-resolving thenable (never await it): wait for embind to publish cv.Mat
  if (self.cv instanceof Promise) self.cv = await self.cv;
  for (let i = 0; !(self.cv && self.cv.Mat); i++) {
    if (i > 600) throw new Error('OpenCV.js did not start');
    await new Promise(r => setTimeout(r, 50));
  }
  cv = self.cv;
  status('Loading the hand model…');
  const { FilesetResolver, HandLandmarker } = await import(MP + 'vision_bundle.mjs');
  const fileset = await FilesetResolver.forVisionTasks(MP + 'wasm');
  const make = async (url, mode, conf) => HandLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: url, delegate: 'CPU' }, runningMode: mode, numHands: 2,
    minHandDetectionConfidence: conf, minHandPresenceConfidence: conf, minTrackingConfidence: conf });
  let url = modelUrl;
  try { const r = await fetch(url, { method: 'HEAD' }); if (!r.ok) url = MP_MODEL_FALLBACK; } catch { url = MP_MODEL_FALLBACK; }
  hands = await make(url, 'VIDEO', 0.3);
  handsImg = await make(url, 'IMAGE', 0.2);
}

// ---------- small numeric helpers ----------
const median = a => { const s = a.filter(v => !Number.isNaN(v)).sort((x, y) => x - y); return s.length ? (s.length % 2 ? s[s.length >> 1] : 0.5 * (s[s.length / 2 - 1] + s[s.length / 2])) : NaN; };
const clip = (v, a, b) => Math.max(a, Math.min(b, v));
const toMat32 = (data, W, H) => cv.matFromArray(H, W, cv.CV_32F, data);
function gauss32(data, W, H, sigma) {
  const m = toMat32(data, W, H), o = new cv.Mat();
  cv.GaussianBlur(m, o, new cv.Size(0, 0), sigma, sigma, cv.BORDER_DEFAULT);
  const out = Float32Array.from(o.data32F); m.delete(); o.delete(); return out;
}
function sobel32(data, W, H, dx, dy) {
  const m = toMat32(data, W, H), o = new cv.Mat();
  cv.Sobel(m, o, cv.CV_32F, dx, dy, 3, 1, 0, cv.BORDER_DEFAULT);
  const out = Float32Array.from(o.data32F, Math.abs); m.delete(); o.delete(); return out;
}
const refl = (i, n) => { if (n === 1) return 0; while (i < 0 || i >= n) i = i < 0 ? -i : 2 * n - 2 - i; return i; };
function blur1(a, k) { const n = a.length, h = k >> 1, o = new Float32Array(n); for (let i = 0; i < n; i++) { let s = 0; for (let j = i - h; j < i - h + k; j++) s += a[refl(j, n)]; o[i] = s / k; } return o; }
function gauss1(a, sigma) { const r = Math.max(1, Math.round(4 * sigma)), w = []; let z = 0; for (let j = -r; j <= r; j++) { const v = Math.exp(-0.5 * j * j / sigma / sigma); w.push(v); z += v; } const n = a.length, o = new Float32Array(n); for (let i = 0; i < n; i++) { let s = 0; for (let j = -r; j <= r; j++) s += a[refl(i + j, n)] * w[j + r]; o[i] = s / z; } return o; }
function blur2(P, W, H, k) {  // box blur k×k, reflect101 (cv2.blur)
  const h = k >> 1, T = new Float32Array(W * H), O = new Float32Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { let s = 0; for (let j = x - h; j < x - h + k; j++) s += P[y * W + refl(j, W)]; T[y * W + x] = s / k; }
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { let s = 0; for (let j = y - h; j < y - h + k; j++) s += T[refl(j, H) * W + x]; O[y * W + x] = s / k; }
  return O;
}
// warp a CV_8UC1 Mat with image->dst matrix M (row-major 9) into W×H; returns Uint8Array
function warp(src, M, W, H, nearest = false) {
  const Mm = cv.matFromArray(3, 3, cv.CV_64F, M), o = new cv.Mat();
  cv.warpPerspective(src, o, Mm, new cv.Size(W, H), nearest ? cv.INTER_NEAREST : cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar(0));
  const out = Uint8Array.from(o.data); Mm.delete(); o.delete(); return out;
}
let ones = null;
function onesLike(g) { if (!ones || ones.rows !== g.rows || ones.cols !== g.cols) { if (ones) ones.delete(); ones = new cv.Mat(g.rows, g.cols, cv.CV_8UC1, new cv.Scalar(255)); } return ones; }
// mask drawn with a 2D canvas (polygons / thick lines), as a CV_8UC1 Mat
function drawMask(W, H, draw) {
  const c = new OffscreenCanvas(W, H), g = c.getContext('2d'); g.fillStyle = g.strokeStyle = '#fff'; g.lineCap = 'round'; g.lineJoin = 'round';
  draw(g);
  const d = g.getImageData(0, 0, W, H).data, m = new Uint8Array(W * H);
  for (let i = 0; i < m.length; i++) m[i] = d[4 * i + 3] > 127 ? 255 : 0;
  return cv.matFromArray(H, W, cv.CV_8UC1, m);
}
function hull(P) {
  const p = P.map(q => q.slice()).sort((a, b) => a[0] - b[0] || a[1] - b[1]), cr = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lo = [], up = [];
  for (const q of p) { while (lo.length >= 2 && cr(lo.at(-2), lo.at(-1), q) <= 0) lo.pop(); lo.push(q); }
  for (const q of p.reverse()) { while (up.length >= 2 && cr(up.at(-2), up.at(-1), q) <= 0) up.pop(); up.push(q); }
  return lo.slice(0, -1).concat(up.slice(0, -1));
}
const handSize = pts => Math.hypot(pts[0][0] - pts[9][0], pts[0][1] - pts[9][1]);
function handMaskFill(g, pts, dil = 0.15) {
  const sz = handSize(pts), r = Math.max(2, 0.18 * sz), k = Math.max(3, dil * sz), hl = hull(pts);
  g.beginPath(); hl.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y))); g.closePath(); g.fill();
  g.lineWidth = k; g.stroke();
  g.lineWidth = r + k;
  for (const [a, b] of CONN) { g.beginPath(); g.moveTo(...pts[a]); g.lineTo(...pts[b]); g.stroke(); }
}

// ---------- fretboard tracker (fretboard.py) ----------
class Fretboard {
  constructor() { this.H = null; this.valid = false; this.lost = 0; this.prev = null; this.lastOk = false; this.fretRange = [0, 99]; this.snapStrings = true; this.stillPx = 0.25; }
  toPlane(x, y) { return applyH(inv3(this.H), x, y); }
  visibleFrets(w, h, margin = 0.02) {
    const out = [];
    for (let n = 0; n <= N_FRETS; n++) {
      const X = fretX(n), ps = [applyH(this.H, X, stringY(0, X)), applyH(this.H, X, stringY(5, X))];
      if (ps.some(([x, y]) => Number.isFinite(x) && x >= -margin * w && x <= (1 + margin) * w && y >= -margin * h && y <= (1 + margin) * h)) out.push(n);
    }
    return out;
  }
  polygon(w, h, frets = null, pad = 0.6) {
    frets = frets || this.visibleFrets(w, h);
    if (frets.length < 2) return null;
    const Xa = fretX(Math.max(frets[0] - 1, 0)), Xb = fretX(Math.min(frets.at(-1) + 1, N_FRETS));
    return [[Xa, stringY(-pad, Xa)], [Xb, stringY(-pad, Xb)], [Xb, stringY(5 + pad, Xb)], [Xa, stringY(5 + pad, Xa)]].map(([x, y]) => applyH(this.H, x, y));
  }
  setPrev(gray) { if (this.prev) this.prev.delete(); this.prev = gray.clone(); }
  predict(gray, handMask) {
    if (!this.H || !this.prev) { this.setPrev(gray); return false; }
    const W = gray.cols, H = gray.rows, poly = this.polygon(W, H);
    let moved = false;
    if (poly) {
      const mask = drawMask(W, H, g => { g.beginPath(); poly.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y))); g.closePath(); g.fill(); });
      if (handMask) { const md = mask.data, hd = handMask.data; for (let i = 0; i < md.length; i++) if (hd[i]) md[i] = 0; }
      const p0 = new cv.Mat(), p1 = new cv.Mat(), p0r = new cv.Mat(), st = new cv.Mat(), st2 = new cv.Mat(), err = new cv.Mat(), err2 = new cv.Mat();
      try {
        cv.goodFeaturesToTrack(this.prev, p0, 300, 0.01, 5, mask);
        if (p0.rows >= 8) {
          const win = new cv.Size(21, 21), crit = new cv.TermCriteria(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01);
          cv.calcOpticalFlowPyrLK(this.prev, gray, p0, p1, st, err, win, 3, crit);
          cv.calcOpticalFlowPyrLK(gray, this.prev, p1, p0r, st2, err2, win, 3, crit);
          const a = p0.data32F, b = p1.data32F, c = p0r.data32F, src = [], dst = [], disp = [];
          for (let i = 0; i < p0.rows; i++) {
            if (st.data[i] !== 1 || st2.data[i] !== 1 || Math.hypot(a[2 * i] - c[2 * i], a[2 * i + 1] - c[2 * i + 1]) >= 1) continue;
            src.push([a[2 * i], a[2 * i + 1]]); dst.push([b[2 * i], b[2 * i + 1]]); disp.push(Math.hypot(b[2 * i] - a[2 * i], b[2 * i + 1] - a[2 * i + 1]));
          }
          if (src.length >= 8) {
            if (median(disp) < this.stillPx) return false;      // held still: keep prev so slow motion accumulates
            const r = findH(src, dst, 2.0, 200);
            if (r && r.inl.filter(Boolean).length >= 8) { this.H = mul3(r.H, this.H); moved = true; }
          }
        }
      } finally { [mask, p0, p1, p0r, st, st2, err, err2].forEach(m => m.delete()); }
    }
    this.setPrev(gray);
    return moved;
  }
  snap(gray, handMask, iters = 2, minInl = 10) {
    let res = null;
    for (let i = 0; i < iters; i++) { const r = this.snapOnce(gray, handMask, minInl); if (r.ok) { this.H = r.H; res = r; } else break; }
    return res || { ok: false, H: this.H };
  }
  snapOnce(gray, handMask, minInl) {
    const Wi = gray.cols, Hi = gray.rows, fail = { ok: false, H: this.H };
    const frets = this.visibleFrets(Wi, Hi).filter(n => n >= 1 && n >= this.fretRange[0] && n <= this.fretRange[1]);
    if (frets.length < 3) return fail;
    // rectify
    const Xa = fretX(Math.max(frets[0] - 1, 0)), Xb = fretX(Math.min(frets.at(-1) + 1, N_FRETS + 1)), ymax = 3.4 * spacing(Xb);
    const pa = applyH(this.H, Xa, 0), pb = applyH(this.H, Xb, 0), imgLen = Math.hypot(pb[0] - pa[0], pb[1] - pa[1]);
    let S = clip(1.3 * imgLen / Math.max(Xb - Xa, 1e-6), 300, 8000);
    const W = Math.floor(Math.min(2400, (Xb - Xa) * S)) + 1; S = (W - 1) / (Xb - Xa);
    const Hh = Math.floor(2 * ymax * S) + 1;
    if (W < 8 || Hh < 8 || Hh > 2000) return fail;
    const A = [S, 0, -Xa * S, 0, S, ymax * S, 0, 0, 1], Hinv = inv3(this.H); if (!Hinv) return fail;
    const M = mul3(A, Hinv);
    const R = warp(gray, M, W, Hh), valid = warp(onesLike(gray), M, W, Hh, true), hm = handMask ? warp(handMask, M, W, Hh, true) : null;
    const G = gauss32(Float32Array.from(R), W, Hh, Math.max(1, S * 0.0012));
    const dxx = sobel32(G, W, Hh, 2, 0), dyy = sobel32(G, W, Hh, 0, 2);
    const bad = new Uint8Array(W * Hh);
    for (let i = 0; i < bad.length; i++) if (!valid[i] || (hm && hm[i])) { bad[i] = 1; dxx[i] = 0; dyy[i] = 0; }
    const badFrac = (y0, y1, x0, x1) => { let s = 0, n = 0; for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) { s += bad[y * W + x]; n++; } return n ? s / n : 1; };
    const colProf = (D, y0, y1, x0, x1) => { const o = []; for (let x = x0; x < x1; x++) { let s = 0; for (let y = y0; y < y1; y++) s += D[y * W + x]; o.push(s / Math.max(1, y1 - y0)); } return o; };
    const rowProf = (D, y0, y1, x0, x1) => { const o = []; for (let y = y0; y < y1; y++) { let s = 0; for (let x = x0; x < x1; x++) s += D[y * W + x]; o.push(s / Math.max(1, x1 - x0)); } return o; };
    const peak = prof => {
      let i = 1; for (let k = 2; k < prof.length - 1; k++) if (prof[k] > prof[i]) i = k;
      const pk = prof[i], base = median(prof.slice()) + 1e-3;
      if (pk < 2 * base || pk < 2) return null;
      const den = prof[i - 1] - 2 * pk + prof[i + 1], off = Math.abs(den) > 1e-6 ? 0.5 * (prof[i - 1] - prof[i + 1]) / den : 0;
      return i + clip(off, -0.5, 0.5);
    };
    const toPl = (x, y) => [Xa + x / S, y / S - ymax];
    const targets = [], found = [], fretHits = {}, strHits = new Set();
    const bands = [[-0.45, 0], [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 5.45]];
    for (const n of frets) {
      const Xn = fretX(n), win = 0.3 * Math.min(Xn - fretX(n - 1), fretX(n + 1) - Xn) * S, xc = (Xn - Xa) * S;
      const x0 = Math.floor(xc - win), x1 = Math.ceil(xc + win);
      if (x0 < 1 || x1 >= W - 1) continue;
      let hits = 0;
      for (const [k0, k1] of bands) {
        let ya = Math.trunc((stringY(k0, Xn) + ymax) * S), yb = Math.trunc((stringY(k1, Xn) + ymax) * S);
        [ya, yb] = [Math.max(Math.min(ya, yb), 0), Math.min(Math.max(ya, yb), Hh - 1)];
        if (yb - ya < 3) continue;
        const pad = Math.max(1, (yb - ya) / 5 | 0);
        if (1 - badFrac(ya + pad, yb - pad + 1, x0, x1 + 1) < 0.7) continue;
        const prof = colProf(dxx, ya + pad, yb - pad + 1, x0 - 1, x1 + 2), i = prof.length > 2 ? peak(prof) : null;
        if (i == null) continue;
        const [Xd, Yd] = toPl(x0 - 1 + i, 0.5 * (ya + yb));
        targets.push([Xn, Yd]); found.push([Xd, Yd]); hits++;
      }
      if (hits) fretHits[n] = hits;
    }
    if (this.snapStrings) for (const n of frets) {
      const Xm = 0.5 * (fretX(n - 1) + fretX(n)), dX = fretX(n) - fretX(n - 1), xc = (Xm - Xa) * S, hw = Math.max(2, Math.trunc(0.25 * dX * S));
      const x0 = Math.trunc(xc - hw), x1 = Math.trunc(xc + hw);
      if (x0 < 0 || x1 >= W) continue;
      const sp = spacing(Xm);
      for (let k = 0; k < 6; k++) {
        const yc = (stringY(k, Xm) + ymax) * S, win = 0.38 * sp * S, y0 = Math.floor(yc - win), y1 = Math.ceil(yc + win);
        if (y0 < 1 || y1 >= Hh - 1 || 1 - badFrac(y0, y1 + 1, x0, x1 + 1) < 0.7) continue;
        const prof = rowProf(dyy, y0 - 1, y1 + 2, x0, x1 + 1), i = peak(prof);
        if (i == null) continue;
        const [Xd, Yd] = toPl(xc, y0 - 1 + i);
        targets.push([Xm, stringY(k, Xm)]); found.push([Xd, Yd]); strHits.add(k);
      }
    }
    if (strHits.size < 3 && this.snapStrings) {        // board edges when the strings are invisible
      const dy = sobel32(G, W, Hh, 0, 1); for (let i = 0; i < bad.length; i++) if (bad[i]) dy[i] = 0;
      for (const n of frets) {
        const Xm = 0.5 * (fretX(n - 1) + fretX(n)), dX = fretX(n) - fretX(n - 1), xc = (Xm - Xa) * S, hw = Math.max(2, Math.trunc(0.3 * dX * S));
        const x0 = Math.trunc(xc - hw), x1 = Math.trunc(xc + hw);
        if (x0 < 0 || x1 >= W) continue;
        const sp = spacing(Xm);
        for (const ke of [-EDGE, 5 + EDGE]) {
          const yc = (stringY(ke, Xm) + ymax) * S, win = 0.6 * sp * S, y0 = Math.trunc(yc - win), y1 = Math.trunc(yc + win);
          if (y0 < 1 || y1 >= Hh - 1 || badFrac(y0, y1 + 1, x0, x1 + 1) > 0.3) continue;
          const prof = rowProf(dy, y0, y1 + 1, x0, x1 + 1); let i = 0; for (let k = 1; k < prof.length; k++) if (prof[k] > prof[i]) i = k;
          if (prof[i] < 3 * (median(prof.slice()) + 1e-3) || prof[i] < 4) continue;
          const [Xd, Yd] = toPl(xc, y0 + i);
          targets.push([Xm, stringY(ke, Xm)]); found.push([Xd, Yd]);
        }
      }
    }
    if (targets.length < minInl) return fail;
    const imgPts = found.map(([x, y]) => applyH(this.H, x, y));
    const thr = Math.max(1.5, 0.003 * Math.hypot(Wi, Hi));
    const r = findH(targets, imgPts, thr, 300);
    if (!r) return fail;
    const nIn = r.inl.filter(Boolean).length, nf = new Set(targets.filter((_, i) => r.inl[i]).map(t => t[0].toFixed(6))).size;
    if (nIn < minInl || nf < 3) return fail;
    const Hn = r.H;
    for (const n of [frets[0], frets[frets.length >> 1], frets.at(-1)]) {       // slip guard: < 0.45 fret
      const o = [n - 1, n, n + 1].map(k => applyH(this.H, fretX(k), 0)), nw = applyH(Hn, fretX(n), 0);
      const fs = 0.5 * Math.hypot(o[2][0] - o[0][0], o[2][1] - o[0][1]);
      if (Math.hypot(nw[0] - o[1][0], nw[1] - o[1][1]) > 0.45 * Math.max(fs, 2)) return fail;
    }
    const Xm = fretX(frets[frets.length >> 1]);
    const so = [applyH(this.H, Xm, stringY(0, Xm)), applyH(this.H, Xm, stringY(5, Xm))], sn = [applyH(Hn, Xm, stringY(0, Xm)), applyH(Hn, Xm, stringY(5, Xm))];
    const ro = Math.hypot(so[1][0] - so[0][0], so[1][1] - so[0][1]), rn = Math.hypot(sn[1][0] - sn[0][0], sn[1][1] - sn[0][1]);
    if (!(rn / Math.max(ro, 1e-6) > 0.8 && rn / Math.max(ro, 1e-6) < 1.25) || Math.hypot(sn[0][0] - so[0][0], sn[0][1] - so[0][1]) > 0.5 * ro) return fail;
    return { ok: true, H: Hn, nIn, nf, fretHits };
  }
  update(gray, handMask) {
    if (!this.H) return false;
    this.predict(gray, handMask);
    const r = this.snap(gray, handMask);
    if (r.ok) { this.lost = 0; this.setPrev(gray); } else this.lost++;
    this.valid = this.lost <= OPT.maxLost; this.lastOk = r.ok;
    return r.ok;
  }
  // Inlay re-verification (PLAN.md P1): the snap guard stops one-frame slips, but over a long take the grid can still
  // creep a fret at a time. Score the dot pattern under the current grid at fret offsets -4..+4 and re-index on a clear win.
  inlayCheck(gray, handMask) {
    const Wi = gray.cols, Hi = gray.rows;
    const frets = this.visibleFrets(Wi, Hi).filter(n => n >= 1);
    if (frets.length < 5) return 0;
    const Xa = fretX(Math.max(frets[0] - 1, 0)), Xb = fretX(frets.at(-1)), ymax = 3.2 * spacing(Xb);
    const pa = applyH(this.H, Xa, 0), pb = applyH(this.H, Xb, 0);
    let S = clip(1.0 * Math.hypot(pb[0] - pa[0], pb[1] - pa[1]) / Math.max(Xb - Xa, 1e-6), 300, 4000);
    const W = Math.floor(Math.min(1600, (Xb - Xa) * S)) + 1; S = (W - 1) / (Xb - Xa);
    const Hh = Math.floor(2 * ymax * S) + 1;
    if (W < 20 || Hh < 12 || Hh > 1500) return 0;
    const M = mul3([S, 0, -Xa * S, 0, S, ymax * S, 0, 0, 1], inv3(this.H));
    const G = gauss32(Float32Array.from(warp(gray, M, W, Hh)), W, Hh, 1.0), valid = warp(onesLike(gray), M, W, Hh, true), hm = handMask ? warp(handMask, M, W, Hh, true) : null;
    const rowOf = (k, X) => (stringY(k, X) + ymax) * S;
    const cells = new Map();
    for (const n of frets) {
      const x0 = (fretX(n - 1) - Xa) * S, x1 = (fretX(n) - Xa) * S, cw = x1 - x0, Xm = 0.5 * (fretX(n - 1) + fretX(n));
      const X0 = Math.trunc(x0 + 0.2 * cw), X1 = Math.trunc(x1 - 0.2 * cw) + 1, Y0 = Math.max(0, Math.trunc(rowOf(-0.2, Xm))), Y1 = Math.min(Hh, Math.trunc(rowOf(5.2, Xm)) + 1);
      if (X0 < 0 || X1 > W || X1 - X0 < 2 || Y1 - Y0 < 6) continue;
      let ok = 0; for (let y = Y0; y < Y1; y++) for (let x = X0; x < X1; x++) { const i = y * W + x; if (valid[i] && !(hm && hm[i])) ok++; }
      if (ok < 0.9 * (X1 - X0) * (Y1 - Y0)) continue;
      const cwid = X1 - X0, ch = Y1 - Y0, cell = new Float32Array(cwid * ch);
      for (let y = 0; y < ch; y++) { const rv = []; for (let x = 0; x < cwid; x++) rv.push(G[(Y0 + y) * W + X0 + x]); const m = median(rv); for (let x = 0; x < cwid; x++) cell[y * cwid + x] = rv[x] - m; }
      const mad = median(Array.from(cell, Math.abs)) + 2, sp = spacing(Xm) * S, k = Math.max(1, Math.trunc(0.18 * Math.min(cw, sp)));
      const at = (q0, q1) => {
        const ra = Math.max(Math.trunc(rowOf(q0, Xm)) - Y0, 0), rb = Math.min(Math.trunc(rowOf(q1, Xm)) - Y0 + 1, ch);
        if (rb - ra < 1) return 0;
        const patch = cell.subarray(ra * cwid, rb * cwid), bl = rb - ra >= k && cwid >= k ? blur2(patch, cwid, rb - ra, k) : patch;
        let mx = 0; for (const v of bl) mx = Math.max(mx, Math.abs(v)); return mx / mad;
      };
      cells.set(n, [at(1.6, 3.4), Math.min(at(-0.2, 1.8), at(3.2, 5.2))]);
    }
    if (cells.size < 5) return 0;
    const ns = [...cells.keys()], z = ns.map(() => [0, 0]);
    for (let q = 0; q < 2; q++) {
      const v = ns.map(n => cells.get(n)[q]), m = median(v.slice()), sd = median(v.map(x => Math.abs(x - m))) * 1.4826 + 0.3;
      v.forEach((x, i) => { z[i][q] = (x - m) / sd; });
    }
    const score = o => { let s = 0; ns.forEach((n, i) => { const f = n + o, [c, d] = z[i]; if (DOUBLE.has(f)) s += d; else if (SINGLE.has(f)) s += c; else s -= 0.5 * (Math.max(c, 0) + Math.max(d, 0)); }); return s; };
    let best = 0, bs = score(0); const s0 = bs;
    for (let o = -4; o <= 4; o++) if (o && frets[0] + o >= 1 && frets.at(-1) + o <= N_FRETS) { const s = score(o); if (s > bs) { bs = s; best = o; } }
    if (!best || bs - s0 < 3.5 || bs < 4) { this.pendingShift = 0; return 0; }
    if (this.pendingShift !== best) { this.pendingShift = best; return 0; }   // act only when two checks in a row agree
    this.pendingShift = 0;
    const plane = [], img = [];
    for (const n of frets) { if (n + best < 0 || n + best > N_FRETS) continue; const X = fretX(n), Xn = fretX(n + best); for (const k of [0, 2.5, 5]) { plane.push([Xn, stringY(k, Xn)]); img.push(applyH(this.H, X, stringY(k, X))); } }
    const r = findH(plane, img);
    if (!r) return 0;
    this.H = r.H;
    return best;
  }
  // Reference view (fretboard.py register_to_reference, used here every few frames, not only after a loss): ORB features
  // on the board in the frame where the grid was set are matched to the current frame, H = M · H_ref. Frame-to-frame KLT
  // and snapping accumulate drift over a long take; this bounds it.
  setReference(gray) {
    this.dropReference();
    const poly = this.polygon(gray.cols, gray.rows, [0, ...this.visibleFrets(gray.cols, gray.rows).filter(n => n >= 1)], 1.5);
    if (!poly) return;
    const mask = drawMask(gray.cols, gray.rows, g => { g.beginPath(); poly.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y))); g.closePath(); g.fill(); });
    try {
      this.orb = this.orb || new cv.ORB(1500);
      const kp = new cv.KeyPointVector(), des = new cv.Mat();
      this.orb.detectAndCompute(gray, mask, kp, des);
      if (kp.size() >= 25) { const pts = []; for (let i = 0; i < kp.size(); i++) { const p = kp.get(i).pt; pts.push([p.x, p.y]); } this.ref = { H: this.H.slice(), pts, des }; }
      else des.delete();
      kp.delete();
    } catch (e) { console.warn('ORB unavailable:', e); } finally { mask.delete(); }
  }
  dropReference() { if (this.ref) { this.ref.des.delete(); this.ref = null; } }
  registerToRef(gray, handMask, minInl = 25) {
    if (!this.ref) return null;
    let mask = null; const kp = new cv.KeyPointVector(), des = new cv.Mat(), bf = new cv.BFMatcher(cv.NORM_HAMMING, false), mt = new cv.DMatchVectorVector();
    try {
      if (handMask) { mask = new cv.Mat(); cv.bitwise_not(handMask, mask); }
      this.orb.detectAndCompute(gray, mask || new cv.Mat(), kp, des);
      if (kp.size() < minInl) return null;
      bf.knnMatch(this.ref.des, des, mt, 2);
      const src = [], dst = [];
      for (let i = 0; i < mt.size(); i++) {
        const m = mt.get(i); if (m.size() < 2) continue;
        const a = m.get(0), b = m.get(1);
        if (a.distance < 0.8 * b.distance) { src.push(this.ref.pts[a.queryIdx]); const p = kp.get(a.trainIdx).pt; dst.push([p.x, p.y]); }
      }
      if (src.length < minInl) return null;
      const r = findH(src, dst, 4.0, 400);
      if (!r || r.inl.filter(Boolean).length < minInl) return null;
      const Hn = mul3(r.H, this.ref.H), X = fretX(5);
      const a0 = applyH(this.ref.H, X, stringY(0, X)), a1 = applyH(this.ref.H, X, stringY(5, X)), b0 = applyH(Hn, X, stringY(0, X)), b1 = applyH(Hn, X, stringY(5, X));
      const ra = Math.hypot(a1[0] - a0[0], a1[1] - a0[1]), rb = Math.hypot(b1[0] - b0[0], b1[1] - b0[1]);
      return rb / Math.max(ra, 1e-6) > 0.5 && rb / Math.max(ra, 1e-6) < 2 ? Hn : null;
    } catch (e) { return null; }
    finally { [kp, des, bf, mt].forEach(m => m.delete()); if (mask) mask.delete(); }
  }
  initFrom(gray, H, calibFrets = null) {
    this.H = H; this.setPrev(gray);
    const vis = this.visibleFrets(gray.cols, gray.rows).filter(n => n >= 1);
    if (calibFrets && vis.length) {          // grow outward from the calibrated frets
      const lo = Math.min(...calibFrets), hi = Math.max(...calibFrets);
      for (let step = 1; step < 30; step++) {
        const a = Math.max(1, lo - 2 * step), b = Math.min(N_FRETS, hi + 2 * step);
        this.fretRange = [Math.min(lo, a), Math.max(hi, b, 3)];
        this.snap(gray, null, 2);
        if (a <= vis[0] && b >= vis.at(-1)) break;
      }
      this.fretRange = [0, 99];
    }
    const r = this.snap(gray, null, 3);
    this.valid = true; this.lost = 0;
    return r;
  }
}

// ---------- automatic initialisation (autodetect.py) ----------
function houghLines(gray) {
  const g = new cv.Mat(), e = new cv.Mat(), L = new cv.Mat();
  cv.GaussianBlur(gray, g, new cv.Size(3, 3), 0);
  cv.Canny(g, e, 40, 120);
  cv.HoughLinesP(e, L, 1, Math.PI / 720, 50, 0.2 * Math.min(gray.rows, gray.cols), 10);
  const out = []; for (let i = 0; i < L.rows; i++) out.push([L.data32S[4 * i], L.data32S[4 * i + 1], L.data32S[4 * i + 2], L.data32S[4 * i + 3]]);
  g.delete(); e.delete(); L.delete(); return out;
}
const angOf = l => ((Math.atan2(l[3] - l[1], l[2] - l[0]) * 180 / Math.PI) % 180 + 180) % 180;
const angDiff = (a, d) => ((((a - d + 90) % 180) + 180) % 180) - 90;
function fitLine(pts) {
  let mx = 0, my = 0; for (const [x, y] of pts) { mx += x; my += y; } mx /= pts.length; my /= pts.length;
  let sxx = 0, sxy = 0, syy = 0; for (const [x, y] of pts) { sxx += (x - mx) ** 2; sxy += (x - mx) * (y - my); syy += (y - my) ** 2; }
  const th = 0.5 * Math.atan2(2 * sxy, sxx - syy); return [Math.cos(th), Math.sin(th), mx, my];
}
function clusterParallel(L, dom, tol = 5, gap = 6) {
  const th = dom * Math.PI / 180, u = [Math.cos(th), Math.sin(th)], n = [-Math.sin(th), Math.cos(th)];
  const S = L.filter(l => Math.abs(angDiff(angOf(l), dom)) < tol);
  if (!S.length) return [];
  const offs = S.map(l => 0.5 * (l[0] + l[2]) * n[0] + 0.5 * (l[1] + l[3]) * n[1]);
  const order = offs.map((_, i) => i).sort((a, b) => offs[a] - offs[b]);
  const groups = []; let cur = [order[0]];
  for (let i = 1; i < order.length; i++) { if (offs[order[i]] - offs[order[i - 1]] > gap) { groups.push(cur); cur = [order[i]]; } else cur.push(order[i]); }
  groups.push(cur);
  return groups.map(gi => {
    const seg = gi.map(i => S[i]), pts = seg.flatMap(l => [[l[0], l[1]], [l[2], l[3]]]);
    let [vx, vy, x0, y0] = fitLine(pts); if (vx * u[0] + vy * u[1] < 0) { vx = -vx; vy = -vy; }
    const proj = pts.map(([x, y]) => x * u[0] + y * u[1]);
    return { p: [x0, y0], d: [vx, vy], off: x0 * n[0] + y0 * n[1], len: seg.reduce((s, l) => s + Math.hypot(l[2] - l[0], l[3] - l[1]), 0), amin: Math.min(...proj), amax: Math.max(...proj) };
  });
}
function mergeClose(cl, minGap) {
  const out = [];
  for (const c of cl.slice().sort((a, b) => a.off - b.off)) { if (out.length && c.off - out.at(-1).off < minGap) { if (c.len > out.at(-1).len) out[out.length - 1] = c; } else out.push(c); }
  return out;
}
function pickStrings(cl) {
  if (cl.length < 6) return null;
  let best = null, bestS = 1e9;
  for (let i = 0; i + 6 <= cl.length; i++) {
    const sub = cl.slice(i, i + 6), g = sub.slice(1).map((c, k) => c.off - sub[k].off);
    if (Math.min(...g) <= 0) continue;
    const m = g.reduce((a, b) => a + b) / 5, sd = Math.sqrt(g.reduce((a, b) => a + (b - m) ** 2, 0) / 5);
    const s = sd / m - 0.02 * Math.log(sub.reduce((a, c) => a + c.len, 0) + 1);
    if (s < bestS) { best = sub; bestS = s; }
  }
  return best && bestS <= 0.35 ? best : null;
}
const linePointAt = (c, a, u) => { const t = (a - (c.p[0] * u[0] + c.p[1] * u[1])) / (c.d[0] * u[0] + c.d[1] * u[1]); return [c.p[0] + t * c.d[0], c.p[1] + t * c.d[1]]; };
function findPeaks(prof, minDist) {
  const pk = []; for (let i = 1; i < prof.length - 1; i++) if (prof[i] >= prof[i - 1] && prof[i] > prof[i + 1]) pk.push(i);
  pk.sort((a, b) => prof[b] - prof[a]);
  const keep = []; for (const i of pk) if (keep.every(j => Math.abs(i - j) >= minDist)) keep.push(i);
  return keep.sort((a, b) => a - b);
}
function regularise(pos, strength) {
  if (pos.length < 4) return null;
  const pmin = Math.min(...pos), pmax = Math.max(...pos);
  let best = null;
  for (let i = 0; i + 1 < pos.length; i++) {
    if (pos[i + 1] - pos[i] < 4) continue;
    const idx = new Map([[0, [pos[i], true]], [1, [pos[i + 1], true]]]);
    let score = strength[i] + strength[i + 1];
    for (const dir of [1, -1]) {
      let k = dir > 0 ? 1 : 0;
      const hist = [idx.get(k - dir)[0], idx.get(k)[0]];
      let miss = 0;
      for (;;) {
        const g1 = Math.abs(hist.at(-1) - hist.at(-2));
        const ratio = hist.length >= 3 ? clip(g1 / Math.max(Math.abs(hist.at(-2) - hist.at(-3)), 1e-6), 0.75, 1.45) : 1;
        const gp = g1 * ratio, pred = hist.at(-1) + dir * gp;
        if (pred < pmin - 0.5 * gp || pred > pmax + 0.5 * gp) break;
        let jn = 0; for (let j = 1; j < pos.length; j++) if (Math.abs(pos[j] - pred) < Math.abs(pos[jn] - pred)) jn = j;
        k += dir;
        if (Math.abs(pos[jn] - pred) < 0.2 * gp) { hist.push(pos[jn]); idx.set(k, [pos[jn], true]); score += strength[jn]; miss = 0; }
        else { if (++miss > 1) break; hist.push(pred); idx.set(k, [pred, false]); }
      }
    }
    const ks = [...idx.keys()].sort((a, b) => a - b);
    while (ks.length && !idx.get(ks[0])[1]) ks.shift();
    while (ks.length && !idx.get(ks.at(-1))[1]) ks.pop();
    if (ks.length < 4) continue;
    if (!best || score > best[0]) best = [score, ks, ks.map(k => idx.get(k)[0])];
  }
  if (!best) return null;
  return { ks: best[1].map(k => k - best[1][0]), pos: best[2] };
}
function findNut(strip, valid, W, Hst, pos, ks, row) {
  if (pos.length < 5) return null;
  const gaps = pos.slice(1).map((p, i) => (p - pos[i]) / Math.max(ks[i + 1] - ks[i], 1));
  const wide = gaps.at(-1) > gaps[0] ? 1 : 0;
  const r0 = Math.trunc(row(0.5)), r1 = Math.trunc(row(4.5)) + 1;
  const colmean = new Float64Array(W).fill(NaN);
  for (let x = 0; x < W; x++) { let s = 0, n = 0; for (let y = r0; y < Math.min(r1, Hst); y++) if (valid[y * W + x]) { s += strip[y * W + x]; n++; } if (n) colmean[x] = s / n; }
  const nanmed = (a, b) => median(Array.from(colmean.slice(a, b)));
  const order = wide ? [...Array(Math.min(3, pos.length)).keys()].map(i => pos.length - 1 - i) : [...Array(Math.min(3, pos.length)).keys()];
  for (const j of order) {
    const x = pos[j], g = gaps[clip(wide ? j - 1 : j, 0, gaps.length - 1)];
    const La = Math.trunc(x - 0.45 * g), Lb = Math.trunc(x - 0.15 * g), Ra = Math.trunc(x + 0.15 * g), Rb = Math.trunc(x + 0.45 * g);
    if (La < 0 || Rb >= W) continue;
    const mL = nanmed(La, Lb + 1), mR = nanmed(Ra, Rb + 1);
    if (Math.abs(mL - mR) / (0.5 * (mL + mR) + 10) > 0.35) return { end: wide, x, j };
  }
  const gl = wide ? gaps.at(-1) : gaps[0], gpv = wide ? gaps.at(-2) : gaps[1];
  const pred = (wide ? pos.at(-1) : pos[0]) + (wide ? 1 : -1) * gl * clip(gl / Math.max(gpv, 1e-6), 0.9, 1.4);
  const lo = Math.trunc(pred - 0.35 * gl), hi = Math.trunc(pred + 0.35 * gl);
  if (lo < 2 || hi > W - 3) return null;
  const seg = Array.from(colmean.slice(lo - 2, hi + 3));
  if (seg.some(Number.isNaN)) return null;
  const sm = seg.map((_, i) => ((seg[i - 1] ?? 0) + seg[i] + (seg[i + 1] ?? 0)) / 3);
  const grad = sm.slice(1).map((v, i) => Math.abs(v - sm[i]));
  let jg = 1; for (let i = 2; i < grad.length - 1; i++) if (grad[i] > grad[jg]) jg = i;
  const inside = [];
  for (let i = 0; i + 1 < pos.length; i++) {
    const xa = Math.trunc(pos[i] + 0.25 * (pos[i + 1] - pos[i])), xb = Math.trunc(pos[i + 1] - 0.25 * (pos[i + 1] - pos[i]));
    if (xb - xa > 2) { const c = Array.from(colmean.slice(xa, xb)); if (!c.some(Number.isNaN)) inside.push(Math.max(...c.slice(1).map((v, k) => Math.abs(v - c[k])))); }
  }
  const ref = inside.length ? median(inside) : 1;
  if (grad[jg] < 4 * Math.max(ref, 1)) return null;
  return { end: wide, x: lo - 2 + jg + 0.5, j: null };
}
let clahe = null;
function autoDetect(gray, handMask) {
  let r = autoDetectOnce(gray, handMask);
  if (r.H) return r;
  try {
    if (!clahe) clahe = new cv.CLAHE(2.5, new cv.Size(8, 8));
    const g2 = new cv.Mat(); clahe.apply(gray, g2);
    const r2 = autoDetectOnce(g2, handMask); g2.delete();
    if (r2.H) { r2.info.contrast = 'clahe'; return r2; }
  } catch (e) { /* CLAHE missing in this build */ }
  return r;
}
function autoDetectOnce(gray, handMask, minScore = 4, minMargin = 2) {
  const info = {}, fail = why => (info.fail = why, { H: null, info });
  const L = houghLines(gray);
  if (L.length < 3) return fail('no lines');
  const ang = L.map(angOf), ln = L.map(l => Math.hypot(l[2] - l[0], l[3] - l[1]));
  const hist = new Float64Array(180); ang.forEach((a, i) => { hist[Math.trunc(a) % 180] += ln[i]; });
  const hs = new Float64Array(180); for (let i = 0; i < 180; i++) for (let d = -2; d <= 2; d++) hs[i] += hist[(i + d + 180) % 180];
  const hmax = Math.max(...hs), hs2 = Float64Array.from(hs), cands = [];
  for (let q = 0; q < 4; q++) { let a = 0; for (let i = 1; i < 180; i++) if (hs2[i] > hs2[a]) a = i; if (hs2[a] <= 0) break; cands.push(a); for (let d = -8; d <= 8; d++) hs2[(a + d + 180) % 180] = 0; }
  const found = [];
  for (let dom of cands) {
    let sw = 0, sd = 0; ang.forEach((a, i) => { const df = angDiff(a, dom); if (Math.abs(df) < 4) { sw += ln[i]; sd += ln[i] * df; } });
    if (sw) dom = ((dom + sd / sw) % 180 + 180) % 180;
    let cl = clusterParallel(L, dom);
    if (cl.length < 2) continue;
    const offs = cl.map(c => c.off).sort((a, b) => a - b), gaps = offs.slice(1).map((o, i) => o - offs[i]);
    cl = mergeClose(cl, Math.max(4, gaps.length ? 0.35 * median(gaps) : 4));
    const strings = pickStrings(cl), sc = (strings ? 2 : 0) + hs[Math.round(dom) % 180] / hmax;
    found.push({ sc, dom, cl, strings });
  }
  if (!found.length) return fail('too few parallel lines');
  // try each candidate neck direction (captions / overlays give strong straight lines too); keep the best grid
  found.sort((a, b) => b.sc - a.sc);
  let out = null;
  for (const c of found.slice(0, 3)) {
    const r = detectDir(gray, handMask, c, minScore, minMargin);
    r.q = r.H ? 100 + (r.info.mode === 'strings' ? 5 : 0) + (r.info.wires || 0) + (r.info.indexFrom === 'inlays' ? 5 : 0) + Math.min(10, r.info.inlayScore || 0) / 2 : 0;
    if (!out || r.q > out.q) out = r;
  }
  return out;
}
function detectDir(gray, handMask, { dom, cl, strings }, minScore, minMargin) {
  const info = {}, fail = why => (info.fail = why, { H: null, info });
  info.angle = dom;
  const th = dom * Math.PI / 180, u = [Math.cos(th), Math.sin(th)], nrm = [-Math.sin(th), Math.cos(th)];
  const Wi = gray.cols, Hi = gray.rows;
  let lines, kcoords, mode;
  if (strings) { mode = 'strings'; lines = strings; kcoords = [0, 1, 2, 3, 4, 5]; }
  else {
    const maxLen = Math.max(...cl.map(c => c.len)), big = cl.filter(c => c.len > 0.25 * maxLen);
    if (big.length < 2) return fail('no strings/edges');
    const gf = gauss32(Float32Array.from(gray.data), Wi, Hi, 1.1);
    const sideCon = c => {
      const vals = [];
      for (let i = 0; i < 40; i++) {
        const a = c.amin + (c.amax - c.amin) * i / 39, p = linePointAt(c, a, u), q1 = [p[0] + 4 * nrm[0], p[1] + 4 * nrm[1]], q2 = [p[0] - 4 * nrm[0], p[1] - 4 * nrm[1]];
        if ([q1, q2].every(([x, y]) => x >= 0 && x < Wi && y >= 0 && y < Hi)) vals.push(gf[Math.trunc(q1[1]) * Wi + Math.trunc(q1[0])] - gf[Math.trunc(q2[1]) * Wi + Math.trunc(q2[0])]);
      }
      return vals.length ? median(vals) : 0;
    };
    const con = big.map(sideCon), diag = Math.hypot(Wi, Hi);
    let pair = null, bv = 0;
    for (let i = 0; i < big.length; i++) for (let j = i + 1; j < big.length; j++) {
      if (big[j].off - big[i].off < 0.03 * diag) continue;
      if (con[i] - con[j] > bv) { bv = con[i] - con[j]; pair = [big[i], big[j]]; }
    }
    lines = pair || [big[0], big.at(-1)]; mode = 'edges'; kcoords = [-EDGE, 5 + EDGE];
  }
  info.mode = mode;
  let a0 = Math.min(...lines.map(c => c.amin)), a1 = Math.max(...lines.map(c => c.amax));
  const cp = [[0, 0], [Wi, 0], [Wi, Hi], [0, Hi]].map(([x, y]) => x * u[0] + y * u[1]);
  [a0, a1] = [Math.max(a0 - 0.15 * (a1 - a0), Math.min(...cp)), Math.min(a1 + 0.15 * (a1 - a0), Math.max(...cp))];
  const first = lines[0], last = lines.at(-1);
  const P = [linePointAt(first, a0, u), linePointAt(first, a1, u), linePointAt(last, a1, u), linePointAt(last, a0, u)];
  const Wst = Math.trunc(a1 - a0), span = (Math.hypot(P[0][0] - P[3][0], P[0][1] - P[3][1]) + Math.hypot(P[1][0] - P[2][0], P[1][1] - P[2][1])) / 2;
  const Hst = Math.max(40, Math.trunc(span));
  if (Wst < 40 || Wst > 4000 || Hst > 2000) return fail('bad strip');
  const kA = kcoords[0], kB = kcoords.at(-1);
  let row = k => (k - kA) / (kB - kA) * (Hst - 1);
  const Mr = findH(P, [[0, 0], [Wst - 1, 0], [Wst - 1, Hst - 1], [0, Hst - 1]]); if (!Mr) return fail('strip homography');
  const M = Mr.H;
  let strip = Float32Array.from(warp(gray, M, Wst, Hst));
  let valid = warp(onesLike(gray), M, Wst, Hst, true);
  if (handMask) { const hm = warp(handMask, M, Wst, Hst, true); for (let i = 0; i < valid.length; i++) if (hm[i]) valid[i] = 0; }
  const bands = [0, 1, 2, 3, 4].map(k => [k + 0.2, k + 0.8]);
  const bandProfiles = (D, V) => bands.map(([b0, b1]) => {
    const r0 = Math.trunc(row(b0)), r1 = Math.trunc(row(b1)) + 1, out = new Float64Array(Wst);
    for (let x = 0; x < Wst; x++) { let num = 0, den = 0; for (let y = r0; y < r1; y++) if (V[y * Wst + x]) { num += D[y * Wst + x]; den++; } out[x] = den > 0.5 * (r1 - r0) ? num / Math.max(den, 1) : NaN; }
    return out;
  });
  // shear search: fret wires vertical in the strip (varying linearly along the neck for perspective)
  let dxx = sobel32(gauss32(strip, Wst, Hst, 1.2), Wst, Hst, 2, 0);
  const P0 = bandProfiles(dxx, valid).map(p => {      // fill NaN gaps linearly (np.interp over the valid samples)
    const ok = []; p.forEach((v, i) => { if (!Number.isNaN(v)) ok.push(i); });
    if (ok.length <= 2) return null;
    const f = new Float64Array(p.length); let j = 0;
    for (let i = 0; i < p.length; i++) {
      if (i <= ok[0]) f[i] = p[ok[0]]; else if (i >= ok.at(-1)) f[i] = p[ok.at(-1)];
      else { while (ok[j + 1] < i) j++; const a = ok[j], b = ok[j + 1]; f[i] = p[a] + (p[b] - p[a]) * (i - a) / (b - a); }
    }
    return f;
  });
  const yb = bands.map(([b0, b1]) => 0.5 * (row(b0) + row(b1))), ym = 0.5 * (Hst - 1), xm = 0.5 * Wst;
  let bestSh = [-1, 0, 0];
  const med = new Float32Array(Wst), vals = new Float64Array(5), kbox = Math.max(3, Math.trunc(Wst / 20));
  for (let a = 0; a < 9; a++) {
    const k1 = (-2 + 4 * a / 8) / Wst;
    for (let b = 0; b < 65; b++) {
      const k0 = -1.6 + 3.2 * b / 64;
      for (let x = 0; x < Wst; x++) {
        const kk = k0 + k1 * (x - xm); let n = 0;
        for (let q = 0; q < 5; q++) {
          const pb = P0[q]; if (!pb) continue;
          const src = x + kk * (yb[q] - ym); if (src < 0 || src > Wst - 1) continue;
          const i0 = Math.floor(src), fr = src - i0; vals[n++] = i0 + 1 < Wst ? pb[i0] * (1 - fr) + pb[i0 + 1] * fr : pb[i0];
        }
        if (!n) { med[x] = 0; continue; }
        const s = Array.from(vals.subarray(0, n)).sort((p, q) => p - q); med[x] = n % 2 ? s[n >> 1] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
      }
      const bl = blur1(med, kbox); let sc = 0;
      for (let x = 0; x < Wst; x++) { const hp = med[x] - bl[x]; if (hp > 0) sc += hp * hp; }
      if (sc > bestSh[0]) bestSh = [sc, k0, k1];
    }
  }
  const [, k0, k1] = bestSh;
  info.shear = [+k0.toFixed(3), +(k1 * Wst).toFixed(3)];
  { // remap
    const s2 = new Float32Array(Wst * Hst), v2 = new Uint8Array(Wst * Hst);
    for (let y = 0; y < Hst; y++) for (let x = 0; x < Wst; x++) {
      const src = x + (k0 + k1 * (x - xm)) * (y - ym), i0 = Math.floor(src), fr = src - i0;
      if (i0 >= 0 && i0 + 1 < Wst) s2[y * Wst + x] = strip[y * Wst + i0] * (1 - fr) + strip[y * Wst + i0 + 1] * fr;
      const ni = Math.round(src); if (ni >= 0 && ni < Wst) v2[y * Wst + x] = valid[y * Wst + ni];
    }
    strip = s2; valid = v2;
  }
  dxx = sobel32(gauss32(strip, Wst, Hst, 1.2), Wst, Hst, 2, 0);
  const profs = bandProfiles(dxx, valid);
  let prof = new Float32Array(Wst);
  for (let x = 0; x < Wst; x++) { const m = median(profs.map(p => p[x])); prof[x] = Number.isNaN(m) ? 0 : m; }
  prof = gauss1(prof, 1.0);
  const base = blur1(prof, Math.max(3, Math.trunc(Wst / 15))), rel = prof.map((v, i) => v / (base[i] + 1e-3));
  const pk = findPeaks(rel, Math.max(3, Math.trunc(Wst / 80))).filter(i => rel[i] > 1.4);
  if (pk.length < 4) return fail(`few fret peaks (${pk.length})`);
  const rg = regularise(pk, pk.map(i => rel[i]));
  if (!rg || rg.ks.length < 4) return fail('no regular fret run');
  let { ks, pos } = rg;
  info.wires = ks.length;
  // across-neck extent check (not in the Python version): fret-wire ridges exist only on the board. If the picked
  // "strings" also took in body/caption lines, the rows where the wires stop mark the board edges: re-map the rows.
  {
    const ridge = new Float64Array(Hst), basev = new Float64Array(Hst);
    for (let y = 0; y < Hst; y++) {
      const on = [], off = [];
      for (let j = 0; j < pos.length; j++) {
        const x = Math.round(pos[j]); if (x < 1 || x >= Wst - 1) continue;
        on.push(Math.max(dxx[y * Wst + x - 1], dxx[y * Wst + x], dxx[y * Wst + x + 1]));
        if (j + 1 < pos.length) off.push(dxx[y * Wst + Math.round(0.5 * (pos[j] + pos[j + 1]))]);
      }
      ridge[y] = median(on); basev[y] = median(off);
    }
    const ratio = gauss1(Float32Array.from(ridge, (v, y) => (v - basev[y]) / (basev[y] + 2)), 2);
    const top = Array.from(ratio).sort((a, b) => b - a)[Math.floor(0.1 * Hst)], thr = 0.35 * top;
    let best = [0, -1], cur = -1;
    for (let y = 0; y <= Hst; y++) {
      if (y < Hst && ratio[y] > thr) { if (cur < 0) cur = y; }
      else if (cur >= 0) { if (y - 1 - cur > best[1] - best[0]) best = [cur, y - 1]; cur = -1; }
    }
    const [y0, y1] = best, len = y1 - y0;
    info.boardRows = [y0, y1, Hst];
    if (top > 0.5 && len > 0.3 * Hst && len < 0.85 * Hst) {
      const e0 = -EDGE, e1 = 5 + EDGE;
      row = k => y0 + (k - e0) / (e1 - e0) * len;
      info.rowsRefined = true;
    }
  }
  // inlays
  const G2 = gauss32(strip, Wst, Hst, 1.0), sp = row(1) - row(0), cells = [];
  for (let j = 0; j + 1 < pos.length; j++) {
    const xa = pos[j], xbb = pos[j + 1], cw = xbb - xa;
    const r = Math.max(1.5, 0.18 * (mode === 'edges' ? Math.min(cw, sp * (5 / Math.max(kB - kA, 1e-6))) : Math.min(cw, sp)));
    const X0 = Math.trunc(xa + 0.2 * cw), X1 = Math.trunc(xbb - 0.2 * cw) + 1, Y0 = Math.max(0, Math.trunc(row(0))), Y1 = Math.min(Hst, Math.trunc(row(5)) + 1);
    let vm = 0; for (let y = Y0; y < Y1; y++) for (let x = X0; x < X1; x++) vm += valid[y * Wst + x] ? 1 : 0;
    if (X1 - X0 < 2 || Y1 - Y0 < 4 || !(vm / ((X1 - X0) * (Y1 - Y0)) > 0.9)) { cells.push([NaN, NaN]); continue; }
    const cwid = X1 - X0, ch = Y1 - Y0, cell = new Float32Array(cwid * ch);
    for (let y = 0; y < ch; y++) {
      const rowv = []; for (let x = 0; x < cwid; x++) rowv.push(G2[(Y0 + y) * Wst + X0 + x]);
      const m = median(rowv); for (let x = 0; x < cwid; x++) cell[y * cwid + x] = rowv[x] - m;
    }
    const mad = median(Array.from(cell, Math.abs)) + 2;
    const at = (q0, q1) => {
      const ra = Math.max(Math.trunc(row(q0)) - Y0, 0), rb = Math.min(Math.trunc(row(q1)) - Y0 + 1, ch);
      if (rb - ra < 1) return 0;
      const ph = rb - ra, patch = cell.subarray(ra * cwid, rb * cwid), k = Math.max(1, Math.trunc(r));
      const bl = ph >= k && cwid >= k ? blur2(patch, cwid, ph, k) : patch;
      let mx = 0; for (const v of bl) mx = Math.max(mx, Math.abs(v)); return mx / mad;
    };
    cells.push([at(1.6, 3.4), Math.min(at(0, 1.8), at(3.2, 5))]);
  }
  const cz = cells.map(c => c.slice());
  for (let q = 0; q < 2; q++) {
    const v = cells.map(c => c[q]), m = median(v), sd = median(v.map(x => Math.abs(x - m))) * 1.4826 + 0.3;
    cz.forEach((c, i) => { const z = (v[i] - m) / sd; c[q] = Number.isFinite(z) ? z : 0; });
  }
  const gaps = pos.slice(1).map((p, i) => (p - pos[i]) / Math.max(ks[i + 1] - ks[i], 1));
  let slope = 0;
  if (gaps.length >= 3) { const lg = gaps.map(g => Math.log(Math.max(g, 1e-3))), n = lg.length, mx = (n - 1) / 2, my = lg.reduce((a, b) => a + b) / n; let sxy = 0, sxx = 0; lg.forEach((y, i) => { sxy += (i - mx) * (y - my); sxx += (i - mx) ** 2; }); slope = sxy / sxx; }
  const hyps = [];
  for (const s of [1, -1]) for (let n0 = 0; n0 <= N_FRETS; n0++) {
    const nums = ks.map(k => n0 + s * k);
    if (Math.min(...nums) < 0 || Math.max(...nums) > N_FRETS) continue;
    let sc = 0;
    for (let j = 0; j + 1 < pos.length; j++) {
      const f = Math.max(nums[j], nums[j + 1]), [c, d] = cz[j];
      if (DOUBLE.has(f)) sc += d; else if (SINGLE.has(f)) sc += c; else sc -= 0.5 * (Math.max(c, 0) + Math.max(d, 0));
    }
    sc += 3 * clip(-s * slope / 0.0578, -2, 2);
    hyps.push([sc, s, n0]);
  }
  hyps.sort((a, b) => b[0] - a[0]);
  if (!hyps.length) return fail('no fret numbering');
  let [sc, s, n0] = hyps[0];
  const margin = sc - (hyps[1] ? hyps[1][0] : -1e9);
  info.inlayScore = +sc.toFixed(2); info.inlayMargin = +margin.toFixed(2);
  let nums = ks.map(k => n0 + s * k), nutX = null;
  if (sc < minScore || margin < minMargin) {
    const nut = findNut(strip, valid, Wst, Hst, pos, ks, row);
    if (!nut) return fail(`inlay pattern not found (score ${sc.toFixed(2)}) and no nut`);
    nutX = nut.x;
    if (nut.j != null) {
      const keep = pos.map((_, i) => (nut.end === 1 ? i < nut.j : i > nut.j));
      pos = pos.filter((_, i) => keep[i]); ks = ks.filter((_, i) => keep[i]); const m0 = Math.min(...ks); ks = ks.map(k => k - m0);
    }
    if (nut.end === 1) { s = -1; n0 = ks.at(-1) + 1; } else { s = 1; n0 = 1; }
    nums = ks.map(k => n0 + s * k);
    info.indexFrom = 'nut';
    if (Math.max(...nums) > N_FRETS + 2) return fail('nut gives too many frets');
  } else info.indexFrom = 'inlays';
  info.frets = [Math.min(...nums), Math.max(...nums)];
  const Minv = inv3(M), plane = [], img = [];
  const wires = pos.map((x, j) => [x, nums[j]]); if (nutX != null) wires.push([nutX, 0]);
  for (const [x, n] of wires) {
    const X = fretX(n);
    for (const k of kcoords) {
      const y = row(k), xs = x + (k0 + k1 * (x - xm)) * (y - ym);
      plane.push([X, stringY(k, X)]); img.push(applyH(Minv, xs, y));
    }
  }
  let r = findH(plane, img);
  if (r) {
    const res = plane.map((p, i) => { const q = applyH(r.H, ...p); return Math.hypot(q[0] - img[i][0], q[1] - img[i][1]); });
    const thr = Math.max(3, 2.5 * median(res.slice())), keep = res.map(v => v < thr), nk = keep.filter(Boolean).length;
    if (nk >= 8 && nk < res.length) r = findH(plane.filter((_, i) => keep[i]), img.filter((_, i) => keep[i])) || r;
  }
  if (!r) return fail('homography');
  return { H: r.H, info };
}

// ---------- hands (hands.py) ----------
class OneEuro {
  constructor(minCutoff = 3, beta = 0.05, dCutoff = 1) { Object.assign(this, { minCutoff, beta, dCutoff, x: null, dx: null, t: null }); }
  static a(c, dt) { const r = 2 * Math.PI * c * dt; return r / (r + 1); }
  filter(x, t) {
    if (!this.x || this.x.length !== x.length) { this.x = x.slice(); this.dx = x.map(() => 0); this.t = t; return x; }
    const dt = Math.max(t - this.t, 1e-3), ad = OneEuro.a(this.dCutoff, dt);
    this.x = x.map((v, i) => {
      this.dx[i] = ad * (v - this.x[i]) / dt + (1 - ad) * this.dx[i];
      const a = OneEuro.a(this.minCutoff + this.beta * Math.abs(this.dx[i]), dt);
      return a * v + (1 - a) * this.x[i];
    });
    this.t = t; return this.x;
  }
  reset() { this.x = null; }
}
let lastTs = -1;
const canvasA = { c: null, g: null }, canvasB = { c: null, g: null };
function detectHands(bmp, t) {
  const W = bmp.width, H = bmp.height;
  if (!canvasA.c || canvasA.c.width !== W || canvasA.c.height !== H) { canvasA.c = new OffscreenCanvas(W, H); canvasA.g = canvasA.c.getContext('2d', { willReadFrequently: true }); }
  canvasA.g.drawImage(bmp, 0, 0);
  let ts = Math.round(t * 1000); if (ts <= lastTs) ts = lastTs + 1; lastTs = ts;
  let res = hands.detectForVideo(canvasA.c, ts), flipped = false;
  if (!res.landmarks?.length) {      // mirrored retry: the palm detector often misses a fretting hand seen from behind
    if (!canvasB.c || canvasB.c.width !== W || canvasB.c.height !== H) { canvasB.c = new OffscreenCanvas(W, H); canvasB.g = canvasB.c.getContext('2d'); }
    canvasB.g.setTransform(-1, 0, 0, 1, W, 0); canvasB.g.drawImage(bmp, 0, 0); canvasB.g.setTransform(1, 0, 0, 1, 0, 0);
    res = handsImg.detect(canvasB.c); flipped = true;
  }
  return (res.landmarks || []).map((lms, i) => {
    const pts = lms.map(l => [(flipped ? 1 - l.x : l.x) * W, l.y * H]);
    let label = null, score = 1;
    const c = (res.handedness || res.handednesses || [])[i]?.[0];
    if (c) { label = c.categoryName; score = c.score; if (flipped === false && (label === 'Left' || label === 'Right')) label = label === 'Left' ? 'Right' : 'Left'; }
    return { pts, label, score };
  });
}
function selectFretting(hs, fb) {
  let best = null, bs = -1e9;
  for (const hd of hs) {
    let s = hd.label === OPT.wantLabel ? 0.5 * hd.score : 0;
    if (fb && fb.H) {
      const Hi = inv3(fb.H);
      for (let i = 0; i < 4; i++) {
        const t = hd.pts[TIPS[i]], d = hd.pts[DIPS[i]], [X, Y] = applyH(Hi, t[0] + 0.25 * (d[0] - t[0]), t[1] + 0.25 * (d[1] - t[1])), sc = planeToString(X, Y);
        if (sc > -1 && sc < 6 && X > -0.02 && X < 0.8) s += 1;
      }
      const w0 = applyH(Hi, ...hd.pts[0]), w9 = applyH(Hi, ...hd.pts[9]);
      s -= 2 * clip(0.5 * (w0[0] + w9[0]), -1, 2);
    }
    if (s > bs) { best = hd; bs = s; }
  }
  return best;
}

// ---------- per-frame driver (track.py main loop) ----------
let fb = new Fretboard(), filt = new OneEuro(), frameI = 0, manual = false, stringsResolved = false, sideVotes = [];
function resetState() { if (fb.prev) fb.prev.delete(); fb = new Fretboard(); filt = new OneEuro(); frameI = 0; manual = false; stringsResolved = false; sideVotes = []; lastTs = -1; }
let lastGray = null;
function frame({ t, bitmap }) {
  const t0 = performance.now(), W = bitmap.width, H = bitmap.height;
  const hs = detectHands(bitmap, t), tm = [performance.now() - t0];
  const id = canvasA.g.getImageData(0, 0, W, H), rgba = cv.matFromImageData(id), gray = new cv.Mat();
  cv.cvtColor(rgba, gray, cv.COLOR_RGBA2GRAY); rgba.delete(); bitmap.close();
  if (lastGray) lastGray.delete(); lastGray = gray.clone();
  frameI++;
  const handMask = hs.length ? drawMask(W, H, g => hs.forEach(h => handMaskFill(g, h.pts))) : null;
  let snapOk = false, justInit = false, flipNow = null;
  if (fb.H && !fb.valid && !manual) { fb.H = null; fb.dropReference(); stringsResolved = false; sideVotes = []; }
  if (!fb.H && !manual && frameI % OPT.autoEvery === 1) {
    const r = autoDetect(gray, handMask);
    if (r.H) { const s = fb.initFrom(gray, r.H); fb.setReference(gray); snapOk = s.ok; justInit = true; status(`Fretboard found (${r.info.mode}, frets ${r.info.frets?.join('–')}, from ${r.info.indexFrom}${r.info.rowsRefined ? ', edges refined' : ''}).`); }
  }
  if (fb.H && !justInit) {
    if (fb.ref && frameI % OPT.regEvery === 0) {         // re-anchor to the reference view, then refine by snapping
      const Hr = fb.registerToRef(gray, handMask);
      if (Hr) { fb.H = Hr; fb.setPrev(gray); }
    }
    snapOk = fb.update(gray, handMask);
    if (manual) fb.valid = true;
    if (fb.valid && frameI % OPT.inlayEvery === 0) {
      const o = fb.inlayCheck(gray, handMask);
      if (o) { fb.setReference(gray); console.log(`grid re-indexed by ${o} frets at t=${t.toFixed(2)}`); }
    }
  }
  tm.push(performance.now() - t0);
  const grid = fb.H && fb.valid;
  const hand = selectFretting(hs, grid ? fb : null);
  const tips = [];
  let unresolved = false;
  tm.push(performance.now() - t0);
  if (hand) {
    const flat = filt.filter(hand.pts.flat(), t), pts = Array.from({ length: 21 }, (_, i) => [flat[2 * i], flat[2 * i + 1]]);
    hand.pts = pts;
    if (grid) {
      if (!stringsResolved) {
        const w = fb.toPlane(...pts[0]), ws = planeToString(w[0], w[1]);
        if (ws < -0.5 || ws > 5.5) sideVotes.push(ws > 5.5);
        if (sideVotes.length >= 15) {
          const flip = sideVotes.filter(Boolean).length > sideVotes.length / 2;
          if (flip) { fb.H = mul3(fb.H, [1, 0, 0, 0, -1, 0, 0, 0, 1]); if (fb.ref) fb.ref.H = mul3(fb.ref.H, [1, 0, 0, 0, -1, 0, 0, 0, 1]); }
          stringsResolved = true; flipNow = flip;
        }
      }
      unresolved = !stringsResolved;
      const Hl = inv3(liftedH(fb.H, W, H, OPT.tipHeight));
      for (let i = 0; i < 4; i++) {
        const tp = pts[TIPS[i]], dp = pts[DIPS[i]], cx = tp[0] + OPT.alpha * (dp[0] - tp[0]), cy = tp[1] + OPT.alpha * (dp[1] - tp[1]);
        const [X, Y] = applyH(Hl, cx, cy), sf = planeToString(X, Y), ff = xToFretFloat(X);
        if (sf >= -0.7 && sf <= 5.7 && ff >= -0.6 && ff <= N_FRETS + 0.5) tips.push({ i, x: cx, y: cy, sf, ff });
      }
    }
  } else filt.reset();
  if (handMask) handMask.delete();
  gray.delete();
  const state = !fb.H ? 'searching' : manual ? 'calibrated' : snapOk ? 'tracked' : fb.valid ? 'coast' : 'lost';
  postMessage({ type: 'result', t, ms: Math.round(performance.now() - t0), state, H: grid ? fb.H : null, frets: grid ? fb.visibleFrets(W, H) : [],
    hand: hand ? hand.pts.map(p => [+p[0].toFixed(1), +p[1].toFixed(1)]) : null, tips, unresolved, flipNow, w: W, h: H, nh: hs.length, tm: tm.map(Math.round) });
}
function calib({ points, frets }) {
  const H = homographyFromCalibration(points, frets);
  if (!H || !lastGray) { postMessage({ type: 'calibrated', ok: false }); return; }
  fb.H = null; const r = fb.initFrom(lastGray, H, frets); fb.setReference(lastGray);
  manual = true; stringsResolved = true; sideVotes = [];
  postMessage({ type: 'calibrated', ok: !!r.ok });
}

let ready = null;
onmessage = async e => {
  const m = e.data;
  try {
    if (m.type === 'init') { ready = ready || init(m.modelUrl); await ready; postMessage({ type: 'ready' }); }
    else if (m.type === 'frame') { await ready; frame(m); }
    else if (m.type === 'calib') { await ready; calib(m); }
    else if (m.type === 'reset') resetState();
  } catch (err) {
    console.error(err);
    if (m.bitmap) try { m.bitmap.close(); } catch { /* already closed */ }
    // a bad frame must not stop the run: answer it with an empty result
    if (m.type === 'frame' && cv && hands) postMessage({ type: 'result', t: m.t, state: 'error', error: String(err && err.message || err), tips: [], H: null });
    else postMessage({ type: 'error', message: String(err && err.message || err) });
  }
};
