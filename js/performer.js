// Performer: turns timeline targets into hand motion (springs + placement + IK) for both hands,
// and computes where each string is pressed / plucked. Shared by the page (main.js) and the
// offline self-check (tools/selfcheck.mjs), so what is measured is what is rendered.
import * as THREE from 'three';
import { contactX, fretX, fbWidth, stringY, stringZ, FRET_H } from './guitar.js';
import { sample, FINGER_KEYS, PLUCK_X, PRESS_Y as PRESS_Y_ } from './timeline.js';
import { Spring3 } from './hand.js';
import { STRUM, STRUM_Z0 } from './strumhand.js';

const D2R = Math.PI / 180;
// fretting-hand placement, fitted offline (node tools/selfcheck.mjs --fit-left) on fingertip error AND
// posture. The old fit (alpha 12, ky 0.025) had the same error but flat knuckles (MCP ~0 deg) and hooked
// fingers (PIP ~89 deg); raising the knuckles gives MCP ~33, PIP ~62 and a more upright fingertip.
export const LEFT = {
  alpha: 16 * D2R,                    // finger direction tilt out of the fingerboard plane
  ky: 0.044,                          // knuckle height above the fingerboard (refit with WRIST on)
  kz: focus => 0.040 - 0.006 * focus, // knuckle distance beyond the treble edge, by string focus
  // wrist: the palm frame turns about the knuckle line instead of staying rigid. Ranges come from MediaPipe
  // hand poses on a realistic reference clip (frets 12-22, see PLAN.md "Wrist data"): palm-direction yaw
  // median +2 deg (p10..p90 -8..+5), roll about the finger axis median -2 deg (p10..p90 -10..+9).
  yaw: fret => Math.max(-3, Math.min(5, 0.35 * (fret - 5))) * D2R,      // fingers lean toward the bridge up the neck
  roll: focus => Math.max(-8, Math.min(8, WRIST.rollK * (focus - 1.5))) * D2R, // hand rolls as the fingers reach the bass strings
};
export const WRIST = { on: true, rollK: -2.5 };   // deg of roll per string of reach (sign picked by the self-check: -2.5 and 0 tie, +4 is worse)
// thumb pad on the back of the neck: dx behind the knuckle line (toward the nut), u = position round the
// neck back (-1 treble edge .. 0 centre .. 1 bass edge), shifted with the string focus. Fitted offline for
// reachability (node tools/fit-thumb.mjs).
// The old values (dx -0.02, u0 -0.9, uk 0.18, depth 0.02) missed by ~13 mm on average with the thumb
// pinned at its abduction limit; these reach to ~2 mm (95th percentile ~6 mm).
export const THUMB = { dx: -0.03, u0: -0.95, uk: 0, uMin: -0.95, uMax: 0.8, depth: 0.004 };
// picking-hand placement (fitted offline, see tools/selfcheck.mjs --fit-right)
export const RIGHT = { beta: 35 * D2R, yaw: 25 * D2R, dx: 0.0, ky: 0.06, kz: -0.015 };
export const rightAxes = () => ({
  l: new THREE.Vector3(-Math.cos(RIGHT.yaw), 0, Math.sin(RIGHT.yaw)),
  f: new THREE.Vector3(-Math.sin(RIGHT.yaw) * Math.cos(RIGHT.beta), -Math.sin(RIGHT.beta), -Math.cos(RIGHT.yaw) * Math.cos(RIGHT.beta)),
});
const ELBOW_R = new THREE.Vector3(0.80, 0.10, 0.20);

const OMEGA = { finger: 55, strike: 95, hand: 34, thumb: 18, right: 60, pick: 70 };

export class Performer {
  constructor(left, right) {
    this.left = left; this.right = right;
    this.events = [];
    this.handSpring = new Spring3(OMEGA.hand, new THREE.Vector3(contactX(5), 2.5, 0));
    this.springs = {};
    for (const k of [...FINGER_KEYS, 'thumb']) this.springs[k] = new Spring3(k === 'thumb' ? OMEGA.thumb : OMEGA.finger);
    this.rsprings = {};
    for (const k of ['thumb', 'index', 'middle', 'ring', 'pick']) this.rsprings[k] = new Spring3(k === 'pick' ? OMEGA.pick : OMEGA.right);
    this.anchor = new THREE.Vector3();
    this.pick = { base: new THREE.Vector3(), tip: new THREE.Vector3(), visible: false };
    this.lastState = null;
    this.strumState = null;
    this.tune = new Spring3(16);    // per-chord placement tweak (dx, dkz, dky) and finger tilt (da), smoothed across chord changes
    this.tuneA = new Spring3(16);
  }

  setEvents(events) {
    this.events = events; this.mode = events.mode || 'fingers'; this._pickReady = false;
    if (events.strum && this.left) this.tuneChords(events);
  }

  /** fretting-hand frame: knuckle anchor + finger/lateral axes (d* = per-chord tweaks, all 0 for tab songs) */
  _leftFrame(anchorX, focus, dkz = 0, dky = 0, dalpha = 0) {
    const alpha = LEFT.alpha + dalpha;
    const anchor = new THREE.Vector3(anchorX, LEFT.ky + dky, -(fbWidth(anchorX) / 2 + LEFT.kz(focus) + dkz));
    const lH = new THREE.Vector3(-1, 0, 0), fH = new THREE.Vector3(0, Math.sin(alpha), Math.cos(alpha));
    if (WRIST.on) {
      const fret = -12 * Math.log2(Math.max(1e-3, 1 - anchorX / 0.648));
      const qYaw = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), LEFT.yaw(fret));
      lH.applyQuaternion(qYaw); fH.applyQuaternion(qYaw);
      lH.applyAxisAngle(fH, LEFT.roll(focus));
    }
    return { anchor, lH, fH };
  }

  /**
   * Chord shapes spread the fingers over all six strings, which the single fitted hand placement of the tab
   * songs does not always reach (7-16 mm misses on G, C, B7). For each distinct shape, search a few placement
   * tweaks (shift along the neck, knuckle distance/height, finger tilt) that minimise the static IK error of
   * the fingertips on their exact fret positions; the winner is stored as ev.hand and blended in by update().
   */
  tuneChords(events) {
    const L = this.left, cache = new Map(), V = THREE.Vector3;
    const DX = [-0.012, -0.006, 0, 0.006], DKZ = [-0.012, -0.006, 0, 0.006, 0.012], DKY = [-0.012, 0, 0.012], DA = [-8, 0, 8].map(d => d * D2R);
    for (const ev of events) {
      if (ev.hand || (events.mixed && !ev.strummed)) continue;   // mixed songs: only the strummed (chord) events are tuned
      const key = ev.pos + '|' + ev.notes.map(n => `${n.s}.${n.f}.${n.finger}`).join(',') + (ev.barre ? 'b' : '');
      if (cache.has(key)) { ev.hand = cache.get(key); continue; }
      const goals = {};
      FINGER_KEYS.forEach((k, i) => { const n = ev.byFinger[i]; if (n) { const x = contactX(n.f); goals[k] = new V(x, PRESS_Y_, stringZ(n.s, x)); } });
      const used = Object.keys(goals);
      let best = { dx: 0, dkz: 0, dky: 0, da: 0 };
      if (used.length) {
        const focus = ev.fretted.length ? ev.fretted.reduce((a, n) => a + n.s, 0) / ev.fretted.length : 2.5;
        const ax = 0.5 * (contactX(ev.pos) + contactX(ev.pos + 3)) - 0.01;
        const cost = c => {
          const fr = this._leftFrame(ax + c.dx, focus, c.dkz, c.dky, c.da);
          L.setPlacement(fr.anchor, fr.lH, fr.fH);
          L.solve(goals);
          let mx = 0, sum = 0;
          for (const k of used) { const e = L.chains[k].err * 1000; mx = Math.max(mx, e); sum += e; }
          return mx + 0.3 * sum / used.length + 15 * (Math.abs(c.dx) + Math.abs(c.dkz) + Math.abs(c.dky)) + 3 * Math.abs(c.da);
        };
        let bc = cost(best);
        if (bc > 0.6) for (const dx of DX) for (const dkz of DKZ) for (const dky of DKY) for (const da of DA) {
          const c = { dx, dkz, dky, da }, v = cost(c);
          if (v < bc - 1e-6) { bc = v; best = c; }
        }
      }
      cache.set(key, best); ev.hand = best;
    }
  }

  update(T, dt, jumped) {
    const st = sample(this.events, T);
    this.lastState = st;
    const tmp = new THREE.Vector3();

    // ---- fretting hand placement ----
    const ax = 0.5 * (contactX(st.pos) + contactX(st.pos + 3)) - 0.01;
    tmp.set(ax, st.focus ?? this.handSpring.x.y, 0);
    if (jumped) this.handSpring.snap(tmp); else this.handSpring.step(tmp, dt);
    // chord events carry a per-shape placement tweak found by tuneChords() (zero for tab songs)
    const ho = st.evH && st.evH.hand;
    tmp.set(ho ? ho.dx : 0, ho ? ho.dkz : 0, ho ? ho.dky : 0);
    if (jumped) this.tune.snap(tmp); else this.tune.step(tmp, dt);
    tmp.set(ho ? ho.da : 0, 0, 0);
    if (jumped) this.tuneA.snap(tmp); else this.tuneA.step(tmp, dt);
    const anchorX = this.handSpring.x.x + this.tune.x.x, focus = this.handSpring.x.y;
    const frame = this._leftFrame(anchorX, focus, this.tune.x.y, this.tune.x.z, this.tuneA.x.x);
    this.anchor.copy(frame.anchor);

    const L = this.left;
    if (L) {
      L.setPlacement(this.anchor, frame.lH, frame.fH);
      const goals = {};
      for (const k of FINGER_KEYS) {
        const g = st.targets[k];
        const sp = this.springs[k];
        sp.w = g.strike ? OMEGA.strike : OMEGA.finger;
        tmp.set(g.x, g.y, g.z);
        if (jumped) sp.snap(tmp); else sp.step(tmp, dt);
        goals[k] = sp.x;
      }
      // thumb pad on the back of the neck, opposite index/middle; it rides further round the back when the
      // fingers work the bass strings (reachable band found offline by IK sweep)
      const u = Math.max(THUMB.uMin, Math.min(THUMB.uMax, THUMB.u0 + THUMB.uk * focus)), wN = fbWidth(anchorX + THUMB.dx) / 2;
      tmp.set(anchorX + THUMB.dx, -0.012 - THUMB.depth * Math.sqrt(1 - u * u), u * wN);
      if (jumped) this.springs.thumb.snap(tmp); else this.springs.thumb.step(tmp, dt);
      goals.thumb = this.springs.thumb.x;
      L.solve(goals);
      L.setForearm(new THREE.Vector3(anchorX + 0.12, -0.13, -0.30));
    }

    // ---- string geometry: pressed points from the *actual* fingertips ----
    const pressed = [null, null, null, null, null, null];
    if (L) for (const k of FINGER_KEYS) {
      const g = st.targets[k];
      if (!(g.press || g.support)) continue;
      const s = g.note ? g.note.s : null;
      const tip = L.tipG(k);
      const sIdx = s ?? nearestString(tip);
      const zStr = stringZ(sIdx, tip.x);
      if (s === null && Math.abs(tip.z - zStr) > 0.0045) continue;
      const h = Math.max(FRET_H + 0.0006, Math.min(stringY(tip.x), tip.y - 0.0016));
      if (h >= stringY(tip.x) - 1e-5) continue;          // not touching yet
      const fret = g.fret ?? Math.ceil(fretIndexAt(tip.x));
      if (!pressed[sIdx] || tip.x > pressed[sIdx].x) pressed[sIdx] = { x: tip.x, y: h, z: tip.z, fretX: fretX(Math.max(1, fret)) };
      // barre chord: the same (index) finger lies across the other strings of the barre and presses them too
      const bar = g.ev && g.ev.barre;
      if (bar && k === 'index' && g.note && g.note.finger === 1) for (const s2 of bar.strings) {
        if (s2 === sIdx) continue;
        const p2 = { x: tip.x, y: h, z: stringZ(s2, tip.x), fretX: fretX(Math.max(1, fret)) };
        if (!pressed[s2] || tip.x > pressed[s2].x) pressed[s2] = p2;
      }
    }

    // ---- picking hand ----
    const R = this.right;
    this.pick.visible = false;
    if (R) {
      const { l: lR, f: fR } = rightAxes();
      // chord strumming (js/chordsong.js); in a mixed song (js/songchords.js buildMixedTimeline) only inside the strummed
      // bars' windows: null elsewhere, and the tab notes are picked below
      const sg = this.events.strum ? this.events.strum.at(T) : null;
      this.strumState = sg;
      if (!sg || sg.blend >= 1) for (const k in this.rsprings) this.rsprings[k].from = null;
      // approach into strummed bars (sg.blend < 1): the hand eases in on its springs from where the picking left it; on
      // the path itself it is exact (crossing times matter) and the springs are kept on it, so picking resumes from there
      const ease = (sp, v) => {
        if (!sg || sg.blend == null) return v;
        if (jumped || sg.blend >= 1) { sp.snap(v); sp.from = null; return v; }
        if (!sp.from) sp.from = sp.x.clone();                       // where the picking left the hand
        const u = sg.blend * sg.blend * (3 - 2 * sg.blend);
        sp.snap(v.lerpVectors(sp.from, v, u));
        return v;
      };
      if (sg && this.mode === 'pick') {
        // pick held in the grip; the pick tip follows the strum path exactly (no spring: crossing times matter),
        // and the wrist swings with the sweep (the tip stays on the path, the pick direction tilts)
        this._pickPose();
        ease(this.rsprings.pick, tmp.set(sg.x, sg.y, sg.z));
        const dirZ = Math.max(-0.7, Math.min(1.0, 0.25 - 14 * (sg.z - STRUM_Z0)));
        R.setPlacementAlign(this.pickDirArm, new THREE.Vector3(0, -1, dirZ), R.axesArm.f, fR.clone().setY(0), this.pickTipArm, tmp);
        this.pick.tip.copy(this.pickTipArm).applyMatrix4(R.place);
        this.pick.base.copy(this.pickBaseArm).applyMatrix4(R.place);
        this.pick.visible = true;
      } else if (sg) {
        // fingers: the index fingertip sweeps the strings, the hand follows half of the sweep, thumb and
        // middle/ring trail the index loosely
        const zc = stringZ(2.5, PLUCK_X);
        R.setPlacement(new THREE.Vector3(PLUCK_X + RIGHT.dx + (sg.x - STRUM.x), stringY(PLUCK_X) + RIGHT.ky + 0.4 * (sg.y - stringY(sg.x)), zc + RIGHT.kz + 0.55 * (sg.z - STRUM_Z0)), lR, fR);
        const goals = { index: ease(this.rsprings.index, new THREE.Vector3(sg.x, sg.y, sg.z)) };
        goals.thumb = ease(this.rsprings.thumb, new THREE.Vector3(sg.x - 0.022, sg.y + 0.010, sg.z - sg.sg * 0.006));
        goals.middle = ease(this.rsprings.middle, new THREE.Vector3(sg.x + 0.003, sg.y + 0.007, sg.z + 0.008));
        goals.ring = ease(this.rsprings.ring, new THREE.Vector3(sg.x + 0.010, sg.y + 0.009, sg.z + 0.015));
        goals.pinky = goals.ring.clone().add(new THREE.Vector3(0.012, 0.004, 0.008));
        R.solve(goals);
      } else if (this.mode === 'pick') {
        this._pickPose();
        const g = st.right.pick;
        const sp = this.rsprings.pick;
        tmp.set(g.x, g.y, g.z);
        if (jumped) sp.snap(tmp); else sp.step(tmp, dt);
        // orient the grip so the pick points down at the strings, then put the pick tip on the sprung target
        R.setPlacementAlign(this.pickDirArm, new THREE.Vector3(0, -1, 0.25), R.axesArm.f, fR.clone().setY(0), this.pickTipArm, sp.x);
        this.pick.tip.copy(this.pickTipArm).applyMatrix4(R.place);
        this.pick.base.copy(this.pickBaseArm).applyMatrix4(R.place);
        this.pick.visible = true;
      } else {
        const zc = stringZ(2.5, PLUCK_X);
        R.setPlacement(new THREE.Vector3(PLUCK_X + RIGHT.dx, stringY(PLUCK_X) + RIGHT.ky, zc + RIGHT.kz), lR, fR);
        const goals = {};
        for (const k of ['thumb', 'index', 'middle', 'ring']) {
          const g = st.right[k];
          if (!g) continue;
          tmp.set(g.x, g.y, g.z);
          const sp = this.rsprings[k];
          if (jumped) sp.snap(tmp); else sp.step(tmp, dt);
          goals[k] = sp.x;
        }
        // pinky follows the ring finger loosely (coupled), slightly curled
        if (goals.ring) goals.pinky = goals.ring.clone().add(new THREE.Vector3(0.016, 0.008, 0.008));
        R.solve(goals);
      }
      R.setForearm(this.strumState ? ELBOW_R.clone().add(new THREE.Vector3(0, 0, 3 * (this.strumState.z - STRUM_Z0))) : ELBOW_R);
    }
    return { st, pressed, anchor: this.anchor };
  }

  /** pick grip: index curled, thumb pad on the index distal phalanx, others loosely curled */
  _pickPose() {
    const R = this.right;
    if (this._pickReady) return;
    R.setAngles('middle', [0, 70 * D2R, 85 * D2R]);
    R.setAngles('ring', [0, 75 * D2R, 90 * D2R]);
    R.setAngles('pinky', [0, 80 * D2R, 90 * D2R]);
    R.setAngles('index', [0, 50 * D2R, 75 * D2R]);
    const ich = R.chains.index;
    const dist = ich.outP[3].clone(), tip = ich.outP[4].clone();
    const { l, n } = R.axesArm;
    const pad = dist.clone().lerp(tip, 0.5).addScaledVector(l, 0.009);
    R.solveLocal('thumb', pad);
    const th = R.chains.thumb.outP[3].clone();
    this.pickBaseArm = th.clone().lerp(pad, 0.5);
    // the pick sticks out of the thumb-index pinch, away from the curled fingers
    const palm = new THREE.Vector3();
    for (const k of ['middle', 'ring', 'pinky']) palm.add(R.chains[k].outP[4]).add(R.chains[k].outP[1]);
    palm.multiplyScalar(1 / 6);
    this.pickDirArm = this.pickBaseArm.clone().sub(palm).normalize().addScaledVector(n, -0.6).normalize();
    this.pickTipArm = this.pickBaseArm.clone().addScaledVector(this.pickDirArm, 0.02);
    this._pickReady = true;
  }
}

function nearestString(p) {
  let best = 0, d = Infinity;
  for (let s = 0; s < 6; s++) { const dd = Math.abs(stringZ(s, p.x) - p.z); if (dd < d) { d = dd; best = s; } }
  return best;
}
function fretIndexAt(x) { return -12 * Math.log2(1 - x / 0.648); }
