// Guitar synth: renders a tab (normalized events) to a WAV Blob with an OfflineAudioContext.
// Used when the original recording isn't available, so the page always has sound that matches the hands.
const OPEN = [64, 59, 55, 50, 45, 40];          // e B G D A E (MIDI)
const SR = 22050;                               // plenty for guitar, halves render time
const mtof = m => 440 * Math.pow(2, (m - 69) / 12);

function distCurve(k) {
  const n = 2048, c = new Float32Array(n);
  for (let i = 0; i < n; i++) { const x = (i * 2) / n - 1; c[i] = ((1 + k) * x) / (1 + k * Math.abs(x)); }
  return c;
}

function buildChain(ctx, mode) {
  const out = ctx.createGain();
  const comp = ctx.createDynamicsCompressor();
  comp.threshold.value = -18; comp.ratio.value = 4;
  comp.connect(ctx.destination);
  if (mode === 'pick') {                          // distorted electric lead
    const hp = ctx.createBiquadFilter(); hp.type = 'highpass'; hp.frequency.value = 110;
    const sh = ctx.createWaveShaper(); sh.curve = distCurve(70); sh.oversample = '4x';
    const scoop = ctx.createBiquadFilter(); scoop.type = 'peaking'; scoop.frequency.value = 450; scoop.gain.value = -5;
    const pres = ctx.createBiquadFilter(); pres.type = 'peaking'; pres.frequency.value = 1900; pres.Q.value = 0.8; pres.gain.value = 5;
    const cab = ctx.createBiquadFilter(); cab.type = 'lowpass'; cab.frequency.value = 4300; cab.Q.value = 0.8;
    const lvl = ctx.createGain(); lvl.gain.value = 0.22;
    out.connect(hp).connect(sh).connect(scoop).connect(pres).connect(cab).connect(lvl).connect(comp);
    const dly = ctx.createDelay(1); dly.delayTime.value = 0.3;
    const fb = ctx.createGain(); fb.gain.value = 0.28;
    const wet = ctx.createGain(); wet.gain.value = 0.16;
    lvl.connect(dly); dly.connect(fb).connect(dly); dly.connect(wet).connect(comp);
  } else {                                        // clean acoustic
    const body = ctx.createBiquadFilter(); body.type = 'peaking'; body.frequency.value = 220; body.Q.value = 1.2; body.gain.value = 4;
    const air = ctx.createBiquadFilter(); air.type = 'highshelf'; air.frequency.value = 3500; air.gain.value = 3;
    const lvl = ctx.createGain(); lvl.gain.value = 0.55;
    out.connect(body).connect(air).connect(lvl).connect(comp);
  }
  return out;
}

function voice(ctx, dest, t, startMidi, dur, o, mode) {
  const dist = mode === 'pick';
  const f = mtof(startMidi);
  const o1 = ctx.createOscillator(); o1.type = 'sawtooth';
  const o2 = ctx.createOscillator(); o2.type = dist ? 'square' : 'triangle'; o2.detune.value = 6;
  const g2 = ctx.createGain(); g2.gain.value = dist ? 0.35 : 1.1;
  const filt = ctx.createBiquadFilter(); filt.type = 'lowpass'; filt.Q.value = dist ? 1.2 : 0.7;
  const g = ctx.createGain();
  const osc = [o1, o2];
  osc.forEach(x => x.frequency.setValueAtTime(f, t));
  let cur = f;
  if (o.bendTo) {
    const fb = mtof(o.bendTo), t1 = t + Math.min(0.05, dur * 0.2), t2 = t + Math.min(0.18, dur * 0.6);
    osc.forEach(x => { x.frequency.setValueAtTime(f, t1); x.frequency.exponentialRampToValueAtTime(fb, t2); });
    cur = fb;
  }
  if (o.relTo) {
    const fr = mtof(o.relTo), t1 = t + dur * 0.45, t2 = t + dur * 0.6;
    osc.forEach(x => { x.frequency.setValueAtTime(cur, t1); x.frequency.exponentialRampToValueAtTime(fr, t2); });
    cur = fr;
  }
  if (o.slideTo) {
    const fs = mtof(o.slideTo), t1 = t + Math.min(dur * 0.35, 0.12), t2 = t + Math.min(dur * 0.95, t1 - t + 0.12);
    osc.forEach(x => { x.frequency.setValueAtTime(cur, t1); x.frequency.exponentialRampToValueAtTime(fs, t2); });
  }
  if (o.vib) {
    const lfo = ctx.createOscillator(); lfo.frequency.value = 5.6;
    const lg = ctx.createGain(); lg.gain.setValueAtTime(0, t); lg.gain.linearRampToValueAtTime(30, t + 0.25);
    lfo.connect(lg); lg.connect(o1.detune); lg.connect(o2.detune);
    lfo.start(t); lfo.stop(t + dur + 0.4);
  }
  const len = o.pm ? Math.min(dur, 0.13) : dur;
  const peak = (o.soft ? 0.6 : 1) * (o.vel ?? 1) * (dist ? 0.5 : 0.32);
  if (dist) {
    filt.frequency.setValueAtTime(o.pm ? 650 : 3400, t);
    filt.frequency.exponentialRampToValueAtTime(o.pm ? 260 : 1500, t + Math.max(0.05, Math.min(0.45, len)));
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(peak, t + 0.005);
    g.gain.exponentialRampToValueAtTime(peak * 0.55, t + Math.max(0.02, Math.min(0.25, len * 0.8)));
    g.gain.setTargetAtTime(0.0001, t + len, 0.035);
  } else {                                        // plucked string: bright attack, long natural decay
    filt.frequency.setValueAtTime(Math.min(9000, f * 9), t);
    filt.frequency.exponentialRampToValueAtTime(Math.max(500, f * 2.2), t + 0.6);
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(peak, t + 0.004);
    g.gain.setTargetAtTime(peak * 0.02, t + 0.004, 0.55);
    g.gain.setTargetAtTime(0.0001, t + len, 0.06);
  }
  o1.connect(filt); o2.connect(g2).connect(filt); filt.connect(g).connect(dest);
  osc.forEach(x => { x.start(t); x.stop(t + len + 0.4); });
}

function encodeWav(buf) {
  const ch = buf.numberOfChannels, n = buf.length, bytes = 44 + n * ch * 2;
  const dv = new DataView(new ArrayBuffer(bytes));
  const w = (o, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(o + i, s.charCodeAt(i)); };
  w(0, 'RIFF'); dv.setUint32(4, bytes - 8, true); w(8, 'WAVE'); w(12, 'fmt ');
  dv.setUint32(16, 16, true); dv.setUint16(20, 1, true); dv.setUint16(22, ch, true);
  dv.setUint32(24, buf.sampleRate, true); dv.setUint32(28, buf.sampleRate * ch * 2, true);
  dv.setUint16(32, ch * 2, true); dv.setUint16(34, 16, true); w(36, 'data'); dv.setUint32(40, n * ch * 2, true);
  const data = Array.from({ length: ch }, (_, c) => buf.getChannelData(c));
  let o = 44;
  for (let i = 0; i < n; i++) for (let c = 0; c < ch; c++) {
    const s = Math.max(-1, Math.min(1, data[c][i]));
    dv.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true); o += 2;
  }
  return new Blob([dv], { type: 'audio/wav' });
}

// events: output of normalizeScore (t, dur, gap, notes[{s,f}], slide, bend, hb, rel, vib, ham, pull, pm)
export async function renderSong(events, mode, seconds) {
  const Ctx = window.OfflineAudioContext || window.webkitOfflineAudioContext;
  const ctx = new Ctx(1, Math.ceil(SR * seconds), SR);
  addNotes(ctx, buildChain(ctx, mode), events, mode);
  return encodeWav(await ctx.startRendering());
}

function addNotes(ctx, dest, events, mode) {
  events.forEach((e, i) => {
    const next = events[i + 1], prev = events[i - 1];
    const legatoIn = prev && (prev.slide || prev.ham || prev.pull);
    const gap = next ? next.t - e.t : 1.5;
    e.notes.forEach(({ s, f }, k) => {
      // let a plucked string ring until the same string is played again (open B / e drones in Choo Lo)
      let ring = gap;
      if (mode !== 'pick') {
        const again = events.slice(i + 1).find(x => x.notes.some(n => n.s === s));
        ring = Math.min(2.4, again ? again.t - e.t : 2.4);
      }
      const dur = Math.max(0.08, Math.min(ring * 0.98, e.vib ? 2.4 : 2.0));
      const slideTarget = e.slide && next ? next.notes.find(n => n.s === s) || next.notes[0] : null;
      voice(ctx, dest, e.t + k * 0.007, OPEN[s] + (e.hb || f), dur, {
        bendTo: e.bend ? OPEN[s] + e.bend : 0,
        relTo: e.rel ? OPEN[s] + f : 0,
        slideTo: slideTarget ? OPEN[slideTarget.s] + slideTarget.f : 0,
        vib: e.vib, pm: e.pm, soft: legatoIn,
      }, mode);
    });
  });
}

// ---------- chord strumming ----------
const OPEN_LOW = [40, 45, 50, 55, 59, 64];      // chords/ order: low E -> high e

/** short percussive "chuck": band-passed noise, no pitch, no sustain */
function chuck(ctx, dest, t, vel) {
  const n = Math.floor(ctx.sampleRate * 0.06), buf = ctx.createBuffer(1, n, ctx.sampleRate), d = buf.getChannelData(0);
  let seed = 12345;
  for (let i = 0; i < n; i++) { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; d[i] = (seed / 2147483648 - 1) * Math.exp(-i / (ctx.sampleRate * 0.012)); }
  const src = ctx.createBufferSource(); src.buffer = buf;
  const bp = ctx.createBiquadFilter(); bp.type = 'bandpass'; bp.frequency.value = 1900; bp.Q.value = 0.9;
  const g = ctx.createGain(); g.gain.setValueAtTime(Math.max(0.05, vel) * 1.1, t);
  g.gain.exponentialRampToValueAtTime(0.0001, t + 0.05);
  src.connect(bp).connect(g).connect(dest);
  src.start(t);
}

/**
 * Render a stroke list (chords/strum_model.js: strings low-E-first, per-string offsets, velocity, muted) to a WAV Blob.
 * mode 'pick' = distorted electric chain, otherwise clean acoustic. opts.click = {bpm, beatsPerBar, startT} adds a metronome.
 */
export async function renderStrokes(strokes, mode, seconds, opts = {}) {
  const Ctx = window.OfflineAudioContext || window.webkitOfflineAudioContext;
  const ctx = new Ctx(1, Math.ceil(SR * seconds), SR);
  addStrokes(ctx, buildChain(ctx, mode === 'pick' ? 'pick' : 'fingers'), strokes, mode);
  if (opts.click) {
    const { bpm, beatsPerBar, startT = 0 } = opts.click, beat = 60 / bpm;
    for (let b = 0, t = startT; t < seconds - 0.1; b++, t = startT + b * beat) {
      const o = ctx.createOscillator(), g = ctx.createGain();
      o.type = 'square'; o.frequency.value = b % beatsPerBar === 0 ? 1500 : 1000;
      g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(b % beatsPerBar === 0 ? 0.16 : 0.1, t + 0.002);
      g.gain.exponentialRampToValueAtTime(0.0001, t + 0.04);
      o.connect(g).connect(ctx.destination); o.start(t); o.stop(t + 0.06);
    }
  }
  return encodeWav(await ctx.startRendering());
}

function addStrokes(ctx, dest, strokes, mode) {
  strokes.forEach((st, i) => {
    const vel = st.velocity ?? 0.7;
    if (st.muted === 'dead') { chuck(ctx, dest, Math.max(0, st.t), vel * 0.8); return; }
    // strings stop ringing when the same string is struck again or the chord changes (the fretting hand lifts)
    const change = strokes.find((x, j) => j > i && x.chord !== st.chord);
    st.strings.forEach((sl, k) => {
      const ts = st.t + st.offsets[k];
      const again = strokes.find((x, j) => j > i && x.strings.includes(sl));
      let end = ts + 2.2;
      if (again) end = Math.min(end, again.t + again.offsets[again.strings.indexOf(sl)]);
      if (change) end = Math.min(end, change.t + 0.03);
      if (Number.isFinite(st.ringUntil)) end = Math.min(end, st.ringUntil);   // mixed songs: stop when the notes take over
      const fret = st.frets ? st.frets[k] : 0;
      voice(ctx, dest, Math.max(0, ts), OPEN_LOW[sl] + fret, Math.max(0.1, end - ts), { pm: !!st.muted, vel: 0.35 + 0.65 * vel * (st.dir === 'U' ? 0.9 : 1) }, mode === 'pick' ? 'pick' : 'fingers');
    });
  });
}

/** a song that mixes single notes (tab events, as renderSong) and strummed bars (strokes, as renderStrokes) in one take */
export async function renderMixed(events, strokes, mode, seconds) {
  const Ctx = window.OfflineAudioContext || window.webkitOfflineAudioContext;
  const ctx = new Ctx(1, Math.ceil(SR * seconds), SR);
  const dest = buildChain(ctx, mode === 'pick' ? 'pick' : 'fingers');
  addNotes(ctx, dest, events, mode === 'pick' ? 'pick' : 'fingers');
  addStrokes(ctx, dest, strokes, mode);
  return encodeWav(await ctx.startRendering());
}
