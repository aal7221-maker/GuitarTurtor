// E2E recorder: serves guitar3d, opens it in headless Chrome (swiftshader) and steps the page clock
// deterministically, capturing frames (-> mp4) and per-step state (-> JSON) of what the 3D guitar does.
//
//   node tests/e2e/record.mjs --song choo-lo [--start 0 --end 39] [--fps 15 --sub 4] [--out tests/e2e/out/choo-lo]
//   node tests/e2e/record.mjs --chords "G D Am Am G D C C" --pattern basic --bpm 69 --bars 1 --hand pick --name knockin
//   options: --no-frames (state only, fast)  --width 540 --height 960  --view front|player|close|back|pick  --keep-frames
//
// How the clock is controlled (no app code is changed): an init script replaces window.Audio with a fake media element
// whose currentTime we set, and replaces requestAnimationFrame/performance.now with a virtual clock. Each capture step
// of 1/fps s runs `sub` animation frames of dt = 1/(fps*sub) s (springs see a steady 60 fps), so results do not depend
// on how slow swiftshader is. State is recorded at every animation frame; frames are captured once per step.
import fs from 'node:fs';
import path from 'node:path';
import net from 'node:net';
import { spawn, execFileSync } from 'node:child_process';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createRequire } from 'node:module';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..', '..');
const PY = process.env.PYTHON || 'C:/Users/sharv/anaconda3/python.exe';
const CHROME = process.env.CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const FFMPEG = process.env.FFMPEG || 'C:/Users/sharv/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-9.1-full_build/bin/ffmpeg.exe';
const PUPPETEER_DIR = process.env.PUPPETEER_DIR || 'C:/Users/sharv/AppData/Local/Temp/claude/dae56179cd8afd99/5914db84-07fe-4dfa-89f2-50a4e5c63e96/scratchpad/nodetest';
let puppeteer;
try { puppeteer = (await import('puppeteer-core')).default; } catch { puppeteer = createRequire(path.join(PUPPETEER_DIR, 'package.json'))('puppeteer-core'); }
const G = await import(pathToFileURL(path.join(ROOT, 'js/guitar.js')).href);   // same geometry as the page

// ---------- args ----------
const argv = process.argv.slice(2);
const opt = (k, d) => { const i = argv.indexOf('--' + k); return i >= 0 ? argv[i + 1] : d; };
const flag = k => argv.includes('--' + k);
const song = opt('song'), chordsArg = opt('chords');
const name = opt('name', song || 'chords');
const FPS = +opt('fps', 15), SUB = +opt('sub', 4), DT = 1 / (FPS * SUB);
const W = +opt('width', 540), H = +opt('height', 960);
const OUT = path.resolve(opt('out', path.join(HERE, 'out', name)));
const FRAMES = !flag('no-frames');
const view = opt('view', 'front');
fs.mkdirSync(OUT, { recursive: true });
const frameDir = path.join(OUT, 'frames');
if (FRAMES) { fs.rmSync(frameDir, { recursive: true, force: true }); fs.mkdirSync(frameDir, { recursive: true }); }

// ---------- server ----------
const port = await new Promise(res => { const s = net.createServer(); s.listen(0, '127.0.0.1', () => { const p = s.address().port; s.close(() => res(p)); }); });
const server = spawn(PY, ['-m', 'http.server', String(port), '--bind', '127.0.0.1'], { cwd: ROOT, stdio: 'ignore' });
const stopServer = () => { try { server.kill(); } catch { /* gone */ } };
process.on('exit', stopServer);
for (let i = 0; i < 100; i++) {   // wait until it answers
  try { const r = await fetch(`http://127.0.0.1:${port}/data/songs.json`); if (r.ok) break; } catch { /* not yet */ }
  await new Promise(r => setTimeout(r, 100));
}

// ---------- page instrumentation (runs before any page script) ----------
const INIT = () => {
  const realNow = performance.now.bind(performance);
  let vnow = 0, q = [];
  performance.now = () => vnow;
  window.requestAnimationFrame = cb => { q.push(cb); return q.length; };
  window.cancelAnimationFrame = () => {};
  window.__tick = ms => { vnow += ms; const cbs = q; q = []; for (const cb of cbs) cb(vnow); };
  window.__auto = setInterval(() => window.__tick(16.7), 17);      // free-run while loading; the recorder stops it
  window.__realNow = realNow;
  class FakeAudio extends EventTarget {
    constructor(src) { super(); this._src = ''; this.currentTime = 0; this.paused = true; this.playbackRate = 1; this.duration = NaN; this.preload = ''; this.volume = 1; if (src) this.src = src; (window.__audios = window.__audios || []).push(this); }
    get src() { return this._src; }
    set src(v) { this._src = String(v || ''); setTimeout(() => this.dispatchEvent(new Event('loadedmetadata')), 0); }
    play() { this.paused = false; return Promise.resolve(); }
    pause() { this.paused = true; }
    load() {}
    removeAttribute() {}
  }
  window.Audio = FakeAudio;
  window.__state = () => {
    const g = window.__g3d, p = g.perf, st = p.lastState, K = ['index', 'middle', 'ring', 'pinky'];
    const v = x => [+x.x.toFixed(5), +x.y.toFixed(5), +x.z.toFixed(5)];
    const cur = st && st.now >= 0 ? g.events[st.now] : null;
    const sg = p.strumState;
    const r = {
      now: st ? st.now : -1, pos: st ? st.pos : null, anchor: v(p.anchor),
      lt: K.map(k => v(g.hand.tipG(k))),
      press: K.map(k => !!(st && st.targets[k] && st.targets[k].press)),
      tgt: K.map(k => { const q = st && st.targets[k]; return q && q.note ? [q.note.s, q.note.f] : null; }),
      rt: ['thumb', 'index', 'middle', 'ring'].map(k => v(g.right.tipG(k))),
      pick: p.pick.visible ? v(p.pick.tip) : null,
      strum: sg ? { c: !!sg.contact, sg: sg.sg, dir: sg.stroke ? sg.stroke.dir : null, z: +sg.z.toFixed(5) } : null,
      glow: g.events.strum ? g.events.strum.glow(window.__T).map(x => +x.toFixed(2)) : null,
      hud: document.getElementById('info').textContent,
      chord: g.chords.active ? document.getElementById('cp-name').textContent : null,
      cur: cur ? cur.notes.map(n => [n.s, n.f, n.finger || 0]) : null,
    };
    return r;
  };
  // one animation frame at time t: set the media clock, keep the render loop awake, run the frame callbacks
  window.__step = (ts, dtms) => {
    const out = [], a = window.__audios[0];
    for (const t of ts) { window.__T = t; a.currentTime = t; window.dispatchEvent(new Event('pointermove')); window.__tick(dtms); out.push(window.__state()); }
    return out;
  };
};

const browser = await puppeteer.launch({ executablePath: CHROME, headless: 'new',
  args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', `--window-size=${W},${H}`] });
const logs = [];
let result;
try {
  const page = await browser.newPage();
  await page.setViewport({ width: W, height: H, deviceScaleFactor: 1 });
  await page.setRequestInterception(true);
  page.on('request', r => { if (/youtube|ytimg|googlevideo/.test(r.url())) r.abort(); else r.continue(); });
  page.on('console', m => { if (m.type() === 'error' || m.type() === 'warning') logs.push(`[${m.type()}] ${m.text()}`); });
  page.on('pageerror', e => logs.push(`[pageerror] ${e.message}`));
  await page.evaluateOnNewDocument(INIT);
  const hash = chordsArg ? 'chords' : song;
  await page.goto(`http://127.0.0.1:${port}/index.html#${hash}`, { waitUntil: 'networkidle0', timeout: 120000 });
  await page.waitForFunction(() => window.__g3d && window.__g3d.perf && window.__g3d.events.length && document.getElementById('play').textContent === 'Play', { timeout: 120000 });

  if (chordsArg) {
    await page.waitForFunction(() => window.__g3d.chords.active && window.__g3d.chords.ready, { timeout: 60000 });
    // drive the chord-mode UI like a user: chords text, strum pattern, bars per chord, strumming hand, tempo
    await page.evaluate(({ chords, pattern, bpm, bars, hand }) => {
      const set = (id, v, ev) => { const el = document.getElementById(id); el.value = v; el.dispatchEvent(new Event(ev, { bubbles: true })); };
      set('cp-loop', null, 'change'); document.getElementById('cp-loop').checked = false; document.getElementById('cp-loop').dispatchEvent(new Event('change'));
      set('cp-pattern', pattern, 'change');
      set('cp-bars', String(bars), 'change');
      set('cp-hand', hand, 'change');
      set('cp-bpm', String(bpm), 'change');
      set('cp-chords', chords, 'input');
    }, { chords: chordsArg, pattern: opt('pattern', 'basic'), bpm: +opt('bpm', 90), bars: +opt('bars', 2), hand: opt('hand', 'pick') });
    const want = chordsArg.trim().split(/\s+/).join(' ');
    await page.waitForFunction(w => { const s = window.__g3d.chords.song; return s && s.chords.map(c => c.symbol).join(' ') === w.split(' ').map(x => x).join(' ') || (s && s.chords.length === w.split(' ').length); }, { timeout: 60000 }, want);
    await page.waitForFunction(() => window.__g3d.events.strum && document.getElementById('play').textContent === 'Play', { timeout: 120000 });
  }
  // let the hand meshes refine and textures arrive, then take over the clock
  await page.waitForFunction(() => window.__g3d.hand.refined && window.__g3d.right.refined, { timeout: 60000 }).catch(() => logs.push('[warn] hands not refined'));
  await new Promise(r => setTimeout(r, 1500));
  await page.evaluate(v => { clearInterval(window.__auto); window.__g3d.setView(v); }, view);

  const meta = await page.evaluate(() => {
    const g = window.__g3d, ev = g.events, cs = g.chords.active ? g.chords.song : null;
    return {
      song: g.song, chordMode: !!cs, mode: ev.mode, bpm: ev.bpm || (cs && cs.bpm) || null,
      duration: +document.getElementById('scrub').max,
      events: ev.map(e => ({ i: e.i, t: e.t, dur: e.dur, pos: e.pos, notes: e.notes.map(n => ({ s: n.s, f: n.f, finger: n.finger || 0, pluck: n.pluck || null })),
        slide: !!e.slide, ham: !!e.ham, pull: !!e.pull, bend: e.bend || 0, hb: e.hb || 0, rel: !!e.rel, vib: !!e.vib, legato: e.legato || null, symbol: e.symbol || null })),
      plucks: ev.plucks ? Object.fromEntries(Object.entries(ev.plucks).map(([k, l]) => [k, l.map(p => ({ t: p.t, s: p.s ?? null, strings: p.strings || null, dir: p.dir || null }))])) : null,
      strokes: ev.strokes ? ev.strokes.map(s => ({ t: s.t, dir: s.dir, strings: s.strings.map(x => 5 - x), offsets: s.offsets, muted: s.muted || null, accent: !!s.accent })) : null,
      chords: cs ? cs.chords.map(c => ({ symbol: c.symbol, t0: c.t0, t1: c.t1, pos: c.pos, frets: c.shape.frets })) : null,
      pattern: cs ? { id: cs.pattern.id, grid: cs.pattern.grid, step_beats: cs.pattern.step_beats } : null,
      status: document.getElementById('status').textContent,
    };
  });
  const t0 = +opt('start', 0), t1 = Math.min(+opt('end', meta.duration), meta.duration);

  await page.evaluate(t => { window.__g3d.seek(t); window.__audios[0].currentTime = t; }, t0);
  await page.evaluate(() => { for (let i = 0; i < 3; i++) window.__tick(1000 / 60); });

  const states = [];
  const nSteps = Math.floor((t1 - t0) * FPS + 1e-6) + 1;
  const wall0 = Date.now();
  for (let k = 0; k < nSteps; k++) {
    const T = t0 + k / FPS;
    const ts = [];
    for (let j = SUB - 1; j >= 0; j--) ts.push(+(T - j * DT).toFixed(6));
    if (k === 0) ts.splice(0, ts.length - 1);
    const recs = await page.evaluate((ts, ms) => window.__step(ts, ms), ts, DT * 1000);
    recs.forEach((r, i) => {
      const t = ts[i];
      // fingertip -> (string float, fret float, height above the string) with the page's own geometry
      const conv = ([x, y, z]) => {
        const zs = [0, 1, 2, 3, 4, 5].map(s => G.stringZ(s, x));
        const sf = (z - zs[0]) / (zs[5] - zs[0]) * 5;
        const ff = x <= 0 ? x / 0.03 : -12 * Math.log2(Math.max(1e-6, 1 - x / G.SCALE));
        return [+sf.toFixed(3), +ff.toFixed(3), +((y - G.stringY(Math.max(0, x))) * 1000).toFixed(2)];
      };
      r.t = t;
      r.ltc = r.lt.map(conv);       // fretting fingertips index..pinky: [string, fret, height mm]
      r.anchorFret = +(-12 * Math.log2(Math.max(1e-6, 1 - r.anchor[0] / G.SCALE))).toFixed(3);
      r.rtc = r.rt.map(conv);       // picking-hand fingertips thumb,index,middle,ring
      if (r.pick) r.pickc = conv(r.pick);
      r.frame = (i === recs.length - 1 && FRAMES) ? k : null;
      states.push(r);
    });
    if (FRAMES) await page.screenshot({ path: path.join(frameDir, `f${String(k).padStart(5, '0')}.png`) });
    if (k % 60 === 0) process.stdout.write(`  ${name}: step ${k}/${nSteps} t=${T.toFixed(2)}s (${((Date.now() - wall0) / 1000).toFixed(0)} s)\n`);
  }
  result = { name, args: argv, fps: FPS, sub: SUB, t0, t1, viewport: [W, H], view, meta, states, logs, wall_s: (Date.now() - wall0) / 1000 };
  fs.writeFileSync(path.join(OUT, 'state.json'), JSON.stringify(result));
  console.log(`state: ${states.length} samples -> ${path.join(OUT, 'state.json')}`);
} finally {
  await browser.close();
  stopServer();
}

if (FRAMES) {
  const mp4 = path.join(OUT, `${name}.mp4`);
  execFileSync(FFMPEG, ['-y', '-loglevel', 'error', '-framerate', String(FPS), '-start_number', '0', '-i', path.join(frameDir, 'f%05d.png'),
    '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23', mp4]);
  console.log(`video: ${mp4}  (frame k is page time t0 + k/${FPS} s)`);
  if (!flag('keep-frames')) fs.rmSync(frameDir, { recursive: true, force: true });
}
if (logs.length) console.log('page logs:\n  ' + logs.slice(0, 20).join('\n  '));
