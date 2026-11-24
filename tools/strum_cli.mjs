// node tools/strum_cli.mjs AUDIO.wav [chords.json] [--full] -> prints score.strum (strumForScore; null when no strum pattern) or,
// with --full, the whole detectStrum analysis. Same code as the browser (js/strumdetect.js).
// AUDIO.wav: 16-bit PCM, mono or stereo, 22.05 kHz (ffmpeg -ac 1 -ar 22050). chords.json: [{t0,t1,name}] (score.chords).
// Without chords.json the chord-change bar-phase vote and the "is it chordal" gate are skipped.
import fs from 'node:fs';
import { detectStrum, strumForScore } from '../js/strumdetect.js';
import { readWav } from './synth_guitar.mjs';
const full = process.argv.includes('--full'), [wav, chordsFile] = process.argv.slice(2).filter(a => a !== '--full');
if (!wav) { console.error('usage: node tools/strum_cli.mjs AUDIO.wav [chords.json]'); process.exit(2); }
const { y, sr } = readWav(wav);
if (sr !== 22050) { console.error('expected 22050 Hz, got ' + sr); process.exit(2); }
const patterns = JSON.parse(fs.readFileSync(new URL('../chords/strum_patterns.json', import.meta.url)));
const chords = chordsFile ? JSON.parse(fs.readFileSync(chordsFile, 'utf8')) : null;
process.stdout.write(JSON.stringify(full ? detectStrum(y, { patterns, chords }) : strumForScore(y, chords, patterns)));
