// Model-level fixtures, NOT captured inference. Same voicings/timing as the KS
// audio in audio.mjs: .5 + 2*c, repeated at +1 s, 12 ms between strings.
// Basic Pitch amplitude is mean frame activation, not waveform amplitude.
import { GUITAR } from './audio.mjs';

export function ksNotes() {
  return ['C', 'G', 'Am', 'F'].flatMap((name, c) => [0, 1].flatMap(beat =>
    GUITAR[name].map((p, s) => ({ p, t: .5 + 2 * c + beat + .012 * s,
      e: 1.5 + 2 * c + beat, a: [.72, .88, .76, .82, .70, .78][s] }))));
}

// Short islands of residual salience: 12-15 frames (139-174 ms), safely above
// the decoder's >9-frame minimum. Every fourth frame crosses .4; intervening
// troughs stay within its 11-frame energy tolerance. Mean activation ~.32-.35
// can therefore survive the remaining-energy pass without an onset >.6.
// This is a plausible decoder-input model, not a prediction of neural outputs.
export function residual(p, t, count = 12) {
  const frames = Array.from({ length: count }, (_, i) => i % 4 === 0 ? .44 : .29);
  return { p, t, e: t + count * 256 / 22050, a: frames.reduce((a, b) => a + b) / count };
}

export function phantomFixtures() {
  const clean = ksNotes();
  const partial = (p, h, t, frames) => residual(Math.round(p + 12 * Math.log2(h)), t, frames);
  // E3 is a fragmented fundamental; A3 is the second partial of A2. The latter
  // is also a played string, so end that string's earlier detection first.
  // B4 and Db5 model the third/fifth partials of E3/A2, outside Am's pitch classes.
  const ghosts = [partial(52, 1, 4.82, 12), partial(45, 2, 5.18, 13),
    partial(52, 3, 5.84, 14), partial(45, 5, 6.16, 15)];
  const fragmented = clean.map(n => ({ ...n,
    e: n.t >= 4.5 && n.t < 5.5 && n.p === 52 ? 4.70 :
      n.t >= 4.5 && n.t < 5.5 && n.p === 57 ? 5.05 : n.e }));
  const melody = [residual(62, 4.82), { ...residual(52, 5.84), e: 6.3 }];
  // A real, quiet three-fret hammer E3 -> G3 during C. E3 ends at the
  // transition; the other strings still establish C. G3 is a chord tone.
  const hammer = clean.map(n => ({ ...n, e: n.t < 1 && n.p === 52 ? 1.08 : n.e }));
  return [
    { name: 'clean KS strums', raw: clean, expected: clean },
    { name: 'KS residual fundamentals and partials', raw: [...fragmented, ...ghosts], expected: fragmented },
    { name: 'soft passing melody and sustained chord tone', raw: [...clean, ...melody], expected: [...clean, ...melody] },
    { name: 'soft chord-tone hammer-on', raw: [...hammer, residual(55, 1.08)], expected: [...hammer, residual(55, 1.08)] },
    { name: 'phantoms plus real melody', raw: [...fragmented, ...ghosts, residual(60, 3.12)],
      expected: [...fragmented, residual(60, 3.12)] },
  ];
}
