"""Generate strum_patterns.json. Pattern string tokens per step (one per grid step):
   D / U = normal down/up   d / u = light (ghost)   > after a letter = accent   x = dead-muted strum (chuck, percussive)
   - = rest (hand keeps moving, no contact).  Palm-mute flag per pattern ('pm': true) or per token with 'p' suffix.
Strings are indexed 0 = low E ... 5 = high e. Down hits strings 0..5 (bass-led), up hits the top 4 (2..5)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))

def P(id, name, ts, grid, sub, bpm, diff, songs, desc, **kw):
    spb = {'4/4': 4, '3/4': 3, '6/8': 2}[ts]               # beats per bar (6/8 counted in dotted-quarter beats)
    n = len(grid.split())
    d = dict(id=id, name=name, time_sig=ts, subdivision=sub, steps_per_bar=n, beats_per_bar=spb, step_beats=spb / n,
             grid=grid, bpm=bpm, difficulty=diff, songs=songs, description=desc, type='strum')
    d.update(kw); return d

S = [
 P('down8', 'All down eighths', '4/4', 'D D D D D D D D', '8th', [90, 160], 1, ['many punk songs', 'driving rock rhythm'], 'Down on every 8th note. Easiest start.', feel='driving'),
 P('quarters', 'Quarter-note downs', '4/4', 'D - D - D - D -', '8th', [60, 120], 1, ['slow ballads'], 'One down per beat.', feel='ballad'),
 P('basic', 'D DU UDU (the universal pattern)', '4/4', 'D - D U - U D U', '8th', [70, 130], 2, ['Wonderwall', 'Let It Be', 'Riptide', 'Brown Eyed Girl'], 'Beats 1, 2&, 3&, 4 and 4& - the most used pop/folk strum. Count 1 2 & & 4 &. Keep the arm swinging, miss the strings on the rests.'),
 P('ddudu', 'D D UDU', '4/4', 'D - D - - U D U', '8th', [70, 120], 2, ['Hey Soul Sister (similar)', 'Hallelujah (fast)'], 'Down on 1, 2, then 3-and-4 pattern: D D (rest) UDU.'),
 P('folk', 'Folk strum', '4/4', 'D - D U D - D U', '8th', [80, 130], 2, ['Mr. Tambourine Man', 'Free Fallin'], 'Down on each beat with an up after 2 and 4 - the Johnny Cash boom-chick feel.'),
 P('pop', 'Pop 8ths with accent', '4/4', 'D> u D u D> u D u', '8th', [90, 130], 2, ['Shape of You (uke/gtr)', 'Counting Stars'], 'Alternate down accents on 1 and 3, light ups.'),
 P('rock8', 'Rock 8ths (palm-muted)', '4/4', 'Dp Dp Dp Dp Dp Dp Dp Dp', '8th', [100, 180], 2, ['Back In Black verses', 'Smoke On The Water (chords)'], 'Steady palm-muted downstrokes, open the mute on accents.', pm=True, feel='driving'),
 P('rock_open', 'Rock 8ths accented', '4/4', 'D> D D> D D> D D> D', '8th', [100, 180], 2, ['Sweet Home Alabama (feel)'], 'Eight downs, accent on the beats.', feel='driving'),
 P('reggae', 'Reggae / ska offbeat', '4/4', '- u - u - u - u', '8th', [60, 160], 2, ['No Woman No Cry', 'Three Little Birds', 'Is This Love'], 'Short, clipped up-strokes on the offbeats (the &), top 3-4 strings only.', stroke_strings='top3'),
 P('funk16', 'Funk 16ths (muted)', '4/4', 'D x D x U x D x D x D x U x D x', '16th', [80, 110], 4, ['Play That Funky Music', 'Superstition (rhythm)'], 'Constant 16th hand motion; x = percussive dead strums with the fretting hand released, accents on D.'),
 P('funk16b', 'Funk 16ths (ghosted)', '4/4', 'D> x u x D x u D> x u x D x u x u', '16th', [85, 105], 4, ['Get Lucky', 'Uptown Funk (rhythm)'], 'Syncopated 16ths with ghost ups and chucks.'),
 P('ballad', 'Slow ballad', '4/4', 'D - - - D - D u', '8th', [50, 80], 1, ['Perfect', 'Thinking Out Loud (strum)'], 'Big down on 1, light on 3, a little pickup into the next bar.', feel='ballad'),
 P('ballad2', 'Ballad 8ths', '4/4', 'D - d u - u D u', '8th', [55, 85], 2, ['Photograph', 'Stand By Me (feel)'], 'Softer version of the universal pattern.', feel='ballad'),
 P('ballad16', 'Slow ballad (16ths)', '4/4', 'D - - - D - D U - U D U - U D U', '16th', [50, 80], 3, ["Knockin' on Heaven's Door (lesson)"], 'Slow 16th-note ballad: down on 1, D D U on 2, then U D U  U D U through beats 3 and 4. Half of it (D D D U) fits a 2-beat chord.', feel='ballad'),
 P('shuffle', 'Shuffle / swing', '4/4', 'D - U D - U D - U D - U', '8th', [80, 140], 3, ['Johnny B. Goode (feel)', 'Blues shuffle'], 'Triplet feel: 12 triplet steps per bar; strum the 1st and 3rd partial of each beat (long-short).', steps_per_bar=12, step_beats=1 / 3, swing=0),
 P('bossa', 'Bossa nova', '4/4', 'D - D u - u D -', '8th', [80, 140], 3, ['The Girl From Ipanema', 'Corcovado'], 'Syncopated bass-then-chord pulse (clave-like), fingers or soft pick.', stroke_strings='bassmix'),
 P('waltz', 'Waltz (3/4)', '3/4', 'D - d - d -', '8th', [90, 180], 1, ['Norwegian Wood', 'Tennessee Waltz', 'Moon River'], 'Boom-chick-chick: bass note on 1, light chord downs on 2 and 3.', stroke_strings='bass_then_top'),
 P('waltz2', 'Waltz strum', '3/4', 'D - D U D U', '8th', [90, 160], 2, ['Amazing Grace', 'Can\'t Help Falling In Love (3/4 feel)'], 'Down on 1, D U on 2, D U on 3.'),
 P('six8', '6/8 strum', '6/8', 'D - - D U -', '8th', [50, 90], 2, ['House of the Rising Sun (feel)', 'Nothing Else Matters'], 'Two dotted-quarter beats, each 3 eighths: D on 1, DU on 4.', feel='ballad'),
 P('six8_arp', '6/8 arpeggio feel', '6/8', 'b - - - - -', '8th', [50, 90], 2, ['House of the Rising Sun'], 'See fingerpicking pattern six8_arp.', type='alias', alias='six8_arp'),
]
S = [s for s in S if s['type'] != 'alias']

FP = [  # fingerpicking: each step is a role: p = bass root, p2 = alt bass (5th), i = G string, m = B, a = e; chords with '+' pluck together
 dict(id='pima', name='p-i-m-a arpeggio', time_sig='4/4', subdivision='8th', steps_per_bar=8, beats_per_bar=4, step_beats=0.5,
      steps=['p', 'i', 'm', 'a', 'm', 'i', 'm', 'i'], difficulty=2, songs=['Dust in the Wind', 'Blackbird (basis)'], description='Thumb on bass, i m a on G B e.', type='pick'),
 dict(id='travis', name='Travis picking', time_sig='4/4', subdivision='8th', steps_per_bar=8, beats_per_bar=4, step_beats=0.5,
      steps=['p', 'i', 'p2', 'm', 'p', 'i', 'p2', 'm'], difficulty=3, songs=['Dust in the Wind (variant)', 'Landslide', 'Freight Train'], description='Thumb alternates root and fifth/neighbour bass (E-D or A-D strings) steadily; fingers pluck treble notes in between.', type='pick'),
 dict(id='arp_up', name='Simple arpeggio up-down', time_sig='4/4', subdivision='8th', steps_per_bar=8, beats_per_bar=4, step_beats=0.5,
      steps=['p', 'i', 'm', 'a', 'm', 'i', 'm', 'a'], difficulty=1, songs=['Stand By Me (arp)', 'Perfect (arp)'], description='Roll the strings of the chord up then back down.', type='pick'),
 dict(id='six8_arp', name='6/8 arpeggio', time_sig='6/8', subdivision='8th', steps_per_bar=6, beats_per_bar=2, step_beats=1 / 3,
      steps=['p', 'i', 'm', 'a', 'm', 'i'], difficulty=2, songs=['House of the Rising Sun'], description='p i m a m i over a 6/8 bar.', type='pick'),
 dict(id='waltz_pick', name='3/4 pick: bass + chord + chord', time_sig='3/4', subdivision='8th', steps_per_bar=3, beats_per_bar=3, step_beats=1,
      steps=['p', 'i+m+a', 'i+m+a'], difficulty=1, songs=['Norwegian Wood (feel)'], description='Bass note, then two plucked treble chords.', type='pick'),
 dict(id='pinch', name='Pinch + arpeggio', time_sig='4/4', subdivision='8th', steps_per_bar=8, beats_per_bar=4, step_beats=0.5,
      steps=['p+a', 'i', 'm', 'i', 'p+a', 'i', 'm', 'i'], difficulty=2, songs=['Fast Car', 'Landslide (feel)'], description='Thumb and ring finger pinch together on beats 1 and 3.', type='pick'),
]

out = dict(
  note='Strings 0..5 = low E..high e. strum grid tokens: D/U down/up, d/u light, ">" accent, "p" suffix palm-muted, x dead-muted chuck, "-" rest. '
       'Down hits all strings (low to high), up hits the top 3 (G B e = 3..5, high to low; set "up4" adds the D string) unless stroke_strings overrides. '
       'Timing defaults (ms): down per-string spread 5-12, up 4-9, stroke duration 90-160 shrinking with tempo; see strum_model.js.',
  tokens={'D': 'down', 'U': 'up', 'd': 'light down', 'u': 'light up', '>': 'accent', 'p': 'palm-mute', 'x': 'dead-muted strum', '-': 'rest'},
  stroke_strings={'default': {'D': [0, 1, 2, 3, 4, 5], 'U': [3, 4, 5]}, 'top3': {'D': [3, 4, 5], 'U': [3, 4, 5]},
                  'bass_then_top': {'D': [0], 'd': [1, 2, 3, 4, 5], 'U': [3, 4, 5]}, 'bassmix': {'D': [0, 3, 4, 5], 'U': [3, 4, 5]},
                  'up4': {'D': [0, 1, 2, 3, 4, 5], 'U': [2, 3, 4, 5]}},
  strum=S, pick=FP)
for s in S:
    s['steps_per_bar'] = len(s['grid'].split()); s['step_beats'] = s['beats_per_bar'] / s['steps_per_bar']
    if s['id'] == 'shuffle': s['swing'] = 0
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'strum_patterns.json'), 'w'), indent=1)
print(len(S), 'strum', len(FP), 'pick', [(s['id'], s['steps_per_bar'], round(s['step_beats'], 3)) for s in S])
