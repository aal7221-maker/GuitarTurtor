// Paired converter measurements; deterministic model-level fixtures, no inference.
import { toScore as before } from './fixtures/songmaker-before-phantoms.mjs';
import { toScore as after } from '../js/songmaker.js';
import { phantomFixtures } from './fixtures/phantom-notes.mjs';
const counts = score => ({ events: score.notes.length, chords: score.notes.filter(n => n.n.length > 1).length });
for (const f of phantomFixtures()) {
  console.log(JSON.stringify({ fixture: f.name, before: counts(before(f.raw)), after: counts(after(f.raw)),
    expectedScoreIdentical: JSON.stringify(after(f.raw)) === JSON.stringify(before(f.expected)) }));
}
