// Recordings stay on this origin. Storage failure must never prevent a tab from being made.
const DB = 'guitar3d-song-audio', STORE = 'audio', TIMEOUT = 5000;

async function access(id, mode, blob) {
  let db;
  try {
    db = await new Promise((resolve, reject) => {
      let settled = false;
      const fail = e => { settled = true; clearTimeout(timer); reject(e); };
      const timer = setTimeout(() => fail(new Error('Audio storage timed out')), TIMEOUT);
      try {
        const req = indexedDB.open(DB, 1);
        req.onupgradeneeded = () => {
          try { req.result.createObjectStore(STORE); } catch (e) { fail(e); }
        };
        req.onsuccess = () => {
          try {
            if (settled) { req.result.close(); return; } // A blocked open can still succeed later.
            settled = true; clearTimeout(timer); resolve(req.result);
          } catch (e) { fail(e); }
        };
        req.onerror = () => fail(req.error);
        req.onblocked = () => fail(new Error('Audio storage is blocked'));
      } catch (e) { fail(e); }
    });
    return await new Promise((resolve, reject) => {
      let tx;
      const fail = e => { clearTimeout(timer); reject(e); };
      const timer = setTimeout(() => {
        try { if (tx) tx.abort(); } catch { /* already complete */ }
        fail(new Error('Audio storage timed out'));
      }, TIMEOUT);
      try {
        tx = db.transaction(STORE, mode === 'get' ? 'readonly' : 'readwrite');
        const store = tx.objectStore(STORE);
        const req = mode === 'get' ? store.get(id) : mode === 'put' ? store.put(blob, id) : store.delete(id);
        tx.oncomplete = () => {
          try { clearTimeout(timer); resolve(mode === 'get' ? req.result || null : true); }
          catch (e) { fail(e); }
        };
        tx.onerror = tx.onabort = () => fail(tx.error);
        req.onerror = () => fail(req.error);
      } catch (e) { fail(e); }
    });
  } catch { return mode === 'get' ? null : false; }
  finally { try { if (db) db.close(); } catch { /* unavailable storage */ } }
}

export const saveSongAudio = (id, blob) => access(id, 'put', blob);
export const loadSongAudio = id => access(id, 'get');
export const deleteSongAudio = id => access(id, 'delete');

// Scores use video time; a recording's clock starts at zero, including its leading silence.
export function shiftScore(score, seconds) {
  if (!seconds) return score;
  const time = t => +(t + seconds).toFixed(3);
  return { ...score, grid: score.grid?.map(time), notes: score.notes.map(n => ({ ...n, t: time(n.t) })),
    ...(score.chords ? { chords: score.chords.map(c => ({ ...c, t0: time(c.t0), t1: time(c.t1) })) } : {}),
    ...(score.strum ? { strum: { ...score.strum, t_bar0: score.strum.t_bar0 == null ? null : time(score.strum.t_bar0),
      sections: score.strum.sections?.map(c => ({ ...c, t0: time(c.t0), t1: time(c.t1) })), strokes: score.strum.strokes.map(k => ({ ...k, t: time(k.t) })) } } : {}) };
}
