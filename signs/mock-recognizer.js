/* signs/mock-recognizer.js: a stand-in for signs/recognizer.js while the sign model is trained, so the lesson can be
 * built and tried end to end. Same createRecognizer() signature, same controls (setTarget, pushFrame, on, setFlip,
 * close) and the same events (docs/signs-design.md), so swapping the app's one import for ./recognizer.js changes
 * nothing else.
 *
 * It looks only at whether the signing hand is up and how fast it moves, as the real one finds a try: a try starts
 * when the hand moves after it was down or still, and ends when the hand goes down or stays still. What is said about
 * a try comes from outside:
 *   - the keyboard (on a laptop): A accepts the sign asked for (the next one in a sequence), M misses it with sample
 *     feedback in the sign's own words (signs/content.json fixes), U is unclear;
 *   - ?mock=auto: about 2 s of hand movement accepts the sign due (and with mock.missFirst = N, the first N verdicts of
 *     each sign in the learn phase are misses, to see that screen). The movement of every try since the last verdict
 *     counts, so several short tries add up, as a learner's repeated tries would.
 * In a sequence each sign due waits for its own 2 s, so the reply's chips light up one by one; after 8 s without the
 * sign due, a hint, as the real one gives.
 *
 * Never for learners: it says yes to anything that moves. */

export const MOCK = true;

export const MOCK_DEFAULTS = Object.freeze({
  moveSpeed: 0.8,        // palm lengths per second: moving at or above (the mean speed of the wrist and fingertips)
  stillSpeed: 0.5,       //                          still below
  speedSpanMs: 100,      // speed over this much time: one frame's jitter is not motion
  goneMs: 250,           // the hand out of view this long: down
  stillEndMs: 600,       // still this long after moving: the try is over
  armStillMs: 300,       // after a verdict or setTarget: still this long (or down) before a new try
  acceptMovingMs: 2000,  // ?mock=auto: this much movement accepts the sign due
  hintMs: 8000,
});

const POINTS = [0, 4, 8, 12, 16, 20];
/* What a miss of each lesson sign "looked like" in the sample feedback: another lesson sign, as the real one names
 * only signs the learner has met. */
const NEIGHBOUR = Object.freeze({ hello: "bye", bye: "hello", yes: "no", no: "yes", please: "thankyou", thankyou: "please", water: "yes" });
const PART_ORDER = ["location", "movement", "handshape"];
const GENERIC = Object.freeze({
  location: "Start with your hand a little higher, near your face.",
  movement: "Make the movement a little bigger.",
  handshape: "Check your hand shape against the demonstration.",
});
const UNCLEAR = ["hand-lost", "too-short", "out-of-frame"];

function firstWords(fix, key) {
  if (fix == null) return null;
  if (typeof fix === "string") return fix;
  return fix[key] ?? fix.default ?? Object.values(fix).find((v) => typeof v === "string") ?? null;
}

/**
 * @param {object} o  the real createRecognizer's options (runner, featuresFn, feedbackFn, cuts... are ignored), plus
 * @param {object} [o.content]  signs/content.json, for the sample feedback's words
 * @param {{mode?: "keys"|"auto", missFirst?: number, keys?: boolean}} [o.mock]
 *   mode "auto": movement accepts; keys (default true in a browser): A / M / U on the window
 */
export function createRecognizer({ labels = null, hand = "right", flip = null, content = null, mock = {}, options = {} } = {}) {
  const o = { ...MOCK_DEFAULTS, ...options };
  const mode = mock.mode === "auto" ? "auto" : "keys";
  const missFirst = Math.max(0, Math.floor(Number(mock.missFirst) || 0));
  const known = Array.isArray(labels) ? new Set(labels) : null;
  let side = flip === null || flip === undefined ? hand === "left" : !!flip;
  const listeners = new Set();
  const stats = { frames: 0, attempts: 0, accepts: 0, misses: 0, unclear: 0, hints: 0 };

  let target = null, idx = 0, done = false, gen = 0;
  let armed = true, attempt = null, indexSince = null, hintCount = 0, movedMs = 0;
  const missesOf = new Map();
  // motion
  const recent = [];
  let lastSeen = null, stillSince = null, act = null, present = false;

  const glossOf = (id) => content?.signs?.[id]?.gloss ?? String(id).toUpperCase();
  const emit = (e) => {
    const g = gen;
    // after the caller's own call returns, as the real one answers once its model run is back; dropped if the target
    // changed meanwhile
    Promise.resolve().then(() => {
      if (g !== gen && e.type !== "error") return;
      for (const cb of [...listeners]) {
        try { cb(e); } catch (err) { console.warn(`mock-recognizer: a listener threw (${err?.message || err})`); }
      }
    });
  };
  const signAt = (i) => target?.signs[i] ?? null;
  const isSeq = () => target?.mode === "sequence";

  /* ---- the verdicts (also the keyboard's) ---- */
  function accept(confidence = 0.93) {
    if (!target || done) return false;
    const sign = signAt(idx);
    stats.accepts++;
    movedMs = 0;
    emit({ type: "accept", sign, index: idx, confidence, early: false });
    if (!isSeq()) { done = true; endAttempt(); return true; }
    idx++;
    hintCount = 0;
    indexSince = lastT;
    if (idx >= target.signs.length) {
      done = true;
      emit({ type: "sequence-done", signs: [...target.signs] });
      endAttempt();
    } else if (attempt) {
      emit({ type: "progress", index: idx, sign: signAt(idx) });
    }
    return true;
  }
  function miss() {
    if (!target || done) return false;
    const sign = signAt(idx);
    const n = missesOf.get(sign) ?? 0;
    missesOf.set(sign, n + 1);
    movedMs = 0;
    const part = PART_ORDER[n % PART_ORDER.length];
    const words = content?.signs?.[sign] ?? {};
    let text = firstWords(words.fixes?.[part], part === "location" ? "higher" : part === "movement" ? "bigger" : "default") ?? GENERIC[part];
    // the first miss of a sign also says what it looked like (the real feedback does when the model is sure)
    const like = n === 0 && NEIGHBOUR[sign] && (!known || known.has(NEIGHBOUR[sign])) ? NEIGHBOUR[sign] : null;
    if (like) text = `That looked like ${glossOf(like)}. ${text}`;
    const checks = { handshape: true, location: true, movement: true };
    checks[part] = false;
    // a part that could not be told, now and then, as the real checks give null
    if (n % 2 === 1) checks[part === "handshape" ? "movement" : "handshape"] = null;
    stats.misses++;
    emit({
      type: "miss", sign, index: idx, confidence: 0.12, looksLike: like, looksLikeP: like ? 0.71 : 0.2,
      top: like ? [{ sign: like, p: 0.71 }, { sign, p: 0.12 }] : [{ sign, p: 0.12 }],
      feedback: { text, checks, looksLike: like ? glossOf(like) : null, part },
    });
    endAttempt();
    return true;
  }
  let unclearTurn = 0;
  function unclear(reason = null) {
    if (!target || done) return false;
    const r = UNCLEAR.includes(reason) ? reason : UNCLEAR[unclearTurn++ % UNCLEAR.length];
    stats.unclear++;
    emit({ type: "unclear", sign: signAt(idx), index: idx, reason: r });
    endAttempt();
    return true;
  }
  function endAttempt() {
    attempt = null;
    armed = false;
  }

  /* ---- frames ---- */
  let lastT = 0;
  function speedOf(f) {
    const h = f.hand;
    // palm lengths per second, over speedSpanMs: the wrist and fingertips' mean displacement
    while (recent.length && f.tMs - recent[0].t > o.speedSpanMs * 1.5) recent.shift();
    const a = f.aspect || 1;
    const pts = POINTS.map((i) => [h[3 * i] * a, h[3 * i + 1]]);
    const palm = Math.hypot((h[27] - h[0]) * a, h[28] - h[1]);
    recent.push({ t: f.tMs, pts, palm });
    const old = recent.find((r) => f.tMs - r.t <= o.speedSpanMs) ?? recent[0];
    const dt = (f.tMs - old.t) / 1000;
    if (old === recent[recent.length - 1] || !(dt > 0) || !(palm > 0)) return null;
    let d = 0;
    for (let k = 0; k < pts.length; k++) d += Math.hypot(pts[k][0] - old.pts[k][0], pts[k][1] - old.pts[k][1]);
    return d / pts.length / palm / dt;
  }
  function pushFrame(f) {
    if (!f || !Number.isFinite(f.tMs)) return;
    stats.frames++;
    const dtMs = lastT ? Math.min(100, Math.max(0, f.tMs - lastT)) : 0;
    lastT = f.tMs;
    indexSince ??= f.tMs;
    present = !!f.hand && Number.isFinite(f.hand[0]);
    if (present) {
      lastSeen = f.tMs;
      act = speedOf(f);
    } else {
      recent.length = 0;
      act = null;
    }
    const goneMs = present ? 0 : lastSeen === null ? Infinity : f.tMs - lastSeen;
    const moving = present && act !== null && act >= o.moveSpeed;
    const still = present && act !== null && act < o.stillSpeed;
    if (still) stillSince ??= f.tMs; else stillSince = null;

    if (!target || done) return;
    // a hand still up from the sign before must come down, or stop, before it can start a try
    if (!armed && (goneMs >= o.goneMs || (stillSince !== null && f.tMs - stillSince >= o.armStillMs))) armed = true;
    if (!attempt && armed && moving) {
      attempt = { t0: f.tMs, movingMs: 0, moved: false };
      stats.attempts++;
      emit({ type: "attempt-start", sign: signAt(idx), index: idx });
      if (isSeq()) emit({ type: "progress", index: idx, sign: signAt(idx) });
    }
    if (attempt) {
      if (moving) { attempt.movingMs += dtMs; attempt.moved = true; movedMs += dtMs; }
      if (mode === "auto" && movedMs >= o.acceptMovingMs) {
        const sign = signAt(idx);
        if (!isSeq() && (missesOf.get(sign) ?? 0) < missFirst) miss();
        else accept();
      } else if (goneMs >= o.goneMs || (attempt.moved && stillSince !== null && f.tMs - stillSince >= o.stillEndMs)) {
        // the try is over without a verdict: the keyboard decides; in auto mode a short try just ends
        attempt = null;
      }
    }
    if (isSeq() && !done && f.tMs - indexSince >= o.hintMs * (hintCount + 1)) {
      hintCount++;
      stats.hints++;
      emit({ type: "hint", index: idx, sign: signAt(idx), waitedMs: f.tMs - indexSince, count: hintCount });
    }
  }

  function setTarget(tg) {
    // checked before anything changes, as the real one does: a wrong name throws and leaves the target as it was
    const signs = !tg ? null : tg.mode === "sequence" ? [...(tg.signs || [])] : [tg.sign];
    if (signs && !signs.length) throw new Error("setTarget: no signs");
    if (signs && known) for (const s of signs) if (!known.has(s)) throw new Error(`setTarget: "${s}" is not one of the model's labels`);
    gen++;
    idx = 0;
    done = false;
    hintCount = 0;
    attempt = null;
    indexSince = null;
    movedMs = 0;
    armed = !present;
    target = signs ? { mode: tg.mode === "sequence" ? "sequence" : "isolated", signs } : null;
  }

  /* the keyboard, on a laptop: A accept, M miss, U unclear */
  let offKeys = null;
  if (mock.keys !== false && typeof window !== "undefined" && typeof window.addEventListener === "function") {
    const onKey = (e) => {
      if (e.repeat || e.metaKey || e.ctrlKey || e.altKey || /^(INPUT|TEXTAREA|SELECT)$/.test(e.target?.tagName ?? "")) return;
      const k = e.key?.toLowerCase();
      if (k === "a") accept();
      else if (k === "m") miss();
      else if (k === "u") unclear();
    };
    window.addEventListener("keydown", onKey);
    offKeys = () => window.removeEventListener("keydown", onKey);
  }

  return {
    setTarget,
    pushFrame,
    on(cb) { listeners.add(cb); return () => listeners.delete(cb); },
    setFlip(b) { side = !!b; },
    setCuts() { /* nothing to cut */ },
    get flip() { return side; },
    get target() { return target ? { ...target, signs: [...target.signs] } : null; },
    get state() {
      return { index: idx, done, armed, movedMs, attempt: attempt ? { movingMs: attempt.movingMs } : null, activity: act, present };
    },
    stats,
    timing: () => ({ mock: true, mode }),
    // the keyboard's verdicts, callable from tests and the console
    accept, miss, unclear,
    mode,
    close() { gen++; target = null; listeners.clear(); offKeys?.(); },
  };
}
