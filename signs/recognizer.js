/* signs/recognizer.js: from frame records to verdicts. Is the learner making the asked-for sign, and when?
 *
 * The lesson always knows which sign it wants (docs/signs-design.md), so the question put to the model is never
 * "what is this?" but "is this THANK-YOU?": p(target) against that sign's own cut. Pure logic, no DOM: the model
 * (a runner, signs/model-runner.js), the features (signs/features.js) and the words of a miss (signs/feedback.js)
 * are handed in, so all of it runs under node --test with a scripted model.
 *
 * TRIES. A try is found from the signing hand alone: its presence and how fast it moves, in shoulder widths per
 * second (the pose's shoulders; 2.75 palm lengths when they are not seen, the median ratio in PopSign's own clips,
 * where the hand is nearer the camera than the shoulders). "How fast" is the mean speed of the hand's five fastest
 * landmarks over 100 ms, because much of a sign is fingers and wrist while the palm stays put: in the middle of
 * PopSign's YES (a fist nodding) and BYE (fingers waving) the palm's centre is under 0.5 shoulder widths a second in
 * 30-73 % of frames, the five fastest landmarks in 0-10 %. In the 21 PopSign clips of the lesson signs (Deaf
 * signers, 30 fps) a still stretch inside a sign lasts at most 11 frames under 0.5 of these units, while a hand coming
 * into view or leaving it moves at 1.3 to 16 (medians 3.8 and 5.3).
 *   - A try starts when the hand moves (2 frames over moveSpeed) after it was down, or still, since the last
 *     verdict or setTarget: a hand still moving on from the sign before does not start one. Up to 0.5 s of the hand
 *     held still before the movement counts as the sign's starting place; a hand that came into view moving counts
 *     from its first frame.
 *   - It ends when the hand is down (out of view 250 ms), or still for 600 ms after moving, or 3 s after the
 *     movement began.
 *   - A miss after stillness waits: a beginner who stops at the starting place to check, then moves, is still in the
 *     same try. If the hand moves again within 1.5 s of stopping, the try goes on, scored both whole and from the new
 *     movement (the better counts), and nothing was said in between; if it stays still or goes down, or 1.5 s pass
 *     whatever it does, the miss is said.
 *   - A shorter stop (250 ms still) and then moving again is a new movement within the same try: from then on the try
 *     is also scored from it, and it is what the same sign twice in a row needs between the two.
 *
 * ISOLATED (the learn phase): the try so far is scored every ~120 ms and accepted early when p(target) is over the
 * sign's accept cut while the hand is slowing or has stopped, or twice in a row over its high cut. At the try's end
 * one more score decides: accept; or unclear (the hand out of the picture at the top or sides mid-try, found in too
 * little of it, or a try too short to judge); or a miss, with what it looked like (another sign the model gives 0.6)
 * and whatever the injected feedbackFn says. After an accept nothing more is said until setTarget.
 *
 * SEQUENCE (the conversation): windows of the last 24, 36 and 48 frames (0.8 to 1.6 s) are scored every ~100 ms
 * while a try is under way; the sign due next is accepted on two ticks in a row over its cut, or one over its high
 * cut (a tick's score is the best of the lengths scored then). Only the sign due next can be accepted, so signs out
 * of order never are. After an accept the windows start at that moment, so the sign just made cannot count again,
 * and the same sign twice in a row needs a new movement in between (the hand down, or a stop, then moving again;
 * no turn of this lesson asks for one, but a later one may). At a try's end the part made since the last
 * accept is scored once more (a sign slower than every window) and accepted if it is over the cut; it is said as a
 * miss only when the hand began that movement after the last accept, so the tail of a sign already accepted is never
 * a miss of the next. After ~8 s without the sign due, a hint, and again every 8 s.
 *
 * Every window is resampled to 30 frames a second by timestamp before features: the model was trained on 30 fps
 * video and a phone sends 24-30, unevenly. Sequence windows are cut to the hand's span as tries are (trimWindows):
 * the model learnt signs from clips cut to most of the hand's span (80-100 %, up to 3 frames either side), and a
 * window opening on frames without the hand, as one does whenever the hand has just come up, is a shape it never saw
 * as a sign (trimWindows: false scores the windows as they are). At most one model run is in flight from here: a
 * poll or tick that comes due while one runs is skipped, a try's last score waits its turn and is never skipped, and
 * when runs take longer than a tick, fewer window lengths are scored per tick, in turn.
 *
 * Events (rec.on(cb)):
 *   {type: "attempt-start", sign, index}
 *   {type: "progress", index, sign}                       sequence: the hand is moving toward sign `index`
 *   {type: "accept", sign, index, confidence, early}      index: position in the sequence (0 when isolated)
 *   {type: "sequence-done", signs}
 *   {type: "miss", sign, index, confidence, looksLike, looksLikeP, top, feedback}   feedback: feedbackFn's answer or null
 *   {type: "unclear", sign, index, reason}                reason: "hand-lost" | "too-short" | "out-of-frame"
 *   {type: "hint", index, sign, waitedMs, count}
 *   {type: "error", message}                              the model or features failed (at most one every 3 s)
 *   {type: "score", purpose, sign, p, T, ms}              only with options.debug: every model answer */

export const NONE_INDEX = 250;          // the model's 251st output, "not a whole sign" (tools/signs/train_signs.py NONE)
export const DEFAULT_CUT = Object.freeze({ accept: 0.5, high: 0.8 });

export const RECOGNIZER_DEFAULTS = Object.freeze({
  fps: 30,
  speedSpanMs: 100,       // speed over this much time (3 frames at 30 fps): one frame's jitter is not motion
  topLandmarks: 5,        // activity = the mean speed of this many fastest landmarks
  moveSpeed: 0.8,         // shoulder widths per second: at or above, moving
  stillSpeed: 0.5,        //                              below, still
  startFrames: 2,         // moving frames in a row that start a try
  goneMs: 250,            // the hand out of view this long: down
  stillEndMs: 600,        // still this long after moving: the try is over (accepted now, or the miss waits)
  stillMissMs: 1500,      // a miss after stillness is said once the hand has stayed still this long
  maxAttemptMs: 3000,     // from the latest movement's start
  armStillMs: 300,        // after a verdict or setTarget: still this long (or down goneMs) before a new try
  prerollMs: 500,         // the hand held still before moving: this much of it is the sign's starting place
  phaseStillMs: 250,      // still this long, then moving: a new movement within the try
  noiseFrames: 4,         // fewer frames with the hand than this is not a try (nothing is said)
  minFrames: 8,           // fewer than this is "too-short"
  minPresence: 0.6,       // the hand found in less of the try than this: "hand-lost"
  maxGap: 3,              // signfeat's MAX_GAP: shorter gaps in the hand are filled in, longer ones are holes
  maxHoleFrac: 0.2,       // holes over this fraction of the try: "hand-lost"
  edgeFrames: 3,          // beyond the picture's top or sides in this many frames mid-try: "out-of-frame"
  slowFrac: 0.5,          // slowing: activity under this fraction of the try's peak
  pollMs: 120,
  tickMs: 100,
  windows: Object.freeze([24, 36, 48]),
  minWindow: 12,          // a window shorter than this (just after an accept) waits for more frames
  trimWindows: true,      // sequence windows cut to the hand's span (+-2 frames) as tries are; false: as they are
  maxWindowMs: 4000,      // a try's whole window keeps at most its last 4 s
  hintMs: 8000,
  looksLikeMin: 0.6,      // "That looked like X" only when the model gives X this much (docs/signs-design.md)
  acceptOnChecks: true,   // a try whose hand shape, place and movement are all within the signers' range passes,
                          // unless the model gives another sign looksLikeMin or more (the owner, 24 Oct: "when all 3 of
                          // these things are green it should just pass")
  checksTopK: 5,          // ... and only when the model has the sign among its first 5 of the 250
  shoulderPerPalm: 2.75,  // the speed unit without shoulders: shoulder widths per palm length (wrist to middle knuckle)
  bufferMs: 8000,
  errorEveryMs: 3000,
  debug: false,
});

const defaultNow = () => (typeof performance !== "undefined" ? performance.now() : Date.now());

/* ---- motion: presence and speed of the signing hand, frame by frame -------------------------------------------- */

/**
 * @param {Partial<typeof RECOGNIZER_DEFAULTS>} [opts]
 * @returns {{push(frame: {tMs: number, aspect: number, hand: ArrayLike<number>|null, pose: ArrayLike<number>|null}):
 *   {t: number, present: boolean, act: number|null, moving: boolean, still: boolean, unit: number|null,
 *    presentSince: number|null, stillSince: number|null, moveRun: number, moveRunStart: number|null,
 *    lastSeen: number|null, goneMs: number}}}
 *   act: shoulder widths per second (null until 100 ms of the hand have been seen).
 */
export function createMotionTracker(opts = {}) {
  const o = { ...RECOGNIZER_DEFAULTS, ...opts };
  const recent = [];          // the last ~300 ms of hand frames: {t, pts: x * aspect, y for the 21 landmarks}
  let shoulder = null, shoulderT = -Infinity, palmUnit = null;
  let lastSeen = null, presentSince = null, stillSince = null, moveRun = 0, moveRunStart = null;
  const speeds = new Float64Array(21);
  const unitAt = (t) => (shoulder !== null && t - shoulderT < 2000 ? shoulder : palmUnit);

  return {
    push(f) {
      const t = f.tMs, aspect = f.aspect || 1, h = f.hand, p = f.pose;
      // shoulder width from the pose when both shoulders are seen, smoothed against the pose's own jitter
      if (p && p.length >= 39 && p[35] >= 0.5 && p[38] >= 0.5) {
        const w = Math.hypot((p[33] - p[36]) * aspect, p[34] - p[37]);
        if (w > 1e-3) { shoulder = shoulder === null ? w : shoulder + 0.1 * (w - shoulder); shoulderT = t; }
      }
      let act = null;
      if (h) {
        if (presentSince === null) presentSince = t;
        lastSeen = t;
        const pts = new Float64Array(42);
        for (let i = 0; i < 21; i++) { pts[2 * i] = h[3 * i] * aspect; pts[2 * i + 1] = h[3 * i + 1]; }
        const palm = Math.hypot(pts[18] - pts[0], pts[19] - pts[1]) * o.shoulderPerPalm;
        if (palm > 1e-3) palmUnit = palmUnit === null ? palm : palmUnit + 0.1 * (palm - palmUnit);
        const unit = unitAt(t);
        // the newest earlier frame at least speedSpanMs back (15 ms of slack for uneven frame times), within 250 ms
        let ref = null;
        for (let k = recent.length - 1; k >= 0; k--) {
          const dt = t - recent[k].t;
          if (dt >= o.speedSpanMs - 15) { if (dt <= 250) ref = recent[k]; break; }
        }
        if (ref && unit) {
          const dt = (t - ref.t) / 1000;
          for (let i = 0; i < 21; i++) speeds[i] = Math.hypot(pts[2 * i] - ref.pts[2 * i], pts[2 * i + 1] - ref.pts[2 * i + 1]) / dt / unit;
          const s = Array.from(speeds).sort((a, b) => b - a);
          let sum = 0;
          for (let i = 0; i < o.topLandmarks; i++) sum += s[i];
          act = sum / o.topLandmarks;
        }
        recent.push({ t, pts });
        while (recent.length && t - recent[0].t > 300) recent.shift();
      } else if (lastSeen !== null && t - lastSeen >= o.goneMs) {
        presentSince = null;
        recent.length = 0;
      }
      const moving = act !== null && act >= o.moveSpeed;
      const still = act !== null && act < o.stillSpeed;
      if (moving) { if (moveRun === 0) moveRunStart = t; moveRun++; } else if (h) moveRun = 0;
      if (still) { if (stillSince === null) stillSince = t; } else if (h || presentSince === null) stillSince = null;
      return {
        t, present: !!h, act, moving, still, unit: unitAt(t), presentSince, stillSince, moveRun, moveRunStart, lastSeen,
        goneMs: h ? 0 : lastSeen === null ? Infinity : t - lastSeen,
      };
    },
  };
}

/* ---- resampling to 30 fps by timestamp ------------------------------------------------------------------------- */

const lerp = (a, b, w) => {
  if (w <= 1e-6) return a;
  if (w >= 1 - 1e-6) return b;
  if (!a || !b) return null;    // a part missing on either side stays missing, as in signfeat.resample
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] + (b[i] - a[i]) * w;
  return out;
};

/**
 * Frame records at a steady rate, the grid ending on the newest frame (a window "ending now" holds the latest
 * picture). Linear between the two frames around each time; a hand or pose missing on either side stays missing.
 * The fallback when signs/features.js has no resampleToFps.
 * @param {{tMs: number, aspect: number, hand: Float32Array|null, pose: Float32Array|null}[]} frames  in time order
 */
export function resampleFrames(frames, fps = 30) {
  if (frames.length < 2) return frames.map((f) => ({ tMs: f.tMs, aspect: f.aspect, hand: f.hand, pose: f.pose }));
  const step = 1000 / fps, t0 = frames[0].tMs, t1 = frames[frames.length - 1].tMs;
  const n = Math.floor((t1 - t0) / step + 1e-6) + 1;
  const out = new Array(n);
  let j = 0;
  for (let k = 0; k < n; k++) {
    const t = t1 - (n - 1 - k) * step;
    while (j < frames.length - 2 && frames[j + 1].tMs <= t) j++;
    const a = frames[j], b = frames[j + 1];
    const w = b.tMs > a.tMs ? Math.min(1, Math.max(0, (t - a.tMs) / (b.tMs - a.tMs))) : 1;
    out[k] = { tMs: t, aspect: w < 0.5 ? a.aspect : b.aspect, hand: lerp(a.hand, b.hand, w), pose: lerp(a.pose, b.pose, w) };
  }
  return out;
}

/* signs/features.js is written alongside this file: its resampleToFps is used when it is there and answers a known
 * input with frame records, resampleFrames otherwise (said once on the console). */
function defaultResample() {
  const fallback = (why) => { console.warn(`recognizer: resampling with resampleFrames (${why})`); return resampleFrames; };
  return import("./features.js").then((m) => {
    if (typeof m.resampleToFps !== "function") return fallback("signs/features.js has no resampleToFps");
    try {
      const hand = new Float32Array(63), pose = new Float32Array(99);
      const r = m.resampleToFps([{ tMs: 0, aspect: 0.75, hand, pose }, { tMs: 100, aspect: 0.75, hand, pose }], 30);
      if (Array.isArray(r) && r.length >= 3 && r.length <= 5 && typeof r[0]?.tMs === "number" && "hand" in r[0]) return m.resampleToFps;
    } catch { /* below */ }
    return fallback("signs/features.js resampleToFps answers in another shape");
  }, () => fallback("signs/features.js did not load"));
}

function edgeOf(hand) {
  for (let i = 0; i < 21; i++) {
    const x = hand[3 * i], y = hand[3 * i + 1];
    if (x < 0 || x > 1 || y < 0) return true;
  }
  return false;
}

/* ---- the recognizer ------------------------------------------------------------------------------------------- */

/**
 * @param {object} o
 * @param {{run(x: Float32Array, T: number): Promise<{probs: ArrayLike<number>, ms?: number, wallMs?: number}|null>}} o.runner
 * @param {(frames: object[], ctx: {flip: boolean, attempt: boolean, hand: string}) => {x: Float32Array, T: number}|null} o.featuresFn
 *   frames: 30 fps frame records {tMs, aspect, hand, pose}. attempt: true asks for signfeat's attempt_window first (the
 *   hand's active span +-2 frames, squeezed to 64): every try is passed so, and so is every sequence window unless
 *   options.trimWindows is false. null: nothing to score. signs/features.js modelInput is this function.
 * @param {string[]} o.labels             the model's outputs in order (250 signs, then "none")
 * @param {Object<string, {accept?: number, high?: number}>} [o.cuts]   per sign, "*" for every sign; defaults 0.5 / 0.8
 * @param {"right"|"left"} [o.hand]       the signing hand; also the default flip (a left hand is mirrored)
 * @param {boolean|null} [o.flip]         the session's canonical side (decided at the camera check)
 * @param {() => number} [o.now]          a clock for latency figures (frames carry their own tMs)
 * @param {(attemptFrames: object[], targetSign: string, probs: ArrayLike<number>, ctx: object) => any} [o.feedbackFn]
 *   called on a miss with the try's frame records as captured; its answer (or its promise's) is the miss's feedback.
 * @param {(frames: object[], fps: number) => object[]} [o.resample]   default: features.js resampleToFps
 * @param {(sign: string) => boolean} [o.looksLikeOk]   which signs a miss may name (met in the lesson, neighbours)
 * @param {Partial<typeof RECOGNIZER_DEFAULTS>} [o.options]
 */
export function createRecognizer({ runner, featuresFn, labels, cuts = {}, hand = "right", flip = null, now = defaultNow,
                                   feedbackFn = null, resample = null, looksLikeOk = null, options = {} } = {}) {
  if (!runner || typeof runner.run !== "function") throw new Error("createRecognizer needs a runner");
  if (typeof featuresFn !== "function") throw new Error("createRecognizer needs featuresFn");
  if (!Array.isArray(labels) || labels.length < 2) throw new Error("createRecognizer needs the model's labels");
  const o = { ...RECOGNIZER_DEFAULTS, ...options };
  const index = new Map(labels.map((l, i) => [l, i]));
  const noneIndex = labels.indexOf("none") >= 0 ? labels.indexOf("none") : labels.length > NONE_INDEX ? NONE_INDEX : -1;
  let cutTable = { ...cuts };
  // a boolean always: features.js modelInput refuses anything else rather than guess a side
  let side = flip === null || flip === undefined ? hand === "left" : !!flip;
  let resampleFn = resample;
  const resampleReady = resample ? Promise.resolve(resample) : defaultResample().then((f) => (resampleFn = f));

  const listeners = new Set();
  const motion = createMotionTracker(o);
  const buffer = [];          // {f: {tMs, aspect, hand, pose}, present}
  let lastM = null;

  // the target, and how far the learner has got
  let gen = 0, target = null, idx = 0, done = false, fence = -Infinity, indexSince = 0, hintCount = 0;
  let armed = false, attempt = null, attemptIds = 0, needNewMovement = false, prevTickOver = false;
  const progressSent = new Set();
  // the one lane to the model
  let laneBusy = false, closed = false, lastError = -Infinity, lastTickT = -Infinity, rotate = 0;
  const recentRuns = [];
  const finals = [];

  const stats = {
    frames: 0, attempts: 0, accepts: 0, earlyAccepts: 0, misses: 0, unclear: 0, polls: 0, pollsSkipped: 0,
    ticks: 0, ticksSkipped: 0, runs: 0, dropped: 0, errors: 0, windowsPerTick: o.windows.length, runMs: null,
    inFlight: 0, maxInFlight: 0, verdictMs: [], pushMs: [], prepMs: [], featMs: [],
  };
  const keep = (a, v) => { a.push(v); if (a.length > 300) a.shift(); };

  const emit = (e) => {
    for (const cb of [...listeners]) {
      try { cb(e); } catch (err) { console.warn(`recognizer: a listener threw (${err?.message || err})`); }
    }
  };
  const cutFor = (sign) => ({ ...DEFAULT_CUT, ...(cutTable["*"] || {}), ...(cutTable[sign] || {}) });
  const signAt = (i) => target?.signs[i];
  const isSeq = () => target?.mode === "sequence";

  /* ---- scoring ---- */

  function framesBetween(t0, t1) {
    const out = [];
    for (const e of buffer) if (e.f.tMs >= t0 && e.f.tMs <= t1) out.push(e.f);
    return out;
  }
  async function to30(frames) {
    const fn = resampleFn ?? (await resampleReady);
    const t = now();
    const out = frames.length > 1 ? fn(frames, o.fps) : frames;
    keep(stats.prepMs, now() - t);
    return out;
  }
  /* One model run on 30 fps frames: p(sign) and everything the verdict needs. null when there is nothing to score
   * or the runner dropped it. */
  async function score(f30, signIdx, purpose, raw = f30, attempt = true) {
    const tf = now();
    const feat = featuresFn(f30, { flip: side, attempt, hand });
    keep(stats.featMs, now() - tf);
    if (!feat || !(feat.T > 0)) return null;
    stats.inFlight++;
    stats.maxInFlight = Math.max(stats.maxInFlight, stats.inFlight);
    let r;
    try { r = await runner.run(feat.x, feat.T); } finally { stats.inFlight--; }
    if (!r) { stats.dropped++; return null; }
    stats.runs++;
    const ms = r.wallMs ?? r.ms;
    if (Number.isFinite(ms)) {
      // the median of the last few: one stalled run (a busy phone, a garbage collection) must not set the pace
      recentRuns.push(ms);
      if (recentRuns.length > 7) recentRuns.shift();
      stats.runMs = [...recentRuns].sort((x, y) => x - y)[(recentRuns.length - 1) >> 1];
    }
    const p = r.probs[signIdx] ?? 0;
    if (o.debug) emit({ type: "score", purpose, sign: labels[signIdx], p, T: feat.T, ms: r.ms });
    return { p, probs: r.probs, frames: raw, f30 };
  }
  const better = (a, b) => (!a ? b : !b ? a : b.p > a.p ? b : a);

  function launch(job) {
    laneBusy = true;
    const g = gen;
    Promise.resolve().then(() => job(g)).catch((err) => {
      stats.errors++;
      const t = now();
      if (t - lastError >= o.errorEveryMs) { lastError = t; emit({ type: "error", message: String(err?.message || err) }); }
    }).finally(() => {
      laneBusy = false;
      if (finals.length && !closed) launch(finals.shift());
    });
  }
  function enqueueFinal(job) {
    if (laneBusy) finals.push(job);
    else launch(job);
  }

  /* ---- verdicts ---- */

  function looksLike(probs, signIdx) {
    let best = -1, bp = 0;
    for (let i = 0; i < labels.length; i++) {
      if (i === signIdx || i === noneIndex) continue;
      if (probs[i] > bp) { bp = probs[i]; best = i; }
    }
    if (best < 0 || bp < o.looksLikeMin || (looksLikeOk && !looksLikeOk(labels[best]))) return { looksLike: null, looksLikeP: bp };
    return { looksLike: labels[best], looksLikeP: bp };
  }
  const top3 = (probs) => Array.from(probs, (p, i) => [i, p]).sort((a, b) => b[1] - a[1]).slice(0, 3)
    .map(([i, p]) => ({ sign: labels[i] ?? String(i), p }));

  /* What kept a try the model did not accept from being judged, from the captured frames themselves. */
  function unclearReason(frames) {
    let hands = 0;
    for (const f of frames) if (f.hand) hands++;
    if (hands < o.minFrames) return "too-short";
    // a hand going down leaves by the bottom or a lower corner at its very end (as PopSign's signers' do), so the
    // first and last 4 frames with the hand do not count
    let seen = 0, edges = 0;
    for (const f of frames) {
      if (!f.hand) continue;
      seen++;
      if (seen > 4 && seen <= hands - 4 && edgeOf(f.hand)) edges++;
    }
    if (edges >= o.edgeFrames) return "out-of-frame";
    // lost: found in too little of the try, or missing for longer than features can bridge (signfeat fills gaps of
    // at most 3 frames; a longer one is a hole the model sees as no hand) in a fifth of it
    const first = frames.findIndex((f) => f.hand);
    let last = frames.length - 1;
    while (last > first && !frames[last].hand) last--;
    const span = last - first + 1;
    let holes = 0, run = 0;
    for (let k = first; k <= last; k++) {
      if (frames[k].hand) { if (run > o.maxGap) holes += run; run = 0; } else run++;
    }
    if (hands / span < o.minPresence || holes / span >= o.maxHoleFrac) return "hand-lost";
    return null;
  }

  function endAttempt(a) {
    a.state = "done";
    if (attempt === a) attempt = null;
    // a new try needs the hand down or still first (setTarget's rule), so the end of this one cannot start it; a try
    // that ended because the hand stopped or went down has had exactly that, and the next movement may start one
    armed = !!a.rested || resting();
  }
  const resting = () => !!lastM && (lastM.goneMs >= o.goneMs || (lastM.stillSince !== null && lastM.t - lastM.stillSince >= o.armStillMs));
  async function say(a, best, why, g) {
    if (g !== gen || a !== attempt) return;
    const sign = signAt(idx), signIdx = index.get(sign);
    if (why) {
      stats.unclear++;
      endAttempt(a);
      emit({ type: "unclear", sign, index: idx, reason: why });
    } else {
      const probs = best?.probs ?? new Float32Array(labels.length);
      let feedback = null;
      if (feedbackFn) {
        try {
          feedback = await feedbackFn(best?.frames ?? [], sign, probs, { flip: side, hand, frames30: best?.f30 ?? [], labels, index: idx });
        } catch (err) { console.warn(`recognizer: feedbackFn threw (${err?.message || err})`); }
        if (g !== gen || a !== attempt) return;
      }
      // all three parts green and no other sign the model is sure of: the sign was made (the model alone can miss a
      // learner's sign that is within fluent signers' range on every part; a green checklist under "Not quite" is a
      // contradiction the learner cannot act on)
      const c = feedback?.checks;
      if (o.acceptOnChecks && c && c.handshape === true && c.location === true && c.movement === true) {
        let other = 0, above = 0;
        for (let i = 0; i < labels.length; i++) {
          if (i === signIdx || i === noneIndex) continue;
          if (probs[i] > other) other = probs[i];
          if (probs[i] > probs[signIdx]) above++;
        }
        // and the model has the sign among its first checksTopK: a try the model is sure is something else is not
        // passed on the checks alone (TALK passed WATER's checks with WATER at 0.00)
        if (other < o.looksLikeMin && above < o.checksTopK) {
          stats.acceptsByChecks = (stats.acceptsByChecks ?? 0) + 1;
          accept(a, best?.p ?? 0, false, lastM?.t ?? now(), true);
          return;
        }
      }
      // the parts all looked right but the model is sure it was not the sign: three green checks under "Not quite"
      // would contradict each other, so the checks are left out and the learner is sent back to the video
      if (c && c.handshape === true && c.location === true && c.movement === true) {
        feedback = { ...feedback, text: "Almost! Watch it once more, then sign it just like the video.", checks: { handshape: null, location: null, movement: null }, part: null };
      }
      stats.misses++;
      endAttempt(a);
      emit({ type: "miss", sign, index: idx, confidence: best?.p ?? 0, ...looksLike(probs, signIdx), top: top3(probs), feedback: feedback ?? null });
    }
    if (a.tDetect !== undefined) keep(stats.verdictMs, now() - a.tDetect);
  }
  function accept(a, p, early, tAt, byChecks = false) {
    const i = idx, sign = signAt(i);
    stats.accepts++;
    if (early) stats.earlyAccepts++;
    if (a && a.tDetect !== undefined) keep(stats.verdictMs, now() - a.tDetect);
    if (!isSeq()) {
      done = true;
      if (a) { a.state = "done"; attempt = null; }
      emit({ type: "accept", sign, index: i, confidence: p, early, byChecks });
      return;
    }
    fence = tAt;
    idx++;
    prevTickOver = false;
    indexSince = tAt;
    hintCount = 0;
    // a try that was ending is over; one still under way goes on toward the next sign
    if (a && a.state !== "active") endAttempt(a);
    if (idx >= target.signs.length) {
      done = true;
      if (attempt) { attempt.state = "done"; attempt = null; }
    } else {
      needNewMovement = target.signs[idx] === sign;
    }
    // a listener may set the next target from inside the accept: then this one's sequence-done is not said
    const g = gen, finished = done, signs = [...target.signs];
    emit({ type: "accept", sign, index: i, confidence: p, early, byChecks });
    if (finished && g === gen) emit({ type: "sequence-done", signs });
  }

  /* ---- the try under way ---- */

  function startAttempt(m, t) {
    const onset = m.moveRunStart ?? t;
    const tWin = Math.max(m.presentSince ?? onset, onset - o.prerollMs);
    attempt = {
      id: ++attemptIds, state: "active", tStart: onset, tWin, tPhase: onset, tWinPhase: tWin, phases: 1,
      peakAct: m.act ?? 0, movingFrames: 0, lastPollT: -Infinity, highRun: 0, paused: null, tEnd: null, tDetect: undefined,
      stillFrom: null,
    };
    stats.attempts++;
    prevTickOver = false;
    if (isSeq() && needNewMovement && onset > fence) needNewMovement = false;
    emit({ type: "attempt-start", sign: signAt(idx), index: idx });
  }
  /* The windows of a try ending at tEnd: whole (from its starting place) and, after a pause, from the newest
   * movement. Sequence windows never reach back past the last accept. */
  function attemptWindows(a, tEnd) {
    const lead = 2.5 * (1000 / o.fps);            // signfeat's +-2 frames, and half a frame for the grid
    const from = (t0) => Math.max(t0 - lead, tEnd - o.maxWindowMs, isSeq() ? fence : -Infinity);
    const ws = [framesBetween(from(a.tWin), tEnd + lead)];
    if (a.phases > 1 && a.tWinPhase > a.tWin) ws.push(framesBetween(from(a.tWinPhase), tEnd + lead));
    return ws;
  }

  function endNow(a, reason, tEnd) {
    a.state = "ending";
    a.tEnd = tEnd;
    a.tDetect = now();
    a.rested = reason !== "long";
    enqueueFinal(async (g) => {
      if (g !== gen || a !== attempt) return;
      try {
        const sign = signAt(idx), signIdx = index.get(sign), cut = cutFor(sign);
        // sequence: a miss of the next sign needs a movement begun after the last accept, and so does the same sign
        // again (what is left of the one just accepted must not count twice)
        const mayMiss = !isSeq() || a.tPhase > fence;
        if (isSeq() && needNewMovement && !mayMiss) { endAttempt(a); return; }
        const ws = attemptWindows(a, tEnd);
        let hands = 0;
        for (const f of ws[0]) if (f.hand) hands++;
        if (hands < o.noiseFrames || (!mayMiss && hands < o.minFrames)) { endAttempt(a); return; }   // a glimpse
        if (hands < o.minFrames) { await say(a, { p: 0, frames: ws[0], f30: ws[0], probs: null }, "too-short", g); return; }
        let best = null;
        for (const w of ws) {
          const s = await score(await to30(w), signIdx, "final", w);
          if (g !== gen || a !== attempt) return;
          best = better(best, s);
        }
        if (best && best.p >= cut.accept) { accept(a, best.p, false, tEnd); return; }
        if (!mayMiss) { endAttempt(a); return; }
        const why = unclearReason(best?.frames ?? ws[0]);
        if (reason === "still") {
          // not yet: the hand may move on from a pause at the starting place
          a.paused = { best, why };
          a.state = "paused";
          return;
        }
        await say(a, best, why, g);
      } catch (err) {
        if (a === attempt && a.state !== "done") endAttempt(a);
        throw err;
      }
    });
  }

  function poll(a, m, t) {
    a.lastPollT = t;
    stats.polls++;
    const sign = signAt(idx), signIdx = index.get(sign), cut = cutFor(sign);
    const slowing = m.present && m.act !== null && (m.still || m.act < o.slowFrac * a.peakAct);
    const ws = attemptWindows(a, t);
    launch(async (g) => {
      let best = null;
      for (const w of ws) {
        const s = await score(await to30(w), signIdx, "poll", w);
        if (g !== gen || a !== attempt || a.state !== "active") return;
        best = better(best, s);
      }
      if (!best) return;
      a.highRun = best.p >= cut.high ? a.highRun + 1 : 0;
      if ((best.p >= cut.accept && slowing) || a.highRun >= 2) accept(a, best.p, true, t);
    });
  }

  function tick(t) {
    lastTickT = t;
    stats.ticks++;
    const i = idx, sign = signAt(i), signIdx = index.get(sign), cut = cutFor(sign);
    // as many window lengths as fit in a tick at the speed runs have been taking, the rest on the next ticks
    const n = o.windows.length;
    const per = stats.runMs ? Math.max(1, Math.min(n, Math.floor(o.tickMs / stats.runMs))) : n;
    stats.windowsPerTick = per;
    const sizes = [];
    for (let k = 0; k < per; k++) sizes.push(o.windows[(rotate + k) % n]);
    rotate = (rotate + per) % n;
    sizes.sort((x, y) => x - y);
    const raw = framesBetween(Math.max(fence, t - (sizes[sizes.length - 1] + 1) * (1000 / o.fps)), t);
    launch(async (g) => {
      const f30 = await to30(raw);
      const seen = new Set();
      let best = null;
      for (const N of sizes) {
        const from = Math.max(0, f30.length - N);
        // windows that reach back past the hand's first frame come down to the same span once trimmed (signfeat's
        // active span +-2), as at the start of every try: one run for all of them
        let a = from, b = f30.length - 1;
        while (a <= b && !f30[a].hand) a++;
        while (b >= a && !f30[b].hand) b--;
        if (a > b) continue;
        const key = o.trimWindows ? `${Math.max(from, a - 2)}:${Math.min(f30.length - 1, b + 2)}` : `${from}`;
        if (f30.length - from < o.minWindow || b - a + 1 < o.minWindow || seen.has(key)) continue;
        seen.add(key);
        const w = f30.slice(from);
        const s = await score(w, signIdx, "tick", w, o.trimWindows);
        if (g !== gen || idx !== i || done) return;
        best = better(best, s);
      }
      if (!best) return;
      const over = best.p >= cut.accept;
      if (best.p >= cut.high || (over && prevTickOver)) { accept(attempt, best.p, true, t); return; }
      prevTickOver = over;
    });
  }

  /* ---- frames ---- */

  function pushFrame(frame) {
    if (closed || !frame) return;
    const tp = now();
    const t = Number.isFinite(frame.tMs) ? frame.tMs : tp;
    // copies: a capture loop may reuse its arrays from frame to frame
    const f = { tMs: t, aspect: frame.aspect || 1, hand: frame.hand ? Float32Array.from(frame.hand) : null,
                pose: frame.pose ? Float32Array.from(frame.pose) : null };
    const m = motion.push(f);
    lastM = m;
    buffer.push({ f, present: m.present });
    while (buffer.length && t - buffer[0].f.tMs > o.bufferMs) buffer.shift();
    stats.frames++;
    if (target && indexSince === null) { indexSince = t; fence = t; }
    if (target && !done) step(m, t);
    keep(stats.pushMs, now() - tp);
  }

  function step(m, t) {
    const g = gen;
    if (t - indexSince >= o.hintMs * (hintCount + 1)) {
      hintCount++;
      emit({ type: "hint", index: idx, sign: signAt(idx), waitedMs: Math.round(t - indexSince), count: hintCount });
      if (g !== gen) return;      // a listener set another target: this frame starts its clock next time
    }
    if (!armed) {
      if (!(m.goneMs >= o.goneMs || (m.stillSince !== null && t - m.stillSince >= o.armStillMs))) return;
      armed = true;
      // sequence windows start from here: what the hand did before it settled belongs to the sign before
      if (isSeq() && !attempt) fence = Math.max(fence, t - o.armStillMs);
    }
    track(m, t);
    if (g !== gen) return;
    // sequence: the chip of the sign due next can light up as soon as the hand moves toward it
    const a = attempt;
    if (isSeq() && !done && !progressSent.has(idx) && a && a.state === "active" && m.moving && t > fence &&
        (!needNewMovement || a.tPhase > fence)) {
      progressSent.add(idx);
      emit({ type: "progress", index: idx, sign: signAt(idx) });
    }
  }

  function track(m, t) {
    const a = attempt;
    if (!a) {
      if (m.present && m.moveRun >= o.startFrames) startAttempt(m, t);
      return;
    }
    if (a.state === "ending" || a.state === "done") return;
    if (m.present) {
      if (m.moving) a.movingFrames++;
      if (m.act !== null) a.peakAct = Math.max(a.peakAct, m.act);
      if (m.stillSince !== null) a.stillFrom = m.stillSince;
    }
    // the hand moving again after a pause is a new movement of the same try: from then on the try is also scored
    // from it, and it is the "new movement" that the same sign twice in a row needs
    const resumed = m.present && m.moveRun === o.startFrames;
    if (resumed && (a.state === "paused" || (a.stillFrom !== null && (m.moveRunStart ?? t) - a.stillFrom >= o.phaseStillMs))) {
      a.state = "active";
      a.paused = null;
      a.tPhase = m.moveRunStart ?? t;
      a.tWinPhase = Math.max(m.presentSince ?? a.tPhase, a.tPhase - o.prerollMs);
      a.phases++;
      a.highRun = 0;
      a.lastPollT = -Infinity;
      a.stillFrom = null;
      if (isSeq() && needNewMovement && a.tPhase > fence) needNewMovement = false;
      return;
    }
    if (resumed) a.stillFrom = null;
    if (a.state === "paused") {
      // said when the hand goes down or stays still, or once the pause has lasted as long whatever the hand does
      if ((!m.present && m.goneMs >= o.goneMs) || (m.stillSince !== null && t - m.stillSince >= o.stillMissMs) ||
          t - a.tEnd >= o.stillMissMs) {
        const { best, why } = a.paused;
        a.rested = resting();
        a.state = "ending";
        a.tDetect = now();
        enqueueFinal((g) => say(a, best, why, g));
      }
      return;
    }
    if (!m.present && m.goneMs >= o.goneMs) { endNow(a, "down", m.lastSeen ?? t); return; }
    if (m.present && a.movingFrames > 0 && m.stillSince !== null && t - m.stillSince >= o.stillEndMs) {
      endNow(a, "still", Math.min(t, m.stillSince + 200));
      return;
    }
    if (t - a.tPhase >= o.maxAttemptMs) { endNow(a, "long", t); return; }
    if (isSeq()) {
      if ((needNewMovement && !(a.tPhase > fence)) || t - lastTickT < o.tickMs) return;
      if (laneBusy || finals.length) { stats.ticksSkipped++; return; }
      tick(t);
    } else {
      if (t - a.lastPollT < o.pollMs) return;
      let hands = 0;
      for (let k = buffer.length - 1; k >= 0 && buffer[k].f.tMs >= a.tWin; k--) if (buffer[k].present) hands++;
      if (hands < o.minFrames) return;
      if (laneBusy || finals.length) { stats.pollsSkipped++; return; }
      poll(a, m, t);
    }
  }

  /* ---- the controls ---- */

  function setTarget(tg) {
    // checked before anything changes: a wrong name throws and leaves the current target as it was
    const signs = !tg ? null : tg.mode === "sequence" ? [...(tg.signs || [])] : [tg.sign];
    if (signs && !signs.length) throw new Error("setTarget: no signs");
    if (signs) for (const s of signs) if (!index.has(s)) throw new Error(`setTarget: "${s}" is not one of the model's labels`);
    gen++;
    finals.length = 0;
    if (attempt) attempt.state = "done";
    attempt = null;
    done = false;
    idx = 0;
    hintCount = 0;
    prevTickOver = false;
    needNewMovement = false;
    progressSent.clear();
    lastTickT = -Infinity;
    // the target's clock (hints, and where sequence windows may start) starts on the next frame: frames carry their
    // own times, and the last one buffered may be long ago or none at all
    fence = Infinity;
    indexSince = null;
    if (!signs) { target = null; return; }
    target = { mode: tg.mode === "sequence" ? "sequence" : "isolated", signs };
    // a hand still up from the sign before must come down, or stop, before it can start a try at this one
    armed = !lastM || lastM.goneMs >= o.goneMs;
  }

  return {
    setTarget,
    pushFrame,
    on(cb) { listeners.add(cb); return () => listeners.delete(cb); },
    setFlip(b) { side = !!b; },
    setCuts(c) { cutTable = { ...c }; },
    get flip() { return side; },
    get target() { return target ? { ...target, signs: [...target.signs] } : null; },
    get state() {
      return { index: idx, done, armed, fence, laneBusy, finals: finals.length,
               attempt: attempt ? { id: attempt.id, state: attempt.state, phases: attempt.phases } : null,
               activity: lastM?.act ?? null, present: lastM?.present ?? false };
    },
    stats,
    timing() {
      const q = (arr, p) => { const b = [...arr].sort((x, y) => x - y); return b.length ? +b[Math.floor(p * (b.length - 1))].toFixed(2) : null; };
      return { verdictP50: q(stats.verdictMs, 0.5), verdictP95: q(stats.verdictMs, 0.95), pushP50: q(stats.pushMs, 0.5),
               pushP95: q(stats.pushMs, 0.95), resampleP95: q(stats.prepMs, 0.95), featuresP50: q(stats.featMs, 0.5),
               featuresP95: q(stats.featMs, 0.95), runMs: stats.runMs === null ? null : +stats.runMs.toFixed(2),
               windowsPerTick: stats.windowsPerTick, ticks: stats.ticks, ticksSkipped: stats.ticksSkipped, polls: stats.polls,
               pollsSkipped: stats.pollsSkipped, runs: stats.runs, dropped: stats.dropped, maxInFlight: stats.maxInFlight };
    },
    close() { closed = true; gen++; finals.length = 0; listeners.clear(); },
  };
}
