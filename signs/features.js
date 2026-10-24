/* signs/features.js: the numbers the sign model reads, made in the browser. The exact twin of
 * tools/signs/signfeat.py.
 *
 * The model learnt from numbers signfeat.py made, so it can only be trusted with numbers made the same way: the same
 * steps, in the same order, rounded the same way. numpy keeps a window's landmarks in float32, so every float32 step
 * there is a Math.fround here; where numpy quietly works in float64 (the weights that fill a gap, an attempt squeezed
 * to 64 frames by resample) this works in float64 too. tools/signs/parity.py runs signfeat on real clips and on the
 * hard cases (the hand lost, gaps of 1-5 frames, no shoulders, only the eyes, one frame, both sides, three picture
 * shapes) and tests/signs-features.test.js checks that this file gives the same numbers (to 1e-5; in fact, the same
 * bits).
 *
 * The steps (signfeat's docstring says why each is there):
 *   canonical    x * aspect (pixel-like units); mirrored when `flip`, so the signing hand is always on the image's
 *                left; the pose's left/right pairs sorted so that slot a is the image-left one.
 *   fillGaps     the hand lost for at most MAX_GAP frames in a row, with a frame that has it on both sides: filled
 *                linearly.
 *   bodyFrame    origin = the shoulders' midpoint, unit = shoulder width (the window's medians); else from the eyes
 *                and nose; else from the hand.
 *   features     per frame: the hand in the body frame (42), the hand around its wrist in palm lengths (40), 13
 *                upper-body points (26), then the lag-1 and lag-2 differences of those 108, NaN -> 0, clipped to
 *                +-10, then 2 presence flags (hand, shoulders): N_FEAT = 326 numbers.
 *   attemptWindow  an isolated attempt: the hand's active span (+-2 frames), squeezed to MAX_T frames when longer.
 *
 * Arrays are flat, frame after frame, and the typed array's class stands for numpy's dtype:
 *   canonical hand  Float32Array(T * HAND_W)  21 landmarks x (x, y)
 *   canonical pose  Float32Array(T * POSE_W)  POSE_KEEP's 13 points x (x, y)
 * with NaN where a part was not found. resample returns a Float64Array, because numpy promotes its result to float64,
 * and every function here then works in float64 on it, as signfeat does.
 *
 * What only the browser needs (no Python twin: the training video is already about 30 fps):
 *   resampleToFps         camera frames at uneven times -> frames at exact 1/30 s steps
 *   canonicalFromRecords  frame records {tMs, aspect, hand, other, pose} (docs/signs-design.md) -> canonical arrays
 *
 * An isolated attempt, from the camera's frame records to the model's input:
 *   const c = canonicalFromRecords(resampleToFps(records), flip);
 *   const w = attemptWindow(c.h, c.p);
 *   const { x, T } = features(w.h, w.p);     // x: Float32Array(T * N_FEAT), the model's input [1, T, 326]
 * which is modelInput(resampleToFps(records), { flip }), with null instead of an error when the hand was never found.
 * A conversation's window is the same without attemptWindow: the last 24, 36 or 48 resampled frames. */

export const FPS = 30;
export const MAX_T = 64;
export const MAX_GAP = 3;
export const POSE_KEEP = Object.freeze([0, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
/* Positions in POSE_KEEP of each left/right pair: eyes, ears, mouth corners, shoulders, elbows, wrists. */
export const PAIRS = Object.freeze([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]].map((pair) => Object.freeze(pair)));
export const SH_A = 7, SH_B = 8;   // shoulders in POSE_KEEP
export const EYE_A = 1, EYE_B = 2;
export const NOSE = 0;
export const N_BASE = 21 * 2 + 20 * 2 + 13 * 2;
export const N_FEAT = 3 * N_BASE + 2;
export const CLIP = 10.0;

/* Numbers per frame in the canonical arrays; and in a frame record, whose hand has x, y, z per landmark and whose pose
 * has x, y, visibility (only x and y are ever read: the model never sees depth or visibility). */
export const HAND_W = 21 * 2;
export const POSE_W = POSE_KEEP.length * 2;
const RECORD_HAND = 21 * 3;
const RECORD_POSE = 33 * 3;

/* float32 arithmetic. A double holds the exact product of two float32 numbers, and holds their sum, quotient or square
 * root closely enough that rounding it to float32 gives the correctly rounded float32 answer (53 >= 2 * 24 + 2 bits),
 * so Math.fround after each step IS numpy's float32 arithmetic, bit for bit. In a float64 array nothing is rounded. */
const f32 = Math.fround;
const f64 = (v) => v;
const roundingOf = (a) => (a instanceof Float64Array ? f64 : f32);
/* A number that is not there is NaN, as numpy reads None. JavaScript arithmetic reads null as 0, which would put a
 * missing landmark in the picture's corner and hand the model a plausible, wrong hand. */
const value = (v) => (typeof v === "number" ? v : NaN);

function framesIn(a, width, what) {
  if (!(a instanceof Float32Array || a instanceof Float64Array) || a.length % width !== 0) {
    const got = a?.constructor?.name ?? typeof a;
    throw new Error(`${what}: expected a Float32Array or Float64Array of frames of ${width} numbers, got ${got} of length ${a?.length}`);
  }
  return a.length / width;
}

/* np.linalg.norm(d, axis=-1) of one 2-vector whose parts are already rounded: the squares, their sum and the root,
 * each rounded in the array's dtype. Not Math.hypot, which rounds differently. */
const norm2 = (dx, dy, r) => r(Math.sqrt(r(r(dx * dx) + r(dy * dy))));

/* np.median: NaN when any value is NaN; else the middle value, or the mean of the two middle ones (their sum rounded,
 * then halved, as np.mean does it). */
function median(values, r) {
  if (values.some(Number.isNaN)) return NaN;
  const s = Float64Array.from(values).sort();
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : r(r(s[m - 1] + s[m]) / 2);
}

/**
 * signfeat.canonical, on the browser's per-frame arrays.
 *
 * @param {(Float32Array|null)[]} hands  per frame: the signing hand, 21 x (x, y, z) in MediaPipe's normalised
 *                                        coordinates of the raw (unmirrored) picture, or null when it was not found
 * @param {(Float32Array|null)[]} poses  per frame: 33 x (x, y, visibility), or null
 * @param {number|number[]} aspect       the picture's width / height: one number for the window, as signfeat takes it,
 *                                        or one per frame (a phone turned on its side mid-window changes its picture)
 * @param {boolean} flip                 mirror, so that the signing hand is on the image's left
 * @returns {{h: Float32Array, p: Float32Array, T: number}}  hand [T, 21, 2] and pose [T, 13, 2], NaN where missing
 */
export function canonical(hands, poses, aspect, flip) {
  const T = hands.length;
  if (poses.length !== T) throw new Error(`canonical: ${T} hand frames but ${poses.length} pose frames`);
  const h = new Float32Array(T * HAND_W).fill(NaN);
  const p = new Float32Array(T * POSE_W).fill(NaN);
  for (let t = 0; t < T; t++) {
    const shape = typeof aspect === "number" ? aspect : aspect[t];
    // A missing aspect would turn every x into NaN, which the model would read as "no hand": fail loudly instead.
    if (!(shape > 0 && shape < Infinity)) throw new Error(`canonical: frame ${t}: aspect must be the picture's width / height, got ${shape}`);
    // numpy multiplies a float32 array by a Python float in float32: the aspect is rounded to float32 first.
    const a = f32(shape);
    const hand = hands[t], pose = poses[t];
    if (hand != null) {
      if (hand.length !== RECORD_HAND) throw new Error(`canonical: frame ${t}: a hand is ${RECORD_HAND} numbers (21 x x, y, z), got ${hand.length}`);
      const o = t * HAND_W;
      for (let i = 0; i < 21; i++) {
        const x = f32(f32(value(hand[3 * i])) * a);
        h[o + 2 * i] = flip ? -x : x;
        h[o + 2 * i + 1] = value(hand[3 * i + 1]);   // stored as float32, as np.array(..., dtype=np.float32) rounds it
      }
    }
    if (pose != null) {
      if (pose.length !== RECORD_POSE) throw new Error(`canonical: frame ${t}: a pose is ${RECORD_POSE} numbers (33 x x, y, visibility), got ${pose.length}`);
      const o = t * POSE_W;
      for (let j = 0; j < POSE_KEEP.length; j++) {
        const k = POSE_KEEP[j];
        const x = f32(f32(value(pose[3 * k])) * a);
        p[o + 2 * j] = flip ? -x : x;
        p[o + 2 * j + 1] = value(pose[3 * k + 1]);
      }
      // Slot a is the image-left one of each pair, whatever the camera did. A tie, or a pair with a missing x, stays.
      for (const [sa, sb] of PAIRS) {
        const ia = o + 2 * sa, ib = o + 2 * sb;
        if (p[ia] > p[ib]) {
          const x = p[ia], y = p[ia + 1];
          p[ia] = p[ib]; p[ia + 1] = p[ib + 1];
          p[ib] = x; p[ib + 1] = y;
        }
      }
    }
  }
  return { h, p, T };
}

/* canonical() on frame records (docs/signs-design.md): each record's own hand, pose and aspect. The arrays are those
 * signfeat.canonical would make from the same landmarks; on records that all share one aspect, the same numbers. */
export function canonicalFromRecords(records, flip) {
  return canonical(records.map((r) => r.hand ?? null), records.map((r) => r.pose ?? null), records.map((r) => r.aspect), flip);
}

/* signfeat.fill_gaps. A frame counts as found when its wrist's x is a number. Returns the same array when nothing is
 * filled, else a filled copy. numpy weighs the two frames in float64 here (w = k / (gap + 1) is a float64 scalar, which
 * NumPy 2 does not demote) and rounds once, on storing into the float32 array; so does this. */
export function fillGaps(h, maxGap = MAX_GAP) {
  const T = framesIn(h, HAND_W, "fillGaps");
  const found = [];
  for (let t = 0; t < T; t++) if (!Number.isNaN(h[t * HAND_W])) found.push(t);
  if (found.length === T || found.length === 0) return h;
  const out = h.slice();
  for (let j = 1; j < found.length; j++) {
    const i0 = found[j - 1], i1 = found[j], gap = i1 - i0 - 1;
    if (gap > 0 && gap <= maxGap) {
      for (let k = 1; k <= gap; k++) {
        const w = k / (gap + 1);
        for (let c = 0; c < HAND_W; c++) out[(i0 + k) * HAND_W + c] = (1 - w) * h[i0 * HAND_W + c] + w * h[i1 * HAND_W + c];
      }
    }
  }
  return out;
}

/**
 * signfeat.body_frame: the window's origin and unit.
 * @returns {{mid: Float32Array, width: number}}  mid is float32 (signfeat casts it); width is the Python float it
 *   returns, which is the float32 median itself for the shoulders, 3.5 eye-spans, or max(4 palm lengths, 1e-3).
 */
export function bodyFrame(h, p) {
  const T = framesIn(p, POSE_W, "bodyFrame pose");
  if (framesIn(h, HAND_W, "bodyFrame hand") !== T) throw new Error(`bodyFrame: ${h.length / HAND_W} hand frames but ${T} pose frames`);
  const rp = roundingOf(p), rh = roundingOf(h);
  const at = (t, j) => t * POSE_W + 2 * j;

  // The shoulders, in the frames that have both (judged by x, as signfeat does).
  const midX = [], midY = [], widths = [];
  for (let t = 0; t < T; t++) {
    const a = at(t, SH_A), b = at(t, SH_B);
    if (Number.isNaN(p[a]) || Number.isNaN(p[b])) continue;
    midX.push(rp(rp(p[a] + p[b]) / 2));
    midY.push(rp(rp(p[a + 1] + p[b + 1]) / 2));
    widths.push(norm2(rp(p[a] - p[b]), rp(p[a + 1] - p[b + 1]), rp));
  }
  if (widths.length) {
    const width = median(widths, rp);
    if (width > 1e-4) return { mid: Float32Array.of(median(midX, rp), median(midY, rp)), width };
  }

  // The eyes and the nose: shoulders sit about 3.5 eye-spans apart and 2.2 below the nose.
  const spans = [], noseX = [], noseY = [];
  for (let t = 0; t < T; t++) {
    const a = at(t, EYE_A), b = at(t, EYE_B), n = at(t, NOSE);
    if (Number.isNaN(p[a]) || Number.isNaN(p[b]) || Number.isNaN(p[n])) continue;
    spans.push(norm2(rp(p[a] - p[b]), rp(p[a + 1] - p[b + 1]), rp));
    noseX.push(p[n]);
    noseY.push(p[n + 1]);
  }
  if (spans.length) {
    const eyes = median(spans, rp);
    if (eyes > 1e-4) {
      // NumPy 2 multiplies a Python float by a float32 scalar in float32, 2.2 rounded to float32 first (a double
      // product rounded once differs in about a quarter of cases); the pair is then cast to float32.
      const below = f32(rp(rp(2.2) * eyes));
      return { mid: Float32Array.of(rp(median(noseX, rp) + 0), rp(median(noseY, rp) + below)), width: rp(3.5 * eyes) };
    }
  }

  // The hand: its median wrist, and 4 palm lengths (wrist to middle knuckle), at least 1e-3.
  const wristX = [], wristY = [], palms = [];
  for (let t = 0; t < T; t++) {
    const o = t * HAND_W;
    if (Number.isNaN(h[o])) continue;
    wristX.push(h[o]);
    wristY.push(h[o + 1]);
    palms.push(norm2(rh(h[o + 18] - h[o]), rh(h[o + 19] - h[o + 1]), rh));
  }
  if (palms.length) {
    const unit = 4 * median(palms, rh);
    // Python's max(a, b) keeps a unless b > a, so a NaN palm stays NaN (and every feature then reads 0).
    return { mid: Float32Array.of(median(wristX, rh), median(wristY, rh)), width: 1e-3 > unit ? 1e-3 : unit };
  }
  return { mid: new Float32Array(2), width: 1.0 };
}

const clipped = (v) => (Number.isNaN(v) ? 0 : v < -CLIP ? -CLIP : v > CLIP ? CLIP : v);

/**
 * signfeat.features: canonical hand and pose (NaN where missing) -> the model's input.
 * @param {Float32Array|Float64Array} h  [T, 21, 2]
 * @param {Float32Array|Float64Array} p  [T, 13, 2]
 * @returns {{x: Float32Array, T: number}}  x is [T, N_FEAT], frame after frame
 */
export function features(h, p) {
  h = fillGaps(h);
  const T = framesIn(h, HAND_W, "features hand");
  if (framesIn(p, POSE_W, "features pose") !== T) throw new Error(`features: ${T} hand frames but ${p.length / POSE_W} pose frames`);
  // signfeat raises on an empty window (D.reshape(0, -1)), so the twin gives no answer either. attemptWindow returns
  // one when the hand was never found: that attempt is "unclear", not something to show the model.
  if (T === 0) throw new Error("features: an empty window has no features (signfeat raises too); check attemptWindow's T first");
  const { mid, width } = bodyFrame(h, p);
  const rh = roundingOf(h), rp = roundingOf(p);
  // Each part keeps its own dtype and their concatenation takes the wider one (only a float64 hand with a float32
  // pose, or the reverse, makes them differ). The Python float `width` meets each array in that array's dtype; the
  // palm's floor is 0.02 * width worked out as a Python float first.
  const base = new (h instanceof Float64Array || p instanceof Float64Array ? Float64Array : Float32Array)(T * N_BASE);
  const rb = roundingOf(base);
  const wh = rh(width), wp = rp(width), palmFloor = rh(0.02 * width);
  for (let t = 0; t < T; t++) {
    const o = t * HAND_W, q = t * POSE_W, b = t * N_BASE;
    for (let i = 0; i < 21; i++) {
      base[b + 2 * i] = rh(rh(h[o + 2 * i] - mid[0]) / wh);
      base[b + 2 * i + 1] = rh(rh(h[o + 2 * i + 1] - mid[1]) / wh);
    }
    let palm = norm2(rh(h[o + 18] - h[o]), rh(h[o + 19] - h[o + 1]), rh);
    palm = Number.isNaN(palm) || Number.isNaN(palmFloor) ? NaN : palm < palmFloor ? palmFloor : palm;   // np.maximum
    for (let i = 1; i < 21; i++) {
      base[b + 40 + 2 * i] = rh(rh(h[o + 2 * i] - h[o]) / palm);
      base[b + 40 + 2 * i + 1] = rh(rh(h[o + 2 * i + 1] - h[o + 1]) / palm);
    }
    for (let j = 0; j < POSE_KEEP.length; j++) {
      base[b + 82 + 2 * j] = rp(rp(p[q + 2 * j] - mid[0]) / wp);
      base[b + 82 + 2 * j + 1] = rp(rp(p[q + 2 * j + 1] - mid[1]) / wp);
    }
  }
  const x = new Float32Array(T * N_FEAT);
  for (let t = 0; t < T; t++) {
    const b = t * N_BASE, o = t * N_FEAT;
    for (let c = 0; c < N_BASE; c++) {
      const v = base[b + c];
      x[o + c] = clipped(v);
      x[o + N_BASE + c] = t >= 1 ? clipped(rb(v - base[b - N_BASE + c])) : 0;
      x[o + 2 * N_BASE + c] = t >= 2 ? clipped(rb(v - base[b - 2 * N_BASE + c])) : 0;
    }
    x[o + 3 * N_BASE] = Number.isNaN(h[t * HAND_W]) ? 0 : 1;
    x[o + 3 * N_BASE + 1] = Number.isNaN(p[t * POSE_W + 2 * SH_A]) ? 0 : 1;
  }
  return { x, T };
}

/* signfeat.active_span: the first and last frame (inclusive) with the hand, widened by `margin`; null if never. */
export function activeSpan(h, margin = 2) {
  const T = framesIn(h, HAND_W, "activeSpan");
  let first = -1, last = -1;
  for (let t = 0; t < T; t++) {
    if (Number.isNaN(h[t * HAND_W])) continue;
    if (first < 0) first = t;
    last = t;
  }
  return first < 0 ? null : [Math.max(0, first - margin), Math.min(T - 1, last + margin)];
}

/**
 * signfeat.resample: [T, ...] -> n frames by linear interpolation; a frame missing on either side stays missing
 * (even at an exact position, where numpy still adds 0 * the next frame, and 0 * NaN is NaN).
 * @param {Float32Array|Float64Array} a  frames of `width` numbers
 * @returns the same array when T === n; the one frame n times, in its own dtype, when T === 1 (np.repeat); otherwise
 *   a Float64Array (numpy's (1 - w) * a[i0] + w * a[i1] with float64 weights).
 */
export function resample(a, n, width) {
  const T = framesIn(a, width, "resample");
  if (T === n) return a;
  if (T === 1) {
    const out = new a.constructor(n * width);
    for (let i = 0; i < n; i++) out.set(a, i * width);
    return out;
  }
  if (T === 0) throw new RangeError("resample: no frames to resample");
  const out = new Float64Array(n * width);
  const step = (T - 1) / (n - 1);
  for (let i = 0; i < n; i++) {
    // np.linspace(0, T - 1, n): i * step, the last one exactly T - 1, and [0] when n is 1.
    const pos = n === 1 ? 0 : i === n - 1 ? T - 1 : i * step;
    const i0 = Math.floor(pos), i1 = Math.min(i0 + 1, T - 1), w = pos - i0;
    for (let c = 0; c < width; c++) out[i * width + c] = (1 - w) * a[i0 * width + c] + w * a[i1 * width + c];
  }
  return out;
}

/**
 * signfeat.attempt_window: an isolated attempt is the hand's active span, squeezed to MAX_T frames when longer.
 * The arrays returned may share memory with h and p (as numpy's slices do); nothing here writes into them.
 * @returns {{h: Float32Array|Float64Array, p: Float32Array|Float64Array, T: number}}
 */
export function attemptWindow(h, p) {
  const T = framesIn(h, HAND_W, "attemptWindow hand");
  if (framesIn(p, POSE_W, "attemptWindow pose") !== T) throw new Error(`attemptWindow: ${T} hand frames but ${p.length / POSE_W} pose frames`);
  const span = activeSpan(h);
  if (span === null) return { h: h.subarray(0, 0), p: p.subarray(0, 0), T: 0 };
  const hw = h.subarray(span[0] * HAND_W, (span[1] + 1) * HAND_W);
  const pw = p.subarray(span[0] * POSE_W, (span[1] + 1) * POSE_W);
  const n = span[1] - span[0] + 1;
  if (n > MAX_T) return { h: resample(hw, MAX_T, HAND_W), p: resample(pw, MAX_T, POSE_W), T: MAX_T };
  return { h: hw, p: pw, T: n };
}

/**
 * The model's input from frame records already on the 30 fps clock (resampleToFps): canonical, then attemptWindow
 * for an isolated attempt, then features, in signfeat's order. null when there is nothing to score: no frames, or an
 * attempt in which the hand was never found (features would throw there, as signfeat does). It has the shape of
 * signs/recognizer.js's featuresFn, so it can be passed as it is.
 * @param {object[]} records  frame records {tMs, aspect, hand, other, pose}
 * @param {{flip: boolean, attempt?: boolean}} o  the session's canonical side; attempt false for a plain window
 * @returns {{x: Float32Array, T: number}|null}
 */
export function modelInput(records, { flip, attempt = true } = {}) {
  // An unset side would silently read every left-handed signer as a mirror image: say so instead.
  if (typeof flip !== "boolean") throw new Error(`modelInput: flip must be true or false (the session's side), got ${flip}`);
  const c = canonicalFromRecords(records, flip);
  const w = attempt ? attemptWindow(c.h, c.p) : c;
  return w.T > 0 ? features(w.h, w.p) : null;
}

/* ---- the browser's clock --------------------------------------------------------------------------------------- */

/**
 * Camera frames at uneven times -> frames at exact 1000 / fps ms steps.
 *
 * signfeat counts time in frames, and the training video ran at about 30 a second; a browser's frames come when the
 * camera and the trackers allow (24-30 a second on a phone, unevenly, with the odd one skipped). So the frames are put
 * back on a 30 fps clock first. Each output frame lies between the two nearest real frames, and each part (the
 * signing hand, the other hand, the pose) is the straight-line blend of the two, landmark by landmark. Where either of
 * the two lacks that part, the output frame lacks it too (signfeat's resample has the same rule). A hand is not blended
 * across a pause longer than maxGapMs either (by default MAX_GAP + 1 frame times: two frames that far apart have the
 * MAX_GAP missing frames between them that fill_gaps would still bridge). Blending across a longer pause would draw a
 * movement nobody made, so the frames in between have no hand, as a camera that kept running but lost the hand would
 * give. The pose is blended across any pause: the body barely moves, and the model never saw
 * a body vanish for a moment. An output frame that falls on a real frame is that frame. The picture's shape is never
 * blended: across a change of shape (a phone turned) the nearer frame is used as it is.
 *
 * @param {{tMs:number, aspect:number, hand:Float32Array|null, other?:Float32Array|null, pose:Float32Array|null}[]} records
 *   in the order they were seen (parts as Float32Arrays, NaN for a missing number); a record without a time, or
 *   whose time is not after the previous one's, is skipped
 * @param {number} [fps]
 * @param {{anchor?: "last"|"first", maxGapMs?: number}} [o]  anchor "last" (the default) puts the last output frame on
 *   the last real frame, so the newest frame is always used whole; "first" starts the clock at the first real frame
 * @returns frame records at exact steps (a real frame's own arrays where an output frame falls on it; never write
 *   into them)
 */
export function resampleToFps(records, fps = FPS, { anchor = "last", maxGapMs = (MAX_GAP + 1) * 1000 / fps } = {}) {
  if (!(fps > 0 && fps < Infinity)) throw new Error(`resampleToFps: fps must be a positive number, got ${fps}`);
  if (anchor !== "last" && anchor !== "first") throw new Error(`resampleToFps: anchor is "last" or "first", got ${anchor}`);
  const src = [];
  for (const r of records) if (Number.isFinite(r.tMs) && (!src.length || r.tMs > src[src.length - 1].tMs)) src.push(r);
  if (src.length < 2) return src.map((r) => frameAt(r, r.tMs));
  const step = 1000 / fps, t0 = src[0].tMs, t1 = src[src.length - 1].tMs;
  // Times are sums of steps, so allow for rounding: a millionth of a frame decides the count, a hundred-thousandth
  // of a frame counts as falling on a real frame.
  const n = Math.floor((t1 - t0) / step + 1e-6) + 1;
  const same = 1e-5 * step;
  const out = new Array(n);
  let j = 0;
  for (let k = 0; k < n; k++) {
    const t = anchor === "last" ? t1 - (n - 1 - k) * step : t0 + k * step;
    while (j < src.length - 2 && src[j + 1].tMs <= t) j++;
    const a = src[j], b = src[j + 1];
    if (t - a.tMs <= same) out[k] = frameAt(a, t);
    else if (b.tMs - t <= same) out[k] = frameAt(b, t);
    else if (a.aspect !== b.aspect) out[k] = frameAt(t - a.tMs <= b.tMs - t ? a : b, t);
    else {
      const w = (t - a.tMs) / (b.tMs - a.tMs), near = b.tMs - a.tMs <= maxGapMs;
      out[k] = {
        tMs: t, aspect: a.aspect,
        hand: near ? blend(a.hand, b.hand, w) : null,
        other: near ? blend(a.other, b.other, w) : null,
        pose: blend(a.pose, b.pose, w),
      };
    }
  }
  return out;
}

const frameAt = (r, tMs) => ({ tMs, aspect: r.aspect, hand: r.hand ?? null, other: r.other ?? null, pose: r.pose ?? null });

function blend(u, v, w) {
  if (u == null || v == null) return null;
  if (u.length !== v.length) throw new Error(`resampleToFps: one part has ${u.length} numbers in one frame and ${v.length} in the next`);
  const out = new Float32Array(u.length);
  for (let i = 0; i < u.length; i++) out[i] = (1 - w) * u[i] + w * v[i];
  return out;
}
