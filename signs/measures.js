/* measures.js: what a sign attempt DID, in the parts a teacher talks about -- where the hand was (place), how it moved
 * (movement) and what shape it held (hand shape) -- so that a missed sign can be answered with the one part that was off.
 *
 * The same function measures the Deaf signers' clips the templates are built from (tools/signs/build-templates.mjs)
 * and the learner's attempt in the browser (signs/feedback.js), so the two are measured alike by construction. It works
 * in the model's own canonical picture (signs/features.js: the signing hand on the image's left, x in the picture's
 * height units) and body frame (origin = the shoulders' midpoint, unit = shoulder width, y downwards).
 *
 *   place     the palm's centre (in the body frame) and the two longest fingertips relative to the mouth, each at the
 *             start, middle and end of the attempt: 12 numbers. Relative to the mouth because the beginner signs that
 *             live on the face (HELLO at the forehead, THANK-YOU and WATER at the chin) are about the face, and a
 *             long neck or a tilted head moves the face against the shoulders.
 *   movement  the palm's net displacement, path length and extent; how many times it turned back along its main
 *             axis (taps, waves); how many times the hand tipped back and forth at the wrist (YES's nod); how many
 *             times the index fingertip closed onto the thumb (NO's snap); how many times the fingers opened and
 *             closed; how many full turns the path made (PLEASE's circle).
 *   shape     tutor/features.js's 46 hand features (the letter grader's), each the median over the middle half of
 *             the attempt, where the hand shape of these signs is steady.
 *
 * Counting a turn-back needs a hysteresis, or tracker jitter counts as a tap: a reversal counts only once the value
 * has come back by more than a set amount from its last extreme. */

import { canonical, fillGaps, bodyFrame, activeSpan, HAND_W, POSE_W } from "./features.js";
import { featuresFromFlat } from "../tutor/features.js";

export const PHASES = Object.freeze([["start", 0, 0.25], ["middle", 0.375, 0.625], ["end", 0.75, 1]]);
export const PLACE_NAMES = Object.freeze(PHASES.flatMap(([ph]) => [`${ph}.palm.x`, `${ph}.palm.y`, `${ph}.tip.x`, `${ph}.tip.y`]));
export const MOVE_NAMES = Object.freeze(["dx", "dy", "path", "extent", "axisTurns", "nods", "snaps", "openClose", "circles"]);
export const COUNT_MEASURES = Object.freeze(["axisTurns", "nods", "snaps", "openClose", "circles"]);
const PALM = [0, 5, 9, 13, 17];
const TIPS = [8, 12];
const MOUTH_A = 5, MOUTH_B = 6, NOSE = 0;   // slots in signs/features.js POSE_KEEP
const MIN_FRAMES = 4;

const finite = Number.isFinite;

function median(vals) {
  const v = vals.filter(finite).sort((a, b) => a - b);
  if (!v.length) return NaN;
  const m = v.length >> 1;
  return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
}

/* Turn-backs of a series with hysteresis H: one each time it comes back by more than H from its last extreme. */
export function reversals(vals, H) {
  let lo = Infinity, hi = -Infinity, dir = 0, ext = 0, count = 0;
  for (const x of vals) {
    if (!finite(x)) continue;
    if (dir === 0) {
      lo = Math.min(lo, x); hi = Math.max(hi, x);
      if (x - lo > H) { dir = 1; ext = x; } else if (hi - x > H) { dir = -1; ext = x; }
    } else if (dir === 1) {
      if (x > ext) ext = x; else if (ext - x > H) { count++; dir = -1; ext = x; }
    } else if (x < ext) ext = x;
    else if (x - ext > H) { count++; dir = 1; ext = x; }
  }
  return count;
}

/* Closings: times the series drops below `closed` after having been above `open`. */
export function closings(vals, open, closed) {
  let state = null, count = 0;
  for (const x of vals) {
    if (!finite(x)) continue;
    if (x > open) state = "open";
    else if (x < closed) { if (state === "open") count++; state = "closed"; }
  }
  return count;
}

const at2 = (a, o) => [a[o], a[o + 1]];

/**
 * @param {{hands: (Float32Array|null)[], poses: (Float32Array|null)[], aspect: number|number[], flip: boolean,
 *          shapes?: (number[]|null)[]}} a
 *   per frame (30 fps): the signing hand 21 x (x, y, z) and the pose 33 x (x, y, visibility), raw picture coordinates;
 *   shapes, when given, the hand-shape features of each frame already worked out (shapeRows), so that measuring many
 *   windows of one attempt does not work them out again
 * @returns {{ok: true, frames: number, place: number[], move: object, shape: number[]|null} | {ok: false, reason: string}}
 */
export function measureAttempt({ hands, poses, aspect, flip, shapes = null }) {
  const { h: raw, p } = canonical(hands, poses, aspect, flip);
  const h = fillGaps(raw);
  const span = activeSpan(h, 0);
  if (!span || span[1] - span[0] + 1 < MIN_FRAMES) return { ok: false, reason: "too-short" };
  const [s, e] = span;
  const n = e - s + 1;
  const hs = h.subarray(s * HAND_W, (e + 1) * HAND_W), ps = p.subarray(s * POSE_W, (e + 1) * POSE_W);
  const { mid, width } = bodyFrame(hs, ps);
  const toBody = (x, y) => [(x - mid[0]) / width, (y - mid[1]) / width];

  // the mouth, once for the attempt (median of the frames that have it; the nose a little lower if not)
  const mx = [], my = [];
  for (let t = 0; t < n; t++) {
    const a = at2(ps, t * POSE_W + 2 * MOUTH_A), b = at2(ps, t * POSE_W + 2 * MOUTH_B);
    if (finite(a[0]) && finite(b[0])) { mx.push((a[0] + b[0]) / 2); my.push((a[1] + b[1]) / 2); }
  }
  let mouth = mx.length ? toBody(median(mx), median(my)) : null;
  if (!mouth) {
    const nx = [], ny = [];
    for (let t = 0; t < n; t++) { const q = at2(ps, t * POSE_W + 2 * NOSE); if (finite(q[0])) { nx.push(q[0]); ny.push(q[1]); } }
    mouth = nx.length ? toBody(median(nx), median(ny) + 0.08 * width) : [NaN, NaN];
  }

  const palm = [], tip = [], pitch = [], pinch = [], open = [];
  for (let t = 0; t < n; t++) {
    const o = t * HAND_W;
    if (!finite(hs[o])) { palm.push(null); tip.push(null); pitch.push(NaN); pinch.push(NaN); open.push(NaN); continue; }
    const pt = (i) => [hs[o + 2 * i], hs[o + 2 * i + 1]];
    let cx = 0, cy = 0;
    for (const i of PALM) { const q = pt(i); cx += q[0]; cy += q[1]; }
    palm.push(toBody(cx / PALM.length, cy / PALM.length));
    let tx = 0, ty = 0;
    for (const i of TIPS) { const q = pt(i); tx += q[0]; ty += q[1]; }
    const tb = toBody(tx / TIPS.length, ty / TIPS.length);
    tip.push([tb[0] - mouth[0], tb[1] - mouth[1]]);
    const w = pt(0), m9 = pt(9), i8 = pt(8), t4 = pt(4);
    const palmLen = Math.hypot(m9[0] - w[0], m9[1] - w[1]) || NaN;
    pitch.push(Math.atan2(m9[1] - w[1], m9[0] - w[0]) * 180 / Math.PI);
    pinch.push(Math.hypot(i8[0] - t4[0], i8[1] - t4[1]) / palmLen);
    let ext = 0;
    for (const i of [8, 12, 16, 20]) { const q = pt(i); ext += Math.hypot(q[0] - w[0], q[1] - w[1]); }
    open.push(ext / 4 / palmLen);
  }
  const present = palm.map((q, t) => (q ? t : -1)).filter((t) => t >= 0);
  if (present.length < MIN_FRAMES) return { ok: false, reason: "too-short" };

  // place: medians over each phase's frames (the nearest frame with the hand when a phase has none)
  const place = [];
  for (const [, a, b] of PHASES) {
    let ts = present.filter((t) => t / (n - 1) >= a - 1e-9 && t / (n - 1) <= b + 1e-9);
    if (!ts.length) {
      const c = ((a + b) / 2) * (n - 1);
      ts = [present.reduce((x, y) => (Math.abs(y - c) < Math.abs(x - c) ? y : x))];
    }
    place.push(median(ts.map((t) => palm[t][0])), median(ts.map((t) => palm[t][1])),
      median(ts.map((t) => tip[t][0])), median(ts.map((t) => tip[t][1])));
  }

  // movement
  const k = Math.max(1, Math.round(present.length * 0.2));
  const first = present.slice(0, k), last = present.slice(-k);
  const c0 = [median(first.map((t) => palm[t][0])), median(first.map((t) => palm[t][1]))];
  const c1 = [median(last.map((t) => palm[t][0])), median(last.map((t) => palm[t][1]))];
  let path = 0, xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  for (let j = 0; j < present.length; j++) {
    const q = palm[present[j]];
    xmin = Math.min(xmin, q[0]); xmax = Math.max(xmax, q[0]); ymin = Math.min(ymin, q[1]); ymax = Math.max(ymax, q[1]);
    if (j && present[j] - present[j - 1] === 1) { const r = palm[present[j - 1]]; path += Math.hypot(q[0] - r[0], q[1] - r[1]); }
  }
  // the main axis of the palm's path (the larger eigenvector of its 2x2 covariance)
  const mxp = median(present.map((t) => palm[t][0])), myp = median(present.map((t) => palm[t][1]));
  let sxx = 0, syy = 0, sxy = 0;
  for (const t of present) { const dx = palm[t][0] - mxp, dy = palm[t][1] - myp; sxx += dx * dx; syy += dy * dy; sxy += dx * dy; }
  const ang = 0.5 * Math.atan2(2 * sxy, sxx - syy);
  const ax = [Math.cos(ang), Math.sin(ang)];
  const proj = present.map((t) => palm[t][0] * ax[0] + palm[t][1] * ax[1]);
  const range = Math.max(...proj) - Math.min(...proj);
  const axisTurns = reversals(proj, Math.max(0.03, 0.2 * range));
  // the wrist's nod, unwrapped so a hand pointing straight up does not jump by 360 degrees
  const pitchU = [];
  let prev = NaN, off = 0;
  for (const v of pitch) {
    if (!finite(v)) { pitchU.push(NaN); continue; }
    if (finite(prev)) { const d = v + off - prev; if (d > 180) off -= 360; else if (d < -180) off += 360; }
    prev = v + off; pitchU.push(v + off);
  }
  const nods = reversals(pitchU, 12);
  const snaps = closings(pinch, 0.75, 0.4);
  const openClose = reversals(open, 0.35);
  // full turns of the path: the summed change of its direction over steps long enough to have one
  let turn = 0, lastDir = NaN, ax0 = NaN, ay0 = NaN;
  for (const t of present) {
    const q = palm[t];
    if (!finite(ax0)) { ax0 = q[0]; ay0 = q[1]; continue; }
    const dx = q[0] - ax0, dy = q[1] - ay0;
    if (Math.hypot(dx, dy) < 0.02) continue;
    const dir = Math.atan2(dy, dx);
    if (finite(lastDir)) { let d = dir - lastDir; while (d > Math.PI) d -= 2 * Math.PI; while (d < -Math.PI) d += 2 * Math.PI; turn += d; }
    lastDir = dir; ax0 = q[0]; ay0 = q[1];
  }
  const move = { dx: c1[0] - c0[0], dy: c1[1] - c0[1], path, extent: Math.hypot(xmax - xmin, ymax - ymin), axisTurns, nods,
    snaps, openClose, circles: Math.abs(turn) / (2 * Math.PI) };

  // shape: the letter grader's features over the middle half, from the raw frames (z included)
  const sign = flip ? -1 : 1;
  const rows = [];
  for (let t = Math.floor(n * 0.25); t <= Math.ceil(n * 0.75) - 1; t++) {
    if (shapes) { if (shapes[s + t]) rows.push(shapes[s + t]); continue; }
    const flat = hands[s + t];
    if (!flat) continue;
    const a = typeof aspect === "number" ? aspect : aspect[s + t];
    try { rows.push(featuresFromFlat(flat, a, sign)); } catch { /* a degenerate frame is skipped */ }
  }
  const shape = rows.length ? rows[0].map((_, j) => median(rows.map((r) => r[j]))) : null;
  return { ok: true, frames: n, place, move, shape: shape && shape.every(finite) ? shape : null };
}

/* The hand-shape features of every frame of an attempt (null where the hand is missing or degenerate), for
 * measureAttempt's `shapes`. */
export function shapeRows({ hands, aspect, flip }) {
  const sign = flip ? -1 : 1;
  return hands.map((flat, t) => {
    if (!flat) return null;
    try {
      const r = featuresFromFlat(flat, typeof aspect === "number" ? aspect : aspect[t], sign);
      return r.every(finite) ? r : null;
    } catch { return null; }
  });
}
