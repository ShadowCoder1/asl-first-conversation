/* feedback.js: after a missed sign, the one thing to fix, in the sign's own words, and a three-part checklist.
 *
 * The model (signs/recognizer.js) decides WHETHER the sign was made; this module says what was off when it was not.
 * The attempt is measured (signs/measures.js) and set against the sign's template (signs/model/templates.json): how
 * the Deaf signers of asl-signs made it. Each part -- place, movement, hand shape -- is "off" only beyond a cut that 95 %
 * of unseen fluent signers' own clips pass (tools/signs/build-templates.mjs), so a learner is not corrected for what
 * fluent signers do anyway. Of the parts that are off, the one furthest beyond its cut is named:
 *
 *   place     the phase (start, middle, end) furthest off, and which way: higher / lower / closer, or "touch" when the
 *             sign touches the face and the hand stayed away from it;
 *   movement  the measure furthest off: too few taps, nods, snaps or circles -> "repeat"; too many for a one-stroke sign
 *             -> "once"; too short / long a path -> "bigger" / "smaller"; the wrong way -> "direction";
 *   shape     which part of the hand: the letter grader's counterfactual (tutor/blame.js) on the sign's hand-shape
 *             Gaussian -- move one block of features to its mean given the rest of the hand, and see which buys the
 *             most -- so orientation can be told apart from the fingers.
 *
 * The words come from signs/content.json (fixes.location.higher, fixes.movement.repeat, ...), each falling back to the
 * part's default sentence. "That looked like X" is not said here: the recognizer's miss event carries it, limited to
 * signs the learner has met (signs/app.js); `looksLike` is still returned for tools.
 *
 * WHICH FRAMES. A try in the app runs from the hand coming up to the hand going down; the signers' clips the templates
 * come from were trimmed to the sign. Measured whole, a correct WATER read as "taps too big" (the hand rising counted as
 * movement: 8 of 21 correct PopSign clips were flagged). So the attempt is measured in windows about as long as the
 * sign's own clips (0.8, 1 and 1.3 times their median length, every 3 frames, and the whole try), and judged by the
 * window that matches the template best: the learner's best go at the sign within the try. What is still off there is
 * really off. */

import { measureAttempt, shapeRows, PHASES } from "./measures.js";
import { FEATURE_BLOCKS, BLOCKS } from "../tutor/features.js";
import { solve, subMatrix } from "../tutor/linalg.js";

export const LOOKS_LIKE_MIN_P = 0.6;
const FACE_NEAR = 0.35;      // shoulder widths: a template fingertip this close to the mouth makes it a face sign
const TOUCH_NEAR = 0.2;
const BLOCK_INDEX = Object.fromEntries(BLOCKS.map((b) => [b, FEATURE_BLOCKS.flatMap((x, i) => (x === b ? [i] : []))]));
const COUNTS = new Set(["axisTurns", "nods", "snaps", "openClose", "circles"]);

function quad(prec, d) {
  let s = 0;
  for (let i = 0; i < d.length; i++) { let r = 0; for (let j = 0; j < d.length; j++) r += prec[i][j] * d[j]; s += d[i] * r; }
  return s;
}

/* The block whose move to its conditional mean (given the rest of the hand) lowers the distance most, per square root
 * of its width (tutor/blame.js RANK_CRITERION). */
export function blameShape(tpl, z) {
  const d = z.map((v, i) => v - tpl.mu[i]);
  const before = quad(tpl.prec, d);
  let best = null;
  for (const block of BLOCKS) {
    const b = BLOCK_INDEX[block];
    const r = d.map((_, i) => i).filter((i) => !b.includes(i));
    const Pbb = subMatrix(tpl.prec, b, b), Pbr = subMatrix(tpl.prec, b, r);
    const rhs = Pbr.map((row) => row.reduce((s, v, k) => s + v * d[r[k]], 0));
    const shift = solve(Pbb, rhs);            // d_b at the conditional mean is -Pbb^-1 Pbr d_r
    const fixed = d.slice();
    b.forEach((i, k) => { fixed[i] = -shift[k]; });
    const gain = before - quad(tpl.prec, fixed);
    const rank = gain / Math.sqrt(b.length);
    if (!best || rank > best.rank) best = { block, gain, rank };
  }
  return { d2: before, ...best };
}

function pick(fix, key) {
  if (fix == null) return null;
  if (typeof fix === "string") return fix;
  return fix[key] ?? fix.default ?? null;
}

/**
 * @param {{templates: object, content: object, labels: string[], scale: {mean: number[], sd: number[]}}} o
 *   scale: tutor/model.json's, which the templates' hand-shape Gaussians are in
 * @returns {(attempt: {hands, poses, aspect, flip}, target: string, probs?: Float32Array) =>
 *   {text: string|null, checks: {handshape: boolean|null, location: boolean|null, movement: boolean|null},
 *    looksLike: string|null, part: string|null, detail: object}}
 */
export const WINDOW_SCALES = Object.freeze([0.8, 1, 1.3]);
const WINDOW_STEP = 3;

/* Every part of one measured window against the template: ratios past each cut (over 1 is off). */
function judge(m, tpl, scale) {
  const P = tpl.place;
  const phase = PHASES.map(([name], k) => {
    let s = 0;
    for (let j = 0; j < 4; j++) s += ((m.place[4 * k + j] - P.median[4 * k + j]) / P.scale[4 * k + j]) ** 2;
    return { name, k, ratio: Math.sqrt(s / 4) / P.phaseCut[k] };
  });
  const worstPhase = phase.reduce((a, b) => (b.ratio > a.ratio ? b : a));
  const M = tpl.move;
  const moves = M.names.map((name, j) => ({ name, j, value: m.move[name], z: (m.move[name] - M.median[j]) / M.scale[j], ratio: Math.abs(m.move[name] - M.median[j]) / M.scale[j] / M.cut[j] }));
  const worstMove = moves.reduce((a, b) => (b.ratio > a.ratio ? b : a));
  let shape = null;
  if (m.shape && tpl.shape) {
    const z = m.shape.map((v, i) => (v - scale.mean[i]) / scale.sd[i]);
    shape = blameShape(tpl.shape, z);
    shape.ratio = Math.sqrt(Math.max(shape.d2, 0) / tpl.shape.d2Cut);
  }
  const worst = Math.max(worstPhase.ratio, worstMove.ratio, shape?.ratio ?? 0);
  return { worstPhase, worstMove, shape, worst };
}

/* The window of the try that matches the template best (see WHICH FRAMES above). */
export function bestWindow(attempt, tpl, scale) {
  const n = attempt.hands.length;
  const shapes = shapeRows(attempt);
  const slice = (s, L) => ({ hands: attempt.hands.slice(s, s + L), poses: attempt.poses.slice(s, s + L),
    aspect: typeof attempt.aspect === "number" ? attempt.aspect : attempt.aspect.slice(s, s + L), flip: attempt.flip, shapes: shapes.slice(s, s + L) });
  let best = null;
  const consider = (s, L) => {
    const m = measureAttempt(slice(s, L));
    if (!m.ok) return;
    const j = judge(m, tpl, scale);
    if (!best || j.worst < best.j.worst) best = { m, j, s, L };
  };
  consider(0, n);
  const L0 = tpl.frames?.median ?? n;
  for (const f of WINDOW_SCALES) {
    const L = Math.round(L0 * f);
    if (L < 6 || L >= n) continue;
    for (let s = 0; s + L <= n; s += WINDOW_STEP) consider(s, L);
  }
  return best;
}

export function createFeedback({ templates, content, labels, scale }) {
  const glossOf = (id) => content?.signs?.[id]?.gloss ?? content?.glosses?.[id] ?? id.toUpperCase();
  return function feedback(attempt, target, probs = null) {
    const tpl = templates?.signs?.[target];
    const words = content?.signs?.[target] ?? {};
    let looksLike = null;
    if (probs) {
      let bi = -1, bp = 0;
      for (let i = 0; i < Math.min(labels.length, probs.length); i++) {
        if (labels[i] === target || labels[i] === "none") continue;
        if (probs[i] > bp) { bp = probs[i]; bi = i; }
      }
      if (bi >= 0 && bp >= LOOKS_LIKE_MIN_P) looksLike = glossOf(labels[bi]);
    }
    const none = { text: null, checks: { handshape: null, location: null, movement: null }, looksLike, part: null, detail: {} };
    if (!tpl) return none;
    const best = bestWindow(attempt, tpl, scale);
    if (!best) return { ...none, detail: { reason: "too-short" } };
    const { m, j: { worstPhase, worstMove, shape } } = best;
    const P = tpl.place, M = tpl.move;
    const checks = { location: worstPhase.ratio <= 1, movement: worstMove.ratio <= 1, handshape: shape ? shape.ratio <= 1 : null };

    // the part to name: the one furthest past its cut (place first on a tie: the easiest to see and fix)
    const parts = [["location", worstPhase.ratio], ["movement", worstMove.ratio], ["handshape", shape?.ratio ?? 0]].filter(([, r]) => r > 1);
    parts.sort((a, b) => b[1] - a[1]);
    const part = parts[0]?.[0] ?? null;
    let text = null, key = null;
    if (part === "location") {
      const k = worstPhase.k;
      const tipT = [P.median[4 * k + 2], P.median[4 * k + 3]];
      const face = Math.hypot(...tipT) < FACE_NEAR;
      const [dx, dy] = face ? [m.place[4 * k + 2] - tipT[0], m.place[4 * k + 3] - tipT[1]]
        : [m.place[4 * k] - P.median[4 * k], m.place[4 * k + 1] - P.median[4 * k + 1]];
      const tipNow = Math.hypot(m.place[4 * k + 2], m.place[4 * k + 3]);
      if (face && Math.hypot(...tipT) < TOUCH_NEAR && tipNow > Math.max(0.3, 2 * Math.hypot(...tipT))) key = "touch";
      else if (Math.abs(dy) >= Math.abs(dx)) key = dy > 0 ? "higher" : "lower";
      else key = Math.abs(face ? m.place[4 * k + 2] : m.place[4 * k]) > Math.abs(face ? tipT[0] : P.median[4 * k]) ? "closer" : "default";
      text = pick(words.fixes?.location, key);
    } else if (part === "movement") {
      const w = worstMove;
      if (COUNTS.has(w.name)) key = w.z < 0 ? "repeat" : (M.median[w.j] <= 1 ? "once" : "default");
      else if (w.name === "path" || w.name === "extent") key = w.z < 0 ? "bigger" : "smaller";
      else key = "direction";
      text = pick(words.fixes?.movement, key);
    } else if (part === "handshape") {
      key = shape.block;
      text = shape.block === "orientation" ? (words.fixes?.orientation ?? pick(words.fixes?.handshape, "default")) : pick(words.fixes?.handshape, "default");
    }
    if (!text && part) text = words.howTo ?? null;
    if (!part) text = words.howTo ? `So close. ${words.howTo}` : null;
    return { text, checks, looksLike, part, detail: { key, phase: worstPhase, move: worstMove, shape, window: [best.s, best.L] } };
  };
}
