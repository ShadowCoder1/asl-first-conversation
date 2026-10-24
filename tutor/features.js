/* features.js: turns 21 canonical hand landmarks into 46 named, interpretable
 * articulatory features -- finger flexion, spread, thumb placement, and palm
 * orientation -- grouped into 7 "blocks": thumb, index, middle, ring, pinky,
 * spread, orientation.
 *
 * A block is the group of features that one articulator -- one finger, the
 * spread between fingers, or the way the palm is held -- actually controls.
 * It is the unit of blame and feedback (spec A3.2): when a practiced shape
 * comes out wrong, the tutor has to say WHICH articulator to fix, not just
 * that the shape is wrong, and a block is the smallest group of numbers that
 * maps back to one thing a learner can move. Every later diagnosis and
 * coaching message is phrased in terms of a block, so which block a feature
 * belongs to is as load-bearing as the feature's name.
 *
 * Orientation (palmnormal_*, handdir_*) is measured in the CAMERA frame and
 * is deliberately NOT normalized away, even though the other 40 features
 * live in a rotation- and size-invariant palm frame. Finger configuration
 * alone cannot tell G from Q, K from P, or U from H apart -- each pair has
 * the same handshape and differs only in which way the hand is held. Drop
 * orientation and those signs collapse into each other.
 *
 * FEATURE_NAMES and FEATURE_BLOCKS are an interface: a Python mirror, a
 * trained model file, and every later module depend on this exact order.
 * To keep the names and the values from ever drifting apart, both are built
 * by the same emit(value, name, block) calls inside buildFeatures below --
 * extractFeatures and the FEATURE_NAMES/FEATURE_BLOCKS exports both come
 * from that one function, never from two separately maintained lists. All
 * three exported arrays are frozen so an importer cannot silently corrupt
 * the interface (e.g. reordering a block in place); tests/features.test.js
 * carries a golden-vector table that pins every (name, value) pair for one
 * fixed pose, so a name/value swap that the structural tests miss still
 * fails a test.
 *
 * A degenerate hand (S = 0, e.g. all 21 landmarks coincident because
 * tracking dropped out) is a real input, not a bug: this module reports it
 * as NaN in every S-normalized feature (thumbtip_*, d_thumbtip_*, cross_im_*,
 * tip_palm_*) rather than throwing, and leaves the S-free features (flexion,
 * abduction, orientation) finite. That NaN is the "not gradable" signal;
 * deciding what to do about it (skip the frame, flag low tracking quality,
 * ...) belongs to the verifier's tracking-quality gate, not to this module. */
import { sub, add, scale, dot, norm, angleDeg, wrapDeg } from "./vec.js";
import { canonicalize, palmFrame } from "./hand-frame.js";
import { toPoints } from "./landmarks.js";

export const BLOCKS = Object.freeze(["thumb", "index", "middle", "ring", "pinky", "spread", "orientation"]);

const FINGERS = ["index", "middle", "ring", "pinky"];
// Landmark indices [mcp/cmc, pip/mcp, dip/ip, tip] per digit.
const CHAIN = { thumb: [1, 2, 3, 4], index: [5, 6, 7, 8], middle: [9, 10, 11, 12], ring: [13, 14, 15, 16], pinky: [17, 18, 19, 20] };

const dist = (a, b) => norm(sub(a, b));
const mean = (points) => scale(points.reduce((acc, p) => add(acc, p)), 1 / points.length);

// The single place all 46 features are computed, in the exact order the
// interface fixes. `emit` pushes a value alongside the name and block that
// describe it, so the three parallel arrays can never fall out of sync.
function buildFeatures(P) {
  const { x, y, n, S } = palmFrame(P);
  const values = [], names = [], blocks = [];
  const emit = (value, name, block) => { values.push(value); names.push(name); blocks.push(block); };

  // 15 flexion angles (0 = straight). Every joint but the first on each
  // chain is angleDeg between the two bones that meet there; the first
  // joint (thumb CMC, each finger's MCP) has no incoming bone of its own, so
  // it uses wrist -> that joint as a stand-in incoming bone.
  emit(angleDeg(sub(P[1], P[0]), sub(P[2], P[1])), "flex_thumb_cmc", "thumb");
  emit(angleDeg(sub(P[2], P[1]), sub(P[3], P[2])), "flex_thumb_mcp", "thumb");
  emit(angleDeg(sub(P[3], P[2]), sub(P[4], P[3])), "flex_thumb_ip", "thumb");
  for (const finger of FINGERS) {
    const [m, p, d, t] = CHAIN[finger];
    emit(angleDeg(sub(P[m], P[0]), sub(P[p], P[m])), `flex_${finger}_mcp`, finger);
    emit(angleDeg(sub(P[p], P[m]), sub(P[d], P[p])), `flex_${finger}_pip`, finger);
    emit(angleDeg(sub(P[d], P[p]), sub(P[t], P[d])), `flex_${finger}_dip`, finger);
  }

  // 3 abduction angles: signed angle, in the palm plane, between adjacent
  // proximal phalanges. Positive = the pair is spread apart.
  const az = {};
  for (const finger of FINGERS) {
    const [m, p] = CHAIN[finger];
    const v = sub(P[p], P[m]);
    az[finger] = Math.atan2(dot(v, x), dot(v, y)) * 180 / Math.PI;
  }
  for (const [a, b] of [["index", "middle"], ["middle", "ring"], ["ring", "pinky"]]) {
    emit(wrapDeg(az[a] - az[b]), `abd_${a}_${b}`, "spread");
  }

  // Thumb tip in the palm frame, with the origin taken at the index MCP.
  const tipVec = scale(sub(P[4], P[5]), 1 / S);
  emit(dot(tipVec, x), "thumbtip_radial", "thumb");
  emit(dot(tipVec, y), "thumbtip_distal", "thumb");
  emit(dot(tipVec, n), "thumbtip_palmar", "thumb");

  // Thumb tip distances to each finger's tip, PIP and MCP -- how close the
  // thumb is to making contact, which is the geometry behind signs like F.
  for (const finger of FINGERS) {
    const [m, p, , t] = CHAIN[finger];
    emit(dist(P[4], P[t]) / S, `d_thumbtip_${finger}_tip`, "thumb");
    emit(dist(P[4], P[p]) / S, `d_thumbtip_${finger}_pip`, "thumb");
    emit(dist(P[4], P[m]) / S, `d_thumbtip_${finger}_mcp`, "thumb");
  }

  // Signed index-middle crossing: positive = index tip is radial of middle
  // tip, negative once the fingers cross, as in signs like R.
  const crossVec = scale(sub(P[8], P[12]), 1 / S);
  emit(dot(crossVec, x), "cross_im_radial", "spread");
  emit(dot(crossVec, n), "cross_im_palmar", "spread");

  // Fingertip distance to palm center (mean of the wrist and the 4 finger
  // MCPs), one per digit including the thumb.
  const palmCenter = mean([P[0], P[5], P[9], P[13], P[17]]);
  emit(dist(P[4], palmCenter) / S, "tip_palm_thumb", "thumb");
  for (const finger of FINGERS) {
    const [, , , t] = CHAIN[finger];
    emit(dist(P[t], palmCenter) / S, `tip_palm_${finger}`, finger);
  }

  // Orientation in the camera frame -- deliberately not normalized away; see
  // the header comment.
  for (let i = 0; i < 3; i++) emit(n[i], `palmnormal_${"xyz"[i]}`, "orientation");
  // handdir_x is FOLDED: fingers pointing to the left of the picture and to
  // the right are one direction. No two letters differ by that alone, and the
  // training data has sideways letters (H, G, P, Q) pointed both ways -- as a
  // signed number H was two opposite clusters, its class overlapped U, and its
  // accept threshold rose until it refused 95% of real H hands
  // (research/allletters-2026-09-22/). |x| is continuous, so an upright hand
  // (x near 0) is not made jittery by it.
  emit(Math.abs(y[0]), "handdir_x", "orientation");
  for (let i = 1; i < 3; i++) emit(y[i], `handdir_${"xyz"[i]}`, "orientation");

  return { values, names, blocks };
}

// A small, non-degenerate 21-point stand-in hand, used only to bootstrap
// FEATURE_NAMES/FEATURE_BLOCKS below (names and blocks never depend on P,
// only on the fixed order in buildFeatures, so any well-formed P would do).
// It is NOT imported from hand-model.js -- that module is downstream of
// this one -- and it is deliberately not the degenerate all-zero point set
// this bootstrap used to run: computing every feature, including the
// S-normalized ones, against a real (if arbitrary) hand shape means the
// bootstrap never depends on 0/0 or Infinity*0 arithmetic to reach a finite
// answer, even though the answer itself is thrown away.
const BOOTSTRAP_HAND = [
  [0, 0, 0], // 0 wrist
  [0.3, 0.2, 0.1], [0.5, 0.4, 0.2], [0.65, 0.55, 0.25], [0.75, 0.7, 0.3], // 1-4 thumb
  [0.3, 0.9, 0], [0.32, 1.2, 0], [0.33, 1.45, 0], [0.34, 1.65, 0], // 5-8 index
  [0.05, 0.95, 0], [0.06, 1.3, 0], [0.07, 1.6, 0], [0.08, 1.85, 0], // 9-12 middle
  [-0.2, 0.9, 0], [-0.22, 1.2, 0], [-0.23, 1.45, 0], [-0.24, 1.65, 0], // 13-16 ring
  [-0.42, 0.75, 0], [-0.44, 1.0, 0], [-0.45, 1.2, 0], [-0.46, 1.35, 0], // 17-20 pinky
];

const BOOTSTRAP = buildFeatures(BOOTSTRAP_HAND);
export const FEATURE_NAMES = Object.freeze(BOOTSTRAP.names);
export const FEATURE_BLOCKS = Object.freeze(BOOTSTRAP.blocks);

// P must already be canonical (right-hand) isotropic points, as produced by
// hand-frame.js's canonicalize -- this function does not canonicalize. It
// must be exactly 21 [x, y, z] points: a flat 63-number array (an easy
// mistake -- that is landmarks.js's own on-the-wire shape) or a short array
// from a dropped landmark would otherwise silently produce garbage (NaNs or
// a truncated/misaligned feature vector) instead of failing loudly.
function assertLandmarks(P) {
  const shapeOk = Array.isArray(P) && P.length === 21 && P.every((p) => Array.isArray(p) && p.length === 3);
  if (!shapeOk) {
    const got = Array.isArray(P) ? `an array of length ${P.length}` : typeof P;
    throw new Error(`extractFeatures: expected 21 [x, y, z] landmark points, got ${got}`);
  }
}

export function extractFeatures(P) {
  assertLandmarks(P);
  return buildFeatures(P).values;
}

// flat63: 21 landmarks x, y, z flattened, in image-relative units, as saved
// by landmarks.js flattenRounded. aspect and sign as landmarks.js's toPoints
// and hand-frame.js's chirality expect: aspect = width/height, sign =
// chirality(...).sign.
export function featuresFromFlat(flat63, aspect, sign) {
  return extractFeatures(canonicalize(toPoints(flat63, aspect), sign));
}

export function blockIndices(block) {
  if (!BLOCKS.includes(block)) {
    throw new Error(`blockIndices: unknown block "${block}" -- valid blocks are ${BLOCKS.join(", ")}`);
  }
  const out = [];
  for (let i = 0; i < FEATURE_BLOCKS.length; i++) if (FEATURE_BLOCKS[i] === block) out.push(i);
  return out;
}
