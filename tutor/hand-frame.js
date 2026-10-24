/* hand-frame.js: which hand is this, and which way is it facing?
 *
 * Camera frame: x right, y down, z away. That frame is right-handed, so for a
 * hand that APPEARS as a right hand, cross(indexMCP - wrist, pinkyMCP - wrist)
 * points out of the palm. We mirror apparent-left hands so one model serves
 * both. MediaPipe's own Left/Right label is stored as reported, never
 * corrected (research/live/REPORT.md section 1 measured it: on this
 * platform's unmirrored video a physical right hand comes back "Right"), but
 * we still derive chirality from geometry rather than trust the label,
 * because the label can flip at frame edges and this test needs no
 * assumption about mirroring either way. We keep MediaPipe's label only as a
 * cross-check. */
import { sub, dot, cross, norm, unit } from "./vec.js";

const CHAINS = { index: [5, 6, 7, 8], middle: [9, 10, 11, 12], ring: [13, 14, 15, 16], pinky: [17, 18, 19, 20] };

// Degenerate input (coincident or collinear points, e.g. all 21 landmarks at
// the same location) yields margin: 0 and a zero normal rather than NaN --
// callers must treat margin ~ 0 as "do not trust," same as a legitimately
// flat hand.
export function chirality(P) {
  const n = unit(cross(sub(P[5], P[0]), sub(P[17], P[0])));
  let s = 0;
  for (const f of ["index", "middle", "ring", "pinky"]) {
    const [m, , d, t] = CHAINS[f];
    s += dot(sub(P[t], P[m]), n) + dot(sub(P[d], P[m]), n);
  }
  s += dot(sub(P[4], P[1]), n);
  // Fingers curl toward the palm. Which side of the MCP plane they curl to
  // tells left from right. A flat hand seen face-on curls nowhere: margin ~ 0.
  return { sign: s >= 0 ? 1 : -1, margin: s };
}

export const canonicalize = (P, sign) => P.map((p) => [p[0] * sign, p[1], p[2]]);

// How big this hand is in the picture: the mean of the five distances that
// span the palm (wrist to each of the four finger MCPs, plus index MCP to
// pinky MCP). It is the S in palmFrame below, i.e. the quantity every
// size-normalized feature divides by, so anything that wants to report a
// length "in hand widths" should use this one and not some other span.
//
// CHIRALITY-INVARIANT, and that is why it is exported on its own: it is built
// from distances, so mirroring x cannot change it, and unlike the rest of
// palmFrame it is valid on RAW points -- a caller that only needs the scale
// does not have to know which hand it is looking at, or canonicalize first.
// A degenerate hand (coincident points) gives 0, not NaN; callers dividing by
// it have to say what they want for that.
export function palmSize(P) {
  return (norm(sub(P[5], P[0])) + norm(sub(P[9], P[0])) + norm(sub(P[13], P[0])) + norm(sub(P[17], P[0])) + norm(sub(P[5], P[17]))) / 5;
}

// P must already be canonical right-hand points (i.e. canonicalize(P, sign)
// has been applied). On a raw left hand this still returns an orthonormal
// frame -- no NaN, no exception -- but n points out of the BACK of the hand
// and x points toward the pinky side, silently. There is no runtime
// assertion here because a flat hand's chirality is legitimately ambiguous
// (margin ~ 0), so asserting on sign would throw on valid input. Callers
// should go through featuresFromFlat (tutor/features.js), which
// canonicalizes first. S alone is safe on raw points: see palmSize.
export function palmFrame(P) {
  const y = unit(sub(P[9], P[0]));
  const n0 = unit(cross(sub(P[5], P[0]), sub(P[17], P[0])));
  const x = unit(cross(y, n0));
  const n = cross(x, y);
  return { x, y, n, S: palmSize(P) };
}
