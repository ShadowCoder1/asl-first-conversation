/* pick-hand.js: when both hands are in the picture, which one is the learner's.
 *
 * The study asks for ONE hand (experiments/asl-study.js studyHand). With the
 * tracker told to find a single hand, MediaPipe returned whichever it was
 * surest of, and the dots jumped to the other hand whenever it came into view
 * (JT, 2026-09-23). So the tracker now looks for two, and this picks the one
 * asked for; the other is never graded, drawn or saved.
 *
 * In order:
 *   1. MediaPipe's label. With two hands in view it rarely labels both the
 *      same; on this platform's UNMIRRORED frames a physical right hand reads
 *      "Right" (research/live/REPORT.md section 1).
 *   2. If the labels do not settle it, the hand's own geometry (tutor/hand-frame.js
 *      chirality), when it is clear: +1 is a right hand.
 *   3. Otherwise, the hand nearer to where the chosen hand was last frame, so
 *      the dots do not flicker between two hands on an ambiguous frame.
 *
 * ONE hand in view is passed through whichever it is: a learner using the
 * wrong hand is told so by the tutor ("Use your right hand."), which needs to
 * see it. */

import { chirality } from "./hand-frame.js";

const WANT_SIGN = { right: 1, left: -1 };
export const PICK_MARGIN = 0.15;   // the engine's HAND_MARGIN: below it the geometry is not trusted

const pointsOf = (lm, aspect) => lm.map((p) => [p.x * aspect, p.y, (p.z ?? 0) * aspect]);

/**
 * @param {{landmarks:object[][], handedness:string[]}} res   the tracker's result
 * @param {"right"|"left"} want
 * @param {{aspect?:number, last?:{x:number,y:number}|null}} [o]
 * @returns {number} the index of the hand to use, or -1 for none
 */
export function pickHand(res, want, { aspect = 4 / 3, last = null } = {}) {
  const n = res?.landmarks?.length ?? 0;
  if (n === 0) return -1;
  if (n === 1) return 0;
  const idx = [...Array(n).keys()];
  const wantLabel = want === "left" ? /^l/i : /^r/i;

  const byLabel = idx.filter((i) => wantLabel.test(res.handedness?.[i] ?? ""));
  if (byLabel.length === 1) return byLabel[0];

  const byShape = idx.filter((i) => {
    const c = chirality(pointsOf(res.landmarks[i], aspect));
    return Math.abs(c.margin) >= PICK_MARGIN && c.sign === WANT_SIGN[want];
  });
  if (byShape.length === 1) return byShape[0];

  if (last) {
    const d = (i) => Math.hypot(res.landmarks[i][0].x - last.x, res.landmarks[i][0].y - last.y);
    return idx.reduce((a, b) => (d(b) < d(a) ? b : a));
  }
  return 0;
}

/* A picker that remembers where the chosen hand was, for step 3. */
export function createHandPicker(want) {
  let last = null;
  return (res, aspect) => {
    const i = pickHand(res, want, { aspect, last });
    last = i >= 0 ? { x: res.landmarks[i][0].x, y: res.landmarks[i][0].y } : null;
    return i;
  };
}

/* Which hand this is, by the engine's own rule (experiments/asl-tutor/engine.js):
 * the geometry when it is clear, else MediaPipe's label, else unknown (null).
 * Used to stop someone who is signing with the other hand (the camera check). */
export function handSide(landmarks, label, aspect = 4 / 3) {
  const c = chirality(pointsOf(landmarks, aspect));
  if (Math.abs(c.margin) >= PICK_MARGIN) return c.sign === 1 ? "right" : "left";
  if (/^r/i.test(label ?? "")) return "right";
  if (/^l/i.test(label ?? "")) return "left";
  return null;
}
