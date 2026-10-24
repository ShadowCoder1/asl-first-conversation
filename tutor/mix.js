/* mix.js: our grader plus three fixes, chosen on the experts' hands (research/combine-2026-10-16/RESULTS.md).
 *
 *   1. P RULE (fixed, not fitted; JT's wrist-angle point). P is K pointing down. On every expert P the middle
 *      finger (landmark 9 -> 12) points below horizontal in the image, on every expert K above. So for a hand
 *      whose middle finger points up: P is never accepted, and P is not counted as a rival, which is what held
 *      upright K's back ("too close to call, rival P").
 *   2. O/E RULE (JT: the kinematic difference between the experts' O and E, as a heuristic). E folds the
 *      fingertips down onto their own knuckles; O arcs them forward to meet the thumb. Fluent O's turned to the
 *      camera look like E's from the front, but the fingertip-to-knuckle distance (tipToKnuckle) still tells
 *      them apart: within every expert every O is above every E. So for a hand at NOT_E.minTipToKnuckle or more:
 *      E is never accepted, and E is not counted as a rival (research/combine-2026-10-16/o_vs_e.py).
 *   3. SECOND OPINION (tutor/openfs.js). When our grader does not accept one of SECOND_OPINION.letters, accept
 *      anyway if OpenFS, reading the whole trial clip, puts the target first among the 24 static letters with
 *      probability >= SECOND_OPINION.minP, and the hand is inside the target's typicality gate. A tracking
 *      complaint (hand at the edge, too small) is never overridden.
 *
 * Held out one expert at a time: 95.6% -> 98.1% of signs recognized, 13 fixed (M 6, K 4, O 3), none broken, wrong
 * letters praised 0.86% -> 0.78%. The settings below are the ones fitted on all six experts. */

import { letterColumn } from "./openfs.js";
import { toPoints } from "./landmarks.js";
import { palmSize, palmFrame, canonicalize } from "./hand-frame.js";
import { sub, norm } from "./vec.js";
import { standardize } from "./verifier.js";
import { looksLike as nameLetter } from "./feedback.js";

export const SECOND_OPINION = Object.freeze({
  letters: Object.freeze(["I", "M", "N", "X"]),
  minP: 0.3,
  tailFrames: 7,     // the frames the hold's pose is the median of (tutor/commit.js window)
  minHoldFrames: 3,  // fewer tracked frames than this inside the hold and there is nothing to read
  maxFrames: 300,    // longer clips are cut to their last 300 tracked frames; on all 3,603 saved signs that changed no decision
});
export const NOT_E = Object.freeze({
  minTipToKnuckle: 0.52,   // experts' E 0.19-0.48, O 0.41-0.95 (all but one 0.54+); chosen held out per person (o_vs_e.py)
});
/* 4. B THUMB RULE (demo only, opt-in { bThumb: true }; the owner, 19 Oct: "i had my hand fully open and it passed B").
 *    B folds the thumb across the palm; an open hand sticks it out to the side, and the grader, which has no "open
 *    hand" class, took one as a confident B (score 9.7 against a cut of -5.1). How far the thumb tip sits out to the
 *    thumb side, in the hand's own frame (palm sizes; tutor/hand-frame.js palmFrame x): fluent signers' B -0.03
 *    typical, 0.58 at most (22 signs); study 1's accepted practice B's 0.65 at most (654); that open hand 0.89. Over
 *    B_THUMB.maxSideways, B is never accepted -- which changes none of those 676 signs. */
export const B_THUMB = Object.freeze({ maxSideways: 0.7 });
export function thumbSideways(flat, aspect, sign = 1) {
  const Q = canonicalize(toPoints(flat, aspect), sign);
  const { x, S } = palmFrame(Q);
  if (!(S > 0)) return NaN;
  const c = [0, 5, 9, 13, 17].reduce((a, i) => a.map((v, k) => v + Q[i][k] / 5), [0, 0, 0]);
  return ((Q[4][0] - c[0]) * x[0] + (Q[4][1] - c[1]) * x[1] + (Q[4][2] - c[2]) * x[2]) / S;
}
const STATIC24 = "ABCDEFGHIKLMNOPQRSTUVWXY".split("");

/** Does the middle finger point below horizontal on screen? (x scaled by the video's aspect; image y points down) */
export function middleFingerDown(flat, aspect) {
  const dx = (flat[12 * 3] - flat[9 * 3]) * aspect, dy = flat[12 * 3 + 1] - flat[9 * 3 + 1];
  const len = Math.hypot(dx, dy);
  return len > 0 ? dy / len >= 0 : true;   // no direction at all: say nothing against P
}

/** How far the four fingertips stand from their own knuckles: the mean 3-D distance tip -> MCP over index, middle,
 * ring and pinky, in palm sizes (hand-frame.js palmSize, the scale of every distance feature). Distances only, so
 * either hand and any palm direction. NaN for a hand with no size. */
export function tipToKnuckle(flat, aspect) {
  const P = toPoints(flat, aspect), S = palmSize(P);
  if (!(S > 0)) return NaN;
  let sum = 0;
  for (const tip of [8, 12, 16, 20]) sum += norm(sub(P[tip], P[tip - 3]));
  return sum / 4 / S;
}

/**
 * OpenFS's read of a hold: the 24-letter probabilities, from the mean log-softmax over the hold's last tracked
 * frames, the whole clip read once.
 * @param {object} openfs                 createOpenFS(...)
 * @param {{tMs:number, flat:number[]}[]} clip  the trial's tracked frames so far, oldest first
 * @param {number} tStart  hold start (ms, trial clock)
 * @param {number} tCommit hold end
 * @returns {{top:string, p:Object<string,number>}|null}
 */
export function openfsRead(openfs, clip, tStart, tCommit) {
  let frames = clip.filter((f) => f.tMs <= tCommit + 1);
  if (frames.length > SECOND_OPINION.maxFrames) frames = frames.slice(-SECOND_OPINION.maxFrames);
  const hold = [];
  frames.forEach((f, k) => { if (f.tMs >= tStart - 1) hold.push(k); });
  if (hold.length < SECOND_OPINION.minHoldFrames) return null;
  const tail = hold.slice(-SECOND_OPINION.tailFrames);
  const T = frames.length, xy = new Float64Array(T * 42);
  frames.forEach((f, t) => { for (let j = 0; j < 21; j++) { xy[t * 42 + 2 * j] = f.flat[3 * j]; xy[t * 42 + 2 * j + 1] = f.flat[3 * j + 1]; } });
  const m = openfs.meanLogSoftmax(xy, T, tail);
  const lp = STATIC24.map((c) => m[letterColumn(c)]);
  const mx = Math.max(...lp), z = lp.reduce((s, v) => s + Math.exp(v - mx), 0);
  const p = {};
  STATIC24.forEach((c, i) => { p[c] = Math.exp(lp[i] - mx) / z; });
  const top = STATIC24.reduce((a, c) => (p[c] > p[a] ? c : a), STATIC24[0]);
  return { top, p };
}

/**
 * A verify() with the two fixes, same signature, so tutor/flow.js takes it as deps.verify.
 * @param {object} o
 * @param {Function} o.verify       tutor/verifier.js verify
 * @param {() => object|null} o.getOpenFS   the loaded OpenFS, or null (then no second opinion)
 * @param {() => {flat:number[], aspect:number, tStart:number, tCommit:number, clip:object[]}|null} o.getCommit
 *        what the engine is grading right now
 * The result is verify's, plus `mix`: { old: {outcome, reason, rival}, pRule, tipToKnuckle, notE, openfs: {top, p}|null,
 * by } where `by` names what decided an outcome that differs from the old grader's ("p-rule" | "o-e-rule" |
 * "second-opinion").
 * looksLike (opt-in, the demo): on a reject, `looksLike` is the letter the hand would be accepted as instead, under the
 * same rules (tutor/feedback.js looksLike), or null. It never changes the verdict.
 */
export function createMixedVerify({ verify, getOpenFS, getCommit, bThumb = false, looksLike = false }) {
  return function mixedVerify(model, x, target, quality, opts) {
    const old = verify(model, x, target, quality, opts);
    const ctx = getCommit();
    if (!ctx) return { ...old, mix: { old: pick(old), pRule: false, tipToKnuckle: null, notE: false, openfs: null, by: null } };
    const pointsUp = !middleFingerDown(ctx.flat, ctx.aspect);
    const reach = tipToKnuckle(ctx.flat, ctx.aspect);
    const notE = Number.isFinite(reach) && reach >= NOT_E.minTipToKnuckle;
    // the letters this hand cannot be, each with the letter it is most likely being mistaken for instead
    const ruledOut = [...(pointsUp ? [["P", "K"]] : []), ...(notE ? [["E", "O"]] : [])].filter(([c]) => c in model.index);
    let v = ruledOut.length ? verify(model, x, target, quality, { ...opts, excludeRivals: ruledOut.map(([c]) => c) }) : old;
    let by = null;
    const ruled = ruledOut.find(([c]) => c === target);
    if (ruled && v.reason !== "tracking" && v.reason !== "not-gradable") {
      v = { ...v, outcome: "reject", reason: null, rival: v.rival && !ruledOut.some(([c]) => c === v.rival) ? v.rival : ruled[1] };
    }
    if (v.outcome !== old.outcome) {
      // which rule changed it: the one about the target if there is one, else the one whose rival was removed
      const which = ruled ? ruled[0] : ruledOut.some(([c]) => c === old.rival) ? old.rival : ruledOut[0]?.[0];
      by = which === "E" ? "o-e-rule" : "p-rule";
    }
    let read = null;
    const openfs = getOpenFS();
    if (v.outcome !== "accept" && !ruled && SECOND_OPINION.letters.includes(target) && openfs
        && v.reason !== "tracking" && v.reason !== "not-gradable" && Number.isFinite(v.d2) && v.d2 <= v.gate) {
      read = openfsRead(openfs, ctx.clip, ctx.tStart, ctx.tCommit);
      if (read && read.top === target && read.p[target] >= SECOND_OPINION.minP) {
        v = { ...v, outcome: "accept", reason: null, rival: null };
        by = "second-opinion";
      }
    }
    let thumbOut = null;
    if (bThumb && target === "B" && v.outcome === "accept") {
      thumbOut = thumbSideways(ctx.flat, ctx.aspect, ctx.sign ?? 1);
      if (thumbOut > B_THUMB.maxSideways) { v = { ...v, outcome: "reject", reason: null, rival: null }; by = "b-thumb"; }
    }
    let named;
    if (looksLike && v.outcome === "reject") {
      const notB = bThumb && (thumbOut ?? thumbSideways(ctx.flat, ctx.aspect, ctx.sign ?? 1)) > B_THUMB.maxSideways;
      named = nameLetter(model, standardize(model, x), target, { exclude: ruledOut.map(([c]) => c), notB });
    }
    return { ...v, ...(looksLike ? { looksLike: named ?? null } : {}),
      mix: { old: pick(old), pRule: pointsUp, tipToKnuckle: Number.isFinite(reach) ? reach : null, notE, thumbOut,
        openfs: read && { top: read.top, pTarget: read.p[target] ?? null }, by } };
  };
}

const pick = (v) => ({ outcome: v.outcome, reason: v.reason ?? null, rival: v.rival ?? null });
