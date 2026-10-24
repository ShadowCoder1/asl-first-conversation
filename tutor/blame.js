/* blame.js: when a hand is rejected, WHICH articulator should the learner
 * move -- and would moving it actually have been enough?
 *
 * A rejection is not feedback. "That is not an R" leaves a learner with 46
 * numbers' worth of ways to be wrong and no idea which one they are. The
 * tutor has to name one thing to fix, and the unit it names is a BLOCK: the
 * group of features one articulator controls (features.js), because a block
 * is the smallest group of numbers that maps back to something a hand can
 * actually do.
 *
 * THE COUNTERFACTUAL. The grader is one full-covariance Gaussian per letter,
 * so "what if this block were right?" has a closed form. For block index set
 * b, the rest of the hand r, and the target letter's mean mu and precision P:
 *
 *     mu_{b|r} = mu_b - P_bb^-1 P_br (z_r - mu_r)
 *
 * -- the mean of that block under the target letter GIVEN the rest of the
 * hand as it actually was. Not the marginal mean mu_b: the features are
 * correlated, and a thumb that is right for this particular index finger is
 * not the same as a thumb that is right on average. Put that block's
 * conditional mean into a copy of z, re-score, and the change in score is
 * what fixing the block would buy. The block that buys the most is blamed.
 *
 * P_bb^-1 is never formed. `solve(P_bb, P_br (z_r - mu_r))` gets the same
 * vector without inverting a 19x19 matrix. In exact arithmetic that solve
 * cannot fail -- a principal submatrix of a positive-definite matrix is
 * positive-definite, and loadModel has already factored the whole thing --
 * but the arithmetic is not exact: `solve` refuses a pivot that is small
 * RELATIVE to the largest entry in the block, and a precision matrix scaled
 * badly enough can pass loadModel's Cholesky and still be too ill-conditioned
 * here. That is a broken model, not a bad hand, so it throws -- naming the
 * block and the letter, because a bare linear-algebra complaint from inside a
 * hint is a stack nobody can place.
 *
 * THE MOVE GOES TO THE CONDITIONAL MEAN, not to the acceptance boundary. The
 * design (A5.3) asks for the smallest move that would be accepted, which is a
 * better description of what a learner has to do -- but "how far is far
 * enough" is a question about real hands, and there are none yet: the model
 * in this slice is bootstrap, fitted on public datasets of posed hands, with
 * no learner of this tutor in it. Moving to the mean is the version whose
 * arithmetic means something without captured data; the boundary refinement
 * waits for it.
 *
 * THE SELF-CONSISTENCY CHECK, AND WHAT IT IS NOT. Before a hint is shown,
 * `chooseCorrection` re-decides the corrected vector and only offers the hint
 * if the model would then accept it. That is the model checking its OWN
 * ARITHMETIC -- nothing more. A learner cannot move their thumb without
 * moving their index finger, their palm and the camera's view of all three,
 * so a hint that passes this check is not thereby shown to be physically
 * actionable, and this module makes no such claim. Whether hints work is an
 * empirical question, answered later by P(accept on the next attempt | hint
 * shown) on captured data (A1.4). What the check does buy is the opposite
 * guarantee, which is worth having: the tutor never says "fix your thumb"
 * about a hand that would still be rejected with a perfect thumb. When no
 * one- or two-block correction gets there, the honest answer is to show the
 * whole sign, and that is what `fallback` says.
 *
 * BLOCKS ARE RANKED PER UNIT OF WIDTH, NOT BY RAW GAIN. The blocks are very
 * different sizes -- 19 features in the thumb, 5 in the spread, 6 in
 * orientation, 4 in each finger -- and a wide block earns score just by being
 * wide: moving 19 features to their conditional mean soaks up whatever
 * ordinary atypicality those 19 happened to carry, whether or not the thumb
 * is what is wrong. Measured on U hands graded as R, the thumb's fix RAISED
 * the rival U's log-likelihood by 2.29 while the spread's LOWERED it by 1.34,
 * and the thumb still won on raw gain. That is the old "19 redundant thumb
 * features outvote 6 orientation features" objection (docs/specs/
 * review-dispositions.md B3) arriving by a different road.
 *
 * So blocks are ordered by `rank` = gain / sqrt(block size), and both numbers
 * are reported: `gain` is what the correction is worth in score, `rank` is
 * what decided the blame. The divisor was not picked by taste -- it was
 * chosen by a pre-declared evaluation over 11 minimal pairs and 330 real
 * hands, against raw gain and against a chi-square-style standardization,
 * choosing on one half of the rows and reporting on the other.
 * docs/BLAME_EVALUATION.md has the table, the protocol and the limits.
 *
 * ORIENTATION FIRST (A5.2). A hand held the wrong way and a hand with the
 * wrong finger flexion are not equally urgent: orientation is what the
 * receiver sees first, and several pairs (G/Q, K/P, U/H) differ in nothing
 * else. So a positive orientation RANK within 80% of the best block's takes
 * the correction even when a finger block ranks a little higher. The rule
 * compares the same number the blame was decided on; comparing gains while
 * ranking by rank would let a block jump a queue it was never near.
 *
 * TIER-BLIND ON PURPOSE. Every score here comes from `decide(..., {
 * ignoreTier: true })`. A tier-2 letter gets no verdict shown to a learner,
 * but a tier-aware `decide` answers "abstain, tier2" for every block and
 * every counterfactual alike, which is no answer at all -- and most letters
 * are tier 2 in the current bootstrap model, so blame would be blind exactly
 * where it is most needed. Nothing here is user-facing on its own: it
 * produces the block names a caller may choose to phrase a hint from, and
 * whether a tier-2 letter gets a hint at all is that caller's decision.
 *
 * This runs once per rejected attempt -- not per frame -- so it is written
 * for clarity rather than for speed: seven blocks, seven small solves and
 * eight re-scores, well under a millisecond on the real model.
 */
import { BLOCKS, blockIndices, FEATURE_NAMES } from "./features.js";
import { decide } from "./verifier.js";
import { solve, subMatrix, matVec } from "./linalg.js";

const F = FEATURE_NAMES.length;

// See "TIER-BLIND ON PURPOSE" above. Frozen so no caller can reach in and
// turn the tier back on for a module that cannot work with it.
const TIER_BLIND = Object.freeze({ ignoreTier: true });

/* How a block's gain is turned into the number the blocks are ordered by.
 * Named and exported so that a log, a test or a later analysis can say which
 * criterion a recorded blame was made under, and so that changing it is a
 * visible change rather than an edited expression. The evaluation that chose
 * it -- over raw gain and over (gain - k/2) / sqrt(k/2) -- is in
 * docs/BLAME_EVALUATION.md; changing this without redoing that evaluation
 * fails tests/blame.model.test.js. */
export const RANK_CRITERION = "gain / sqrt(block size)";
const rankOf = (gain, size) => gain / Math.sqrt(size);

/* A5.2. Orientation jumps the queue when its rank is at least this fraction
 * of the best block's -- a near-tie, not a win. A positive rank is required
 * as well: without it, a hand whose orientation is already right (rank 0)
 * would satisfy 0 >= 0.8 * 0 on every hand where nothing else helps either,
 * and the tutor would blame the palm on principle. */
const ORIENTATION_NEAR_TIE = 0.8;

/* The one block this module names out loud. blockIndices throws on a block
 * features.js does not have, so if the blocks are ever renamed this module
 * fails at import rather than quietly never applying the rule again. */
const ORIENTATION = "orientation";
blockIndices(ORIENTATION);

/* Which feature indices are in each block and which are not, worked out once
 * at load. FEATURE_BLOCKS is frozen and fixed at import, so these never go
 * stale, and blame does not rebuild them on every rejected attempt. In BLOCKS
 * order, which is the tie-break order for equal gains. */
const PARTITION = BLOCKS.map((block) => {
  const inBlock = blockIndices(block);
  const inside = new Set(inBlock);
  const rest = [];
  for (let i = 0; i < F; i++) if (!inside.has(i)) rest.push(i);
  return { block, inBlock, rest, size: inBlock.length };
});

const PART_BY_BLOCK = new Map(PARTITION.map((p) => [p.block, p]));

function checkVector(z, who) {
  if (z.length !== F) throw new Error(`${who}: expected ${F} standardized features, got ${z.length}`);
  for (let i = 0; i < F; i++) {
    /* A caller only blames an attempt the verifier rejected, and a rejected
     * attempt was gradable by construction -- so a NaN here is a caller that
     * skipped the verdict, not a hand that could not be measured. Every
     * number below would come back NaN and the sort would order the blocks
     * arbitrarily, which looks exactly like an answer. */
    if (!Number.isFinite(z[i])) throw new Error(`${who}: z[${i}] (${FEATURE_NAMES[i]}) is ${z[i]}, not a finite number -- a hand that was not gradable has nothing to blame`);
  }
}

/* A copy of z with one block replaced by its conditional mean under `cls`.
 * The copy is always fresh: a caller sorts these, hands them on, and edits
 * them, and two blocks' answers that shared a buffer would silently be the
 * same answer. */
function moveBlock(cls, z, part, target) {
  const { inBlock, rest } = part;
  const Pbb = subMatrix(cls.prec, inBlock, inBlock);
  const Pbr = subMatrix(cls.prec, inBlock, rest);
  const restDiff = new Array(rest.length);
  for (let k = 0; k < rest.length; k++) restDiff[k] = z[rest[k]] - cls.mu[rest[k]];
  let y;
  try {
    y = solve(Pbb, matVec(Pbr, restDiff));
  } catch (e) {
    // See the header: positive-definite is not the same as well-conditioned,
    // and the model file is where such a failure comes from.
    throw new Error(`blameBlocks: cannot compute the ${part.block} block's conditional mean for target ${JSON.stringify(target)} -- ${e.message}`);
  }
  const zFixed = Float64Array.from(z);
  for (let k = 0; k < inBlock.length; k++) zFixed[inBlock[k]] = cls.mu[inBlock[k]] - y[k];
  return zFixed;
}

const scoreOf = (model, z, target) => decide(model, z, target, TIER_BLIND).score;

/* One block moved to its conditional mean under `target`, given the rest of this hand (what blameBlocks does for every
 * block, for one): a fresh vector. For a caller that already has the gains (chooseCorrection's) and wants only which
 * way one named block has to go (tutor/feedback.js directionOf). */
export function blockAtTarget(model, z, target, block) {
  checkVector(z, "blockAtTarget");
  const part = PART_BY_BLOCK.get(block);
  if (!part) throw new Error(`blockAtTarget: unknown block "${block}"`);
  return moveBlock(model.classes[target], z, part, target);
}

/* Every block, with what moving it to its conditional mean would do to the
 * score, best first. One entry per block -- including the blocks that would
 * buy nothing, because "the ring finger was not the problem" is part of the
 * answer and a caller that logs blame wants the whole picture.
 *
 *   gain   the change in score: what the correction is worth.
 *   rank   RANK_CRITERION applied to it: what the blame was decided on, and
 *          what the entries are sorted by. Both are kept -- an attempt log
 *          that recorded only one of them could not be re-read against a
 *          later criterion.
 *   zFixed a Float64Array, the same shape `standardize` produces, the
 *          caller's to keep or edit. */
export function blameBlocks(model, z, target) {
  checkVector(z, "blameBlocks");
  // An unknown target raises the verifier's error, not a second copy of it.
  const before = decide(model, z, target, TIER_BLIND);
  if (!Number.isFinite(before.score)) {
    // decide refused to grade a finite vector: the arithmetic overflowed.
    // Gains would all be Infinity - Infinity, and a sort over NaN would hand
    // back an order that looks like a ranking.
    throw new Error(`blameBlocks: the verifier cannot grade this hand against ${JSON.stringify(target)} (${before.outcome}, ${before.reason}), so there is no gain to measure`);
  }
  const cls = model.classes[target];

  const entries = PARTITION.map((part) => {
    const zFixed = moveBlock(cls, z, part, target);
    const gain = scoreOf(model, zFixed, target) - before.score;
    return { block: part.block, gain, rank: rankOf(gain, part.size), zFixed };
  });
  // Array#sort is stable, so blocks with equal ranks -- every block on a hand
  // that is already on the target's mean, say -- keep BLOCKS order.
  entries.sort((a, b) => b.rank - a.rank);
  return entries;
}

/* The candidates in the order the correction is tried: A5.2's perceptual
 * problems first, then simply the biggest gain. `entries` is already sorted.
 */
function candidateOrder(entries) {
  const orientation = entries.find((e) => e.block === ORIENTATION);
  if (orientation.rank > 0 && orientation.rank >= ORIENTATION_NEAR_TIE * entries[0].rank) {
    return [orientation, ...entries.filter((e) => e !== orientation)];
  }
  return entries;
}

/* The one correction to show, or two, or none.
 *
 * PRECONDITION: call this on an attempt the verifier did NOT accept. A
 * correction is an answer to "why was that not an R?", and on a hand that WAS
 * an R there is no such question -- but there is almost always still a block
 * with a positive gain, because no real hand sits exactly on a mean, so
 * without the guard below this would hand back a confident, self-consistent
 * "move your thumb" about a sign the learner just got right. The guard is
 * here rather than left to the caller because the failure is silent: a tutor
 * wired to blame every attempt instead of every rejected one would produce
 * fluent nonsense and nothing would look broken.
 *
 *   blocks         the blocks to tell the learner about, in the order they
 *                  were applied; [] when there is no hint to give.
 *   selfConsistent the model's own arithmetic accepts the hand once these
 *                  blocks are corrected. See the header for what that does
 *                  and does not mean.
 *   fallback       true when the hand was rejected and no correction one or
 *                  two blocks wide would have been enough, so the caller
 *                  should show the whole sign instead. Its RATE is a number
 *                  worth watching per letter (A5.3): a tutor that falls back
 *                  all the time is a demo arm wearing a corrective arm's name.
 *   nothingToSay   true when the hand needs no correction at all -- the model
 *                  already accepts it. Distinct from `fallback`, which means
 *                  "something is wrong and I cannot say what in one block".
 *                  Exactly one of blocks/fallback/nothingToSay is the answer.
 *   gains          every block's gain from the first round of blame, for
 *                  logging -- including when the hint is empty.
 *
 * The second block's conditional mean is recomputed against the vector the
 * first correction produced, not against the original hand: after the first
 * block moves, what the rest of the hand implies about the second block has
 * moved with it. It must still have earned something of its own, though --
 * see below.
 */
export function chooseCorrection(model, z, target) {
  const entries = blameBlocks(model, z, target);
  const gains = {};
  for (const e of entries) gains[e.block] = e.gain;
  const noHint = { blocks: [], selfConsistent: false, fallback: true, nothingToSay: false, gains };

  // The precondition above, enforced. `fallback` stays false: there is
  // nothing to correct and nothing to fall back to either.
  if (decide(model, z, target, TIER_BLIND).outcome === "accept") {
    return { blocks: [], selfConsistent: false, fallback: false, nothingToSay: true, gains };
  }

  /* No block to blame: nothing the model can fix by itself helps at all.
   * Tested on the best RANK, the number the order was made on -- and since a
   * rank is a gain divided by a positive width, the best-ranked block having
   * no gain means no block has one. */
  if (!(entries[0].rank > 0)) return noHint;

  const accepts = (v) => decide(model, v, target, TIER_BLIND).outcome === "accept";
  const order = candidateOrder(entries);
  const first = order[0];
  if (accepts(first.zFixed)) return { blocks: [first.block], selfConsistent: true, fallback: false, nothingToSay: false, gains };

  /* One more, and then no more (A5.3): a third block is not a hint any
   * longer, it is a description of the whole hand -- which is exactly what
   * falling back and showing the sign already does, more honestly.
   *
   * The second block has to have earned something of its own. Reaching accept
   * is not enough on its own: a block can be merely ATYPICAL rather than
   * wrong -- costing distance while costing no score, so its gain is zero --
   * and moving it would then carry the pair past the typicality gate without
   * that block ever having been the problem. Telling a learner to move a
   * finger that was not wrong is worse than showing them the sign. */
  const second = order[1];
  if (!(second.rank > 0)) return noHint;
  const both = moveBlock(model.classes[target], first.zFixed, PART_BY_BLOCK.get(second.block), target);
  if (accepts(both)) return { blocks: [first.block, second.block], selfConsistent: true, fallback: false, nothingToSay: false, gains };
  return noHint;
}
