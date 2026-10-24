/* schedule-dropout.js: decides which letter comes next in PRACTICE -- a
 * seed-reproducible DROP-OUT schedule whose actual letter order depends on
 * how the learner performs.
 *
 * WHAT THIS IS: new letters are introduced one at a time, in the order the
 * caller gives them (never later than every `introEvery` trials -- see C1
 * below). A letter answered correctly and unassisted comes back after a
 * growing gap (`lags`, by default 2, then 5, then 10 other trials) and is
 * retired once it has cleared all of them; any other outcome -- wrong, or
 * right only with help -- sends it back to the front of that queue (due
 * again after `lags[0]`). When there is nothing due, no new letter to
 * introduce, and at least one letter has already retired, an idle trial is
 * spent reviewing the least-recently-seen retired letter rather than pulling
 * an active letter's own test in early -- see F2 below. The only randomness
 * (from the seed) is which of several EQUALLY due, equally new, or equally
 * overdue-for-review letters goes first; the SEQUENCE is otherwise driven
 * entirely by what the learner gets right and wrong.
 *
 * WHAT THIS IS FOR: the playable tutor itself, and piloting -- letting a
 * real or simulated learner practice fingerspelling with spaced repetition
 * that responds to how they're doing, the way a human coach would decide
 * what to drill next.
 *
 * WHAT THIS IS NOT FOR, AND WHY (correcting an earlier claim in this same
 * file): this is NOT the design spec's pre-generated, equal-exposure study
 * schedule, and it must never be used as one. A feedback experiment needs
 * every participant's *exposure* -- which letters, how many times, how far
 * apart -- to be independent of how they perform, so a difference in
 * outcomes can be attributed to the feedback condition instead of to "this
 * participant's mistakes changed what they practiced." That is exactly what
 * this module does NOT give you: a wrong or assisted answer here changes
 * what gets served next, so two participants who happen to get different
 * answers right will not see the same sequence even on the same seed. A
 * true fixed/equal-exposure schedule -- generated once from a seed, with no
 * feedback loop from performance back into scheduling -- is a separate,
 * later module; this one is not a substitute for it.
 *
 * WHY DETERMINISM STILL MATTERS HERE: not for equal exposure across
 * participants, but for REPLAY. `next()` never touches Math.random or Date,
 * only the `prng` built from `seed` (see tutor/prng.js), and only to break
 * ties. That means a single participant's exact sequence can always be
 * re-derived after the fact from two things saved with the session: the
 * seed, and the ordered list of `report()` outcomes -- useful for QA,
 * debugging a strange session, and for the snapshot the study log keeps
 * regardless of which schedule generated it.
 *
 * C1 (INTRODUCTION GUARANTEE): due-first priority, applied with no other
 * rule, has a lockstep failure mode. Once several already-introduced letters
 * are perpetually due (an always-wrong learner is the deterministic case;
 * a low-scoring one hits it often), the "never repeat the letter just
 * served" rule always has another due letter to offer instead of falling
 * through to "introduce the next new letter" -- so once that happens, no
 * further letter is ever introduced. `introEvery` (default 6) bounds how
 * long that can go on: if a not-yet-introduced letter remains and at least
 * `introEvery` trials have been served since the last introduction (or
 * since the session started), introducing it takes priority over even the
 * most overdue due letter. Below that threshold, priority is unchanged.
 *
 * F2 (REVIEW FILLERS): the earlier version of this module, faced with
 * nothing due and no new letter to introduce, filled the idle trial by
 * serving a not-yet-due active letter early -- which could crush that
 * letter's own gap requirement down to nothing. A retired letter needs no
 * protecting: reviewing one costs it nothing (it does not change its
 * retired status) and gives every ACTIVE letter's due date one more "other
 * trial" to be satisfied by, instead of stealing one. So once at least one
 * letter has retired, an idle trial reviews the least-recently-served
 * retired one (kind "review") rather than testing an active one early; the
 * old "serve the not-yet-due letter soonest" behavior is now the last
 * resort, only reachable before any letter has ever retired. */
import { createPrng } from "./prng.js";

function validateInputs({ letters, seed, lags, introEvery, maxTrials }) {
  if (!Array.isArray(letters) || letters.length === 0) {
    throw new Error(`createDropoutSchedule: letters must be a non-empty array of distinct strings, got ${JSON.stringify(letters)}`);
  }
  for (const l of letters) {
    if (typeof l !== "string") {
      throw new Error(`createDropoutSchedule: letters must all be strings, found ${JSON.stringify(l)}`);
    }
  }
  if (new Set(letters).size !== letters.length) {
    throw new Error(`createDropoutSchedule: letters must be distinct, got ${JSON.stringify(letters)}`);
  }

  if (!Number.isInteger(seed)) {
    throw new Error(`createDropoutSchedule: seed must be an integer, got ${JSON.stringify(seed)}`);
  }

  if (!Array.isArray(lags) || lags.length === 0) {
    throw new Error(`createDropoutSchedule: lags must be a non-empty array of positive integers, got ${JSON.stringify(lags)}`);
  }
  for (const k of lags) {
    if (!Number.isInteger(k) || k <= 0) {
      throw new Error(`createDropoutSchedule: lags must all be positive integers, found ${JSON.stringify(k)}`);
    }
  }

  if (!Number.isInteger(introEvery) || introEvery <= 0) {
    throw new Error(`createDropoutSchedule: introEvery must be a positive integer, got ${JSON.stringify(introEvery)}`);
  }

  if (!Number.isInteger(maxTrials) || maxTrials <= 0) {
    throw new Error(`createDropoutSchedule: maxTrials must be a positive integer, got ${JSON.stringify(maxTrials)}`);
  }
}

export function createDropoutSchedule({ letters, seed, lags = [2, 5, 10], introEvery = 6, maxTrials }) {
  validateInputs({ letters, seed, lags, introEvery, maxTrials });

  const prng = createPrng(seed);
  const order = []; // { letter, kind }
  const state = {}; // letter -> { introduced, retired, lagIndex, lastServedAt, dueAt, tests, errors, reviews }
  for (const letter of letters) {
    state[letter] = {
      introduced: false, retired: false, lagIndex: 0, lastServedAt: null, dueAt: null,
      tests: 0, errors: 0, reviews: 0, ungraded: 0,
    };
  }

  let trialsServed = 0;
  let introducedCount = 0; // how many of `letters`, in order, have been introduced
  let trialsSinceIntro = 0; // C1: trials served since the last intro (or since the start)
  let lastServedLetter = null;
  let pending = null; // { letter, kind } for the trial next() just returned, until report() clears it

  // True once `trialsServed` has reached (or passed) `dueAt` for a letter served at `t`
  // with the lag that now applies: due from trial index t + k + 1 onward (ruling 2).
  const isDue = (letter) => {
    const s = state[letter];
    return s.introduced && !s.retired && s.dueAt !== null && trialsServed >= s.dueAt;
  };

  // Picks one letter out of a tied group deterministically from the seed. `group` must
  // already be in a fixed order (the order `letters` was given in), so that which index
  // the PRNG lands on is reproducible for a given seed regardless of how the group was
  // assembled.
  const breakTie = (group) => (group.length === 1 ? group[0] : prng.pick(group));

  // The full ruling-3 + F2 selection, restricted to letters not in `excluded` (the letter
  // just served, or nothing). Returns { letter, kind } or null if every stage comes up empty
  // (only possible when `excluded` removed the only eligible letter).
  function selectGiven(excluded) {
    const notExcluded = (l) => !excluded.has(l);

    // Stage 1: among introduced, not-retired, due letters, the most overdue (ties by PRNG).
    const due = letters.filter((l) => isDue(l) && notExcluded(l));
    if (due.length > 0) {
      let best = -Infinity;
      for (const l of due) best = Math.max(best, trialsServed - state[l].dueAt);
      return { letter: breakTie(due.filter((l) => trialsServed - state[l].dueAt === best)), kind: "test" };
    }

    // Stage 2: introduce the next new letter, in the given `letters` order. `fresh` has never
    // been served, so it can never be the just-served (excluded) letter -- no membership
    // check needed here, unlike the due/retired pools above which mix served letters.
    if (introducedCount < letters.length) {
      return { letter: letters[introducedCount], kind: "intro" };
    }

    // Stage 3 (F2): nothing due, nothing new -- review the least-recently-served retired
    // letter (ties by PRNG), which costs that letter nothing and buys every still-active
    // letter one more "other trial" toward its own due date.
    const retired = letters.filter((l) => state[l].retired && notExcluded(l));
    if (retired.length > 0) {
      let best = Infinity;
      for (const l of retired) best = Math.min(best, state[l].lastServedAt);
      return { letter: breakTie(retired.filter((l) => state[l].lastServedAt === best)), kind: "review" };
    }

    // Stage 4: never stall -- nothing due, nothing new, and nothing retired to review yet
    // (only reachable before any letter has ever retired). Serve the not-retired letter due
    // soonest, even if it is not actually due yet.
    const notRetired = letters.filter((l) => !state[l].retired && notExcluded(l));
    if (notRetired.length > 0) {
      let best = Infinity;
      for (const l of notRetired) best = Math.min(best, state[l].dueAt);
      return { letter: breakTie(notRetired.filter((l) => state[l].dueAt === best)), kind: "test" };
    }

    return null;
  }

  function serve(letter, kind) {
    const t = trialsServed;
    const s = state[letter];
    s.lastServedAt = t;
    if (kind === "intro") {
      s.introduced = true;
      s.lagIndex = 0;
      s.dueAt = t + lags[0] + 1;
      introducedCount++;
      trialsSinceIntro = 0;
    } else {
      trialsSinceIntro++;
    }
    order.push({ letter, kind });
    trialsServed++;
    lastServedLetter = letter;
    pending = { letter, kind };
    return { letter, kind };
  }

  function next() {
    if (pending !== null) {
      throw new Error(`createDropoutSchedule: next() was called again before report() for "${pending.letter}" -- call report() exactly once after every next()`);
    }
    if (trialsServed >= maxTrials) return null;

    const anyNonRetired = letters.some((l) => !state[l].retired);
    if (!anyNonRetired) return null; // every letter retired -- never reviews forever (F2)

    // C1: the introduction guarantee overrides due-first priority once introEvery trials
    // have passed without an introduction, so already-introduced letters cannot perpetually
    // crowd out ones that have never been served at all.
    if (introducedCount < letters.length && trialsSinceIntro >= introEvery) {
      return serve(letters[introducedCount], "intro");
    }

    // Ruling 3: never repeat the letter just served unless it is the only eligible one left.
    // Try the selection with that letter excluded first, and only fall back to allowing it
    // when excluding it leaves every stage empty -- which only happens when it really is the
    // sole remaining option.
    const excluded = lastServedLetter === null ? new Set() : new Set([lastServedLetter]);
    const picked = selectGiven(excluded) ?? selectGiven(new Set());
    if (picked === null) return null; // unreachable given anyNonRetired, but never stall on a surprise

    return serve(picked.letter, picked.kind);
  }

  function report({ letter, firstAttemptCorrect, assisted, ungraded }) {
    if (pending === null) {
      throw new Error("createDropoutSchedule: report() was called with no trial pending -- call next() first");
    }
    if (pending.letter !== letter) {
      throw new Error(`createDropoutSchedule: report() was called for "${letter}" but the last trial served was "${pending.letter}"`);
    }
    if (ungraded !== undefined && typeof ungraded !== "boolean") {
      throw new Error(`createDropoutSchedule: report()'s ungraded must be a boolean when it is given, got ${JSON.stringify(ungraded)}`);
    }
    const kind = pending.kind;
    const s = state[letter];

    /* THE THIRD ANSWER: the trial happened and the learner was never graded.
     * The camera lost the hand three times running, the time ran out before
     * anything was held still, a replay ran dry. Before this existed a caller
     * had to pick one of the two booleans anyway, and the only honest-looking
     * choice -- firstAttemptCorrect: false -- recorded an ERROR: the letter's
     * lag index went back to zero and its mastery went backwards, immediately
     * after the tutor had told the learner it could not see their hand. That
     * is grading somebody on the tracker's failure.
     *
     * So an ungraded trial costs the letter nothing: no test, no error, no
     * change to the lag index or to a retired letter's status. It is COUNTED,
     * because a session full of them is a real finding about the camera or
     * the lighting rather than about the learner. And the letter still has to
     * be tested, so it comes back after lags[0] other trials -- the shortest
     * gap, not the one it had earned, since nothing was learned either way. */
    if (ungraded === true) {
      if (kind !== "intro") {
        s.ungraded++;
        // A retired letter keeps dueAt null: it has no gap requirement left,
        // and giving it one would put it back in the due pool it graduated
        // from. It is only ever served again as a review filler.
        if (!s.retired) s.dueAt = s.lastServedAt + lags[0] + 1;
      }
      pending = null;
      return;
    }

    if (kind === "test") {
      // I1: a non-boolean here was previously coerced silently -- "assisted": "true" (a
      // truthy string) advanced a lag it should have reset, and firstAttemptCorrect: 1 was
      // recorded as an error instead of being rejected. Grading depends on these two flags
      // meaning exactly what they say, so a caller passing anything else is a bug to surface
      // loudly, not a value to guess the intent of.
      if (typeof firstAttemptCorrect !== "boolean") {
        throw new Error(`createDropoutSchedule: report() for a test trial needs firstAttemptCorrect to be a boolean, got ${JSON.stringify(firstAttemptCorrect)}`);
      }
      if (typeof assisted !== "boolean") {
        throw new Error(`createDropoutSchedule: report() for a test trial needs assisted to be a boolean, got ${JSON.stringify(assisted)}`);
      }

      s.tests++;
      const succeeded = firstAttemptCorrect && !assisted;
      if (succeeded) {
        s.lagIndex++;
        if (s.lagIndex >= lags.length) {
          s.retired = true;
          s.dueAt = null;
        } else {
          s.dueAt = s.lastServedAt + lags[s.lagIndex] + 1;
        }
      } else {
        s.errors++;
        s.lagIndex = 0;
        s.dueAt = s.lastServedAt + lags[0] + 1;
      }
    } else if (kind === "review") {
      // F2: a review is reported like any other trial, but it is graded on nothing -- the
      // flags are ignored and the retired letter's status never changes. It still counts.
      s.reviews++;
    }
    // kind === "intro": firstAttemptCorrect/assisted are ignored (ruling 1); the letter's
    // lag index and due date were already set by serve() when it was introduced.

    pending = null;
  }

  function snapshot() {
    const letterSnapshots = {};
    for (const letter of letters) {
      const s = state[letter];
      letterSnapshots[letter] = {
        introduced: s.introduced, retired: s.retired, lagIndex: s.lagIndex,
        lastServedAt: s.lastServedAt, dueAt: s.dueAt, tests: s.tests, errors: s.errors, reviews: s.reviews,
        // How many trials of this letter ended with nobody graded (see report).
        // A high count here is a finding about the camera, not the learner.
        ungraded: s.ungraded,
      };
    }
    return {
      seed, lags: lags.slice(), introEvery, maxTrials, trialsServed, trialsSinceLastIntro: trialsSinceIntro,
      order: order.map((o) => ({ letter: o.letter, kind: o.kind })),
      letters: letterSnapshots,
    };
  }

  return { next, report, snapshot };
}
