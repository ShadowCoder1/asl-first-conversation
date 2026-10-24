/* commit.js: turns a stream of tracked-hand frames into discrete ATTEMPTS.
 *
 * FRAMES ARE NOT ANSWERS. While a learner is still forming a handshape, every
 * intermediate frame is a real hand caught mid-motion; grading it would
 * punish the formation phase, not the shape. So this module waits for the
 * hand to go STILL -- a hold-still ring fills while landmark displacement
 * stays under tolerance -- and only the moment that hold completes is an
 * attempt.
 *
 * WHY THE LEARNER DECLARES THE ATTEMPT, rather than the tutor deciding for
 * them (design spec A4): "stable" cannot mean "correct," because a resting
 * hand is stable too, and MediaPipe's VIDEO-mode tracking smooths frame to
 * frame, so even a wrong or half-formed pose reads as motionless once it has
 * held for a few frames. Grading whatever happens to be still -- mid-formation
 * or at rest -- would reward the wrong thing. The hold-still ring asks for the
 * same thing a human coach would ("hold it there"), not a correctness check.
 *
 * CORRECTNESS-BLIND BY CONSTRUCTION: this module never imports the model,
 * the letter target, or anything from js/core/ or config.js, and never asks
 * "is this right" -- only "has this stopped moving." That separation means
 * committing an attempt cannot itself leak whether the shape is correct: the
 * commit boundary is drawn without ever looking at a grade.
 *
 * FIX ROUND 1 (jitter is ABSOLUTE, tolerance was RELATIVE): the original
 * stillness check compared a single anchor frame to a single current frame
 * and divided by palm size -- a NOISE-VS-NOISE comparison, since MediaPipe's
 * per-frame jitter has a fixed size in image units regardless of how big the
 * hand looks. For a small or distant hand (small palm size S), the same
 * absolute jitter becomes a LARGER fraction of S, so a purely relative
 * tolerance (stillTol*S) shrinks toward zero exactly where jitter does not --
 * measured at 99.9% of 600ms holds spuriously resetting at hand size 0.15.
 * Two independent fixes, both required:
 *   - a tolerance FLOOR, `absTol` (raw isotropic units, not a fraction of S):
 *     `tol = max(stillTol*S, absTol)`. For a small hand this floor is what
 *     keeps jitter noise from resetting the ring at all -- stillTol*S alone
 *     is too small to see over the noise, so a fixed absolute minimum takes
 *     over.
 *   - SMOOTHING: compare the per-coordinate MEDIAN of several rows, not one
 *     frame, on both sides (median of the still period's first `SMOOTH_N`
 *     usable rows as a FIXED anchor once collected; median of the last
 *     `SMOOTH_N` usable rows as the current pose). One frame's jitter is
 *     mostly noise; five combined is close to the true position.
 *
 * FIX ROUND 2 (three problems the reviewer found in fix round 1's build):
 *   - C1, `minPalm`: the floor from round 1 has to be a fixed size in raw
 *     units, which means it does NOT shrink for a small hand -- and a small
 *     hand's SMOOTHED anchor/current comparison also carries a ~4-frame
 *     (~133ms) lag (median-of-5 vs. median-of-5 compares window CENTERS, not
 *     the latest sample), so a slow drift's measured displacement at
 *     t=holdMs is `v*(holdMs - ~133ms)`, not `v*holdMs`. At hand size 0.10
 *     that shortfall is enough for the check to never fire: a 0.2 palm/s
 *     drift measurably, deterministically COMMITTED. Below `minPalm`,
 *     stillness is not a measurable question -- signal and noise are the
 *     same size -- so the honest behavior is to refuse rather than guess:
 *     `state.tooSmall` goes true and no still period can run at all.
 *   - I2, `reset()`: fix round 1 read the controller's "the monotonicity
 *     guard survives reset()" as surviving a BARE reset() specifically,
 *     requiring an explicit reset(tMs) to cross an old high-water mark. That
 *     was the wrong reading -- js/core/recorder.js's tMs restarts near 0 on
 *     EVERY trial, and the tutor calls the plain reset() at every trial
 *     boundary, so the old reading crashed a normal session on trial 2. A
 *     bare reset() now always clears the guard; reset() takes no argument.
 *   - I3, the progress latch: `progress` is `(tMs-periodStart)/holdMs`
 *     clipped to 1, and once it hits 1 it STAYS there as tMs keeps climbing,
 *     with no mechanism to unstick it. If a still hold's present-fraction
 *     never clears `presentFrac` (e.g. one dropped frame in three, each
 *     individually well under the 300ms dropout threshold), the hold-time
 *     target is reached but the commit condition never passes -- so the old
 *     code just sat at progress=1 forever, the ring reading "done" while
 *     nothing happened. Now: reaching hold time without enough presence
 *     RESETS the period (restarting from this frame) instead of latching.
 *     `state.lastResetReason` says which of the four reasons the last reset
 *     was, so the UI can react appropriately.
 * FIX ROUND 3 (the minPalm boundary itself flapped): size 0.13's palmSize
 * (~0.1200) sits within single-frame measurement noise (std ~0.0019) of
 * `minPalm`'s default (0.12) -- round 2's RAW single-frame gate flipped
 * `tooSmall` on and off almost every frame for a hand at exactly that size,
 * and every flip reset the still period. A hard cutoff always has SOME
 * boundary; the fix is to make the boundary calm and documented rather than
 * moving it. Two changes: the gate now reads a SMOOTHED palm size (the
 * median of the last `SMOOTH_N` usable frames' `palmSize`, mirroring the
 * position smoothing above -- fix round 2's proposal (b)); and it uses
 * HYSTERESIS, not one threshold: `tooSmall` turns on when smoothed S drops
 * below `minPalm`, but only turns off once smoothed S climbs above
 * `minPalm * MIN_PALM_EXIT_FACTOR`. Without the gap, a hand sitting on the
 * threshold would flip state every few frames purely from size-measurement
 * noise; with it, a hand whose TRUE size sits below the exit threshold (as
 * 0.13's does -- 0.1200 can never clear 0.12*1.08=0.1296) enters `tooSmall`
 * at most once and then stays there, calmly, for as long as it stays that
 * size -- which is the honest answer, not a hidden default change.
 *
 * THREE ZONES, in plain words (wrist-to-middle-knuckle length as a fraction
 * of image height): >= 0.15 is the SUPPORTED range -- measured reliably,
 * and what the statistical battery in tests/commit.test.js asserts (0.15,
 * 0.25, 0.40 at 30fps and 12fps: spurious-reset rate < 2%, zero commits
 * under a 0.2 palm/s drift with jitter, a real fingertip change always
 * resets, and the hand is never `tooSmall`). Below ~0.12 the module REFUSES
 * outright (`state.tooSmall` true, no still period can run -- "move your
 * hand closer" is the honest answer, not a guess). In between (roughly
 * 0.12-0.15) is a GRAY ZONE: the module may call a hand in that range
 * measurable or too-small -- it is not required to say the same thing every
 * session, or even every few seconds, at that size -- but per this round it
 * MUST NOT FLAP within one session once it settles.
 *
 * See tests/commit.test.js for the measurements behind these numbers: the
 * jitter spurious-reset battery ("C1 acceptance: still-hand spurious-reset
 * rate < 2% at 0.15/0.25/0.40, 30fps and 12fps"); the drift battery ("C1
 * acceptance: 0.2 palm/s drift + jitter never commits at 0.15/0.25/0.40,
 * 30fps and 12fps (zero of >=300 trials)", together with "C1
 * characterization: noiseless drift at 12fps slips through at hand sizes
 * 0.13 and 0.15 (known concern, see report)" for the characterization of
 * exactly how small a hand can slip a slow drift through `minPalm`); and the
 * size/hysteresis test ("fix round 3: at hand size 0.13 held still for 3s
 * with standard jitter, tooSmall settles CALMLY -- changes at most once per
 * run, 30fps and 12fps, 100 seeds each") for the boundary-noise measurement
 * that motivated the hysteresis. Measured cost: ~19us/push (fix round 1
 * reviewer benchmark; later rounds add a few more scalar operations per push
 * and stay in the same range) -- cheap for a 30-60fps loop with headroom to
 * spare.
 *
 * Time comes ONLY from the `tMs` each caller passes to push() -- never
 * Date/performance/frame counts -- so a saved frame stream replays to the
 * exact same attempts every time. See push() for why `tMs` must never go
 * backwards, and reset() for how a new trial's clock is allowed to restart. */
import { toPoints, medianFlat } from "./landmarks.js";

/* The committer's geometry in units of the picture's SHORT side. toPoints measures in picture heights, and two floors
 * here are absolute in those units (minPalm, absTol): on a phone held upright, whose camera sends 480 x 640, a height
 * is 640 px, so the same hand at the same distance had to be a third bigger in pixels than on a laptop's 640 x 480
 * before a hold could start (the phone version's review, 22 Oct: a propped-up phone got a green glove and a ring that
 * never filled). A landscape picture's short side IS its height, so nothing changes there; an upright one is measured
 * in its width, the same 480 px. Everything else here is relative to the palm, so only those two floors move. */
function pointsOf(flat, aspect) {
  const P = toPoints(flat, aspect);
  if (!(aspect < 1)) return P;
  const k = 1 / aspect;
  for (const p of P) { p[0] *= k; p[1] *= k; p[2] *= k; }
  return P;
}
import { palmSize } from "./hand-frame.js";

// MediaPipe's z is triangulated from a single 2-D view rather than measured
// directly, so at rest it jitters more than x/y. Down-weighting it in the
// displacement measure keeps that extra depth noise from resetting a hold
// that is, in x/y -- the axes a learner can actually see themselves hold
// still in -- genuinely still. (See tests/commit.test.js's jitter
// spurious-reset battery, "C1 acceptance: still-hand spurious-reset rate <
// 2% at 0.15/0.25/0.40, 30fps and 12fps", for the measurement that picked
// 0.5 rather than some other value.)
export const Z_WEIGHT = 0.5;

// How many usable rows are combined -- by per-coordinate median, reusing
// medianFlat's own semantics (odd count, unusable rows skipped) -- into the
// "anchor" and "current" poses that stillness and rearm are judged against.
// See the header for why a single frame is not enough on its own.
export const SMOOTH_N = 5;

// A gap this long is not "a still hand we briefly lost sight of," it is "we
// do not know what the hand did for a third of a second" -- long enough that
// neither continuing the old hold nor trusting the old commit is safe. It
// both breaks an in-progress hold (ruling 2) and, on its own, re-arms a
// disarmed committer (ruling 4).
const DROPOUT_MS = 300;

// Hysteresis on the minPalm gate (fix round 3): a hand whose smoothed size
// sits within noise of `minPalm` would otherwise flip `tooSmall` on and off
// every few frames, and every flip resets the still period. `tooSmall` turns
// off only once the smoothed size climbs 8% above `minPalm` -- wide enough
// that a hand which is only ever a hair below the line (like size 0.13,
// whose true palmSize never clears this margin) settles into `tooSmall` and
// stays there instead of bouncing, while staying tight enough that a hand
// which has genuinely grown is recognized without much delay.
export const MIN_PALM_EXIT_FACTOR = 1.08;

// `settleMs` is the part of the hold during which the ring shows NOTHING. A
// still period starts on the first measurable frame, so a hand that is
// merely passing through the picture is always "0 ms into a hold"; drawing
// that as progress made the ring twitch for every moving hand (first camera
// session, 2026-09-20). Progress stays 0 until the hand has been still for
// settleMs, then runs 0 -> 1 over the rest of holdMs. The commit itself still
// lands at holdMs exactly, so settleMs changes what is SHOWN, never what is
// graded.
//
// `stillSkip` / `rearmSkip`: how many of the LARGEST per-landmark
// displacements are ignored before the comparison (0 = the plain maximum).
// `stillPerFinger`: instead, take each finger's SECOND largest landmark
// displacement and then the largest of those five. Measured on real tracked
// video (research/live/REPORT.md, 60 labeled held letters, three signers):
// with the plain maximum at stillTol 0.06, 12% of the frames INSIDE a held
// letter broke the hold and 11 of 60 holds ever closed a 600 ms ring -- single
// landmarks (an occluded fingertip, the wrist) hop while the hand is still.
// Ignoring the k largest overall fixes that and opens a hole: a finger curling
// at its middle joint moves only its last two landmarks, which are exactly the
// two ignored. Per finger, one hopping landmark in EACH finger is ignored,
// and a finger with two landmarks travelling together is a moving finger.
// (research/live/eval-still-policies.mjs compares them on the real video.)
// The module's defaults stay where its statistical battery characterizes
// them; the tutor passes the measured values (TUTOR_COMMIT_OPTS).
//
// `placeTol`: when it is a finite number, stillness is judged in two parts.
// SHAPE -- the pose with its palm centre removed -- has to stay within
// stillTol; PLACE -- the palm centre itself -- only within placeTol (both in
// palm widths). Grading does not depend on where the hand is, so a
// hand wobbling as a whole on the end of a raised arm is not a hand that is
// changing shape, and on the real video it was whole-hand wobble, not fingers,
// that broke genuinely held letters: split this way with a TIGHTER shape
// tolerance, a comparison script at a 600 ms hold closed 42 of 60 real holds
// against 11 for the plain rule, and the grader accepted every pose those
// holds committed. (A comparison, on fluent signers holding 0.7-1.7 s: not a
// rate to expect from this module at a longer hold. tests/real-holds.test.js
// runs the real module on a real clip at learner-like speed.)
// Infinity (the default) is the original single comparison.
//
// `stillAlign`: SHAPE is compared twice and the SMALLER distance counts: as
// above, and again after the current pose has been turned and scaled in the
// picture plane to sit on the anchor's palm (alignedTo). Without it a hand is
// only "still" if it is not turning: a fingertip sits about two palm widths
// from the palm centre, so THREE DEGREES of wrist rock moves it the whole 0.10
// tolerance, and on the first recorded session a synthetic 12-degree rock
// closed 10 of 57 rings and half a palm of sway closed 5
// (research/recordings-2026-09-21/). Turning the whole hand is not a change of
// shape; a finger bending still is, aligned or not, because the alignment is
// fitted to the palm and a finger moves against it. The smaller of the two
// rather than the aligned one alone: fitting an angle to five noisy palm
// landmarks adds a little noise of its own, and a hand that IS still must
// never do worse than it did before. With stillAlign, placeTol Infinity means
// "anywhere" instead of "one comparison".
//
// `heldOverShape` / `heldOverPlace`: what it takes to re-arm a committer that
// nextTrial() carried over disarmed. Deliberately MUCH more than rearmTol. The
// two mistakes are not alike: re-arming too readily grades the last letter's
// pose against the next letter (the burst), and tracker noise on a hand nobody
// has moved reaches rearmTol within a second on real video; re-arming too
// reluctantly asks the learner to do what they do between two signs anyway --
// the status line says so -- and any two different letters differ by far more
// than this. Shape in palm widths (shapeDistance), place likewise, or the
// hand gone for DROPOUT_MS. heldOverPlace Infinity: place never re-arms --
// needed once a MOVING hand can commit (stillAlign), or a learner swaying with
// the last letter still up re-arms by place and has it graded as the next.
//
// `rearmAfterMs`: a disarmed committer normally re-arms only when the hand
// moves away from the pose it committed. Some real corrections are smaller
// than any safe movement threshold ("spread your fingers a little"), and the
// learner would face a dead ring for doing as they were told. After this long
// disarmed, WITHIN a trial, the committer re-arms anyway: a pose held that
// long after a verdict is a second answer. Never across a trial boundary
// (nextTrial): that is exactly the burst. Infinity turns it off.
export const DEFAULTS = Object.freeze({ holdMs: 600, settleMs: 250, stillTol: 0.06, stillSkip: 0, stillPerFinger: false, stillAlign: false, presentFrac: 0.8, rearmTol: 0.15, rearmSkip: 0, rearmAfterMs: Infinity, heldOverShape: 0.35, heldOverPlace: 0.6, placeTol: Infinity, window: 7, absTol: 0.010, minPalm: 0.12 });

function assertOption(opts, key, ok, why) {
  const v = opts[key];
  if (typeof v !== "number" || !Number.isFinite(v) || !ok(v)) {
    throw new Error(`createCommitter: opts.${key} = ${JSON.stringify(v)} is invalid -- ${why}`);
  }
}

function validateOptions(cfg) {
  assertOption(cfg, "holdMs", (v) => v > 0, "must be a positive number of milliseconds");
  assertOption(cfg, "settleMs", (v) => v >= 0 && v < cfg.holdMs, "must be at least 0 and less than holdMs");
  assertOption(cfg, "stillTol", (v) => v > 0, "must be a positive number, in units of palm size");
  assertOption(cfg, "presentFrac", (v) => v > 0 && v <= 1, "must be in (0, 1]");
  assertOption(cfg, "rearmTol", (v) => v > 0, "must be a positive number, in units of palm size");
  assertOption(cfg, "window", (v) => Number.isInteger(v) && v > 0 && v % 2 === 1, "must be a positive odd integer");
  if (typeof cfg.stillAlign !== "boolean") throw new Error(`createCommitter: opts.stillAlign = ${JSON.stringify(cfg.stillAlign)} is invalid -- must be true or false`);
  if (typeof cfg.stillPerFinger !== "boolean") throw new Error(`createCommitter: opts.stillPerFinger = ${JSON.stringify(cfg.stillPerFinger)} is invalid -- must be true or false`);
  assertOption(cfg, "stillSkip", (v) => Number.isInteger(v) && v >= 0 && v <= 10, "must be a whole number from 0 to 10 (landmarks ignored)");
  assertOption(cfg, "rearmSkip", (v) => Number.isInteger(v) && v >= 0 && v <= 10, "must be a whole number from 0 to 10 (landmarks ignored)");
  if (cfg.placeTol !== Infinity) assertOption(cfg, "placeTol", (v) => v > 0, "must be a positive number in units of palm size, or Infinity for a single comparison");
  assertOption(cfg, "heldOverShape", (v) => v > 0, "must be a positive number, in units of palm size");
  if (cfg.heldOverPlace !== Infinity) assertOption(cfg, "heldOverPlace", (v) => v > 0, "must be a positive number in units of palm size, or Infinity so that only a change of shape re-arms");
  if (cfg.rearmAfterMs !== Infinity) assertOption(cfg, "rearmAfterMs", (v) => v > 0, "must be a positive number of milliseconds, or Infinity for never");
  assertOption(cfg, "absTol", (v) => v > 0, "must be a positive number, in the same raw isotropic units as displacement");
  assertOption(cfg, "minPalm", (v) => v > 0, "must be a positive number, in the same palm-size units as palmSize()");
}

// Whether this row has a hand at all, and its RAW (un-smoothed) palm size if
// so. "No hand" covers any non-finite coordinate (ruling 1) and a
// geometrically degenerate hand (palmSize 0, e.g. every landmark collapsed
// onto the same point -- nothing to normalize a displacement against, so
// this reads exactly like a missing detection rather than dividing by zero
// downstream). The size itself is deliberately NOT classified here as
// too-small-or-not -- that decision needs the smoothed, hysteresis-tracked
// size the caller maintains across frames (fix round 3), not one raw value.
function frameStatus(points) {
  if (points === null || !points.every((p) => p.every(Number.isFinite))) {
    return { present: false, S: NaN };
  }
  const S = palmSize(points);
  if (!Number.isFinite(S) || S <= 0) return { present: false, S: NaN }; // degenerate
  return { present: true, S };
}

// Median of a small array of raw numbers, with the same odd-count
// convention as medianFlat (drop the oldest on an even count) -- used only
// for the minPalm gate's smoothed size, where medianFlat's per-coordinate
// machinery would be overkill for a single scalar per frame.
function medianScalar(values) {
  let sample = values;
  if (sample.length % 2 === 0) sample = sample.slice(1);
  return sample.slice().sort((a, b) => a - b)[(sample.length - 1) / 2];
}

// Max over the 21 landmarks of the z-down-weighted 3-D distance between two
// point sets, in raw (un-normalized) units -- callers combine this with the
// current frame's palmSize and absTol themselves, so this one helper serves
// the anchor comparison (stillness) and the committed-pose comparison
// (rearm) alike.
export function maxDisplacement(a, b) {
  return displacement(a, b, 0);
}

// The same distance, with the `skip` largest landmarks ignored: skip 0 is the
// maximum, skip 2 the third largest. See DEFAULTS for why a tutor wants that.
export function displacement(a, b, skip) {
  if (skip === 0) {
    let m = 0;
    for (let i = 0; i < a.length; i++) {
      const dx = a[i][0] - b[i][0], dy = a[i][1] - b[i][1], dz = (a[i][2] - b[i][2]) * Z_WEIGHT;
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (d > m) m = d;
    }
    return m;
  }
  const d = new Array(a.length);
  for (let i = 0; i < a.length; i++) {
    const dx = a[i][0] - b[i][0], dy = a[i][1] - b[i][1], dz = (a[i][2] - b[i][2]) * Z_WEIGHT;
    d[i] = Math.sqrt(dx * dx + dy * dy + dz * dz);
  }
  d.sort((x, y) => y - x);
  return d[Math.min(skip, d.length - 1)];
}

// Per finger, the second largest of its four landmarks' displacements; then the
// largest of the five. See DEFAULTS.stillPerFinger.
const FINGER_BASES = [1, 5, 9, 13, 17];
export function fingerDisplacement(a, b) {
  let m = 0;
  for (const base of FINGER_BASES) {
    let first = 0, second = 0;
    for (let i = base; i < base + 4; i++) {
      const dx = a[i][0] - b[i][0], dy = a[i][1] - b[i][1], dz = (a[i][2] - b[i][2]) * Z_WEIGHT;
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (d > first) { second = first; first = d; } else if (d > second) second = d;
    }
    if (second > m) m = second;
  }
  return m;
}

// The palm's centre: the mean of the four finger MCPs. Used to split a pose
// into where it is and what shape it is (see DEFAULTS.placeTol). NOT the
// wrist: it is the landmark that hops most on real video (the largest mover on
// 64% of in-hold frames), and one-fifth of every wrist hop would leak into
// every finger's shape term.
const PALM_CENTRE = [5, 9, 13, 17];
export function palmCentre(P) {
  const c = [0, 0, 0];
  for (const i of PALM_CENTRE) { c[0] += P[i][0]; c[1] += P[i][1]; c[2] += P[i][2]; }
  return [c[0] / PALM_CENTRE.length, c[1] / PALM_CENTRE.length, c[2] / PALM_CENTRE.length];
}
const recentred = (P, c) => P.map((p) => [p[0] - c[0], p[1] - c[1], p[2] - c[2]]);

// `C` turned and scaled, in the picture plane, to lie on `A` -- both already
// recentred on their palm centres. The turn and the scale are the least-squares
// ones for the PALM (wrist and four knuckles) and are then applied to every
// landmark, so the fingers are carried along and not fitted: a finger that has
// bent shows up as far from its old place as it did before. The wrist is in
// the fit although it hops (see palmCentre): without it the fit is to four
// knuckles almost in a line, and an angle read off a line one palm wide is
// noisier than one read off a triangle two palms tall.
const ALIGN_ON = [0, 5, 9, 13, 17];
export function alignedTo(A, C) {
  let dot = 0, cross = 0, aa = 0, cc = 0;
  for (const i of ALIGN_ON) {
    dot += C[i][0] * A[i][0] + C[i][1] * A[i][1];
    cross += C[i][0] * A[i][1] - C[i][1] * A[i][0];
    aa += A[i][0] * A[i][0] + A[i][1] * A[i][1];
    cc += C[i][0] * C[i][0] + C[i][1] * C[i][1];
  }
  if (!(cc > 0) || !(aa > 0)) return C;
  const k = Math.sqrt(aa / cc), th = Math.atan2(cross, dot), cs = k * Math.cos(th), sn = k * Math.sin(th);
  return C.map((p) => [cs * p[0] - sn * p[1], sn * p[0] + cs * p[1], k * p[2]]);
}

// How different two poses are AS SHAPES, in palm widths of the second: palm
// centres removed, then fingerDisplacement. "Is this the pose I already
// graded?" is this question, and both the held-over disarm below and the
// tutor's same-answer guard (experiments/asl-tutor/engine.js) ask it.
// Turned in the picture plane, it is still the same shape: the smaller of the
// plain and the aligned comparison (DEFAULTS.stillAlign says why the smaller).
export function shapeDistance(a, b) {
  const S = palmSize(b);
  if (!(S > 0)) return Infinity;
  const A = recentred(a, palmCentre(a)), B = recentred(b, palmCentre(b));
  return Math.min(fingerDisplacement(A, B), fingerDisplacement(A, alignedTo(A, B))) / S;
}
export function placeDistance(a, b) {
  const S = palmSize(b);
  if (!(S > 0)) return Infinity;
  const ca = palmCentre(a), cb = palmCentre(b);
  return Math.hypot(ca[0] - cb[0], ca[1] - cb[1]) / S;
}

export function createCommitter(opts = {}) {
  const cfg = { ...DEFAULTS, ...opts };
  validateOptions(cfg);

  // All mutable state lives here. reset() is the one place that defines
  // "empty," so the constructor and a mid-session reset can never disagree
  // about what a fresh committer looks like.
  let armed, committedPose, armRows, lastCommitT, lastHandT, lastPushT, stateTooSmall, lastResetReason, sizeRows;
  let holdOffUntil, carriedDisarm, heldOver;
  let inStillPeriod, periodStart, anchorRows, recentRows, history, totalFrames, handFrames;

  function clearPeriod() {
    inStillPeriod = false;
    periodStart = null;
    anchorRows = [];
    recentRows = [];
    history = [];
    totalFrames = 0;
    handFrames = 0;
  }

  function startPeriod(tMs, flat) {
    inStillPeriod = true;
    periodStart = tMs;
    anchorRows = [flat];
    recentRows = [flat];
    history = [flat];
    totalFrames = 1;
    handFrames = 1;
  }

  // The smoothed pose represented by a bounded row buffer, or null if the
  // buffer is still empty. `rows` may hold fewer than SMOOTH_N entries (the
  // "running median of what exists" while a period/rearm window is still
  // filling up) -- medianFlat already handles any count gracefully.
  function smoothedPose(rows, aspect) {
    if (rows.length === 0) return null;
    return pointsOf(medianFlat(rows), aspect);
  }

  // Purely a function of (armed, inStillPeriod, periodStart, lastPushT): it
  // is never stored, so push()'s return value and the `state` getter cannot
  // drift out of sync with each other or with ruling 4 ("while disarmed,
  // progress is 0").
  function currentProgress() {
    if (!armed || !inStillPeriod) return 0;
    const elapsed = lastPushT - periodStart;
    if (elapsed < cfg.settleMs) return 0;
    return Math.min(1, Math.max(0, (elapsed - cfg.settleMs) / (cfg.holdMs - cfg.settleMs)));
  }

  function reset() {
    armed = true;
    committedPose = null;
    armRows = [];
    lastCommitT = null;
    lastHandT = null;
    stateTooSmall = false;
    lastResetReason = null;
    sizeRows = [];
    clearPeriod();
    // Fix round 2, I2: a bare reset() ALWAYS permits the next push to carry
    // any tMs (including one earlier than what came before). A new trial's
    // clock genuinely restarts near 0 (js/core/recorder.js's tMs is
    // "milliseconds since THIS TRIAL started"), and the tutor's natural call
    // at a trial boundary is the plain reset() -- so the guard has to give
    // way here, not just when a caller remembers to say so explicitly.
    lastPushT = null;
    holdOffUntil = null;
    carriedDisarm = false;
    heldOver = false;
  }
  reset();

  /* A new trial on the same learner, as opposed to reset()'s blank slate.
   *
   * The caller's clock restarts (so the time guard is cleared, as in
   * reset()), but a DISARMED committer stays disarmed against the pose it
   * last committed. Without this the tutor graded one held pose against
   * letter after letter: a trial ended on a commit, the next trial reset()
   * re-armed the committer, and the hand -- still up, still in the last
   * letter's shape -- committed again 600 ms later against the NEW letter,
   * was rejected three times in a row and the trial after that did the same
   * (first camera session, 2026-09-20: "it just bursts through a bunch of
   * letters"). The learner now has to let go of the old shape (move it by
   * rearmTol, or take the hand away) before anything new can be committed,
   * which is what a person does between two signs anyway.
   *
   * The dropout clock cannot cross the boundary (the old trial's tMs means
   * nothing on the new clock), so the first frame of the new trial re-bases
   * it: the hand counts as having been there up to that frame, whether or not
   * the tracker happened to see it on that one, and only a real absence of
   * DROPOUT_MS from then on re-arms. */
  function nextTrial() {
    lastPushT = null;
    lastCommitT = null;
    holdOffUntil = null;
    lastResetReason = null;
    clearPeriod();
    if (committedPose !== null) {
      // Whatever `armed` says. A committer that re-armed during the last
      // trial's closing seconds (a timed re-arm landing on the frame a dwell
      // ended did exactly this) would otherwise walk into the new trial armed,
      // under the same unmoved hand. If the hand really has left that pose,
      // the first frames of the new trial re-arm it at once.
      armed = false;
      armRows = [];
      // Said out loud, because otherwise nothing is: a disarmed committer
      // shows an empty ring and has no reset to report, so a learner who kept
      // their hand up would sit in front of a tutor doing nothing at all.
      lastResetReason = "held-over";
      heldOver = true;   // stricter re-arm, and no rearmAfterMs: see DEFAULTS
    }
    carriedDisarm = true;
    lastHandT = null;
  }

  /* No hold may run before `untilT` (on the caller's clock). The tutor uses it
   * for the moments a learner is READING -- a new letter, a hint -- when a
   * hand that happens to be still is not an answer. Rearm detection carries
   * on underneath, so moving the hand during a hold-off still counts. */
  function holdOff(untilT) {
    if (typeof untilT !== "number" || !Number.isFinite(untilT)) {
      throw new Error(`commit.holdOff: untilT must be a finite number, got ${JSON.stringify(untilT)}`);
    }
    holdOffUntil = untilT;
    clearPeriod();
  }

  function push(tMs, flat, aspect) {
    if (typeof tMs !== "number" || !Number.isFinite(tMs)) {
      throw new Error(`commit.push: tMs must be a finite number, got ${JSON.stringify(tMs)}`);
    }
    if (lastPushT !== null && tMs < lastPushT) {
      throw new Error(`commit.push: tMs went backwards (${tMs} < ${lastPushT}) -- time comes only from the caller, so this is a replay bug, not a clock glitch (call reset() if the caller's own clock is restarting)`);
    }
    lastPushT = tMs;

    const points = flat === null ? null : pointsOf(flat, aspect);
    const { present: hasHand, S: rawS } = frameStatus(points);
    // Fix round 3: the minPalm gate reads a SMOOTHED size (median of the
    // last SMOOTH_N usable frames' raw palmSize), not this one frame's, and
    // applies hysteresis rather than a single threshold -- see the header
    // for why. The size buffer is independent of the position buffers below
    // (anchorRows/recentRows/history): it tracks "how big does the hand
    // look right now" across period boundaries, gaps, and rearms alike, and
    // only clears on a frame with no hand at all or on reset().
    if (hasHand) {
      sizeRows.push(rawS);
      if (sizeRows.length > SMOOTH_N) sizeRows.shift();
    } else {
      sizeRows = [];
    }
    if (hasHand) {
      const smoothedS = medianScalar(sizeRows);
      if (!stateTooSmall && smoothedS < cfg.minPalm) stateTooSmall = true;
      else if (stateTooSmall && smoothedS > cfg.minPalm * MIN_PALM_EXIT_FACTOR) stateTooSmall = false;
      // else: within the hysteresis band, or not enough history yet -- hold
      // the current classification rather than guess from thin evidence.
    } else {
      stateTooSmall = false; // no hand at all is not "present but small"
    }
    const tooSmall = stateTooSmall;
    const measurable = hasHand && !tooSmall;
    // How long, up to and including this frame, the hand has been absent --
    // computed from the PREVIOUS sighting, before it is updated below, so it
    // is well-defined both on a still-missing frame and on the frame the
    // hand reappears after a gap. A too-small hand still counts as present
    // here: the hand IS there, it is just too small to measure (C1).
    if (carriedDisarm) {
      // First frame after nextTrial(): re-base the dropout clock (see there).
      // UNCONDITIONALLY. Re-basing only when this frame has a hand meant one
      // dropped tracker frame at the boundary read as "gone for ever", re-armed
      // the committer at once and brought the burst straight back -- and real
      // tracking drops about one frame in eight. Absence now has to LAST
      // DROPOUT_MS on the new clock, like any other dropout.
      carriedDisarm = false;
      lastHandT = tMs;
    }
    const gap = lastHandT === null ? Infinity : tMs - lastHandT;
    if (hasHand) lastHandT = tMs;
    const heldOff = holdOffUntil !== null && tMs < holdOffUntil;

    if (!armed) {
      // Rearm (ruling 4): either the hand moved away from the COMMITTED pose
      // by more than rearmTol palm sizes, or it has simply been gone for
      // long enough that "still the same attempt" is no longer a safe
      // assumption. Either is sufficient on its own. The comparison uses the
      // same smoothed/floored measure as stillness (fix round 1, ruling 4),
      // and only ever from MEASURABLE rows -- a too-small frame cannot
      // reliably say whether the hand moved either.
      if (measurable) {
        armRows.push(flat);
        if (armRows.length > SMOOTH_N) armRows.shift();
      }
      const movedAway = measurable && (() => {
        const curPts = smoothedPose(armRows, aspect);
        if (heldOver) {
          return shapeDistance(committedPose, curPts) > cfg.heldOverShape
              || placeDistance(committedPose, curPts) > cfg.heldOverPlace;
        }
        const S = palmSize(curPts);
        const tol = Math.max(cfg.rearmTol * S, cfg.absTol);
        return displacement(committedPose, curPts, cfg.rearmSkip) > tol;
      })();
      // Not while held off: a decided trial's dwell is a hold-off that lasts to
      // the end of the trial, and re-arming inside it would hand nextTrial()
      // an ARMED committer -- the burst again, five seconds late.
      const waitedOut = !heldOver && !heldOff && lastCommitT !== null && tMs - lastCommitT >= cfg.rearmAfterMs;
      if (movedAway || gap >= DROPOUT_MS || waitedOut) {
        armed = true;
        armRows = [];
        heldOver = false;
        if (lastResetReason === "held-over") lastResetReason = null;
      }
    }

    if (heldOff) {
      clearPeriod();
    } else if (armed && !inStillPeriod) {
      // "The still period starts at the first frame with a hand after a
      // reset" (ruling 2) -- including the reset that is disarm -> rearm.
      // Only a MEASURABLE frame can seed a period (C1).
      if (measurable) startPeriod(tMs, flat);
    } else if (armed && inStillPeriod) {
      if (gap > DROPOUT_MS) {
        // The dropout itself, not the eventual reappearance, is what breaks
        // the hold; if the hand is already back (and measurable) this same
        // frame the new period starts here rather than staying broken an
        // extra tick.
        clearPeriod();
        if (measurable) startPeriod(tMs, flat);
        lastResetReason = "dropout";
      } else if (tooSmall) {
        // C1: the hand is present but below minPalm -- stillness is not
        // measurable at this size, so the honest behavior is to refuse
        // rather than guess. The period ends outright; it cannot restart on
        // this same frame the way a dropout or a moved-anchor reset can,
        // because this frame itself is not a usable anchor.
        clearPeriod();
        lastResetReason = "too-small";
      } else if (measurable) {
        // anchorRows fills once (first SMOOTH_N usable rows of the period)
        // and then stays fixed -- it must not track a slow drift, or a
        // steady drift would never look like movement to itself. recentRows
        // is a rolling window of the last SMOOTH_N usable rows.
        if (anchorRows.length < SMOOTH_N) anchorRows.push(flat);
        recentRows.push(flat);
        if (recentRows.length > SMOOTH_N) recentRows.shift();
        const anchorPts = smoothedPose(anchorRows, aspect);
        const curPts = smoothedPose(recentRows, aspect);
        const S = palmSize(curPts);
        const tol = Math.max(cfg.stillTol * S, cfg.absTol);
        const stat = cfg.stillPerFinger ? fingerDisplacement : (a, b) => displacement(a, b, cfg.stillSkip);
        let movedTooFar;
        if (cfg.placeTol === Infinity && !cfg.stillAlign) {
          movedTooFar = stat(anchorPts, curPts) >= tol;
        } else {
          const ca = palmCentre(anchorPts), cc = palmCentre(curPts);
          const place = Math.hypot(ca[0] - cc[0], ca[1] - cc[1]);   // in the picture plane; z is noise here
          const A = recentred(anchorPts, ca), C = recentred(curPts, cc);
          let shape = stat(A, C);
          if (cfg.stillAlign) shape = Math.min(shape, stat(A, alignedTo(A, C)));
          movedTooFar = shape >= tol || (cfg.placeTol !== Infinity && place >= Math.max(cfg.placeTol * S, cfg.absTol));
        }
        if (movedTooFar) {
          // Moved too far from the anchor: the hold restarts here, and its
          // history is discarded -- a mid-formation frame from the broken
          // hold must never survive into a later commit's median (ruling 3).
          clearPeriod();
          startPeriod(tMs, flat);
          lastResetReason = "moved";
        } else {
          // Still within tolerance: the hold continues. `history` is a
          // sliding window of the last `window` pushed rows of THIS period,
          // present or not -- medianFlat skips the not-present ones on its
          // own, and is separate from the SMOOTH_N buffers above (ruling 6:
          // bounded, but a different bound for a different purpose).
          totalFrames++;
          history.push(flat);
          if (history.length > cfg.window) history.shift();
          handFrames++;
        }
      } else {
        // No hand at all, but not a long-enough dropout: still counts
        // toward the period's frame/presence tally and its history, just
        // not toward handFrames or the smoothing buffers.
        totalFrames++;
        history.push(flat);
        if (history.length > cfg.window) history.shift();
      }
    }

    const progress = currentProgress();
    let committed = null;
    if (inStillPeriod && progress >= 1) {
      if (handFrames / totalFrames >= cfg.presentFrac) {
        const medianRow = medianFlat(history);
        if (medianRow !== null) {
          committed = { tStart: periodStart, tCommit: tMs, flat: medianRow, handPresentFrac: handFrames / totalFrames, frames: totalFrames };
          committedPose = pointsOf(medianRow, aspect);
          lastCommitT = tMs;
          armed = false;
          armRows = [];
          clearPeriod();
          lastResetReason = null;
        }
      } else {
        // Fix round 2, I3: the hold time was reached, but tracking was too
        // spotty along the way to trust (e.g. every third frame dropped --
        // each individually well under the 300ms dropout threshold, so
        // nothing above ever reset the period, yet the hand was only
        // present ~67% of the time). Without this, `progress` would sit at
        // 1 forever -- the ring reading "done" while nothing ever commits.
        // Reset honestly instead of latching; restart from this frame if it
        // is itself measurable.
        clearPeriod();
        if (measurable) startPeriod(tMs, flat);
        lastResetReason = "unsteady-tracking";
      }
    }

    return { progress, committed };
  }

  return {
    push,
    reset,
    nextTrial,
    holdOff,
    // A plain, JSON-safe snapshot -- the tutor logs this directly (ruling 5).
    get state() {
      return { armed, inStillPeriod, periodStart, progress: currentProgress(), lastCommitT, tooSmall: stateTooSmall, lastResetReason, holdOffUntil };
    },
  };
}
