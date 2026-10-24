/* verifier.js: grades one hand against a KNOWN target letter, in the
 * browser, every frame.
 *
 * The target is known on every trial, so this is VERIFICATION, not
 * recognition: "is this an acceptable R?", answered with accept, reject, or
 * abstain. Abstain is a first-class answer, not a failure -- a tutor that
 * must produce a verdict will produce a wrong one, and a wrong verdict on a
 * hand the camera barely saw is worse for a learner than silence.
 *
 * TWO NUMBERS, DOING DIFFERENT JOBS (training/gaussian_model.py has the long
 * version). `score` is RELATIVE: the target's log-likelihood minus the best
 * of every other class, the 23 other letters and the "_background" class
 * alike. It answers "more like the target than like anything else I know?"
 * and it cannot reject a hand that is unlike every letter, because such a
 * hand is still relatively most like one of them -- the spike measured an
 * ungated score accepting 100% of non-letter hands as some letter. `d2` is
 * ABSOLUTE: squared Mahalanobis distance from the target's own Gaussian,
 * i.e. "a typical instance of this letter at all?". That is what rejects
 * garbage, and every accept is gated on it.
 *
 * THE ORDER OF THE CHECKS IN `verify`, AND WHY EACH IS WHERE IT IS:
 *
 *   1. a non-finite feature -> abstain "not-gradable". features.js reports
 *      NaN when the hand was degenerate (tracking dropped out), so the hand
 *      was never measured and every number below would be built on nothing.
 *      It is first because it is the only failure that makes the others
 *      meaningless -- including the tier, which would otherwise log a score
 *      for a hand nobody saw.
 *   2. standardize -- the Gaussians live in a fixed mean/sd space, so a raw
 *      angle in degrees and a palm-width-normalized distance are comparable.
 *   3. tier 2 -> abstain "tier2", score and distance still computed and
 *      returned for logging. A tier-2 letter did not clear even the lenient
 *      development bar (training/MODEL_REPORT.md), so it is taught and shown
 *      but never given a verdict. It comes BEFORE the tracking check because
 *      "fix your camera" would be a lie about a letter that gets no verdict
 *      however well it was tracked.
 *   4. tracking quality -> abstain "tracking". The hand may be perfect; we
 *      could not see it well enough to say so.
 *   5. decide -- the gate, then the two thresholds, then the band between
 *      them where the evidence supports nothing.
 *
 * `decide` is exported separately and does step 5 (plus 1 and 3) on an
 * ALREADY-STANDARDIZED vector, with no quality checks at all: the block
 * blame that comes next re-decides on modified standardized vectors and
 * must reach exactly the same verdict machinery, not a second copy of it.
 * Both it and `verify` take an options object whose one option,
 * `ignoreTier` (strictly `true`, nothing truthy), skips step 3 for callers
 * that need the verdict the tier hides -- never for anything a learner
 * sees. Any result it changed is marked `tierIgnored: true`, and every
 * other result carries `tierIgnored: false`, so a consumer or a log can
 * always tell a tier-blind decision from a real one without knowing how it
 * was called. See `decide` below.
 *
 * PINNED TO PYTHON. training/decide.py carries the same rules, branch for
 * branch, and tests/fixtures/verifier-parity.json holds real hands scored by
 * it; tests/verifier.test.js demands agreement to 1e-6 and refuses to run at
 * all if the fixture was generated from a different tutor/model.json.
 *
 * HOW CLOSE "AGREE" IS. Both sides read the same written JSON numbers (8
 * significant digits) and `standardize` is the same two IEEE operations in
 * the same order on both, so the standardized vector is identical bit for
 * bit. The SCORE is not: numpy accumulates its sums in a different order
 * from the fused row loop here, and measured over the fixture and a sweep of
 * model variants the two differ by up to about 1e-10 in score and distance.
 * That is seven orders of magnitude inside the 1e-6 the parity test demands,
 * and it means exactly one thing for behavior: a hand whose score sits
 * within ~1e-10 of a threshold can fall on different sides of it in the two
 * implementations. Nothing rides on such a hand -- the thresholds are
 * bootstrap numbers with three significant digits of meaning, and a score
 * that close to a threshold is a coin toss about a borderline hand either
 * way -- but "bit for bit" would be a false claim and this is the true one.
 *
 * NOTHING HERE IS CALIBRATED. Every threshold is read from the model file,
 * never written down here, and those thresholds are BOOTSTRAP: fitted on
 * public datasets of posed hands, cross-source rather than cross-signer,
 * with no learner of this tutor contributing a single hand.
 */
import { FEATURE_NAMES, FEATURE_BLOCKS } from "./features.js";
import { quadForm, cholesky } from "./linalg.js";

const F = FEATURE_NAMES.length;
const SCHEMA = 1;

// A-Z less the two motion letters (J, Z), which have no static handshape to
// verify. A model that grades a different number of letters is not this one.
const STATIC_LETTER_COUNT = 24;

// The class of non-letter hands. It can beat the target and so appear as a
// rival, but it is never a target and has no thresholds of its own.
export const BACKGROUND = "_background";

// The rival reported when the hand is past the target's gate and is not a
// plausible instance of any letter either: it is not "a bad R that looks
// like a U", it is a hand the model has no name for. Deliberately not a
// letter, so a caller cannot mistake it for one.
export const ATYPICAL = "_atypical";

/* TRACKING QUALITY. These are properties of the CAMERA VIEW, not of the
 * handshape, and they are why a verdict is withheld rather than a reason to
 * reject a hand. They are here rather than in the model file because they
 * describe this tutor's capture setup (one webcam, one hand, hand held up in
 * frame), not the letters.
 *
 * handPresentFrac: the fraction of the scoring window in which a hand was
 * tracked at all. Below this, the "hand" being graded is mostly interpolated
 * across frames where there wasn't one.
 * palmSizePx: palm width in pixels. Landmarks from a hand this small in
 * frame carry jitter comparable to the differences between M, N and S. */
export const MIN_HAND_PRESENT_FRAC = 0.8;
export const MIN_PALM_SIZE_PX = 40;

/* Two entries of a precision matrix that should mirror each other may differ
 * by the last bits of the 8 significant digits the model file holds. More
 * than this relative difference is a corrupted file, not rounding, and a
 * Mahalanobis distance from an asymmetric precision is not a distance. */
const PREC_SYMMETRY_TOLERANCE = 1e-9;

/* A squared distance cannot be negative. In exact arithmetic v' P v > 0 for
 * a positive-definite P, but a hand sitting almost exactly on a mean can
 * come out a hair under zero through cancellation, and that is rounding, not
 * a broken model: it is clamped to zero. Anything more negative than this is
 * a precision matrix that is not what it claims to be, and no verdict comes
 * out of it. */
const D2_NEGATIVE_TOLERANCE = 1e-9;

// --------------------------------------------------------------- loading

const fail = (msg) => { throw new Error(`loadModel: ${msg}`); };
const isFiniteNumber = (v) => typeof v === "number" && Number.isFinite(v);

// What the bad value WAS, in an error message. Not JSON.stringify alone:
// it turns NaN and Infinity -- the two most likely corruptions here -- into
// "null", which would send a reader looking for the wrong thing.
const show = (v) => (typeof v === "number" ? String(v) : JSON.stringify(v) ?? String(v));

function checkVector(v, len, who) {
  if (!Array.isArray(v) || v.length !== len) fail(`${who} is not an array of ${len} numbers (got ${Array.isArray(v) ? `${v.length} entries` : typeof v})`);
  for (let i = 0; i < len; i++) if (!isFiniteNumber(v[i])) fail(`${who}[${i}] is ${show(v[i])}, not a finite number`);
  return Float64Array.from(v);
}

function checkPrecision(rows, name) {
  if (!Array.isArray(rows) || rows.length !== F) fail(`class ${name}: prec is not ${F}x${F} (got ${Array.isArray(rows) ? `${rows.length} rows` : typeof rows})`);
  const out = [];
  for (let i = 0; i < F; i++) {
    const row = rows[i];
    if (!Array.isArray(row) || row.length !== F) fail(`class ${name}: prec[${i}] is not a row of ${F} numbers (got ${Array.isArray(row) ? `${row.length} entries` : typeof row})`);
    for (let j = 0; j < F; j++) if (!isFiniteNumber(row[j])) fail(`class ${name}: prec[${i}][${j}] is ${show(row[j])}, not a finite number`);
    out.push(Float64Array.from(row));
  }
  for (let i = 0; i < F; i++) {
    for (let j = i + 1; j < F; j++) {
      const a = out[i][j], b = out[j][i];
      const scale = Math.max(Math.abs(a), Math.abs(b));
      if (Math.abs(a - b) > PREC_SYMMETRY_TOLERANCE * scale) {
        fail(`class ${name}: prec[${i}][${j}] is ${a} but prec[${j}][${i}] is ${b} -- a precision matrix must be symmetric`);
      }
    }
  }
  return out;
}

/* Turn the parsed tutor/model.json into the object every function here
 * takes. It does NOT fetch: reading the file is the caller's job, so this
 * module never touches the network and can be tested with a model built in
 * a test.
 *
 * Everything it can check, it checks, and every error names what is wrong.
 * A model file is generated by training/build_model.py and then shipped as
 * a static asset; the failure mode this guards against is not a malicious
 * one, it is a half-written or half-updated file producing verdicts that
 * look like numbers. */
export function loadModel(json) {
  if (!json || typeof json !== "object") fail(`expected the parsed contents of tutor/model.json, got ${json === null ? "null" : typeof json}`);
  if (json.schema !== SCHEMA) fail(`schema is ${show(json.schema)}, but this verifier reads schema ${SCHEMA}`);

  for (const [key, want] of [["featureNames", FEATURE_NAMES], ["blocks", FEATURE_BLOCKS]]) {
    const got = json[key];
    if (!Array.isArray(got) || got.length !== want.length) {
      fail(`${key} is not an array of ${want.length} names (got ${Array.isArray(got) ? `${got.length} entries` : typeof got}) -- the model was trained against a different features.js`);
    }
    for (let i = 0; i < want.length; i++) {
      if (got[i] !== want[i]) fail(`${key}[${i}] is ${show(got[i])} but this build's feature ${i} is ${show(want[i])} -- the model was trained against a different features.js`);
    }
  }

  if (!json.scale || typeof json.scale !== "object") fail("scale is missing");
  const mean = checkVector(json.scale.mean, F, "scale.mean");
  const sd = checkVector(json.scale.sd, F, "scale.sd");
  for (let i = 0; i < F; i++) {
    if (!(sd[i] > 0)) fail(`scale.sd[${i}] is ${sd[i]}; standardization divides by it, so every sd must be positive`);
  }

  const letters = json.letters;
  if (!Array.isArray(letters) || letters.length !== STATIC_LETTER_COUNT) {
    fail(`letters is not an array of the ${STATIC_LETTER_COUNT} static letters (got ${Array.isArray(letters) ? `${letters.length} entries` : typeof letters})`);
  }
  if (new Set(letters).size !== letters.length) fail("letters contains a duplicate");
  if (!json.classes || typeof json.classes !== "object") fail("classes is missing");

  const known = new Set([...letters, BACKGROUND]);
  for (const name of Object.keys(json.classes)) {
    if (!known.has(name)) fail(`class ${JSON.stringify(name)} is neither one of the ${STATIC_LETTER_COUNT} letters nor ${JSON.stringify(BACKGROUND)}`);
  }
  if (!json.classes[BACKGROUND]) fail(`there is no ${JSON.stringify(BACKGROUND)} class -- without it the score has nothing to reject a non-letter hand with`);

  const thresholds = Object.create(null);
  const tiers = Object.create(null);
  for (const L of letters) {
    if (!json.classes[L]) fail(`letter ${L}: no class in the model`);
    const t = json.thresholds?.[L];
    if (!t || typeof t !== "object") fail(`letter ${L}: no thresholds in the model`);
    for (const key of ["accept", "reject", "gate"]) {
      if (!isFiniteNumber(t[key])) fail(`letter ${L}: threshold ${show(key)} is ${show(t[key])}, not a finite number`);
    }
    const tier = json.tiers?.[L];
    if (tier !== 1 && tier !== 2) fail(`letter ${L}: tier is ${show(tier)}, expected 1 or 2`);
    // Frozen: a threshold is a fitted number from the model file, not a knob
    // a later module gets to nudge at runtime.
    thresholds[L] = Object.freeze({ accept: t.accept, reject: t.reject, gate: t.gate });
    tiers[L] = tier;
  }

  // Sorted so that a tie between two equally likely rivals is broken the
  // same way here and in Python (which scores columns in sorted order).
  const names = Object.keys(json.classes).sort();
  const classes = Object.create(null);
  for (const name of names) {
    const c = json.classes[name];
    if (!c || typeof c !== "object") fail(`class ${name}: not an object`);
    if (!isFiniteNumber(c.logdet)) fail(`class ${name}: logdet is ${show(c.logdet)}, not a finite number`);
    const prec = checkPrecision(c.prec, name);
    /* The right shape, finite and symmetric is not enough to be a precision
     * matrix, and the two ways it can fail are both silent. All zeros -- a
     * plausible half-written file -- makes d2 zero for every letter at once,
     * so every hand is a perfect B and a perfect S simultaneously. A sign
     * flip makes d2 negative, so `d2 > gate` can never fire and the
     * typicality gate is gone without anything looking wrong. Factoring is
     * the cheap honest test, it runs once at load, and it takes well under a
     * millisecond for all 25 classes. */
    try {
      cholesky(prec, `class ${name}: prec`);
    } catch (e) {
      fail(e.message);
    }
    classes[name] = { mu: checkVector(c.mu, F, `class ${name}: mu`), prec, logdet: c.logdet, n: c.n };
  }

  return Object.freeze({
    schema: json.schema,
    provenance: json.provenance,
    confusions: json.confusions,
    letters: Object.freeze([...letters]),
    names: Object.freeze(names),
    classes,
    thresholds,
    tiers,
    mean,
    sd,
    // Everything a per-frame call would otherwise allocate or look up by
    // name. verify runs on every camera frame on a laptop, so the only
    // object it creates is the result it hands back.
    order: names.map((n) => classes[n]),
    index: names.reduce((acc, n, i) => { acc[n] = i; return acc; }, Object.create(null)),
    work: { z: new Float64Array(F), diff: new Float64Array(F), d2: new Float64Array(names.length), ll: new Float64Array(names.length) },
  });
}

// --------------------------------------------------------------- scoring

/* Raw 46-feature vector -> the standardized space the Gaussians live in.
 * `out` lets the per-frame path reuse a buffer; without it a fresh one is
 * returned, because a caller that keeps the answer must not have it
 * overwritten by the next frame.
 *
 * A division, not a multiply by a precomputed reciprocal: 1/sd rounds, and
 * training/decide.py divides. The two have to agree bit for bit. */
export function standardize(model, x46, out) {
  if (x46.length !== F) throw new Error(`standardize: expected ${F} features, got ${x46.length}`);
  const z = out ?? new Float64Array(F);
  for (let i = 0; i < F; i++) z[i] = (x46[i] - model.mean[i]) / model.sd[i];
  return z;
}

/* d2 and log-likelihood for every class, into caller-owned buffers, in
 * model.names order. ll drops the shared -0.5 * F * log(2 pi): it is
 * identical for every class, so it cancels in the score and no threshold
 * here can see it (this is exactly what Python's log_lik does).
 *
 * A non-finite z yields d2 +Infinity and ll -Infinity -- gaussian_model.py's
 * convention for a row it cannot score -- rather than NaN, which would
 * compare false against everything and quietly fall through to a verdict. */
function scoreInto(model, z, d2, ll) {
  const diff = model.work.diff;
  let finite = true;
  for (let i = 0; i < F; i++) if (!Number.isFinite(z[i])) { finite = false; break; }
  if (!finite) {
    d2.fill(Infinity);
    ll.fill(-Infinity);
    return;
  }
  const order = model.order;
  for (let c = 0; c < order.length; c++) {
    const cls = order[c];
    const mu = cls.mu;
    for (let i = 0; i < F; i++) diff[i] = z[i] - mu[i];
    // The difference is formed first rather than expanding the quadratic
    // into z'Pz - 2z'Pmu + mu'Pmu with a precomputed P mu: the expanded
    // form subtracts two large nearly equal numbers and loses the low bits
    // of a small distance, which is precisely the range a typicality gate
    // lives in -- and it would break parity with numpy.
    const d = quadForm(cls.prec, diff);
    // A hair below zero on a hand sitting almost on the mean is
    // cancellation; anything more negative is left alone so that `decide`
    // can refuse to grade it.
    d2[c] = d < 0 && d > -D2_NEGATIVE_TOLERANCE ? 0 : d;
    ll[c] = -0.5 * (d2[c] + cls.logdet);
  }
}

/* Every class's log-likelihood and typicality distance, as plain objects
 * keyed by class name. For logging, debugging and the research export; the
 * per-frame path does not use it, because it allocates. */
export function scoreAll(model, z) {
  if (z.length !== F) throw new Error(`scoreAll: expected ${F} standardized features, got ${z.length}`);
  const { d2: d2buf, ll: llbuf } = model.work;
  scoreInto(model, z, d2buf, llbuf);
  const ll = {}, d2 = {};
  model.names.forEach((name, i) => { ll[name] = llbuf[i]; d2[name] = d2buf[i]; });
  return { ll, d2 };
}

// --------------------------------------------------------------- deciding

function thresholdsFor(model, target) {
  const t = typeof target === "string" ? model.thresholds[target] : undefined;
  if (!t) throw new Error(`${JSON.stringify(target)} is not a graded letter; this model grades ${model.letters.join("")}`);
  return t;
}

/* Every result has the same seven-and-a-bit fields, always, so a consumer
 * never has to test whether a key is there. `tierIgnored` defaults to false
 * because the paths that do not pass it -- a hand that was never scored,
 * arithmetic with no answer -- come back the same whatever the tier says,
 * so the bypass cannot have changed them. */
const result = (outcome, reason, score, d2, gate, rival, tier, tierIgnored = false) =>
  ({ outcome, reason, score, d2, gate, rival, tier, tierIgnored });

/* The verdict on one ALREADY-STANDARDIZED vector, with no tracking quality
 * considered -- a hand can be perfectly tracked and still not be an R, and
 * the block-blame module re-decides on vectors that never came from a
 * camera frame at all. See the header for the order of the rules. */
export function decide(model, z, target, opts) {
  const th = thresholdsFor(model, target);
  const gate = th.gate;
  const tier = model.tiers[target];
  if (z.length !== F) throw new Error(`decide: expected ${F} standardized features, got ${z.length}`);

  // The INPUT is what decides gradability, not the distance that comes out
  // of it: a finite vector whose distance overflowed to Infinity is an
  // infinitely atypical hand, which is a reject, and Python reads it the
  // same way (training/decide.py tests np.isfinite(z), not the distance).
  for (let i = 0; i < F; i++) {
    // Nothing was measured, so nothing is claimed.
    if (!Number.isFinite(z[i])) return result("abstain", "not-gradable", -Infinity, Infinity, gate, null, tier);
  }

  const { d2: d2buf, ll: llbuf } = model.work;
  scoreInto(model, z, d2buf, llbuf);
  const t = model.index[target];

  // The best of every OTHER class: the 23 other letters and the background
  // alike. Strictly greater, so a tie goes to the first in names order --
  // the same tie-break Python's argmax makes.
  // opts.excludeRivals: classes that cannot be the rival for this hand (tutor/mix.js: P, when the middle finger
  // points up). Absent, nothing changes.
  const excluded = opts?.excludeRivals ?? null;
  let b = -1, impossible = false;
  for (let c = 0; c < llbuf.length; c++) {
    if (d2buf[c] < 0) impossible = true;
    if (c === t) continue;
    if (excluded !== null && excluded.includes(model.names[c])) continue;
    if (b < 0 || llbuf[c] > llbuf[b]) b = c;
  }
  const best = model.names[b];
  const score = llbuf[t] - llbuf[b];
  const d2 = d2buf[t];

  /* A score or a distance that is not a finite number is a computation that
   * blew up, not evidence: a hand far enough from every class overflows to
   * d2 = Infinity and a score of Infinity - Infinity = NaN, and NaN compares
   * false against every threshold, so it would fall through to the last
   * branch and come back as a confident "reject, rival _atypical" or a
   * "borderline" that is a lie about arithmetic with no answer. A negative
   * squared distance anywhere means a precision matrix is not one. Either
   * way the honest answer is the same as for a hand that was never seen.
   * (A score of +Infinity -- every rival overflowed while the target did
   * not -- is refused too: it is the same untrustworthy arithmetic, and
   * silence is the conservative direction.) */
  if (impossible || !Number.isFinite(score) || !Number.isFinite(d2)) {
    return result("abstain", "not-gradable", -Infinity, Infinity, gate, null, tier);
  }

  let outcome, reason = null, rival = null;
  if (d2 > gate) {
    // Atypical: rejected whatever the relative score says. The rival then
    // answers a different question -- is this hand a plausible instance of
    // some other LETTER (inside that letter's own gate), or of nothing?
    const rivalThresholds = model.thresholds[best];
    outcome = "reject";
    rival = rivalThresholds !== undefined && d2buf[b] <= rivalThresholds.gate ? best : ATYPICAL;
  } else if (score >= th.accept) {
    outcome = "accept";
  } else if (score < th.reject) {
    outcome = "reject";
    rival = best;
  } else {
    // Reported, not blamed: `best` is what held the score down, which is
    // worth logging even though no verdict is given.
    outcome = "abstain";
    reason = "borderline";
    rival = best;
  }

  /* opts.ignoreTier: decide a tier-2 letter exactly as if it were tier 1.
   * The block-blame module needs it -- it asks "would fixing this block have
   * changed the verdict?", and on a tier-2 letter a tier-aware decide can
   * only ever answer "abstain", for every block, which is no answer at all.
   * The result still carries `tier`: the caller is opting out of the
   * downgrade, not out of knowing about it. Nothing user-facing may use
   * this; a tier-2 letter gets no verdict shown to a learner.
   *
   * STRICTLY true, and training/decide.py tests `is True` for the same
   * reason: a truthy string or 1 arriving from a config or a query
   * parameter must not be able to turn a tier-2 letter into a graded one.
   *
   * `tierIgnored` on the result is the structural half of that rule. It is
   * true only where the bypass ACTUALLY changed the answer -- a tier-2
   * target with the flag strictly set -- and false for a tier-1 target even
   * if the flag was passed, so filtering on it gives exactly the verdicts
   * the tier would otherwise have hidden. A comment saying "do not show
   * these to a learner" is prose; this is the thing a consumer or a log
   * line can test. */
  const tierIgnored = tier === 2 && opts?.ignoreTier === true;
  if (tier === 2 && !tierIgnored) {
    outcome = "abstain";
    reason = "tier2";
  }
  return result(outcome, reason, score, d2, gate, rival, tier, tierIgnored);
}

/* quality: { handPresentFrac, palmSizePx, centered, chiralityMargin }.
 *
 * chiralityMargin (how confidently the hand was read as left or right) is
 * accepted and meant to be LOGGED by callers, but it decides nothing in
 * this slice: features are computed on a canonicalized right hand, and what
 * a weak chirality reading should do to a verdict is not something this
 * model was evaluated on. It is here so the number is carried alongside the
 * ones that do decide, not so it quietly starts mattering. */
function qualityIsPoor(quality) {
  return quality.handPresentFrac < MIN_HAND_PRESENT_FRAC || quality.palmSizePx < MIN_PALM_SIZE_PX || !quality.centered;
}

/* The two reasons `decide` can reach that outrank a tracking complaint.
 * "tier2" because the letter gets no verdict however well it was tracked,
 * and "not-gradable" because a feature that looked finite can still
 * standardize to an infinity (a tiny sd) -- and "fix your camera" would be
 * the wrong thing to say about a hand that was never scored. "borderline"
 * is deliberately NOT here: when the tracking was poor as well, the
 * actionable reason is the tracking. */
const OUTRANKS_TRACKING = new Set(["tier2", "not-gradable"]);

// One hand, one target letter, one verdict. See the header for the order of
// the checks and the reason for each.
export function verify(model, x46, target, quality, opts) {
  const th = thresholdsFor(model, target);
  if (!quality || typeof quality !== "object" || !Number.isFinite(quality.handPresentFrac)
      || !Number.isFinite(quality.palmSizePx) || typeof quality.centered !== "boolean") {
    throw new Error("verify: quality must be { handPresentFrac, palmSizePx, centered, chiralityMargin } -- a missing tracking quality must not read as a good one");
  }
  if (x46.length !== F) throw new Error(`verify: expected ${F} features, got ${x46.length}`);

  for (let i = 0; i < F; i++) {
    if (!Number.isFinite(x46[i])) {
      // features.js reports NaN for a hand it could not measure. No score,
      // no tier, no tracking verdict: this hand was never seen.
      return result("abstain", "not-gradable", -Infinity, Infinity, th.gate, null, model.tiers[target]);
    }
  }

  const r = decide(model, standardize(model, x46, model.work.z), target, opts);
  if (OUTRANKS_TRACKING.has(r.reason)) return r;
  if (qualityIsPoor(quality)) {
    // The score stays on the result: what the hand would have scored is
    // worth logging even though the tutor says nothing about it.
    r.outcome = "abstain";
    r.reason = "tracking";
  }
  return r;
}
