/* feedback.js: what the tutor says about a hand, said specifically (JT's class, 19 Oct 2026: when told "that is a
 * different sign", the students asked to be told WHICH sign it was, and how to fix it).
 *
 * Three questions, each answered with the grader's own arithmetic and nothing else:
 *
 *   WHICH SIGN IS IT?  looksLike: the letter this hand WOULD be accepted as, had that letter been asked for -- the same
 *       test the target gets (inside that letter's typicality gate, and its score against every other class, the
 *       target included, over that letter's accept cut), with tutor/mix.js's rules (no P with the middle finger up, no
 *       E with the fingertips standing off the knuckles, no B with the thumb out). When no letter passes, no letter is
 *       named. That is the CHI 2026 egocentric-ASL paper's confidence gate (Cai, Li et al.: report an error only when
 *       the detector is sure), in this grader's terms: a learner is never told "that looks like S" about a hand the
 *       grader would not call an S.
 *   WHAT TO CHANGE?  the one block tutor/blame.js's chooseCorrection names (the articulator whose correction alone
 *       would make the grader accept), phrased by tutor/letters.js for the TARGET letter -- the sentence that says
 *       what the target's thumb or fingers do, which is the difference that matters. When no one- or two-block
 *       correction is enough and no letter is named, the two blocks that would help most are named as the ones to
 *       check (not as a fix: the model cannot say they are enough).
 *   HOW FAR OFF IS EACH PART?  blockLevels: per articulator, how much fixing it alone would raise the grade (blame's
 *       gain, per square root of the block's width -- blame's own ranking), as a 0..1 level for the per-finger meters
 *       and the teacher's notes. Not each finger's typicality on its own: measured on the six fluent signers, an S's
 *       thumb is a TYPICAL A thumb and a V's spread a typical U spread -- the letters differ in how the parts go
 *       together, which only the grader's relative score sees.
 *
 * The learner sees words and pictures, never a number (experiments/asl-tutor/ui.js); the numbers go to teacherNotes,
 * the structured record behind each hint (the paper's teacher_notes: primary error, its parameter, details).
 *
 * Pure: no DOM, no clock. */

import { BLOCKS, FEATURE_NAMES, blockIndices } from "./features.js";
import { scoreAll } from "./verifier.js";
import { blameBlocks, blockAtTarget } from "./blame.js";
import { hintFor, letterSpec } from "./letters.js";

/* The paper's four manual parameters, for each of the grader's blocks. Location and movement are not blocks: location
 * is whether the whole hand is in view (the tracking check), movement is J and Z (tutor/motion.js). */
export const PARAMETER_OF = Object.freeze({
  thumb: "handshape", index: "handshape", middle: "handshape", ring: "handshape", pinky: "handshape", spread: "handshape",
  orientation: "orientation",
});

/* NAMING IS STRICTER THAN ACCEPTING. A letter's own accept cut is set for a learner who is TRYING to make it, and most
 * are below zero (F -23, W -35): a hand only somewhat like an F passes as an F when F is asked for. To be named, a letter
 * must also beat every other class outright -- the target included -- and hold at least NAME_MIN_P of the likelihood
 * (the paper's gate for a primary error was 0.6). On the six fluent signers' 528 signs, each offered as every other
 * letter the grader rejects it as (11,827): the letter the signer made is named 94.9%, another letter 3.2% (about the
 * grader's own misreadings of those signs: an M it reads as S is named S), none 1.9%. Without the extra conditions,
 * 4.2% wrong; study 1's novices' rejected test signs get a letter named 80% of the time, and drawn out they do look
 * like it (an I with the index up is a D, a fist with the thumb across an S). */
export const NAME_MIN_P = 0.6;

/**
 * The letter this hand would be accepted as, if that letter were the one asked for -- and is more like than anything
 * else; null if none (the target itself is never named: a hand accepted as it gets no correction).
 * @param {object} model   a loaded model
 * @param {ArrayLike<number>} z  the hand, standardized
 * @param {string} target  the letter asked for
 * @param {object} [o]
 * @param {string[]} [o.exclude]  letters this hand cannot be (tutor/mix.js's P and O/E rules): never named, never rivals
 * @param {boolean} [o.notB]      B is ruled out too (tutor/mix.js's B thumb rule)
 * @returns {{letter:string, margin:number, p:number}|null}  margin: its score over its own cut; p: its share of the
 *   likelihood over every class it competes with (equal priors) -- for the notes, never for the learner
 */
export function looksLike(model, z, target, { exclude = [], notB = false } = {}) {
  for (let i = 0; i < z.length; i++) if (!Number.isFinite(z[i])) return null;
  const { ll, d2 } = scoreAll(model, z);
  const names = model.names.filter((n) => !exclude.includes(n));
  let best = null;
  for (const L of model.letters) {
    if (L === target || exclude.includes(L) || (notB && L === "B")) continue;
    const th = model.thresholds[L];
    if (!(d2[L] <= th.gate)) continue;
    let rival = -Infinity;
    for (const n of names) if (n !== L && ll[n] > rival) rival = ll[n];
    const score = ll[L] - rival, cut = Math.max(th.accept, 0);
    if (score >= cut && (best === null || ll[L] > best.ll)) best = { letter: L, ll: ll[L], margin: score - cut };
  }
  if (best === null) return null;
  let mx = -Infinity;
  for (const n of names) if (ll[n] > mx) mx = ll[n];
  let sum = 0;
  for (const n of names) sum += Math.exp(ll[n] - mx);
  const p = Math.exp(best.ll - mx) / sum;
  return p >= NAME_MIN_P ? { letter: best.letter, margin: best.margin, p } : null;
}

/* Each block, as it is named in a sentence. */
export const PART_WORDS = Object.freeze({
  thumb: "your thumb", index: "your index finger", middle: "your middle finger", ring: "your ring finger",
  pinky: "your pinky", spread: "the gaps between your fingers", orientation: "which way your hand points",
});

/* tutor/mix.js's rules, as the block each one is about (its `by`): the hint for that block of the target letter is the
 * sentence the rule is about (B's thumb across the palm; P's fingers pointing down; E's fingertips folded down). */
export const RULE_BLOCK = Object.freeze({ "b-thumb": "thumb", "p-rule": "orientation", "o-e-rule": "index" });

/* FROM THE LETTER MADE TO THE LETTER ASKED FOR. One sentence per pair learners mix up (target first: "AS" is "asked for
 * A, made S"), saying what to move FROM what TO what -- the form that worked best in Shao et al. 2020 (one expert
 * instruction per detected -> target handshape) and in the CHI 2026 paper's best feedback ("used a BENT-L ... should
 * be OPEN-A ... go back to ..."). Checked against Handspeak (Lapiak), Lifeprint (Vicars) and Keane 2014's handshape
 * tables. `parts`: the blocks the sentence is about. It is used only when the grader's blame names one of them (or
 * names nothing), so the words never point at one finger while the golden finger moves another. */
const FINGERS4 = ["index", "middle", "ring", "pinky"];
const pair = (text, ...parts) => Object.freeze({ text, parts: Object.freeze(parts.flat()) });
export const CONTRAST = Object.freeze({
  AS: pair("For A, move your thumb from across your fingers to the side of your index finger.", "thumb"),
  AT: pair("For A, pull your thumb out from under your index finger and rest it at the side.", "thumb"),
  AM: pair("For A, pull your thumb out from under your fingers and rest it at the side of your index finger.", "thumb"),
  AN: pair("For A, pull your thumb out from under your fingers and rest it at the side of your index finger.", "thumb"),
  AE: pair("For A, curl your fingers into a fist and rest your thumb against the side of your index finger.", FINGERS4, "thumb"),
  SA: pair("For S, wrap your thumb across the front of your fingers, not at their side.", "thumb"),
  ST: pair("For S, pull your thumb out from between your fingers and wrap it across their front.", "thumb"),
  SM: pair("For S, pull your thumb out from under your fingers and wrap it across their front.", "thumb"),
  SN: pair("For S, pull your thumb out from under your fingers and wrap it across their front.", "thumb"),
  SE: pair("For S, curl your fingers into a tight fist and wrap your thumb across the front.", FINGERS4, "thumb"),
  EA: pair("For E, fold your thumb across your palm and rest your bent fingertips on it.", "thumb", FINGERS4),
  ES: pair("For E, tuck your thumb under your fingertips instead of across the front of your fingers, and rest your fingertips on it.", "thumb", FINGERS4),
  EM: pair("For E, slide your thumb out from under your fingers and rest your fingertips on top of it.", "thumb", FINGERS4),
  EN: pair("For E, slide your thumb out from under your fingers and rest your fingertips on top of it.", "thumb", FINGERS4),
  ET: pair("For E, slide your thumb out from under your fingers and rest your fingertips on top of it.", "thumb", FINGERS4),
  EO: pair("For E, fold your thumb across your palm and rest your bent fingertips on it.", "thumb", FINGERS4),
  MN: pair("For M, drape three fingers over your thumb, not two.", "ring", "thumb"),
  MT: pair("For M, tuck your thumb under three fingers, with its tip between your ring finger and pinky.", "thumb", "middle", "ring"),
  MA: pair("For M, tuck your thumb under your index, middle and ring fingers.", "thumb", "index", "middle", "ring"),
  MS: pair("For M, tuck your thumb under your index, middle and ring fingers.", "thumb", "index", "middle", "ring"),
  ME: pair("For M, tuck your thumb under your index, middle and ring fingers.", "thumb", "index", "middle", "ring"),
  NM: pair("For N, drape two fingers over your thumb, not three.", "ring", "thumb"),
  NT: pair("For N, tuck your thumb under two fingers, with its tip between your middle and ring fingers.", "thumb", "middle"),
  NA: pair("For N, tuck your thumb under your index and middle fingers.", "thumb", "index", "middle"),
  NS: pair("For N, tuck your thumb under your index and middle fingers.", "thumb", "index", "middle"),
  NE: pair("For N, tuck your thumb under your index and middle fingers.", "thumb", "index", "middle"),
  TA: pair("For T, tuck your thumb under your index finger, with its tip between your index and middle fingers.", "thumb", "index"),
  TS: pair("For T, tuck your thumb under your index finger, with its tip between your index and middle fingers.", "thumb", "index"),
  TE: pair("For T, tuck your thumb under your index finger, with its tip between your index and middle fingers.", "thumb", "index"),
  TM: pair("For T, tuck your thumb under your index finger only.", "thumb", "middle", "ring"),
  TN: pair("For T, tuck your thumb under your index finger only.", "thumb", "middle"),
  CO: pair("For C, open a gap between your fingertips and your thumb.", "thumb", FINGERS4),
  OC: pair("For O, close the gap: bring your fingertips round to touch your thumb.", "thumb", FINGERS4),
  OE: pair("For O, curve your fingers forward so their tips meet your thumb tip.", FINGERS4, "thumb"),
  OD: pair("For O, curve your index finger down to meet your thumb as well.", "index"),
  OF: pair("For O, curve your middle, ring and pinky fingers down to meet your thumb as well.", "middle", "ring", "pinky"),
  DO: pair("For D, lift your index finger straight up; keep the others touching your thumb.", "index"),
  DF: pair("For D, point your index finger up and touch your thumb with the other three.", FINGERS4, "thumb"),
  FO: pair("For F, keep your index finger on your thumb and raise your other three fingers.", "middle", "ring", "pinky"),
  FD: pair("For F, touch your index finger to your thumb and raise the other three.", FINGERS4),
  FW: pair("For F, touch your thumb to your index finger instead of your pinky.", "thumb", "index", "pinky"),
  GH: pair("For G, fold your middle finger down; only your index finger and thumb stick out.", "middle", "spread"),
  GQ: pair("For G, lift your hand so your index finger points across your body, not down.", "orientation"),
  GL: pair("For G, turn your hand sideways and bring your thumb parallel to your index finger.", "orientation", "thumb"),
  HG: pair("For H, add your middle finger beside your index finger and tuck your thumb in; both fingers point sideways.", "middle", "spread", "thumb"),
  HU: pair("For H, turn your hand sideways so both fingers point across your body.", "orientation"),
  HV: pair("For H, press your two fingers together and point them sideways.", "spread", "orientation"),
  HR: pair("For H, uncross your fingers and point them sideways, side by side.", "spread", "orientation"),
  HK: pair("For H, keep both fingers straight and together, pointing sideways, with your thumb tucked in.", "spread", "middle", "thumb", "orientation"),
  QG: pair("For Q, point your index finger and thumb down.", "orientation"),
  QP: pair("For Q, fold your middle finger in; only your index finger and thumb point down.", "middle"),
  LG: pair("For L, point your index finger straight up and stick your thumb out to the side.", "orientation", "thumb"),
  XD: pair("For X, bend your index finger into a hook; curl the other fingers into a fist.", FINGERS4, "thumb"),
  XG: pair("For X, point your index finger up, bend it into a hook, and fold your thumb in.", "orientation", "index", "thumb"),
  KP: pair("For K, turn your hand upright so your index finger points up.", "orientation"),
  KV: pair("For K, put your thumb tip between your two fingers, against your middle finger.", "thumb", "middle"),
  KU: pair("For K, part the two fingers and bring your thumb tip up between them.", "thumb", "spread", "middle"),
  PK: pair("For P, tip your hand so your fingers point down.", "orientation"),
  PQ: pair("For P, add your middle finger pointing down, with your thumb touching it.", "middle", "thumb"),
  RU: pair("For R, cross your middle finger over the back of your index finger.", "spread"),
  RV: pair("For R, cross your middle finger over the back of your index finger instead of spreading them apart.", "spread"),
  UV: pair("For U, press your two raised fingers together, with no gap.", "spread"),
  UR: pair("For U, uncross your two fingers and press them together.", "spread"),
  UH: pair("For U, point your two fingers straight up.", "orientation"),
  VU: pair("For V, spread your two raised fingers apart.", "spread"),
  VW: pair("For V, fold your ring finger down; keep only your index and middle fingers up.", "ring"),
  VK: pair("For V, fold your thumb over your ring finger and spread your two fingers apart.", "thumb", "middle", "spread"),
  WV: pair("For W, raise your ring finger too, spread apart from the other two.", "ring"),
  WF: pair("For W, hold your pinky down with your thumb and raise your index, middle and ring fingers.", "index", "pinky", "thumb"),
  IY: pair("For I, fold your thumb across your curled fingers; only your pinky stays up.", "thumb"),
  ID: pair("For I, curl your index finger down, fold your thumb across your fingers, and raise your pinky instead.", "index", "pinky", "thumb"),
  IB: pair("For I, curl your index, middle and ring fingers down and fold your thumb across them; only your pinky stays up.", "index", "middle", "ring", "thumb"),
  YI: pair("For Y, stick your thumb straight out to the side.", "thumb"),
  YV: pair("For Y, curl your index and middle fingers down, and stick out your thumb and pinky.", "index", "middle", "thumb", "pinky"),
  DI: pair("For D, raise your index finger and curl your pinky down to your thumb.", "index", "pinky"),
});

/* The pair's sentence and its parts, if the letters asked for and made are a pair learners mix up. */
export const contrastFor = (letter, looks) => (looks ? CONTRAST[`${letter}${looks}`] ?? null : null);

/* WHICH PARTS TO SHOW (the golden finger, the meters' amber) when the hand looks like another letter. The pair's
 * sentence is what the learner is told, so the parts shown are the ones it is about -- unless blame's own part is
 * already one of them (then blame's, which is this hand's) or the sentence is about more than MAX_SHOWN parts (then
 * blame's: five golden fingers at once is no longer a hint). On study 1's novices' rejected signs, the pair's parts
 * and blame's disagreed on 32 of the 274 named mix-ups that have a pair sentence (an O made for F: blame says the
 * thumb, the fix is to raise the other three fingers). */
export const MAX_SHOWN = 3;
export function partsToShow(letter, looks, blocks) {
  const c = contrastFor(letter, looks);
  if (!c || (blocks.length && c.parts.includes(blocks[0])) || c.parts.length > MAX_SHOWN) return blocks;
  return c.parts.slice();
}

/* tutor/letters.js has ONE sentence for all of a letter's finger blocks ("fingers / spread"), which is right for most
 * letters but says the wrong thing when the finger blamed is not the one the sentence is about: F's is about the three
 * raised fingers, so a hand whose INDEX is wrong was told to keep the other three straight. Here, the sentence for that
 * finger. (The study's words are tutor/letters.js's, unchanged.) */
const CURL_RP = "Curl your ring finger and pinky down, with your thumb over them.";
/* R: the middle finger crosses over the BACK of the index, so from the palm side the index is in front (Helen Keller
 * National Center: "the middle finger wraps behind the index finger"; the tutor's own R picture). letters.js's "cross
 * your index finger over your middle finger" can be read the other way round. */
const CROSS_R = "Cross your middle finger over the back of your index finger.";
export const PART_HINTS = Object.freeze({
  "R.index": CROSS_R, "R.middle": CROSS_R, "R.spread": CROSS_R,
  "F.index": "Bend your index finger down so its tip touches your thumb tip.",
  "R.ring": CURL_RP, "R.pinky": CURL_RP,
  "U.ring": CURL_RP, "U.pinky": CURL_RP,
  "V.ring": CURL_RP, "V.pinky": CURL_RP,
  "W.pinky": "Fold your pinky down and hold it with your thumb.",
  "X.middle": "Curl your middle, ring and pinky fingers into a fist.", "X.ring": "Curl your middle, ring and pinky fingers into a fist.",
  "X.pinky": "Curl your middle, ring and pinky fingers into a fist.",
});
export const partHint = (letter, block) => PART_HINTS[`${letter}.${block}`] ?? hintFor(letter, block);

const lowerFirst = (s) => (s ? s[0].toLowerCase() + s.slice(1) : s);

/**
 * The one line after a hold that was not accepted.
 * @param {object} o
 * @param {string} o.letter         the target
 * @param {string|null} [o.looks]   looksLike's letter
 * @param {string|null} [o.block]   the block to change (chooseCorrection's, as the flow chose it)
 * @param {string|null} [o.rule]    a mix.js rule that refused a hand the plain grader accepts ("b-thumb", "p-rule", "o-e-rule")
 * @param {string[]} [o.check]      when there is no block and no letter: the (at most two) parts that would help most
 */
export function correctionText({ letter, looks = null, block = null, rule = null, check = [] }) {
  const named = looks ? `That looks like ${looks}. ` : "";
  // the expert sentence for this mix-up, when there is one: it says what to move from what to what
  const contrast = contrastFor(letter, looks);
  if (contrast) return named + contrast.text;
  if (rule && RULE_BLOCK[rule]) {
    const hint = partHint(letter, RULE_BLOCK[rule]);
    return looks ? named + hint : `Almost — ${lowerFirst(hint)}`;
  }
  if (block) return named + partHint(letter, block);
  if (looks) return `${named}For ${letter}: ${lowerFirst(letterSpec(letter).describe)}`;
  const parts = check.filter((b) => PART_WORDS[b]).slice(0, 2).map((b) => PART_WORDS[b]);
  if (parts.length) return `Not quite — check ${parts.join(" and ")} against the picture.`;
  return "Not quite — have a look at the shape again.";
}

/* BLOCK LEVELS. A block's rank (blame's gain / sqrt(width)) at or under RANK_OK is not what is holding the letter back
 * (study 1's 66 novices: 95% of the blocks of the holds the grader ACCEPTED rank under 2.7, thumb included); at
 * RANK_BAD or over, it plainly is (90% of the blocks blame NAMED rank over 3.3-10, by block). The level runs 1 -> 0
 * between them on a log scale. */
export const RANK_OK = 2.5;
export const RANK_BAD = 8;
export const levelOfRank = (rank) => {
  if (!(rank > RANK_OK)) return 1;
  if (rank >= RANK_BAD) return 0;
  return 1 - Math.log(rank / RANK_OK) / Math.log(RANK_BAD / RANK_OK);
};

/* Which way a block has to go, in the hand's own terms, from the block's corrected value (blame: its conditional mean
 * under the target, given the rest of this hand): a finger BENDS or STRAIGHTENS (its three joint angles together, by
 * more than DIR_DEG), the fingers go APART or TOGETHER (the gap that has to change most, by more than DIR_DEG / 2) or CROSS (R);
 * the thumb and the hand's direction have no one word, and get none. */
export const DIR_DEG = 20;
const IX = Object.fromEntries(FEATURE_NAMES.map((n, i) => [n, i]));
const raw = (model, z, name) => z[IX[name]] * model.sd[IX[name]] + model.mean[IX[name]];
export function directionOf(model, z, zFixed, block) {
  const delta = (names) => names.reduce((s, n) => s + raw(model, zFixed, n) - raw(model, z, n), 0);
  if (["index", "middle", "ring", "pinky"].includes(block)) {
    const d = delta([`flex_${block}_mcp`, `flex_${block}_pip`, `flex_${block}_dip`]);
    return d > DIR_DEG ? "bend" : d < -DIR_DEG ? "straighten" : null;
  }
  if (block === "spread") {
    // CROSS only for a target that crosses (R: the index tip ends up on the middle finger's far side) and a hand that
    // does not; closing a V into a U also moves the index tip that way, without crossing
    if (raw(model, zFixed, "cross_im_radial") < 0 && raw(model, z, "cross_im_radial") > 0) return "cross";
    // the gap that has to change most decides (a V from a U opens only the index-middle gap)
    const d = ["abd_index_middle", "abd_middle_ring", "abd_ring_pinky"].map((n) => delta([n])).reduce((a, b) => (Math.abs(b) > Math.abs(a) ? b : a), 0);
    return d > DIR_DEG / 2 ? "apart" : d < -DIR_DEG / 2 ? "together" : null;
  }
  return null;
}

/**
 * Every block's standing for the target: { block: { rank, gain, level, dir } }, from tutor/blame.js's gains.
 * `accepted`: the grader accepts the hand -- then nothing is holding it back: every level is 1, and nothing is computed.
 * `named`: the blocks the tutor is naming now -- they are never shown at the top (at most NAMED_MAX), so the meter
 * agrees with the words, and only they get a direction. `gains`: chooseCorrection's, when the caller has them already
 * (the live coach, every frame), so blame is not run twice.
 */
export const NAMED_MAX = 0.6;
const WIDTH = Object.fromEntries(BLOCKS.map((b) => [b, Math.sqrt(blockIndices(b).length)]));
export function blockLevels(model, z, target, { accepted = false, named = [], gains = null } = {}) {
  const out = {};
  if (accepted) {
    for (const b of BLOCKS) out[b] = { rank: 0, gain: 0, level: 1, dir: null };
    return out;
  }
  const g = gains ?? Object.fromEntries(blameBlocks(model, z, target).map((e) => [e.block, e.gain]));
  for (const b of BLOCKS) {
    const gain = g[b], rank = gain / WIDTH[b];
    let level = levelOfRank(rank);
    let dir = null;
    if (named.includes(b)) {
      level = Math.min(level, NAMED_MAX);
      dir = directionOf(model, z, blockAtTarget(model, z, target, b), b);
    }
    out[b] = { rank, gain, level, dir };
  }
  return out;
}

/* The two blocks most worth checking when there is no fix: the best two by rank, with something to gain. */
export function partsToCheck(levels) {
  return Object.entries(levels).filter(([, v]) => v.rank > RANK_OK).sort((a, b) => b[1].rank - a[1].rank).slice(0, 2).map(([b]) => b);
}

/* How far the hand's facing and pointing are from the target's average hand, in degrees (camera frame; the pointing
 * direction's left-right is folded, as the features fold it). For the notes. */
function angleBetween(a, b) {
  const na = Math.hypot(...a), nb = Math.hypot(...b);
  if (!(na > 0 && nb > 0)) return null;
  const c = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (na * nb);
  return Math.round((Math.acos(Math.max(-1, Math.min(1, c))) * 180) / Math.PI);
}
export function orientationDegrees(model, z, target) {
  const mu = model.classes[target].mu;
  const v = (zz, names) => names.map((n) => raw(model, zz, n));
  const palm = ["palmnormal_x", "palmnormal_y", "palmnormal_z"], dir = ["handdir_x", "handdir_y", "handdir_z"];
  return { palm: angleBetween(v(z, palm), v(mu, palm)), pointing: angleBetween(v(z, dir), v(mu, dir)) };
}

/**
 * The structured record behind one graded hold, after the CHI 2026 paper's JSON (student_feedback, teacher_notes:
 * primary_error, confidence, details) in this grader's terms. For a teacher's view and a log; never shown to a learner.
 * @param {object} o
 * @param {object} o.model
 * @param {ArrayLike<number>} o.z         the hand, standardized
 * @param {string} o.letter               the target
 * @param {object} o.verdict              the grader's (verify / mixed verify) result
 * @param {string[]} [o.blocks]           the blocks the hint names
 * @param {string|null} [o.looks]         looksLike's letter
 * @param {object} [o.quality]            { centered, palmSizePx }
 * @param {object|null} [o.motion]        tutor/motion.js's verdict on a J or Z
 * @param {string} [o.text]               what the learner was told
 */
export function teacherNotes({ model, z, letter, verdict, blocks = [], looks = null, quality = null, motion = null, text = "" }) {
  const accepted = verdict?.outcome === "accept";
  const finite = z && Array.from(z).every(Number.isFinite) && letter in model.thresholds;
  const levels = finite ? blockLevels(model, z, letter, { accepted, named: blocks }) : null;
  const off = levels ? BLOCKS.filter((b) => b !== "orientation" && (blocks.includes(b) || levels[b].level < 1)) : [];
  const orient = finite ? orientationDegrees(model, z, letter) : null;
  const part = (b) => (b ? { parameter: PARAMETER_OF[b], part: b, change: levels?.[b]?.dir ?? null } : null);
  return {
    target: letter,
    accepted,
    looksLike: looks,
    primaryError: accepted ? null : part(blocks[0] ?? null),
    secondaryError: accepted ? null : part(blocks[1] ?? null),
    parameters: {
      handshape: levels ? { ok: accepted || off.length === 0, off: accepted ? [] : off } : null,
      orientation: levels ? { ok: accepted || (!blocks.includes("orientation") && levels.orientation.level === 1), palmDeg: orient.palm, pointingDeg: orient.pointing } : null,
      // fingerspelling's location (beside the shoulder) is not measured: only whether the whole hand is in view
      location: quality ? { ok: null, measured: false, wholeHandInView: !!quality.centered, palmPx: Math.round(quality.palmSizePx ?? 0) } : null,
      movement: motion ? { ok: !!motion.ok, reason: motion.reason ?? null } : { ok: null, note: "held still" },
    },
    levels: levels && Object.fromEntries(Object.entries(levels).map(([b, v]) => [b, Math.round(v.level * 100) / 100])),
    grader: verdict ? { outcome: verdict.outcome, reason: verdict.reason ?? null, rival: verdict.rival ?? null, by: verdict.mix?.by ?? null,
      score: round2(verdict.score), accept: round2(model.thresholds[letter]?.accept), d2: round2(verdict.d2), gate: round2(verdict.gate) } : null,
    studentFeedback: text,
  };
}
const round2 = (v) => (Number.isFinite(v) ? Math.round(v * 100) / 100 : null);
