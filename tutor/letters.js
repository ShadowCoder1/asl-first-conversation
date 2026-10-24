/* letters.js: per-letter fingerspelling specs -- what each handshape looks
 * like, and the one short categorical sentence shown when a block
 * (features.js BLOCKS: thumb, index, middle, ring, pinky, spread,
 * orientation) is blamed for a rejected sign.
 *
 * WHY CATEGORICAL, NOT NUMERIC. Piloting this tutor with a number in the
 * hint ("bend your index finger 30 degrees") overwhelmed learners mid
 * practice -- they do not have a mental ruler for their own joints. A short
 * gestalt phrase ("Only the index finger is straight.") does not ask for
 * that ruler, so every sentence in this file is categorical: no digits, no
 * degree sign, one short second-person sentence.
 *
 * SCOPE. This module teaches fingerspelling HANDSHAPES only. It makes no
 * claim to teach American Sign Language, which is its own language with its
 * own grammar, none of which lives here.
 *
 * DOM-free, fetch-free, Math.random-free, no dependencies: this is plain
 * data plus lookup functions, imported by both the practice UI and this
 * module's own test. The only import is BLOCKS from features.js, so
 * hintFor's block validation can never name a block the scoring model does
 * not actually have. */
import { BLOCKS } from "./features.js";

export const LETTERS = Object.freeze("ABCDEFGHIJKLMNOPQRSTUVWXYZ".split(""));

// J and Z are traced in the air, not held in place. The bootstrap grader
// (model.json) is fit on static posed hands and defines no class for
// either, so they are the only two letters excluded from STATIC_LETTERS and
// the only two with `motion: true`.
const MOTION_LETTERS = Object.freeze(["J", "Z"]);

export const STATIC_LETTERS = Object.freeze(LETTERS.filter((l) => !MOTION_LETTERS.includes(l)));

// Hard-coded rather than imported: letters.js is DOM-free and the browser
// build has no JSON-import step, so model.json's `confusions` table is
// copied here by hand, letter for letter. tests/letters.test.js reads
// model.json with `fs` and asserts this object is deeply equal to
// model.confusions, so the two copies can never silently drift apart.
//
// J and Z have no entry in model.json (the model has no class for either --
// see MOTION_LETTERS above), so they are NOT invented a confusion set here.
// They get the empty string, same shape as a static letter with no
// confusions (B, L, W, X), which is honest: the bootstrap model has never
// scored either letter, so it has no evidence of what they are confused
// with.
export const CONFUSIONS = Object.freeze({
  A: "EMNST", B: "", C: "O", D: "FO", E: "AMNST", F: "DO", G: "HQ",
  H: "GKQRUV", I: "Y", J: "", K: "HPRUV", L: "", M: "AENST", N: "AEMST",
  O: "CDF", P: "K", Q: "GH", R: "HKUV", S: "AEMNT", T: "AEMNS",
  U: "HKRV", V: "HKRU", W: "", X: "", Y: "I", Z: "",
});

// One sentence per block, shown only when hintFor is asked about a block a
// letter's own table has no sentence for. None of the 26 letters below
// actually hits this path today (the phrase table is filled for every
// block of every letter) -- it exists as the documented, tested fallback
// for whatever letter or block is added later without full data, so the
// tutor never renders `undefined` on screen instead of a sentence.
export const GENERIC_HINTS = Object.freeze({
  thumb: "Adjust your thumb to match the target letter.",
  index: "Adjust your index finger to match the target letter.",
  middle: "Adjust your middle finger to match the target letter.",
  ring: "Adjust your ring finger to match the target letter.",
  pinky: "Adjust your pinky finger to match the target letter.",
  spread: "Adjust how far apart your fingers are to match the target letter.",
  orientation: "Adjust which way your palm is facing to match the target letter.",
});

// The five blocks whose text lives in one shared "fingers" sentence per
// letter (the brief's "fingers / spread" table column), rather than one
// sentence per block -- a hand's fingers move together, so a separate
// sentence per finger would just repeat itself five times.
const FINGER_BLOCKS = Object.freeze(["index", "middle", "ring", "pinky", "spread"]);

// One row per letter, authored verbatim from task-5-brief.md's phrase
// table (American spelling, second person, one short sentence each).
// `fingers` is the brief's single "fingers / spread" sentence; buildSpec
// below expands it into all five of blocks.index/middle/ring/pinky/spread
// so that letterSpec(L).blocks matches the documented interface shape and
// hintFor needs no special-casing beyond the lookup itself.
const TABLE = {
  A: { motion: false, describe: "A fist with the thumb resting against the side of the index finger.", thumb: "Rest your thumb against the SIDE of your index finger, pointing up — not across the front.", fingers: "Curl all four fingers into a tight fist.", orientation: "Palm faces forward." },
  B: { motion: false, describe: "A flat hand, fingers together and straight up, thumb folded across the palm.", thumb: "Fold your thumb across your palm.", fingers: "Keep all four fingers straight and pressed together.", orientation: "Palm faces forward, fingers up." },
  C: { motion: false, describe: "Curve your whole hand into the shape of a C.", thumb: "Curve your thumb so it mirrors your fingers.", fingers: "Curve all four fingers together — not a fist, not flat.", orientation: "Turn so the C opening faces sideways." },
  D: { motion: false, describe: "Index finger straight up; the other fingertips meet the thumb in a circle.", thumb: "Touch your thumb tip to your middle fingertip.", fingers: "Only the index finger is straight; curl the other three down to the thumb.", orientation: "Palm faces forward." },
  E: { motion: false, describe: "Fingers bent down so the tips rest on the thumb, which lies across the palm.", thumb: "Tuck your thumb across your palm, UNDER your fingertips.", fingers: "Bend all four fingers so the tips come down to the thumb — do not make a fist.", orientation: "Palm faces forward." },
  F: { motion: false, describe: "Index fingertip and thumb tip touch in a circle; the other three fingers are straight and spread.", thumb: "Touch your thumb tip to your index fingertip.", fingers: "Keep your middle, ring and pinky fingers straight and apart.", orientation: "Palm faces forward." },
  G: { motion: false, describe: "Index finger and thumb point sideways, parallel, like measuring something small.", thumb: "Hold your thumb parallel to your index finger.", fingers: "Only the index finger is straight; the rest are curled.", orientation: "Point your index finger SIDEWAYS, palm toward you." },
  H: { motion: false, describe: "Index and middle fingers together, pointing sideways.", thumb: "Tuck your thumb in against your curled fingers.", fingers: "Index and middle fingers straight and TOGETHER; ring and pinky curled.", orientation: "Point the two fingers SIDEWAYS." },
  I: { motion: false, describe: "A fist with only the pinky straight up.", thumb: "Fold your thumb across the front of your fingers.", fingers: "Only your pinky is straight.", orientation: "Palm faces forward." },
  J: { motion: true, describe: "Make an I, then draw a J in the air with your pinky.", thumb: "Fold your thumb across the front of your fingers.", fingers: "Only your pinky is straight.", orientation: "Start palm forward and trace the hook of a J." },
  K: { motion: false, describe: "Index up, middle finger angled forward, thumb tip resting between them.", thumb: "Put your thumb tip on the middle of your middle finger.", fingers: "Index straight up, middle finger angled forward; ring and pinky curled.", orientation: "Palm faces forward, fingers up." },
  L: { motion: false, describe: "Index finger up, thumb straight out: an L.", thumb: "Stick your thumb straight out to the side.", fingers: "Only the index finger is straight.", orientation: "Palm faces forward." },
  M: { motion: false, describe: "Thumb tucked under the first THREE fingers.", thumb: "Tuck your thumb under your index, middle and ring fingers.", fingers: "Fold three fingers down over the thumb; pinky curled beside them.", orientation: "Palm faces forward." },
  N: { motion: false, describe: "Thumb tucked under the first TWO fingers.", thumb: "Tuck your thumb under your index and middle fingers.", fingers: "Fold two fingers down over the thumb; ring and pinky curled.", orientation: "Palm faces forward." },
  O: { motion: false, describe: "All fingertips curve down to meet the thumb tip: an O.", thumb: "Bring your thumb tip up to meet your fingertips.", fingers: "Curve all four fingers so the tips meet the thumb.", orientation: "Turn so the O opening faces sideways." },
  P: { motion: false, describe: "A K pointed downward.", thumb: "Put your thumb tip on the middle of your middle finger.", fingers: "Index straight, middle finger angled; ring and pinky curled.", orientation: "Tip your hand so the fingers point DOWN." },
  Q: { motion: false, describe: "A G pointed downward.", thumb: "Hold your thumb parallel to your index finger.", fingers: "Only the index finger is straight.", orientation: "Point your index finger and thumb DOWN." },
  R: { motion: false, describe: "Index and middle fingers crossed.", thumb: "Fold your thumb over your curled ring finger.", fingers: "CROSS your index finger over your middle finger.", orientation: "Palm faces forward, fingers up." },
  S: { motion: false, describe: "A fist with the thumb across the FRONT of the fingers.", thumb: "Wrap your thumb across the front of your fingers — not alongside them.", fingers: "Curl all four fingers into a tight fist.", orientation: "Palm faces forward." },
  T: { motion: false, describe: "A fist with the thumb tucked between the index and middle fingers.", thumb: "Tuck your thumb between your index and middle fingers.", fingers: "Curl all four fingers; the index folds over the thumb.", orientation: "Palm faces forward." },
  U: { motion: false, describe: "Index and middle fingers straight up and together.", thumb: "Fold your thumb over your curled ring finger.", fingers: "Index and middle fingers straight and TOGETHER — no gap.", orientation: "Palm faces forward, fingers up." },
  V: { motion: false, describe: "Index and middle fingers straight up and apart: a V.", thumb: "Fold your thumb over your curled ring finger.", fingers: "SPREAD your index and middle fingers apart.", orientation: "Palm faces forward, fingers up." },
  W: { motion: false, describe: "Index, middle and ring fingers up and apart; thumb holds down the pinky.", thumb: "Hold your pinky down with your thumb.", fingers: "Three fingers straight and apart.", orientation: "Palm faces forward, fingers up." },
  X: { motion: false, describe: "A fist with the index finger raised and bent into a hook.", thumb: "Fold your thumb across the front of your fingers.", fingers: "Raise your index finger and bend it into a hook.", orientation: "Palm faces sideways." },
  Y: { motion: false, describe: "Thumb and pinky out, the three middle fingers curled.", thumb: "Stick your thumb straight out.", fingers: "Pinky straight out; index, middle and ring fingers curled.", orientation: "Palm faces forward." },
  Z: { motion: true, describe: "Point your index finger and draw a Z in the air.", thumb: "Fold your thumb across the front of your fingers.", fingers: "Only the index finger is straight.", orientation: "Trace a Z with your fingertip." },
};

// Mnemonics only for letters whose handshape actually resembles the printed
// letter -- a mnemonic that does not resemble anything is just noise.
const MNEMONICS = Object.freeze({
  C: "It looks like the letter C.",
  I: "It looks like the letter I.",
  J: "It looks like the letter J.",
  L: "It looks like the letter L.",
  O: "It looks like the letter O.",
  V: "It looks like the letter V.",
  W: "It looks like the letter W.",
  Y: "It looks like the letter Y.",
  Z: "It looks like the letter Z.",
});

function buildSpec(letter) {
  const row = TABLE[letter];
  const blocks = { thumb: row.thumb, orientation: row.orientation };
  for (const block of FINGER_BLOCKS) blocks[block] = row.fingers;
  const spec = {
    letter,
    motion: row.motion,
    describe: row.describe,
    blocks: Object.freeze(blocks),
    confusions: CONFUSIONS[letter],
  };
  if (MNEMONICS[letter]) spec.mnemonic = MNEMONICS[letter];
  return Object.freeze(spec);
}

// Built once, deep-frozen: LETTERS, STATIC_LETTERS and every value nested
// under a spec are frozen so an importer cannot mutate shared state (e.g.
// reorder or overwrite one letter's blocks in place) and corrupt every
// other module reading it.
const SPECS = Object.freeze(Object.fromEntries(LETTERS.map((l) => [l, buildSpec(l)])));

export function letterSpec(letter) {
  const spec = SPECS[letter];
  if (!spec) throw new Error(`letterSpec: unknown letter "${letter}"`);
  return spec;
}

// The categorical phrase for a blamed block: the letter's own sentence for
// that block if it has one, else the generic per-block phrase (see
// GENERIC_HINTS above). `resolveHint` is exported separately from `hintFor`
// solely so that fallback branch -- unreachable through the 26 real letters
// below, whose table is fully populated for every block -- still has an
// honest, direct test instead of a fabricated "letter with a missing
// sentence" (see tests/letters.test.js).
export function resolveHint(blocks, block) {
  return blocks[block] ?? GENERIC_HINTS[block];
}

export function hintFor(letter, block) {
  const spec = letterSpec(letter); // throws for an unknown letter
  if (!BLOCKS.includes(block)) throw new Error(`hintFor: unknown block "${block}"`);
  return resolveHint(spec.blocks, block);
}

// -- Reference pictures --------------------------------------------------
// Per-letter public-domain SVGs pulled from Wikimedia Commons (the
// "Sign language <L>.svg" / wpclipart set); see assets/letters/LICENSES.md
// for the license, source and retrieval date recorded for every letter,
// shipped or not. Only letters whose file was confirmed public domain AND
// successfully downloaded are listed here -- no picture is shipped on a
// guess. K, L, M and O were rate-limited by upload.wikimedia.org on the
// first fetch pass (2026-09-20); a later retry that same day, spaced out
// and with a Special:FilePath fallback ready, got all four through, so all
// 26 letters now have a shipped picture.
export const PICTURE_LETTERS = Object.freeze(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]);

/* WHICH WAY ROUND. The clip art is not consistent: 20 letters are drawn as
 * the OTHER person sees a right hand, and G H P Q X Z as the signer sees their
 * own right hand. The camera picture on the page is mirrored, so a learner
 * matched an "other person's view" drawing on screen by using their LEFT hand
 * -- the owner did exactly that for two sessions (every hold read as a left
 * hand), and JT's testers switched hands. So the 20 are shown flipped: every
 * picture now shows what a correct RIGHT hand looks like on the learner's own
 * (mirrored) screen. Found by thumb side and forearm side, letter by letter,
 * 2026-09-22; the tracker cannot read line drawings to check it. */
export const PICTURES_DRAWN_FROM_VIEWER = Object.freeze(["A", "B", "C", "D", "E", "F", "I", "J", "K", "L", "M", "N", "O", "R", "S", "T", "U", "V", "W", "Y"]);
export const PICTURE_CAPTION = "How it looks on your screen.";

/* Should this letter's picture be flipped left-right, for a learner signing
 * with `hand`? For a right hand, the 20 viewer-view drawings; for a left hand,
 * the other six (which makes every picture a left hand on screen). */
export function pictureMirrored(letter, hand = "right") {
  const viewer = PICTURES_DRAWN_FROM_VIEWER.includes(letter);
  return hand === "left" ? !viewer : viewer;
}

export function pictureUrl(letter) {
  return PICTURE_LETTERS.includes(letter) ? `assets/letters/${letter}.svg` : null;
}
