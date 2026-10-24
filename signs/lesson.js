/* lesson.js: First Conversation as data -- the seven signs in the order they are taught, and the chat with Maya that
 * puts them to use (docs/signs-design.md, "What the learner does"). Signs are named by the sign model's own labels
 * (signs/model/labels.json), so a turn's reply is exactly what the recognizer is told to expect:
 * rec.setTarget({ mode: "sequence", signs: turn.reply }). What each sign means, how it is made and how a miss is
 * corrected are in signs/content.json, which the app fetches; this file holds no words about the signs themselves.
 *
 * The chat is ordinary ASL: a reply is the signs a Deaf person would answer with, in ASL order (WATER PLEASE, NO
 * THANK-YOU), and `english` is what that reply means, shown under the chips. Every sign taught is used at least once,
 * and no reply has the same sign twice in a row (in a sequence the recognizer needs the hand to move on between two of
 * the same sign).
 *
 * DOM-free and fetch-free like tutor/letters.js: plain data plus a check, imported by the app and by
 * tests/signs-content.test.js. */

const deepFreeze = (o) => {
  if (o && typeof o === "object" && !Object.isFrozen(o)) {
    Object.freeze(o);
    for (const v of Object.values(o)) deepFreeze(v);
  }
  return o;
};

export const LESSON = deepFreeze({
  id: "first-conversation",
  title: "Your first ASL conversation",
  intro: "Learn 7 signs, then use them in a real back-and-forth.",
  partner: { name: "Maya" },
  // the teaching order of the design: the greeting first, the goodbye last, the rest in roughly the chat's order
  signs: ["hello", "yes", "no", "please", "thankyou", "water", "bye"],
  conversation: [
    { partner: "Hi! I'm Maya.", reply: ["hello"], english: "Hello!" },
    { partner: "Are you thirsty?", reply: ["yes"], english: "Yes." },
    { partner: "What would you like to drink?", reply: ["water", "please"], english: "Water, please." },
    { partner: "Here you go!", reply: ["thankyou"], english: "Thank you!" },
    { partner: "Anything else?", reply: ["no", "thankyou"], english: "No, thank you." },
    { partner: "Okay! See you later.", reply: ["bye"], english: "Bye!" },
  ],
});

/* The demonstrations are PopSign signers' own clips (docs/signs-design.md, "Demonstrations"), and the licence
 * (CC BY 4.0) asks for this credit wherever they are shown. */
export const ATTRIBUTION_VIDEO = "Sign videos: PopSign ASL v1.0 (Starner et al., 2023), CC BY 4.0. Clips were trimmed, scaled and re-encoded.";
export const ATTRIBUTION_LINKS = deepFreeze({
  dataset: "https://signdata.cc.gatech.edu/view/datasets/popsign_v1_0/",
  license: "https://creativecommons.org/licenses/by/4.0/",
});

// The four parts a teacher names (signs/feedback.js measures three of them; orientation comes out of the hand shape's
// blame), and the correction keys feedback.js can pick for place and movement. A key outside these is never shown.
export const PARTS = Object.freeze(["handshape", "location", "movement", "orientation"]);
export const FIX_KEYS = deepFreeze({
  location: ["default", "higher", "lower", "closer", "touch"],
  movement: ["default", "bigger", "smaller", "direction", "repeat", "once", "still"],
});

/** A label's gloss for the screen: the lesson sign's own, then content.glosses (other labels a miss may be named as,
 *  e.g. minemy -> MY), then the label in capitals. */
export function signGloss(content, id) {
  return content?.signs?.[id]?.gloss ?? content?.glosses?.[id] ?? String(id).toUpperCase();
}

const text = (v) => typeof v === "string" && v.trim().length > 0;

/**
 * What would stop the app from teaching this lesson with this content: a sign the model does not know, a sign with
 * missing words, a reply that uses a sign not taught. Empty when all is well.
 * @param {typeof LESSON} lesson
 * @param {object} content  signs/content.json
 * @param {string[]} [labels]  signs/model/labels.json's labels (checked only when given)
 * @returns {string[]}
 */
export function lessonProblems(lesson, content, labels = null) {
  const out = [];
  const known = Array.isArray(labels) ? new Set(labels) : null;
  const signs = Array.isArray(lesson?.signs) ? lesson.signs : [];
  if (!signs.length) out.push("the lesson has no signs");
  if (new Set(signs).size !== signs.length) out.push("a sign is taught twice");
  for (const id of signs) {
    if (known && !known.has(id)) out.push(`${id}: not a label of the sign model`);
    const c = content?.signs?.[id];
    if (!c) { out.push(`${id}: no words in content.json`); continue; }
    for (const f of ["english", "gloss", "howTo", "tip", "variants", "popsignForm"]) if (!text(c[f])) out.push(`${id}: no ${f}`);
    for (const p of PARTS) if (!text(c.parts?.[p])) out.push(`${id}: no parts.${p}`);
    if (!text(c.fixes?.handshape)) out.push(`${id}: no fixes.handshape`);
    if (!text(c.fixes?.orientation)) out.push(`${id}: no fixes.orientation`);
    if (!text(c.fixes?.location?.default)) out.push(`${id}: no fixes.location.default`);
    if (!text(c.fixes?.movement?.default)) out.push(`${id}: no fixes.movement.default`);
    if (!Array.isArray(c.commonMistakes) || !c.commonMistakes.length || !c.commonMistakes.every(text)) out.push(`${id}: no commonMistakes`);
    for (const n of content?.neighbours?.[id] ?? []) {
      if (n === id || n === "none") out.push(`${id}: neighbour ${n} is not another sign`);
      else if (known && !known.has(n)) out.push(`${id}: neighbour ${n} is not a label of the sign model`);
    }
  }
  const taught = new Set(signs);
  (lesson?.conversation ?? []).forEach((turn, k) => {
    if (!text(turn?.partner)) out.push(`turn ${k + 1}: no partner line`);
    if (!text(turn?.english)) out.push(`turn ${k + 1}: no english`);
    const reply = Array.isArray(turn?.reply) ? turn.reply : [];
    if (!reply.length) out.push(`turn ${k + 1}: no reply`);
    reply.forEach((id, j) => {
      if (!taught.has(id)) out.push(`turn ${k + 1}: ${id} is not taught in the lesson`);
      if (j && reply[j - 1] === id) out.push(`turn ${k + 1}: ${id} twice in a row`);
    });
  });
  return out;
}
