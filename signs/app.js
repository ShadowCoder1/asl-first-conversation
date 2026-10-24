/* signs/app.js: First Conversation, a beginner ASL lesson (docs/signs-design.md). The owner, 23 Oct: "actually
 * teaching ppl how to communicate in asl. very basic. just a few signs ... smooth and work incredible."
 *
 * The learner meets 7 one-handed signs one at a time (a demonstration first, then their own try on camera, told at once
 * whether it was the sign and, if not, which part to change), then uses them: Maya writes, the learner answers in ASL,
 * and the reply's signs light up one by one as each is recognised. Phone first (held in the other hand at arm's length,
 * as PopSign's signers held theirs), laptop too.
 *
 * Screens: welcome -> the lesson (one screen with the camera: the camera check, then each sign's demonstration card
 * and try, then the chat) -> results. The camera is opened once, in the Start tap, and never moves; cards come and go
 * over it, and while one covers it nothing is tracked.
 *
 * The parts:
 *   signs/capture.js       the camera, HandLandmarker + PoseLandmarker, frame records, the drawing
 *   the recognizer         signs/recognizer.js's events (attempt-start, progress, accept, sequence-done, miss, unclear,
 *                          hint); the mock only with ?mock=, behind ONE import below
 *   signs/lesson.js        the lesson: its signs and the conversation's turns
 *   signs/content.json     each sign's words: meaning, how to make it, a tip, its parts, the corrections
 *   the demonstrations     real Deaf signers' PopSign clips of each sign (CC BY 4.0, credited under the clip): the
 *                          lesson's signs published with the site (signs/media/popsign, tools/build-signs-site.sh), and
 *                          on this Mac every sign (signs/local -> ~/Projects/handful-local-media, never committed)
 * Until lesson.js and content.json are in, the lesson runs on the stub below; a sign with no clip shows its name on a
 * plain tile. (The drawn signer that stood in for the clips is gone: the owner, 24 Oct, "remove that drawing version
 * completely ... just keep the actual videos".)
 *
 * URL: ?hand=left (sign with the left hand), ?mock=auto (the mock accepts ~2 s of movement; &mockmiss=N: the first N
 *      tries of each sign are misses), ?fps=1 (frame rate and per-frame cost on the camera view), ?autostart=1 (past
 *      the welcome, for driving the page where nothing can tap), ?srcmirror=1 (a camera that sends a mirrored
 *      picture; js/core/tracker.js), ?signmodel=<path> (localhost only, with the real recognizer: another model file). */

import { createCapture, bodyInView, wristOffset, decideFlip } from "./capture.js";
import { handSide } from "../tutor/pick-hand.js";
/* THE RECOGNIZER: signs/recognizer.js with the sign model (signs/model/signs.onnx), which the app downloads with the
 * feedback's data itself (makeRecognizer below); the stand-in (mock-recognizer.js, MOCK = true) only with ?mock=. */
// ?mock=... keeps the stand-in (design reviews, tests); otherwise the real sign model.
const recognition = await import(new URLSearchParams(location.search).has("mock") ? "./mock-recognizer.js" : "./recognizer.js");

const $ = (s) => document.querySelector(s);
const params = new URLSearchParams(location.search);
const HAND = params.get("hand") === "left" ? "left" : "right";
const OTHER = HAND === "right" ? "left" : "right";
const SHOW_PERF = params.get("fps") === "1";
const MOCK = recognition.MOCK === true;
const MOCK_MODE = params.get("mock") === "auto" ? "auto" : "keys";
let PARTNER = "Maya";
const SOUND_KEY = "signs-sound";
const AUTOSTART_KEY = "signs-autostart";
if (HAND === "left") document.documentElement.classList.add("hand-left");

/* How long things take. A made sign stays green this long before the next one (the celebration, then a breath). */
const ACCEPT_HOLD_MS = 1600;
/* A third miss in a row: the feedback is read for this long, then the demonstration comes back, slowed, with a skip. */
const MISS_TO_DEMO_MS = 2600;
const MISSES_BEFORE_DEMO = 3;
/* A miss's words stay up at least this long, even when the next try starts straight away. */
const MISS_HOLD_MS = 2000;
/* No try for this long: say how a try starts (the recognizer needs the hand down or still first). */
const IDLE_HINT_MS = 9000;
/* A sign not made after this long can be skipped, so nobody is stuck on one. */
const SKIP_AFTER_MS = 25000;
/* Maya "typing", then her message. */
const TYPING_MS = 850;
const TURN_DONE_MS = 1300;

const stats = window.__signs = {
  screen: "welcome", phase: null, mock: MOCK ? MOCK_MODE : null, loaded: 0, flip: null, flipInfo: null,
  signs: [], turns: [], events: [], errors: [], stub: [], demo: null, capture: null,
};
addEventListener("error", (e) => stats.errors.push(String(e.message || e)));
addEventListener("unhandledrejection", (e) => stats.errors.push(String(e.reason?.message || e.reason)));

/* ---- what kind of phone and browser (mobile/app.js) ----------------------------------------------------------- */
const UA = navigator.userAgent || "";
const IOS = /iPad|iPhone|iPod/.test(UA) || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
const ANDROID = /Android/i.test(UA);
const IN_APP = (() => {
  const plain = (IOS && !/Safari\//.test(UA) && !navigator.standalone) || (ANDROID && /; wv\)/.test(UA));
  const m = UA.match(/\b(FBAN|FBAV|FB_IAB|Instagram|LinkedInApp|Snapchat|TikTok|musical_ly|BytedanceWebview|Line\/|KAKAOTALK|MicroMessenger|Pinterest|Twitter)\b/i);
  if (!m) return plain ? "this app" : null;
  const k = m[1].toLowerCase();
  return k.startsWith("fb") ? "Facebook" : k === "instagram" ? "Instagram" : k === "linkedinapp" ? "LinkedIn" : k.includes("tiktok") || k.includes("musical") || k.includes("bytedance") ? "TikTok" : "this app";
})();
const BROWSER = IOS ? "Safari" : ANDROID ? "Chrome" : "your browser";
const PHONE = IOS || ANDROID;
if (PHONE) document.documentElement.classList.add("is-phone");
/* A phone's front camera never sends a mirror image: a ?srcmirror=1 made on a laptop would flip every phone's hand. */
if (PHONE && params.get("srcmirror") === "1") {
  const q = new URLSearchParams(location.search);
  q.set("srcmirror", "0");
  location.replace(`?${q}`);
}
const REDUCED_MOTION = matchMedia("(prefers-reduced-motion: reduce)").matches;
/* The link to pass on: without this device's own settings or the testing options. */
function shareableUrl() {
  const q = new URLSearchParams(location.search);
  for (const k of ["srcmirror", "autostart", "fps", "mock", "mockmiss", "delegate"]) q.delete(k);
  const qs = q.toString();
  return `${location.origin}${location.pathname}${qs ? `?${qs}` : ""}`;
}
const esc = (v) => String(v ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/* ---- the lesson's words ---------------------------------------------------------------------------------------- */
/* Used until signs/lesson.js and signs/content.json are in (they are written alongside this file), and for any sign
 * content.json leaves out. The words follow docs/signs-design.md; content.json's are the ones to trust. */
const STUB_LESSON = Object.freeze({
  title: "Your first ASL conversation",
  intro: "Learn 7 signs, then use them in a real back-and-forth.",
  signs: ["hello", "yes", "no", "please", "thankyou", "water", "bye"],
  conversation: [
    { partner: "Hi!", reply: ["hello"], english: "Hello!" },
    { partner: "Are you thirsty?", reply: ["yes"], english: "Yes." },
    { partner: "What would you like?", reply: ["water", "please"], english: "Water, please." },
    { partner: "Here you go!", reply: ["thankyou"], english: "Thank you!" },
    { partner: "Anything else?", reply: ["no", "thankyou"], english: "No, thank you." },
    { partner: "See you!", reply: ["bye"], english: "Bye!" },
  ],
});
const STUB_CONTENT = Object.freeze({ signs: {
  hello: { english: "hello", gloss: "HELLO", howTo: "Flat hand at the side of your forehead, then move it out and away, like a small salute.",
    tip: "Fingers together, palm facing out.", parts: { handshape: "Flat hand", location: "Forehead", movement: "Out and away" } },
  yes: { english: "yes", gloss: "YES", howTo: "Make a fist and nod it down and up, the way a head nods.",
    tip: "Nod from the wrist, two or three times.", parts: { handshape: "Fist", location: "In front of your shoulder", movement: "Nod down and up" } },
  no: { english: "no", gloss: "NO", howTo: "Index and middle fingers out together, then snap them shut onto your thumb.",
    tip: "Snap twice, quickly.", parts: { handshape: "Two fingers and thumb", location: "In front of your shoulder", movement: "Snap shut" } },
  please: { english: "please", gloss: "PLEASE", howTo: "Flat hand on your chest, palm in, and rub a small circle.",
    tip: "Keep it on the middle of your chest.", parts: { handshape: "Flat hand", location: "Chest", movement: "Small circle" } },
  thankyou: { english: "thank you", gloss: "THANK-YOU", howTo: "Fingertips of a flat hand at your chin, then move your hand forward and down.",
    tip: "Finish with your palm up, toward the person you thank.", parts: { handshape: "Flat hand", location: "Chin", movement: "Forward and down" } },
  water: { english: "water", gloss: "WATER", howTo: "Make a W (three fingers up) and tap your index finger on your chin twice.",
    tip: "Your thumb holds down your little finger.", parts: { handshape: "W hand", location: "Chin", movement: "Tap twice" } },
  bye: { english: "bye", gloss: "BYE", howTo: "Open hand, palm out, and fold your fingers down and up together: a small wave.",
    tip: "Keep your hand still; only the fingers move.", parts: { handshape: "Open hand", location: "Beside your shoulder", movement: "Fingers fold and open" } },
} });
let LESSON = STUB_LESSON;
let CONTENT = STUB_CONTENT;
function usableLesson(l) {
  return l && Array.isArray(l.signs) && l.signs.length && l.signs.every((s) => typeof s === "string")
    && Array.isArray(l.conversation) && l.conversation.every((t) => t && typeof t.partner === "string" && Array.isArray(t.reply) && t.reply.length);
}
async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res.json();
}
/* The credit the clips' licence asks for (signs/lesson.js has the words; these until it is in). */
let CREDIT = {
  video: "Sign videos: PopSign ASL v1.0 (Starner et al., 2023), CC BY 4.0. Clips were trimmed, scaled and re-encoded.",
};
let lessonModule = null;
const contentReady = (async () => {
  const [mod, content] = await Promise.all([
    import("./lesson.js").catch(() => null),
    fetchJson(new URL("content.json", import.meta.url)).catch(() => null),
  ]);
  lessonModule = mod;
  const lesson = mod?.LESSON ?? mod?.default ?? null;
  if (usableLesson(lesson)) LESSON = lesson; else stats.stub.push("lesson");
  if (typeof LESSON.partner?.name === "string" && LESSON.partner.name) PARTNER = LESSON.partner.name;
  if (typeof mod?.ATTRIBUTION_VIDEO === "string") CREDIT.video = mod.ATTRIBUTION_VIDEO;
  if (content?.signs && typeof content.signs === "object") {
    // a sign content.json leaves out keeps the stub's words rather than none
    CONTENT = { ...content, signs: { ...STUB_CONTENT.signs, ...content.signs } };
  } else stats.stub.push("content");
  stats.lesson = { signs: LESSON.signs, turns: LESSON.conversation.length, partner: PARTNER };
})();
const signOf = (id) => CONTENT.signs?.[id] ?? STUB_CONTENT.signs[id] ?? {};
/* A label's gloss for the screen (signs/lesson.js signGloss): the lesson sign's own, then content.glosses (the other
 * labels a miss may be named as), then the label in capitals. */
const glossOf = (id) => CONTENT.signs?.[id]?.gloss ?? CONTENT.glosses?.[id] ?? STUB_CONTENT.signs[id]?.gloss ?? String(id).toUpperCase();
const englishOf = (id) => signOf(id).english ?? String(id);
/* What a sign means, where its name does not say it already (HELLO is "hello", THANK-YOU "thank you"): shown under the
 * name only then. */
function meaningOf(id) {
  const plain = (s) => String(s).toLowerCase().replace(/[^a-z]/g, "");
  const e = englishOf(id);
  return plain(e) === plain(glossOf(id)) ? "" : e;
}
const TOTAL_STEPS = () => LESSON.signs.length + LESSON.conversation.length;

/* ---- demonstrations ------------------------------------------------------------------------------------------- */
/* Real Deaf signers' PopSign clips (CC BY 4.0), mirrored when the signer used the other hand: the lesson's signs as
 * published with the site (signs/media/popsign, tools/build-signs-site.sh; the owner, 24 Oct: "i wanted to see those
 * not these animations"), and on this Mac every sign (signs/local, git-ignored). */
const LOCAL = /^(localhost|127\.0\.0\.1|\[::1\])$/.test(location.hostname);
let localClips = null, clipsOnThisComputer = false;
const localReady = (async () => {
  for (const base of LOCAL ? ["local/popsign", "media/popsign"] : ["media/popsign"]) {
    try {
      const m = await fetchJson(new URL(`${base}/manifest.json`, import.meta.url));
      clipsOnThisComputer = base.startsWith("local");
      return [m, base];
    } catch { /* the next place, or no clips */ }
  }
  throw new Error("no clips");
})().then(([m, base]) => {
  localClips = new Map();
  for (const s of m.signs ?? []) {
    const clips = (s.clips ?? []).map((c) => ({ ...c, url: new URL(`${base}/${c.file}`, import.meta.url).href }));
    // the learner's own hand first (those play as they were filmed), then the clips of the demo frame's own 3:4 shape
    const shapeOff = (c) => (c.w > 0 && c.h > 0 ? Math.abs(c.w / c.h - 0.75) : 1);
    clips.sort((a, b) => ((a.hand === HAND ? 0 : 1) - (b.hand === HAND ? 0 : 1)) || shapeOff(a) - shapeOff(b));
    if (clips.length) localClips.set(s.id, clips);
  }
  return localClips;
}).catch(() => null);
const clipTurn = new Map();
const HAND_SVG = `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 13V5.5a1.5 1.5 0 0 1 3 0V12m0-1V4a1.5 1.5 0 0 1 3 0v7m0-.5V5.5a1.5 1.5 0 0 1 3 0V14c0 4-2.5 7-6.5 7S6 18.5 5 16l-1.6-3.3a1.4 1.4 0 0 1 2.4-1.4L8 14"/></svg>`;

/**
 * One demonstration in `box`: a signer's clip, or the sign's name on a plain tile when no clip could be had.
 * @returns {Promise<{kind: string, setSlow(b: boolean): void, destroy(): void, hasVideo: boolean}>}
 */
async function mountDemo(box, id, { slow = false, small = false } = {}) {
  const token = String(Math.random());
  box.dataset.token = token;
  const clips = await localReady.then(() => localClips?.get(id) ?? null);
  const stale = () => box.dataset.token !== token;
  const hasVideo = !!clips?.length;
  const handle = { kind: "placeholder", hasVideo, setSlow() {}, destroy() {} };
  if (stale()) return handle;
  box.replaceChildren();
  if (hasVideo) {
    const clip = clips[(clipTurn.get(id) ?? 0) % clips.length];
    const v = document.createElement("video");
    v.muted = true; v.loop = true; v.playsInline = true; v.autoplay = true; v.preload = "auto";
    v.setAttribute("muted", ""); v.setAttribute("playsinline", ""); v.disablePictureInPicture = true;
    if (clip.hand && clip.hand !== HAND) v.classList.add("mirrored");
    const rate = () => { v.playbackRate = slowNow ? 0.5 : 1; };
    let slowNow = slow;
    v.addEventListener("loadedmetadata", rate);
    v.src = clip.url;
    box.append(v);
    rate();
    v.play().catch(() => {});
    Object.assign(handle, {
      kind: "video",
      setSlow(b) { slowNow = !!b; rate(); },
      destroy() { v.pause(); v.removeAttribute("src"); v.load(); v.remove(); },
    });
  } else {
    const tile = document.createElement("div");
    tile.className = "demo-placeholder";
    tile.innerHTML = `${HAND_SVG}<b>${esc(glossOf(id))}</b>${small ? "" : "<span>The video didn't load</span>"}`;
    box.append(tile);
    handle.destroy = () => tile.remove();
  }
  stats.demo = handle.kind;
  return handle;
}

/* ---- sound and a buzz (mobile/app.js) ------------------------------------------------------------------------ */
let audio = null;
let soundOn = (() => { try { return localStorage.getItem(SOUND_KEY) !== "off"; } catch { return true; } })();
function unlockAudio() {
  try {
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return;
    audio ??= new AC();
    if (audio.state === "suspended") audio.resume();
    // iOS opens the audio path on a sound started inside the tap
    const b = audio.createBuffer(1, 1, 22050), s = audio.createBufferSource();
    s.buffer = b; s.connect(audio.destination); s.start(0);
  } catch { audio = null; }
}
function tone(freq, at, dur, peak = 0.16, type = "triangle") {
  const o = audio.createOscillator(), g = audio.createGain();
  o.type = type;
  o.frequency.setValueAtTime(freq, at);
  g.gain.setValueAtTime(0.0001, at);
  g.gain.exponentialRampToValueAtTime(peak, at + 0.012);
  g.gain.exponentialRampToValueAtTime(0.0001, at + dur);
  o.connect(g).connect(audio.destination);
  o.start(at); o.stop(at + dur + 0.02);
}
function chime(kind) {
  if (kind === "made") navigator.vibrate?.(35);
  if (kind === "done") navigator.vibrate?.([30, 70, 40]);
  if (!soundOn || !audio) return;
  try {
    if (audio.state === "suspended") audio.resume();
    const t = audio.currentTime + 0.01;
    if (kind === "made") { tone(784, t, 0.16); tone(1175, t + 0.09, 0.28); }
    else if (kind === "chip") { tone(988, t, 0.18, 0.12); }
    else if (kind === "sent") { tone(659, t, 0.12, 0.11); tone(988, t + 0.08, 0.22, 0.12); }
    else if (kind === "maya") { tone(523, t, 0.09, 0.07, "sine"); tone(784, t + 0.05, 0.12, 0.06, "sine"); }
    else if (kind === "miss") { tone(392, t, 0.14, 0.06, "sine"); tone(349, t + 0.1, 0.2, 0.05, "sine"); }
    else if (kind === "done") { [523, 659, 784, 1047].forEach((f, i) => tone(f, t + i * 0.1, 0.35, 0.13)); }
  } catch { /* no sound is fine */ }
}
function renderSound() { $("#btn-sound").setAttribute("aria-pressed", String(soundOn)); }

/* ---- screens -------------------------------------------------------------------------------------------------- */
function show(screen) {
  document.body.dataset.screen = screen;
  stats.screen = screen;
  window.scrollTo(0, 0);
  if (screen === "results" || screen === "error") $(`#${screen} h1`)?.focus({ preventScroll: true });
  if (screen === "lesson") requestAnimationFrame(layout);
}
function phase(p) {
  document.body.dataset.phase = p;
  stats.phase = p;
}
let toastTimer = null;
function toast(text) {
  const t = $("#toast");
  t.textContent = text;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, 2400);
}
function announce(text) { $("#announce").textContent = text; }
function stageMessage(text) {
  const m = $("#stage-message");
  m.hidden = !text;
  if (text) $("#stage-message-text").textContent = text;
}
const stageEl = $("#stage");
let stageTimer = null;
function stageState(s, backToIdleMs = 0) {
  clearTimeout(stageTimer);
  stageEl.dataset.state = s;
  if (backToIdleMs) stageTimer = setTimeout(() => { stageEl.dataset.state = "idle"; }, backToIdleMs);
}
let stepsDone = 0;
function setProgress(n) {
  stepsDone = n;
  const total = TOTAL_STEPS() || 1;
  const fill = $("#progress-fill");
  fill.style.width = `${Math.round((100 * n) / total)}%`;
  fill.classList.toggle("progress-zero", n === 0);
  const bar = $("#progress");
  bar.setAttribute("aria-valuemax", String(total));
  bar.setAttribute("aria-valuenow", String(n));
}

/* ---- the camera and the trackers -------------------------------------------------------------------------------- */
const video = $("#cam");
const canvas = $("#overlay");
/* Loading: MediaPipe's runtime and both models (capture.js) and, with the real recognizer, the sign model and
 * onnxruntime's WebAssembly, as one progress line weighed by their sizes. */
const loadParts = new Map([["capture", [0, 25.4]]]);
if (!MOCK) loadParts.set("model", [0, 18.5]);
let stageWaiting = false;
function loadProgress(part, frac, text) {
  const w = loadParts.get(part);
  if (!w) return;
  w[0] = Math.max(0, Math.min(1, frac));
  let a = 0, b = 0;
  for (const [f, size] of loadParts.values()) { a += f * size; b += size; }
  const all = b ? a / b : 1;
  stats.loaded = all;
  $("#loadbar-fill").style.width = `${Math.round(all * 100)}%`;
  if (text && all < 1) $("#loadline-text").textContent = text;
  if (all >= 1) $("#loadline").classList.add("loadline-done");
  if (stageWaiting) stageMessage(`Getting the lesson ready… ${Math.round(all * 100)}%`);
}
const capture = createCapture({
  video, hand: HAND, canvas,
  onProgress: (p) => loadProgress("capture", p.frac, p.text === "Ready" ? null : p.text),
  onError: (err) => fail(err),
  onResize: () => layout(),
});
stats.capture = capture.stats;
let loadError = null;
function loadFailed(err) {
  if (loadError) return;
  loadError = err;
  if (stats.screen !== "welcome") return;
  $("#loadline").classList.remove("loadline-done");
  $("#loadline-text").textContent = "Couldn't load the lesson. Check your connection, then try again.";
  $("#btn-start").textContent = "Try again";   // start() reloads
}
capture.ready.catch((err) => loadFailed(err));

/* The camera view's size: as big as the room allows at the camera's own shape (mobile/app.js layout). Upright, it
 * takes the width and as much height as leaves the words their room; on its side and on a laptop it fills the space
 * beside them. */
const PANEL_MIN_PX = 172;
const landscape = () => matchMedia("(orientation: landscape)").matches;
function layout() {
  const main = $("#lesson-main"), wrap = $("#stage-wrap");
  if (!main || !wrap || !main.clientHeight) return;
  const { aspect } = capture.source();
  let W, H;
  if (landscape()) {
    wrap.style.flex = "";
    wrap.style.height = "";
    W = wrap.clientWidth; H = wrap.clientHeight;
  } else {
    const cs = getComputedStyle(main);
    W = main.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
    H = main.clientHeight - parseFloat(cs.paddingBottom) - parseFloat(cs.rowGap || cs.gap || 0) - PANEL_MIN_PX;
  }
  if (!(W > 0 && H > 0)) return;
  let w = W, h = W / aspect;
  if (h > H) { h = H; w = H * aspect; }
  w = Math.floor(w); h = Math.floor(h);
  if (!landscape()) { wrap.style.flex = "0 0 auto"; wrap.style.height = `${h}px`; }
  stageEl.style.aspectRatio = "auto";
  stageEl.style.width = `${w}px`;
  stageEl.style.height = `${h}px`;
}
new ResizeObserver(() => layout()).observe($("#lesson-main"));
addEventListener("orientationchange", () => setTimeout(layout, 250));

/* ---- the recognizer --------------------------------------------------------------------------------------------- */
const met = new Set();
let rec = null;
let onRec = null;
let routeFrames = false;
let frameHook = null;
async function makeRecognizer() {
  await contentReady;
  const labels = (await fetchJson(new URL("model/labels.json", import.meta.url))).labels;
  // what would stop this lesson being taught with this content and model (signs/lesson.js): said, not shown
  try {
    const problems = lessonModule?.lessonProblems?.(LESSON, CONTENT, labels) ?? [];
    if (problems.length) { stats.problems = problems; console.warn(`signs: the lesson's content has problems: ${problems.join("; ")}`); }
  } catch { /* the check itself is optional */ }
  if (MOCK) {
    return recognition.createRecognizer({
      labels, hand: HAND, content: CONTENT,
      mock: { mode: MOCK_MODE, missFirst: Number(params.get("mockmiss")) || 0 },
    });
  }
  // The sign model (in a Worker) and what a miss is measured against: the Deaf signers' templates and the letter
  // grader's hand-shape scale (signs/feedback.js). A miss's frames are put on the 30 fps clock first, as the model's.
  const [{ createModelRunner }, { modelInput, resampleToFps }, { createFeedback }] = await Promise.all([
    import("./model-runner.js"), import("./features.js"), import("./feedback.js"),
  ]);
  // ?signmodel=<path> on this Mac only: another model file (a smoke model, a candidate run) without touching the app
  const modelPath = LOCAL && params.get("signmodel") ? new URL(params.get("signmodel"), location.href) : new URL("model/signs.onnx", import.meta.url);
  const runner = createModelRunner({
    modelUrl: modelPath,
    onProgress: (p) => loadProgress("model", p.frac, "Getting the sign model ready…"),
  });
  const [templates, grader, cuts] = await Promise.all([
    fetchJson(new URL("model/templates.json", import.meta.url)),
    fetchJson(new URL("../tutor/model.json", import.meta.url)),
    fetchJson(new URL("model/cuts.json", import.meta.url)).catch(() => ({})),
  ]);
  const words = createFeedback({ templates, content: CONTENT, labels, scale: grader.scale });
  let r = null;
  const feedbackFn = (frames, sign, probs, ctx) => {
    const f = resampleToFps(frames, 30);
    return words({ hands: f.map((x) => x.hand ?? null), poses: f.map((x) => x.pose ?? null), aspect: f.map((x) => x.aspect),
      flip: ctx?.flip ?? r?.flip ?? HAND === "left" }, sign, probs);
  };
  r = recognition.createRecognizer({
    runner, featuresFn: modelInput, labels, cuts: cuts?.cuts ?? cuts, hand: HAND, feedbackFn,
    // "That looked like X" names only a sign the learner has met, or a known neighbour of one (content.json
    // neighbours: signs that share a hand shape, place or movement with it)
    looksLikeOk: (s) => met.has(s) || [...met].some((m) => CONTENT.neighbours?.[m]?.includes(s)),
  });
  await runner.ready;
  return r;
}
const recReady = makeRecognizer();
recReady.catch((err) => loadFailed(err));
function listen(r) {
  r.on((e) => {
    stats.events.push({ type: e.type, sign: e.sign ?? null, index: e.index ?? null, reason: e.reason ?? null, t: Math.round(performance.now()),
      confidence: e.confidence ?? null, looksLike: e.looksLike ?? null, top: e.top ? [...e.top].slice(0, 3) : null, early: e.early ?? null });
    if (stats.events.length > 400) stats.events.splice(0, 200);
    onRec?.(e);
  });
}
capture.onFrame((r, raw) => {
  frameHook?.(r, raw);
  if (routeFrames && rec) rec.pushFrame(r);
});
function target(t) {
  try { rec.setTarget(t); }
  catch (err) { fail(new Error(`The lesson asked for a sign the recognizer does not know (${err?.message || err}).`)); }
}

/* ---- welcome ---------------------------------------------------------------------------------------------------- */
async function renderWelcome() {
  for (const el of document.querySelectorAll(".hand-word")) el.textContent = HAND;
  for (const el of document.querySelectorAll(".other-hand-word")) el.textContent = OTHER;
  if (!PHONE) {
    $("#step-phone").textContent = "Sit about an arm's length from the camera, so your head and shoulders fit.";
    // a laptop, not a phone
    $("#step-phone").previousElementSibling.innerHTML = `<svg viewBox="0 0 24 24"><rect x="4" y="5" width="16" height="11" rx="1.5"/><path d="M2.5 19h19"/><circle cx="12" cy="10.5" r="1.6"/></svg>`;
  }
  $("#privacy-line").textContent = `Video stays on your ${PHONE ? "phone" : "computer"}. Nothing is recorded or saved.`;
  if (IN_APP) { $("#inapp-browser").textContent = BROWSER; $("#inapp-note").hidden = false; }
  await contentReady;
  if (LESSON.title) $("#welcome-title").textContent = LESSON.title;
  if (LESSON.intro) $("#welcome-lede").textContent = LESSON.intro;
  $("#interlude-title").textContent = `${LESSON.signs.length} signs learned!`;
}
/* The welcome's three signers: stills from the lesson's own clips, made with the site (tools/build-signs-site.sh; on
 * this Mac, tools/signs/make_stills.py into signs/local). A picture that will not load takes the row with it. */
function heroPictures() {
  const hero = $("#hero");
  for (const img of hero.querySelectorAll("img")) {
    const missing = () => {
      const local = new URL(`local/popsign/stills/${img.dataset.sign}.jpg`, import.meta.url).href;
      if (LOCAL && img.src !== local) img.src = local;
      else hero.hidden = true;
    };
    img.addEventListener("error", missing);
    if (img.complete && !img.naturalWidth) missing();
  }
}

/* ---- the camera check ----------------------------------------------------------------------------------------- */
/* Head and shoulders in view for a moment, then the signing hand up for a moment (which side of the body it is on
 * decides the session's flip, capture.js decideFlip), then "Great! Now lower your hand to begin." (the letter apps'
 * words): a hand still up would be read as the start of the first sign. */
const CHECK = Object.freeze({ bodyHoldMs: 600, handHoldMs: 1000, graceMs: 250, hintMs: 8000, anywayMs: 15000, lowerGoneMs: 300, lowerMaxMs: 4000, otherHandMs: 1500 });
const NO_HAND_HINT = "We can't see a hand yet. Move somewhere brighter, and keep your whole hand in the picture.";
function cameraCheck() {
  phase("check");
  const status = $("#check-status"), guide = $("#guide"), actions = $("#check-actions");
  const say = (text, tone = "") => {
    if (status.textContent !== text) status.textContent = text;
    const c = `status${tone ? ` ${tone}` : ""}`;
    if (status.className !== c) status.className = c;
  };
  const step = (id, state) => { $(id).className = state; };
  $("#chk-hand-text").textContent = `Raise your ${HAND} hand`;
  step("#chk-body", "now"); step("#chk-hand", "");
  guide.classList.add("guide-on");
  actions.replaceChildren();
  const button = (text, onClick) => {
    const b = document.createElement("button");
    b.type = "button"; b.className = "btn btn-ghost"; b.textContent = text;
    b.addEventListener("click", onClick);
    actions.append(b);
    return b;
  };
  return new Promise((resolve) => {
    let stage = "body", stageAt = performance.now(), lastHandAt = performance.now();
    let bodySince = null, bodySeenAt = -Infinity, handSince = null, handSeenAt = -Infinity, objectionSince = null, goneAt = null;
    let anyway = null, offer = null;
    const offsets = [];
    const toHand = () => {
      stage = "hand"; stageAt = performance.now();
      step("#chk-body", "ok"); step("#chk-hand", "now");
      guide.classList.remove("guide-on");
      anyway?.remove(); anyway = null;
    };
    let finished = false;
    const finish = () => {
      if (finished) return;
      finished = true;
      frameHook = null;
      actions.replaceChildren();
      resolve(decideFlip(offsets, HAND === "left"));
    };
    frameHook = (r, raw) => {
      const now = performance.now();
      if (stage === "body") {
        const b = bodyInView(r.pose);
        const ok = b.head && b.shoulders;
        if (ok) { bodySeenAt = now; bodySince ??= now; } else if (now - bodySeenAt > CHECK.graceMs) bodySince = null;
        // the outline is where to go: once the body is there, it is in the way
        guide.classList.toggle("guide-on", now - bodySeenAt > 800);
        if (bodySince !== null && now - bodySince >= CHECK.bodyHoldMs) { toHand(); return; }
        if (ok) say("Hold it there…", "good");
        else if (!r.pose) say(PHONE ? "Hold your phone at arm's length, so your head and shoulders fit." : "Sit back a little, so your head and shoulders fit.");
        else if (!b.head) say("Move so your whole face is in the picture.");
        else say(PHONE ? "Hold the phone a little further away, so both shoulders show." : "Sit back a little, so both shoulders show.");
        // a body the tracker never finds (a dark room, a phone too close) must not be a dead end
        if (!anyway && now - stageAt > CHECK.anywayMs) anyway = button("Continue anyway", toHand);
        return;
      }
      if (stage === "hand") {
        let objection = false;
        // one hand in view that is plainly the other one (its shape, then MediaPipe's label): say so, as the letter
        // apps do; with both hands up, the picker has already taken the signing one
        if (r.hand && !r.other && raw?.k >= 0) {
          try { objection = handSide(raw.handRes.landmarks[raw.k], raw.handRes.handedness?.[raw.k], r.aspect) === OTHER; }
          catch { objection = false; }
        }
        if (r.hand) lastHandAt = now;
        objectionSince = objection ? objectionSince ?? now : null;
        if (objectionSince !== null && now - objectionSince >= CHECK.otherHandMs && !offer) {
          offer = button(`I sign with my ${OTHER} hand`, () => restartWith({ hand: OTHER }));
          // a laptop camera that sends a mirror image shows a right hand as a left one (the owner's Mac does)
          if (!PHONE && params.get("srcmirror") !== "1") button(`This is my ${HAND} hand`, () => restartWith({ srcmirror: "1" }));
        }
        const good = !!r.hand && !objection;
        if (good) {
          handSeenAt = now;
          if (handSince === null) { handSince = now; offsets.length = 0; }
          offsets.push(wristOffset(r));
        } else if (now - handSeenAt > CHECK.graceMs) handSince = null;
        if (handSince !== null && now - handSince >= CHECK.handHoldMs) {
          stage = "lower"; stageAt = now;
          step("#chk-hand", "ok");
          actions.replaceChildren(); offer = null;
          say("Great! Now lower your hand to begin.", "good");
          announce("Great! Now lower your hand to begin.");
          return;
        }
        if (objection) say(PHONE ? `That looks like your ${OTHER} hand. Hold the phone in your ${OTHER} hand, and raise your ${HAND}.` : `That looks like your ${OTHER} hand. Raise your ${HAND} hand.`, "bad");
        else if (good) say("Hold it there…", "good");
        else if (now - lastHandAt > CHECK.hintMs) say(NO_HAND_HINT);
        else say(`Raise your ${HAND} hand beside your face.`);
        if (!anyway && now - stageAt > CHECK.anywayMs) anyway = button("Continue anyway", () => { stage = "lower"; stageAt = performance.now(); finish(); });
        return;
      }
      // lowering: gone for a moment, or at most a few seconds
      if (!r.hand) { goneAt ??= now; if (now - goneAt >= CHECK.lowerGoneMs) finish(); }
      else goneAt = null;
      if (stage === "lower" && now - stageAt >= CHECK.lowerMaxMs) finish();
    };
  });
}
/* The same link with ?hand= or ?srcmirror= changed, straight back into the lesson after one tap. */
function restartWith(change) {
  const q = new URLSearchParams(location.search);
  for (const [k, v] of Object.entries(change)) q.set(k, v);
  try { sessionStorage.setItem(AUTOSTART_KEY, "1"); } catch { /* the welcome again */ }
  capture.stop();
  location.href = `?${q}`;
}

/* ---- learning one sign: watch it, then make it ------------------------------------------------------------------ */
const watchEl = $("#watch");
let watchDemo = null;
let watchCard = 0;
let pipDemo = null;
let pipSlow = false;
let pipSign = null;
/* A sign's parts as a teacher names them (content.json parts), by the names the miss checklist later uses. */
const PART_NAMES = { handshape: "Hand shape", location: "Place", movement: "Movement", orientation: "Palm" };
function renderParts(parts) {
  if (!parts || typeof parts !== "object" || Array.isArray(parts)) return "";
  return Object.keys(PART_NAMES).filter((k) => typeof parts[k] === "string" && parts[k])
    .map((k) => `<li><span>${PART_NAMES[k]}</span><p>${esc(parts[k])}</p></li>`).join("");
}
/* The sign's card over the camera: its demonstration (looping), what it means, how to make it, its parts, a tip. mode
 * "first" (a new sign), "rewatch" (asked for again), "again" (after three misses: slowed, and it can be skipped).
 * Resolves "try" or "skip". */
function showWatch(id, i, mode = "first") {
  phase("watch");
  capture.pause();
  hidePip();
  routeFrames = false;
  const s = signOf(id);
  const n = LESSON.signs.length;
  const step = $("#watch-step");
  step.textContent = mode === "again" ? "Let's watch it slowly" : mode === "rewatch" ? "Watch it again" : `Sign ${i + 1} of ${n}`;
  step.classList.toggle("watch-step-again", mode === "again");
  $("#watch-gloss").textContent = glossOf(id);
  const meaning = meaningOf(id);
  $("#watch-english").textContent = meaning ? `“${meaning}”` : "";
  $("#watch-english").hidden = !meaning;
  $("#watch-howto").textContent = s.howTo ?? "";
  $("#watch-parts").innerHTML = renderParts(s.parts);
  const tip = $("#watch-tip");
  tip.hidden = !s.tip;
  if (s.tip) tip.innerHTML = `<b>Tip:</b> ${esc(s.tip)}`;
  $("#btn-try").textContent = mode === "first" ? "Try it" : "Try again";
  $("#btn-watch-skip").hidden = mode !== "again";
  let slow = mode === "again";
  const slowBtn = $("#btn-slow");
  slowBtn.setAttribute("aria-pressed", String(slow));
  const media = $("#demo-media");
  const credit = $("#demo-credit");
  // a demo still loading when the card closes (a quick tap on Try) must not start playing under the camera
  const card = ++watchCard;
  const mount = async () => {
    watchDemo?.destroy();
    watchDemo = null;
    const d = await mountDemo(media, id, { slow });
    if (card !== watchCard) { d.destroy(); return; }
    watchDemo = d;
    $("#btn-signer").hidden = d.kind !== "video" || (localClips?.get(id)?.length ?? 0) < 2;
    credit.hidden = d.kind !== "video";
    credit.textContent = d.kind === "video" ? `${CREDIT.video}${clipsOnThisComputer ? " Shown on this computer only." : ""}` : "";
  };
  mount();
  watchEl.hidden = false;
  watchEl.classList.remove("watch-out");
  watchEl.classList.add("watch-in");
  $("#watch .watch-card").scrollTop = 0;
  announce(`New sign: ${glossOf(id)}, meaning ${englishOf(id)}. ${s.howTo ?? ""}`);
  setTimeout(() => $("#btn-try").focus({ preventScroll: true }), 50);
  return new Promise((resolve) => {
    const offs = [];
    const on = (el, type, fn) => { el.addEventListener(type, fn); offs.push(() => el.removeEventListener(type, fn)); };
    const done = (choice) => {
      offs.forEach((f) => f());
      watchCard++;
      watchDemo?.destroy();
      watchDemo = null;
      watchEl.classList.remove("watch-in");
      if (REDUCED_MOTION) watchEl.hidden = true;
      else {
        watchEl.classList.add("watch-out");
        setTimeout(() => { if (watchEl.classList.contains("watch-out")) { watchEl.hidden = true; watchEl.classList.remove("watch-out"); } }, 260);
      }
      resolve(choice);
    };
    on($("#btn-try"), "click", () => done("try"));
    on($("#btn-watch-skip"), "click", () => done("skip"));
    on(slowBtn, "click", () => {
      slow = !slow;
      slowBtn.setAttribute("aria-pressed", String(slow));
      watchDemo?.setSlow(slow);
    });
    on($("#btn-signer"), "click", () => { clipTurn.set(id, (clipTurn.get(id) ?? 0) + 1); mount(); });
  });
}
/* The demonstration, small, in the corner away from the signing hand while the learner tries. A tap slows it. */
async function showPip(id) {
  const pip = $("#pip");
  if (pipSign === id && !pip.hidden) return;
  hidePip();
  pipSign = id;
  pipSlow = false;
  pip.classList.remove("pip-slow");
  pip.setAttribute("aria-label", `Demonstration of ${glossOf(id)}. Tap for slow motion.`);
  pip.hidden = false;
  const d = await mountDemo($("#pip-media"), id, { small: true });
  if (pipSign !== id || pip.hidden) { d.destroy(); return; }
  pipDemo = d;
}
function hidePip() {
  pipDemo?.destroy();
  pipDemo = null;
  pipSign = null;
  $("#pip").hidden = true;
  $("#pip-media").dataset.token = "";
  $("#pip-media").replaceChildren();
}
$("#pip").addEventListener("click", () => {
  pipSlow = !pipSlow;
  $("#pip").classList.toggle("pip-slow", pipSlow);
  pipDemo?.setSlow(pipSlow);
});

function setWord(id) {
  const pill = $("#word-pill");
  if (!id) { pill.hidden = true; return; }
  $("#word-gloss").textContent = glossOf(id);
  pill.hidden = false;
  pill.classList.remove("word-in");
  void pill.offsetWidth;
  pill.classList.add("word-in");
}
const tryCard = $("#try-card");
function setTry(state, title, html, { checks = null, looks = null } = {}) {
  tryCard.dataset.state = state;
  $("#try-title").textContent = title;
  $("#try-text").innerHTML = html ?? "";
  const list = $("#try-checks");
  list.hidden = !checks;
  if (checks) for (const li of list.children) li.dataset.ok = String(checks[li.dataset.part] ?? null);
  const lk = $("#try-looks");
  lk.hidden = !looks;
  if (looks) lk.innerHTML = `That looked like <b>${esc(looks)}</b>.`;
  if (!REDUCED_MOTION) { tryCard.classList.remove("swap"); void tryCard.offsetWidth; tryCard.classList.add("swap"); }
}
const PRAISE = ["Nice!", "Great job!", "You got it!", "Perfect!", "Nailed it!", "Beautiful!"];
let praiseTurn = 0;
/* A miss's words: one sentence (feedback.js's, without its own "That looked like ..." which is said apart, from the
 * recognizer's looksLike: it names only signs the learner has met), else the sign's how-to. */
function missWords(e) {
  const text = String(e.feedback?.text ?? "").replace(/^\s*That looked like [^.]*\.\s*/i, "").trim();
  return text || signOf(e.sign).howTo || `Watch the demonstration, then try ${glossOf(e.sign)} again.`;
}
const UNCLEAR_WORDS = {
  "hand-lost": ["We lost your hand", "Keep your whole hand in the picture while you sign."],
  "too-short": ["A little longer", "Make the whole sign, then lower your hand."],
  "out-of-frame": ["Out of the picture", PHONE ? "Your hand left the picture. Hold the phone a little further away." : "Your hand left the picture. Sit back a little."],
};
const unclearWords = (reason) => UNCLEAR_WORDS[reason] ?? ["Let's try that again", "Keep your hand in the picture, and make the whole sign."];

/* One try (or several) at a sign. Resolves "accept", "skip", "watch" (Watch again) or "again" (three misses). */
function trySign(id, res) {
  return new Promise((resolve) => {
    phase("try");
    setWord(id);
    showPip(id);
    capture.calm();
    capture.resume();
    stageState("idle");
    const gloss = glossOf(id);
    const idleText = `Sign <b>${esc(gloss)}</b> to the camera.`;
    setTry("idle", "Your turn", idleText);
    $("#btn-skip").hidden = true;
    target({ mode: "isolated", sign: id });
    routeFrames = true;
    let settled = false, idleTimer = null;
    const armIdle = () => {
      clearTimeout(idleTimer);
      idleTimer = setTimeout(() => {
        if (tryCard.dataset.state === "idle") setTry("idle", "Your turn", `Lower your hand, then sign <b>${esc(gloss)}</b>.`);
      }, IDLE_HINT_MS);
    };
    armIdle();
    const skipTimer = setTimeout(() => { $("#btn-skip").hidden = false; }, SKIP_AFTER_MS);
    const offs = [];
    const on = (el, type, fn) => { el.addEventListener(type, fn); offs.push(() => el.removeEventListener(type, fn)); };
    const done = (outcome) => {
      if (settled) return;
      settled = true;
      clearTimeout(idleTimer); clearTimeout(skipTimer);
      onRec = null;
      routeFrames = false;
      offs.forEach((f) => f());
      resolve(outcome);
    };
    // a correction stays up this long even if the next try starts at once: it is what the learner should read first
    let holdUntil = 0, holdTimer = null;
    const watching = () => setTry("watching", "Watching…", `Sign <b>${esc(gloss)}</b>, then lower your hand.`);
    offs.push(() => clearTimeout(holdTimer));
    onRec = (e) => {
      if (settled || e.sign !== id) return;
      if (e.type !== "attempt-start") clearTimeout(holdTimer);
      if (e.type === "attempt-start") {
        clearTimeout(idleTimer);
        stageState("watching");
        const wait = holdUntil - performance.now();
        if (wait > 0) holdTimer = setTimeout(watching, wait);
        else watching();
      } else if (e.type === "accept") {
        res.tries = res.misses + 1;
        onRec = null;
        capture.celebrate();
        stageState("good");
        const title = res.misses ? "You got it!" : PRAISE[praiseTurn++ % PRAISE.length];
        setTry("good", title, `That's <b>${esc(gloss)}</b>${res.misses ? "" : ", first try"}.`);
        chime("made");
        announce(`${title} That's ${gloss}.`);
        setTimeout(() => done("accept"), ACCEPT_HOLD_MS);
      } else if (e.type === "miss") {
        res.misses++;
        const like = e.looksLike ? glossOf(e.looksLike) : null;
        if (like) res.looks.push(like);
        stageState("miss", 1100);
        const title = res.misses === 1 ? "Not quite" : res.misses === 2 ? "Almost" : "Let's watch it again";
        setTry("miss", title, esc(missWords(e)), { checks: e.feedback?.checks ?? null, looks: like });
        holdUntil = performance.now() + MISS_HOLD_MS;
        chime("miss");
        announce(`${title}. ${missWords(e)}${like ? ` That looked like ${like}.` : ""}`);
        armIdle();
        if (res.misses % MISSES_BEFORE_DEMO === 0) { onRec = null; setTimeout(() => done("again"), MISS_TO_DEMO_MS); }
      } else if (e.type === "unclear") {
        res.unclear++;
        stageState("idle");
        const [title, text] = unclearWords(e.reason);
        setTry("unclear", title, esc(text));
        holdUntil = performance.now() + MISS_HOLD_MS * 0.75;
        announce(`${title}. ${text}`);
        armIdle();
      } else if (e.type === "hint" && tryCard.dataset.state !== "miss" && tryCard.dataset.state !== "watching") {
        // a long wait (the recognizer's own clock): how a try starts, then the sign's own words
        const s = signOf(id);
        setTry("idle", "Your turn", e.count > 1 && s.howTo ? esc(s.howTo) : `Lower your hand, then sign <b>${esc(gloss)}</b>.`);
      }
    };
    on($("#btn-rewatch"), "click", () => done("watch"));
    on($("#btn-skip"), "click", () => done("skip"));
  });
}
async function learnSign(id, i) {
  const res = { id, gloss: glossOf(id), tries: 0, misses: 0, unclear: 0, skipped: false, looks: [], ms: 0 };
  const t0 = performance.now();
  met.add(id);
  let mode = "first";
  for (;;) {
    const choice = await showWatch(id, i, mode);
    if (choice === "skip") { res.skipped = true; break; }
    const outcome = await trySign(id, res);
    if (outcome === "accept") break;
    if (outcome === "skip") { res.skipped = true; break; }
    mode = outcome === "again" ? "again" : "rewatch";
  }
  res.ms = Math.round(performance.now() - t0);
  capture.calm();
  setWord(null);
  hidePip();
  return res;
}

/* ---- the conversation ------------------------------------------------------------------------------------------- */
const chatEl = $("#chat");
let calmTimer = null;
function clearChat() { chatEl.replaceChildren(); }
/* Maya "types", then her message slides in; what was said before slides away, so the camera stays clear. */
async function mayaSays(text) {
  for (const old of [...chatEl.children]) {
    if (REDUCED_MOTION) old.remove();
    else { old.classList.add("msg-out"); setTimeout(() => old.remove(), 230); }
  }
  const row = document.createElement("div");
  row.className = "msg msg-maya msg-in";
  row.innerHTML = `<span class="bubble typing" aria-label="${esc(PARTNER)} is typing"><i></i><i></i><i></i></span>`;
  chatEl.append(row);
  await sleep(REDUCED_MOTION ? 250 : TYPING_MS);
  row.innerHTML = `<span class="bubble">${esc(text)}</span>`;
  row.classList.remove("msg-in");
  void row.offsetWidth;
  row.classList.add("msg-in");
  chime("maya");
  announce(`${PARTNER}: ${text}`);
}
function youSay(turn, skipped) {
  const row = document.createElement("div");
  row.className = "msg msg-you msg-in";
  const glosses = turn.reply.map(glossOf).join(" ");
  row.innerHTML = `<span class="bubble">${esc(skipped ? "(skipped)" : turn.english ?? glosses)}<small>${esc(glosses)}</small></span>`;
  chatEl.append(row);
  announce(`You: ${turn.english ?? glosses}`);
}
function renderChips(reply) {
  $("#chips").innerHTML = reply.map((id, k) => `<span class="chip" data-index="${k}" data-state="${k === 0 ? "next" : "pending"}"><b>${esc(glossOf(id))}</b>${meaningOf(id) ? `<small>${esc(meaningOf(id))}</small>` : ""}</span>`).join("");
}
function chipState(k, s) {
  const c = $(`#chips .chip[data-index="${k}"]`);
  if (c && c.dataset.state !== "done") c.dataset.state = s;
}
function replyHint(html) {
  const h = $("#reply-hint");
  h.hidden = !html;
  h.innerHTML = html ?? "";
}
async function interlude() {
  phase("interlude");
  capture.pause();
  routeFrames = false;
  const el = $("#interlude");
  const badge = el.querySelector(".interlude-badge");
  badge.querySelectorAll(".confetti").forEach((c) => c.remove());
  if (!REDUCED_MOTION) badge.insertAdjacentHTML("beforeend", confetti(16));
  el.hidden = false;
  el.classList.add("watch-in");
  chime("done");
  setTimeout(() => $("#btn-chat").focus({ preventScroll: true }), 50);
  await new Promise((r) => $("#btn-chat").addEventListener("click", r, { once: true }));
  el.hidden = true;
  el.classList.remove("watch-in");
}
async function chatTurn(turn, k) {
  const res = { partner: turn.partner, reply: [...turn.reply], english: turn.english, done: false, skipped: false, hints: 0, misses: 0, ms: 0 };
  phase("chat");
  hidePip();
  replyHint(null);
  $("#reply-card").classList.remove("sent");
  $("#reply-label").textContent = k === 0 ? "Sign your reply" : "Your reply";
  renderChips(turn.reply);
  $("#btn-skip-reply").hidden = true;
  capture.calm();
  capture.resume();
  stageState("idle");
  await mayaSays(turn.partner);
  const t0 = performance.now();
  target({ mode: "sequence", signs: turn.reply });
  routeFrames = true;
  await new Promise((resolve) => {
    let settled = false;
    const skipBtn = $("#btn-skip-reply");
    const finish = () => {
      if (settled) return;
      settled = true;
      onRec = null;
      routeFrames = false;
      skipBtn.removeEventListener("click", onSkip);
      resolve();
    };
    const onSkip = () => { res.skipped = true; finish(); };
    skipBtn.addEventListener("click", onSkip);
    onRec = (e) => {
      if (settled) return;
      if (e.type === "attempt-start") stageState("watching");
      else if (e.type === "progress") chipState(e.index, "now");
      else if (e.type === "accept") {
        if (e.sign !== turn.reply[e.index]) return;
        chipState(e.index, "done");
        if (e.index + 1 < turn.reply.length) chipState(e.index + 1, "next");
        chime("chip");
        capture.celebrate();
        clearTimeout(calmTimer);
        calmTimer = setTimeout(() => capture.calm(), 900);
        stageState("good", 900);
        replyHint(null);
        if (pipSign === turn.reply[e.index]) hidePip();
      } else if (e.type === "sequence-done") {
        res.done = true;
        finish();
      } else if (e.type === "hint") {
        res.hints++;
        const s = signOf(e.sign);
        if (e.count <= 1) replyHint(`Next: <b>${esc(glossOf(e.sign))}</b>. ${esc(s.howTo ?? "")}`);
        else replyHint(`Take your time. Lower your hand, then sign <b>${esc(glossOf(e.sign))}</b>.`);
        showPip(e.sign);
        if (e.count >= 2) skipBtn.hidden = false;
      } else if (e.type === "miss") {
        res.misses++;
        stageState("miss", 1100);
        replyHint(`Not quite. ${esc(missWords(e))}`);
        chime("miss");
      } else if (e.type === "unclear") {
        stageState("idle");
        replyHint(esc(unclearWords(e.reason)[1]));
      }
    };
  });
  res.ms = Math.round(performance.now() - t0);
  clearTimeout(calmTimer);
  hidePip();
  replyHint(null);
  if (!res.skipped) {
    $("#reply-card").classList.add("sent");
    $("#reply-label").textContent = "Sent";
    capture.celebrate();
    stageState("good");
    chime("sent");
  } else {
    for (let i = 0; i < turn.reply.length; i++) chipState(i, "skipped");
  }
  youSay(turn, res.skipped);
  await sleep(TURN_DONE_MS);
  capture.calm();
  stageState("idle");
  return res;
}

/* ---- the lesson, start to end ------------------------------------------------------------------------------------ */
async function lessonFlow() {
  for (let i = 0; i < LESSON.signs.length; i++) {
    const r = await learnSign(LESSON.signs[i], i);
    stats.signs.push(r);
    setProgress(stepsDone + 1);
  }
  await interlude();
  clearChat();
  for (let k = 0; k < LESSON.conversation.length; k++) {
    const r = await chatTurn(LESSON.conversation[k], k);
    stats.turns.push(r);
    setProgress(stepsDone + 1);
  }
  await sleep(500);
  finish();
}

/* ---- the end ---------------------------------------------------------------------------------------------------- */
/* A small burst round a badge (the letter app's results confetti). */
function confetti(n) {
  const colors = ["#1e9e6a", "#f5a524", "#38bdf8", "#a78bfa", "#f472b6", "#34d399"];
  return Array.from({ length: n }, (_, k) => `<i class="confetti" style="--a:${Math.round((k * 360) / n)}deg;--d:${70 + (k % 3) * 18}px;--delay:${120 + (k % 4) * 40}ms;background:${colors[k % colors.length]}"></i>`).join("");
}
const CHECK_SVG = `<svg viewBox="0 0 24 24"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>`;
const CHAT_SVG = `<svg viewBox="0 0 24 24"><path d="M5 5.5h14a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-8l-4.5 3.5v-3.5H5a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2z"/></svg>`;
const SKIP_SVG = `<svg viewBox="0 0 24 24"><path d="M7 12h10"/></svg>`;
function finish() {
  routeFrames = false;
  onRec = null;
  capture.stop();
  releaseAwake();
  phase(null);
  const signs = stats.signs, turns = stats.turns;
  const learned = signs.filter((s) => !s.skipped).length, replied = turns.filter((t) => t.done).length;
  const allDone = learned === signs.length && replied === turns.length;
  $("#results-title").textContent = allDone ? "You had your first ASL conversation!" : learned + replied ? "Nice work!" : "Good start!";
  $("#results-lede").textContent = allDone
    ? `${signs.length} signs learned, and every reply to ${PARTNER} signed.`
    : learned
      ? `You learned ${learned} of ${signs.length} signs and signed ${replied} of ${turns.length} replies.`
      : replied
        ? `You signed ${replied} of ${turns.length} replies. The signs you skipped are here when you want another go.`
        : "Every sign gets easier with practice. Try them again when you're ready.";
  const trophy = $("#trophy");
  trophy.className = `trophy${allDone ? "" : " trophy-partial"}`;
  trophy.innerHTML = `<div class="trophy-disc">${allDone ? CHAT_SVG : CHECK_SVG}</div>` + (learned + replied && !REDUCED_MOTION ? confetti(18) : "");
  $("#sign-results").innerHTML = signs.map((s, k) => {
    const what = s.skipped ? "Skipped" : s.tries <= 1 ? "First try" : `${s.tries} tries`;
    return `<li class="${s.skipped ? "sign-skipped" : "sign-made"}" style="--delay:${150 + k * 70}ms">
        <span class="sign-mark" aria-hidden="true">${s.skipped ? SKIP_SVG : CHECK_SVG}</span>
        <span class="sign-text"><b>${esc(s.gloss)}</b><span>${esc(what)}</span></span></li>`;
  }).join("");
  $("#transcript").innerHTML = turns.map((t, k) => {
    const glosses = t.reply.map(glossOf).join(" ");
    return `<div class="msg msg-maya" style="--delay:${300 + k * 120}ms"><span class="bubble">${esc(t.partner)}</span></div>
      <div class="msg msg-you${t.skipped ? " msg-skipped" : ""}" style="--delay:${360 + k * 120}ms"><span class="bubble">${esc(t.skipped ? "(skipped)" : t.english ?? glosses)}<small>${esc(glosses)}</small></span></div>`;
  }).join("");
  $("#transcript-title").hidden = !turns.length;
  $("#transcript").hidden = !turns.length;
  show("results");
  chime("done");
}

/* ---- when something goes wrong (mobile/app.js) ------------------------------------------------------------------ */
function errorCopy(err) {
  const kind = err?.kind ?? null;
  const browserSteps = IOS
    ? (/CriOS/.test(UA)
      ? `<ol><li>Open your iPhone's <b>Settings</b>, scroll to <b>Chrome</b>, and turn <b>Camera</b> on.</li><li>Come back here and tap <b>Try again</b>.</li></ol>`
      : `<ol><li>Tap <b>aA</b> (or the page icon) at the side of the address bar.</li><li>Tap <b>Website Settings</b>, then set <b>Camera</b> to <b>Allow</b>.</li><li>Tap <b>Try again</b>.</li></ol>`)
    : ANDROID
      ? `<ol><li>Tap the icon at the left of the address bar.</li><li>Tap <b>Permissions</b> (or <b>Site settings</b>) and allow the <b>Camera</b>.</li><li>Tap <b>Try again</b>.</li></ol>`
      : `<ol><li>Click the camera icon in the address bar and choose <b>Allow</b>.</li><li>Click <b>Try again</b>.</li></ol>`;
  const detail = esc(String(err?.message || err)).replace(/\n/g, "<br>");
  if (kind === "unsupported" || (IN_APP && (kind === "NotAllowedError" || kind === "NotSupportedError"))) {
    return { title: "Open this page in your browser", copy: true,
      body: `<p>${IN_APP ? `${IN_APP}'s built-in browser` : "This browser"} can't use the camera.</p>
        <p>Open the page in <b>${BROWSER}</b>: ${IN_APP ? "tap the <b>•••</b> menu and choose <b>Open in browser</b>, or " : ""}copy the link and paste it there.</p>` };
  }
  if (kind === "NotAllowedError" || kind === "SecurityError" || kind === "PermissionDeniedError") {
    return { title: "Allow the camera", copy: true, body: `<p>The camera is how the lesson sees your signs. Nothing is recorded or sent anywhere.</p>${browserSteps}
      <p>Opened from another app? Then open the page in <b>${BROWSER}</b> instead: copy the link and paste it there.</p>` };
  }
  if (kind === "NotReadableError" || kind === "TrackStartError" || kind === "AbortError") {
    return { title: "Your camera is busy", body: `<p>Another app may be using the camera (a video call, the Camera app). Close it, then try again.</p>` };
  }
  if (kind === "NotFoundError" || kind === "OverconstrainedError" || kind === "DevicesNotFoundError") {
    return { title: "No camera found", body: `<p>This device didn't offer a front camera. Try another phone, or a laptop with a webcam.</p>` };
  }
  if (kind === "insecure") return { title: "Open the secure link", copy: true, body: `<p>Browsers only allow the camera on <b>https://</b> pages.</p>` };
  if (kind === "NoFrames") return { title: "The camera didn't start", body: `<p>The camera turned on but sent no picture. Close other apps that use it and try again.</p>` };
  if (kind === "no-simd") return { title: "This browser is too old", body: `<p>The lesson needs a newer browser (Safari 16.4 or later, or a recent Chrome or Firefox).</p>` };
  if (kind === "load" || /Could not load|Failed to fetch|NetworkError|Load failed|network|Importing a module script failed|module script|dynamically imported module|HTTP \d/i.test(String(err?.message || ""))) {
    return { title: "Couldn't load the lesson", body: `<p>Part of the lesson didn't download. Check your connection, then try again.</p><p class="fineprint">${detail}</p>` };
  }
  if (/hand tracking model could not start|Hand tracking stopped working/i.test(String(err?.message || ""))) {
    return { title: "The hand tracker stopped", body: `<p>The hand tracker couldn't keep running on this device's graphics. Close other tabs and apps, then try again.</p>` };
  }
  return { title: "Something went wrong", body: `<p>${detail}</p>` };
}
let failed = false;
function fail(err) {
  console.error("signs:", err);
  if (failed) return;
  failed = true;
  stats.errors.push(String(err?.kind || err?.message || err));
  routeFrames = false;
  onRec = null;
  releaseAwake();
  try { capture.stop(); } catch { /* already stopped */ }
  hidePip();
  watchEl.hidden = true;
  $("#interlude").hidden = true;
  const e = errorCopy(err);
  $("#error-title").textContent = e.title;
  $("#error-body").innerHTML = e.body;
  $("#btn-copy").hidden = !e.copy;
  show("error");
}

/* ---- keeping the screen on, the back button, coming back to the page ------------------------------------------ */
let wake = null;
async function keepAwake() {
  if (wake && !wake.released) return;
  try { wake = await navigator.wakeLock?.request("screen"); } catch { wake = null; }
}
function releaseAwake() { try { wake?.release(); } catch { /* already gone */ } wake = null; }
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState !== "visible") return;
  if (stats.screen === "lesson" && capture.running) keepAwake();
  if (audio?.state === "suspended") audio.resume().catch(() => {});
});
// back to a page restored from the browser's memory: the camera is gone, start clean
addEventListener("pageshow", (e) => { if (e.persisted) location.reload(); });
/* Back during the lesson (an Android edge swipe brushed while holding the phone; in Slack's or Gmail's browser it would
 * close the page) opens "Stop the lesson?" instead of leaving. */
function guardBack() {
  try { if (history.state?.lesson !== 1) history.pushState({ lesson: 1 }, ""); } catch { /* no history API */ }
}
addEventListener("popstate", () => {
  if (stats.screen !== "lesson") return;
  guardBack();
  openSheet();
});
function openSheet() { $("#quit-sheet").hidden = false; $("#btn-keep").focus({ preventScroll: true }); }
const closeSheet = () => { $("#quit-sheet").hidden = true; $("#btn-close").focus({ preventScroll: true }); };

/* ---- the frame-rate readout (?fps=1) ----------------------------------------------------------------------------- */
if (SHOW_PERF) {
  const el = $("#perf");
  el.hidden = false;
  let lastN = 0, lastT = performance.now();
  const med = (a) => { if (!a.length) return 0; const b = a.slice(-60).sort((x, y) => x - y); return b[Math.floor(b.length / 2)]; };
  setInterval(() => {
    const s = capture.stats, t = performance.now(), fps = ((s.frames - lastN) * 1000) / (t - lastT);
    lastN = s.frames; lastT = t;
    stats.fps = +fps.toFixed(1);
    const { width, height } = capture.source();
    el.textContent = `${fps.toFixed(0)} fps · hand ${med(s.handMs).toFixed(0)} ms · pose ${med(s.poseMs).toFixed(0)} ms · draw ${med(s.drawMs).toFixed(1)} ms · ${s.delegate ?? "…"} · ${width}×${height}`;
  }, 500);
}

/* ---- start ------------------------------------------------------------------------------------------------------- */
let started = false;
/* `tapped`: called from a tap (Start). A page that starts by itself (after "I sign with my left hand") asks for one tap
 * first: sound and the screen's wake lock are only granted inside a tap. */
async function start(tapped = true) {
  if (started) return;
  if (loadError && stats.screen === "welcome") { location.reload(); return; }
  started = true;
  if (tapped) { keepAwake(); unlockAudio(); guardBack(); }
  show("lesson");
  phase("check");
  setProgress(0);
  layout();
  try {
    if (!tapped) await waitForTap();
    stageMessage("Starting your camera…");
    $("#check-status").textContent = "Tap Allow when your browser asks to use the camera.";
    const playing = () => { if (stats.loaded < 1) { stageWaiting = true; loadProgress("capture", capture.stats.loaded); } };
    video.addEventListener("playing", playing, { once: true });
    await capture.start();
    rec = await recReady;
    stageWaiting = false;
    stageMessage(null);
    listen(rec);
    $("#check-status").textContent = "";
    const flip = await cameraCheck();
    stats.flip = flip.flip;
    stats.flipInfo = flip;
    rec.setFlip(flip.flip);
    await lessonFlow();
  } catch (err) {
    fail(err);
  }
}
async function waitForTap() {
  const card = $("#check-card"), b = document.createElement("button");
  b.type = "button"; b.className = "btn btn-primary"; b.textContent = "Start";
  b.style.maxWidth = "220px";
  $("#check-title").textContent = "Ready?";
  $("#check-status").textContent = `Tap Start, then raise your ${HAND} hand.`;
  card.append(b);
  await new Promise((r) => b.addEventListener("click", r, { once: true }));
  b.remove();
  $("#check-title").textContent = "Let's get you in the picture";
  keepAwake();
  unlockAudio();
  guardBack();
}
/* "Practice again": the same page, trackers and recognizer (no download, no new camera prompt to grant), the camera
 * reopened inside the tap, the session's flip kept, straight to the first sign. */
async function again() {
  if (!rec) { location.reload(); return; }
  keepAwake();
  unlockAudio();
  guardBack();
  stats.signs = [];
  stats.turns = [];
  met.clear();
  clearChat();
  setProgress(0);
  show("lesson");
  phase("watch");
  stageState("idle");
  layout();
  try {
    stageMessage("Starting your camera…");
    await capture.start();
    stageMessage(null);
    await lessonFlow();
  } catch (err) {
    fail(err);
  }
}

$("#btn-start").addEventListener("click", () => start(true));
/* Safari ignores overscroll-behavior on a page that does not scroll: a drag on the lesson screen (outside a card whose
 * words overflow) would rubber-band the page or pull it to refresh mid-sign. Taps are untouched. */
$("#lesson").addEventListener("touchmove", (e) => {
  const c = e.target.closest?.(".card, .watch-card");
  if (!c || c.scrollHeight <= c.clientHeight) e.preventDefault();
}, { passive: false });
$("#btn-close").addEventListener("click", openSheet);
$("#btn-keep").addEventListener("click", closeSheet);
$("#quit-sheet").addEventListener("click", (e) => { if (e.target === e.currentTarget) closeSheet(); });
$("#btn-quit").addEventListener("click", () => { capture.stop(); location.replace(location.href); });
$("#btn-sound").addEventListener("click", () => {
  soundOn = !soundOn;
  try { localStorage.setItem(SOUND_KEY, soundOn ? "on" : "off"); } catch { /* this visit only */ }
  renderSound();
  if (soundOn) { unlockAudio(); chime("chip"); }
});
$("#btn-again").addEventListener("click", () => again());
$("#btn-retry").addEventListener("click", () => location.reload());
/* Copy the link; where neither the clipboard nor a prompt works (many in-app browsers), show it to copy by hand. */
async function copyLink() {
  const url = shareableUrl();
  try { await navigator.clipboard.writeText(url); toast("Link copied"); return; } catch { /* below */ }
  const box = $("#error-body");
  if (document.body.dataset.screen === "error" && box && !box.querySelector(".link-out")) {
    box.insertAdjacentHTML("beforeend", `<p>Copy this link: <code class="link-out"></code></p>`);
    box.querySelector(".link-out").textContent = url;
  } else {
    toast(url);
  }
}
$("#btn-copy").addEventListener("click", copyLink);
$("#btn-copy-inapp").addEventListener("click", copyLink);
$("#btn-share").addEventListener("click", async () => {
  const url = shareableUrl();
  if (navigator.share) {
    try { await navigator.share({ title: "First Conversation", text: "I just had my first conversation in ASL. Try it:", url }); return; }
    catch (err) { if (err?.name === "AbortError") return; }
  }
  copyLink();
});
// a tap (touchend / click carry the user activation; pointerdown does not) opens the audio path, after a page that
// started without one or audio interrupted by a call, and asks for the wake lock again while there is a camera
for (const type of ["touchend", "click"]) {
  addEventListener(type, () => {
    if (!audio || audio.state !== "running") unlockAudio();
    if (capture.running && (!wake || wake.released)) keepAwake();
  }, { passive: true, capture: true });
}
// Enter on a card: its button (laptops; the button also has focus, this covers a click elsewhere first)
addEventListener("keydown", (e) => {
  if (e.key !== "Enter" || e.repeat || e.target?.closest?.("button")) return;
  if (!watchEl.hidden) $("#btn-try").click();
  else if (!$("#interlude").hidden) $("#btn-chat").click();
});

renderSound();
heroPictures();
renderWelcome();
$("#btn-start").disabled = false;
$("#btn-start").textContent = "Start";
let autostart = false;
try { autostart = sessionStorage.getItem(AUTOSTART_KEY) === "1"; sessionStorage.removeItem(AUTOSTART_KEY); } catch { /* none */ }
if (params.get("autostart") === "1") start(true);
else if (autostart) start(false);
