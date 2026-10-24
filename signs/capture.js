/* signs/capture.js: the front camera and the two trackers, frame by frame, as the frame records the recognizer reads
 * (docs/signs-design.md), and the drawing over the camera view.
 *
 * A sign is a hand moving against the body, so every camera frame goes through MediaPipe's HandLandmarker (two hands:
 * the one that signs is picked by tutor/pick-hand.js, the way the letter apps pick theirs) AND its PoseLandmarker lite
 * (where the shoulders and face are, which is what "at the chin" or "on the chest" is measured against). The robust
 * parts are the phone app's (mobile/app.js), for the same reasons:
 *
 *   - MediaPipe's runtime and both models (about 25 MB) download while the welcome screen is read, with a progress
 *     line; a download that goes quiet for 15 s is given up and tried again, three tries. They are handed to
 *     js/core/tracker.js as they are (the runtime as a blob URL, the models as bytes), so nothing is fetched twice.
 *   - The GPU, falling back to the CPU (js/core/tracker.js), and a tracker lost with the GPU (a phone locked on some
 *     iOS versions) is built again rather than ending the lesson.
 *   - Each camera frame is processed once (requestVideoFrameCallback; the screen's repaints where there is none), and
 *     nothing is tracked while the video has no picture. A frozen picture is never re-tracked: to the recognizer it
 *     would be a perfectly still hand.
 *   - The picture's shape is read on every frame: a phone turned on its side turns its camera picture, and every
 *     record carries the true width / height (features.js multiplies x by it).
 *   - A camera that stops sending frames while the page is in view is reopened (three tries), and so is one that
 *     ended while the page was hidden.
 *
 * Coordinates are those of the raw (unmirrored) camera picture, as the letter apps keep them; the page shows the
 * picture mirrored with CSS, and the canvas sits inside the same mirrored box, so it draws in raw coordinates too.
 *
 * createCapture({ video, hand, canvas, onProgress, onError, onResize }) ->
 *   { ready, start(), stop(), pause(), resume(), onFrame(cb), celebrate(), calm(), source(), stats } */

import { createTracker } from "../js/core/tracker.js";
import { MEDIAPIPE } from "../config.js";
import { createHandPicker } from "../tutor/pick-hand.js";

/* The letter apps' camera: 640 x 480 (480 x 640 on a phone held upright), 30 frames a second. PopSign's signers were
 * filmed at 480 x 640 / 30 fps, and the trackers never need more. */
export const CAMERA = Object.freeze({ width: 640, height: 480, frameRate: 30 });
const CDN = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE.version}`;
/* What each download yields when read (the runtime arrives compressed; these are the stream's sizes), for the
 * progress line only. */
export const BYTES = Object.freeze({ hand: 7819105, pose: 5777746, wasm: Object.freeze({ simd: 11756954, nosimd: 10960242 }) });
const STALL_MS = 15000;
/* No new frame for this long while the page is in view: the camera is reopened (at most every REOPEN_GAP_MS). */
const STALL_FRAMES_MS = 2500;
const REOPEN_GAP_MS = 6000;
/* requestVideoFrameCallback chains can go quiet (a camera swapped under the video element): armed again after this. */
const VFC_QUIET_MS = 1000;
/* A pose point counts as seen at this visibility (MediaPipe's own default for drawing). */
export const SEEN = 0.5;

export class CameraError extends Error {
  constructor(kind, detail) { super(kind); this.kind = kind; this.detail = detail; }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/* ---- frame records ------------------------------------------------------------------------------------------- */

/** 21 x (x, y, z), MediaPipe's normalised coordinates of the raw picture. */
export function packHand(lm) {
  const out = new Float32Array(63);
  for (let i = 0; i < 21; i++) {
    const p = lm[i];
    out[3 * i] = p?.x ?? NaN;
    out[3 * i + 1] = p?.y ?? NaN;
    out[3 * i + 2] = p?.z ?? NaN;
  }
  return out;
}

/** 33 x (x, y, visibility). */
export function packPose(lm) {
  const out = new Float32Array(99);
  for (let i = 0; i < 33; i++) {
    const p = lm[i];
    out[3 * i] = p?.x ?? NaN;
    out[3 * i + 1] = p?.y ?? NaN;
    out[3 * i + 2] = p?.visibility ?? NaN;
  }
  return out;
}

/**
 * One frame record (docs/signs-design.md) from the two trackers' answers.
 * @param {{tMs:number, aspect:number, handRes:{landmarks:object[][], handedness:string[]}|null,
 *          poseRes:{landmarks:object[][]}|null, pick?:(res:object, aspect:number) => number, k?:number}} o
 *   pick: tutor/pick-hand.js createHandPicker(hand), which remembers where the signing hand was; or `k`, the
 *   signing hand's index when it has already been picked (a picker is asked once per frame)
 * @returns {{tMs:number, aspect:number, hand:Float32Array|null, other:Float32Array|null, pose:Float32Array|null}}
 */
export function makeRecord({ tMs, aspect, handRes, poseRes, pick = null, k = null }) {
  const hands = handRes?.landmarks ?? [];
  if (!Number.isInteger(k)) k = hands.length && pick ? pick(handRes, aspect) : -1;
  let o = -1;
  for (let i = 0; i < hands.length; i++) if (i !== k) { o = i; break; }
  const body = poseRes?.landmarks?.[0];
  return {
    tMs,
    aspect,
    hand: k >= 0 && hands[k]?.length === 21 ? packHand(hands[k]) : null,
    other: o >= 0 && hands[o]?.length === 21 ? packHand(hands[o]) : null,
    pose: body && body.length === 33 ? packPose(body) : null,
  };
}

const seen = (pose, i) => pose[3 * i + 2] >= SEEN;
const inside = (pose, i) => pose[3 * i] >= 0 && pose[3 * i] <= 1 && pose[3 * i + 1] >= 0 && pose[3 * i + 1] <= 1;

/**
 * Is the learner's head (nose and both eyes), and are both shoulders, in the picture? The camera check asks for both:
 * the recognizer measures every sign against the shoulders (features.js body frame) and the face (the place checks).
 * @returns {{head:boolean, shoulders:boolean, width:number|null}}  width: shoulder width in picture widths
 */
export function bodyInView(pose) {
  if (!pose) return { head: false, shoulders: false, width: null };
  const head = [0, 2, 5].every((i) => seen(pose, i) && inside(pose, i));
  const shoulders = [11, 12].every((i) => seen(pose, i) && inside(pose, i));
  const width = shoulders ? Math.abs(pose[3 * 11] - pose[3 * 12]) : null;
  return { head, shoulders, width };
}

/**
 * Where the signing hand's wrist is across the body: its x minus the shoulders' midpoint, in shoulder widths, in the
 * raw picture (+ is the picture's right). null without the hand or both shoulders.
 */
export function wristOffset(rec) {
  if (!rec?.hand || !rec.pose) return null;
  const p = rec.pose;
  if (!(seen(p, 11) && seen(p, 12))) return null;
  const width = Math.abs(p[33] - p[36]);
  if (!(width > 0.02)) return null;
  const d = (rec.hand[0] - (p[33] + p[36]) / 2) / width;
  return Number.isFinite(d) ? d : null;
}

/* Under this (in shoulder widths) a raised hand is too near the middle of the body to say which side it is on. */
export const FLIP_MARGIN = 0.15;
const FLIP_MIN_SAMPLES = 5;

/**
 * The session's canonical side (docs/signs-design.md): the model always sees the signing hand on the picture's left
 * (a right-handed signer in an unmirrored selfie), so flip is true when the raised signing hand's wrist was to the
 * picture-RIGHT of the shoulders' midpoint -- a left-handed learner, or a camera that mirrors. The median of the frames
 * of the camera check's hold; too few of them, or a hand held in the middle, gives `fallback` (the side the learner's
 * hand would be on in an unmirrored picture).
 * @param {(number|null)[]} offsets  wristOffset() of each frame
 * @param {boolean} fallback
 * @returns {{flip:boolean, decided:boolean, median:number|null, n:number}}
 */
export function decideFlip(offsets, fallback) {
  const v = offsets.filter((x) => Number.isFinite(x)).sort((a, b) => a - b);
  if (v.length < FLIP_MIN_SAMPLES) return { flip: !!fallback, decided: false, median: null, n: v.length };
  const m = v.length % 2 ? v[(v.length - 1) / 2] : (v[v.length / 2 - 1] + v[v.length / 2]) / 2;
  if (Math.abs(m) < FLIP_MARGIN) return { flip: !!fallback, decided: false, median: m, n: v.length };
  return { flip: m > 0, decided: true, median: m, n: v.length };
}

/* ---- drawing ------------------------------------------------------------------------------------------------- */

const BONES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]];
const ARMS = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16]];
/* The letter apps' look (experiments/asl-demo.js drawHand, copied rather than imported: the letter apps stay as they
 * are): the signing hand a sky-blue skeleton with white joints; made, it turns green in 150 ms, its joints give one
 * small bounce and a smaller second one, one soft ring leaves the palm and the fingertips twinkle once. With reduced
 * motion it only turns green. The body is drawn faintly, so the learner sees that the camera has found their
 * shoulders and face without it competing with the hand. */
const SKY = [56, 189, 248], MINT = [34, 197, 94];
export const CELEBRATE = Object.freeze({ greenMs: 150, ringMs: 650, twinkleMs: 520 });
const REDUCED_MOTION = typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;
/* One bump up and back by 260 ms, then a smaller one by 480 ms: 0..1. */
export function celebratePop(ms) {
  if (!(ms >= 0)) return 0;
  if (ms < 260) return Math.sin(Math.PI * (ms / 260)) * Math.exp(-ms / 300);
  if (ms < 480) return 0.3 * Math.sin(Math.PI * ((ms - 260) / 220)) * Math.exp(-(ms - 260) / 250);
  return 0;
}

function drawBody(ctx, W, H, pose, s) {
  if (!pose) return;
  const x = (i) => pose[3 * i] * W, y = (i) => pose[3 * i + 1] * H;
  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "rgba(255, 255, 255, 0.4)";
  ctx.lineWidth = s * 1.1;
  ctx.beginPath();
  for (const [a, b] of ARMS) if (seen(pose, a) && seen(pose, b)) { ctx.moveTo(x(a), y(a)); ctx.lineTo(x(b), y(b)); }
  ctx.stroke();
  ctx.fillStyle = "rgba(255, 255, 255, 0.55)";
  for (const i of [11, 12, 13, 14]) if (seen(pose, i)) { ctx.beginPath(); ctx.arc(x(i), y(i), s * 1.5, 0, 2 * Math.PI); ctx.fill(); }
  // the face: one light ring round it, sized by the ears (or the eyes when an ear is hidden), so a head turned a
  // little does not shrink it
  if (seen(pose, 0)) {
    const ears = seen(pose, 7) && seen(pose, 8) ? Math.hypot(x(7) - x(8), y(7) - y(8)) : null;
    const eyes = seen(pose, 2) && seen(pose, 5) ? Math.hypot(x(2) - x(5), y(2) - y(5)) * 2.4 : null;
    const r = 0.62 * (ears ?? eyes ?? 0);
    if (r > 4) {
      ctx.strokeStyle = "rgba(255, 255, 255, 0.32)";
      ctx.lineWidth = s;
      ctx.beginPath();
      ctx.ellipse(x(0), y(0) - 0.12 * r, r, r * 1.22, 0, 0, 2 * Math.PI);
      ctx.stroke();
    }
  }
  ctx.restore();
}

function drawOtherHand(ctx, W, H, h, s) {
  if (!h) return;
  ctx.save();
  ctx.lineCap = "round";
  ctx.strokeStyle = "rgba(255, 255, 255, 0.3)";
  ctx.lineWidth = s * 0.9;
  ctx.beginPath();
  for (const [a, b] of BONES) { ctx.moveTo(h[3 * a] * W, h[3 * a + 1] * H); ctx.lineTo(h[3 * b] * W, h[3 * b + 1] * H); }
  ctx.stroke();
  ctx.restore();
}

/* `sinceMs`: how long ago the sign was made (null: not made). */
function drawHand(ctx, W, H, h, s, sinceMs = null) {
  if (!h) return;
  const X = (i) => h[3 * i] * W, Y = (i) => h[3 * i + 1] * H;
  const won = sinceMs !== null && sinceMs >= 0;
  const green = won ? Math.min(1, sinceMs / CELEBRATE.greenMs) : 0;
  const col = SKY.map((c, i) => Math.round(c + (MINT[i] - c) * green)).join(", ");
  const pop = won && !REDUCED_MOTION ? celebratePop(sinceMs) : 0;
  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  const bones = () => { ctx.beginPath(); for (const [a, b] of BONES) { ctx.moveTo(X(a), Y(a)); ctx.lineTo(X(b), Y(b)); } };
  if (green > 0) {
    // a soft green glow under the lines (a wide faint stroke: no per-frame blur)
    ctx.strokeStyle = `rgba(${MINT.join(", ")}, ${(0.3 * green).toFixed(3)})`;
    ctx.lineWidth = s * (3.4 + 1.2 * pop);
    bones(); ctx.stroke();
  }
  ctx.strokeStyle = `rgba(${col}, 0.95)`;
  ctx.lineWidth = s * (1 + 0.35 * pop + 0.5 * green);
  bones(); ctx.stroke();
  ctx.fillStyle = "#ffffff";
  const r = 1.6 * s * (1 + 0.7 * pop + 0.1 * green);
  for (let i = 0; i < 21; i++) { ctx.beginPath(); ctx.arc(X(i), Y(i), r, 0, 2 * Math.PI); ctx.fill(); }
  if (won && !REDUCED_MOTION) {
    const palm = Math.hypot(X(9) - X(0), Y(9) - Y(0));
    if (sinceMs < CELEBRATE.ringMs) {
      const k = sinceMs / CELEBRATE.ringMs, ease = 1 - (1 - k) ** 3;
      const cx = [0, 5, 9, 13, 17].reduce((a, i) => a + X(i), 0) / 5, cy = [0, 5, 9, 13, 17].reduce((a, i) => a + Y(i), 0) / 5;
      ctx.strokeStyle = `rgba(${MINT.join(", ")}, ${(0.55 * (1 - k)).toFixed(3)})`;
      ctx.lineWidth = s * 1.6;
      ctx.beginPath(); ctx.arc(cx, cy, palm * (0.9 + 1.5 * ease), 0, 2 * Math.PI); ctx.stroke();
    }
    if (sinceMs < CELEBRATE.twinkleMs) {
      const k = sinceMs / CELEBRATE.twinkleMs, size = s * 3.2 * Math.sin(Math.PI * k);
      ctx.fillStyle = `rgba(255, 255, 255, ${(0.95 * (1 - k)).toFixed(3)})`;
      for (const i of [4, 8, 12, 16, 20]) {
        const x = X(i), y = Y(i);
        ctx.beginPath();
        ctx.moveTo(x, y - size); ctx.lineTo(x + size * 0.28, y - size * 0.28); ctx.lineTo(x + size, y); ctx.lineTo(x + size * 0.28, y + size * 0.28);
        ctx.lineTo(x, y + size); ctx.lineTo(x - size * 0.28, y + size * 0.28); ctx.lineTo(x - size, y); ctx.lineTo(x - size * 0.28, y - size * 0.28);
        ctx.closePath(); ctx.fill();
      }
    }
  }
  ctx.restore();
}

/** The whole overlay for one frame record. */
export function drawFrame(ctx, rec, { sinceMs = null } = {}) {
  const W = ctx.canvas.width, H = ctx.canvas.height, s = Math.max(1.8, H / 300);
  ctx.clearRect(0, 0, W, H);
  if (!rec) return;
  drawBody(ctx, W, H, rec.pose, s);
  drawOtherHand(ctx, W, H, rec.other, s);
  drawHand(ctx, W, H, rec.hand, s, sinceMs);
}

/* ---- downloads ---------------------------------------------------------------------------------------------- */

/* One file with progress (mobile/app.js download()). A response that stops sending for STALL_MS is given up and tried
 * again (three tries, a moment apart): on a phone a download that goes quiet without failing is ordinary. The runtime
 * comes back as a Blob built from its chunks and a model as bytes in one buffer sized up front, so the peak memory is
 * about one copy of each. */
async function download(url, bytes, onBytes, { asBlob = false } = {}) {
  let last = null;
  for (let attempt = 0; attempt < 3; attempt++) {
    if (attempt) { onBytes(0, true); await sleep(1000 * attempt); }
    const ctrl = new AbortController();
    let timer = null;
    const arm = () => { clearTimeout(timer); timer = setTimeout(() => ctrl.abort(), STALL_MS); };
    try {
      arm();
      const res = await fetch(url, { signal: ctrl.signal });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      if (!res.body) {
        const whole = asBlob ? await res.blob() : new Uint8Array(await res.arrayBuffer());
        clearTimeout(timer);
        onBytes(bytes);
        return whole;
      }
      const reader = res.body.getReader();
      const parts = [];
      let out = asBlob ? null : new Uint8Array(bytes), got = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        arm();
        if (asBlob) parts.push(value);
        else {
          if (got + value.byteLength > out.length) { const more = new Uint8Array(Math.ceil((got + value.byteLength) * 1.25)); more.set(out.subarray(0, got)); out = more; }
          out.set(value, got);
        }
        got += value.byteLength;
        onBytes(got);
      }
      clearTimeout(timer);
      return asBlob ? new Blob(parts, { type: "application/wasm" }) : out.subarray(0, got);
    } catch (err) {
      clearTimeout(timer);
      last = err;
    }
  }
  const e = new Error(`Could not load the hand tracker (${last?.name === "AbortError" ? "the download stalled" : last?.message || last}).`);
  e.kind = "load";
  throw e;
}

/* The first picture with a hand in it compiles the hand model's GPU programs: up to a second or two on a phone, which
 * would freeze the first camera frame. A glove render is a hand MediaPipe finds, so both trackers see it now; nothing
 * here is shown or recorded. */
async function warmUp(hand, pose) {
  const img = new Image();
  img.src = new URL("../assets/gloves/X.png", import.meta.url).href;
  await img.decode();
  const c = document.createElement("canvas");
  c.width = 360; c.height = 480;
  const g = c.getContext("2d");
  g.fillStyle = "#6b6259";
  g.fillRect(0, 0, c.width, c.height);
  g.drawImage(img, 30, 60, 300, 300 * img.height / img.width);
  const t = performance.now();
  for (let i = 0; i < 3; i++) { hand.track(c, t + i); pose.track(c, t + i); }
}

/* ---- the capture ------------------------------------------------------------------------------------------ */

/**
 * @param {{video: HTMLVideoElement, hand?: "right"|"left", canvas?: HTMLCanvasElement|null,
 *          onProgress?: (p: {frac: number, text: string}) => void, onError?: (err: Error) => void,
 *          onResize?: (s: {width: number, height: number, aspect: number}) => void}} o
 */
export function createCapture({ video, hand = "right", canvas = null, onProgress = null, onError = null, onResize = null } = {}) {
  const stats = {
    frames: 0, handMs: [], poseMs: [], drawMs: [], delegate: null, poseDelegate: null, video: null, warm: false,
    loaded: 0, rebuilt: 0, reopened: 0, errors: [],
  };
  const keep = (a, v) => { a.push(v); if (a.length > 600) a.splice(0, 300); };
  const listeners = new Set();
  const pick = createHandPicker(hand);
  const ctx = canvas?.getContext("2d") ?? null;
  let tracker = null, poseTracker = null;
  let stream = null, running = false, paused = false, halted = false, clock = null, celebratedAt = null;

  /* ---- loading, with progress ---- */
  const got = new Map();
  const progress = (text) => {
    let a = 0, b = 0;
    for (const [x, y] of got.values()) { a += x; b += y; }
    stats.loaded = b ? 0.9 * (a / b) : 0;
    try { onProgress?.({ frac: stats.loaded, text }); } catch (err) { console.warn("capture: onProgress threw", err); }
  };
  const ready = (async () => {
    const { FilesetResolver } = await import(`${CDN}/vision_bundle.mjs`);
    const simd = await FilesetResolver.isSimdSupported();
    const name = `vision_wasm${simd ? "" : "_nosimd"}_internal`;
    const wasmUrl = `${CDN}/wasm/${name}.wasm`, wasmBytes = simd ? BYTES.wasm.simd : BYTES.wasm.nosimd;
    got.set("wasm", [0, wasmBytes]); got.set("hand", [0, BYTES.hand]); got.set("pose", [0, BYTES.pose]);
    const track = (key, total) => (n, retry = false) => { got.set(key, [Math.min(n, total), total]); progress(retry ? "Slow connection, trying again…" : "Getting the hand tracker ready…"); };
    const [wasm, handModel, poseModel] = await Promise.all([
      download(wasmUrl, wasmBytes, track("wasm", wasmBytes), { asBlob: true }),
      download(MEDIAPIPE.models.hand, BYTES.hand, track("hand", BYTES.hand)),
      download(MEDIAPIPE.models.pose, BYTES.pose, track("pose", BYTES.pose)),
    ]);
    // kept for the page's life: tracker.js keeps the file set it was first given, and a tracker rebuilt after a lost GPU
    // loads its runtime from it again
    const blobUrl = URL.createObjectURL(wasm);
    stats.loaded = 0.93;
    try { onProgress?.({ frac: 0.93, text: "Starting the hand tracker…" }); } catch { /* reported above */ }
    const fileset = { wasmLoaderPath: `${CDN}/wasm/${name}.js`, wasmBinaryPath: blobUrl };
    tracker = await createTracker("hand", { numHands: 2, fileset, modelAssetBuffer: handModel });
    poseTracker = await createTracker("pose", { fileset, modelAssetBuffer: poseModel });
    stats.delegate = tracker.delegate;
    stats.poseDelegate = poseTracker.delegate;
    try { await warmUp(tracker, poseTracker); stats.warm = true; } catch (err) { console.warn(`capture: warm-up skipped (${err?.message || err})`); }
    stats.loaded = 1;
    try { onProgress?.({ frac: 1, text: "Ready" }); } catch { /* reported above */ }
    return { delegate: stats.delegate, poseDelegate: stats.poseDelegate };
  })();
  ready.catch(() => { /* the caller's own await reports it */ });

  /* ---- the camera ---- */
  async function openCamera() {
    if (!window.isSecureContext) throw new CameraError("insecure");
    if (TEST_CAMERA) {
      stream = testCameraStream();
    } else {
    if (!navigator.mediaDevices?.getUserMedia) throw new CameraError("unsupported");
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: "user", width: { ideal: CAMERA.width }, height: { ideal: CAMERA.height }, frameRate: { ideal: CAMERA.frameRate } },
      });
    } catch (err) {
      throw new CameraError(err?.name || "unknown", err?.message);
    }
    }
    const track = stream.getVideoTracks()[0];
    track?.addEventListener("ended", () => { if (document.visibilityState === "visible" && running) resumeCamera(); });
    video.srcObject = stream;
    video.muted = true;
    video.playsInline = true;
    // play() settles only once a picture arrives: a camera held by a call never sends one, so it is not awaited unbounded
    await Promise.race([video.play().catch(() => {}), sleep(5000)]);
    for (let i = 0; i < 100 && !(video.videoWidth > 0); i++) await sleep(50);
    if (!(video.videoWidth > 0)) throw new CameraError("NoFrames");
    stats.video = { w: video.videoWidth, h: video.videoHeight, settings: track?.getSettings?.() ?? null };
    sourceSize();
    aliveAt = performance.now();
  }
  function stopCamera() {
    stream?.getTracks().forEach((t) => t.stop());
    stream = null;
    video.srcObject = null;
  }
  let resuming = null, reopenedAt = 0, aliveAt = 0;
  /* After a lock, an app switch or a stall: play again, or reopen the camera (`force`: even if the track says it is
   * live). A camera just let go by another app can refuse once, so a reopen is tried three times. The guard is set
   * before the body can finish, so a return that finds the camera live cannot leave it holding a settled promise. */
  function resumeCamera(force = false) {
    if (resuming || halted) return resuming;
    let done;
    resuming = new Promise((r) => { done = r; });
    (async () => {
      try {
        const t = stream?.getVideoTracks()[0];
        if (force || !t || t.readyState === "ended") {
          for (let attempt = 0; ; attempt++) {
            try { stopCamera(); await openCamera(); break; }
            catch (err) { if (attempt >= 2) throw err; await sleep(700); }
          }
          stats.reopened++;
          clock?.rearm();
        } else if (video.paused) {
          await Promise.race([video.play().catch(() => {}), sleep(3000)]);
        }
      } catch (err) {
        fail(err);
      } finally {
        resuming = null;
        done();
      }
    })();
    return resuming;
  }
  const watchdog = setInterval(() => {
    // not while paused: without requestVideoFrameCallback, frames are only counted as they are processed
    if (!running || paused || halted || resuming || !stream || document.visibilityState !== "visible") return;
    const now = performance.now();
    if (now - aliveAt > STALL_FRAMES_MS && now - reopenedAt > REOPEN_GAP_MS) { reopenedAt = now; resumeCamera(true); }
  }, 1000);
  const onVisible = () => {
    if (document.visibilityState !== "visible" || !running || !stream) return;
    // a return gives frames 2.5 s before the watchdog reopens the camera
    aliveAt = performance.now();
    resumeCamera();
  };
  document.addEventListener("visibilitychange", onVisible);

  /* ---- the picture's shape, read on every frame ---- */
  let srcW = innerWidth > innerHeight ? CAMERA.width : CAMERA.height;
  let srcH = innerWidth > innerHeight ? CAMERA.height : CAMERA.width;
  function sourceSize() {
    const w = video.videoWidth, h = video.videoHeight;
    if (w > 0 && h > 0 && (w !== srcW || h !== srcH)) {
      srcW = w; srcH = h;
      sizeCanvas();
      try { onResize?.({ width: srcW, height: srcH, aspect: srcW / srcH }); } catch (err) { console.warn("capture: onResize threw", err); }
    }
    return { width: srcW, height: srcH, aspect: srcW / srcH };
  }
  /* The drawing at the screen's pixel density, at most 2 canvas pixels per camera pixel: crisp lines on a phone without
   * a huge canvas, and a hand keeps its size on the canvas when the view changes size. */
  function sizeCanvas() {
    if (!canvas) return;
    const k = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
    const cw = Math.round(srcW * k), ch = Math.round(srcH * k);
    if (canvas.width !== cw || canvas.height !== ch) { canvas.width = cw; canvas.height = ch; }
  }
  sizeCanvas();

  /* ---- one new camera frame at a time ---- */
  /* With requestVideoFrameCallback each camera frame is flagged once, as it is shown, and the loop (on the screen's
   * repaints) processes it once; the frame's time is when the camera took it, where the browser says so. Until the
   * first callback, or where there is none, the test is whether the video's time has moved. Once the callbacks have
   * started they are the only test: a picture that stops coming is not re-tracked; if they go quiet for a second the
   * chain is armed again (one chain counts, by generation). Nothing is tracked while the video has no picture. */
  function createClock(src) {
    let seq = 0, doneSeq = 0, lastTime = -1, vfcAt = -Infinity, gen = 0, meta = null;
    const vfc = typeof src.requestVideoFrameCallback === "function";
    const arm = () => {
      const g = ++gen;
      const tick = () => src.requestVideoFrameCallback((now, m) => {
        if (g !== gen) return;
        seq++; vfcAt = performance.now(); aliveAt = vfcAt;
        const cap = m?.captureTime;
        // the camera's own time when it is on the page's clock and recent; else when the frame was handed over
        meta = Number.isFinite(cap) && Math.abs(now - cap) < 1000 ? cap : now;
        tick();
      });
      tick();
    };
    if (vfc) arm();
    return {
      hasNew() {
        if (!(src.readyState >= 2 && src.videoWidth > 0)) return false;
        if (vfcAt === -Infinity) return src.currentTime !== lastTime;
        if (performance.now() - vfcAt > VFC_QUIET_MS) { vfcAt = performance.now(); arm(); }
        return seq !== doneSeq;
      },
      done() {
        if (vfcAt === -Infinity && src.currentTime !== lastTime) aliveAt = performance.now();
        doneSeq = seq; lastTime = src.currentTime;
      },
      frameTime: () => (vfcAt === -Infinity || meta === null ? performance.now() : meta),
      rearm() { if (vfc) { vfcAt = performance.now(); arm(); } },
      next: () => (halted || !running ? Promise.resolve() : new Promise((r) => requestAnimationFrame(() => r()))),
    };
  }

  /* ---- tracking ---- */
  const NO_HANDS = Object.freeze({ landmarks: [], worldLandmarks: [], handedness: [], handednessScore: [] });
  const rebuilding = { hand: null, pose: null };
  /* tracker.js gives up after ~3 s of failed frames (a GPU context lost when the phone was locked). Rather than end the
   * lesson, build a new tracker (from the kept runtime and the browser's cached model) and carry on; only a failed
   * rebuild, or a third loss, ends it. */
  function rebuild(kind, err) {
    if (rebuilding[kind]) return;
    if (stats.rebuilt >= 2) { fail(err); return; }
    rebuilding[kind] = (async () => {
      await null;
      try {
        const t = await createTracker(kind, kind === "hand" ? { numHands: 2 } : {});
        const old = kind === "hand" ? tracker : poseTracker;
        try { old?.close(); } catch { /* already gone */ }
        if (kind === "hand") tracker = t; else poseTracker = t;
        stats.rebuilt++;
      } catch (e) {
        fail(e);
      } finally {
        rebuilding[kind] = null;
      }
    })();
  }
  // each landmarker wants its own times strictly increasing
  let tsHand = 0, tsPose = 0;
  let frameErrors = 0;
  function processFrame() {
    const tMs = clock.frameTime();
    const { aspect } = sourceSize();
    const t0 = performance.now();
    let handRes = NO_HANDS, poseRes = null;
    if (!rebuilding.hand) {
      tsHand = Math.max(tsHand + 1, tMs);
      try { handRes = tracker.track(video, tsHand); } catch (err) { rebuild("hand", err); }
    }
    const t1 = performance.now();
    if (!rebuilding.pose) {
      tsPose = Math.max(tsPose + 1, tMs);
      try { poseRes = poseTracker.track(video, tsPose); } catch (err) { rebuild("pose", err); }
    }
    const t2 = performance.now();
    keep(stats.handMs, t1 - t0);
    keep(stats.poseMs, t2 - t1);
    const k = handRes.landmarks?.length ? pick(handRes, aspect) : -1;
    const rec = makeRecord({ tMs, aspect, handRes, poseRes, k });
    stats.frames++;
    if (ctx) {
      drawFrame(ctx, rec, { sinceMs: celebratedAt === null ? null : performance.now() - celebratedAt });
      keep(stats.drawMs, performance.now() - t2);
    }
    // the record, and the trackers' own answer (the camera check reads MediaPipe's handedness label from it)
    const raw = { handRes, poseRes, k };
    for (const cb of [...listeners]) {
      try { cb(rec, raw); } catch (err) { console.warn(`capture: a frame listener threw (${err?.message || err})`); }
    }
  }
  async function loop() {
    while (running && !halted) {
      if (!paused && clock.hasNew()) {
        clock.done();
        try { processFrame(); frameErrors = 0; }
        catch (err) {
          // one bad frame is skipped; a second of them in a row is not a hiccup
          if (++frameErrors === 1) console.error("capture: a frame failed and was skipped", err);
          if (frameErrors >= 30) { fail(err); return; }
        }
      }
      await clock.next();
    }
  }

  function fail(err) {
    if (halted) return;
    halted = true;
    running = false;
    stats.errors.push(String(err?.kind || err?.message || err));
    stopCamera();
    try { onError?.(err); } catch (e) { console.error("capture: onError threw", e); }
  }

  return {
    ready,
    stats,
    /** Opens the camera (call it from a tap: the permission prompt and iOS's camera want one), waits for the trackers,
     *  and starts the frame loop. Rejects with a CameraError or the load's error. */
    async start() {
      if (running) return;
      halted = false;
      try {
        await openCamera();
        await ready;
      } catch (err) {
        stopCamera();
        throw err;
      }
      running = true;
      paused = false;
      clock ??= createClock(video);
      clock.rearm();
      loop();
    },
    stop() {
      running = false;
      stopCamera();
      if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
    },
    /** No tracking while a card covers the camera (the demo plays smoother and a phone stays cooler); the camera stays
     *  open, so going on is instant. */
    pause() { paused = true; if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height); },
    resume() { paused = false; aliveAt = performance.now(); },
    get paused() { return paused; },
    get running() { return running; },
    onFrame(cb) { listeners.add(cb); return () => listeners.delete(cb); },
    /** The signing hand turns green, with the celebration, until calm(). */
    celebrate() { celebratedAt = performance.now(); },
    calm() { celebratedAt = null; },
    source: () => ({ width: srcW, height: srcH, aspect: srcW / srcH }),
    resumeCamera,
    close() {
      running = false;
      clearInterval(watchdog);
      document.removeEventListener("visibilitychange", onVisible);
      stopCamera();
    },
  };
}

/* ---- a test camera (localhost only, ?camsrc=1) -------------------------------------------------------------------
 * The camera is replaced by a canvas stream drawn from video files that a test harness chooses, try by try
 * (window.__signsCam.play(url) resolves when the clip has ended; its last picture stays up). Everything after the
 * camera is the app's own: the trackers, the recognizer, the screens. So a real signer's footage can be put in front
 * of exactly the sign the lesson is asking for, which a looping fake camera cannot do. Never on a deployed site. */
const TEST_CAMERA = typeof location !== "undefined" && /^(localhost|127\.0\.0\.1)$/.test(location.hostname) &&
  new URLSearchParams(location.search).get("camsrc") === "1";
function testCameraStream() {
  const canvas = document.createElement("canvas");
  canvas.width = 360; canvas.height = 640;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#000"; ctx.fillRect(0, 0, canvas.width, canvas.height);
  const player = document.createElement("video");
  player.muted = true; player.playsInline = true; player.preload = "auto";
  const draw = () => {
    if (player.readyState >= 2 && player.videoWidth) {
      if (canvas.width !== player.videoWidth || canvas.height !== player.videoHeight) { canvas.width = player.videoWidth; canvas.height = player.videoHeight; }
      ctx.drawImage(player, 0, 0, canvas.width, canvas.height);
    } else {
      ctx.fillStyle = "#000"; ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
    requestAnimationFrame(draw);
  };
  requestAnimationFrame(draw);
  window.__signsCam = {
    play(url) {
      return new Promise((resolve, reject) => {
        player.onended = () => resolve(true);
        player.onerror = () => reject(new Error(`test camera: cannot play ${url}`));
        player.src = url;
        player.play().catch(reject);
      });
    },
    get playing() { return !player.paused && !player.ended; },
  };
  return canvas.captureStream(30);
}
