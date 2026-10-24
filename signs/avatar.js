/* avatar.js: the demonstration signer of the public build -- a calm, friendly line-art person who signs with a Deaf
 * signer's own motion.
 *
 * The PopSign clips cannot be shown in a product (the signers consented to research databases and presentations;
 * ~/Projects/handful-local-media/popsign/README.md), so the public build draws a person instead.
 * tools/signs/extract_demos.py tracked one signer's sign with MediaPipe (hand, pose, face mesh), cut it to the sign
 * itself, steadied the hand-held camera, smoothed it gently and stored the landmarks (signs/model/demos/<id>.json).
 * This file turns them into a figure on a canvas. The motion is the signer's: every finger, the path, the brows and
 * the mouth. Only the picture is ours.
 *
 * WHAT IS DRAWN, back to front: a plain shirt (shoulders, a torso hint, a crew neck), the neck, ears, face, short hair,
 * brows, eyes, a nose hint, lips (in ASL the face is part of the sign: the brows ask, the mouth adds meaning), then
 * the signing arm in its sleeve and the hand. The other arm is not drawn: the signers held their phone in it.
 *   - The hand is large and in the accent colour, so the eye goes to it, and drawn the way the letter app draws its
 *     ghost hand (experiments/asl-tutor/ghost.js), the same bone widths: each finger one tapered shape, painted back
 *     to front by MediaPipe's depth, a finger put in front of the palm only when it is clearly nearer (a fist's folded
 *     fingers), one clean outline round the whole hand and a thin line only where parts cross. Its palm is lighter
 *     than its back, and the palm's colour follows which way it faces, so "palm toward you" can be seen.
 *   - The shirt is plain and dark, as an interpreter's is, so the hand stands out against it.
 *   - The eyes are simple dark ovals: the signers looked at their phone's screen, below its camera, and drawn irises
 *     would stare past the learner. The ovals still close when the signer blinked or squinted.
 *
 * TIME. A pass is: the first pose held (demo.hold.start, ~0.4 s), the sign at the demo's 30 fps, the last pose held
 * (~0.5 s); when looping, `pauseMs` more on the last pose and then a quick glide back to the first. Drawn at up to
 * 60 frames a second, between the 30 fps samples by Catmull-Rom interpolation, so slow motion is smooth rather than
 * stepped. prefers-reduced-motion: nothing plays by itself; the key frame (demo.key, where the hand has arrived and
 * holds its shape) is shown, and play() still works when the learner asks for it.
 *
 * COLOURS come from CSS custom properties on the canvas (or its ancestors), so the app's light and dark themes apply:
 * --avatar-skin, --avatar-hair, --avatar-shirt, --avatar-ink, --avatar-lips, --avatar-hand (else the app's --accent),
 * --avatar-hand-ink, --avatar-bg (only drawn with { background: true }). Unset ones come from the light or dark palette
 * below, chosen by the page's --bg (else prefers-color-scheme). { theme: "light" | "dark" } uses that palette as it is;
 * { theme: { skin: "#c68863", ... } } sets single colours over the page's.
 *
 *   import { loadDemo, createAvatar, unionView } from "./avatar.js";
 *   const demo = await loadDemo("model/demos/hello.json");
 *   const av = createAvatar(canvas, demo, {
 *     mirror: false,     // true: a left-handed signer (for a left-handed learner)
 *     speed: 1,          // 0.5: slow motion
 *     loop: true, pauseMs: 500,
 *     theme: "auto", autoplay: true, background: false,
 *     view: null,        // a box in shoulder widths; unionView(demos) gives a lesson's signs one framing
 *   });
 *   av.play(); av.pause(); av.setSpeed(0.5); av.setMirror(true); av.setView(v); av.setTheme("dark");
 *   av.seek(0.8);        // seconds into the pass, drawn at once; av.seekKey() shows the key frame
 *   const off = av.onLoop(() => ...);   // after each pass (once, at the end, with loop: false); off() unsubscribes
 *   av.duration, av.cycle, av.time, av.playing, av.reducedMotion, av.stats;  av.destroy();
 * The canvas is sized by CSS; the drawing follows its size (ResizeObserver) at up to 2 canvas pixels per CSS pixel, and
 * stops while it is off screen or the page is hidden.
 */

/* ---- the data ----------------------------------------------------------------------------------------------------- */

/* Face-mesh points drawn, by feature, in drawing order (the same lists as tools/signs/extract_demos.py FACE_GROUPS). */
export const FACE_GROUPS = Object.freeze({
  oval: [10, 297, 284, 389, 454, 361, 397, 379, 400, 152, 176, 150, 172, 132, 234, 162, 54, 67],
  browR: [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
  browL: [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
  eyeR: [33, 160, 159, 158, 133, 153, 145, 144],
  eyeL: [263, 387, 386, 385, 362, 380, 374, 373],
  nose: [6, 195, 4, 2, 98, 327],
  lipsOuter: [61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91],
  lipsInner: [78, 81, 13, 311, 308, 402, 14, 178],
});
// the upper half of the oval, ear to ear over the top of the head: the hair is built on it
const OVAL_TOP = [234, 162, 54, 67, 10, 297, 284, 389, 454];

/* A demo file (tools/signs/extract_demos.py) -> the same object with `values`: Float32Array(frames * channels), in
 * picture heights, plus what the drawing needs once (offsets, the body's size and middle, the framing). Throws on a
 * file that does not hold together, so a broken download fails loudly at load and not as a blank canvas later. */
export function decodeDemo(json) {
  if (json?.values instanceof Float32Array) return json;
  const T = json?.frames | 0, C = json?.channels | 0;
  const pose = json?.pose ?? [], face = json?.face ?? [];
  if (!(T > 0) || C !== 2 * pose.length + 63 + 2 * face.length || !Array.isArray(json.data) || json.data.length !== T * C) {
    throw new Error(`avatar: demo ${json?.id ?? "?"} is not a demo file (frames ${T}, channels ${C}, data ${json?.data?.length})`);
  }
  if (json.encoding !== "delta") throw new Error(`avatar: demo ${json.id}: unknown encoding ${json.encoding}`);
  if (![11, 12, 14].every((id) => pose.includes(id))) throw new Error(`avatar: demo ${json.id}: needs pose points 11, 12 and 14`);
  const s = 1 / (json.scale || 1000);
  const values = new Float32Array(T * C);
  const acc = new Float64Array(C);
  for (let t = 0, i = 0; t < T; t++) {
    for (let c = 0; c < C; c++, i++) { acc[c] += json.data[i]; values[i] = acc[c] * s; }
  }
  const poseOff = {};
  pose.forEach((id, i) => { poseOff[id] = 2 * i; });
  const handOff = 2 * pose.length;
  const faceSlot = new Map(face.map((id, i) => [id, handOff + 63 + 2 * i]));
  const groups = {};
  for (const [k, ids] of Object.entries(FACE_GROUPS)) {
    if (ids.every((id) => faceSlot.has(id))) groups[k] = Int32Array.from(ids, (id) => faceSlot.get(id));
  }
  const ovalTop = OVAL_TOP.every((id) => faceSlot.has(id)) ? Int32Array.from(OVAL_TOP, (id) => faceSlot.get(id)) : null;
  const hold = { start: Math.max(0, +json.hold?.start || 0), end: Math.max(0, +json.hold?.end || 0) };
  const d = { ...json, data: undefined, values, frames: T, channels: C, fps: +json.fps || 30, hold, poseOff, handOff, faceSlot, groups, ovalTop };
  d.body = measureBody(d);
  return d;
}

export async function loadDemo(url, { signal } = {}) {
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error(`avatar: ${url}: HTTP ${res.status}`);
  return decodeDemo(await res.json());
}

const median = (xs) => { const a = Float64Array.from(xs).sort(); return a.length ? a[a.length >> 1] : NaN; };

/* Once per demo: the shoulders' middle and width (the body's unit), the face's size, the hand's palm size (3-D, so a
 * hand turned edge-on keeps its finger widths) and the box the figure needs (in shoulder widths from the shoulders'
 * middle: the head with its hair, the shoulders, the hand wherever it goes). */
function measureBody(d) {
  const { values: v, frames: T, channels: C, poseOff: po, handOff: ho, faceSlot: fs } = d;
  const sw = [], mx = [], my = [], fh = [], fw = [], palm = [];
  for (let t = 0; t < T; t++) {
    const o = t * C;
    const ax = v[o + po[11]], ay = v[o + po[11] + 1], bx = v[o + po[12]], by = v[o + po[12] + 1];
    sw.push(Math.hypot(ax - bx, ay - by)); mx.push((ax + bx) / 2); my.push((ay + by) / 2);
    if (fs.has(10) && fs.has(152) && fs.has(234) && fs.has(454)) {
      const f10 = o + fs.get(10), f152 = o + fs.get(152), f234 = o + fs.get(234), f454 = o + fs.get(454);
      fh.push(Math.hypot(v[f10] - v[f152], v[f10 + 1] - v[f152 + 1]));
      fw.push(Math.hypot(v[f234] - v[f454], v[f234 + 1] - v[f454 + 1]));
    }
    const h = o + ho;
    const dist = (i, j) => Math.hypot(v[h + 3 * i] - v[h + 3 * j], v[h + 3 * i + 1] - v[h + 3 * j + 1], v[h + 3 * i + 2] - v[h + 3 * j + 2]);
    palm.push((dist(5, 0) + dist(9, 0) + dist(13, 0) + dist(17, 0) + dist(5, 17)) / 5);   // tutor/hand-frame.js palmSize
  }
  const u = median(sw), cx = median(mx), cy = median(my);
  const faceH = fh.length ? median(fh) : 0.7 * u, faceW = fw.length ? median(fw) : 0.55 * u;
  // the box: how far up, down and to either side the figure reaches. Not centred on the body: the hand's side gets
  // the room the hand needs and the other side only the shoulder's, so the figure is drawn as large as it can be.
  let top = Infinity, bottom = -Infinity, left = -SHOULDER_BOX * u, right = SHOULDER_BOX * u;
  for (let t = 0; t < T; t++) {
    const o = t * C;
    if (d.groups.oval) {
      for (const k of d.groups.oval) {
        const x = v[o + k] - cx;
        top = Math.min(top, v[o + k + 1] - 0.4 * faceH);
        left = Math.min(left, x - 0.14 * faceW); right = Math.max(right, x + 0.14 * faceW);
      }
    }
    for (let j = 0; j < 21; j++) {
      const x = v[o + ho + 3 * j] - cx, y = v[o + ho + 3 * j + 1];
      top = Math.min(top, y - 0.14 * u); bottom = Math.max(bottom, y + 0.16 * u);
      left = Math.min(left, x - 0.16 * u); right = Math.max(right, x + 0.16 * u);
    }
  }
  if (!Number.isFinite(top)) top = cy - 1.4 * u;
  const view = { left: left / u, right: right / u, top: Math.min(-1.25, (top - cy) / u), bottom: Math.max(0.9, (bottom - cy) / u) };
  return { u, cx, cy, faceH, faceW, palm: median(palm), view };
}
// the shirt reaches this far either side of the shoulders' middle (shoulder widths), with a little room
const SHOULDER_BOX = 0.86;

/* Where a view goes on a W x H canvas: canvas pixels per picture height (k) and the offset (ox, oy), the box centred
 * across and its spare height split 35 % above the head, 65 % below it (more chest, never a floating head). */
export function fitView(body, view, W, H) {
  const u = body.u;
  const bw = (view.right - view.left) * u, bh = (view.bottom - view.top) * u;
  const k = Math.min(W / bw, H / bh);
  const x0 = body.cx + view.left * u, y0 = body.cy + view.top * u;
  return { k, ox: (W - k * bw) / 2 - k * x0, oy: (H - k * bh) * 0.35 - k * y0 };
}

/* The view that fits every one of these demos, so a lesson's signs share one framing (the figure does not grow,
 * shrink or shift from sign to sign). Pass it as createAvatar's { view }, or to setView(). */
export function unionView(demos) {
  const vs = demos.map((d) => decodeDemo(d).body.view);
  return {
    left: Math.min(...vs.map((x) => x.left)), right: Math.max(...vs.map((x) => x.right)),
    top: Math.min(...vs.map((x) => x.top)), bottom: Math.max(...vs.map((x) => x.bottom)),
  };
}

/* ---- colours ------------------------------------------------------------------------------------------------------ */
const PALETTES = {
  light: {
    ink: "#2b2420", skin: "#e5b18b", hair: "#3a2b24", shirt: "#3b5469",
    lips: "#c4766a", hand: "#f5a524", bg: "#ffffff",
  },
  // dark: the hair a lighter brown, or the head's top is lost against the page
  dark: {
    ink: "#120f0d", skin: "#e2ad87", hair: "#6a4d3e", shirt: "#56718b",
    lips: "#c4766a", hand: "#f5a524", bg: "#1f1c19",
  },
};
const TOKENS = {
  ink: ["--avatar-ink"], skin: ["--avatar-skin"], hair: ["--avatar-hair"], shirt: ["--avatar-shirt"],
  lips: ["--avatar-lips"], hand: ["--avatar-hand", "--accent"], handInk: ["--avatar-hand-ink"], bg: ["--avatar-bg", "--panel", "--bg"],
};

function parseColor(s) {
  s = String(s || "").trim();
  let m = s.match(/^#([0-9a-f]{3,8})$/i);
  if (m) {
    let h = m[1];
    if (h.length <= 4) h = [...h].map((c) => c + c).join("");
    return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
  }
  m = s.match(/^rgba?\(\s*([\d.]+)[\s,]+([\d.]+)[\s,]+([\d.]+)/i);
  return m ? [+m[1], +m[2], +m[3]] : null;
}
const mix = (a, b, t) => [0, 1, 2].map((i) => a[i] + (b[i] - a[i]) * t);
const css = (c, alpha = 1) => (alpha >= 1 ? `rgb(${c.map(Math.round).join(",")})` : `rgba(${c.map(Math.round).join(",")},${alpha})`);
const luminance = (c) => (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) / 255;
const WHITE = [255, 255, 255], BLACK = [0, 0, 0];

/* Every colour the drawing uses, as CSS strings (made once per theme, never per frame), plus the hand's two ends for
 * the palm's turn. */
function resolveColors(canvas, theme) {
  const cs = typeof getComputedStyle === "function" && canvas?.isConnected ? getComputedStyle(canvas) : null;
  const token = (names) => { for (const n of names) { const c = parseColor(cs?.getPropertyValue(n)); if (c) return c; } return null; };
  let dark;
  if (theme === "dark" || theme === "light") dark = theme === "dark";
  else {
    const bg = token(["--bg", "--panel"]);
    dark = bg ? luminance(bg) < 0.35 : typeof matchMedia === "function" && matchMedia("(prefers-color-scheme: dark)").matches;
  }
  const base = PALETTES[dark ? "dark" : "light"];
  // a forced "light" or "dark" is that palette as it is; otherwise the page's tokens come first, and an object's own
  // colours before them
  const usePage = theme !== "light" && theme !== "dark";
  const own = theme && typeof theme === "object" ? theme : {};
  const pick = (k) => parseColor(own[k]) || (usePage && token(TOKENS[k] ?? [])) || parseColor(base[k]);
  const ink = pick("ink"), skin = pick("skin"), hair = pick("hair"), shirt = pick("shirt"), lips = pick("lips"), hand = pick("hand");
  const handInk = pick("handInk") ?? mix(hand, [70, 40, 0], 0.72);
  const bg = pick("bg") ?? parseColor(base.bg);
  return {
    dark,
    bg: css(bg), ink: css(ink), inkSoft: css(ink, 0.55), inkFaint: css(ink, 0.32),
    skin: css(skin), skinShade: css(mix(skin, [120, 60, 40], 0.16)), cheek: css([236, 110, 100], dark ? 0.16 : 0.2),
    hair: css(hair), hairShine: css(mix(hair, WHITE, dark ? 0.2 : 0.16)), brow: css(mix(hair, ink, 0.3)), eye: css(mix(hair, ink, 0.6)), eyeShine: css(WHITE, 0.9),
    lips: css(lips), lipsLine: css(mix(lips, [90, 30, 30], 0.45)), mouth: css(mix(lips, [60, 15, 20], 0.72)),
    shirt: css(shirt), shirtShade: css(mix(shirt, BLACK, 0.22)), sleeve: css(mix(shirt, WHITE, 0.12)),
    hand: css(hand), handInk: css(handInk), handLine: css(mix(hand, handInk, 0.55)),
    handRGB: hand, palmRGB: mix(hand, WHITE, 0.5), handShadow: css(BLACK, dark ? 0.32 : 0.14),
  };
}

/* ---- the hand (as experiments/asl-tutor/ghost.js draws its ghost hand) ------------------------------------------- */
const FINGERS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]];
// bone widths at the base, middle and tip joints, in palm sizes (ghost.js WIDTH), a little fuller for a drawn hand
const WIDTH = [[0.33, 0.28, 0.245, 0.215], [0.26, 0.235, 0.21, 0.19], [0.27, 0.245, 0.22, 0.2], [0.255, 0.228, 0.205, 0.188], [0.225, 0.205, 0.186, 0.172]];
const PALM = [0, 1, 5, 9, 13, 17];
const PALM_ROUND = 0.13;      // the palm's corners are rounded by this much (palm sizes)
const IN_FRONT = 0.12;        // a finger goes in front of the palm only when nearer than the palm by this (palm sizes)

/* A tapered capsule (the hull of two circles) added to the current path (ghost.js capsule). */
function capsule(ctx, ax, ay, r1, bx, by, r2) {
  const dx = bx - ax, dy = by - ay, L = Math.hypot(dx, dy);
  if (L <= Math.abs(r1 - r2) + 1e-9) {
    const big = r1 >= r2;
    const cx = big ? ax : bx, cy = big ? ay : by, r = big ? r1 : r2;
    ctx.moveTo(cx + r, cy);
    ctx.arc(cx, cy, r, 0, 2 * Math.PI);
    return;
  }
  const th = Math.atan2(dy, dx), al = Math.acos((r1 - r2) / L);
  ctx.moveTo(ax + r1 * Math.cos(th + al), ay + r1 * Math.sin(th + al));
  ctx.arc(ax, ay, r1, th + al, th - al + 2 * Math.PI);
  ctx.arc(bx, by, r2, th - al, th + al);
  ctx.closePath();
}

/* ---- the controller ----------------------------------------------------------------------------------------------- */
const DPR_MAX = 2;            // canvas pixels per CSS pixel, at most: crisp on a phone without a huge canvas (mobile/app.js)
const FRAME_MS = 1000 / 60;
const RETURN_MS = 280;        // the glide from the last pose back to the first, inside the pause between loops

export function createAvatar(canvas, demoIn, {
  mirror = false, speed = 1, loop = true, pauseMs = 500, theme = "auto", autoplay = true, background = false, view = null,
} = {}) {
  const demo = decodeDemo(demoIn);
  const ctx = canvas.getContext("2d");
  const { frames: T, channels: C, values: V, fps, hold, poseOff: PO, handOff: HO, groups: G, body } = demo;
  const cur = new Float32Array(C);            // the frame being drawn
  const u = body.u;
  const reduce = typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;
  const st = {
    t: 0, playing: false, raf: 0, lastDraw: 0, visible: true, destroyed: false,
    speed: clampSpeed(speed), mirror: !!mirror, loop: !!loop, pauseMs: Math.max(0, +pauseMs || 0),
    colors: null, theme, W: 0, H: 0, dpr: 1, view: view ?? body.view, drawMs: 0, drawN: 0, drawMax: 0,
  };
  const loopCbs = new Set();
  const move = (T - 1) / fps;
  const passLen = () => hold.start + move + hold.end;
  const cycleLen = () => passLen() + (st.loop ? st.pauseMs / 1000 : 0);

  if (!canvas.hasAttribute("role")) canvas.setAttribute("role", "img");
  if (!canvas.hasAttribute("aria-label")) canvas.setAttribute("aria-label", `A signer showing the sign ${demo.gloss ?? demo.id}`);

  function clampSpeed(s) { s = +s; return Number.isFinite(s) && s > 0 ? Math.min(4, Math.max(0.05, s)) : 1; }

  /* -- sampling: pass time -> a frame (Catmull-Rom between the 30 fps samples) -- */
  function sampleAt(f) {
    const i = Math.max(0, Math.min(T - 1, Math.floor(f)));
    const a = f - i;
    const o1 = i * C;
    if (a <= 1e-6 || i >= T - 1) { cur.set(V.subarray(o1, o1 + C)); return; }
    const o0 = Math.max(0, i - 1) * C, o2 = Math.min(T - 1, i + 1) * C, o3 = Math.min(T - 1, i + 2) * C;
    const a2 = a * a, a3 = a2 * a;
    for (let c = 0; c < C; c++) {
      const p0 = V[o0 + c], p1 = V[o1 + c], p2 = V[o2 + c], p3 = V[o3 + c];
      cur[c] = 0.5 * (2 * p1 + (p2 - p0) * a + (2 * p0 - 5 * p1 + 4 * p2 - p3) * a2 + (3 * p1 - p0 - 3 * p2 + p3) * a3);
    }
  }
  function frameFor(t) {
    if (t < hold.start) { sampleAt(0); return; }
    if (t < hold.start + move) { sampleAt((t - hold.start) * fps); return; }
    sampleAt(T - 1);
    const ret = Math.min(RETURN_MS, st.pauseMs * 0.6) / 1000;
    const end = cycleLen();
    if (st.loop && ret > 0 && t > end - ret) {
      const w = (t - (end - ret)) / ret, e = w * w * (3 - 2 * w);       // eased back to the first pose
      for (let c = 0; c < C; c++) cur[c] += (V[c] - cur[c]) * e;
    }
  }

  /* -- size and framing -- */
  function resize() {
    const r = canvas.getBoundingClientRect();
    const dpr = Math.min(DPR_MAX, Math.max(1, globalThis.devicePixelRatio || 1));
    const W = Math.max(1, Math.round(r.width * dpr)), H = Math.max(1, Math.round(r.height * dpr));
    if (W !== canvas.width || H !== canvas.height) { canvas.width = W; canvas.height = H; }
    st.W = W; st.H = H; st.dpr = dpr;
  }
  const transform = () => fitView(body, st.view, st.W, st.H);

  /* -- drawing -- */
  const P = (off) => [cur[off], cur[off + 1]];
  function smoothClosed(idx) {
    const n = idx.length;
    ctx.moveTo((cur[idx[n - 1]] + cur[idx[0]]) / 2, (cur[idx[n - 1] + 1] + cur[idx[0] + 1]) / 2);
    for (let i = 0; i < n; i++) {
      const a = idx[i], b = idx[(i + 1) % n];
      ctx.quadraticCurveTo(cur[a], cur[a + 1], (cur[a] + cur[b]) / 2, (cur[a + 1] + cur[b + 1]) / 2);
    }
    ctx.closePath();
  }

  function render() {
    const t0 = performance.now();
    const col = st.colors;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, st.W, st.H);
    if (background) { ctx.fillStyle = col.bg; ctx.fillRect(0, 0, st.W, st.H); }
    const { k, ox, oy } = transform();
    if (st.mirror) ctx.setTransform(-k, 0, 0, k, st.W - ox, oy);
    else ctx.setTransform(k, 0, 0, k, ox, oy);
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    // outlines: about 1.4 CSS px on a phone-sized figure, growing with it, never under 1 px
    const ink = Math.max((1.0 * st.dpr) / k, 0.0125 * u);

    const [s11x, s11y] = P(PO[11]), [s12x, s12y] = P(PO[12]);
    let ex = s11x - s12x, ey = s11y - s12y;
    const sw = Math.hypot(ex, ey) || u;
    ex /= sw; ey /= sw;                         // across the shoulders, toward the other (non-signing) side
    const dx = -ey, dy = ex;                    // down the body
    const mx = (s11x + s12x) / 2, my = (s11y + s12y) / 2;
    const head = G.oval ? headFrame() : null;
    // a mirrored demo's face points keep their own left and right (extract_demos.py mirrors positions, not names), so
    // the head's "across" can point against the body's: turn it to match, or the neck's sides cross and the hair's
    // sweep and the eyes' shine change side from one sign to the next
    if (head && head.ax * ex + head.ay * ey < 0) { head.ax = -head.ax; head.ay = -head.ay; head.flip = true; }

    // the neck first: its top goes under the face and its foot under the shirt's collar
    if (head) drawNeck(col, ink, head, mx, my, ex, ey, dx, dy);
    drawShirt(col, ink, mx, my, ex, ey, dx, dy, s11x, s11y, s12x, s12y);
    drawRestingArm(col, ink, s11x, s11y, ex, ey, dx, dy);
    if (head) {
      drawEars(col, ink, head);
      drawFace(col, ink, head);
    }
    drawArm(col, ink, s12x, s12y, dx, dy);
    drawHand(col, ink);
    const ms = performance.now() - t0;
    st.drawMs += ms; st.drawN++; st.drawMax = Math.max(st.drawMax, ms);
  }

  /* The head's own axes this frame: centre, up (chin -> forehead), across, height and width. */
  function headFrame() {
    const fs = demo.faceSlot;
    const [tx, ty] = P(fs.get(10)), [bx, by] = P(fs.get(152)), [lx, ly] = P(fs.get(234)), [rx, ry] = P(fs.get(454));
    const h = Math.hypot(tx - bx, ty - by) || body.faceH, w = Math.hypot(rx - lx, ry - ly) || body.faceW;
    const ux = (tx - bx) / h, uy = (ty - by) / h;           // up the face
    let ax = -uy, ay = ux;                                   // across the face
    if (ax * (rx - lx) + ay * (ry - ly) < 0) { ax = -ax; ay = -ay; }   // pointing from 234 to 454 (render() may turn it)
    return { cx: (tx + bx) / 2, cy: (ty + by) / 2, ux, uy, ax, ay, h, w, lx, ly, rx, ry, bx, by, flip: false };
  }

  /* The shirt: MediaPipe's shoulder points are the joints, so the outline is built round them -- the collar a little
   * above the joints' line, the top of the shoulders sloping gently out to the joints and rounding over them, the
   * sides falling straight past the bottom of the canvas. */
  function drawShirt(col, ink, mx, my, ex, ey, dx, dy, s11x, s11y, s12x, s12y) {
    // per side (-1: the signing shoulder, 12; +1: the other, 11): collar end, the shoulder's top curve, the arm's side
    // the joints sit well inside a body's outline (the signers' shoulders, arms at their sides, are about three faces
    // wide, the joints under two), so the outline reaches about a quarter of a shoulder width past each joint
    const pts = (sg, sx, sy) => {
      const nx = mx + sg * 0.16 * u * ex - 0.11 * u * dx, ny = my + sg * 0.16 * u * ey - 0.11 * u * dy;
      const c1x = nx + sg * 0.24 * u * ex + 0.015 * u * dx, c1y = ny + sg * 0.24 * u * ey + 0.015 * u * dy;
      const c2x = sx + sg * 0.17 * u * ex - 0.13 * u * dx, c2y = sy + sg * 0.17 * u * ey - 0.13 * u * dy;
      const ox = sx + sg * 0.25 * u * ex + 0.1 * u * dx, oy = sy + sg * 0.25 * u * ey + 0.1 * u * dy;
      const bx = ox + 3 * u * dx - sg * 0.08 * u * ex, by = oy + 3 * u * dy - sg * 0.08 * u * ey;
      return { nx, ny, c1x, c1y, c2x, c2y, ox, oy, bx, by };
    };
    const L = pts(-1, s12x, s12y), R = pts(1, s11x, s11y);
    const cx = mx + 0.03 * u * dx, cy = my + 0.03 * u * dy;      // the crew neck: a shallow curve between the collar's ends
    ctx.beginPath();
    ctx.moveTo(L.nx, L.ny);
    ctx.bezierCurveTo(L.c1x, L.c1y, L.c2x, L.c2y, L.ox, L.oy);
    ctx.lineTo(L.bx, L.by);
    ctx.lineTo(R.bx, R.by);
    ctx.lineTo(R.ox, R.oy);
    ctx.bezierCurveTo(R.c2x, R.c2y, R.c1x, R.c1y, R.nx, R.ny);
    ctx.quadraticCurveTo(cx, cy, L.nx, L.ny);
    ctx.closePath();
    ctx.fillStyle = col.shirt;
    ctx.fill();
    ctx.strokeStyle = col.ink;
    ctx.lineWidth = ink;
    ctx.stroke();
    // the collar's band
    ctx.beginPath();
    ctx.moveTo(R.nx, R.ny);
    ctx.quadraticCurveTo(cx, cy, L.nx, L.ny);
    ctx.strokeStyle = col.shirtShade;
    ctx.lineWidth = 0.045 * u;
    ctx.stroke();
  }

  function drawNeck(col, ink, hd, mx, my, ex, ey, dx, dy) {
    // from under the jaw (the head's own axes) to under the collar (the body's), a little narrower at the top
    const tw = 0.29 * hd.w, bw = 0.17 * u;
    const tx = hd.bx + hd.ux * 0.3 * hd.h, ty = hd.by + hd.uy * 0.3 * hd.h;
    const bx = mx - 0.02 * u * dx, by = my - 0.02 * u * dy;
    ctx.beginPath();
    ctx.moveTo(tx - hd.ax * tw, ty - hd.ay * tw);
    ctx.lineTo(tx + hd.ax * tw, ty + hd.ay * tw);
    ctx.lineTo(bx + ex * bw, by + ey * bw);
    ctx.lineTo(bx - ex * bw, by - ey * bw);
    ctx.closePath();
    ctx.fillStyle = col.skinShade;
    ctx.fill();
    ctx.strokeStyle = col.ink;
    ctx.lineWidth = ink;
    ctx.beginPath();
    ctx.moveTo(tx - hd.ax * tw, ty - hd.ay * tw); ctx.lineTo(bx - ex * bw, by - ey * bw);
    ctx.moveTo(tx + hd.ax * tw, ty + hd.ay * tw); ctx.lineTo(bx + ex * bw, by + ey * bw);
    ctx.stroke();
  }

  function drawEars(col, ink, hd) {
    // behind the face, on its sides at eye-to-nose height; a turned head hides one of them behind the face
    const ang = Math.atan2(hd.ay, hd.ax);
    ctx.fillStyle = col.skin;
    ctx.strokeStyle = col.ink;
    ctx.lineWidth = ink;
    for (const [px, py, sg] of [[hd.lx, hd.ly, -1], [hd.rx, hd.ry, 1]]) {
      const cx = px + sg * hd.ax * 0.04 * hd.w - hd.ux * 0.06 * hd.h, cy = py + sg * hd.ay * 0.04 * hd.w - hd.uy * 0.06 * hd.h;
      ctx.beginPath();
      ctx.ellipse(cx, cy, 0.085 * hd.w, 0.14 * hd.h, ang, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
    }
  }

  function drawFace(col, ink, hd) {
    // the face
    ctx.beginPath();
    smoothClosed(G.oval);
    ctx.fillStyle = col.skin;
    ctx.fill();
    ctx.strokeStyle = col.ink;
    ctx.lineWidth = ink;
    ctx.stroke();
    // cheeks: a faint warmth under each eye, toward the outside
    if (G.eyeR && G.eyeL) {
      ctx.fillStyle = col.cheek;
      for (const e of [G.eyeR, G.eyeL]) {
        const [ox, oy] = P(e[0]), [ix, iy] = P(e[4]);
        const cx = ox + (ox - ix) * 0.05 - hd.ux * 0.2 * hd.h, cy = oy + (oy - iy) * 0.05 - hd.uy * 0.2 * hd.h;
        ctx.beginPath();
        ctx.ellipse(cx, cy, 0.11 * hd.w, 0.065 * hd.h, Math.atan2(hd.ay, hd.ax), 0, 2 * Math.PI);
        ctx.fill();
      }
    }
    drawHair(col, ink, hd);
    // brows: filled, tapered shapes, a little bolder than the mesh's, as brows read from afar
    for (const b of [G.browR, G.browL]) {
      if (!b) continue;
      ctx.beginPath();
      smoothClosed(b);
      ctx.fillStyle = col.brow;
      ctx.fill();
      ctx.strokeStyle = col.brow;
      ctx.lineWidth = 0.022 * hd.h;
      ctx.stroke();
    }
    // eyes: dark ovals, as open as the signer's
    for (const e of [G.eyeR, G.eyeL]) {
      if (!e) continue;
      const [ox, oy] = P(e[0]), [ix, iy] = P(e[4]);
      const [tx, ty] = P(e[2]), [bx, by] = P(e[6]);
      const ew = Math.hypot(ox - ix, oy - iy) || 0.2 * hd.w;
      const open = Math.hypot(tx - bx, ty - by) / ew;            // ~0.3 open, under 0.1 closed
      const cx = (ox + ix + tx + bx) / 4, cy = (oy + iy + ty + by) / 4;
      const rx = 0.22 * ew, ry = Math.max(0.035, Math.min(1, (open - 0.08) / 0.22)) * 0.3 * ew;
      const ang = Math.atan2(iy - oy, ix - ox);
      ctx.beginPath();
      ctx.ellipse(cx, cy, rx, Math.max(ry, 0.02 * ew), ang, 0, 2 * Math.PI);
      ctx.fillStyle = col.eye;
      ctx.fill();
      if (ry > 0.12 * ew) {
        ctx.beginPath();
        ctx.arc(cx - 0.06 * ew * hd.ax + 0.09 * ew * hd.ux, cy - 0.06 * ew * hd.ay + 0.09 * ew * hd.uy, 0.055 * ew, 0, 2 * Math.PI);
        ctx.fillStyle = col.eyeShine;
        ctx.fill();
      }
    }
    // nose: a small curve under its tip, its ends turned up, and a soft line down one side (a whole nostril-to-nostril
    // line read as a moustache)
    if (G.nose) {
      const n = G.nose;   // 6 bridge, 195, 4 tip, 2 base, 98, 327 wings
      const bx = cur[n[3]], by = cur[n[3] + 1], up = 0.012 * hd.h;
      const lx = bx + (cur[n[4]] - bx) * 0.55 + hd.ux * up, ly = by + (cur[n[4] + 1] - by) * 0.55 + hd.uy * up;
      const rx = bx + (cur[n[5]] - bx) * 0.55 + hd.ux * up, ry = by + (cur[n[5] + 1] - by) * 0.55 + hd.uy * up;
      ctx.strokeStyle = col.inkSoft;
      ctx.lineWidth = ink * 0.85;
      ctx.beginPath();
      ctx.moveTo(lx, ly);
      ctx.quadraticCurveTo(bx - hd.ux * 1.6 * up, by - hd.uy * 1.6 * up, rx, ry);
      ctx.stroke();
      const side = (cur[n[2]] - hd.cx) * hd.ax + (cur[n[2] + 1] - hd.cy) * hd.ay < 0 ? n[5] : n[4];
      ctx.strokeStyle = col.inkFaint;
      ctx.beginPath();
      ctx.moveTo(cur[n[1]], cur[n[1] + 1]);
      ctx.quadraticCurveTo(cur[n[2]], cur[n[2] + 1], (cur[n[2]] * 2 + cur[side]) / 3, (cur[n[2] + 1] * 2 + cur[side + 1]) / 3);
      ctx.stroke();
    }
    // lips, and the mouth's opening when it opens
    if (G.lipsOuter && G.lipsInner) {
      ctx.beginPath();
      smoothClosed(G.lipsOuter);
      ctx.fillStyle = col.lips;
      ctx.fill();
      ctx.beginPath();
      smoothClosed(G.lipsInner);
      ctx.fillStyle = col.mouth;
      ctx.fill();
      ctx.strokeStyle = col.lipsLine;
      ctx.lineWidth = ink * 0.8;
      ctx.stroke();
    }
  }

  /* Short hair, built on the upper half of the face oval: a dome pushed out from the face's top, and a hairline a
   * little under the top of the forehead. It is drawn, not tracked, so it only follows the head. */
  // face heights outward, ear to ear: close at the temples, fuller on top
  const HAIR_OUT = [0.02, 0.07, 0.18, 0.26, 0.28, 0.26, 0.19, 0.08, 0.02];
  // the hairline, face heights inward: a fringe swept to one side
  const HAIR_IN = [0.0, 0.06, 0.13, 0.15, 0.12, 0.08, 0.06, 0.03, 0.0];
  function drawHair(col, ink, hd) {
    const top = demo.ovalTop;
    if (!top) return;
    const n = top.length;
    const out = new Array(n), inn = new Array(n);
    for (let i = 0; i < n; i++) {
      const x = cur[top[i]], y = cur[top[i] + 1];
      let vx = x - hd.cx, vy = y - hd.cy;
      const L = Math.hypot(vx, vy) || 1;
      vx /= L; vy /= L;
      const j = hd.flip ? n - 1 - i : i;          // the sweep goes the same way in every sign
      out[i] = [x + vx * HAIR_OUT[j] * hd.h, y + vy * HAIR_OUT[j] * hd.h];
      inn[i] = [x - vx * HAIR_IN[j] * hd.h, y - vy * HAIR_IN[j] * hd.h];
    }
    ctx.beginPath();
    ctx.moveTo(out[0][0], out[0][1]);
    for (let i = 1; i < n - 1; i++) ctx.quadraticCurveTo(out[i][0], out[i][1], (out[i][0] + out[i + 1][0]) / 2, (out[i][1] + out[i + 1][1]) / 2);
    ctx.lineTo(out[n - 1][0], out[n - 1][1]);
    ctx.lineTo(inn[n - 1][0], inn[n - 1][1]);
    for (let i = n - 2; i > 0; i--) ctx.quadraticCurveTo(inn[i][0], inn[i][1], (inn[i][0] + inn[i - 1][0]) / 2, (inn[i][1] + inn[i - 1][1]) / 2);
    ctx.lineTo(inn[0][0], inn[0][1]);
    ctx.closePath();
    ctx.fillStyle = col.hair;
    ctx.fill();
    ctx.strokeStyle = col.ink;
    ctx.lineWidth = ink;
    ctx.stroke();
    // a soft shine across the top
    const at = (i, f) => [out[i][0] + (inn[i][0] - out[i][0]) * f, out[i][1] + (inn[i][1] - out[i][1]) * f];
    const k = (i) => (hd.flip ? n - 1 - i : i);
    const [ax, ay] = at(k(5), 0.34), [cx, cy] = at(k(6), 0.3), [ex, ey] = at(k(7), 0.42);
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.quadraticCurveTo(cx, cy, ex, ey);
    ctx.strokeStyle = col.hairShine;
    ctx.lineWidth = 0.035 * hd.h;
    ctx.stroke();
  }

  /* The other arm, resting at the side. It is not tracked (the signers held their phone in it), so it hangs from its
   * shoulder, a little away from the body, and goes out of the picture: a figure with one arm reads as a diagram. */
  function drawRestingArm(col, ink, sx, sy, ex, ey, dx, dy) {
    const ax = sx + 0.06 * u * ex + 0.05 * u * dx, ay = sy + 0.06 * u * ey + 0.05 * u * dy;
    const bx = ax + 0.14 * u * ex + 1.6 * u * dx, by = ay + 0.14 * u * ey + 1.6 * u * dy;
    const w = 0.27 * u;
    ctx.strokeStyle = col.ink;
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.lineWidth = w + 2 * ink; ctx.stroke();
    ctx.strokeStyle = col.shirt;
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.lineWidth = w; ctx.stroke();
  }

  function drawArm(col, ink, sx, sy, dx, dy) {
    // the signing arm in its sleeve: shoulder -> elbow -> just short of the wrist (the hand covers the cuff)
    const [elx, ely] = P(PO[14]);
    const wx = cur[HO], wy = cur[HO + 1];
    const fx = wx + (elx - wx) * 0.08, fy = wy + (ely - wy) * 0.08;
    const ax = sx + 0.04 * u * dx, ay = sy + 0.04 * u * dy;
    const up = 0.3 * u, fore = 0.22 * u;
    ctx.strokeStyle = col.ink;
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(elx, ely); ctx.lineWidth = up + 2 * ink; ctx.stroke();
    ctx.beginPath(); ctx.moveTo(elx, ely); ctx.lineTo(fx, fy); ctx.lineWidth = fore + 2 * ink; ctx.stroke();
    // a shade lighter than the shirt: the arm is in front of the body, and it must read against it
    ctx.strokeStyle = col.sleeve;
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(elx, ely); ctx.lineWidth = up; ctx.stroke();
    ctx.beginPath(); ctx.moveTo(elx, ely); ctx.lineTo(fx, fy); ctx.lineWidth = fore; ctx.stroke();
    // the cuff
    ctx.strokeStyle = col.shirtShade;
    ctx.lineWidth = fore * 0.35;
    const cx = fx + (elx - fx) * 0.12, cy = fy + (ely - fy) * 0.12;
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(fx, fy); ctx.stroke();
  }

  const order = [0, 1, 2, 3, 4, 5];          // painting order of the palm (0) and fingers (1..5), reused every frame
  const depth = new Float32Array(6);
  /* A joint nearer the camera than the wrist is drawn a little thicker, as the camera saw it: the signers held their
   * phone at arm's length and signed toward it, so a finger pointing at the lens looked fat and short (NO, YES). */
  const persp = (o, s) => Math.max(0.85, Math.min(1.35, 1 - (0.3 * (cur[o + 2] - cur[HO + 2])) / s));
  function handPath(part, s, grow) {
    const h = HO;
    ctx.beginPath();
    if (part === 0) {
      let f = 0;
      for (let k = 0; k < PALM.length; k++) {
        const o = h + 3 * PALM[k];
        f += persp(o, s);
        if (k) ctx.lineTo(cur[o], cur[o + 1]); else ctx.moveTo(cur[o], cur[o + 1]);
      }
      ctx.closePath();
      return (PALM_ROUND * s * f) / PALM.length + grow;
    }
    const f = FINGERS[part - 1], w = WIDTH[part - 1];
    for (let b = 0; b < 3; b++) {
      const a = h + 3 * f[b], c = h + 3 * f[b + 1];
      capsule(ctx, cur[a], cur[a + 1], (w[b] * s * persp(a, s)) / 2 + grow, cur[c], cur[c + 1], (w[b + 1] * s * persp(c, s)) / 2 + grow);
    }
    return grow;
  }
  /* The whole hand as ONE path, filled once, so its translucent shadow is even where parts overlap: the capsules all
   * wind the same way, and the palm (grown by moving its corners out from its middle) is turned to wind the same way
   * too, or nonzero filling would leave holes where they cross. */
  function handShadow(s, grow, offX, offY, color) {
    const h = HO;
    let cx = 0, cy = 0;
    for (const j of PALM) { cx += cur[h + 3 * j]; cy += cur[h + 3 * j + 1]; }
    cx /= PALM.length; cy /= PALM.length;
    let area = 0;
    for (let k = 0; k < PALM.length; k++) {
      const a = h + 3 * PALM[k], b = h + 3 * PALM[(k + 1) % PALM.length];
      area += cur[a] * cur[b + 1] - cur[b] * cur[a + 1];
    }
    const pad = PALM_ROUND * s + grow;
    ctx.save();
    ctx.translate(offX, offY);
    ctx.beginPath();
    for (let k = 0; k < PALM.length; k++) {
      const j = PALM[area >= 0 ? k : PALM.length - 1 - k], o = h + 3 * j;
      const vx = cur[o] - cx, vy = cur[o + 1] - cy, L = Math.hypot(vx, vy) || 1;
      const x = cur[o] + (vx / L) * pad, y = cur[o + 1] + (vy / L) * pad;
      if (k) ctx.lineTo(x, y); else ctx.moveTo(x, y);
    }
    ctx.closePath();
    for (let i = 0; i < 5; i++) {
      const f = FINGERS[i], w = WIDTH[i];
      for (let b = 0; b < 3; b++) {
        const a = h + 3 * f[b], c = h + 3 * f[b + 1];
        capsule(ctx, cur[a], cur[a + 1], (w[b] * s * persp(a, s)) / 2 + grow, cur[c], cur[c + 1], (w[b + 1] * s * persp(c, s)) / 2 + grow);
      }
    }
    ctx.fillStyle = color;
    ctx.fill("nonzero");
    ctx.restore();
  }

  function drawHand(col, ink) {
    const h = HO, s = body.palm;
    // which way the palm faces: the normal of (index knuckle - wrist) x (pinky knuckle - wrist); a right hand (every
    // demo is one) shows its palm to the camera when the normal points at it (z < 0: MediaPipe's z grows away)
    const ax = cur[h + 15] - cur[h], ay = cur[h + 16] - cur[h + 1], az = cur[h + 17] - cur[h + 2];
    const bx = cur[h + 51] - cur[h], by = cur[h + 52] - cur[h + 1], bz = cur[h + 53] - cur[h + 2];
    const nz = ax * by - ay * bx, nl = Math.hypot(ay * bz - az * by, az * bx - ax * bz, nz) || 1;
    const facing = Math.max(0, Math.min(1, (-nz / nl + 0.15) / 0.65));
    const palmFill = css(mix(col.handRGB, col.palmRGB, facing * facing * (3 - 2 * facing)));
    // depth order, farthest first; a finger counts as in front of the palm only when clearly nearer
    let pd = 0;
    for (const j of PALM) pd += cur[h + 3 * j + 2];
    pd /= PALM.length;
    depth[0] = pd;
    for (let i = 0; i < 5; i++) {
      const f = FINGERS[i];
      const d = (cur[h + 3 * f[1] + 2] + cur[h + 3 * f[2] + 2] + cur[h + 3 * f[3] + 2]) / 3;
      depth[i + 1] = d < pd - IN_FRONT * s ? d : pd + 1e-6 * (i + 1);
    }
    order.sort((a, b) => depth[b] - depth[a] || a - b);
    // 0. a soft shadow a little below and to the side: the hand is in front of the face and body, and reads so
    handShadow(s, ink * 1.15, 0.022 * u, 0.034 * u, col.handShadow);
    // 1. one clean outline round the whole hand
    ctx.strokeStyle = col.handInk;
    ctx.fillStyle = col.handInk;
    for (const p of order) {
      const pad = handPath(p, s, 0);
      ctx.lineWidth = 2 * (pad + ink * 1.15);
      ctx.stroke();
      ctx.fill();
    }
    // 2. the parts back to front, each with a thin line round it that shows only where it crosses another part
    for (const p of order) {
      const pad = handPath(p, s, 0);
      const fill = p === 0 ? palmFill : col.hand;
      ctx.strokeStyle = col.handLine;
      ctx.lineWidth = 2 * pad + ink * 0.9;
      ctx.stroke();
      ctx.fillStyle = fill;
      if (pad > 0) { ctx.strokeStyle = fill; ctx.lineWidth = 2 * pad; ctx.stroke(); }
      ctx.fill();
    }
  }

  /* -- the clock -- */
  function frame(now) {
    st.raf = 0;
    if (st.destroyed) return;
    if (st.playing) {
      // 60 fps is plenty, even on a 120 Hz screen; the time is measured from the last DRAWN frame, so a skipped one
      // is not lost (the sign would play at half speed on such a screen)
      if (st.lastDraw && now - st.lastDraw < FRAME_MS - 2) { schedule(); return; }
      const dt = st.lastDraw ? Math.min(0.1, (now - st.lastDraw) / 1000) : 0;
      st.lastDraw = now;
      st.t += dt * st.speed;
      const end = cycleLen();
      if (st.t >= (st.loop ? end : passLen())) {
        if (st.loop) {
          st.t %= end;
          for (const cb of loopCbs) { try { cb(); } catch (err) { console.warn("avatar: onLoop", err); } }
        } else {
          st.t = passLen();
          st.playing = false;
          for (const cb of loopCbs) { try { cb(); } catch (err) { console.warn("avatar: onLoop", err); } }
        }
      }
    }
    draw();
    if (st.playing) schedule();
  }
  function schedule() {
    if (!st.raf && st.visible && !st.destroyed && !(typeof document !== "undefined" && document.hidden)) st.raf = requestAnimationFrame(frame);
  }
  function draw() {
    if (!st.W || !st.H) resize();
    if (!st.colors) st.colors = resolveColors(canvas, st.theme);
    frameFor(st.t);
    render();
  }

  /* -- what the page sees -- */
  const ro = typeof ResizeObserver === "function" ? new ResizeObserver(() => { resize(); if (!st.playing) draw(); }) : null;
  ro?.observe(canvas);
  const io = typeof IntersectionObserver === "function" ? new IntersectionObserver((es) => {
    st.visible = es.some((e) => e.isIntersecting);
    if (st.visible && st.playing) { st.lastDraw = 0; schedule(); }
  }) : null;
  io?.observe(canvas);
  const onVis = () => { if (!document.hidden && st.playing) { st.lastDraw = 0; schedule(); } };
  if (typeof document !== "undefined") document.addEventListener("visibilitychange", onVis);
  const mqDark = typeof matchMedia === "function" ? matchMedia("(prefers-color-scheme: dark)") : null;
  const onScheme = () => { if (st.theme === "auto" || st.theme == null) { st.colors = null; if (!st.playing) draw(); } };
  mqDark?.addEventListener?.("change", onScheme);

  const api = {
    play() {
      if (st.destroyed || st.playing) return;
      if (!st.loop && st.t >= passLen()) st.t = 0;
      st.playing = true; st.lastDraw = 0;
      schedule();
    },
    pause() { st.playing = false; if (st.raf) { cancelAnimationFrame(st.raf); st.raf = 0; } },
    setSpeed(s) { st.speed = clampSpeed(s); },
    setMirror(b) { st.mirror = !!b; if (!st.playing) draw(); },
    setLoop(b) { st.loop = !!b; },
    setTheme(t) { st.theme = t; st.colors = null; if (!st.playing) draw(); },
    setView(vw) { st.view = vw ?? body.view; if (!st.playing) draw(); },
    /* t: seconds into the pass (0 .. duration; when looping, on into the pause and the glide back), clamped; drawn at
     * once */
    seek(t) { st.t = Math.max(0, Math.min(st.loop ? cycleLen() : passLen(), +t || 0)); draw(); },
    /* the key frame: where the hand has arrived and holds its shape */
    seekKey() { api.seek(hold.start + (demo.key ?? 0) / fps); },
    onLoop(cb) { loopCbs.add(cb); return () => loopCbs.delete(cb); },
    destroy() {
      st.destroyed = true;
      api.pause();
      ro?.disconnect(); io?.disconnect();
      if (typeof document !== "undefined") document.removeEventListener("visibilitychange", onVis);
      mqDark?.removeEventListener?.("change", onScheme);
      loopCbs.clear();
    },
    get playing() { return st.playing; },
    get time() { return st.t; },
    get duration() { return passLen(); },
    /* a whole loop: the pass, the pause on the last pose and the glide back (= duration when not looping) */
    get cycle() { return cycleLen(); },
    get reducedMotion() { return reduce; },
    get demo() { return demo; },
    /* drawing cost so far, ms: { avg, max, n } (performance.now() around each frame's drawing) */
    get stats() { return { avg: st.drawN ? st.drawMs / st.drawN : 0, max: st.drawMax, n: st.drawN }; },
    resetStats() { st.drawMs = 0; st.drawN = 0; st.drawMax = 0; },
  };

  resize();
  if (reduce || !autoplay) api.seekKey();
  else { draw(); api.play(); }
  return api;
}
