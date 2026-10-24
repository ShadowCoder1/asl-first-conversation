/* motion.js: J and Z, the two letters that are MOVEMENTS.
 *
 * Every other letter is a handshape held still, graded by the model. J is an
 * I (pinky out) that draws a J in the air; Z is an index finger that draws a
 * Z. There is no public data of either as landmark sequences, so this is a
 * RULE, stated in numbers, not a trained model:
 *
 *   shape  a finger is "out" when its tip is at least EXT times as far from
 *          the wrist as its knuckle, "curled" when at most CURL times. On the
 *          owner's recorded holds straight fingers read 1.49-2.05 and curled
 *          ones 0.38-1.13 (research/allletters-2026-09-22/).
 *            J: pinky out; index, middle, ring curled.
 *            Z: index out; middle, ring, pinky curled.
 *   motion over the frames in that shape during the few seconds before the
 *          hand came to rest, the moving fingertip (smoothed) must have:
 *            J: travelled at least J_DROP palm-lengths down and J_HOOK across.
 *            Z: spanned Z_WIDTH across and Z_DROP down, and reversed its
 *               sideways direction at least twice (right, back, right).
 *
 * Palm-length = wrist to middle knuckle, so the rule does not care how far
 * the hand is from the camera. The movement is checked on every frame of a J
 * or Z trial and a recognised one is the answer at once, with no hold
 * (experiments/asl-tutor/engine.js, JT 2026-09-24); holding still is only the
 * fallback that grades a movement that was not recognised. */

export const MOTION_LETTERS = Object.freeze(["J", "Z"]);
export const EXT = 1.45;
export const CURL = 1.3;
export const WINDOW_MS = 3500;       // how far back from the hold to look for the movement
export const MIN_SHAPE_FRAMES = 6;
export const J_DROP = 0.45, J_HOOK = 0.25;
export const Z_WIDTH = 0.6, Z_DROP = 0.35, Z_REVERSALS = 2, Z_HYST = 0.15;
/* The same numbers as one object: the engine takes it as an option, so a
 * replay of recorded trials can try other values (research/prolific-pilot-2026-09-28). */
export const MOTION_DEFAULTS = Object.freeze({ WINDOW_MS, EXT, CURL, MIN_SHAPE_FRAMES, J_DROP, J_HOOK, Z_WIDTH, Z_DROP, Z_REVERSALS, Z_HYST });

const FINGER = { index: [5, 8], middle: [9, 12], ring: [13, 16], pinky: [17, 20] };
const SHAPE = {
  J: { out: ["pinky"], curled: ["index", "middle", "ring"], tip: 20 },
  Z: { out: ["index"], curled: ["middle", "ring", "pinky"], tip: 8 },
};

const pt = (flat, i, aspect) => [flat[3 * i] * aspect, flat[3 * i + 1]];
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

export function extension(flat, finger, aspect) {
  const [m, t] = FINGER[finger];
  const w = pt(flat, 0, aspect);
  const d0 = dist(pt(flat, m, aspect), w);
  return d0 > 0 ? dist(pt(flat, t, aspect), w) / d0 : NaN;
}

export function inShape(letter, flat, aspect, p = MOTION_DEFAULTS) {
  const s = SHAPE[letter];
  if (!s || !flat || flat.some((v) => !Number.isFinite(v))) return false;
  return s.out.every((f) => extension(flat, f, aspect) >= p.EXT) && s.curled.every((f) => extension(flat, f, aspect) <= p.CURL);
}

function smooth(points, k = 5) {
  return points.map((_, i) => {
    const lo = Math.max(0, i - (k >> 1)), hi = Math.min(points.length, i + (k >> 1) + 1);
    let x = 0, y = 0;
    for (let j = lo; j < hi; j++) { x += points[j][0]; y += points[j][1]; }
    return [x / (hi - lo), y / (hi - lo)];
  });
}

/* Sideways direction changes of a 1-D track, counting only swings larger than
 * `hyst` (so tracker jitter is not a reversal). */
export function reversals(xs, hyst) {
  let dir = 0, anchor = xs[0], count = 0;
  for (const x of xs) {
    if (dir >= 0 && x < anchor - hyst) { if (dir > 0) count++; dir = -1; anchor = x; }
    else if (dir <= 0 && x > anchor + hyst) { if (dir < 0) count++; dir = 1; anchor = x; }
    else if ((dir > 0 && x > anchor) || (dir < 0 && x < anchor)) anchor = x;
  }
  return count;
}

/**
 * @param {"J"|"Z"} letter
 * @param {{tMs:number, flat:number[]|null}[]} frames  the frames before and during the hold
 * @param {number} aspect
 * @param {object} [p]  thresholds (MOTION_DEFAULTS)
 * @returns {{ok:boolean, reason:string|null, stats:object}}
 */
export function motionVerdict(letter, frames, aspect, p = MOTION_DEFAULTS) {
  const s = SHAPE[letter];
  const kept = frames.filter((f) => inShape(letter, f.flat, aspect, p));
  if (kept.length < p.MIN_SHAPE_FRAMES) return { ok: false, reason: "shape", stats: { shapeFrames: kept.length } };
  const palm = kept.map((f) => dist(pt(f.flat, 0, aspect), pt(f.flat, 9, aspect))).sort((a, b) => a - b)[kept.length >> 1];
  const tip = smooth(kept.map((f) => pt(f.flat, s.tip, aspect))).map(([x, y]) => [x / palm, y / palm]);
  const xs = tip.map((p) => p[0]), ys = tip.map((p) => p[1]);
  const width = Math.max(...xs) - Math.min(...xs);
  const stats = { shapeFrames: kept.length, width: +width.toFixed(3) };
  if (letter === "J") {
    // down, then across: the lowest point must come after a drop from the start
    // region, and there must be sideways travel around/after it
    // the drop: from the highest point to the lowest one AFTER it; the hook:
    // sideways travel from where 70% of that drop is done onwards (jitter
    // makes "the lowest frame" a poor place to start measuring from)
    const top = ys.indexOf(Math.min(...ys));
    const after = ys.slice(top);
    const drop = Math.max(...after) - ys[top];
    const i70 = top + Math.max(0, after.findIndex((y) => y - ys[top] >= 0.7 * drop));
    const tail = xs.slice(i70);
    const hook = Math.max(...tail) - Math.min(...tail);
    Object.assign(stats, { drop: +drop.toFixed(3), hook: +hook.toFixed(3) });
    const ok = drop >= p.J_DROP && hook >= p.J_HOOK;
    return { ok, reason: ok ? null : "motion", stats };
  }
  const drop = Math.max(...ys) - Math.min(...ys);
  const rev = reversals(xs, p.Z_HYST);
  Object.assign(stats, { drop: +drop.toFixed(3), reversals: rev });
  const ok = width >= p.Z_WIDTH && drop >= p.Z_DROP && rev >= p.Z_REVERSALS;
  return { ok, reason: ok ? null : "motion", stats };
}

export const MOTION_HINTS = Object.freeze({
  J: { shape: "For J, start with an I: only your pinky up, the other fingers curled.", motion: "For J, draw a J in the air with your pinky: down, then curve it round." },
  Z: { shape: "For Z, point with your index finger only; curl the others.", motion: "For Z, draw a Z in the air with your index finger: across, back diagonally, across again." },
});
