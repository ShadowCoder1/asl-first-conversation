/* tracker.js: a thin wrapper around MediaPipe Tasks Vision.
 *
 * Two things happen here:
 *   1. we download the WASM runtime + the model file (a few MB, cached after
 *      the first visit), and
 *   2. we expose one simple call, `track(video, timestampMs)`, that returns
 *      the landmarks for the current video frame.
 *
 * To use a different body part, add a case to createTracker() below.
 * Everything else in the codebase is tracker-agnostic. */

import { MEDIAPIPE } from "../../config.js";

const CDN = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE.version}`;

let visionFileset = null;   // shared WASM runtime, loaded once

/**
 * Build a landmark tracker.
 *
 * @param {"hand"|"pose"} kind
 * @param {object} options  passed through to MediaPipe, e.g. { numHands: 2 }
 * @returns {Promise<{kind:string, numLandmarks:number, track:Function, close:Function}>}
 *
 * The returned `track(video, tMs)` gives you:
 *   {
 *     landmarks:      [[{x,y,z}, ...21], ...]   image coords, 0..1, for drawing
 *     worldLandmarks: [[{x,y,z}, ...21], ...]   metres, origin at hand center,
 *                                               use these for real measurements
 *     handedness:     ["Left"|"Right", ...]     (hand tracker only)
 *     handednessScore:[0..1 | null, ...]        how sure it is of each label
 *   }
 * A NOTE ABOUT `handedness`: measured on this platform's UNMIRRORED camera
 * image (research/live/REPORT.md section 1), a physical RIGHT hand is reported
 * "Right" (a mirrored/hflipped frame reports the opposite). It is passed on
 * and saved exactly as reported rather than corrected, so that code which works
 * the hand out from the geometry -- which needs no assumption about mirroring,
 * and stays right even when the label flips at frame edges -- has something
 * independent to check itself against.
 * Each outer array has one entry per detected hand/person. It is EMPTY when
 * nothing was detected in that frame, always check `.landmarks.length`.
 */
export async function createTracker(kind = "hand", options = {}) {
  const { FilesetResolver, HandLandmarker, PoseLandmarker } =
    await import(`${CDN}/vision_bundle.mjs`);

  /* options.fileset / options.modelAssetBuffer (the phone version, mobile/app.js): the runtime and the model already
   * downloaded by the caller (with a progress bar), handed over as they are so nothing is fetched twice. Without
   * them, as always: MediaPipe fetches both itself. */
  if (!visionFileset) {
    visionFileset = options.fileset ?? await FilesetResolver.forVisionTasks(`${CDN}/wasm`);
  }

  if (kind === "hand") {
    const { landmarker, delegate } = await createWithFallback(HandLandmarker, visionFileset, {
      baseOptions: options.modelAssetBuffer ? { modelAssetBuffer: options.modelAssetBuffer } : { modelAssetPath: MEDIAPIPE.models.hand },
      runningMode: "VIDEO",
      numHands: options.numHands ?? 1,
      minHandDetectionConfidence: options.minDetectionConfidence ?? 0.5,
      minHandPresenceConfidence:  options.minPresenceConfidence  ?? 0.5,
      minTrackingConfidence:      options.minTrackingConfidence  ?? 0.5,
    });
    return wrap("hand", 21, landmarker, delegate, (r) => ({
      landmarks: r.landmarks ?? [],
      worldLandmarks: r.worldLandmarks ?? [],
      handedness: (r.handedness ?? []).map((h) => h[0]?.categoryName ?? "?"),
      // The confidence that goes with each label, kept alongside rather than
      // folded into `handedness`, so experiments written against the old shape
      // keep working unchanged. null where the model gave no score.
      handednessScore: (r.handedness ?? []).map(
        (h) => (typeof h[0]?.score === "number" ? h[0].score : null)
      ),
    }));
  }

  if (kind === "pose") {
    // the same hand-over as the hand model's (signs/capture.js downloads the pose model with its progress line)
    const { landmarker, delegate } = await createWithFallback(PoseLandmarker, visionFileset, {
      baseOptions: options.modelAssetBuffer ? { modelAssetBuffer: options.modelAssetBuffer } : { modelAssetPath: MEDIAPIPE.models.pose },
      runningMode: "VIDEO",
      numPoses: options.numPoses ?? 1,
    });
    return wrap("pose", 33, landmarker, delegate, (r) => ({
      landmarks: r.landmarks ?? [],
      worldLandmarks: r.worldLandmarks ?? [],
      handedness: [],            // whole-body tracking has no handedness
      handednessScore: [],
    }));
  }

  throw new Error(`Unknown tracker "${kind}". Use "hand" or "pose".`);
}


/* ---------------------------------------------------------------------------
 * Try the GPU, fall back to the CPU.
 *
 * MediaPipe's GPU path needs a WebGL context, and there are plenty of real
 * machines that cannot give it one: hardware acceleration switched off,
 * a blocklisted driver, a virtual machine, a remote desktop, or simply too many
 * WebGL contexts already open in other tabs. Those participants used to see
 * "emscripten_webgl_create_context() returned error 0" and could not take part
 * at all. The CPU path is slower but works everywhere, which matters far more
 * for a study people join from their own computers.
 *
 * Which one was used is reported as `tracker.delegate`, stored with the session,
 * and shown by check.html. CPU machines run at a lower frame rate, which is
 * worth knowing when a participant's data looks unusual.
 * -------------------------------------------------------------------------*/
export async function createWithFallback(Landmarker, fileset, options) {
  const failures = [];

  for (const delegate of delegateOrder()) {
    try {
      const landmarker = await Landmarker.createFromOptions(fileset, {
        ...options,
        baseOptions: { ...options.baseOptions, delegate },
      });
      if (delegate === "CPU") {
        console.warn(
          "MediaPipe could not use the GPU on this machine, so it is running on " +
          "the CPU instead. Everything works; the frame rate will be lower.\n" +
          `GPU error was: ${failures[0]}`
        );
      }
      return { landmarker, delegate };
    } catch (err) {
      failures.push(err?.message || String(err));
    }
  }

  throw new Error(
    "The hand tracking model could not start on this computer, on either the " +
    "GPU or the CPU.\n\nTry a different browser (Chrome, Edge, or Firefox), " +
    "close other tabs, and check that hardware acceleration is enabled in your " +
    "browser settings.\n\nDetails: " + failures.join(" | ")
  );
}

/** Which delegates to try, in order. See MEDIAPIPE.delegate in config.js. */
export function delegateOrder() {
  const fromUrl = typeof location !== "undefined"
    ? new URLSearchParams(location.search).get("delegate")
    : null;
  const choice = String(fromUrl || MEDIAPIPE.delegate || "auto").toUpperCase();

  if (choice === "CPU") return ["CPU"];
  if (choice === "GPU") return ["GPU"];
  return ["GPU", "CPU"];          // "auto"
}

/** Shared shape for every tracker, so the rest of the code never branches. */
function wrap(kind, numLandmarks, landmarker, delegate, adapt) {
  let consecutiveErrors = 0;
  return {
    kind,
    numLandmarks,
    delegate,
    errorCount: 0,
    track(video, tMs) {
      try {
        const out = adapt(landmarksUnmirror(landmarker.detectForVideo(video, tMs)));
        consecutiveErrors = 0;
        return out;
      } catch (err) {
        // A GPU context can be lost mid-session (the machine sleeps, the driver
        // resets). Treat a hiccup as a frame with nothing detected rather than
        // throwing away the participant's whole session, but do not hide a
        // permanent failure.
        this.errorCount++;
        if (++consecutiveErrors > 90) {   // ~3 seconds of solid failure
          throw new Error(
            "Hand tracking stopped working partway through. This usually means " +
            "the browser lost access to the graphics card. Please reload the " +
            `page and start again. Details: ${err?.message || err}`
          );
        }
        return { landmarks: [], worldLandmarks: [], handedness: [], handednessScore: [] };
      }
    },
    close: () => landmarker.close(),
  };
}

/* ---------------------------------------------------------------------------
 * Named landmark indices, so your code can say HAND.INDEX_TIP instead of 8.
 * Full maps: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
 * -------------------------------------------------------------------------*/
export const HAND = {
  WRIST: 0,
  THUMB_CMC: 1, THUMB_MCP: 2, THUMB_IP: 3, THUMB_TIP: 4,
  INDEX_MCP: 5, INDEX_PIP: 6, INDEX_DIP: 7, INDEX_TIP: 8,
  MIDDLE_MCP: 9, MIDDLE_PIP: 10, MIDDLE_DIP: 11, MIDDLE_TIP: 12,
  RING_MCP: 13, RING_PIP: 14, RING_DIP: 15, RING_TIP: 16,
  PINKY_MCP: 17, PINKY_PIP: 18, PINKY_DIP: 19, PINKY_TIP: 20,
};

export const POSE = {
  NOSE: 0, LEFT_SHOULDER: 11, RIGHT_SHOULDER: 12, LEFT_ELBOW: 13,
  RIGHT_ELBOW: 14, LEFT_WRIST: 15, RIGHT_WRIST: 16, LEFT_HIP: 23,
  RIGHT_HIP: 24, LEFT_KNEE: 25, RIGHT_KNEE: 26, LEFT_ANKLE: 27, RIGHT_ANKLE: 28,
};

/** Straight-line 3-D distance between two landmarks. */
export function distance3d(a, b) {
  const dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/* ?srcmirror=1 is for a camera that already sends a mirrored picture (a physical right hand looks like a left one;
 * the owner's Mac does this, jt-demo-film 2026-10-06). The landmarks are turned back to the true view, so the hand
 * check, the grader and the ghost see the real hand; css/style.css shows the picture as it comes and flips only the
 * drawing on top of it. */
const SRC_MIRROR = typeof location !== "undefined" && new URLSearchParams(location.search).get("srcmirror") === "1";
if (SRC_MIRROR && typeof document !== "undefined") document.documentElement.classList.add("src-mirror");
function landmarksUnmirror(r) {
  if (!SRC_MIRROR || !r) return r;
  const flip = (hands) => hands?.map((h) => h.map((p) => ({ ...p, x: 1 - p.x })));
  const flipW = (hands) => hands?.map((h) => h.map((p) => ({ ...p, x: -p.x })));
  const other = (n) => (n === "Left" ? "Right" : n === "Right" ? "Left" : n);
  const swap = (cats) => cats?.map((c) => c.map((k) => ({ ...k, categoryName: other(k.categoryName), displayName: other(k.displayName) })));
  return { ...r, landmarks: flip(r.landmarks), worldLandmarks: flipW(r.worldLandmarks), handedness: swap(r.handedness), handednesses: swap(r.handednesses) };
}
