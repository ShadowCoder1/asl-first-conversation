/* landmarks.js: the single place landmarks are rounded, reshaped and smoothed.
 *
 * The grader reads the same rounded numbers the recorder saves. If the two
 * ever round differently, a sign graded "accept" online can come out "reject"
 * when re-graded from the saved data, and nobody will know which to believe. */
export const DECIMALS = 4;

function round(v, decimals) {
  if (typeof v !== "number" || !Number.isFinite(v)) return null;
  const p = 10 ** decimals;
  return Math.round(v * p) / p;
}

// Returns (number|null)[]: one rounded number per coordinate, or null where
// the input coordinate was missing or non-finite.
export function flattenRounded(landmarks, decimals = DECIMALS) {
  const out = new Array(landmarks.length * 3);
  for (let i = 0; i < landmarks.length; i++) {
    out[i * 3] = round(landmarks[i].x, decimals);
    out[i * 3 + 1] = round(landmarks[i].y, decimals);
    out[i * 3 + 2] = round(landmarks[i].z, decimals);
  }
  return out;
}

// A coordinate that is missing stays missing. flattenRounded writes null where
// the tracker gave nothing, and plain arithmetic on null silently reads it as
// 0: the landmark jumps to the origin and the feature code then returns two
// dozen plausible, finite numbers for a hand it could not see. Worse, the
// offline mirror disagrees -- numpy reads the same null as NaN -- so the
// online grade and the re-derived one part ways on exactly the frames where
// tracking failed. NaN is the "not gradable" signal both sides already
// understand, so that is what a missing coordinate becomes here.
const finite = (v) => (Number.isFinite(v) ? v : NaN);

// MediaPipe gives x as a fraction of image WIDTH and y as a fraction of HEIGHT
// (z is on the x scale). Angles are only meaningful once both axes share a
// unit, so multiply x and z by width/height. A guessed aspect ratio cost 1-5
// points of accuracy in the spike; always pass the real video dimensions.
export function toPoints(flat, aspect) {
  const pts = new Array(flat.length / 3);
  for (let i = 0; i < pts.length; i++) {
    pts[i] = [finite(flat[3 * i]) * aspect, finite(flat[3 * i + 1]), finite(flat[3 * i + 2]) * aspect];
  }
  return pts;
}

// Median, not mean: one wild frame should not move the answer. An odd count
// means the median is always a value that was actually saved.
export function medianFlat(rows) {
  // A row with even one missing/non-finite coordinate is not gradable, and
  // dropouts are never interpolated -- so it is skipped exactly like a fully
  // missing (null) row, rather than letting that null coordinate sort as 0
  // and silently bias the result.
  const usable = rows.filter((r) => r !== null && r !== undefined && r.every((v) => Number.isFinite(v)));
  if (!usable.length) return null;
  const len = usable[0].length;
  for (const r of usable) {
    if (r.length !== len) throw new Error(`medianFlat: rows have differing lengths (${len} vs ${r.length})`);
  }
  let sample = usable;
  if (sample.length % 2 === 0) sample = sample.slice(1);
  const mid = (sample.length - 1) / 2, out = new Array(len);
  for (let k = 0; k < out.length; k++) out[k] = sample.map((r) => r[k]).sort((a, b) => a - b)[mid];
  return out;
}
