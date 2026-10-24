/* vec.js: 3-vector helpers on plain [x, y, z] arrays. */
export const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
export const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const norm = (a) => Math.sqrt(dot(a, a));
export const unit = (a) => { const n = norm(a); return n > 1e-12 ? scale(a, 1 / n) : [0, 0, 0]; };

// atan2 of |a x b| and a . b, not acos of the normalized dot product: acos
// throws away precision near 0 and 180 degrees, which is exactly where a
// straight finger lives, and engines disagree in the last digits of acos.
export const angleDeg = (a, b) => Math.atan2(norm(cross(a, b)), dot(a, b)) * 180 / Math.PI;

// JS % keeps the sign of the dividend, so wrap explicitly. Result in [-180, 180).
export const wrapDeg = (d) => ((((d + 180) % 360) + 360) % 360) - 180;
