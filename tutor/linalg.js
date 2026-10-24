/* linalg.js: the four dense-matrix operations the verifier needs, on plain
 * arrays of rows (or typed rows -- anything indexable with a `length`).
 *
 * There is no matrix library here and there is not going to be one: the
 * tutor ships as plain ES modules with no build step and no dependencies,
 * and the whole job is 46-dimensional. What matters instead is that these
 * agree with numpy to the last few bits, because tutor/verifier.js is pinned
 * to the Python scorer in training/gaussian_model.py by a fixture, and a
 * difference in how a sum is accumulated is the kind of thing that shows up
 * only on the hand that was about to be graded.
 *
 * quadForm is the hot one -- it runs 25 times per frame (once per class) --
 * so it allocates nothing and takes the vector it is given rather than
 * building a difference vector of its own. solve and subMatrix are not on
 * the per-frame path; they are here for the conditional-Gaussian block blame
 * that comes next, which needs a sub-block of the precision matrix and one
 * small linear solve.
 */

function checkSquare(M, who) {
  const n = M.length;
  for (let i = 0; i < n; i++) {
    if (M[i].length !== n) throw new Error(`${who}: expected a square matrix, row ${i} has ${M[i].length} entries but there are ${n} rows`);
  }
  return n;
}

// M: (n, m) rows. v: length m. Returns a new plain array of length n.
export function matVec(M, v) {
  const n = M.length;
  const m = M[0] === undefined ? 0 : M[0].length;
  if (v.length !== m) throw new Error(`matVec: matrix has ${m} column(s) but the vector has length ${v.length}`);
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    const row = M[i];
    if (row.length !== m) throw new Error(`matVec: row ${i} has ${row.length} entries but row 0 has ${m}`);
    let s = 0;
    for (let j = 0; j < m; j++) s += row[j] * v[j];
    out[i] = s;
  }
  return out;
}

/* v' P v, accumulated the same way as matVec's rows so that the two agree
 * exactly (quadForm(P, v) === dot(v, matVec(P, v)) up to the final sum's
 * ordering). Written as one fused pass because the verifier calls it once
 * per class per frame and the intermediate P v is never wanted. */
export function quadForm(P, v) {
  const n = checkSquare(P, "quadForm");
  if (v.length !== n) throw new Error(`quadForm: matrix is ${n}x${n} but the vector has length ${v.length}`);
  let total = 0;
  for (let i = 0; i < n; i++) {
    const row = P[i];
    let s = 0;
    for (let j = 0; j < n; j++) s += row[j] * v[j];
    total += v[i] * s;
  }
  return total;
}

/* Gaussian elimination with partial pivoting. Copies A and b, so a caller
 * can pass a model's own matrix without it being eaten.
 *
 * A singular matrix THROWS rather than returning infinities or NaNs: every
 * caller here is asking a question about a real hand, and an answer built
 * out of Infinity would flow into a score and then into a verdict looking
 * like a number. The pivot test is relative to the largest entry in A, so it
 * does not depend on the units the caller happens to be working in. */
const SINGULAR_RELATIVE_TOLERANCE = 1e-12;

export function solve(A, b) {
  const n = checkSquare(A, "solve");
  if (b.length !== n) throw new Error(`solve: matrix has ${n} row(s) but the right-hand side has length ${b.length}`);
  const M = [];
  let scale = 0;
  for (let i = 0; i < n; i++) {
    const row = new Float64Array(n + 1);
    for (let j = 0; j < n; j++) {
      row[j] = A[i][j];
      const a = Math.abs(row[j]);
      if (a > scale) scale = a;
    }
    row[n] = b[i];
    M.push(row);
  }
  const tiny = SINGULAR_RELATIVE_TOLERANCE * (scale > 0 ? scale : 1);

  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[pivot][col])) pivot = r;
    if (!(Math.abs(M[pivot][col]) > tiny)) {
      throw new Error(`solve: matrix is singular at column ${col} (largest remaining pivot ${Math.abs(M[pivot][col])})`);
    }
    if (pivot !== col) { const t = M[pivot]; M[pivot] = M[col]; M[col] = t; }
    const p = M[col];
    for (let r = col + 1; r < n; r++) {
      const row = M[r];
      const f = row[col] / p[col];
      if (f === 0) continue;
      for (let j = col; j <= n; j++) row[j] -= f * p[j];
    }
  }

  const x = new Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = M[i][n];
    for (let j = i + 1; j < n; j++) s -= M[i][j] * x[j];
    x[i] = s / M[i][i];
  }
  return x;
}

/* Lower-triangular L with L L' = M, by the standard Cholesky recurrence.
 *
 * Its real job here is the failure, not the factor: Cholesky succeeds if and
 * only if M is positive-definite, which is the property that makes v' M v a
 * squared distance. A matrix can be the right shape, finite and perfectly
 * symmetric and still fail it -- all zeros, or a sign flip -- and then
 * v' M v is zero for every v, or negative, and a typicality gate built on it
 * silently never fires. There is no cheaper honest test.
 *
 * A non-finite entry fails the same check as a non-positive one: NaN is not
 * greater than zero. `who` labels the error, so a caller can say WHICH class
 * of a model is broken. Only the lower triangle is read; symmetry is the
 * caller's to check (loadModel does). */
export function cholesky(M, who = "cholesky") {
  const n = checkSquare(M, who);
  const L = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = M[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      if (i !== j) {
        L[i][j] = s / L[j][j];
      } else if (s > 0) {
        L[i][i] = Math.sqrt(s);
      } else {
        throw new Error(`${who}: not positive-definite -- pivot ${i} is ${s}`);
      }
    }
  }
  return L;
}

// The rows x cols block of M, copied into fresh plain arrays -- a block-blame
// caller edits what it gets back, and the model must not move with it.
export function subMatrix(M, rows, cols) {
  const out = [];
  for (const r of rows) {
    if (!(r >= 0 && r < M.length)) throw new Error(`subMatrix: row index ${r} is outside a matrix with ${M.length} row(s)`);
    const src = M[r];
    const row = new Array(cols.length);
    for (let k = 0; k < cols.length; k++) {
      const c = cols[k];
      if (!(c >= 0 && c < src.length)) throw new Error(`subMatrix: column index ${c} is outside a row with ${src.length} entr${src.length === 1 ? "y" : "ies"}`);
      row[k] = src[c];
    }
    out.push(row);
  }
  return out;
}
