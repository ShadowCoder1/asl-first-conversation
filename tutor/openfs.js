/* openfs.js: KETI's OpenFS letter reader (arXiv 2602.22949, github.com/AIRC-KETI/OpenFS), run in the browser.
 *
 * WHY IT IS HERE. On the experts' hands (research/combine-2026-10-16/RESULTS.md) OpenFS reads the M's our
 * grader misses, so tutor/mix.js asks it for a second opinion on a few letters when our grader says no.
 *
 * WHAT IT IS. Not a hand-shape classifier: the FSboard recognizer's transformer ENCODER over the whole clip,
 * then the frame classifier's head on each frame. Every frame's output sees every other frame (full
 * self-attention), so the input is the trial's clip, not one pose. Written out here in plain JS, so it runs
 * synchronously inside the grader like everything else, with no runtime to download:
 *
 *   per frame, x and y of the 21 landmarks, raw (no mirroring; they feed left hands as left hands)
 *     -> their normalize(): per frame and per axis, centre the bounding box and scale it to [-0.5, 0.5]
 *     -> Linear 42->256, LayerNorm, ReLU, Linear 256->256, LayerNorm, ReLU
 *     -> + sinusoidal encoding of the frame index (0..T-1) + the same encoding of 0 (their rlh_seg is zeros)
 *     -> 3 x post-norm TransformerEncoderLayer (8 heads, feed-forward 2048, exact GELU), no padding
 *     -> head: Linear 256->256, LayerNorm, ReLU, Linear 256->61 (index 0 = no letter, then their characters)
 *
 * Only the last few frames' outputs are ever read (the held pose), so the last layer is computed for those rows
 * alone: their keys and values still come from every frame, so the answer is the same.
 *
 * The weights (tutor/openfs/openfs-weights.bin + .json) are made by research/combine-2026-10-16/export_openfs.py
 * and are NOT in git: OpenFS is "for non-commercial research purposes only". tests/openfs.test.js pins this file
 * to their PyTorch model on real expert clips. */

const D = 256, H = 8, HD = D / H, FF = 2048, IN = 42, CLASSES = 61, EPS = 1e-5;
const CHARS = "{ !#$%&'()*+,-./0123456789:;=?@[_abcdefghijklmnopqrstuvwxyz~";

/** Column of a letter in the 61 outputs. */
export const letterColumn = (letter) => CHARS.indexOf(letter.toLowerCase());

/* erf by Abramowitz & Stegun 7.1.26: absolute error under 1.5e-7, which is below float32's own rounding of a
 * GELU input; one exp per call, so the 2048-wide feed-forward stays cheap. */
function erf(x) {
  const s = x < 0 ? -1 : 1, a = Math.abs(x), t = 1 / (1 + 0.3275911 * a);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a);
  return s * y;
}
const gelu = (x) => 0.5 * x * (1 + erf(x / Math.SQRT2));

/* out[r*outDim + o] = b[o] + sum_i X[r*inDim + i] * W[o*inDim + i] (W is PyTorch's (out, in)). Four rows share
 * each pass over a weight row: the feed-forward weights are 4 MB, so reading them a quarter as often is most of
 * the speed. */
function linear(X, rows, inDim, W, b, outDim, out) {
  let r = 0;
  for (; r + 4 <= rows; r += 4) {
    const x0 = r * inDim, x1 = x0 + inDim, x2 = x1 + inDim, x3 = x2 + inDim;
    const o0 = r * outDim, o1 = o0 + outDim, o2 = o1 + outDim, o3 = o2 + outDim;
    for (let o = 0; o < outDim; o++) {
      const wo = o * inDim;
      let s0 = b[o], s1 = s0, s2 = s0, s3 = s0;
      for (let i = 0; i < inDim; i++) {
        const wv = W[wo + i];
        s0 += X[x0 + i] * wv; s1 += X[x1 + i] * wv; s2 += X[x2 + i] * wv; s3 += X[x3 + i] * wv;
      }
      out[o0 + o] = s0; out[o1 + o] = s1; out[o2 + o] = s2; out[o3 + o] = s3;
    }
  }
  for (; r < rows; r++) {
    const xo = r * inDim, oo = r * outDim;
    for (let o = 0; o < outDim; o++) {
      const wo = o * inDim;
      let s = b[o];
      for (let i = 0; i < inDim; i++) s += X[xo + i] * W[wo + i];
      out[oo + o] = s;
    }
  }
}

function layerNormRows(X, rows, w, b) {
  for (let r = 0; r < rows; r++) {
    const o = r * D;
    let m = 0;
    for (let i = 0; i < D; i++) m += X[o + i];
    m /= D;
    let v = 0;
    for (let i = 0; i < D; i++) { const d = X[o + i] - m; v += d * d; }
    const inv = 1 / Math.sqrt(v / D + EPS);
    for (let i = 0; i < D; i++) X[o + i] = (X[o + i] - m) * inv * w[i] + b[i];
  }
}

const relu = (X) => { for (let i = 0; i < X.length; i++) if (X[i] < 0) X[i] = 0; };

/** Their lib/utils/dataset.py normalize(), in float32 as their loader does: per frame, per axis. */
export function normalizeClip(xy, T) {
  const out = new Float64Array(T * IN);
  for (let t = 0; t < T; t++) {
    for (let ax = 0; ax < 2; ax++) {
      let lo = Infinity, hi = -Infinity;
      for (let j = 0; j < 21; j++) { const v = Math.fround(xy[t * IN + 2 * j + ax]); if (v < lo) lo = v; if (v > hi) hi = v; }
      const c = Math.fround((lo + hi) / 2);
      let m = 0;
      for (let j = 0; j < 21; j++) m = Math.max(m, Math.abs(Math.fround(Math.fround(xy[t * IN + 2 * j + ax]) - c)));
      const den = Math.fround(m + 1e-6);
      for (let j = 0; j < 21; j++) {
        const v = Math.fround(Math.fround(xy[t * IN + 2 * j + ax]) - c);
        out[t * IN + 2 * j + ax] = Math.fround(Math.fround(v / den) * 0.5);
      }
    }
  }
  return out;
}

/* Their positional_encoding_from_idx, float32 like theirs: pe[2i] = sin(t * div_i), pe[2i+1] = cos(t * div_i). */
const DIV = Float64Array.from({ length: D / 2 }, (_, i) => Math.fround(Math.exp(Math.fround(Math.fround(2 * i) * Math.fround(-Math.log(10000) / D)))));
function addPositions(X, T) {
  for (let t = 0; t < T; t++) {
    for (let i = 0; i < D / 2; i++) {
      const ang = Math.fround(t * DIV[i]);
      X[t * D + 2 * i] += Math.fround(Math.sin(ang));
      X[t * D + 2 * i + 1] += Math.fround(Math.cos(ang)) + 1;   // + the encoding of rlh_seg = 0: sin 0 = 0, cos 0 = 1
    }
  }
}

/**
 * @param {object} manifest  openfs-weights.json
 * @param {ArrayBuffer} buffer openfs-weights.bin (little-endian float32)
 */
export function createOpenFS(manifest, buffer) {
  const all = Float64Array.from(new Float32Array(buffer));   // converted once, so the inner loops never do
  if (all.length !== manifest.floats) throw new Error(`openfs: weights file has ${all.length} floats, manifest says ${manifest.floats}`);
  const w = (name) => {
    const t = manifest.tensors[name];
    if (!t) throw new Error(`openfs: no tensor ${name}`);
    return all.subarray(t.offset, t.offset + t.shape.reduce((a, b) => a * b, 1));
  };
  const L = (i, n) => w(`rec.encoder.layers.${i}.${n}`);
  const layers = [0, 1, 2].map((i) => ({
    inW: L(i, "self_attn.in_proj_weight"), inB: L(i, "self_attn.in_proj_bias"),
    outW: L(i, "self_attn.out_proj.weight"), outB: L(i, "self_attn.out_proj.bias"),
    w1: L(i, "linear1.weight"), b1: L(i, "linear1.bias"), w2: L(i, "linear2.weight"), b2: L(i, "linear2.bias"),
    n1w: L(i, "norm1.weight"), n1b: L(i, "norm1.bias"), n2w: L(i, "norm2.weight"), n2b: L(i, "norm2.bias"),
  }));

  /* One encoder layer. X is (T, D); rows = which rows of the output are needed (all of them except in the last
   * layer). Returns the new (rows.length, D) block. */
  function encoderLayer(lw, X, T, rows) {
    const qkv = new Float64Array(T * 3 * D);
    linear(X, T, D, lw.inW, lw.inB, 3 * D, qkv);
    const R = rows.length, att = new Float64Array(R * D), p = new Float64Array(T), scale = 1 / Math.sqrt(HD);
    for (let k = 0; k < R; k++) {
      const r = rows[k];
      for (let h = 0; h < H; h++) {
        const qo = r * 3 * D + h * HD;
        let mx = -Infinity;
        for (let s = 0; s < T; s++) {
          const ko = s * 3 * D + D + h * HD;
          let dot = 0;
          for (let i = 0; i < HD; i++) dot += qkv[qo + i] * qkv[ko + i];
          p[s] = dot * scale;
          if (p[s] > mx) mx = p[s];
        }
        let z = 0;
        for (let s = 0; s < T; s++) { p[s] = Math.exp(p[s] - mx); z += p[s]; }
        const ao = k * D + h * HD;
        for (let s = 0; s < T; s++) {
          const ps = p[s] / z, vo = s * 3 * D + 2 * D + h * HD;
          for (let i = 0; i < HD; i++) att[ao + i] += ps * qkv[vo + i];
        }
      }
    }
    const y = new Float64Array(R * D);
    linear(att, R, D, lw.outW, lw.outB, D, y);
    for (let k = 0; k < R; k++) for (let i = 0; i < D; i++) y[k * D + i] += X[rows[k] * D + i];
    layerNormRows(y, R, lw.n1w, lw.n1b);
    const hid = new Float64Array(R * FF), f = new Float64Array(R * D);
    linear(y, R, D, lw.w1, lw.b1, FF, hid);
    for (let i = 0; i < hid.length; i++) hid[i] = gelu(hid[i]);
    linear(hid, R, FF, lw.w2, lw.b2, D, f);
    for (let i = 0; i < y.length; i++) y[i] += f[i];
    layerNormRows(y, R, lw.n2w, lw.n2b);
    return y;
  }

  return {
    /**
     * Per-frame logits for the frames in `tail`, the whole clip read once.
     * @param {ArrayLike<number>} xy  T*42: frame-major, then landmark, then x, y (raw image fractions)
     * @param {number} T
     * @param {number[]} tail         frame indices whose outputs are wanted, ascending
     * @returns {Float64Array[]}      one 61-vector of logits per tail frame
     */
    tailLogits(xy, T, tail) {
      if (!(T > 0) || xy.length < T * IN) throw new Error(`openfs: expected ${T} frames of ${IN} numbers`);
      const x0 = normalizeClip(xy, T);
      let X = new Float64Array(T * D);
      linear(x0, T, IN, w("rec.pose_embedding.0.weight"), w("rec.pose_embedding.0.bias"), D, X);
      layerNormRows(X, T, w("rec.pose_embedding.1.weight"), w("rec.pose_embedding.1.bias")); relu(X);
      const X2 = new Float64Array(T * D);
      linear(X, T, D, w("rec.pose_embedding.3.weight"), w("rec.pose_embedding.3.bias"), D, X2);
      layerNormRows(X2, T, w("rec.pose_embedding.4.weight"), w("rec.pose_embedding.4.bias")); relu(X2);
      addPositions(X2, T);
      X = X2;
      const everyRow = Array.from({ length: T }, (_, i) => i);
      X = encoderLayer(layers[0], X, T, everyRow);
      X = encoderLayer(layers[1], X, T, everyRow);
      const E = encoderLayer(layers[2], X, T, tail);
      const h = new Float64Array(tail.length * D);
      linear(E, tail.length, D, w("cls.linear.0.weight"), w("cls.linear.0.bias"), D, h);
      layerNormRows(h, tail.length, w("cls.linear.1.weight"), w("cls.linear.1.bias")); relu(h);
      const out = new Float64Array(tail.length * CLASSES);
      linear(h, tail.length, D, w("cls.linear.4.weight"), w("cls.linear.4.bias"), CLASSES, out);
      return tail.map((_, k) => out.subarray(k * CLASSES, (k + 1) * CLASSES));
    },

    /** The mean over the tail frames of each frame's log-softmax over all 61 classes (what the analysis used). */
    meanLogSoftmax(xy, T, tail) {
      const rows = this.tailLogits(xy, T, tail);
      const m = new Float64Array(CLASSES);
      for (const z of rows) {
        let mx = -Infinity;
        for (const v of z) if (v > mx) mx = v;
        let s = 0;
        for (const v of z) s += Math.exp(v - mx);
        const lse = mx + Math.log(s);
        for (let c = 0; c < CLASSES; c++) m[c] += (z[c] - lse) / rows.length;
      }
      return m;
    },
  };
}

/** Fetch and build. Rejects (never hangs) if the files are missing, so the caller can grade without it. */
export async function loadOpenFS(baseUrl = new URL("./openfs/", import.meta.url).href) {
  const [mr, br] = await Promise.all([fetch(`${baseUrl}openfs-weights.json`), fetch(`${baseUrl}openfs-weights.bin`)]);
  if (!mr.ok || !br.ok) throw new Error(`openfs: weights not found under ${baseUrl}`);
  return createOpenFS(await mr.json(), await br.arrayBuffer());
}
