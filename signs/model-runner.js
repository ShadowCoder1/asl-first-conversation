/* signs/model-runner.js: the sign model (ONNX), run by onnxruntime-web off the frame loop.
 *
 * The camera loop must never wait for the model: on a phone HandLandmarker and PoseLandmarker already take about
 * 27 ms of a 33-40 ms frame (docs/signs-design.md, performance budget). So the model runs in a module Worker
 * (signs/recognizer.worker.js) and this file is the page's side of it:
 *
 *   - The model and onnxruntime's WebAssembly (11.2 MB, 2.4 MB over the wire) are downloaded here with a progress
 *     callback, the way the phone app downloads MediaPipe's files (mobile/app.js download()): a download that goes
 *     quiet for 15 s is given up and tried again, three tries. Both are handed over as bytes, so neither is fetched
 *     twice whatever the browser's cache does with large responses.
 *   - onnxruntime-web 1.22.0 from jsdelivr, its WASM backend, ONE thread. More threads need SharedArrayBuffer, so a
 *     cross-origin-isolated page, and GitHub Pages cannot send the headers that make one.
 *   - At most one run in flight and one queued; a newer run replaces the queued one, which resolves null. The
 *     recognizer asks for a window every ~100 ms: on a phone slower than that, old windows are dropped, never piled up.
 *   - If the module Worker cannot start (Firefox before 114 has none) or cannot load the model, the same session runs
 *     on the main thread instead: the frames get slower, the lesson still works.
 *
 * Nothing at the top level touches the page, so the worker imports createOrtSession from here: one place for the
 * onnxruntime version and session options, whichever thread runs the model. */

export const ORT_VERSION = "1.22.0";
export const ORT_CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
/* The WASM-only build (48 KB of JavaScript): no WebGL or WebGPU code to download for a model this small. */
export const ORT_MODULE = `${ORT_CDN}ort.wasm.min.mjs`;
export const ORT_WASM = `${ORT_CDN}ort-wasm-simd-threaded.wasm`;
/* What the stream yields. jsdelivr sends it brotli-compressed, so its Content-Length (2.4 MB) is not the size read. */
export const ORT_WASM_BYTES = 11210254;
/* signfeat's features per frame (tools/signs/signfeat.py N_FEAT). */
export const N_FEAT = 326;
/* Run once at load: the first run in a thread is several times slower than the rest (31 ms against 4.4 ms at T = 24,
 * Chrome on a laptop, the shipping model's size), and it would land on the learner's first sign. 64, the longest
 * window, so that onnxruntime's memory has already grown to what any later run needs. A length new to the session
 * costs no more than one seen before (measured for T = 9 to 48). */
export const WARM_T = Object.freeze([64]);

const STALL_MS = 15000;
const TRIES = 3;
/* The worker's first message. It needs only this file (same origin, already loaded by the page), so seconds of
 * silence mean the browser could not start it. */
const HELLO_MS = 10000;
/* A run that never comes back (a wedged worker) is given up, so the queue behind it keeps moving. */
const RUN_TIMEOUT_MS = 10000;
const TIMES_KEPT = 300;

/* The test onnxruntime itself makes (wasm-factory.ts isSimdSupported): 1.22 has no build without SIMD, so a browser
 * without it (iOS before 16.4) is told so before 11 MB are downloaded for nothing. */
const SIMD_PROBE = [0, 97, 115, 109, 1, 0, 0, 0, 1, 4, 1, 96, 0, 0, 3, 2, 1, 0, 10, 30, 1, 28, 0, 65, 0, 253, 15, 253, 12,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 253, 186, 1, 26, 11];
export function simdSupported() {
  try { return typeof WebAssembly === "object" && WebAssembly.validate(new Uint8Array(SIMD_PROBE)); } catch { return false; }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const msg = (err) => String(err?.message || err);
function kindError(kind, text, cause) {
  const e = new Error(text);
  e.kind = kind;
  if (cause) e.cause = cause;
  return e;
}

/* ---- the session: the same code in the worker and on the main thread ------------------------------------------ */

/**
 * @param {ArrayBuffer|Uint8Array} model  the ONNX file
 * @param {{wasmBinary?: ArrayBuffer|null, warm?: number[]}} [o]
 * @returns {Promise<{run(x: Float32Array, T: number): Promise<{probs: Float32Array, embedding: Float32Array, ms: number}>,
 *                    release(): void, info: object}>}
 */
export async function createOrtSession(model, { wasmBinary = null, warm = WARM_T } = {}) {
  const ort = await import(ORT_MODULE);
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;      // this already is the thread that should run it
  ort.env.wasm.wasmPaths = ORT_CDN;
  if (wasmBinary) ort.env.wasm.wasmBinary = wasmBinary;
  const t0 = performance.now();
  const session = await ort.InferenceSession.create(model instanceof Uint8Array ? model : new Uint8Array(model), {
    executionProviders: ["wasm"], graphOptimizationLevel: "all",
  });
  const createMs = performance.now() - t0;
  // the compiled module keeps what it needs; the 11 MB of bytes can go
  ort.env.wasm.wasmBinary = undefined;
  const inName = session.inputNames.includes("x") ? "x" : session.inputNames[0];
  const probsName = session.outputNames.includes("probs") ? "probs" : session.outputNames[0];
  const embName = session.outputNames.includes("embedding") ? "embedding" : session.outputNames.find((n) => n !== probsName);

  async function run(x, T) {
    if (!(x instanceof Float32Array) || !Number.isInteger(T) || T < 1 || x.length !== T * N_FEAT) {
      throw new Error(`model input: ${x?.length} numbers for T=${T}, expected T x ${N_FEAT}`);
    }
    const t = performance.now();
    const input = new ort.Tensor("float32", x, [1, T, N_FEAT]);
    const out = await session.run({ [inName]: input });
    const ms = performance.now() - t;
    // copies: a result may be a view on onnxruntime's (shared) memory, which cannot be posted or kept
    const probs = new Float32Array(out[probsName].data);
    const embedding = embName && out[embName] ? new Float32Array(out[embName].data) : new Float32Array(0);
    for (const k of Object.keys(out)) out[k].dispose?.();
    return { probs, embedding, ms };
  }

  const tw = performance.now();
  let nOut = 0, nEmbed = 0;
  for (const T of warm) {
    const r = await run(new Float32Array(T * N_FEAT), T);
    nOut = r.probs.length; nEmbed = r.embedding.length;
  }
  const info = {
    ortVersion: ORT_VERSION, createMs, warmMs: performance.now() - tw, warmT: [...warm],
    inputNames: [...session.inputNames], outputNames: [...session.outputNames], nOut, nEmbed,
  };
  return { run, info, release: () => { try { session.release?.(); } catch { /* already gone */ } } };
}

/* ---- downloads ------------------------------------------------------------------------------------------------- */

/**
 * The whole response as one ArrayBuffer, read as a stream so progress can be told, sized up front when the size is
 * known (one copy in memory). A response that stops sending for stallMs is given up and tried again.
 * @param {string|URL} url
 * @param {{bytes?: number, onBytes?: (got: number, total: number) => void, stallMs?: number, tries?: number}} [o]
 */
export async function fetchBytes(url, { bytes = 0, onBytes = null, stallMs = STALL_MS, tries = TRIES } = {}) {
  let last = null;
  for (let attempt = 0; attempt < tries; attempt++) {
    if (attempt) { onBytes?.(0, bytes); await sleep(1000 * attempt); }
    const ctrl = new AbortController();
    let timer = null;
    const arm = () => { clearTimeout(timer); timer = setTimeout(() => ctrl.abort(), stallMs); };
    try {
      arm();
      const res = await fetch(url, { signal: ctrl.signal });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      // Content-Length is the size on the wire; for a compressed response it is not what the stream yields
      const enc = res.headers.get("content-encoding");
      const len = Number(res.headers.get("content-length")) || 0;
      const total = bytes || (!enc || enc === "identity" ? len : 0);
      if (!res.body) {
        const whole = await res.arrayBuffer();
        clearTimeout(timer);
        onBytes?.(whole.byteLength, total || whole.byteLength);
        return whole;
      }
      const reader = res.body.getReader();
      let out = new Uint8Array(total || 1 << 20), got = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        arm();
        if (got + value.byteLength > out.length) {
          const more = new Uint8Array(Math.ceil((got + value.byteLength) * 1.25));
          more.set(out.subarray(0, got));
          out = more;
        }
        out.set(value, got);
        got += value.byteLength;
        onBytes?.(got, total);
      }
      clearTimeout(timer);
      return got === out.length ? out.buffer : out.slice(0, got).buffer;
    } catch (err) {
      clearTimeout(timer);
      last = err;
    }
  }
  const name = String(url).split("/").pop();
  throw kindError("load", `Could not load ${name} (${last?.name === "AbortError" ? "the download stalled" : msg(last)}).`, last);
}

/* ---- the queue: one run in flight, one waiting, the newest wins --------------------------------------------------- */

/**
 * @param {(x: Float32Array, T: number) => Promise<{probs: Float32Array, embedding: Float32Array, ms: number}>} exec
 * @param {{timeoutMs?: number, now?: () => number}} [o]
 */
export function createRunQueue(exec, { timeoutMs = RUN_TIMEOUT_MS, now = () => performance.now() } = {}) {
  let inFlight = null, queued = null, started = false, failed = null, closed = false;
  const t = { runs: 0, dropped: 0, errors: 0, timeouts: 0, ms: [], wallMs: [], waitMs: [], maxQueued: 0 };
  const keep = (a, v) => { a.push(v); if (a.length > TIMES_KEPT) a.shift(); };

  function run(x, T) {
    return new Promise((resolve, reject) => {
      if (closed) { resolve(null); return; }
      if (failed) { reject(failed); return; }
      if (queued) { t.dropped++; queued.resolve(null); }
      queued = { x, T, resolve, reject, tCall: now() };
      t.maxQueued = Math.max(t.maxQueued, (inFlight ? 1 : 0) + 1);
      pump();
    });
  }
  function pump() {
    if (!started || closed || inFlight || !queued) return;
    const job = queued;
    queued = null;
    inFlight = job;
    const tStart = now();
    // the slot is freed BEFORE the caller hears back: a run asked for right after `await run()` starts at once
    // instead of waiting behind a run that has already finished
    let open = true;
    const settle = (fn) => {
      if (!open) return;
      open = false;
      clearTimeout(timer);
      if (inFlight === job) inFlight = null;
      fn();
      pump();
    };
    job.cancel = () => settle(() => job.resolve(null));
    const timer = setTimeout(() => settle(() => {
      t.timeouts++; t.errors++;
      job.reject(new Error(`the sign model did not answer in ${timeoutMs} ms`));
    }), timeoutMs);
    Promise.resolve().then(() => exec(job.x, job.T)).then((r) => settle(() => {
      const tEnd = now();
      t.runs++;
      keep(t.ms, r.ms); keep(t.wallMs, tEnd - job.tCall); keep(t.waitMs, tStart - job.tCall);
      job.resolve({ probs: r.probs, embedding: r.embedding, ms: r.ms, wallMs: tEnd - job.tCall });
    }), (err) => settle(() => {
      t.errors++;
      job.reject(err);
    }));
  }
  const q = (a, p) => {
    if (!a.length) return null;
    const b = [...a].sort((x, y) => x - y);
    return +b[Math.floor(p * (b.length - 1))].toFixed(2);
  };
  return {
    run,
    start() { started = true; pump(); },
    fail(err) { failed = err; if (queued) { queued.reject(err); queued = null; } },
    close() {
      closed = true;
      if (queued) { queued.resolve(null); queued = null; }
      inFlight?.cancel();
    },
    get busy() { return !!inFlight; },
    get pending() { return (inFlight ? 1 : 0) + (queued ? 1 : 0); },
    timing() {
      const mean = t.ms.length ? +(t.ms.reduce((s, v) => s + v, 0) / t.ms.length).toFixed(2) : null;
      return { runs: t.runs, dropped: t.dropped, errors: t.errors, timeouts: t.timeouts, maxQueued: t.maxQueued,
               p50: q(t.ms, 0.5), p95: q(t.ms, 0.95), mean, max: t.ms.length ? +Math.max(...t.ms).toFixed(2) : null,
               wallP50: q(t.wallMs, 0.5), wallP95: q(t.wallMs, 0.95), waitP95: q(t.waitMs, 0.95) };
    },
  };
}

/* ---- the worker ------------------------------------------------------------------------------------------------ */

function startWorker(url) {
  return new Promise((resolve, reject) => {
    let w;
    try { w = new Worker(url, { type: "module" }); } catch (err) { reject(err); return; }
    const timer = setTimeout(() => { w.terminate(); reject(new Error("the model worker did not start")); }, HELLO_MS);
    w.onmessage = (e) => {
      if (e.data?.type !== "hello") return;
      clearTimeout(timer);
      w.onmessage = null; w.onerror = null;
      resolve(w);
    };
    // a browser that ignores {type: "module"} runs the file as a classic script, and its import is a syntax error
    w.onerror = (e) => {
      e.preventDefault?.();
      clearTimeout(timer);
      w.terminate();
      reject(new Error(e.message || "the model worker failed to start"));
    };
  });
}

/* The worker's half of the protocol (signs/recognizer.worker.js): load once, then run by id. */
function workerBackend(w, model, wasm, warm) {
  let nextId = 1, dead = null;
  const pending = new Map();
  const ready = new Promise((resolve, reject) => {
    w.onmessage = (e) => {
      const m = e.data || {};
      if (m.type === "ready") { resolve(m.info); return; }
      if (m.type === "error") { reject(new Error(m.message)); return; }
      const p = pending.get(m.id);
      if (!p) return;              // an answer to a run already given up on
      pending.delete(m.id);
      if (m.error) p.reject(new Error(m.error));
      else p.resolve({ probs: m.probs, embedding: m.embedding, ms: m.ms });
    };
    w.onerror = (e) => {
      e.preventDefault?.();
      dead = new Error(`the model worker stopped (${e.message || "no reason given"})`);
      reject(dead);
      for (const p of pending.values()) p.reject(dead);
      pending.clear();
    };
  });
  w.postMessage({ type: "load", modelBuffer: model, wasmBinary: wasm, warm: [...warm] }, [model, wasm].filter(Boolean));
  return {
    ready,
    exec(x, T) {
      if (dead) return Promise.reject(dead);
      const id = nextId++;
      return new Promise((resolve, reject) => {
        pending.set(id, { resolve, reject });
        // x is copied, not transferred: the caller may still want its own array
        w.postMessage({ type: "run", id, x, T });
      });
    },
    close() { try { w.postMessage({ type: "close" }); } catch { /* gone */ } setTimeout(() => w.terminate(), 50); },
  };
}

/* ---- the runner --------------------------------------------------------------------------------------------------- */

/**
 * @param {{modelUrl?: string|URL, modelBuffer?: ArrayBuffer|Uint8Array, modelBytes?: number,
 *          onProgress?: (p: {stage: string, frac: number, loadedBytes: number, totalBytes: number}) => void,
 *          useWorker?: boolean, workerUrl?: string|URL, warm?: number[]}} o
 *   modelBytes: the model's size when the server will not say (only for the progress fraction).
 *   onProgress stages: "download" (frac 0..0.9), "starting" (0.95), "ready" (1).
 * @returns {{ready: Promise<object>, run(x: Float32Array, T: number): Promise<{probs: Float32Array,
 *           embedding: Float32Array, ms: number, wallMs: number} | null>, close(): void, timing(): object,
 *           stats: object, readonly backend: "worker"|"main"|null, readonly pending: number}}
 *   ready rejects with err.kind "no-simd" (the browser is too old), "load" (a download failed three times) or
 *   "closed". run resolves null when a newer run replaced it before it started, or after close(); it rejects when the
 *   model failed (the worker stopped, or 10 s without an answer). ms: the model's own time; wallMs: from run() to
 *   the answer, waiting included.
 */
export function createModelRunner({ modelUrl = null, modelBuffer = null, modelBytes = 0, onProgress = null,
                                    useWorker = true, workerUrl = new URL("./recognizer.worker.js", import.meta.url),
                                    warm = WARM_T } = {}) {
  if (!modelUrl && !modelBuffer) throw new Error("createModelRunner needs modelUrl or modelBuffer");
  const stats = { backend: null, info: null, workerError: null, downloadMs: null, loadMs: null, error: null };
  let backend = null, closed = false;
  const queue = createRunQueue((x, T) => backend.exec(x, T));

  const got = { model: [0, modelBytes || 7.5e6], runtime: [0, ORT_WASM_BYTES] };
  if (modelBuffer) got.model = [1, 1];
  let stageNow = "download";
  const progress = (stage, frac) => {
    // only forward: fetching the files again for the main thread (from the cache) does not take the bar back
    if (stage === "download" && stageNow !== "download") return;
    stageNow = stage;
    let a = 0, b = 0;
    for (const [x, y] of Object.values(got)) { a += x; b += y; }
    try {
      onProgress?.({ stage, frac: frac ?? (b ? 0.9 * Math.min(1, a / b) : 0), loadedBytes: a, totalBytes: b });
    } catch (err) { console.warn(`model-runner: onProgress threw (${msg(err)})`); }
  };
  const ownCopy = (buf) => (buf instanceof Uint8Array ? buf.slice() : new Uint8Array(buf).slice()).buffer;
  const getModel = () => (modelBuffer ? Promise.resolve(ownCopy(modelBuffer)) : fetchBytes(modelUrl, {
    bytes: modelBytes, onBytes: (n, total) => { got.model = [n, total || got.model[1]]; progress("download"); },
  }));
  const getWasm = () => fetchBytes(ORT_WASM, {
    bytes: ORT_WASM_BYTES, onBytes: (n) => { got.runtime = [n, ORT_WASM_BYTES]; progress("download"); },
  });

  const ready = (async () => {
    const t0 = performance.now();
    if (!simdSupported()) {
      throw kindError("no-simd", "This browser cannot run the sign model: it needs WebAssembly SIMD (Safari 16.4 or later, Chrome 91 or later, Firefox 89 or later).");
    }
    // the worker starts while the files download; a browser that cannot start one is found out meanwhile
    const workerP = useWorker && typeof Worker !== "undefined"
      ? startWorker(workerUrl).catch((err) => { stats.workerError = msg(err); return null; })
      : Promise.resolve(null);
    progress("download");
    let [model, wasm] = await Promise.all([getModel(), getWasm()]);
    stats.downloadMs = Math.round(performance.now() - t0);
    if (closed) {
      (await workerP)?.terminate();
      throw kindError("closed", "the model runner was closed");
    }
    progress("starting", 0.95);
    const w = await workerP;
    if (w) {
      const wb = workerBackend(w, model, wasm, warm);
      try {
        stats.info = await wb.ready;
        backend = wb;
        stats.backend = "worker";
      } catch (err) {
        stats.workerError = msg(err);
        wb.close();
        // the bytes went to the worker with it; the browser's cache has them again
        model = null; wasm = null;
      }
    }
    if (!backend) {
      if (!model || !model.byteLength) model = await getModel();
      if (!wasm || !wasm.byteLength) wasm = await getWasm();
      const local = await createOrtSession(model, { wasmBinary: wasm, warm });
      model = null; wasm = null;
      stats.info = local.info;
      backend = { exec: (x, T) => local.run(x, T), close: () => local.release() };
      stats.backend = "main";
    }
    if (closed) { backend.close(); throw kindError("closed", "the model runner was closed"); }
    stats.loadMs = Math.round(performance.now() - t0);
    progress("ready", 1);
    return { backend: stats.backend, ...stats.info };
  })();
  ready.then(() => queue.start(), (err) => { stats.error = msg(err); queue.fail(err); });
  ready.catch(() => { /* the caller's own await reports it */ });

  return {
    ready,
    run: (x, T) => queue.run(x, T),
    timing: () => ({ backend: stats.backend, ...queue.timing() }),
    stats,
    get backend() { return stats.backend; },
    get pending() { return queue.pending; },
    close() {
      if (closed) return;
      closed = true;
      queue.close();
      backend?.close();
    },
  };
}
