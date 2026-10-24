/* signs/recognizer.worker.js: the sign model's own thread.
 *
 * A module Worker (signs/model-runner.js starts it). It holds one onnxruntime-web session and answers runs in the
 * order they come; the page never sends a second run before the first is answered, and the chain below keeps that
 * true even if it did. The session code is model-runner.js's createOrtSession, the same that runs on the main thread
 * when a browser cannot start this worker, so both give the same numbers.
 *
 * Messages in:   {type: "load", modelBuffer, wasmBinary, warm}   once
 *                {type: "run", id, x: Float32Array, T}            x is [1, T, 326] flattened
 *                {type: "close"}
 * Messages out:  {type: "hello"}                                  as soon as this file has loaded
 *                {type: "ready", info} | {type: "error", message}  after load
 *                {type: "result", id, probs: Float32Array(251), embedding: Float32Array(384), ms} | {type: "result", id, error}
 * ms is the model's own time in this thread, without the trip between threads. */

import { createOrtSession } from "./model-runner.js";

let session = null;
let chain = Promise.resolve();

self.onmessage = (e) => {
  const m = e.data || {};
  if (m.type === "load") chain = chain.then(() => load(m));
  else if (m.type === "run") chain = chain.then(() => run(m));
  else if (m.type === "close") chain = chain.then(() => { session?.release(); session = null; self.close(); });
};

async function load({ modelBuffer, wasmBinary, warm }) {
  try {
    const t0 = performance.now();
    session = await createOrtSession(modelBuffer, { wasmBinary, warm });
    self.postMessage({ type: "ready", info: { ...session.info, loadMs: performance.now() - t0 } });
  } catch (err) {
    self.postMessage({ type: "error", message: String(err?.message || err) });
  }
}

async function run({ id, x, T }) {
  if (!session) { self.postMessage({ type: "result", id, error: "the sign model is not loaded" }); return; }
  try {
    const r = await session.run(x, T);
    self.postMessage({ type: "result", id, probs: r.probs, embedding: r.embedding, ms: r.ms }, [r.probs.buffer, r.embedding.buffer]);
  } catch (err) {
    self.postMessage({ type: "result", id, error: String(err?.message || err) });
  }
}

self.postMessage({ type: "hello" });
