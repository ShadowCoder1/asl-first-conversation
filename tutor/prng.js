/* prng.js: a small seeded random number generator.
 *
 * Math.random cannot be seeded, so a session that used it can never be replayed.
 * Every random choice in the tutor goes through one of these, and the seed is
 * saved with the session. mulberry32: tiny, fast, good enough for shuffling. */
export function createPrng(seed) {
  let a = seed >>> 0;
  const float = () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const int = (n) => Math.floor(float() * n);
  const pick = (arr) => arr[int(arr.length)];
  const shuffle = (arr) => {
    const out = arr.slice();
    for (let i = out.length - 1; i > 0; i--) { const j = int(i + 1); [out[i], out[j]] = [out[j], out[i]]; }
    return out;
  };
  const normal = () => {                       // Box-Muller; 1 - float() avoids log(0)
    const u = 1 - float(), v = float();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  return { seed: seed >>> 0, float, int, pick, shuffle, normal };
}
