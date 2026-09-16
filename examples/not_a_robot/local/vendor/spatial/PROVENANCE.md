# Spatial task dependencies

## Chess rules

`chess-1.4.0.js` is the unchanged ESM distribution of chess.js 1.4.0,
downloaded from https://cdn.jsdelivr.net/npm/chess.js@1.4.0/dist/esm/chess.js.
Upstream: https://github.com/jhlywa/chess.js/tree/v1.4.0.
License: BSD-2-Clause, retained in `LICENSE.chess.js`.

- JS SHA256: `76c7c34f0e2e9ab076521a5d6fe786a9cce537bb1b6f29d32a9c9970b5b232d2`
- License SHA256: `0b3a3c2b4432a26bb18f9d06f5bba4de015bcc980306b7db28b06025495e2186`

This library validates chess rules; it does not provide an opponent. Its exact
version differs from the unpinned original library. In particular, en passant
FEN serialization and repetition counting can differ. The local game adapts
invalid-move exceptions and uses coordinate moves rather than SAN input.

## Stockfish opponent

`stockfish-17-lite-single.js` and `stockfish-17-lite-single.wasm` are unchanged
paired files from the official stockfish.js repository, tag `v17.1.0`, fixed
commit `f9512ef9aff391026813a56855dd086cb72a2d58`:

- [JavaScript loader](https://github.com/nmrugg/stockfish.js/blob/f9512ef9aff391026813a56855dd086cb72a2d58/src/stockfish-17-lite-single.js)
- [WebAssembly binary](https://github.com/nmrugg/stockfish.js/blob/f9512ef9aff391026813a56855dd086cb72a2d58/src/stockfish-17-lite-single.wasm)
- [Corresponding upstream source](https://github.com/nmrugg/stockfish.js/tree/f9512ef9aff391026813a56855dd086cb72a2d58/src)
- [Upstream Copying.txt](https://github.com/nmrugg/stockfish.js/blob/f9512ef9aff391026813a56855dd086cb72a2d58/Copying.txt), retained unchanged as `LICENSE.stockfish` (GPL-3.0).

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `stockfish-17-lite-single.js` | 20181 | `ef06615dc8cf5974e9f3e73d1663b9ca87f2bd59b97e661af0b22e80801b6409` |
| `stockfish-17-lite-single.wasm` | 7157593 | `8e7d58fd36242f9163fb4881781cbb59b73c2b00067e6f4b02d31a48c58e5fe8` |
| `LICENSE.stockfish` | 35147 | `8ceb4b9ee5adedde47b31e975c1d90c73ad27b6b165a1dcd80c7c545eb65b903` |

The loader matches the captured original loader's bytes. The original deployed
WASM was not acquired, so byte identity of that deployed binary is unverified.
This is an official paired build, not proof of identical results on every device.
Real-time search can vary with scheduling, hardware and engine state.

The local game uses one same-origin Worker, UCI Skill Level 10 and
`go movetime 1000`. There is no heuristic fallback. Loading and protocol errors
remain infrastructure errors, not game losses or wins. Refresh drains an
in-flight old result; closing the task terminates its owned Worker.

These dependencies are not covered by a first-party project license. Retain their
notices; binary redistribution requires appropriate corresponding-source handling
under the upstream license. Their inclusion does not license the original site's
artwork, sounds or other assets, which remain separate fidelity gaps.
