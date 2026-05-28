# morsel

Embeddable inference primitives for tiny neural networks in Rust.

## TL;DR

`morsel` is the missing middle between "real ML in Rust" (candle, burn, ONNX
runtime — 200MB+ of dependencies and assets) and "no ML at all"
(`if x.contains("meow")` brittle heuristics). It provides a small library of
scalar Rust implementations of common neural-network layer primitives — dense
layers, element-wise activations, single-layer LSTM cells, argmax — designed to
be consumed by a downstream "model crate" that ships its trained weights as
`const` arrays in Rust source.

One `cargo add morsel`, one `use cat_meow::is_cat;`, done. No init, no
`Engine::new()`, no model load. Inference is allocation-free, panic-free in the
happy path, and bit-deterministic.

## Design contract

- **Inference only.** No training, no backpropagation.
- **Allocation-free.** Every primitive writes into a caller-provided
  `&mut [f32]` output slice; no `Vec`, no `Box`, no internal arena.
- **Deterministic.** Same inputs produce bit-identical outputs on the same CPU.
- **Safe Rust.** `unsafe_code` is forbidden in non-test builds.
- **No panics in the happy path.** Shape mismatches are caught by
  `debug_assert!`; release builds trust the caller.

## What's in v0.1

| Module | Primitives |
| --- | --- |
| `linear` | `linear`, `linear_flat`, `linear_flat_accumulate` |
| `activation` | `sigmoid`, `tanh`, `relu`, `softmax` (+ scalar variants) |
| `lstm` | `lstm_step`, `lstm_scan` (canonical PyTorch LSTMCell formula) |
| `classify` | `argmax` |

The crate's only runtime dependency is the Rust standard library. `proptest`
is a dev-dependency for property-based tests.

## Acceptance criteria (this v0.1 satisfies)

| ID | Level | Description |
| --- | --- | --- |
| AC1 | MUST | Linear (dense) layer: `y = W * x + b` matches a hand-computed reference within `1e-6`. |
| AC2 | MUST | Activation functions (sigmoid, tanh, relu, softmax) match numeric references within `1e-6` and are numerically stable for large inputs. |
| AC3 | MUST | Single-layer LSTM cell step matches the canonical PyTorch LSTMCell formula within `1e-5`. |
| AC4 | MUST | LSTM scan across a sequence matches stepping the cell manually within `1e-5`. |
| AC5 | MUST | Argmax over `&[f32]`: ties pick lowest index; verified against a naive oracle on 1000 randomized cases. |
| AC6 | MUST | End-to-end composition test: a hand-designed sign-of-sum classifier (LSTM scan → linear → softmax → argmax) classifies ≥3/4 canonical inputs correctly. |
| AC7 | MUST | Allocation-free composed pipeline (LSTM step + linear + sigmoid) runs 100 iterations against pre-allocated buffers without panic. |
| AC8 | SHOULD | Determinism: same inputs produce bit-identical f32 outputs across runs. |
| AC9 | SHOULD | Every public item carries a doc comment with a runnable example or a math formula (lint-enforced via an integration test). |
| AC10 | SHOULD | Domain invariants (sigmoid ∈ [0,1] and monotonic; tanh ∈ [-1,1]; softmax is a probability distribution; argmax order-preserving under softmax) verified via 7 proptest properties + a 200-case acceptance test. |

## Install

```toml
[dependencies]
morsel = { git = "https://github.com/j0yen/morsel" }
```

Once published to crates.io:

```toml
[dependencies]
morsel = "0.1"
```

## Example

```rust
use morsel::activation::sigmoid;
use morsel::linear::linear;

// Weights typically live in a generated `weights.rs` as `const` arrays.
let w = [[0.5_f32, -0.5_f32], [0.25_f32, 0.75_f32]];
let b = [0.1_f32, -0.1_f32];
let x = [1.0_f32, 2.0_f32];

let mut y = [0.0_f32; 2];
linear(&w, &b, &x, &mut y);
sigmoid(&mut y);
// y is now in [0, 1] and ready to feed a classification head.
```

## Non-goals (v0.1)

- Model training — `morsel` is inference-only.
- GPU backends — pure CPU.
- Dynamic model loading — weights are baked into the consumer crate.
- Quantization research — f32 only in v0.1.
- The `morsel bake` CLI (safetensors → `weights.rs`) — follow-on PRD.
- Audio frontends (`LogMel`, `Mfcc`) — follow-on PRD.
- `Conv1d`, `Embedding`, `Gru` — follow-on PRD.
- SIMD acceleration — follow-on PRD.
- `no_std` — follow-on PRD (libm dep was excluded for v0.1).

## License

Dual-licensed under MIT and Apache-2.0. See `LICENSE-MIT` and `LICENSE-APACHE`.

## Provenance

`morsel` was built end-to-end by the `autobuilder` skill (5-stage pipeline:
intake → scaffold → iterate-and-prove → 25-receipt risk gate → postmortem) from
[`PRD-morsel.md`](https://github.com/j0yen/autobuilder/blob/main/PRD-morsel.md).
The intent-card, iteration receipts, and gate verdict live in
`target/autobuilder/` (excluded from this repo via `.gitignore` but reproducible
from a clean clone by running `bash scripts/run-metrics.sh`).
