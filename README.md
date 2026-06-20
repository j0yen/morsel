# morsel

Scalar, allocation-free inference primitives for tiny neural networks in Rust — dense layers, activations, an LSTM cell, and an argmax head, with the trained weights baked into the consumer crate as `const` arrays.

## Why it exists

There is a gap between the two ways Rust programs do classification. On one side, the real ML stacks — candle, burn, an ONNX runtime — which bring 200MB+ of dependencies and a model file you have to load at startup. On the other, the brittle heuristic: `if x.contains("meow")`, which is free but wrong the moment the input drifts.

`morsel` is the middle. A model small enough to fit in a few hundred trained floats — a keyword spotter, a tiny intent classifier, a sign-of-sum toy — does not need a runtime. It needs four operations: a dense layer, an activation, maybe an LSTM step, and an argmax. `morsel` is those operations and nothing else. The weights live in your binary as `const` arrays, so there is no init, no `Engine::new()`, no file to ship alongside the executable. Inference is a function call.

## Design contract

These are the invariants the crate holds, enforced by tests and lints rather than promised in prose:

- **Inference only.** No training, no backpropagation.
- **Allocation-free.** Every primitive writes into a caller-provided `&mut [f32]`; no `Vec`, no `Box`, no internal arena.
- **Deterministic.** The same inputs produce bit-identical outputs on the same CPU — the order of float operations is fixed.
- **Safe Rust.** `unsafe_code` is denied outside test builds.
- **No panics on the happy path.** Shape mismatches are caught by `debug_assert!`; release builds trust the caller.

## What's in v0.1

| Module | Primitives |
| --- | --- |
| `linear` | `linear`, `linear_flat`, `linear_flat_accumulate` |
| `activation` | `sigmoid`, `tanh`, `relu`, `softmax` (and `*_scalar` variants) |
| `lstm` | `lstm_step`, `lstm_scan` — the canonical PyTorch `LSTMCell` formula |
| `classify` | `argmax` (ties pick the lowest index) |

The only runtime dependency is the standard library. `proptest` is a dev-dependency for the property tests.

## Install

The crate is not on crates.io. Depend on it by git:

```toml
[dependencies]
morsel = { git = "https://github.com/j0yen/morsel" }
```

## First run

A linear layer feeding a sigmoid — the smallest useful pipeline. Weights would normally be `const` arrays in a generated `weights.rs`; here they are inline:

```rust
use morsel::activation::sigmoid;
use morsel::linear::linear;

let w = [[0.5_f32, -0.5_f32], [0.25_f32, 0.75_f32]];
let b = [0.1_f32, -0.1_f32];
let x = [1.0_f32, 2.0_f32];

let mut y = [0.0_f32; 2];
linear(&w, &b, &x, &mut y);
// Pre-sigmoid:  y = [-0.4, 1.65]
sigmoid(&mut y);
// Post-sigmoid: y ≈ [0.401, 0.839], ready for a classification head.
```

The output slice is yours; `morsel` only writes into it. That is what lets a full pipeline — LSTM step, linear, activation, argmax — run against pre-allocated buffers without touching the heap.

## How it works

Each primitive is a plain scalar loop, written for correctness and determinism rather than speed. `linear` comes in a const-generic form (dimensions known at compile time) and a slice-typed `linear_flat` form (dimensions known at runtime, used internally by the LSTM cell). `softmax` subtracts the max before exponentiating so large logits don't overflow. The activations and the argmax tie-break rule are checked against naive oracles and against domain invariants — sigmoid stays in `[0, 1]` and monotonic, tanh in `[-1, 1]`, softmax sums to one — under property tests.

## Where it fits

`morsel` is the runtime half of a two-crate pair. Its sibling, [`morsel-bake`](https://github.com/j0yen/morsel-bake), takes a trained model and emits the `weights.rs` that a consumer crate compiles against `morsel`. Train elsewhere, bake the weights to Rust source, link `morsel`, ship one binary.

## Status

v0.1, and honest about its edges. No training, no GPU, no dynamic model loading, f32 only. `Conv1d`, `Embedding`, `Gru`, SIMD, audio frontends, and `no_std` are deliberately out of scope for this version — each is a follow-on, not a hidden limitation.

## License

Dual-licensed under MIT and Apache-2.0. See `LICENSE-MIT` and `LICENSE-APACHE`.
