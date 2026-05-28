//! morsel — embeddable inference primitives for tiny neural networks.
//!
//! `morsel` provides a small library of scalar Rust implementations of common
//! neural-network layer primitives: dense (linear) layers, element-wise
//! activations, single-layer LSTM cell stepping, and an argmax classification
//! head. The crate is designed to be consumed by a downstream "model crate"
//! that ships its trained weights as `const` arrays in Rust source.
//!
//! # Design contract
//!
//! - **Inference only.** No training, no backpropagation.
//! - **Allocation-free.** Every primitive writes into a caller-provided
//!   `&mut [f32]` output slice; no `Vec`, no `Box`, no internal arena.
//! - **Deterministic.** Same inputs produce bit-identical outputs on the same
//!   CPU; ordering of f32 ops is fixed.
//! - **Safe Rust.** `unsafe_code` is forbidden in non-test builds.
//! - **No panics in the happy path.** Shape mismatches are caught by
//!   `debug_assert!`; release builds trust the caller.
//!
//! # Modules
//!
//! - [`linear`] — dense `y = W x + b`.
//! - [`activation`] — element-wise sigmoid, tanh, relu, and softmax.
//! - [`lstm`] — single-layer LSTM cell step and scan.
//! - [`classify`] — argmax over a slice of logits.
//!
//! # Worked-example shape
//!
//! ```
//! use morsel::activation::sigmoid;
//! use morsel::linear::linear;
//!
//! // Weights are typically `const` arrays in a generated `weights.rs`.
//! let w = [[0.5_f32, -0.5_f32], [0.25_f32, 0.75_f32]];
//! let b = [0.1_f32, -0.1_f32];
//! let x = [1.0_f32, 2.0_f32];
//! let mut y = [0.0_f32; 2];
//! linear(&w, &b, &x, &mut y);
//! // Pre-sigmoid: y = [-0.4, 1.65]; post-sigmoid: approximately [0.401, 0.839].
//! sigmoid(&mut y);
//! assert!((y[0] - 0.401_f32).abs() < 1e-2);
//! assert!((y[1] - 0.839_f32).abs() < 1e-2);
//! ```

#![cfg_attr(not(test), forbid(unsafe_code))]
#![allow(clippy::indexing_slicing)]
#![allow(clippy::as_conversions)]
#![allow(clippy::float_arithmetic)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::similar_names)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::suboptimal_flops)]

pub mod activation;
pub mod classify;
pub mod linear;
pub mod lstm;

pub use activation::{relu, sigmoid, softmax, tanh};
pub use classify::argmax;
pub use linear::linear;
pub use lstm::{lstm_scan, lstm_step};
