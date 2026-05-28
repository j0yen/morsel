//! Property-based invariant tests.
//!
//! Read-only after scaffold. The edit-agent must NOT modify proptests.
//! Add invariants here when the intake surfaces a domain-level invariant
//! that survives across iterations (e.g. "reverse is its own inverse").

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_arithmetic,
    clippy::indexing_slicing,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::redundant_clone,
    clippy::float_cmp
)]

use morsel::activation::{relu, sigmoid, softmax, tanh};
use morsel::classify::argmax;
use morsel::linear::linear_flat;
use proptest::prelude::*;

proptest! {
    /// Sigmoid output lies strictly in (0, 1) for finite inputs, and the
    /// function is monotonic non-decreasing.
    #[test]
    fn sigmoid_in_unit_interval(xs in proptest::collection::vec(-50.0_f32..50.0_f32, 1..32)) {
        let mut y = xs.clone();
        sigmoid(&mut y);
        for v in &y {
            prop_assert!(v.is_finite(), "sigmoid produced non-finite for {xs:?}");
            prop_assert!(*v >= 0.0 && *v <= 1.0, "sigmoid out of [0,1]: {v}");
        }
        // Monotonicity on a sorted copy.
        let mut sorted = xs.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mut sorted_y = sorted.clone();
        sigmoid(&mut sorted_y);
        for i in 1..sorted_y.len() {
            prop_assert!(sorted_y[i] >= sorted_y[i - 1] - 1e-5,
                "sigmoid not monotonic: {} < {}", sorted_y[i], sorted_y[i - 1]);
        }
    }

    /// Tanh output lies in [-1, 1].
    #[test]
    fn tanh_in_bounded_interval(xs in proptest::collection::vec(-100.0_f32..100.0_f32, 1..32)) {
        let mut y = xs.clone();
        tanh(&mut y);
        for v in &y {
            prop_assert!(v.is_finite(), "tanh non-finite for {xs:?}");
            prop_assert!(*v >= -1.0 && *v <= 1.0, "tanh out of [-1,1]: {v}");
        }
    }

    /// ReLU never produces negative values, and never increases a positive
    /// value.
    #[test]
    fn relu_clamps_at_zero(xs in proptest::collection::vec(-100.0_f32..100.0_f32, 1..32)) {
        let mut y = xs.clone();
        relu(&mut y);
        for i in 0..xs.len() {
            prop_assert!(y[i] >= 0.0, "relu produced negative: {}", y[i]);
            if xs[i] >= 0.0 {
                prop_assert_eq!(y[i], xs[i], "relu changed non-negative input");
            } else {
                prop_assert_eq!(y[i], 0.0_f32, "relu didn't zero a negative input");
            }
        }
    }

    /// Softmax outputs form a probability distribution (sum to 1, all >= 0).
    #[test]
    fn softmax_is_probability_distribution(xs in proptest::collection::vec(-20.0_f32..20.0_f32, 1..16)) {
        let mut y = xs;
        softmax(&mut y);
        let sum: f32 = y.iter().sum();
        prop_assert!((sum - 1.0).abs() < 1e-5, "softmax doesn't sum to 1: {sum}");
        for v in &y {
            prop_assert!(*v >= 0.0 && *v <= 1.0, "softmax out of [0,1]: {v}");
            prop_assert!(v.is_finite(), "softmax non-finite: {v}");
        }
    }

    /// Argmax of softmax(x) equals argmax(x) (softmax is order-preserving).
    #[test]
    fn argmax_invariant_under_softmax(xs in proptest::collection::vec(-10.0_f32..10.0_f32, 1..16)) {
        let raw_argmax = argmax(&xs);
        let mut sm = xs.clone();
        softmax(&mut sm);
        let sm_argmax = argmax(&sm);
        prop_assert_eq!(raw_argmax, sm_argmax,
            "argmax changed by softmax: raw={:?} sm={:?} xs={:?}",
            raw_argmax, sm_argmax, xs);
    }

    /// Linear with zero weights and zero bias yields zero output for any input.
    #[test]
    fn linear_zero_zero(x in proptest::collection::vec(-100.0_f32..100.0_f32, 1..8)) {
        let in_dim = x.len();
        let out_dim = 4;
        let w = vec![0.0_f32; in_dim * out_dim];
        let b = vec![0.0_f32; out_dim];
        let mut y = vec![1.0_f32; out_dim];
        linear_flat(&w, &b, &x, &mut y, in_dim, out_dim);
        for v in &y {
            prop_assert_eq!(*v, 0.0_f32, "linear with zero W,b should give zeros");
        }
    }

    /// Argmax returns Some(_) for nonempty input.
    #[test]
    fn argmax_some_for_nonempty(xs in proptest::collection::vec(-1e6_f32..1e6_f32, 1..32)) {
        let r = argmax(&xs);
        prop_assert!(r.is_some(), "argmax returned None for nonempty input");
        let idx = r.unwrap();
        prop_assert!(idx < xs.len(), "argmax index out of bounds");
        // Returned index must hold the maximum.
        for (j, v) in xs.iter().enumerate() {
            prop_assert!(*v <= xs[idx] + 1e-6,
                "argmax not actually max: xs[{j}]={v} > xs[{idx}]={}", xs[idx]);
        }
    }
}
