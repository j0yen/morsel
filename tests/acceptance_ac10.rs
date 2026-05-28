//! Acceptance test for AC10 (SHOULD) — added 2026-05-27 via intent-card amendment.
//!
//! Project: morsel (lib)
//! AC description: Domain-level invariants hold under randomized inputs (see
//!   tests/proptest_invariants.rs for the canonical property assertions). This
//!   acceptance test runs a representative random-input smoke check against
//!   the same invariants to ensure the metric harness counts the SHOULD-level
//!   property coverage as an acceptance pass when proptest itself is green.
//!
//! The proptest macro creates its own #[test] functions that the harness counts
//! as test_NAME, not acceptance_NAME — so a small acceptance_ac10 wrapper test
//! is required to register the AC in the unfakeable-metric counter
//! (which grep-matches `^test acceptance_[a-z0-9_]+ \.\.\. ok`).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::doc_markdown,
    clippy::float_arithmetic,
    clippy::indexing_slicing,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::as_conversions,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::many_single_char_names,
    clippy::suboptimal_flops,
    clippy::float_cmp
)]

use morsel::activation::{relu, sigmoid, softmax, tanh};
use morsel::classify::argmax;
use morsel::linear::linear_flat;

/// Tiny LCG for reproducibility in the acceptance harness (proptest is
/// already exercised in tests/proptest_invariants.rs; this test runs a
/// fixed-seed random batch to count as an "acceptance" hit).
struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        (self.0 >> 16) as u32
    }
    fn next_f32_range(&mut self, lo: f32, hi: f32) -> f32 {
        let u = self.next_u32();
        let frac = (u as f32) / (u32::MAX as f32);
        lo + frac * (hi - lo)
    }
}

#[test]
fn acceptance_ac10() {
    let mut rng = Lcg(0xC0FF_EEEE);

    for _ in 0..200 {
        let n = ((rng.next_u32() % 30) + 1) as usize;

        // sigmoid in [0, 1] + finite
        let mut s: Vec<f32> = (0..n).map(|_| rng.next_f32_range(-30.0, 30.0)).collect();
        sigmoid(&mut s);
        for &v in &s {
            assert!(v.is_finite(), "sigmoid non-finite");
            assert!((0.0..=1.0).contains(&v), "sigmoid out of range: {v}");
        }

        // tanh in [-1, 1] + finite
        let mut t: Vec<f32> = (0..n).map(|_| rng.next_f32_range(-50.0, 50.0)).collect();
        tanh(&mut t);
        for &v in &t {
            assert!(v.is_finite(), "tanh non-finite");
            assert!((-1.0..=1.0).contains(&v), "tanh out of range: {v}");
        }

        // relu clamps at zero
        let xs: Vec<f32> = (0..n).map(|_| rng.next_f32_range(-100.0, 100.0)).collect();
        let mut r = xs.clone();
        relu(&mut r);
        for i in 0..n {
            assert!(r[i] >= 0.0, "relu negative");
            if xs[i] >= 0.0 {
                assert_eq!(r[i].to_bits(), xs[i].to_bits(), "relu altered non-negative");
            } else {
                assert_eq!(r[i], 0.0_f32, "relu didn't zero negative");
            }
        }

        // softmax is a probability distribution
        let mut sm: Vec<f32> = (0..n).map(|_| rng.next_f32_range(-10.0, 10.0)).collect();
        softmax(&mut sm);
        let sum: f32 = sm.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sum {sum}");
        for &v in &sm {
            assert!(v.is_finite(), "softmax non-finite");
            assert!((0.0..=1.0).contains(&v), "softmax out of [0,1]");
        }

        // argmax order-preserving under softmax
        let raw_xs: Vec<f32> = (0..n).map(|_| rng.next_f32_range(-5.0, 5.0)).collect();
        let raw_idx = argmax(&raw_xs);
        let mut sm2 = raw_xs.clone();
        softmax(&mut sm2);
        let sm_idx = argmax(&sm2);
        assert_eq!(raw_idx, sm_idx, "argmax changed by softmax");

        // linear with zero weights/bias yields zeros
        let in_dim = 4;
        let out_dim = 3;
        let x_lin: Vec<f32> = (0..in_dim).map(|_| rng.next_f32_range(-1.0, 1.0)).collect();
        let w = vec![0.0_f32; in_dim * out_dim];
        let b = vec![0.0_f32; out_dim];
        let mut y = vec![1.0_f32; out_dim];
        linear_flat(&w, &b, &x_lin, &mut y, in_dim, out_dim);
        for &v in &y {
            assert_eq!(v, 0.0_f32, "linear zero-zero should yield zeros");
        }
    }
}
