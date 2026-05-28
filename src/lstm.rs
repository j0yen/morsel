//! Single-layer LSTM cell stepping and scan.
//!
//! Implements the canonical PyTorch `nn.LSTMCell` formulation:
//!
//! ```text
//! gates = W_ih * x + W_hh * h_prev + b               // [4 * hidden]
//! i = sigmoid(gates[0*hidden..1*hidden])
//! f = sigmoid(gates[1*hidden..2*hidden])
//! g = tanh   (gates[2*hidden..3*hidden])
//! o = sigmoid(gates[3*hidden..4*hidden])
//! c = f * c_prev + i * g
//! h = o * tanh(c)
//! ```
//!
//! Gate order matches PyTorch (`i, f, g, o`). The fused bias `b` of length
//! `4 * hidden` carries the combined `b_ih + b_hh` term — PyTorch's
//! convention. `morsel-bake` will collapse the two PyTorch biases into one
//! buffer at bake time.
//!
//! Weights are passed as flat row-major slices:
//! - `w_ih`: `[4 * hidden, input_dim]`, length `4 * hidden * input_dim`.
//! - `w_hh`: `[4 * hidden, hidden]`, length `4 * hidden * hidden`.
//! - `b`:    length `4 * hidden`.

use crate::activation::{sigmoid_scalar, tanh_scalar};
use crate::linear::{linear_flat, linear_flat_accumulate};

/// Step a single LSTM cell forward by one timestep.
///
/// Updates `h` and `c` in place using the gate equations above. `gates_buf`
/// is a caller-provided scratch buffer of length `4 * hidden`; the caller
/// owns it so the step itself allocates nothing.
///
/// # Shape contract
///
/// - `x.len() == input_dim`
/// - `h.len() == hidden`
/// - `c.len() == hidden`
/// - `w_ih.len() == 4 * hidden * input_dim`
/// - `w_hh.len() == 4 * hidden * hidden`
/// - `b.len() == 4 * hidden`
/// - `gates_buf.len() == 4 * hidden`
///
/// All checked via `debug_assert!`.
///
/// # Example
///
/// ```
/// use morsel::lstm::lstm_step;
///
/// // 2-input, 3-hidden LSTM with hand-set weights and zero state.
/// let input_dim = 2;
/// let hidden = 3;
/// let four_h = 4 * hidden;
/// let w_ih = vec![0.0_f32; four_h * input_dim];
/// let w_hh = vec![0.0_f32; four_h * hidden];
/// let b = vec![0.0_f32; four_h];
/// let mut h = vec![0.0_f32; hidden];
/// let mut c = vec![0.0_f32; hidden];
/// let mut gates_buf = vec![0.0_f32; four_h];
/// let x = vec![0.1_f32, 0.2_f32];
/// lstm_step(
///     &x, &mut h, &mut c, &w_ih, &w_hh, &b, &mut gates_buf,
///     input_dim, hidden,
/// );
/// // Zero weights + bias produce zero gates pre-activation. After
/// // sigmoid the i/f/o gates are 0.5; the g gate (tanh) is 0. c stays 0
/// // and h stays 0.
/// for &v in &h { assert!(v.abs() < 1e-6); }
/// for &v in &c { assert!(v.abs() < 1e-6); }
/// ```
pub fn lstm_step(
    x: &[f32],
    h: &mut [f32],
    c: &mut [f32],
    w_ih: &[f32],
    w_hh: &[f32],
    b: &[f32],
    gates_buf: &mut [f32],
    input_dim: usize,
    hidden: usize,
) {
    let four_h = 4 * hidden;
    debug_assert_eq!(x.len(), input_dim, "x shape");
    debug_assert_eq!(h.len(), hidden, "h shape");
    debug_assert_eq!(c.len(), hidden, "c shape");
    debug_assert_eq!(w_ih.len(), four_h * input_dim, "w_ih shape");
    debug_assert_eq!(w_hh.len(), four_h * hidden, "w_hh shape");
    debug_assert_eq!(b.len(), four_h, "b shape");
    debug_assert_eq!(gates_buf.len(), four_h, "gates_buf shape");

    // gates = W_ih * x + b
    linear_flat(w_ih, b, x, gates_buf, input_dim, four_h);
    // gates += W_hh * h_prev
    linear_flat_accumulate(w_hh, h, gates_buf, hidden, four_h);

    // Split gates into i, f, g, o slices and apply activations + cell update.
    // PyTorch convention: gates are laid out [i; f; g; o] contiguously.
    for k in 0..hidden {
        let i = sigmoid_scalar(gates_buf[k]);
        let f = sigmoid_scalar(gates_buf[hidden + k]);
        let g = tanh_scalar(gates_buf[2 * hidden + k]);
        let o = sigmoid_scalar(gates_buf[3 * hidden + k]);
        let new_c = f * c[k] + i * g;
        c[k] = new_c;
        h[k] = o * new_c.tanh();
    }
}

/// Scan an LSTM cell across a sequence of input frames, mutating `h` and
/// `c` in place. `frames` is laid out flat row-major:
/// `frames[t * input_dim + i]` is the `i`-th feature of the `t`-th frame.
///
/// On return, `h` and `c` hold the final hidden and cell states. This is
/// what a downstream head (linear + softmax + argmax) would consume.
///
/// # Example
///
/// ```
/// use morsel::lstm::lstm_scan;
///
/// let input_dim = 1;
/// let hidden = 2;
/// let four_h = 4 * hidden;
/// let frames = vec![0.0_f32; 5 * input_dim]; // 5 timesteps of zeros
/// let w_ih = vec![0.0_f32; four_h * input_dim];
/// let w_hh = vec![0.0_f32; four_h * hidden];
/// let b = vec![0.0_f32; four_h];
/// let mut h = vec![0.0_f32; hidden];
/// let mut c = vec![0.0_f32; hidden];
/// let mut gates_buf = vec![0.0_f32; four_h];
/// lstm_scan(
///     &frames, &mut h, &mut c, &w_ih, &w_hh, &b, &mut gates_buf,
///     input_dim, hidden,
/// );
/// for &v in &h { assert!(v.abs() < 1e-6); }
/// ```
#[allow(clippy::too_many_arguments)]
pub fn lstm_scan(
    frames: &[f32],
    h: &mut [f32],
    c: &mut [f32],
    w_ih: &[f32],
    w_hh: &[f32],
    b: &[f32],
    gates_buf: &mut [f32],
    input_dim: usize,
    hidden: usize,
) {
    debug_assert_eq!(
        frames.len() % input_dim,
        0,
        "frames.len() must be a multiple of input_dim"
    );
    let n_frames = frames.len() / input_dim;
    for t in 0..n_frames {
        let start = t * input_dim;
        let end = start + input_dim;
        let frame = &frames[start..end];
        lstm_step(
            frame, h, c, w_ih, w_hh, b, gates_buf, input_dim, hidden,
        );
    }
}
