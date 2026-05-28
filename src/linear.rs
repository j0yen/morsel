//! Dense (linear) layer: `y = W * x + b`.
//!
//! The weight matrix is stored as a `&[[f32; IN]; OUT]` — row-major, one
//! row per output unit. This is the shape `morsel-bake` will emit and
//! matches PyTorch's `nn.Linear.weight` (out_features × in_features).

/// Computes `y = W * x + b` in scalar f32.
///
/// `w` is row-major: `w[o][i]` is the weight from input `i` to output `o`.
/// `b` is the bias vector of length `OUT`. `x` is the input slice of length
/// `IN`. `y` is the output slice of length `OUT`; it is overwritten.
///
/// # Shape contract
///
/// In debug builds, the lengths of `b`, `x`, and `y` are checked against the
/// dimensions of `w` via `debug_assert!`. In release builds the caller is
/// trusted; passing mismatched slices is undefined-behavior-free (safe Rust)
/// but will produce a runtime panic from indexing.
///
/// # Determinism
///
/// The inner-product accumulator iterates `i` from `0..IN` in order. The
/// result is bit-identical across runs on the same CPU.
///
/// # Example
///
/// ```
/// use morsel::linear::linear;
///
/// // 2-input, 2-output identity-ish layer.
/// let w = [[1.0_f32, 0.0_f32], [0.0_f32, 1.0_f32]];
/// let b = [0.0_f32, 0.0_f32];
/// let x = [3.0_f32, 4.0_f32];
/// let mut y = [0.0_f32; 2];
/// linear(&w, &b, &x, &mut y);
/// assert_eq!(y, [3.0_f32, 4.0_f32]);
/// ```
pub fn linear<const IN: usize, const OUT: usize>(
    w: &[[f32; IN]; OUT],
    b: &[f32; OUT],
    x: &[f32; IN],
    y: &mut [f32; OUT],
) {
    for (o, y_o) in y.iter_mut().enumerate() {
        let row = &w[o];
        let mut acc = b[o];
        for i in 0..IN {
            acc += row[i] * x[i];
        }
        *y_o = acc;
    }
}

/// Slice-typed variant of [`linear`] for callers whose dimensions are not
/// known at compile time. This is the form the LSTM cell uses internally
/// (with `4 * hidden_size`-shaped weight rows).
///
/// `w` is a flat row-major buffer of length `out_dim * in_dim`. `b` has
/// length `out_dim`. `x` has length `in_dim`. `y` has length `out_dim`.
///
/// # Shape contract
///
/// `debug_assert!`s check `w.len() == out_dim * in_dim`,
/// `b.len() == out_dim`, `x.len() == in_dim`, `y.len() == out_dim`.
///
/// # Example
///
/// ```
/// use morsel::linear::linear_flat;
///
/// // 3-input, 2-output, row-major.
/// // y0 = 1*x0 + 2*x1 + 3*x2 + b0
/// // y1 = 4*x0 + 5*x1 + 6*x2 + b1
/// let w = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let b = [0.0_f32, 0.0];
/// let x = [1.0_f32, 1.0, 1.0];
/// let mut y = [0.0_f32; 2];
/// linear_flat(&w, &b, &x, &mut y, 3, 2);
/// assert_eq!(y, [6.0_f32, 15.0_f32]);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn linear_flat(
    w: &[f32],
    b: &[f32],
    x: &[f32],
    y: &mut [f32],
    in_dim: usize,
    out_dim: usize,
) {
    debug_assert_eq!(w.len(), in_dim * out_dim, "w shape");
    debug_assert_eq!(b.len(), out_dim, "b shape");
    debug_assert_eq!(x.len(), in_dim, "x shape");
    debug_assert_eq!(y.len(), out_dim, "y shape");
    for (o, y_o) in y.iter_mut().enumerate() {
        let row_start = o * in_dim;
        let mut acc = b[o];
        for i in 0..in_dim {
            acc += w[row_start + i] * x[i];
        }
        *y_o = acc;
    }
}

/// Accumulates `y += W * x` (no bias, no overwrite). Used inside the LSTM
/// cell to add the input-to-hidden and hidden-to-hidden contributions into
/// a single buffer that already contains the bias.
///
/// Formula: `y[o] = y[o] + sum_{i in 0..in_dim} w[o * in_dim + i] * x[i]`.
///
/// `w` is row-major flat, length `out_dim * in_dim`. `x` length `in_dim`.
/// `y` length `out_dim`.
///
/// # Example
///
/// ```
/// use morsel::linear::linear_flat_accumulate;
///
/// // Identity-shape 2x2: y0 += 1*x0 + 0*x1; y1 += 0*x0 + 1*x1
/// let w = [1.0_f32, 0.0, 0.0, 1.0];
/// let x = [3.0_f32, 4.0];
/// let mut y = [10.0_f32, 20.0];
/// linear_flat_accumulate(&w, &x, &mut y, 2, 2);
/// assert_eq!(y, [13.0_f32, 24.0_f32]);
/// ```
pub fn linear_flat_accumulate(
    w: &[f32],
    x: &[f32],
    y: &mut [f32],
    in_dim: usize,
    out_dim: usize,
) {
    debug_assert_eq!(w.len(), in_dim * out_dim, "w shape");
    debug_assert_eq!(x.len(), in_dim, "x shape");
    debug_assert_eq!(y.len(), out_dim, "y shape");
    for (o, y_o) in y.iter_mut().enumerate() {
        let row_start = o * in_dim;
        let mut acc = *y_o;
        for i in 0..in_dim {
            acc += w[row_start + i] * x[i];
        }
        *y_o = acc;
    }
}
