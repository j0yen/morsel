//! Element-wise activation functions: sigmoid, tanh, relu, softmax.
//!
//! All activations operate in-place on `&mut [f32]` so they compose cleanly
//! with [`crate::linear::linear`]'s output buffer pattern. None of them
//! allocate.

/// In-place sigmoid: `y[i] = 1 / (1 + exp(-y[i]))`.
///
/// # Numeric stability
///
/// `sigmoid` is implemented as `1 / (1 + exp(-x))` for `x >= 0` and as
/// `exp(x) / (1 + exp(x))` for `x < 0` to avoid overflow when `x` is large
/// and negative.
///
/// # Example
///
/// ```
/// use morsel::activation::sigmoid;
///
/// let mut y = [0.0_f32, 1.0_f32, -1.0_f32];
/// sigmoid(&mut y);
/// assert!((y[0] - 0.5).abs() < 1e-6);
/// assert!((y[1] - 0.7310586).abs() < 1e-5);
/// ```
pub fn sigmoid(y: &mut [f32]) {
    for v in y.iter_mut() {
        *v = sigmoid_scalar(*v);
    }
}

/// Scalar sigmoid. Exposed so LSTM and other compound primitives can apply
/// it to a single gate value without going through a slice.
#[inline]
#[must_use]
pub fn sigmoid_scalar(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// In-place hyperbolic tangent: `y[i] = tanh(y[i])`.
///
/// # Example
///
/// ```
/// use morsel::activation::tanh;
///
/// let mut y = [0.0_f32, 1.0_f32];
/// tanh(&mut y);
/// assert!((y[0]).abs() < 1e-6);
/// assert!((y[1] - 0.7615942).abs() < 1e-5);
/// ```
pub fn tanh(y: &mut [f32]) {
    for v in y.iter_mut() {
        *v = v.tanh();
    }
}

/// Scalar tanh wrapper for parity with [`sigmoid_scalar`].
#[inline]
#[must_use]
pub fn tanh_scalar(x: f32) -> f32 {
    x.tanh()
}

/// In-place rectified linear unit: `y[i] = max(0, y[i])`.
///
/// # Example
///
/// ```
/// use morsel::activation::relu;
///
/// let mut y = [-1.0_f32, 0.0_f32, 1.0_f32];
/// relu(&mut y);
/// assert_eq!(y, [0.0_f32, 0.0_f32, 1.0_f32]);
/// ```
pub fn relu(y: &mut [f32]) {
    for v in y.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

/// In-place softmax across the whole slice.
///
/// Computes the numerically-stable softmax: subtracts the max before
/// exponentiation to avoid overflow, then normalizes by the sum.
///
/// # Example
///
/// ```
/// use morsel::activation::softmax;
///
/// let mut y = [1.0_f32, 2.0_f32, 3.0_f32];
/// softmax(&mut y);
/// let sum: f32 = y.iter().sum();
/// assert!((sum - 1.0).abs() < 1e-6);
/// assert!(y[2] > y[1] && y[1] > y[0]);
/// ```
pub fn softmax(y: &mut [f32]) {
    if y.is_empty() {
        return;
    }
    // Find the max for numeric stability.
    let mut m = y[0];
    for &v in y.iter() {
        if v > m {
            m = v;
        }
    }
    // Exponentiate shifted values and accumulate the sum.
    let mut sum: f32 = 0.0;
    for v in y.iter_mut() {
        *v = (*v - m).exp();
        sum += *v;
    }
    // Normalize.
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for v in y.iter_mut() {
            *v *= inv;
        }
    }
}
