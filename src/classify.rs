//! Small classification heads. v0.1 ships argmax over a slice of logits.

/// Return the index of the maximum value in `logits`.
///
/// Ties are broken by lowest index (matches the convention of every other
/// "stable argmax" in the ML ecosystem). Returns `None` for an empty slice
/// — argmax of nothing is undefined.
///
/// # Example
///
/// ```
/// use morsel::classify::argmax;
///
/// assert_eq!(argmax(&[0.1, 0.5, 0.3, 0.5]), Some(1));
/// assert_eq!(argmax(&[]), None);
/// ```
#[must_use]
pub fn argmax(logits: &[f32]) -> Option<usize> {
    if logits.is_empty() {
        return None;
    }
    let mut best_idx: usize = 0;
    let mut best_val: f32 = logits[0];
    for (i, &v) in logits.iter().enumerate().skip(1) {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    Some(best_idx)
}
