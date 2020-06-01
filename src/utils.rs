use tch::Kind;

/// Returns the smallest representable floating point number such that 1.0 + eps != 1.0.
pub fn eps(kind: Kind) -> Option<f64> {
    Some(match kind {
        Kind::Half => 0.0009765625,
        Kind::Float => std::f32::EPSILON as _,
        Kind::Double => std::f64::EPSILON,
        _ => return None,
    })
}

/// Returns the smallest representable floating point number.
pub fn tiny(kind: Kind) -> Option<f64> {
    Some(match kind {
        Kind::Half => 6.103515625e-05,
        Kind::Float => std::f32::MIN as _,
        Kind::Double => std::f64::MIN,
        _ => return None,
    })
}
