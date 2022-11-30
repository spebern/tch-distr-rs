use tch::{Kind, Tensor};

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
        Kind::Float => std::f32::MIN_POSITIVE as _,
        Kind::Double => std::f64::MIN_POSITIVE,
        _ => return None,
    })
}

/// Returns the smallest representable number (typically -max).
pub fn min(kind: Kind) -> Option<f64> {
    // TODO: need refine
    Some(match kind {
        // Kind::Half => 6.103515625e-05,
        Kind::Float => -3.4028234663852886e+38,
        Kind::Double => -1.7976931348623157e+308,
        _ => return None,
    })
}

fn clamp_probs(probs: &Tensor) -> Tensor {
    let eps = eps(probs.kind()).unwrap();
    probs.clamp(eps, 1.0 - eps)
}

pub fn probs_to_logits(probs: &Tensor, is_binary: bool) -> Tensor {
    let ps_clamped = clamp_probs(probs);
    if is_binary {
        ps_clamped.log() - (-ps_clamped).log1p()
    } else {
        ps_clamped.log()
    }
}

pub fn logits_to_probs(logits: &Tensor, is_binary: bool) -> Tensor {
    if is_binary {
        logits.sigmoid()
    } else {
        logits.softmax(-1, logits.kind())
    }
}

pub fn infinity(kind: Kind) -> Tensor {
    match kind {
        Kind::Float => std::f32::INFINITY.into(),
        Kind::Double => std::f64::INFINITY.into(),
        k => panic!("{:?} cannot represent infinity", k),
    }
}

pub fn standard_normal(shape: &[i64], dtype: tch::Kind, device: tch::Device) -> Tensor {
    Tensor::empty(shape, (dtype, device)).normal_(0., 1.)
}
