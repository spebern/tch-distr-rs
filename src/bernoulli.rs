use super::Distribution;
use crate::utils::eps;
use tch::{Kind, Reduction, Tensor};

/// A Bernoulli distribution.
pub struct Bernoulli {
    probs: Tensor,
    logits: Tensor,
}

impl Bernoulli {
    /// Creates a Bernoulli distribution from probabilities.
    pub fn from_probs(probs: Tensor) -> Self {
        Self {
            logits: probs_to_logits(&probs, true),
            probs,
        }
    }

    /// Creates a Bernoulli distribution from logits.
    pub fn from_logits(logits: Tensor) -> Self {
        Self {
            probs: logits_to_probs(&logits, true),
            logits,
        }
    }
}

fn clamp_probs(probs: &Tensor) -> Tensor {
    let eps = eps(probs.kind()).unwrap();
    probs.clamp(eps, 1.0 - eps)
}

fn probs_to_logits(probs: &Tensor, is_binary: bool) -> Tensor {
    let ps_clamped = clamp_probs(probs);
    if is_binary {
        ps_clamped.log() - (-ps_clamped).log1p()
    } else {
        ps_clamped.log()
    }
}

fn logits_to_probs(logits: &Tensor, is_binary: bool) -> Tensor {
    if is_binary {
        logits.sigmoid()
    } else {
        logits.softmax(-1, logits.kind())
    }
}

impl Distribution for Bernoulli {
    fn entropy(&self) -> Tensor {
        self.logits.binary_cross_entropy_with_logits::<Tensor>(
            &self.probs,
            None,
            None,
            Reduction::None,
        )
    }

    fn log_prob(&self, val: &Tensor) -> Tensor {
        -self
            .logits
            .binary_cross_entropy_with_logits::<Tensor>(val, None, None, Reduction::None)
    }

    fn sample(&self, shape: &[i64]) -> Tensor {
        tch::no_grad(|| {
            Tensor::empty(shape, (Kind::Bool, self.probs.device())).bernoulli_(&self.probs)
        })
    }
}
