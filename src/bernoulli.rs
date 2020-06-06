use crate::{
    utils::{logits_to_probs, probs_to_logits},
    Distribution,
};
use tch::{Kind, Reduction, Tensor};

/// A Bernoulli distribution.
#[derive(Debug)]
pub struct Bernoulli {
    probs: Tensor,
    logits: Tensor,
}

impl Clone for Bernoulli {
    fn clone(&self) -> Self {
        Self {
            probs: self.probs.copy(),
            logits: self.logits.copy(),
        }
    }
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
