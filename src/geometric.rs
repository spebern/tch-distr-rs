use super::Distribution;
use crate::utils::{logits_to_probs, probs_to_logits, tiny};
use tch::{Reduction, Tensor};

/// A Geometric distribution.
pub struct Geometric {
    probs: Tensor,
    logits: Tensor,
}

impl Geometric {
    /// Creates a Geometric distribution from probabilities.
    pub fn from_probs(probs: Tensor) -> Self {
        Self {
            logits: probs_to_logits(&probs, true),
            probs,
        }
    }

    /// Creates a Geometric distribution from logits.
    pub fn from_logits(logits: Tensor) -> Self {
        Self {
            probs: logits_to_probs(&logits, true),
            logits,
        }
    }

    /// Returns the logits of the distribution.
    pub fn logits(&self) -> &Tensor {
        &self.logits
    }

    /// Returns the probabilities of the distribution.
    pub fn probs(&self) -> &Tensor {
        &self.probs
    }
}

impl Distribution for Geometric {
    fn log_prob(&self, val: &Tensor) -> Tensor {
        let cond = &self
            .probs
            .f_eq(1)
            .unwrap()
            .logical_and(&val.f_eq(0).unwrap());
        let probs = self.probs.where1(&cond.logical_not(), &0.0.into());
        val * (-probs).log1p() + self.probs.log()
    }

    fn sample(&self, shape: &[i64]) -> Tensor {
        let tiny = tiny(self.probs.kind()).unwrap();
        let u = Tensor::empty(shape, (self.probs.kind(), self.probs.device())).uniform_(tiny, 1.0);
        (u.log() / (-&self.probs).log1p()).floor()
    }

    fn entropy(&self) -> Tensor {
        self.logits.binary_cross_entropy_with_logits::<Tensor>(
            &self.probs,
            None,
            None,
            Reduction::None,
        ) / &self.probs
    }
}
