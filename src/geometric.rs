use crate::{
    utils::{logits_to_probs, probs_to_logits, tiny},
    Distribution, KullackLeiberDivergence,
};
use tch::{Reduction, Tensor};

/// A Geometric distribution.
#[derive(Debug)]
pub struct Geometric {
    probs: Tensor,
    logits: Tensor,
    batch_shape: Vec<i64>,
}

impl Clone for Geometric {
    fn clone(&self) -> Self {
        Self {
            probs: self.probs.copy(),
            logits: self.logits.copy(),
            batch_shape: self.batch_shape.clone(),
        }
    }
}

impl Geometric {
    /// Creates a Geometric distribution from probabilities.
    pub fn from_probs(probs: Tensor) -> Self {
        let batch_shape = probs.size();
        Self {
            logits: probs_to_logits(&probs, true),
            probs,
            batch_shape,
        }
    }

    /// Creates a Geometric distribution from logits.
    pub fn from_logits(logits: Tensor) -> Self {
        let batch_shape = logits.size();
        Self {
            probs: logits_to_probs(&logits, true),
            logits,
            batch_shape,
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
        let probs = self.probs.where_self(&cond.logical_not(), &0.0.into());
        val * (-probs).log1p() + self.probs.log()
    }

    fn sample(&self, shape: &[i64]) -> Tensor {
        let shape = self.extended_shape(shape);
        let tiny = tiny(self.probs.kind()).unwrap();
        tch::no_grad(|| {
            let u =
                Tensor::empty(&shape, (self.probs.kind(), self.probs.device())).uniform_(tiny, 1.0);
            (u.log() / (-&self.probs).log1p()).floor()
        })
    }

    fn entropy(&self) -> Tensor {
        self.logits.binary_cross_entropy_with_logits::<Tensor>(
            &self.probs,
            None,
            None,
            Reduction::None,
        ) / &self.probs
    }

    fn batch_shape(&self) -> &[i64] {
        &self.batch_shape
    }
}

impl KullackLeiberDivergence<Self> for Geometric {
    fn kl_divergence(&self, other: &Self) -> Tensor {
        -self.entropy() - (-other.probs()).log1p() / self.probs() - other.logits()
    }
}
