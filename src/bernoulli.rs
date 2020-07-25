use crate::{
    utils::{infinity, logits_to_probs, probs_to_logits},
    Distribution, KullackLeiberDivergence,
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

    /// Returns the probabilities of the distribution.
    pub fn probs(&self) -> &Tensor {
        &self.probs
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

impl KullackLeiberDivergence<Self> for Bernoulli {
    fn kl_divergence(&self, other: &Self) -> Tensor {
        let t1: Tensor = self.probs() * (self.probs() / other.probs()).log();
        let t1 = t1.where1(&other.probs().f_ne(0.0).unwrap(), &infinity(t1.kind()));
        let t1 = t1.where1(&self.probs().f_ne(0.0).unwrap(), &0.0.into());

        let t2: Tensor = (&1.0.into() - self.probs())
            * ((&1.0.into() - self.probs()) / (&1.0.into() - other.probs())).log();
        let t2 = t2.where1(&other.probs().f_ne(1.0).unwrap(), &infinity(t1.kind()));
        let t2 = t2.where1(&self.probs().f_ne(1.0).unwrap(), &0.0.into());

        t1 + t2
    }
}
