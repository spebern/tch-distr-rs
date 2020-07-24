use crate::{Distribution, KullackLeiberDivergence};
use tch::Tensor;

/// A Gamma distribution.
#[derive(Debug)]
pub struct Gamma {
    concentration: Tensor,
    rate: Tensor,
    batch_shape: Vec<i64>,
}

impl Clone for Gamma {
    fn clone(&self) -> Self {
        Self {
            concentration: self.concentration.copy(),
            rate: self.rate.copy(),
            batch_shape: self.batch_shape.clone(),
        }
    }
}

impl Gamma {
    // Creates a gamma distribution with`concentration` and `rate`.
    pub fn new(concentration: Tensor, rate: Tensor) -> Self {
        let batch_shape = concentration.size();
        Self {
            concentration,
            rate,
            batch_shape,
        }
    }

    /// Returns shape parameter of the distribution (often referred to as alpha).
    pub fn concentration(&self) -> &Tensor {
        &self.concentration
    }

    /// Returns rate = 1 / scale of the distribution (often referred to as beta).
    pub fn rate(&self) -> &Tensor {
        &self.rate
    }
}

impl Distribution for Gamma {
    fn log_prob(&self, val: &Tensor) -> Tensor {
        &self.concentration * self.rate.log() + (&self.concentration - 1) * val.log()
            - &self.rate * val
            - self.concentration.lgamma()
    }

    fn sample(&self, shape: &[i64]) -> Tensor {
        let _shape = self.extended_shape(shape);
        todo!("seems like some bindings are missing")
    }

    fn entropy(&self) -> Tensor {
        &self.concentration - self.rate.log()
            + self.concentration.lgamma()
            + (1.0 - &self.concentration) * self.concentration.digamma()
    }

    fn batch_shape(&self) -> &[i64] {
        &self.batch_shape
    }
}

impl KullackLeiberDivergence<Self> for Gamma {
    fn kl_divergence(&self, other: &Self) -> Tensor {
        let t1 = other.concentration() * (self.rate() / other.rate()).log();
        let t2 = other.concentration().lgamma() - self.concentration().lgamma();
        let t3 = (self.concentration() - other.concentration()) * self.concentration().digamma();
        let t4 = (other.rate() - self.rate()) * (self.concentration() / self.rate());
        t1 + t2 + t3 + t4
    }
}
