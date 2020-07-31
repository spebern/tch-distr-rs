use crate::{Distribution, KullackLeiberDivergence};
use tch::Tensor;

/// An Exponential distribution.
#[derive(Debug)]
pub struct Exponential {
    rate: Tensor,
    batch_shape: Vec<i64>,
}

impl Clone for Exponential {
    fn clone(&self) -> Self {
        Self {
            rate: self.rate.copy(),
            batch_shape: self.batch_shape.clone(),
        }
    }
}

impl Exponential {
    /// Creates a new `Exponential` distribution with `rate`.
    pub fn new(rate: Tensor) -> Self {
        let batch_shape = rate.size();
        Self { rate, batch_shape }
    }

    /// Returns the rate of the distribution.
    pub fn rate(&self) -> &Tensor {
        &self.rate
    }
}

impl Distribution for Exponential {
    fn log_prob(&self, val: &Tensor) -> Tensor {
        self.rate.log() - &self.rate * val
    }

    fn sample(&self, shape: &[i64]) -> Tensor {
        let shape = self.extended_shape(shape);
        Tensor::empty(&shape, (self.rate.kind(), self.rate.device())).exponential_(1.0) / &self.rate
    }

    fn cdf(&self, val: &Tensor) -> Tensor {
        1.0f64 - (-&self.rate * val).exp()
    }

    fn entropy(&self) -> Tensor {
        1.0f64 - self.rate.log()
    }

    fn icdf(&self, val: &Tensor) -> Tensor {
        -(1.0f64 - val).log() / &self.rate
    }

    fn batch_shape(&self) -> &[i64] {
        &self.batch_shape
    }
}

impl KullackLeiberDivergence<Self> for Exponential {
    fn kl_divergence(&self, other: &Self) -> Tensor {
        let rate_ratio = other.rate() / self.rate();
        let t1 = -rate_ratio.log();
        t1 + rate_ratio - &1.0.into()
    }
}
