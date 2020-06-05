use crate::Distribution;
use tch::Tensor;

/// A Poisson distribution.
pub struct Poisson {
    rate: Tensor,
}

impl Poisson {
    /// Creates a new `Poisson` distribution with `rate`.
    pub fn new(rate: Tensor) -> Self {
        Self { rate }
    }

    /// Returns the rate of the distribution.
    pub fn rate(&self) -> &Tensor {
        &self.rate
    }
}

impl Distribution for Poisson {
    fn log_prob(&self, val: &Tensor) -> Tensor {
        (self.rate.log() * val) - &self.rate - (val + 1).lgamma()
    }

    fn sample(&self, shape: &[i64]) -> Tensor {
        Tensor::empty(shape, (self.rate.kind(), self.rate.device())).poisson()
    }
}
