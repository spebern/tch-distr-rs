use super::Distribution;
use tch::Tensor;

/// An Exponential distribution.
pub struct Exponential {
    rate: Tensor,
}

impl Exponential {
    /// Creates a new `Exponential` distribution with `rate`.
    pub fn new(rate: Tensor) -> Self {
        Self { rate }
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
        Tensor::empty(shape, (self.rate.kind(), self.rate.device())).exponential_(1.0) / &self.rate
    }

    fn cdf(&self, val: &Tensor) -> Tensor {
        1 - (-&self.rate * val).exp()
    }

    fn entropy(&self) -> Tensor {
        1.0 - self.rate.log()
    }

    fn icdf(&self, val: &Tensor) -> Tensor {
        -(1 - val).log() / &self.rate
    }
}
