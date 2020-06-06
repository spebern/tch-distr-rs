use crate::Distribution;
use tch::Tensor;

/// A Poisson distribution.
#[derive(Debug)]
pub struct Poisson {
    rate: Tensor,
    batch_shape: Vec<i64>,
}

impl Clone for Poisson {
    fn clone(&self) -> Self {
        Self {
            rate: self.rate.copy(),
            batch_shape: self.batch_shape.clone(),
        }
    }
}

impl Poisson {
    /// Creates a new `Poisson` distribution with `rate`.
    pub fn new(rate: Tensor) -> Self {
        let batch_shape = rate.size();
        Self { rate, batch_shape }
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
        let shape = self.extended_shape(shape);
        Tensor::empty(&shape, (self.rate.kind(), self.rate.device())).poisson()
    }

    fn batch_shape(&self) -> &[i64] {
        &self.batch_shape
    }
}
