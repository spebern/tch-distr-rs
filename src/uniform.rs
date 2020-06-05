use crate::Distribution;
use tch::Tensor;

/// A Uniform distribution.
pub struct Uniform {
    low: Tensor,
    high: Tensor,
}

impl Uniform {
    // Generates uniformly distributed random samples from the half-open interval [low, high).
    pub fn new(low: Tensor, high: Tensor) -> Self {
        Self { low, high }
    }

    /// Returns the lower range (inclusive).
    pub fn low(&self) -> &Tensor {
        &self.low
    }

    /// Returns the upper range (exclusive).
    pub fn high(&self) -> &Tensor {
        &self.high
    }
}

impl Distribution for Uniform {
    fn cdf(&self, val: &Tensor) -> Tensor {
        ((val - &self.low) / (&self.high - &self.low)).clamp(0.0, 1.0)
    }

    fn entropy(&self) -> Tensor {
        (&self.high - &self.low).log()
    }

    fn icdf(&self, val: &Tensor) -> Tensor {
        val * (&self.high - &self.low) + &self.low
    }

    fn log_prob(&self, val: &Tensor) -> Tensor {
        let lb = self.low.le1(val).type_as(&self.low);
        let ub = self.high.gt1(val).type_as(&self.low);
        (&lb * &ub).log() - (&self.high - &self.low).log()
    }

    fn sample(&self, shape: &[i64]) -> Tensor {
        let rand = Tensor::randn(shape, (self.low.kind(), self.high.device()));
        &self.low + &rand * (&self.high - &self.low)
    }
}
