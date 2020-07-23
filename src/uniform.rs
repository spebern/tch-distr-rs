use crate::{utils::infinity, Distribution, KullackLeiberDivergence};
use tch::Tensor;

/// A Uniform distribution.
#[derive(Debug)]
pub struct Uniform {
    low: Tensor,
    high: Tensor,
    batch_shape: Vec<i64>,
}

impl Clone for Uniform {
    fn clone(&self) -> Self {
        Self {
            low: self.low.copy(),
            high: self.high.copy(),
            batch_shape: self.batch_shape.clone(),
        }
    }
}

impl Uniform {
    // Generates uniformly distributed random samples from the half-open interval [low, high).
    pub fn new(low: Tensor, high: Tensor) -> Self {
        let batch_shape = low.size();
        Self {
            low,
            high,
            batch_shape,
        }
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
        let shape = self.extended_shape(shape);
        let rand = Tensor::rand(&shape, (self.low.kind(), self.high.device()));
        &self.low + &rand * (&self.high - &self.low)
    }

    fn batch_shape(&self) -> &[i64] {
        &self.batch_shape
    }
}

impl KullackLeiberDivergence<Self> for Uniform {
    fn kl_divergence(&self, other: &Self) -> Tensor {
        let result = ((other.high() - other.low()) / (self.high() - self.low())).log();
        result.where1(
            &other
                .low()
                .le1(self.low())
                .logical_and(&other.high().ge1(self.high())),
            &infinity(result.kind()),
        )
    }
}
