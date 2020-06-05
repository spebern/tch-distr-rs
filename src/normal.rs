use crate::Distribution;
use std::f64::consts::PI;
use tch::Tensor;

/// A Guassian distribution.
pub struct Normal {
    mean: Tensor,
    stddev: Tensor,
}

impl Normal {
    /// Creates a new `Normal` distribution with a standard deviation `stddev` around `mean`.
    pub fn new(mean: Tensor, stddev: Tensor) -> Self {
        Self { mean, stddev }
    }

    /// Returns the mean of the distribution.
    pub fn mean(&self) -> &Tensor {
        &self.mean
    }

    /// Returns the standard deviation of the distribution.
    pub fn stddev(&self) -> &Tensor {
        &self.stddev
    }
}

impl Distribution for Normal {
    fn entropy(&self) -> Tensor {
        0.5 + 0.5 * (2.0 * PI).ln() + self.stddev.log()
    }

    fn sample(&self, shape: &[i64]) -> Tensor {
        Tensor::normal_out2(
            &Tensor::empty(shape, (self.mean.kind(), self.mean.device())),
            &self.mean,
            &self.stddev,
        )
    }

    fn log_prob(&self, val: &Tensor) -> Tensor {
        let var = self.stddev.pow(2);
        -(val - &self.mean).pow(2) / (2.0 * var) - self.stddev().log() - (2.0 * PI).sqrt().ln()
    }

    fn cdf(&self, val: &Tensor) -> Tensor {
        0.5 * (1.0 + ((val - &self.mean) * self.stddev.reciprocal() / 2.0f64.sqrt()).erf())
    }

    fn icdf(&self, val: &Tensor) -> Tensor {
        &self.mean + &self.stddev * (2.0 * val - 1.0).erfinv() * 2.0f64.sqrt()
    }
}
