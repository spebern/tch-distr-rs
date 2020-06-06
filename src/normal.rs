use crate::Distribution;
use std::f64::consts::PI;
use tch::Tensor;

/// A Guassian distribution.
#[derive(Debug)]
pub struct Normal {
    mean: Tensor,
    stddev: Tensor,
    batch_shape: Vec<i64>,
}

impl Clone for Normal {
    fn clone(&self) -> Self {
        Self {
            mean: self.mean.copy(),
            stddev: self.stddev.copy(),
            batch_shape: self.batch_shape.clone(),
        }
    }
}

impl Normal {
    /// Creates a new `Normal` distribution with a standard deviation `stddev` around `mean`.
    pub fn new(mean: Tensor, stddev: Tensor) -> Self {
        let batch_shape = mean.size();
        Self {
            mean,
            stddev,
            batch_shape,
        }
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
        let shape = self.extended_shape(shape);
        Tensor::normal_out2(
            &Tensor::empty(&shape, (self.mean.kind(), self.mean.device())),
            &self.mean.expand(&shape, false),
            &self.stddev.expand(&shape, false),
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

    fn batch_shape(&self) -> &[i64] {
        &self.batch_shape
    }
}
