use crate::Distribution;
use std::f64::consts::PI;
use tch::Tensor;

/// A Cauchy distribution.
pub struct Cauchy {
    median: Tensor,
    scale: Tensor,
    batch_shape: Vec<i64>,
}

impl Cauchy {
    /// Creates a new `Cauchy` distribution `median` and `scale` as half width of the maximum.
    pub fn new(median: Tensor, scale: Tensor) -> Self {
        let batch_shape = median.size();
        Self {
            median,
            scale,
            batch_shape,
        }
    }

    /// Returns the median of the distribution.
    pub fn median(&self) -> &Tensor {
        &self.median
    }

    /// Returns the scale of the distribution.
    pub fn scale(&self) -> &Tensor {
        &self.scale
    }
}

impl Distribution for Cauchy {
    fn log_prob(&self, val: &Tensor) -> Tensor {
        -PI.ln() - self.scale.log() - (1 + ((val - &self.median) / &self.scale).pow(2)).log()
    }

    fn sample(&self, shape: &[i64]) -> Tensor {
        let shape = self.extended_shape(shape);
        let eps =
            Tensor::empty(&shape, (self.median.kind(), self.median.device())).cauchy_(0.0, 1.0);
        &self.median + eps * &self.scale
    }

    fn cdf(&self, val: &Tensor) -> Tensor {
        ((val - &self.median) / &self.scale).atan() / PI + 0.5
    }

    fn entropy(&self) -> Tensor {
        (4.0 * PI).ln() + self.scale.log()
    }

    fn icdf(&self, val: &Tensor) -> Tensor {
        (PI * (val - 0.5)).tan() * &self.scale + &self.median
    }

    fn batch_shape(&self) -> &[i64] {
        &self.batch_shape
    }
}
