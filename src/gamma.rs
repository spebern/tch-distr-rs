use super::Distribution;
use crate::utils::tiny;
use tch::Tensor;

/// A Gamma distribution.
pub struct Gamma {
    concentration: Tensor,
    rate: Tensor,
}

impl Gamma {
    // Creates a gamma distribution with`concentration` and `rate`.
    pub fn new(concentration: Tensor, rate: Tensor) -> Self {
        Self {
            concentration,
            rate,
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
        let samples = Tensor::empty(shape, (self.rate.kind(), self.rate.device()))
            .internal_standard_gamma()
            / &self.rate;
        let tiny = tiny(samples.kind()).unwrap();
        samples.detach().clamp_min(tiny)
    }

    fn entropy(&self) -> Tensor {
        &self.concentration - self.rate.log()
            + self.concentration.lgamma()
            + (1.0 - &self.concentration) * self.concentration.digamma()
    }
}
