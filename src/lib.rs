use tch::Tensor;

mod bernoulli;
mod normal;
mod poisson;
mod uniform;
mod utils;

pub trait Distribution {
    /// Returns the cumulative density/mass function evaluated at `val`.
    fn cdf(&self, _val: &Tensor) -> Tensor {
        unimplemented!()
    }

    /// Returns entropy of distribution, batched over batch_shape.
    fn entropy(&self) -> Tensor {
        unimplemented!()
    }

    /// Returns the inverse cumulative density/mass function evaluated at `val`.
    fn icdf(&self, _val: &Tensor) -> Tensor {
        unimplemented!()
    }

    /// Returns the inverse cumulative density/mass function evaluated at `val`.
    fn log_prob(&self, val: &Tensor) -> Tensor;

    /// Generates a sample_shape shaped sample or sample_shape shaped batch of
    /// samples if the distribution parameters are batched.
    fn sample(&self, shape: &[i64]) -> Tensor;
}

pub use bernoulli::Bernoulli;
pub use normal::Normal;
pub use poisson::Poisson;
pub use uniform::Uniform;
