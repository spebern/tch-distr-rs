use tch::Tensor;

mod normal;

pub trait Distribution {
    /// Returns the cumulative density/mass function evaluated at `val`.
    fn cdf(&self, val: &Tensor) -> Tensor;

    /// Returns entropy of distribution, batched over batch_shape.
    fn entropy(&self) -> Tensor;

    /// Returns the inverse cumulative density/mass function evaluated at `val`.
    fn icdf(&self, val: &Tensor) -> Tensor;

    /// Returns the inverse cumulative density/mass function evaluated at `val`.
    fn log_prob(&self, val: &Tensor) -> Tensor;

    /// Generates a sample_shape shaped sample or sample_shape shaped batch of
    /// samples if the distribution parameters are batched.
    fn sample(&self, shape: &[i64]) -> Tensor;
}

pub use normal::Normal;
