use tch::Tensor;

mod bernoulli;
mod cauchy;
mod exponential;
mod gamma;
mod geometric;
mod multivariate_normal;
mod normal;
mod poisson;
mod uniform;
mod utils;
mod categorical;

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
    fn sample(&self, _shape: &[i64]) -> Tensor {
        unimplemented!()
    }

    #[doc(hidden)]
    fn batch_shape(&self) -> &[i64] {
        &[]
    }

    #[doc(hidden)]
    fn event_shape(&self) -> &[i64] {
        &[]
    }

    #[doc(hidden)]
    fn extended_shape(&self, shape: &[i64]) -> Vec<i64> {
        [shape, self.batch_shape(), self.event_shape()].concat()
    }
}

pub trait KullackLeiberDivergence<D: Distribution> {
    /// Calculates the Kullack Leiber Divergence between this distribution and another
    /// distribution.
    fn kl_divergence(&self, other: &D) -> Tensor;
}

pub use bernoulli::Bernoulli;
pub use cauchy::Cauchy;
pub use exponential::Exponential;
pub use gamma::Gamma;
pub use geometric::Geometric;
pub use multivariate_normal::MultivariateNormal;
pub use normal::Normal;
pub use poisson::Poisson;
pub use uniform::Uniform;
pub use categorical::Categorical;
