# tch-distr-rs

This is an attempt of mimicking the python `torch.distributions` module. 

Rust has a very good wrapper around `libtorch`: [https://github.com/LaurentMazare/tch-rs](tch-rs). 
However, the `torch.distributions` module only exists in python and therefor needs to
be implemented in rust which is a lot of work. Also, the python module is very well tested.

To ease the work for porting the distributions `tch-distr-rs`'s output is tested against
the python implementations.

If you want to help out and implement a missing distribution feel free to add it, implement
the `Distribution` trait and include some tests in `tests/against_python.rs`.

Currently, the `Distribution` trait is not stable and it will most likely change (Suggestions welcome!).


# Notes on the tests

The tests require nightly, because of the python wrapper `pyo3` and also and the
python `torch` library.

On arch linux the tests segfaulted for me when I used the `aur/libtorch` together with tch, which
seems to have been a conflict between the python `torch` and `libtorch`.

Using `community/python-pytorch` instead worked for me.

# Distributions
- [x] bernoulli
- [ ] beta
- [ ] binomial
- [ ] categorical
- [ ] cauchy
- [ ] chi2
- [ ] constraint_registry
- [ ] constraints
- [ ] continuous_bernoulli
- [ ] dirichlet
- [ ] distribution
- [ ] exp_family
- [ ] exponential
- [ ] fishersnedecor
- [ ] gamma
- [ ] geometric
- [ ] gumbel
- [ ] half_cauchy
- [ ] half_normal
- [ ] independent
- [ ] kl
- [ ] laplace
- [ ] log_normal
- [ ] logistic_normal
- [ ] lowrank_multivariate_normal
- [ ] mixture_same_family
- [ ] multinomial
- [ ] multivariate_normal
- [ ] negative_binomial
- [x] normal
- [ ] one_hot_categorical
- [ ] pareto
- [ ] poisson
- [ ] relaxed_bernoulli
- [ ] relaxed_categorical
- [ ] studentT
- [ ] transformed_distribution
- [ ] transforms
- [x] uniform
- [ ] von_mises
- [ ] weibull
