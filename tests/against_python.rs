use float_cmp::assert_approx_eq;
use ndarray::{array, ArrayD};
use numpy::{PyArrayDyn, ToPyArray};
use pyo3::{prelude::*, types::PyTuple};
use serial_test::serial;
use std::convert::{TryFrom, TryInto};
use tch::Tensor;
use tch_distr::{
    Bernoulli, Categorical, Cauchy, Distribution, Exponential, Gamma, Geometric,
    KullackLeiberDivergence, MultivariateNormal, Normal, Poisson, Uniform,
};

const SEED: i64 = 42;

struct PyEnv<'py> {
    py: Python<'py>,
    torch: &'py PyModule,
    distributions: &'py PyModule,
    kl: &'py PyModule,
}

impl<'py> PyEnv<'py> {
    fn new(gil: &'py GILGuard) -> Self {
        let py = gil.python();

        let torch = PyModule::import(py, "torch").unwrap();
        let distributions = PyModule::import(py, "torch.distributions").unwrap();
        let kl = PyModule::import(py, "torch.distributions.kl").unwrap();

        Self {
            py,
            torch,
            distributions,
            kl,
        }
    }
}

struct TestCases {
    entropy: bool,
    log_prob: Option<Vec<Tensor>>,
    cdf: Option<Vec<Tensor>>,
    icdf: Option<Vec<Tensor>>,
    sample: Option<Vec<Vec<i64>>>,
}

impl Default for TestCases {
    fn default() -> Self {
        Self {
            entropy: true,
            log_prob: Some(vec![
                1.0.into(),
                2.0.into(),
                Tensor::of_slice(&[1.0, 1.0]),
                Tensor::of_slice(&[2.0, 2.0]),
            ]),
            cdf: Some(vec![
                1.0.into(),
                2.0.into(),
                Tensor::of_slice(&[1.0, 1.0]),
                Tensor::of_slice(&[2.0, 2.0]),
            ]),
            icdf: Some(vec![
                0.5.into(),
                0.7.into(),
                Tensor::of_slice(&[0.3, 0.4]),
                Tensor::of_slice(&[0.2, 0.7]),
            ]),
            sample: None,
        }
    }
}

fn tensor_to_py_obj<'py>(py_env: &'py PyEnv, t: &Tensor) -> &'py PyAny {
    let array: ndarray::ArrayD<f64> = t.try_into().unwrap();
    py_env
        .torch
        .getattr("from_numpy")
        .expect("call from_numpy failed")
        .call1((array.to_pyarray(py_env.py),))
        .unwrap()
}

fn assert_tensor_eq(py: Python<'_>, t: &Tensor, py_t: &PyAny) {
    // transfer type to f64(Double)
    let python_side_array: &PyArrayDyn<f64> = if t.kind() == tch::Kind::Int64 {
        let tmp_pyarray: &PyArrayDyn<i64> = py_t
            .call_method0("contiguous")
            .unwrap()
            .call_method0("numpy")
            .unwrap()
            .extract()
            .unwrap();
        tmp_pyarray
            .call_method1("astype", ("float64",))
            .unwrap()
            .extract()
            .unwrap()
    } else {
        py_t.call_method0("contiguous")
            .unwrap()
            .call_method0("numpy")
            .unwrap()
            .extract()
            .unwrap()
    };

    let rust_side_array: ArrayD<f64> = t.try_into().unwrap();
    let rust_side_array: &PyArrayDyn<f64> = rust_side_array.to_pyarray(py);

    let python_side_array = python_side_array.as_cell_slice().unwrap().to_vec();
    let rust_side_array = rust_side_array.as_cell_slice().unwrap().to_vec();
    assert_eq!(python_side_array.len(), rust_side_array.len());
    for (a, b) in python_side_array.iter().zip(rust_side_array.iter()) {
        let a = a.get();
        let b = b.get();
        assert_approx_eq!(f64, a, b, ulps = 2);
    }
}

fn test_entropy<D: Distribution>(py_env: &PyEnv, dist_rs: &D, dist_py: &PyAny) {
    let entropy_py = dist_py.call_method0("entropy").unwrap();
    let entropy_rs = dist_rs.entropy();
    assert_tensor_eq(py_env.py, &entropy_rs, entropy_py);
}

fn test_log_prob<D: Distribution>(py_env: &PyEnv, dist_rs: &D, dist_py: &PyAny, args: &[Tensor]) {
    for args in args.iter() {
        let args_py = PyTuple::new(py_env.py, vec![tensor_to_py_obj(py_env, args)]);
        let log_prob_py = dist_py.call_method1("log_prob", args_py).unwrap();
        let log_prob_rs = dist_rs.log_prob(args);
        assert_tensor_eq(py_env.py, &log_prob_rs, log_prob_py);
    }
}

fn test_cdf<D: Distribution>(py_env: &PyEnv, dist_rs: &D, dist_py: &PyAny, args: &[Tensor]) {
    for args in args.iter() {
        let args_py = PyTuple::new(py_env.py, vec![tensor_to_py_obj(py_env, args)]);
        let log_prob_py = dist_py.call_method1("cdf", args_py).unwrap();
        let log_prob_rs = dist_rs.cdf(args);
        assert_tensor_eq(py_env.py, &log_prob_rs, log_prob_py);
    }
}

fn test_icdf<D: Distribution>(py_env: &PyEnv, dist_rs: &D, dist_py: &PyAny, args: &[Tensor]) {
    for args in args.iter() {
        let args_py = PyTuple::new(py_env.py, vec![tensor_to_py_obj(py_env, args)]);
        let log_prob_py = dist_py.call_method1("icdf", args_py).unwrap();
        let log_prob_rs = dist_rs.icdf(args);
        assert_tensor_eq(py_env.py, &log_prob_rs, log_prob_py);
    }
}

fn test_sample<D: Distribution>(py_env: &PyEnv, dist_rs: &D, dist_py: &PyAny, args: &[Vec<i64>]) {
    for args in args.iter() {
        // We need to ensure that we always start with the same seed.
        tch::manual_seed(SEED);
        let samples_py = dist_py
            .call_method1("sample", (args.to_object(py_env.py),))
            .unwrap();
        tch::manual_seed(SEED);
        let samples_rs = dist_rs.sample(args);
        assert_tensor_eq(py_env.py, &samples_rs, samples_py);
    }
}

fn test_rsample_of_normal_distribution(
    py_env: &PyEnv,
    dist_rs: &Normal,
    dist_py: &PyAny,
    args: &[Vec<i64>],
) {
    for args in args.iter() {
        // We need to ensure that we always start with the same seed.
        tch::manual_seed(SEED);
        let samples_py = dist_py
            .call_method1("rsample", (args.to_object(py_env.py),))
            .unwrap();
        tch::manual_seed(SEED);
        let samples_rs = dist_rs.rsample(args);
        assert_tensor_eq(py_env.py, &samples_rs, samples_py);
    }
}

fn test_rsample_of_multi_var_normal_distribution(
    py_env: &PyEnv,
    dist_rs: &MultivariateNormal,
    dist_py: &PyAny,
    args: &[Vec<i64>],
) {
    for args in args.iter() {
        // We need to ensure that we always start with the same seed.
        tch::manual_seed(SEED);
        let samples_py = dist_py
            .call_method1("rsample", (args.to_object(py_env.py),))
            .unwrap();
        tch::manual_seed(SEED);
        let samples_rs = dist_rs.rsample(args);
        assert_tensor_eq(py_env.py, &samples_rs, samples_py);
    }
}

fn test_kl_divergence<P, Q>(
    py_env: &PyEnv,
    dist_p_rs: &P,
    dist_q_rs: &Q,
    dist_p_py: &PyAny,
    dist_q_py: &PyAny,
) where
    P: Distribution + KullackLeiberDivergence<Q>,
    Q: Distribution,
{
    let args_py = PyTuple::new(py_env.py, vec![dist_p_py, dist_q_py]);
    let kl_divergence_py = py_env.kl.call_method1("kl_divergence", args_py).unwrap();
    let kl_divergence_rs = dist_p_rs.kl_divergence(dist_q_rs);
    assert_tensor_eq(py_env.py, &kl_divergence_rs, kl_divergence_py);
}

fn run_test_cases<D>(py_env: &PyEnv, dist_rs: D, dist_py: &PyAny, test_cases: &TestCases)
where
    D: Distribution,
{
    if test_cases.entropy {
        test_entropy(py_env, &dist_rs, dist_py);
    }
    if let Some(log_prob) = test_cases.log_prob.as_ref() {
        test_log_prob(py_env, &dist_rs, dist_py, log_prob);
    }
    if let Some(cdf) = test_cases.cdf.as_ref() {
        test_cdf(py_env, &dist_rs, dist_py, cdf);
    }
    if let Some(icdf) = test_cases.icdf.as_ref() {
        test_icdf(py_env, &dist_rs, dist_py, icdf);
    }
    if let Some(sample) = test_cases.sample.as_ref() {
        test_sample(py_env, &dist_rs, dist_py, sample);
    }
}

#[test]
#[serial]
fn normal() {
    let gil = Python::acquire_gil();
    let py_env = PyEnv::new(&gil);

    let args: Vec<(Tensor, Tensor)> = vec![
        (1.0.into(), 2.0.into()),
        (2.0.into(), 4.0.into()),
        (Tensor::of_slice(&[1.0, 1.0]), Tensor::of_slice(&[2.0, 2.0])),
        (
            Tensor::try_from(array![[1.0], [1.0]]).unwrap(),
            Tensor::try_from(array![[2.0], [2.0]]).unwrap(),
        ),
        (
            Tensor::try_from(array![[1.0, 0.5], [1.0, 2.0]]).unwrap(),
            Tensor::try_from(array![[2.0, 1.0], [2.0, 1.0]]).unwrap(),
        ),
    ];

    let mut test_cases = TestCases::default();
    test_cases.sample = Some(vec![vec![1], vec![1, 2]]);

    for (mean, std) in args.into_iter() {
        let dist_py = py_env
            .distributions
            .getattr("Normal")
            .expect("call Normal failed")
            .call1((
                tensor_to_py_obj(&py_env, &mean),
                tensor_to_py_obj(&py_env, &std),
            ))
            .unwrap();
        let dist_rs = Normal::new(mean, std);

        // The test of resampling is not in function `run_test_cases`,
        // because `rsample` is not a method of trait `Distribution`
        if let Some(sample) = &test_cases.sample {
            test_rsample_of_normal_distribution(&py_env, &dist_rs, dist_py, sample);
        }

        run_test_cases(&py_env, dist_rs, dist_py, &test_cases);
    }

    let p_q_mean_std: Vec<((Tensor, Tensor), (Tensor, Tensor))> =
        vec![((1.0.into(), 2.0.into()), (2.0.into(), 3.0.into()))];

    for ((p_mean, p_std), (q_mean, q_std)) in p_q_mean_std {
        let dist_p_py = py_env
            .distributions
            .getattr("Normal")
            .expect("call Normal failed")
            .call1((
                tensor_to_py_obj(&py_env, &p_mean),
                tensor_to_py_obj(&py_env, &p_std),
            ))
            .unwrap();
        let dist_p_rs = Normal::new(p_mean, p_std);

        let dist_q_py = py_env
            .distributions
            .getattr("Normal")
            .expect("call Normal failed")
            .call1((
                tensor_to_py_obj(&py_env, &q_mean),
                tensor_to_py_obj(&py_env, &q_std),
            ))
            .unwrap();
        let dist_q_rs = Normal::new(q_mean, q_std);

        test_kl_divergence(&py_env, &dist_p_rs, &dist_q_rs, dist_p_py, dist_q_py);
    }
}

#[test]
#[serial]
fn uniform() {
    let gil = Python::acquire_gil();
    let py_env = PyEnv::new(&gil);

    let args: Vec<(Tensor, Tensor)> = vec![
        (1.0.into(), 2.0.into()),
        ((-1.0).into(), 4.0.into()),
        (Tensor::of_slice(&[1.0, 1.0]), Tensor::of_slice(&[2.0, 2.0])),
    ];

    let mut test_cases = TestCases::default();

    // NOTE: because all samples of `log_prob` would be testes for each of `args`,
    // so within distribution uniform, the VALID parameter parsed to `log_prob` should be
    // in the range of intersection of all `args`
    test_cases.log_prob = Some(vec![
        1.2.into(),
        2.0.into(),
        Tensor::of_slice(&[1.2, 1.4]),
        Tensor::of_slice(&[2.0, 1.5]),
    ]);
    test_cases.sample = Some(vec![vec![1], vec![2], vec![1, 4], vec![2, 3]]);

    for (low, high) in args.into_iter() {
        let dist_py = py_env
            .distributions
            .getattr("Uniform")
            .expect("call Uniform failed")
            .call1((
                tensor_to_py_obj(&py_env, &low),
                tensor_to_py_obj(&py_env, &high),
            ))
            .unwrap();
        let dist_rs = Uniform::new(low, high);
        run_test_cases(&py_env, dist_rs, dist_py, &test_cases);
    }

    let p_q_mean_std: Vec<((Tensor, Tensor), (Tensor, Tensor))> = vec![
        ((0.0.into(), 3.0.into()), (1.0.into(), 3.0.into())),
        ((1.0.into(), 2.0.into()), (0.0.into(), 3.0.into())),
    ];

    for ((p_low, p_high), (q_low, q_high)) in p_q_mean_std {
        let dist_p_py = py_env
            .distributions
            .getattr("Uniform")
            .expect("call Uniform failed")
            .call1((
                tensor_to_py_obj(&py_env, &p_low),
                tensor_to_py_obj(&py_env, &p_high),
            ))
            .unwrap();
        let dist_p_rs = Uniform::new(p_low, p_high);

        let dist_q_py = py_env
            .distributions
            .getattr("Uniform")
            .expect("call Uniform failed")
            .call1((
                tensor_to_py_obj(&py_env, &q_low),
                tensor_to_py_obj(&py_env, &q_high),
            ))
            .unwrap();
        let dist_q_rs = Uniform::new(q_low, q_high);

        test_kl_divergence(&py_env, &dist_p_rs, &dist_q_rs, dist_p_py, dist_q_py);
    }
}

#[test]
#[serial]
fn bernoulli() {
    let gil = Python::acquire_gil();
    let py_env = PyEnv::new(&gil);

    let probs: Vec<Tensor> = vec![0.1337.into(), 0.6667.into()];

    let mut test_cases = TestCases::default();
    test_cases.icdf = None;
    test_cases.cdf = None;
    // NOTE: for distribution bernoulli, only 1 or 0 is valid for method `log_prob`
    test_cases.log_prob = Some(vec![
        0.0.into(),
        1.0.into(),
        Tensor::of_slice(&[1.0, 0.0]),
        Tensor::of_slice(&[0.0, 1.0]),
    ]);
    test_cases.sample = Some(vec![vec![1], vec![1, 2]]);

    for probs in probs.into_iter() {
        let dist_py = py_env
            .distributions
            .getattr("Bernoulli")
            .expect("call Bernoulli failed")
            .call1((tensor_to_py_obj(&py_env, &probs),))
            .unwrap();
        let dist_rs = Bernoulli::from_probs(probs);
        run_test_cases(&py_env, dist_rs, dist_py, &test_cases);
    }

    let logits: Vec<Tensor> = vec![0.1337.into(), 0.6667.into()];

    let mut test_cases = TestCases::default();
    test_cases.icdf = None;
    test_cases.cdf = None;
    // NOTE: for distribution bernoulli, only 1 or 0 is valid for method `log_prob`
    test_cases.log_prob = Some(vec![
        0.0.into(),
        1.0.into(),
        Tensor::of_slice(&[1.0, 0.0]),
        Tensor::of_slice(&[0.0, 1.0]),
    ]);
    test_cases.sample = Some(vec![vec![1], vec![1, 2]]);

    for logits in logits.into_iter() {
        let dist_py = py_env
            .distributions
            .getattr("Bernoulli")
            .expect("call Bernoulli failed")
            .call1((
                pyo3::Python::None(py_env.py),
                tensor_to_py_obj(&py_env, &logits).to_object(py_env.py),
            ))
            .unwrap();
        let dist_rs = Bernoulli::from_logits(logits);
        run_test_cases(&py_env, dist_rs, dist_py, &test_cases);
    }

    let p_q_probs: Vec<(Tensor, Tensor)> =
        vec![(0.3.into(), 0.65.into()), (0.11237.into(), 0.898.into())];

    for (p_probs, q_probs) in p_q_probs {
        let dist_p_py = py_env
            .distributions
            .getattr("Bernoulli")
            .expect("call Bernoulli failed")
            .call1((tensor_to_py_obj(&py_env, &p_probs),))
            .unwrap();
        let dist_p_rs = Bernoulli::from_probs(p_probs);

        let dist_q_py = py_env
            .distributions
            .getattr("Bernoulli")
            .expect("call Bernoulli failed")
            .call1((tensor_to_py_obj(&py_env, &q_probs),))
            .unwrap();
        let dist_q_rs = Bernoulli::from_probs(q_probs);

        test_kl_divergence(&py_env, &dist_p_rs, &dist_q_rs, dist_p_py, dist_q_py);
    }
}

#[test]
#[serial]
fn poisson() {
    let gil = Python::acquire_gil();
    let py_env = PyEnv::new(&gil);

    let rates: Vec<Tensor> = vec![
        0.1337.into(),
        0.6667.into(),
        Tensor::of_slice(&[0.156, 0.33]),
    ];

    let mut test_cases = TestCases::default();
    test_cases.cdf = None;
    test_cases.icdf = None;
    test_cases.entropy = false;
    test_cases.sample = Some(vec![vec![1], vec![1, 2]]);

    for rate in rates.into_iter() {
        let dist_py = py_env
            .distributions
            .getattr("Poisson")
            .expect("call Poisson failed")
            .call1((tensor_to_py_obj(&py_env, &rate),))
            .unwrap();
        let dist_rs = Poisson::new(rate);
        run_test_cases(&py_env, dist_rs, dist_py, &test_cases);
    }

    let p_q_rate: Vec<(Tensor, Tensor)> = vec![(1.0.into(), 2.0.into()), (3.0.into(), 4.0.into())];

    for (p_rate, q_rate) in p_q_rate {
        let dist_p_py = py_env
            .distributions
            .getattr("Poisson")
            .expect("call Poisson failed")
            .call1((tensor_to_py_obj(&py_env, &p_rate),))
            .unwrap();
        let dist_p_rs = Poisson::new(p_rate);

        let dist_q_py = py_env
            .distributions
            .getattr("Poisson")
            .expect("call Poisson failed")
            .call1((tensor_to_py_obj(&py_env, &q_rate),))
            .unwrap();
        let dist_q_rs = Poisson::new(q_rate);

        test_kl_divergence(&py_env, &dist_p_rs, &dist_q_rs, dist_p_py, dist_q_py);
    }
}

#[test]
#[serial]
fn exponential() {
    let gil = Python::acquire_gil();
    let py_env = PyEnv::new(&gil);

    let rates: Vec<Tensor> = vec![
        0.1337.into(),
        0.6667.into(),
        Tensor::of_slice(&[0.156, 0.33]),
    ];

    let mut test_cases = TestCases::default();
    test_cases.sample = Some(vec![vec![1], vec![1, 2]]);

    for rate in rates.into_iter() {
        let dist_py = py_env
            .distributions
            .getattr("Exponential")
            .expect("call Exponential failed")
            .call1((tensor_to_py_obj(&py_env, &rate),))
            .unwrap();
        let dist_rs = Exponential::new(rate);
        run_test_cases(&py_env, dist_rs, dist_py, &test_cases);
    }

    let p_q_rate: Vec<(Tensor, Tensor)> = vec![(0.3.into(), 0.7.into()), (0.6.into(), 0.5.into())];

    for (p_rate, q_rate) in p_q_rate {
        let dist_p_py = py_env
            .distributions
            .getattr("Exponential")
            .expect("call Exponential failed")
            .call1((tensor_to_py_obj(&py_env, &p_rate),))
            .unwrap();
        let dist_p_rs = Exponential::new(p_rate);

        let dist_q_py = py_env
            .distributions
            .getattr("Exponential")
            .expect("call Exponential failed")
            .call1((tensor_to_py_obj(&py_env, &q_rate),))
            .unwrap();
        let dist_q_rs = Exponential::new(q_rate);

        test_kl_divergence(&py_env, &dist_p_rs, &dist_q_rs, dist_p_py, dist_q_py);
    }
}

#[test]
#[serial]
fn cauchy() {
    let gil = Python::acquire_gil();
    let py_env = PyEnv::new(&gil);

    let args: Vec<(Tensor, Tensor)> = vec![
        (1.0.into(), 2.0.into()),
        (2.0.into(), 4.0.into()),
        (Tensor::of_slice(&[1.0, 1.0]), Tensor::of_slice(&[2.0, 2.0])),
    ];

    let mut test_cases = TestCases::default();
    test_cases.sample = Some(vec![vec![1], vec![1, 2]]);

    for (median, scale) in args.into_iter() {
        let dist_py = py_env
            .distributions
            .getattr("Cauchy")
            .expect("call Cauchy failed")
            .call1((
                tensor_to_py_obj(&py_env, &median),
                tensor_to_py_obj(&py_env, &scale),
            ))
            .unwrap();
        let dist_rs = Cauchy::new(median, scale);
        run_test_cases(&py_env, dist_rs, dist_py, &test_cases);
    }
}

#[test]
#[serial]
fn gamma() {
    let gil = Python::acquire_gil();
    let py_env = PyEnv::new(&gil);

    let args: Vec<(Tensor, Tensor)> = vec![
        (1.0.into(), 2.0.into()),
        (2.0.into(), 4.0.into()),
        (Tensor::of_slice(&[1.0, 1.0]), Tensor::of_slice(&[2.0, 2.0])),
    ];

    let mut test_cases = TestCases::default();
    test_cases.cdf = None;
    test_cases.icdf = None;

    for (concentration, rate) in args.into_iter() {
        let dist_py = py_env
            .distributions
            .getattr("Gamma")
            .expect("call Gamma failed")
            .call1((
                tensor_to_py_obj(&py_env, &concentration),
                tensor_to_py_obj(&py_env, &rate),
            ))
            .unwrap();
        let dist_rs = Gamma::new(concentration, rate);
        run_test_cases(&py_env, dist_rs, dist_py, &test_cases);
    }

    let p_q_concentration_rate: Vec<((Tensor, Tensor), (Tensor, Tensor))> =
        vec![((0.3.into(), 0.7.into()), (0.6.into(), 0.5.into()))];

    for ((p_concentration, p_rate), (q_concentration, q_rate)) in p_q_concentration_rate {
        let dist_p_py = py_env
            .distributions
            .getattr("Gamma")
            .expect("call Gamma failed")
            .call1((
                tensor_to_py_obj(&py_env, &p_concentration),
                tensor_to_py_obj(&py_env, &p_rate),
            ))
            .unwrap();
        let dist_p_rs = Gamma::new(p_concentration, p_rate);

        let dist_q_py = py_env
            .distributions
            .getattr("Gamma")
            .expect("call Gamma failed")
            .call1((
                tensor_to_py_obj(&py_env, &q_concentration),
                tensor_to_py_obj(&py_env, &q_rate),
            ))
            .unwrap();
        let dist_q_rs = Gamma::new(q_concentration, q_rate);

        test_kl_divergence(&py_env, &dist_p_rs, &dist_q_rs, dist_p_py, dist_q_py);
    }
}

#[test]
#[serial]
fn geometric() {
    let gil = Python::acquire_gil();
    let py_env = PyEnv::new(&gil);

    let probs: Vec<Tensor> = vec![0.1337.into(), 0.6667.into(), 1.0.into()];

    let mut test_cases = TestCases::default();
    test_cases.icdf = None;
    test_cases.cdf = None;
    test_cases.sample = Some(vec![vec![1], vec![1, 2]]);

    for probs in probs.into_iter() {
        let dist_py = py_env
            .distributions
            .getattr("Geometric")
            .expect("call Geometric failed")
            .call1((tensor_to_py_obj(&py_env, &probs),))
            .unwrap();
        let dist_rs = Geometric::from_probs(probs);
        run_test_cases(&py_env, dist_rs, dist_py, &test_cases);
    }

    let logits: Vec<Tensor> = vec![0.1337.into(), 0.6667.into(), 1.0.into()];

    let mut test_cases = TestCases::default();
    test_cases.icdf = None;
    test_cases.cdf = None;

    for logits in logits.into_iter() {
        let dist_py = py_env
            .distributions
            .getattr("Geometric")
            .expect("call Geometric failed")
            .call1((
                pyo3::Python::None(py_env.py),
                tensor_to_py_obj(&py_env, &logits).to_object(py_env.py),
            ))
            .unwrap();
        let dist_rs = Geometric::from_logits(logits);
        run_test_cases(&py_env, dist_rs, dist_py, &test_cases);
    }

    let p_q_probs: Vec<(Tensor, Tensor)> = vec![(0.3.into(), 0.7.into()), (0.6.into(), 0.5.into())];

    for (p_probs, q_probs) in p_q_probs {
        let dist_p_py = py_env
            .distributions
            .getattr("Geometric")
            .expect("call Geometric failed")
            .call1((tensor_to_py_obj(&py_env, &p_probs),))
            .unwrap();
        let dist_p_rs = Geometric::from_probs(p_probs);

        let dist_q_py = py_env
            .distributions
            .getattr("Geometric")
            .expect("call Geometric failed")
            .call1((tensor_to_py_obj(&py_env, &q_probs),))
            .unwrap();
        let dist_q_rs = Geometric::from_probs(q_probs);

        test_kl_divergence(&py_env, &dist_p_rs, &dist_q_rs, dist_p_py, dist_q_py);
    }
}

#[test]
#[serial]
fn multivariate_normal() {
    let gil = Python::acquire_gil();
    let py_env = PyEnv::new(&gil);

    // 1.init/test with mean and covariance
    // NOTE: as pytorch takes float64 as default number type,
    // Here we use tch::Kind::Double to make consistent
    let mean_and_covs: Vec<(Tensor, Tensor)> = vec![
        (
            Tensor::ones(&[1], (tch::Kind::Double, tch::Device::Cpu)),
            Tensor::eye(1, (tch::Kind::Double, tch::Device::Cpu)),
        ),
        (
            Tensor::of_slice(&[1f64, 2.0, 3.0]),
            Tensor::try_from(array![[3f64, 0.0, 0.0], [0.0, 7.0, 0.0], [0.0, 0.0, 10.0]]).unwrap(),
        ),
        (
            Tensor::ones(&[1, 4], (tch::Kind::Double, tch::Device::Cpu)),
            Tensor::eye(4, (tch::Kind::Double, tch::Device::Cpu)),
        ),
        (
            Tensor::ones(&[4, 2], (tch::Kind::Double, tch::Device::Cpu)),
            Tensor::eye(2, (tch::Kind::Double, tch::Device::Cpu)),
        ),
        (
            Tensor::ones(&[3], (tch::Kind::Double, tch::Device::Cpu)),
            Tensor::eye(3, (tch::Kind::Double, tch::Device::Cpu)),
        ),
    ];

    let mut test_cases = TestCases::default();
    test_cases.icdf = None;
    test_cases.cdf = None;
    test_cases.entropy = false;
    test_cases.sample = Some(vec![vec![1], vec![1, 2]]);
    test_cases.log_prob = None;

    for (mean, cov) in mean_and_covs.into_iter() {
        let dist_py = py_env
            .distributions
            .getattr("MultivariateNormal")
            .expect("call MultivariateNormal failed")
            .call1((
                tensor_to_py_obj(&py_env, &mean),
                tensor_to_py_obj(&py_env, &cov),
            ))
            .unwrap();
        let dist_rs = MultivariateNormal::from_cov(mean, cov);

        // The test of resampling is not in function `run_test_cases`,
        // because `rsample` is not a method of trait `Distribution`
        if let Some(sample) = &test_cases.sample {
            test_rsample_of_multi_var_normal_distribution(&py_env, &dist_rs, dist_py, sample);
        }

        // run_test_cases(&py_env, dist_rs, dist_py, &test_cases);
    }

    // 2.init/test with mean and precisions
    // NOTE: as pytorch tasks float64 as default number type,
    // Here we use tch::Kind::Double to make consistent
    let mean_and_precisions: Vec<(Tensor, Tensor)> = vec![
        (
            Tensor::ones(&[1, 1], (tch::Kind::Double, tch::Device::Cpu)),
            Tensor::of_slice(&[0.6f64]).reshape(&[1, 1]), //precision has to be float32, not int, not double
        ),
        (
            Tensor::ones(&[1, 2], (tch::Kind::Double, tch::Device::Cpu)),
            Tensor::of_slice(&[0.6f64, 0.4, 0.5, 0.5]).reshape(&[2, 2]),
        ),
    ];
    let mut test_cases = TestCases::default();
    test_cases.icdf = None;
    test_cases.cdf = None;
    test_cases.sample = Some(vec![vec![1], vec![1, 2]]);
    for (mean, precision) in mean_and_precisions.into_iter() {
        let dist_py = py_env
            .distributions
            .getattr("MultivariateNormal")
            .expect("call MultivariateNormal failed")
            .call1((
                tensor_to_py_obj(&py_env, &mean),
                pyo3::Python::None(py_env.py),
                tensor_to_py_obj(&py_env, &precision),
            ))
            .unwrap();
        let dist_rs = MultivariateNormal::from_precision(mean, precision);
        // The test of rsampling is not in function `run_test_cases`,
        // because `rsample` is not a method of trait `Distribution`
        if let Some(sample) = &test_cases.sample {
            test_rsample_of_multi_var_normal_distribution(&py_env, &dist_rs, dist_py, sample);
        }
        // run_test_cases(&py_env, dist_rs, dist_py, &test_cases);
    }

    // 3.init/test with mean and scale_trils
    // NOTE: as pytorch takes float64 as default number type,
    // Here we use tch::Kind::Double to make consistent
    let mean_and_scale_trils: Vec<(Tensor, Tensor)> = vec![
        (
            Tensor::ones(&[1], (tch::Kind::Double, tch::Device::Cpu)),
            Tensor::eye(1, (tch::Kind::Double, tch::Device::Cpu)),
        ),
        (
            Tensor::of_slice(&[1f64, 2.0, 3.0]),
            Tensor::try_from(array![[3f64, 0.0, 0.0], [0.0, 7.0, 0.0], [0.0, 0.0, 10.0]]).unwrap(),
        ),
        (
            Tensor::ones(&[1, 4], (tch::Kind::Double, tch::Device::Cpu)),
            Tensor::eye(4, (tch::Kind::Double, tch::Device::Cpu)),
        ),
        (
            Tensor::ones(&[4, 2], (tch::Kind::Double, tch::Device::Cpu)),
            Tensor::eye(2, (tch::Kind::Double, tch::Device::Cpu)),
        ),
        (
            Tensor::ones(&[3], (tch::Kind::Double, tch::Device::Cpu)),
            Tensor::eye(3, (tch::Kind::Double, tch::Device::Cpu)),
        ),
    ];

    let mut test_cases = TestCases::default();
    test_cases.icdf = None;
    test_cases.cdf = None;
    test_cases.sample = Some(vec![vec![1], vec![1, 2]]);

    for (mean, scale_tril) in mean_and_scale_trils.into_iter() {
        let dist_py = py_env
            .distributions
            .getattr("MultivariateNormal")
            .expect("call MultivariateNormal failed")
            .call1((
                tensor_to_py_obj(&py_env, &mean),
                pyo3::Python::None(py_env.py),
                pyo3::Python::None(py_env.py),
                tensor_to_py_obj(&py_env, &scale_tril),
            ))
            .unwrap();
        let dist_rs = MultivariateNormal::from_scale_tril(mean, scale_tril);
        if let Some(sample) = &test_cases.sample {
            test_rsample_of_multi_var_normal_distribution(&py_env, &dist_rs, dist_py, sample);
        }
        // run_test_cases(&py_env, dist_rs, dist_py, &test_cases);
    }
}

#[test]
#[serial]
fn categorical() {
    let gil = Python::acquire_gil();
    let py_env = PyEnv::new(&gil);

    // 1.init/test with probabilities
    let prob_args_vec: Vec<Tensor> = vec![
        Tensor::of_slice(&[0.25, 0.25, 0.1, 0.4]),
        Tensor::try_from(array![[0.1, 0.1, 0.8], [0.1, 0.7, 0.2], [0.3, 0.4, 0.3]])
            .expect("initial from array failed"),
        Tensor::try_from(array![
            [[0.4, 0.4, 0.2], [0.7, 0.2, 0.1], [0.7, 0.2, 0.1]],
            [[0.3, 0.6, 0.1], [0.4, 0.1, 0.5], [0.7, 0.2, 0.1]]
        ])
        .expect("initial from array failed"),
    ];

    let mut test_cases = TestCases::default();
    test_cases.icdf = None;
    test_cases.cdf = None;
    test_cases.log_prob = Some(vec![
        1.0.into(),
        2.0.into(),
        Tensor::try_from(array![[1.0], [1.0]]).unwrap(),
        Tensor::try_from(array![[1, 0, 2], [0, 1, 2]]).unwrap(),
        // Tensor::of_slice(&[2.0, 2.0]),//invalid paramters
    ]);
    test_cases.sample = Some(vec![vec![1], vec![3]]);
    for probs in prob_args_vec.into_iter() {
        let dist_py = py_env
            .distributions
            .getattr("Categorical")
            .expect("call Categorical with probs failed")
            .call1((tensor_to_py_obj(&py_env, &probs),))
            .unwrap();
        let dist_rs = Categorical::from_probs(probs);

        // // test property mean
        // let mean_py = dist_py.getattr("mean")
        // .expect("call property mean failed");
        // let mean_rs = dist_rs.mean();
        // assert_tensor_eq(py_env.py, &mean_rs, mean_py);

        // // test property variance
        // let variance_py = dist_py.getattr("variance")
        // .expect("call property variance failed");
        // let variance_rs = dist_rs.variance();
        // assert_tensor_eq(py_env.py, &variance_rs, variance_py);

        // common test
        run_test_cases(&py_env, dist_rs, dist_py, &test_cases);
    }

    // 2.init/test with log probabilities
    let log_args_vec: Vec<Tensor> = vec![
        Tensor::of_slice(&[0.25, 0.25, 0.1, 0.4]),
        Tensor::try_from(array![[0.1, 0.1, 0.8], [0.1, 0.7, 0.2], [0.3, 0.4, 0.3]])
            .expect("initial from array failed"),
        Tensor::try_from(array![
            [[0.4, 0.4, 0.2], [0.7, 0.2, 0.1], [0.7, 0.2, 0.1]],
            [[0.3, 0.6, 0.1], [0.4, 0.1, 0.5], [0.7, 0.2, 0.1]]
        ])
        .expect("initial from array failed"),
    ];

    let mut test_cases = TestCases::default();
    test_cases.icdf = None;
    test_cases.cdf = None;
    test_cases.log_prob = Some(vec![
        1.0.into(),
        2.0.into(),
        Tensor::try_from(array![[1.0], [1.0]]).unwrap(),
        Tensor::try_from(array![[1, 0, 2], [0, 1, 2]]).unwrap(),
        // Tensor::of_slice(&[2.0, 2.0]),//invalid paramters
    ]);
    test_cases.sample = Some(vec![vec![1], vec![3]]);
    for logits in log_args_vec.into_iter() {
        let dist_py = py_env
            .distributions
            .getattr("Categorical")
            .expect("call Categorical with logits failed")
            .call1((
                pyo3::Python::None(py_env.py),
                tensor_to_py_obj(&py_env, &logits).to_object(py_env.py),
            ))
            .unwrap();
        let dist_rs = Categorical::from_logits(logits);

        // // test property mean
        // let mean_py = dist_py.getattr("mean")
        // .expect("call property mean failed");
        // let mean_rs = dist_rs.mean();
        // assert_tensor_eq(py_env.py, &mean_rs, mean_py);

        // // test property variance
        // let variance_py = dist_py.getattr("variance")
        // .expect("call property variance failed");
        // let variance_rs = dist_rs.variance();
        // assert_tensor_eq(py_env.py, &variance_rs, variance_py);

        // common test
        run_test_cases(&py_env, dist_rs, dist_py, &test_cases);
    }

    // TODO: test kl divergence?
}
