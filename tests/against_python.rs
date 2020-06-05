use ndarray::ArrayD;
use numpy::{PyArrayDyn, ToPyArray};
use pyo3::{prelude::*, types::PyTuple};
use serial_test::serial;
use std::convert::TryInto;
use tch::Tensor;
use tch_distr::{
    Bernoulli, Cauchy, Distribution, Exponential, Gamma, Geometric, Normal, Poisson, Uniform,
};

const SEED: i64 = 42;

struct PyEnv<'py> {
    py: Python<'py>,
    torch: &'py PyModule,
    distributions: &'py PyModule,
}

impl<'py> PyEnv<'py> {
    fn new(gil: &'py GILGuard) -> Self {
        let py = gil.python();

        let torch = PyModule::import(py, "torch").unwrap();
        let distributions = PyModule::import(py, "torch.distributions").unwrap();

        Self {
            py,
            torch,
            distributions,
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
        .call1("from_numpy", (array.to_pyarray(py_env.py),))
        .unwrap()
}

fn assert_tensor_eq<'py>(py: Python<'py>, t: &Tensor, py_t: &PyAny) {
    let pyarray: &PyArrayDyn<f64> = py_t.call_method0("numpy").unwrap().extract().unwrap();
    let array: ArrayD<f64> = t.try_into().unwrap();
    assert_eq!(
        array.to_pyarray(py).as_slice().unwrap(),
        pyarray.as_slice().unwrap()
    );
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
    for args in args.into_iter() {
        let args_py = PyTuple::new(py_env.py, vec![tensor_to_py_obj(py_env, args)]);
        let log_prob_py = dist_py.call_method1("icdf", args_py).unwrap();
        let log_prob_rs = dist_rs.icdf(args);
        assert_tensor_eq(py_env.py, &log_prob_rs, log_prob_py);
    }
}

fn test_sample<D: Distribution>(py_env: &PyEnv, dist_rs: &D, dist_py: &PyAny, args: &[Vec<i64>]) {
    for args in args.into_iter() {
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

fn run_test_cases<D>(py_env: &PyEnv, dist_rs: D, dist_py: &PyAny, test_cases: &TestCases)
where
    D: Distribution,
{
    if test_cases.entropy {
        test_entropy(py_env, &dist_rs, dist_py);
    }
    if let Some(log_prob) = test_cases.log_prob.as_ref() {
        test_log_prob(py_env, &dist_rs, dist_py, &log_prob);
    }
    if let Some(cdf) = test_cases.cdf.as_ref() {
        test_cdf(py_env, &dist_rs, dist_py, &cdf);
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
    ];

    let mut test_cases = TestCases::default();
    test_cases.sample = Some(vec![vec![1], vec![1, 2]]);

    for (mean, std) in args.into_iter() {
        let dist_py = py_env
            .distributions
            .call1(
                "Normal",
                (
                    tensor_to_py_obj(&py_env, &mean),
                    tensor_to_py_obj(&py_env, &std),
                ),
            )
            .unwrap();
        let dist_rs = Normal::new(mean, std);
        run_test_cases(&py_env, dist_rs, dist_py, &test_cases);
    }
}

#[test]
#[serial]
fn uniform() {
    let gil = Python::acquire_gil();
    let py_env = PyEnv::new(&gil);

    let args: Vec<(Tensor, Tensor)> = vec![
        (1.0.into(), 2.0.into()),
        (2.0.into(), 4.0.into()),
        (Tensor::of_slice(&[1.0, 1.0]), Tensor::of_slice(&[2.0, 2.0])),
    ];

    let mut test_cases = TestCases::default();
    test_cases.sample = Some(vec![vec![1], vec![1, 2]]);

    for (low, high) in args.into_iter() {
        let dist_py = py_env
            .distributions
            .call1(
                "Uniform",
                (
                    tensor_to_py_obj(&py_env, &low),
                    tensor_to_py_obj(&py_env, &high),
                ),
            )
            .unwrap();
        let dist_rs = Uniform::new(low, high);
        run_test_cases(&py_env, dist_rs, dist_py, &test_cases);
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

    for probs in probs.into_iter() {
        let dist_py = py_env
            .distributions
            .call1("Bernoulli", (tensor_to_py_obj(&py_env, &probs),))
            .unwrap();
        let dist_rs = Bernoulli::from_probs(probs);
        run_test_cases(&py_env, dist_rs, dist_py, &test_cases);
    }

    let logits: Vec<Tensor> = vec![0.1337.into(), 0.6667.into()];

    let mut test_cases = TestCases::default();
    test_cases.icdf = None;
    test_cases.cdf = None;

    for logits in logits.into_iter() {
        let dist_py = py_env
            .distributions
            .call1(
                "Bernoulli",
                (
                    pyo3::Python::None(py_env.py),
                    tensor_to_py_obj(&py_env, &logits).to_object(py_env.py),
                ),
            )
            .unwrap();
        let dist_rs = Bernoulli::from_logits(logits);
        run_test_cases(&py_env, dist_rs, dist_py, &test_cases);
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

    for rate in rates.into_iter() {
        let dist_py = py_env
            .distributions
            .call1("Poisson", (tensor_to_py_obj(&py_env, &rate),))
            .unwrap();
        let dist_rs = Poisson::new(rate);
        run_test_cases(&py_env, dist_rs, dist_py, &test_cases);
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

    let test_cases = TestCases::default();

    for rate in rates.into_iter() {
        let dist_py = py_env
            .distributions
            .call1("Exponential", (tensor_to_py_obj(&py_env, &rate),))
            .unwrap();
        let dist_rs = Exponential::new(rate);
        run_test_cases(&py_env, dist_rs, dist_py, &test_cases);
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

    let test_cases = TestCases::default();

    for (median, scale) in args.into_iter() {
        let dist_py = py_env
            .distributions
            .call1(
                "Cauchy",
                (
                    tensor_to_py_obj(&py_env, &median),
                    tensor_to_py_obj(&py_env, &scale),
                ),
            )
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
            .call1(
                "Gamma",
                (
                    tensor_to_py_obj(&py_env, &concentration),
                    tensor_to_py_obj(&py_env, &rate),
                ),
            )
            .unwrap();
        let dist_rs = Gamma::new(concentration, rate);
        run_test_cases(&py_env, dist_rs, dist_py, &test_cases);
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

    for probs in probs.into_iter() {
        let dist_py = py_env
            .distributions
            .call1("Geometric", (tensor_to_py_obj(&py_env, &probs),))
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
            .call1(
                "Geometric",
                (
                    pyo3::Python::None(py_env.py),
                    tensor_to_py_obj(&py_env, &logits).to_object(py_env.py),
                ),
            )
            .unwrap();
        let dist_rs = Geometric::from_logits(logits);
        run_test_cases(&py_env, dist_rs, dist_py, &test_cases);
    }
}
