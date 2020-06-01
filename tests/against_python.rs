use ndarray::ArrayD;
use numpy::{PyArrayDyn, ToPyArray};
use pyo3::{prelude::*, types::PyTuple};
use std::convert::TryInto;
use tch::Tensor;
use tch_distr::{Bernoulli, Distribution, Exponential, Normal, Poisson, Uniform};

struct TestCases {
    log_prob: Vec<Tensor>,
    cdf: Vec<Tensor>,
    icdf: Vec<Tensor>,
}

impl Default for TestCases {
    fn default() -> Self {
        Self {
            log_prob: vec![
                1.0.into(),
                2.0.into(),
                Tensor::of_slice(&[1.0, 1.0]),
                Tensor::of_slice(&[2.0, 2.0]),
            ],
            cdf: vec![
                1.0.into(),
                2.0.into(),
                Tensor::of_slice(&[1.0, 1.0]),
                Tensor::of_slice(&[2.0, 2.0]),
            ],
            icdf: vec![
                0.5.into(),
                0.7.into(),
                Tensor::of_slice(&[0.3, 0.4]),
                Tensor::of_slice(&[0.2, 0.7]),
            ],
        }
    }
}

fn tensor_to_py_obj<'py>(py: Python<'py>, torch: &'py PyModule, t: &Tensor) -> &'py PyAny {
    let array: ndarray::ArrayD<f64> = t.try_into().unwrap();
    torch.call1("from_numpy", (array.to_pyarray(py),)).unwrap()
}

fn assert_tensor_eq<'py>(py: Python<'py>, t: &Tensor, py_t: &PyAny) {
    let pyarray: &PyArrayDyn<f64> = py_t.call_method0("numpy").unwrap().extract().unwrap();
    let array: ArrayD<f64> = t.try_into().unwrap();
    assert_eq!(
        array.to_pyarray(py).as_slice().unwrap(),
        pyarray.as_slice().unwrap()
    );
}

fn test_entropy<'py, D: Distribution>(py: Python<'py>, dist_rs: &D, dist_py: &PyAny) {
    let entropy_py = dist_py.call_method("entropy", (), None).unwrap();
    let entropy_rs = dist_rs.entropy();
    assert_tensor_eq(py, &entropy_rs, entropy_py);
}

fn test_log_prob<'py, D: Distribution>(
    py: Python<'py>,
    torch: &'py PyModule,
    dist_rs: &D,
    dist_py: &PyAny,
    args: &[Tensor],
) {
    for args in args.iter() {
        let args_py = PyTuple::new(py, vec![tensor_to_py_obj(py, torch, args)]);
        let log_prob_py = dist_py.call_method("log_prob", args_py, None).unwrap();
        let log_prob_rs = dist_rs.log_prob(args);
        assert_tensor_eq(py, &log_prob_rs, log_prob_py);
    }
}

fn test_cdf<'py, D: Distribution>(
    py: Python<'py>,
    torch: &'py PyModule,
    dist_rs: &D,
    dist_py: &PyAny,
    args: &[Tensor],
) {
    for args in args.iter() {
        let args_py = PyTuple::new(py, vec![tensor_to_py_obj(py, torch, args)]);
        let log_prob_py = dist_py.call_method("cdf", args_py, None).unwrap();
        let log_prob_rs = dist_rs.cdf(args);
        assert_tensor_eq(py, &log_prob_rs, log_prob_py);
    }
}

fn test_icdf<'py, D: Distribution>(
    py: Python<'py>,
    torch: &'py PyModule,
    dist_rs: &D,
    dist_py: &PyAny,
    args: &[Tensor],
) {
    for args in args.into_iter() {
        let args_py = PyTuple::new(py, vec![tensor_to_py_obj(py, torch, args)]);
        let log_prob_py = dist_py.call_method("icdf", args_py, None).unwrap();
        let log_prob_rs = dist_rs.icdf(args);
        assert_tensor_eq(py, &log_prob_rs, log_prob_py);
    }
}

fn run_test_cases<'py, D, T, U>(
    py: Python<'py>,
    torch: &'py PyModule,
    dist_rs: D,
    dist_name: &str,
    dist_args: impl IntoIterator<Item = T, IntoIter = U>,
    test_cases: &TestCases,
) where
    D: Distribution,
    T: ToPyObject,
    U: ExactSizeIterator<Item = T>,
{
    let distributions = PyModule::import(py, "torch.distributions").unwrap();

    let dist_args = PyTuple::new(py, dist_args);
    let dist_py = distributions.call1(dist_name, dist_args).unwrap();

    test_entropy(py, &dist_rs, dist_py);
    test_log_prob(py, torch, &dist_rs, dist_py, &test_cases.log_prob);
    test_cdf(py, torch, &dist_rs, dist_py, &test_cases.cdf);
    test_icdf(py, torch, &dist_rs, dist_py, &test_cases.icdf);
}

#[test]
fn normal() {
    let gil = Python::acquire_gil();
    let py = gil.python();

    let torch = PyModule::import(py, "torch").unwrap();

    let args: Vec<(Tensor, Tensor)> = vec![
        (1.0.into(), 2.0.into()),
        (2.0.into(), 4.0.into()),
        (Tensor::of_slice(&[1.0, 1.0]), Tensor::of_slice(&[2.0, 2.0])),
    ];

    let test_cases = TestCases::default();
    for (mean, std) in args.into_iter() {
        let args_py = vec![
            tensor_to_py_obj(py, torch, &mean),
            tensor_to_py_obj(py, torch, &std),
        ];
        let dist_rs = Normal::new(mean, std);

        run_test_cases(py, torch, dist_rs, "Normal", args_py, &test_cases);
    }
}

#[test]
fn uniform() {
    let gil = Python::acquire_gil();
    let py = gil.python();

    let torch = PyModule::import(py, "torch").unwrap();

    let args: Vec<(Tensor, Tensor)> = vec![
        (1.0.into(), 2.0.into()),
        (2.0.into(), 4.0.into()),
        (Tensor::of_slice(&[1.0, 1.0]), Tensor::of_slice(&[2.0, 2.0])),
    ];

    let test_cases = TestCases::default();
    for (low, high) in args.into_iter() {
        let args_py = vec![
            tensor_to_py_obj(py, torch, &low),
            tensor_to_py_obj(py, torch, &high),
        ];
        let dist_rs = Uniform::new(low, high);

        run_test_cases(py, torch, dist_rs, "Uniform", args_py, &test_cases);
    }
}

#[test]
fn bernoulli() {
    let gil = Python::acquire_gil();
    let py = gil.python();

    let torch = PyModule::import(py, "torch").unwrap();
    let distributions = PyModule::import(py, "torch.distributions").unwrap();

    let probs: Vec<Tensor> = vec![0.1337.into(), 0.6667.into()];

    let test_cases = TestCases::default();
    for probs in probs.into_iter() {
        let args_py = vec![tensor_to_py_obj(py, torch, &probs)];
        let dist_py = distributions
            .call1("Bernoulli", PyTuple::new(py, args_py))
            .unwrap();

        let dist_rs = Bernoulli::from_probs(probs);
        test_entropy(py, &dist_rs, dist_py);
        test_log_prob(py, torch, &dist_rs, dist_py, &test_cases.log_prob);
    }

    let logits: Vec<Tensor> = vec![0.1337.into(), 0.6667.into()];

    let test_cases = TestCases::default();
    for logits in logits.into_iter() {
        let dist_py = distributions
            .call1(
                "Bernoulli",
                (pyo3::Python::None(py), tensor_to_py_obj(py, torch, &logits)),
            )
            .unwrap();

        let dist_rs = Bernoulli::from_logits(logits);
        test_entropy(py, &dist_rs, dist_py);
        test_log_prob(py, torch, &dist_rs, dist_py, &test_cases.log_prob);
    }
}

#[test]
fn poisson() {
    let gil = Python::acquire_gil();
    let py = gil.python();

    let torch = PyModule::import(py, "torch").unwrap();
    let distributions = PyModule::import(py, "torch.distributions").unwrap();

    let rates: Vec<Tensor> = vec![
        0.1337.into(),
        0.6667.into(),
        Tensor::of_slice(&[0.156, 0.33]),
    ];

    let test_cases = TestCases::default();
    for rate in rates.into_iter() {
        let args_py = vec![tensor_to_py_obj(py, torch, &rate)];
        let dist_py = distributions
            .call1("Poisson", PyTuple::new(py, args_py))
            .unwrap();

        let dist_rs = Poisson::new(rate);
        test_log_prob(py, torch, &dist_rs, dist_py, &test_cases.log_prob);
    }
}

#[test]
fn exponential() {
    let gil = Python::acquire_gil();
    let py = gil.python();

    let torch = PyModule::import(py, "torch").unwrap();

    let rates: Vec<Tensor> = vec![
        0.1337.into(),
        0.6667.into(),
        Tensor::of_slice(&[0.156, 0.33]),
    ];

    let test_cases = TestCases::default();
    for rate in rates.into_iter() {
        let args_py = vec![tensor_to_py_obj(py, torch, &rate)];
        let dist_rs = Exponential::new(rate);

        run_test_cases(py, torch, dist_rs, "Exponential", args_py, &test_cases);
    }
}
