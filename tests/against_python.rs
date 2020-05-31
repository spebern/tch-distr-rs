use ndarray::ArrayD;
use numpy::{PyArrayDyn, ToPyArray};
use pyo3::{prelude::*, types::PyTuple};
use std::convert::TryInto;
use tch::Tensor;
use tch_distr::{Distribution, Normal};

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
    args: Vec<Tensor>,
) {
    for args in args.into_iter() {
        let args_py = PyTuple::new(py, vec![tensor_to_py_obj(py, torch, &args)]);
        let log_prob_py = dist_py.call_method("log_prob", args_py, None).unwrap();
        let log_prob_rs = dist_rs.log_prob(&args);
        assert_tensor_eq(py, &log_prob_rs, log_prob_py);
    }
}

fn test_cdf<'py, D: Distribution>(
    py: Python<'py>,
    torch: &'py PyModule,
    dist_rs: &D,
    dist_py: &PyAny,
    args: Vec<Tensor>,
) {
    for args in args.into_iter() {
        let args_py = PyTuple::new(py, vec![tensor_to_py_obj(py, torch, &args)]);
        let log_prob_py = dist_py.call_method("cdf", args_py, None).unwrap();
        let log_prob_rs = dist_rs.cdf(&args);
        assert_tensor_eq(py, &log_prob_rs, log_prob_py);
    }
}

fn test_icdf<'py, D: Distribution>(
    py: Python<'py>,
    torch: &'py PyModule,
    dist_rs: &D,
    dist_py: &PyAny,
    args: Vec<Tensor>,
) {
    for args in args.into_iter() {
        let args_py = PyTuple::new(py, vec![tensor_to_py_obj(py, torch, &args)]);
        let log_prob_py = dist_py.call_method("icdf", args_py, None).unwrap();
        let log_prob_rs = dist_rs.icdf(&args);
        assert_tensor_eq(py, &log_prob_rs, log_prob_py);
    }
}

#[test]
fn normal() {
    let gil = Python::acquire_gil();
    let py = gil.python();

    let torch = PyModule::import(py, "torch").unwrap();
    let distributions = PyModule::import(py, "torch.distributions").unwrap();

    let args: Vec<(Tensor, Tensor)> = vec![
        (1.0.into(), 2.0.into()),
        (2.0.into(), 4.0.into()),
        (Tensor::of_slice(&[1.0, 1.0]), Tensor::of_slice(&[2.0, 2.0])),
    ];

    for (mean, std) in args.into_iter() {
        let args_py = PyTuple::new(
            py,
            vec![
                tensor_to_py_obj(py, torch, &mean),
                tensor_to_py_obj(py, torch, &std),
            ],
        );
        let dist_py = distributions.call1("Normal", args_py).unwrap();
        let dist_rs = Normal::new(mean, std);

        test_entropy(py, &dist_rs, dist_py);

        let args = vec![
            1.0.into(),
            2.0.into(),
            Tensor::of_slice(&[1.0, 1.0]),
            Tensor::of_slice(&[2.0, 2.0]),
        ];
        test_log_prob(py, torch, &dist_rs, dist_py, args);

        let args = vec![
            1.0.into(),
            2.0.into(),
            Tensor::of_slice(&[1.0, 1.0]),
            Tensor::of_slice(&[2.0, 2.0]),
        ];
        test_cdf(py, torch, &dist_rs, dist_py, args);

        let args = vec![
            0.5.into(),
            0.7.into(),
            Tensor::of_slice(&[0.3, 0.4]),
            Tensor::of_slice(&[0.2, 0.7]),
        ];
        test_icdf(py, torch, &dist_rs, dist_py, args);
    }
}
