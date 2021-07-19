use crate::Distribution;
use std::f64::consts::PI;
use tch::{
    Kind::{Double, Float},
    Tensor,
};

/// A Multivariate Normal distribution.
#[derive(Debug)]
pub struct MultivariateNormal {
    mean: Tensor,
    scale_tril: Tensor,
    cov: Tensor,
    batch_shape: Vec<i64>,
    event_shape: Vec<i64>,
}

fn split_shapes(shape: &[i64]) -> (Vec<i64>, Vec<i64>) {
    match shape.split_last() {
        Some((last, before)) => (vec![*last], before.to_vec()),
        None => (vec![1], vec![0]),
    }
}

impl MultivariateNormal {
    /// Creates a Multivariate Normal distribution with `mean` and covariance matrix `cov`.
    pub fn from_cov(mean: Tensor, cov: Tensor) -> Self {
        let mean_ = mean.unsqueeze(-1);
        let cov_mean = Tensor::broadcast_tensors(&[cov.copy(), mean_]);
        let mean_size = cov_mean[1]
            .size()
            .split_last()
            .map(|(_, before)| before.to_vec())
            .unwrap_or_else(Vec::new);
        let (event_shape, batch_shape) = split_shapes(&mean_size);
        Self {
            mean: cov_mean[1].mean_dim(&[-1], false, cov_mean[1].kind()),
            scale_tril: cov.cholesky(false),
            cov: cov_mean[0].copy(),
            batch_shape,
            event_shape,
        }
    }

    /// Creates a Multivariate Normal distribution with `mean` and precision matrix `precision`.
    pub fn from_precision(mean: Tensor, precision: Tensor) -> Self {
        let mean_ = mean.unsqueeze(-1);
        let precision_mean = Tensor::broadcast_tensors(&[precision.copy(), mean_]);
        let mean_size = precision_mean[1]
            .size()
            .split_last()
            .map(|(_, before)| before.to_vec())
            .unwrap_or_else(Vec::new);
        let (event_shape, batch_shape) = split_shapes(&mean_size);
        let scale_tril = precision_to_scale_tril(&precision_mean[0]);
        let cov = scale_tril.matmul(&scale_tril.transpose(-1, -2)).expand(
            &[
                batch_shape.as_slice(),
                event_shape.as_slice(),
                event_shape.as_slice(),
            ]
            .concat(),
            true,
        );
        Self {
            mean: precision_mean[1].mean_dim(&[-1], false, precision_mean[1].kind()),
            scale_tril,
            cov,
            batch_shape,
            event_shape,
        }
    }

    /// Creates a Multivariate Normal distribution with `mean` and scale tril matrix `scale_tril`.
    pub fn from_scale_tril(mean: Tensor, scale_tril: Tensor) -> Self {
        let mean_ = mean.unsqueeze(-1);
        let scale_tril_mean = Tensor::broadcast_tensors(&[scale_tril.copy(), mean_]);
        let mean_size = scale_tril_mean[1]
            .size()
            .split_last()
            .map(|(_, before)| before.to_vec())
            .unwrap_or_else(Vec::new);
        let (event_shape, batch_shape) = split_shapes(&mean_size);
        let cov = scale_tril.matmul(&scale_tril.transpose(-1, -2)).expand(
            &[
                batch_shape.as_slice(),
                event_shape.as_slice(),
                &event_shape.as_slice(),
            ]
            .concat(),
            true,
        );
        Self {
            mean: scale_tril_mean[1].mean_dim(&[-1], false, scale_tril_mean[1].kind()),
            scale_tril: scale_tril_mean[0].copy(),
            cov,
            batch_shape,
            event_shape,
        }
    }
}

impl Distribution for MultivariateNormal {
    fn entropy(&self) -> Tensor {
        let half_log_det = self
            .scale_tril
            .diagonal(0, -2, -1)
            .log()
            .sum_dim_intlist(&[-1], true, Double);
        let h = (0.5 * self.event_shape[0] as f64) * (1.0 + (2.0 * PI).ln()) + half_log_det;
        if self.batch_shape.is_empty() {
            h
        } else {
            h.expand(&self.batch_shape, true)
        }
    }

    fn sample(&self, shape: &[i64]) -> Tensor {
        let shape = self.extended_shape(shape);
        let eps = Tensor::normal_tensor_tensor_out(
            &Tensor::empty(&shape, (self.mean.kind(), self.mean.device())),
            &Tensor::from(0.0).expand(&shape, false),
            &Tensor::from(1.0).expand(&shape, false),
        );
        &self.mean + &self.scale_tril.matmul(&eps.unsqueeze(-1)).squeeze_dim(-1)
    }

    fn log_prob(&self, val: &Tensor) -> Tensor {
        let diff = val - &self.mean;
        let m = batch_mahalanobis(&self.scale_tril, &diff).totype(Double);
        let half_log_det = self
            .scale_tril
            .diagonal(0, -2, -1)
            .log()
            .sum_dim_intlist(&[-1], true, Double);
        -0.5 * (self.event_shape[0] as f64 * (2.0 * PI).ln() + m) - half_log_det
    }

    fn batch_shape(&self) -> &[i64] {
        &self.batch_shape
    }

    fn event_shape(&self) -> &[i64] {
        &self.event_shape
    }
}

fn precision_to_scale_tril(precision_matrix: &Tensor) -> Tensor {
    let l_f = precision_matrix.flip(&[-2, -1]);
    let l_inv = l_f.flip(&[-2, -1]).transpose(-2, -1);
    let (l, _) = Tensor::eye(
        *precision_matrix.size().last().unwrap(),
        (precision_matrix.kind(), precision_matrix.device()),
    )
    .triangular_solve(&l_inv, false, false, false);
    l
}

fn batch_mahalanobis(b_l: &Tensor, b_x: &Tensor) -> Tensor {
    let n = *b_x.size().last().unwrap();
    let mut b_x_batch_shape = b_x.size();
    b_x_batch_shape.pop();

    let b_x_batch_dims = b_x_batch_shape.len() as i64;
    let b_l_batch_dims = b_l.dim() as i64 - 2;

    let outer_batch_dims = b_x_batch_dims - b_l_batch_dims;
    let old_batch_dims = outer_batch_dims + b_l_batch_dims;
    let new_batch_dims = outer_batch_dims + 2 * b_l_batch_dims;

    let mut bx_new_shape = Vec::new();
    for (&s_x, &s_l) in b_l.size().iter().take(b_l.size().len().max(2) - 2).zip(
        b_x_batch_shape
            .iter()
            .skip(outer_batch_dims.min(0) as usize),
    ) {
        bx_new_shape.push(s_x / s_l);
        bx_new_shape.push(s_l);
    }
    bx_new_shape.push(n);

    let b_x = b_x.reshape(&bx_new_shape);
    let permute_dims = (0..outer_batch_dims)
        .chain((outer_batch_dims..new_batch_dims).step_by(2))
        .chain((outer_batch_dims + 1..new_batch_dims).step_by(2))
        .chain(vec![new_batch_dims])
        .collect::<Vec<_>>();
    let b_x = b_x.permute(&permute_dims);

    let flat_l = b_l.reshape(&[-1, n, n]);
    let flat_x = b_x.reshape(&[-1, flat_l.size()[0], n]);
    let flat_x_swap = flat_x.permute(&[1, 2, 0]);

    let m_swap = flat_x_swap
        .triangular_solve(&flat_l, false, false, false)
        .0
        .pow(2)
        .sum_dim_intlist(&[-2], true, Float);
    let m = m_swap.transpose(0, 1);

    let permuted_m = m.reshape(&b_x_batch_shape);

    let mut permute_inv_dims =
        Vec::with_capacity(outer_batch_dims as usize + (b_l_batch_dims.min(0) * 2) as usize);
    for i in 0..outer_batch_dims {
        permute_inv_dims.push(i);
    }
    for i in 0..b_l_batch_dims {
        permute_inv_dims.push(outer_batch_dims + i);
        permute_inv_dims.push(old_batch_dims + i);
    }

    let reshaped_m = permuted_m.permute(&permute_inv_dims);
    reshaped_m.reshape(&b_x_batch_shape)
}
