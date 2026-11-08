use crate::data::batch_indices;
use crate::optim::Optimizer;
use candle::{DType, Result, Tensor};

#[derive(Debug, Clone, Copy)]
pub struct FitConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub shuffle: bool,
    pub seed: u64,
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            epochs: 1,
            batch_size: 32,
            shuffle: true,
            seed: 42,
        }
    }
}

pub fn fit<O, F>(
    opt: &mut O,
    loss_fn: F,
    x: &Tensor,
    y: &Tensor,
    config: FitConfig,
) -> Result<Vec<f64>>
where
    O: Optimizer,
    F: Fn(&Tensor, &Tensor) -> Result<Tensor>,
{
    if config.batch_size == 0 {
        candle::bail!("batch_size must be positive");
    }
    let n = x.dim(0)?;
    let mut history = Vec::new();
    for epoch in 0..config.epochs {
        let batches = match config.shuffle {
            true => batch_indices(n, config.batch_size, config.seed + epoch as u64),
            false => (0..n)
                .collect::<Vec<usize>>()
                .chunks(config.batch_size)
                .map(<[usize]>::to_vec)
                .collect(),
        };
        let mut total = 0.0f64;
        let mut count = 0usize;
        for indices in batches {
            let len = indices.len();
            let idx = Tensor::from_vec(
                indices.iter().map(|&i| i as u32).collect::<Vec<u32>>(),
                len,
                x.device(),
            )?;
            let xb = x.index_select(&idx, 0)?;
            let yb = y.index_select(&idx, 0)?;
            let loss = loss_fn(&xb, &yb)?;
            opt.backward_step(&loss)?;
            total += loss.to_dtype(DType::F64)?.to_scalar::<f64>()? * len as f64;
            count += len;
        }
        let average = if count > 0 { total / count as f64 } else { 0.0 };
        history.push(average);
    }
    Ok(history)
}
