use crate::data::batch_indices;
use crate::model::Model;
use crate::optim::{GradNorm, Optimizer, OptimizerState};
use candle::{DType, Result, Tensor, Var};

#[derive(Debug, Clone, Copy)]
pub struct LrPoint {
    pub lr: f64,
    pub loss: f64,
    pub grad_norm: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct LrFinder {
    min_lr: f64,
    max_lr: f64,
    steps: usize,
    stop_div: f64,
    seed: u64,
}

impl LrFinder {
    pub fn new(min_lr: f64, max_lr: f64, steps: usize) -> Result<Self> {
        if min_lr <= 0.0 {
            candle::bail!("min_lr must be positive, got {min_lr}");
        }
        if max_lr <= min_lr {
            candle::bail!("max_lr {max_lr} must be greater than min_lr {min_lr}");
        }
        if steps < 2 {
            candle::bail!("steps must be at least 2, got {steps}");
        }
        Ok(Self {
            min_lr,
            max_lr,
            steps,
            stop_div: 4.0,
            seed: 42,
        })
    }

    pub fn with_stop_div(mut self, stop_div: f64) -> Result<Self> {
        if stop_div <= 1.0 {
            candle::bail!("stop_div must be greater than 1, got {stop_div}");
        }
        self.stop_div = stop_div;
        Ok(self)
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn run<O, F>(
        &self,
        model: &Model,
        opt: &mut O,
        x: &Tensor,
        y: &Tensor,
        loss_fn: F,
        batch_size: usize,
    ) -> Result<Vec<LrPoint>>
    where
        O: Optimizer + OptimizerState + GradNorm,
        F: Fn(&Tensor, &Tensor) -> Result<Tensor>,
    {
        if batch_size == 0 {
            candle::bail!("batch_size must be positive");
        }
        let n = x.dim(0)?;
        if n == 0 {
            candle::bail!("lr finder needs a non-empty dataset");
        }
        let weights: Vec<Var> = model.vars();
        let saved: Vec<Tensor> = weights
            .iter()
            .map(|var| var.as_tensor().copy())
            .collect::<Result<Vec<_>>>()?;
        let state = opt.state_tensors()?;
        let base_lr = opt.learning_rate();
        let batches = batch_indices(n, batch_size, self.seed);
        let ratio = (self.max_lr / self.min_lr).powf(1.0 / (self.steps - 1) as f64);
        let mut points = Vec::new();
        let mut best = f64::INFINITY;
        for step in 0..self.steps {
            let indices = &batches[step % batches.len()];
            let idx = Tensor::from_vec(
                indices.iter().map(|&i| i as u32).collect::<Vec<u32>>(),
                indices.len(),
                x.device(),
            )?;
            let xb = x.index_select(&idx, 0)?;
            let yb = y.index_select(&idx, 0)?;
            let lr = self.min_lr * ratio.powi(step as i32);
            opt.set_learning_rate(lr);
            let loss = loss_fn(&xb, &yb)?;
            let value = loss.to_dtype(DType::F64)?.to_scalar::<f64>()?;
            opt.backward_step(&loss)?;
            let grad_norm = opt.last_grad_norm().unwrap_or(f64::NAN);
            points.push(LrPoint {
                lr,
                loss: value,
                grad_norm,
            });
            if value < best {
                best = value;
            }
            if value.is_nan() || value.is_infinite() || value > self.stop_div * best {
                break;
            }
        }
        for (var, tensor) in weights.iter().zip(&saved) {
            var.set(tensor)?;
        }
        opt.restore_state(&state)?;
        opt.set_learning_rate(base_lr);
        Ok(points)
    }
}

pub fn suggested_lr(points: &[LrPoint]) -> Option<f64> {
    if points.len() < 2 {
        return None;
    }
    let mut best: Option<(f64, f64)> = None;
    for pair in points.windows(2) {
        let dlr = (pair[1].lr / pair[0].lr).ln();
        if dlr <= 0.0 {
            continue;
        }
        let slope = (pair[1].loss - pair[0].loss) / dlr;
        if best.is_none_or(|(_, s)| slope < s) {
            best = Some((pair[0].lr, slope));
        }
    }
    best.map(|(lr, _)| lr)
}
