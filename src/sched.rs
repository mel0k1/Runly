use crate::optim::Optimizer;

pub trait LrScheduler {
    fn lr(&mut self) -> f64;

    fn apply<O: Optimizer>(&mut self, opt: &mut O) {
        let lr = self.lr();
        opt.set_learning_rate(lr);
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ExponentialLR {
    base_lr: f64,
    gamma: f64,
    step: usize,
}

impl ExponentialLR {
    pub fn new(base_lr: f64, gamma: f64) -> candle::Result<Self> {
        if !(0.0..=1.0).contains(&gamma) {
            candle::bail!("gamma must be in [0, 1], got {gamma}");
        }
        Ok(Self {
            base_lr,
            gamma,
            step: 0,
        })
    }
}

impl LrScheduler for ExponentialLR {
    fn lr(&mut self) -> f64 {
        let lr = self.base_lr * self.gamma.powi(self.step as i32);
        self.step += 1;
        lr
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StepLR {
    base_lr: f64,
    step_size: usize,
    gamma: f64,
    step: usize,
}

impl StepLR {
    pub fn new(base_lr: f64, step_size: usize, gamma: f64) -> candle::Result<Self> {
        if step_size == 0 {
            candle::bail!("step_size must be positive");
        }
        if !(0.0..=1.0).contains(&gamma) {
            candle::bail!("gamma must be in [0, 1], got {gamma}");
        }
        Ok(Self {
            base_lr,
            step_size,
            gamma,
            step: 0,
        })
    }
}

impl LrScheduler for StepLR {
    fn lr(&mut self) -> f64 {
        let lr = self.base_lr * self.gamma.powi((self.step / self.step_size) as i32);
        self.step += 1;
        lr
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CosineAnnealingLR {
    base_lr: f64,
    t_max: usize,
    eta_min: f64,
    step: usize,
}

impl CosineAnnealingLR {
    pub fn new(base_lr: f64, t_max: usize, eta_min: f64) -> candle::Result<Self> {
        if t_max == 0 {
            candle::bail!("t_max must be positive");
        }
        if eta_min > base_lr {
            candle::bail!("eta_min {eta_min} is above base lr {base_lr}");
        }
        Ok(Self {
            base_lr,
            t_max,
            eta_min,
            step: 0,
        })
    }
}

impl LrScheduler for CosineAnnealingLR {
    fn lr(&mut self) -> f64 {
        let t = self.step as f64 / self.t_max as f64;
        let lr = self.eta_min
            + (self.base_lr - self.eta_min) * (1.0 + (std::f64::consts::PI * t).cos()) / 2.0;
        self.step += 1;
        lr
    }
}
