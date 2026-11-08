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

#[derive(Debug, Clone, Copy)]
pub struct Warmup<S> {
    inner: S,
    warmup_steps: usize,
    peak_lr: f64,
    step: usize,
}

impl<S: LrScheduler> Warmup<S> {
    pub fn new(inner: S, warmup_steps: usize, peak_lr: f64) -> candle::Result<Self> {
        if warmup_steps == 0 {
            candle::bail!("warmup_steps must be positive");
        }
        if peak_lr < 0.0 {
            candle::bail!("peak lr {peak_lr} is negative");
        }
        Ok(Self {
            inner,
            warmup_steps,
            peak_lr,
            step: 0,
        })
    }
}

impl<S: LrScheduler> LrScheduler for Warmup<S> {
    fn lr(&mut self) -> f64 {
        if self.step < self.warmup_steps {
            let lr = self.peak_lr * (self.step + 1) as f64 / self.warmup_steps as f64;
            self.step += 1;
            lr
        } else {
            self.inner.lr()
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct OneCycleLR {
    peak_lr: f64,
    initial_lr: f64,
    final_lr: f64,
    warmup_steps: usize,
    total_steps: usize,
    step: usize,
}

impl OneCycleLR {
    pub fn new(
        peak_lr: f64,
        total_steps: usize,
        warmup_frac: f64,
        initial_lr: f64,
        final_lr: f64,
    ) -> candle::Result<Self> {
        if total_steps < 2 {
            candle::bail!("total_steps must be at least 2");
        }
        if !(0.0..1.0).contains(&warmup_frac) {
            candle::bail!("warmup fraction must be in [0, 1), got {warmup_frac}");
        }
        if !(0.0..=peak_lr).contains(&initial_lr) {
            candle::bail!("initial lr {initial_lr} is outside [0, peak {peak_lr}]");
        }
        if !(0.0..=peak_lr).contains(&final_lr) {
            candle::bail!("final lr {final_lr} is outside [0, peak {peak_lr}]");
        }
        let warmup_steps =
            ((total_steps as f64 * warmup_frac).round() as usize).min(total_steps - 1);
        Ok(Self {
            peak_lr,
            initial_lr,
            final_lr,
            warmup_steps,
            total_steps,
            step: 0,
        })
    }
}

impl LrScheduler for OneCycleLR {
    fn lr(&mut self) -> f64 {
        let lr = if self.step < self.warmup_steps {
            self.initial_lr
                + (self.peak_lr - self.initial_lr) * (self.step + 1) as f64
                    / self.warmup_steps as f64
        } else {
            let span = (self.total_steps - self.warmup_steps) as f64;
            let progress = ((self.step - self.warmup_steps) as f64 / span).min(1.0);
            self.final_lr
                + (self.peak_lr - self.final_lr) * (1.0 + (std::f64::consts::PI * progress).cos())
                    / 2.0
        };
        self.step += 1;
        lr
    }
}
