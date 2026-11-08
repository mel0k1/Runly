use candle::backprop::GradStore;
use candle::{Result, Var};
pub use candle_nn::optim::Optimizer;

#[derive(Debug, Clone, Copy)]
pub struct AdamConfig {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub max_grad_norm: Option<f64>,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            max_grad_norm: None,
        }
    }
}

fn validate_max_grad_norm(max_grad_norm: Option<f64>) -> Result<()> {
    if let Some(max_norm) = max_grad_norm
        && max_norm <= 0.0
    {
        candle::bail!("max_grad_norm {max_norm} must be positive");
    }
    Ok(())
}

fn grad_norm_of(grads: &GradStore, vars: &[&Var]) -> Result<f64> {
    let mut total = 0.0f64;
    for var in vars {
        if let Some(g) = grads.get(var.as_tensor()) {
            total += g.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
        }
    }
    Ok(total.sqrt())
}

fn clip_scale(norm: f64, max_norm: f64) -> Option<f64> {
    if norm > max_norm {
        Some(max_norm / norm)
    } else {
        None
    }
}

#[derive(Debug)]
struct AdamState {
    var: Var,
    m: Var,
    v: Var,
}

#[derive(Debug)]
pub struct Adam {
    states: Vec<AdamState>,
    step_t: usize,
    config: AdamConfig,
    last_norm: Option<f64>,
}

impl Optimizer for Adam {
    type Config = AdamConfig;

    fn new(vars: Vec<Var>, config: AdamConfig) -> Result<Self> {
        validate_max_grad_norm(config.max_grad_norm)?;
        let states = vars
            .into_iter()
            .filter(|var| var.dtype().is_float())
            .map(|var| {
                let m = Var::zeros(var.shape(), var.dtype(), var.device())?;
                let v = Var::zeros(var.shape(), var.dtype(), var.device())?;
                Ok(AdamState { var, m, v })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            states,
            step_t: 0,
            config,
            last_norm: None,
        })
    }

    fn step(&mut self, grads: &GradStore) -> Result<()> {
        self.step_t += 1;
        let c = self.config;
        let scale_m = 1f64 / (1f64 - c.beta1.powi(self.step_t as i32));
        let scale_v = 1f64 / (1f64 - c.beta2.powi(self.step_t as i32));
        let clip = match c.max_grad_norm {
            None => None,
            Some(max_norm) => {
                let vars: Vec<&Var> = self.states.iter().map(|s| &s.var).collect();
                let norm = grad_norm_of(grads, &vars)?;
                self.last_norm = Some(norm);
                clip_scale(norm, max_norm)
            }
        };
        for state in &mut self.states {
            if let Some(g) = grads.get(state.var.as_tensor()) {
                let mut g = g.clone();
                if let Some(clip) = clip {
                    g = (&g * clip)?;
                }
                if c.weight_decay != 0.0 {
                    g = (&g + (state.var.as_tensor() * c.weight_decay)?)?;
                }
                let next_m = ((state.m.as_tensor() * c.beta1)? + (&g * (1f64 - c.beta1))?)?;
                let next_v = ((state.v.as_tensor() * c.beta2)? + (g.sqr()? * (1f64 - c.beta2))?)?;
                let m_hat = &next_m * scale_m;
                let v_hat = &next_v * scale_v;
                let denom = (v_hat?.sqrt()? + c.eps)?;
                let update = ((m_hat? / &denom)? * c.lr)?;
                let next_theta = (state.var.as_tensor() - &update)?;
                state.m.set(&next_m)?;
                state.v.set(&next_v)?;
                state.var.set(&next_theta)?;
            }
        }
        Ok(())
    }

    fn learning_rate(&self) -> f64 {
        self.config.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.config.lr = lr;
    }
}

impl Adam {
    pub fn new_lr(vars: Vec<Var>, lr: f64) -> Result<Self> {
        Self::new(
            vars,
            AdamConfig {
                lr,
                ..Default::default()
            },
        )
    }

    pub fn set_max_grad_norm(&mut self, max_norm: Option<f64>) {
        self.config.max_grad_norm = max_norm;
    }

    pub fn last_grad_norm(&self) -> Option<f64> {
        self.last_norm
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SgdConfig {
    pub lr: f64,
    pub momentum: f64,
    pub dampening: f64,
    pub weight_decay: f64,
    pub max_grad_norm: Option<f64>,
}

impl Default for SgdConfig {
    fn default() -> Self {
        Self {
            lr: 0.01,
            momentum: 0.9,
            dampening: 0.0,
            weight_decay: 0.0,
            max_grad_norm: None,
        }
    }
}

#[derive(Debug)]
struct SgdState {
    var: Var,
    buf: Option<Var>,
}

#[derive(Debug)]
pub struct SGD {
    states: Vec<SgdState>,
    config: SgdConfig,
    last_norm: Option<f64>,
}

impl Optimizer for SGD {
    type Config = SgdConfig;

    fn new(vars: Vec<Var>, config: SgdConfig) -> Result<Self> {
        validate_max_grad_norm(config.max_grad_norm)?;
        let states = vars
            .into_iter()
            .filter(|var| var.dtype().is_float())
            .map(|var| SgdState { var, buf: None })
            .collect();
        Ok(Self {
            states,
            config,
            last_norm: None,
        })
    }

    fn step(&mut self, grads: &GradStore) -> Result<()> {
        let c = self.config;
        let clip = match c.max_grad_norm {
            None => None,
            Some(max_norm) => {
                let vars: Vec<&Var> = self.states.iter().map(|s| &s.var).collect();
                let norm = grad_norm_of(grads, &vars)?;
                self.last_norm = Some(norm);
                clip_scale(norm, max_norm)
            }
        };
        for state in &mut self.states {
            if let Some(g) = grads.get(state.var.as_tensor()) {
                let mut g = g.clone();
                if let Some(clip) = clip {
                    g = (&g * clip)?;
                }
                if c.weight_decay != 0.0 {
                    g = (&g + (state.var.as_tensor() * c.weight_decay)?)?;
                }
                let buf = match &state.buf {
                    Some(b) => ((b.as_tensor() * c.momentum)? + (&g * (1f64 - c.dampening))?)?,
                    None => g.clone(),
                };
                let next = (state.var.as_tensor() - (&buf * c.lr)?)?;
                state.buf = Some(Var::from_tensor(&buf)?);
                state.var.set(&next)?;
            }
        }
        Ok(())
    }

    fn learning_rate(&self) -> f64 {
        self.config.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.config.lr = lr;
    }
}

impl SGD {
    pub fn new_lr(vars: Vec<Var>, lr: f64) -> Result<Self> {
        Self::new(
            vars,
            SgdConfig {
                lr,
                ..Default::default()
            },
        )
    }

    pub fn set_max_grad_norm(&mut self, max_norm: Option<f64>) {
        self.config.max_grad_norm = max_norm;
    }

    pub fn last_grad_norm(&self) -> Option<f64> {
        self.last_norm
    }
}

pub use candle_nn::optim::{AdamW, ParamsAdamW};
