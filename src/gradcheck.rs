use candle::{DType, Result, Tensor, Var};

#[derive(Debug, Clone, Copy)]
pub struct GradCheckConfig {
    pub eps: f64,
    pub atol: f64,
    pub rtol: f64,
    pub max_per_var: usize,
}

impl Default for GradCheckConfig {
    fn default() -> Self {
        Self {
            eps: 2e-3,
            atol: 2e-4,
            rtol: 1e-2,
            max_per_var: 64,
        }
    }
}

#[derive(Debug)]
pub struct GradReport {
    pub name: String,
    pub checked: usize,
    pub max_err: f64,
    pub ok: bool,
}

pub fn check<F>(
    vars: &[(String, Var)],
    mut loss: F,
    config: GradCheckConfig,
) -> Result<Vec<GradReport>>
where
    F: FnMut() -> Result<Tensor>,
{
    let grads = loss()?.backward()?;
    let mut reports = Vec::new();
    for (name, var) in vars {
        let name = name.clone();
        let analytic = grads
            .get(var.as_tensor())
            .ok_or_else(|| candle::Error::Msg(format!("no gradient for {name}")))?
            .to_dtype(DType::F64)?
            .flatten_all()?
            .to_vec1::<f64>()?;
        let base = var.as_tensor().flatten_all()?.to_vec1::<f32>()?;
        let shape = var.shape().clone();
        let device = var.device();
        let total = base.len();
        let step = if total <= config.max_per_var {
            1
        } else {
            total / config.max_per_var
        };
        let mut max_err = 0.0f64;
        let mut bad = 0usize;
        let mut checked = 0usize;
        for index in (0..total).step_by(step) {
            let mut plus = base.clone();
            plus[index] += config.eps as f32;
            var.set(&Tensor::from_vec(plus, &shape, device)?)?;
            let lp = loss()?.to_dtype(DType::F64)?.to_scalar::<f64>()?;
            let mut minus = base.clone();
            minus[index] -= config.eps as f32;
            var.set(&Tensor::from_vec(minus, &shape, device)?)?;
            let lm = loss()?.to_dtype(DType::F64)?.to_scalar::<f64>()?;
            var.set(&Tensor::from_vec(base.clone(), &shape, device)?)?;
            let numeric = (lp - lm) / (2.0 * config.eps);
            let err = (numeric - analytic[index]).abs();
            if err > max_err {
                max_err = err;
            }
            if err > config.atol + config.rtol * analytic[index].abs() {
                bad += 1;
            }
            checked += 1;
        }
        reports.push(GradReport {
            name,
            checked,
            max_err,
            ok: bad == 0,
        });
    }
    Ok(reports)
}

pub fn check_model<L>(
    model: &crate::model::Model,
    x: &Tensor,
    loss: L,
    config: GradCheckConfig,
) -> Result<Vec<GradReport>>
where
    L: Fn(&Tensor) -> Result<Tensor>,
{
    let vars: Vec<(String, Var)> = {
        let data = model.varmap().data().lock().expect("varmap poisoned");
        let mut vars: Vec<(String, Var)> = data
            .iter()
            .map(|(name, var)| (name.clone(), var.clone()))
            .collect();
        vars.sort_by(|a, b| a.0.cmp(&b.0));
        vars
    };
    check(&vars, || loss(&model.forward(x)?), config)
}
