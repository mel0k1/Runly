use crate::model::Model;
use crate::optim::OptimizerState;
use candle::Result;
use std::path::{Path, PathBuf};

pub struct Checkpointer {
    path: PathBuf,
    best: Option<f64>,
}

impl Checkpointer {
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            best: None,
        }
    }

    pub fn update(&mut self, model: &Model, metric: f64) -> Result<bool> {
        if self.best.is_none_or(|best| metric < best) {
            model.save(&self.path)?;
            self.best = Some(metric);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub fn update_with_optimizer<O: OptimizerState>(
        &mut self,
        model: &Model,
        opt: &O,
        metric: f64,
    ) -> Result<bool> {
        if self.best.is_none_or(|best| metric < best) {
            model.save(&self.path)?;
            let mut tensors = candle::safetensors::load(&self.path, model.device())?;
            for (name, tensor) in opt.state_tensors()? {
                tensors.insert(name, tensor);
            }
            candle::safetensors::save(&tensors, &self.path)?;
            self.best = Some(metric);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub fn best(&self) -> Option<f64> {
        self.best
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

pub struct EarlyStopping {
    checkpointer: Checkpointer,
    patience: usize,
    min_delta: f64,
    bad_epochs: usize,
    stopped: bool,
}

impl EarlyStopping {
    pub fn new<P: AsRef<Path>>(path: P, patience: usize, min_delta: f64) -> Result<Self> {
        if patience == 0 {
            candle::bail!("patience must be positive");
        }
        if min_delta < 0.0 {
            candle::bail!("min_delta {min_delta} is negative");
        }
        Ok(Self {
            checkpointer: Checkpointer::new(path),
            patience,
            min_delta,
            bad_epochs: 0,
            stopped: false,
        })
    }

    pub fn update(&mut self, model: &Model, metric: f64) -> Result<bool> {
        if self.stopped {
            return Ok(true);
        }
        let improved = match self.checkpointer.best() {
            None => true,
            Some(best) => metric < best - self.min_delta,
        };
        if improved {
            self.checkpointer.update(model, metric)?;
            self.bad_epochs = 0;
        } else {
            self.bad_epochs += 1;
            if self.bad_epochs >= self.patience {
                self.stopped = true;
            }
        }
        Ok(self.stopped)
    }

    pub fn update_with_optimizer<O: OptimizerState>(
        &mut self,
        model: &Model,
        opt: &O,
        metric: f64,
    ) -> Result<bool> {
        if self.stopped {
            return Ok(true);
        }
        let improved = match self.checkpointer.best() {
            None => true,
            Some(best) => metric < best - self.min_delta,
        };
        if improved {
            self.checkpointer
                .update_with_optimizer(model, opt, metric)?;
            self.bad_epochs = 0;
        } else {
            self.bad_epochs += 1;
            if self.bad_epochs >= self.patience {
                self.stopped = true;
            }
        }
        Ok(self.stopped)
    }

    pub fn best(&self) -> Option<f64> {
        self.checkpointer.best()
    }

    pub fn bad_epochs(&self) -> usize {
        self.bad_epochs
    }

    pub fn stopped(&self) -> bool {
        self.stopped
    }

    pub fn checkpointer(&self) -> &Checkpointer {
        &self.checkpointer
    }

    pub fn path(&self) -> &Path {
        self.checkpointer.path()
    }
}
