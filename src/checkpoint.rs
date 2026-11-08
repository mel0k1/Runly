use crate::model::Model;
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

    pub fn best(&self) -> Option<f64> {
        self.best
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}
