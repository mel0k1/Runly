pub mod checkpoint;
pub mod data;
pub mod fit;
pub mod gradcheck;
pub mod loss;
pub mod lr_finder;
pub mod metrics;
pub mod model;
pub mod optim;
pub mod sched;

#[cfg(feature = "mnist")]
pub mod mnist;

pub use candle;

use candle::Device;

pub fn device() -> candle::Result<Device> {
    Device::cuda_if_available(0)
}

pub mod prelude {
    pub use crate::checkpoint::{Checkpointer, EarlyStopping};
    pub use crate::data::{
        CsvConfig, Dataset, batch_indices, load_csv, load_csv_named, load_csv_with, spirals, xor,
    };
    pub use crate::device;
    pub use crate::fit::{FitConfig, fit};
    pub use crate::gradcheck::{GradCheckConfig, GradReport, check, check_model};
    pub use crate::loss::{binary_cross_entropy_with_logit, cross_entropy, huber, mse};
    pub use crate::lr_finder::{LrFinder, LrPoint, suggested_lr};
    pub use crate::metrics::{
        ClassMetrics, accuracy, argmax, confusion_matrix, macro_f1, per_class_metrics,
    };
    pub use crate::model::{
        Act, Model, adaptive_avg_pool, adaptive_max_pool, avg_pool, conv_transpose2d, embedding,
        global_avg_pool, max_pool, select_last_step,
    };
    pub use crate::optim::{
        Adam, AdamConfig, AdamW, GradNorm, Optimizer, OptimizerState, ParamsAdamW, SGD, SgdConfig,
    };
    pub use crate::sched::{
        CosineAnnealingLR, ExponentialLR, LrScheduler, OneCycleLR, StepLR, Warmup,
    };
    pub use candle::{D, DType, Device, IndexOp, Result, Tensor, Var};
}
