pub mod data;
pub mod gradcheck;
pub mod loss;
pub mod metrics;
pub mod model;
pub mod optim;

#[cfg(feature = "mnist")]
pub mod mnist;

pub use candle;

pub mod prelude {
    pub use crate::data::{Dataset, batch_indices, spirals, xor};
    pub use crate::gradcheck::{GradCheckConfig, GradReport, check_model};
    pub use crate::loss::{binary_cross_entropy_with_logit, cross_entropy, huber, mse};
    pub use crate::metrics::{accuracy, argmax};
    pub use crate::model::{Act, Model};
    pub use crate::optim::{Adam, AdamConfig, AdamW, Optimizer, ParamsAdamW, SGD, SgdConfig};
    pub use candle::{D, DType, Device, Result, Tensor, Var};
}
