use candle::{Result, Tensor};

pub fn mse(inp: &Tensor, target: &Tensor) -> Result<Tensor> {
    candle_nn::loss::mse(inp, target)
}

pub fn cross_entropy(inp: &Tensor, target: &Tensor) -> Result<Tensor> {
    candle_nn::loss::cross_entropy(inp, target)
}

pub fn binary_cross_entropy_with_logit(inp: &Tensor, target: &Tensor) -> Result<Tensor> {
    candle_nn::loss::binary_cross_entropy_with_logit(inp, target)
}

pub fn huber(inp: &Tensor, target: &Tensor, delta: f64) -> Result<Tensor> {
    candle_nn::loss::huber(inp, target, delta)
}
