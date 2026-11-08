use candle::{D, DType, Result, Tensor};

pub fn argmax(logits: &Tensor) -> Result<Tensor> {
    logits.argmax(D::Minus1)
}

pub fn accuracy(logits: &Tensor, targets: &Tensor) -> Result<f64> {
    let hits = logits
        .argmax(D::Minus1)?
        .eq(targets)?
        .to_dtype(DType::F64)?
        .sum_all()?
        .to_scalar::<f64>()?;
    Ok(hits / targets.dims()[0] as f64)
}
