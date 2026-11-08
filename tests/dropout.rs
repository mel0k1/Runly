use runly::prelude::*;

#[test]
fn dropout_zeros_and_scales_in_training() -> Result<()> {
    let device = Device::Cpu;
    let mut model = Model::new(&device);
    model.dropout(0.5)?;
    let x = Tensor::ones((4096, 64), DType::F32, &device)?;
    let out = model.forward(&x)?;
    let values = out.flatten_all()?.to_vec1::<f32>()?;
    let zeros = values.iter().filter(|v| **v == 0.0).count();
    let frac = zeros as f64 / values.len() as f64;
    assert!((frac - 0.5).abs() < 0.02, "zero fraction {frac}");
    let mean = values.iter().sum::<f32>() as f64 / values.len() as f64;
    assert!((mean - 1.0).abs() < 0.02, "scaled mean {mean}");
    Ok(())
}

#[test]
fn dropout_is_identity_in_eval() -> Result<()> {
    let device = Device::Cpu;
    let mut model = Model::new(&device);
    model.dropout(0.5)?;
    let x = Tensor::randn(0f32, 1f32, (16, 8), &device)?;
    let out = model.forward_eval(&x)?;
    let diff = (out - &x)?.abs()?.sum_all()?.to_scalar::<f32>()?;
    assert!(diff < 1e-6, "eval diff {diff}");
    Ok(())
}

#[test]
fn dropout_with_zero_probability_is_identity() -> Result<()> {
    let device = Device::Cpu;
    let mut model = Model::new(&device);
    model.dropout(0.0)?;
    let x = Tensor::randn(0f32, 1f32, (16, 8), &device)?;
    let out = model.forward(&x)?;
    let diff = (out - &x)?.abs()?.sum_all()?.to_scalar::<f32>()?;
    assert!(diff < 1e-6, "p=0 diff {diff}");
    Ok(())
}
