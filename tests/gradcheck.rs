use runly::prelude::*;

#[test]
fn cross_entropy_gradients_match_numerical() -> Result<()> {
    let device = Device::Cpu;
    let mut model = Model::new(&device);
    model.linear(3, 5)?;
    model.tanh();
    model.linear(5, 4)?;
    model.sigmoid();
    model.linear(4, 2)?;
    let x = Tensor::randn(0f32, 1f32, (6, 3), &device)?;
    let y = Tensor::from_vec(vec![0u32, 1, 0, 1, 1, 0], 6, &device)?;
    let reports = check_model(
        &model,
        &x,
        |logits| cross_entropy(logits, &y),
        GradCheckConfig::default(),
    )?;
    assert!(!reports.is_empty());
    for report in &reports {
        assert!(report.ok, "{} max_err {:.2e}", report.name, report.max_err);
    }
    Ok(())
}

#[test]
fn mse_gradients_match_numerical() -> Result<()> {
    let device = Device::Cpu;
    let mut model = Model::new(&device);
    model.linear(2, 6)?;
    model.tanh();
    model.linear(6, 1)?;
    model.sigmoid();
    let x = Tensor::randn(0f32, 1f32, (5, 2), &device)?;
    let y = Tensor::randn(0.5f32, 0.2f32, (5, 1), &device)?;
    let reports = check_model(
        &model,
        &x,
        |logits| mse(logits, &y),
        GradCheckConfig::default(),
    )?;
    for report in &reports {
        assert!(report.ok, "{} max_err {:.2e}", report.name, report.max_err);
    }
    Ok(())
}
