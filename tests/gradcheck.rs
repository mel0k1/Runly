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
fn conv_gradients_match_numerical() -> Result<()> {
    let device = Device::Cpu;
    let mut model = Model::new(&device);
    model.conv2d(1, 2, 3, 1, 1)?;
    model.tanh();
    model.flatten();
    model.linear(32, 2)?;
    let x = Tensor::randn(0f32, 1f32, (4, 1, 4, 4), &device)?;
    let y = Tensor::from_vec(vec![0u32, 1, 1, 0], 4, &device)?;
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
fn conv_transpose_gradients_match_numerical() -> Result<()> {
    let device = Device::Cpu;
    let mut model = Model::new(&device);
    model.conv_transpose2d(1, 2, 2, 2, 0)?;
    model.tanh();
    model.flatten();
    model.linear(2 * 4 * 4, 2)?;
    let x = Tensor::randn(0f32, 1f32, (3, 1, 2, 2), &device)?;
    let y = Tensor::from_vec(vec![0u32, 1, 0], 3, &device)?;
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
fn adaptive_avg_pool_gradients_match_numerical() -> Result<()> {
    let device = Device::Cpu;
    let values = vec![
        -2.0f32, 1.0, 0.5, 3.0, -1.0, 4.0, 2.0, 0.0, 5.0, -3.0, 1.5, 2.5, 0.25, 1.25, 3.5, -0.5,
    ];
    let x = Var::from_tensor(&Tensor::from_vec(values, (1, 1, 4, 4), &device)?)?;
    let reports = check(
        &[("x".to_string(), x.clone())],
        || adaptive_avg_pool(x.as_tensor(), 3)?.sqr()?.sum_all(),
        GradCheckConfig::default(),
    )?;
    for report in &reports {
        assert!(report.ok, "{} max_err {:.2e}", report.name, report.max_err);
    }
    Ok(())
}

#[test]
fn adaptive_max_pool_gradients_match_numerical() -> Result<()> {
    let device = Device::Cpu;
    let values = vec![
        -2.0f32, 1.0, 0.5, 3.0, -1.0, 4.0, 2.0, 0.0, 5.0, -3.0, 1.5, 2.5, 0.25, 1.25, 3.5, -0.5,
    ];
    let x = Var::from_tensor(&Tensor::from_vec(values, (1, 1, 4, 4), &device)?)?;
    let reports = check(
        &[("x".to_string(), x.clone())],
        || adaptive_max_pool(x.as_tensor(), 3)?.sqr()?.sum_all(),
        GradCheckConfig::default(),
    )?;
    for report in &reports {
        assert!(report.ok, "{} max_err {:.2e}", report.name, report.max_err);
    }
    Ok(())
}

#[test]
fn batch_norm_gradients_match_numerical() -> Result<()> {
    let device = Device::Cpu;
    let mut model = Model::new(&device);
    model.linear(3, 6)?;
    model.batch_norm(6)?;
    model.tanh();
    model.linear(6, 2)?;
    let x = Tensor::randn(0f32, 1f32, (8, 3), &device)?;
    let y = Tensor::from_vec(vec![0u32, 1, 0, 1, 1, 0, 0, 1], 8, &device)?;
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
fn max_pool_gradients_match_numerical() -> Result<()> {
    let device = Device::Cpu;
    let values = vec![
        -2.0f32, 1.0, 0.5, 3.0, -1.0, 4.0, 2.0, 0.0, 5.0, -3.0, 1.5, 2.5, 0.25, 1.25, 3.5, -0.5,
    ];
    let x = Var::from_tensor(&Tensor::from_vec(values, (1, 1, 4, 4), &device)?)?;
    let reports = check(
        &[("x".to_string(), x.clone())],
        || max_pool(x.as_tensor(), 2)?.sqr()?.sum_all(),
        GradCheckConfig::default(),
    )?;
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

#[test]
fn avg_pool_gradients_match_numerical() -> Result<()> {
    let device = Device::Cpu;
    let values = vec![
        -2.0f32, 1.0, 0.5, 3.0, -1.0, 4.0, 2.0, 0.0, 5.0, -3.0, 1.5, 2.5, 0.25, 1.25, 3.5, -0.5,
    ];
    let x = Var::from_tensor(&Tensor::from_vec(values, (1, 1, 4, 4), &device)?)?;
    let reports = check(
        &[("x".to_string(), x.clone())],
        || avg_pool(x.as_tensor(), 2)?.sqr()?.sum_all(),
        GradCheckConfig::default(),
    )?;
    for report in &reports {
        assert!(report.ok, "{} max_err {:.2e}", report.name, report.max_err);
    }
    Ok(())
}

#[test]
fn global_avg_pool_gradients_match_numerical() -> Result<()> {
    let device = Device::Cpu;
    let values = vec![
        -2.0f32, 1.0, 0.5, 3.0, -1.0, 4.0, 2.0, 0.0, 5.0, -3.0, 1.5, 2.5, 0.25, 1.25, 3.5, -0.5,
    ];
    let x = Var::from_tensor(&Tensor::from_vec(values, (1, 1, 4, 4), &device)?)?;
    let reports = check(
        &[("x".to_string(), x.clone())],
        || global_avg_pool(x.as_tensor())?.sqr()?.sum_all(),
        GradCheckConfig::default(),
    )?;
    for report in &reports {
        assert!(report.ok, "{} max_err {:.2e}", report.name, report.max_err);
    }
    Ok(())
}
