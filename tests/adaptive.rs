use runly::prelude::*;

#[test]
fn adaptive_avg_pool_matches_reference() -> Result<()> {
    let device = Device::Cpu;
    let values: Vec<f32> = (1..=16).map(|v| v as f32).collect();
    let x = Tensor::from_vec(values, (1, 1, 4, 4), &device)?;
    let pooled = adaptive_avg_pool(&x, 2)?;
    assert_eq!(pooled.dims(), [1, 1, 2, 2]);
    let got = pooled.flatten_all()?.to_vec1::<f32>()?;
    let expected = [3.5, 5.5, 11.5, 13.5];
    for (g, e) in got.iter().zip(expected) {
        assert!((g - e).abs() < 1e-6, "{g} vs {e}");
    }
    Ok(())
}

#[test]
fn adaptive_max_pool_matches_reference() -> Result<()> {
    let device = Device::Cpu;
    let values: Vec<f32> = (1..=16).map(|v| v as f32).collect();
    let x = Tensor::from_vec(values, (1, 1, 4, 4), &device)?;
    let pooled = adaptive_max_pool(&x, 2)?;
    assert_eq!(pooled.dims(), [1, 1, 2, 2]);
    let got = pooled.flatten_all()?.to_vec1::<f32>()?;
    let expected = [6.0, 8.0, 14.0, 16.0];
    for (g, e) in got.iter().zip(expected) {
        assert!((g - e).abs() < 1e-6, "{g} vs {e}");
    }
    Ok(())
}

#[test]
fn adaptive_pools_handle_indivisible_input() -> Result<()> {
    let device = Device::Cpu;
    let values: Vec<f32> = (1..=9).map(|v| v as f32).collect();
    let x = Tensor::from_vec(values, (1, 1, 3, 3), &device)?;
    let avg = adaptive_avg_pool(&x, 2)?;
    let avg = avg.flatten_all()?.to_vec1::<f32>()?;
    let expected = [3.0, 4.0, 6.0, 7.0];
    for (g, e) in avg.iter().zip(expected) {
        assert!((g - e).abs() < 1e-6, "avg {g} vs {e}");
    }
    let max = adaptive_max_pool(&x, 2)?;
    let max = max.flatten_all()?.to_vec1::<f32>()?;
    let expected = [5.0, 6.0, 8.0, 9.0];
    for (g, e) in max.iter().zip(expected) {
        assert!((g - e).abs() < 1e-6, "max {g} vs {e}");
    }
    Ok(())
}

#[test]
fn adaptive_pools_upscale_small_inputs() -> Result<()> {
    let device = Device::Cpu;
    let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 1, 2, 2), &device)?;
    let pooled = adaptive_avg_pool(&x, 4)?;
    assert_eq!(pooled.dims(), [1, 1, 4, 4]);
    let got = pooled.flatten_all()?.to_vec1::<f32>()?;
    let expected = [
        1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0,
    ];
    for (g, e) in got.iter().zip(expected) {
        assert!((g - e).abs() < 1e-6, "{g} vs {e}");
    }
    Ok(())
}

#[test]
fn adaptive_pools_reject_zero_output() -> Result<()> {
    let device = Device::Cpu;
    let x = Tensor::randn(0f32, 1f32, (1, 1, 4, 4), &device)?;
    assert!(adaptive_avg_pool(&x, 0).is_err());
    assert!(adaptive_max_pool(&x, 0).is_err());
    Ok(())
}

#[test]
fn adaptive_pools_keep_batch_and_channels() -> Result<()> {
    let device = Device::Cpu;
    let x = Tensor::randn(0f32, 1f32, (3, 4, 7, 9), &device)?;
    assert_eq!(adaptive_avg_pool(&x, 3)?.dims(), [3, 4, 3, 3]);
    assert_eq!(adaptive_max_pool(&x, 3)?.dims(), [3, 4, 3, 3]);
    assert_eq!(adaptive_avg_pool(&x, 1)?.dims(), [3, 4, 1, 1]);
    assert_eq!(adaptive_max_pool(&x, 1)?.dims(), [3, 4, 1, 1]);
    Ok(())
}

#[test]
fn model_with_adaptive_pool_round_trips() -> Result<()> {
    let device = Device::Cpu;
    let mut model = Model::new(&device);
    model.conv2d(1, 2, 3, 1, 1)?;
    model.relu();
    model.adaptive_max_pool(3);
    model.flatten();
    model.linear(2 * 3 * 3, 2)?;
    let path = std::env::temp_dir().join("runly_adaptive_roundtrip.safetensors");
    model.save(&path)?;
    let restored = Model::load(&path, &device)?;
    let x = Tensor::randn(0f32, 1f32, (2, 1, 5, 7), &device)?;
    let diff = (model.forward_eval(&x)? - restored.forward_eval(&x)?)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    assert!(diff < 1e-5, "max diff {diff}");
    std::fs::remove_file(&path).ok();
    Ok(())
}

#[test]
fn adaptive_pool_accepts_variable_input_sizes() -> Result<()> {
    let device = Device::Cpu;
    let mut model = Model::new(&device);
    model.conv2d(1, 2, 3, 1, 1)?;
    model.relu();
    model.adaptive_avg_pool(2);
    model.flatten();
    model.linear(8, 2)?;
    let small = Tensor::randn(0f32, 1f32, (2, 1, 4, 4), &device)?;
    let large = Tensor::randn(0f32, 1f32, (2, 1, 8, 8), &device)?;
    assert_eq!(model.forward_eval(&small)?.dims(), [2, 2]);
    assert_eq!(model.forward_eval(&large)?.dims(), [2, 2]);
    Ok(())
}
