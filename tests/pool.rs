use runly::prelude::*;

#[test]
fn avg_pool_values_match_reference() -> Result<()> {
    let device = Device::Cpu;
    let values = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    let x = Tensor::from_vec(values, (1, 1, 4, 4), &device)?;
    let pooled = avg_pool(&x, 2)?;
    assert_eq!(pooled.dims(), [1, 1, 2, 2]);
    let got = pooled.flatten_all()?.to_vec1::<f32>()?;
    let expected = [3.5, 5.5, 11.5, 13.5];
    for (g, e) in got.iter().zip(expected) {
        assert!((g - e).abs() < 1e-6, "{g} vs {e}");
    }
    Ok(())
}

#[test]
fn avg_pool_keeps_batch_and_channels() -> Result<()> {
    let device = Device::Cpu;
    let x = Tensor::randn(0f32, 1f32, (3, 4, 6, 6), &device)?;
    let pooled = avg_pool(&x, 3)?;
    assert_eq!(pooled.dims(), [3, 4, 2, 2]);
    Ok(())
}

#[test]
fn avg_pool_rejects_indivisible_input() -> Result<()> {
    let device = Device::Cpu;
    let x = Tensor::randn(0f32, 1f32, (1, 1, 5, 4), &device)?;
    assert!(avg_pool(&x, 2).is_err());
    Ok(())
}

#[test]
fn model_with_avg_pool_round_trips() -> Result<()> {
    let device = Device::Cpu;
    let mut model = Model::new(&device);
    model.conv2d(1, 2, 3, 1, 1)?;
    model.relu();
    model.avg_pool(2);
    model.flatten();
    model.linear(2 * 2 * 2, 2)?;
    let path = std::env::temp_dir().join("runly_avg_pool_roundtrip.safetensors");
    model.save(&path)?;
    let restored = Model::load(&path, &device)?;
    let x = Tensor::randn(0f32, 1f32, (2, 1, 4, 4), &device)?;
    let a = model.forward_eval(&x)?;
    let b = restored.forward_eval(&x)?;
    let diff = (a - b)?.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(diff < 1e-5, "max diff {diff}");
    std::fs::remove_file(&path).ok();
    Ok(())
}

#[test]
fn avg_pool_network_learns_bright_half() -> Result<()> {
    let device = Device::Cpu;
    let mut pattern = Vec::new();
    let mut y_vec = Vec::new();
    for i in 0..60 {
        let left = i % 2 == 0;
        y_vec.push(if left { 0u32 } else { 1 });
        for _row in 0..8 {
            for col in 0..8 {
                let bright = if left { col < 4 } else { col >= 4 };
                pattern.push(if bright { 1.0f32 } else { -1.0 });
            }
        }
    }
    let base = Tensor::from_vec(pattern, (60, 1, 8, 8), &device)?;
    let noise = Tensor::randn(0f32, 0.3f32, (60, 1, 8, 8), &device)?;
    let x = (base + noise)?;
    let y = Tensor::from_vec(y_vec, 60, &device)?;
    let mut model = Model::new(&device);
    model.conv2d(1, 2, 3, 1, 1)?;
    model.relu();
    model.avg_pool(2);
    model.conv2d(2, 2, 3, 1, 1)?;
    model.relu();
    model.avg_pool(2);
    model.flatten();
    model.linear(2 * 2 * 2, 2)?;
    let mut opt = Adam::new_lr(model.vars(), 0.01)?;
    for _ in 0..200 {
        let logits = model.forward(&x)?;
        let loss = cross_entropy(&logits, &y)?;
        opt.backward_step(&loss)?;
    }
    let acc = accuracy(&model.forward(&x)?, &y)?;
    assert!(acc >= 0.95, "accuracy {acc}");
    Ok(())
}

#[test]
fn global_avg_pool_matches_reference() -> Result<()> {
    let device = Device::Cpu;
    let values = vec![1.0f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
    let x = Tensor::from_vec(values, (2, 1, 2, 2), &device)?;
    let pooled = global_avg_pool(&x)?;
    assert_eq!(pooled.dims(), [2, 1, 1, 1]);
    let got = pooled.flatten_all()?.to_vec1::<f32>()?;
    assert!((got[0] - 2.5).abs() < 1e-6, "{}", got[0]);
    assert!((got[1] - 25.0).abs() < 1e-6, "{}", got[1]);
    Ok(())
}

#[test]
fn global_avg_pool_accepts_any_spatial_size() -> Result<()> {
    let device = Device::Cpu;
    let small = Tensor::randn(0f32, 1f32, (2, 3, 4, 4), &device)?;
    let large = Tensor::randn(0f32, 1f32, (2, 3, 7, 9), &device)?;
    assert_eq!(global_avg_pool(&small)?.dims(), [2, 3, 1, 1]);
    assert_eq!(global_avg_pool(&large)?.dims(), [2, 3, 1, 1]);
    Ok(())
}

#[test]
fn global_avg_pool_rejects_rank_two() -> Result<()> {
    let device = Device::Cpu;
    let x = Tensor::randn(0f32, 1f32, (4, 4), &device)?;
    assert!(global_avg_pool(&x).is_err());
    Ok(())
}

#[test]
fn model_with_global_avg_pool_round_trips() -> Result<()> {
    let device = Device::Cpu;
    let mut model = Model::new(&device);
    model.conv2d(1, 2, 3, 1, 1)?;
    model.relu();
    model.global_avg_pool();
    model.flatten();
    model.linear(2, 2)?;
    let path = std::env::temp_dir().join("runly_gap_roundtrip.safetensors");
    model.save(&path)?;
    let restored = Model::load(&path, &device)?;
    let x = Tensor::randn(0f32, 1f32, (2, 1, 6, 6), &device)?;
    let diff = (model.forward_eval(&x)? - restored.forward_eval(&x)?)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    assert!(diff < 1e-5, "max diff {diff}");
    std::fs::remove_file(&path).ok();
    Ok(())
}

#[test]
fn global_avg_pool_net_learns_bright_half() -> Result<()> {
    let device = Device::Cpu;
    let mut pattern = Vec::new();
    let mut y_vec = Vec::new();
    for i in 0..60 {
        let left = i % 2 == 0;
        y_vec.push(if left { 0u32 } else { 1 });
        for _row in 0..8 {
            for col in 0..8 {
                let bright = if left { col < 4 } else { col >= 4 };
                pattern.push(if bright { 1.0f32 } else { -1.0 });
            }
        }
    }
    let base = Tensor::from_vec(pattern, (60, 1, 8, 8), &device)?;
    let noise = Tensor::randn(0f32, 0.3f32, (60, 1, 8, 8), &device)?;
    let x = (base + noise)?;
    let y = Tensor::from_vec(y_vec, 60, &device)?;
    let mut model = Model::new(&device);
    model.conv2d(1, 4, 3, 1, 1)?;
    model.relu();
    model.global_avg_pool();
    model.flatten();
    model.linear(4, 2)?;
    let mut opt = Adam::new_lr(model.vars(), 0.01)?;
    for _ in 0..200 {
        let logits = model.forward(&x)?;
        let loss = cross_entropy(&logits, &y)?;
        opt.backward_step(&loss)?;
    }
    let acc = accuracy(&model.forward(&x)?, &y)?;
    assert!(acc >= 0.95, "accuracy {acc}");
    Ok(())
}
