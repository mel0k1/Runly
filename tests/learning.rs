use runly::prelude::*;

#[test]
fn sgd_momentum_steps_match_reference() -> Result<()> {
    let device = Device::Cpu;
    let w = Var::new(1.0f32, &device)?;
    let mut opt = SGD::new_lr(vec![w.clone()], 0.1)?;
    for step in 0..2 {
        let loss = (w.as_tensor() - 3f64)?.sqr()?;
        let grads = loss.backward()?;
        opt.step(&grads)?;
        let expected = if step == 0 { 1.4 } else { 2.08 };
        let got = w.as_tensor().to_scalar::<f32>()? as f64;
        assert!(
            (got - expected).abs() < 1e-5,
            "step {step}: {got} vs {expected}"
        );
    }
    Ok(())
}

#[test]
fn adam_step_matches_reference() -> Result<()> {
    let device = Device::Cpu;
    let w = Var::new(1.0f32, &device)?;
    let mut opt = Adam::new_lr(vec![w.clone()], 0.05)?;
    let loss = (w.as_tensor() - 3f64)?.sqr()?;
    let grads = loss.backward()?;
    opt.step(&grads)?;
    let got = w.as_tensor().to_scalar::<f32>()? as f64;
    let expected = 1.0 + 0.05 * (4.0 / (4.0 + 1e-8));
    assert!((got - expected).abs() < 1e-6, "{got} vs {expected}");
    Ok(())
}

#[test]
fn xor_is_learned() -> Result<()> {
    let device = Device::Cpu;
    let data = xor(32, 0.12, 11);
    let (x, y) = data.tensors(&device)?;
    let mut model = Model::new(&device);
    model.linear(2, 8)?;
    model.tanh();
    model.linear(8, 2)?;
    let mut opt = Adam::new_lr(model.vars(), 0.05)?;
    let mut loss = f64::INFINITY;
    for _ in 0..500 {
        let logits = model.forward(&x)?;
        let l = cross_entropy(&logits, &y)?;
        opt.backward_step(&l)?;
        loss = l.to_dtype(DType::F64)?.to_scalar::<f64>()?;
    }
    let acc = accuracy(&model.forward(&x)?, &y)?;
    assert!(loss < 0.1, "final loss {loss}");
    assert!(acc >= 0.97, "accuracy {acc}");
    Ok(())
}

#[test]
fn spirals_is_learned() -> Result<()> {
    let device = Device::Cpu;
    let data = spirals(150, 1.75, 0.2, 3);
    let (train, test) = data.train_test_split(0.2, 5);
    let (x, y) = train.tensors(&device)?;
    let (xt, yt) = test.tensors(&device)?;
    let mut model = Model::new(&device);
    model.linear(2, 48)?;
    model.relu();
    model.linear(48, 48)?;
    model.relu();
    model.linear(48, 2)?;
    let mut opt = Adam::new_lr(model.vars(), 0.008)?;
    for _ in 0..800 {
        let logits = model.forward(&x)?;
        let loss = cross_entropy(&logits, &y)?;
        opt.backward_step(&loss)?;
    }
    let train_acc = accuracy(&model.forward(&x)?, &y)?;
    let test_acc = accuracy(&model.forward(&xt)?, &yt)?;
    assert!(train_acc >= 0.92, "train {train_acc}");
    assert!(test_acc >= 0.8, "test {test_acc}");
    Ok(())
}

#[test]
fn save_load_roundtrip_preserves_predictions() -> Result<()> {
    let device = Device::Cpu;
    let data = xor(16, 0.1, 2);
    let (x, y) = data.tensors(&device)?;
    let mut model = Model::new(&device);
    model.linear(2, 8)?;
    model.tanh();
    model.linear(8, 2)?;
    let mut opt = Adam::new_lr(model.vars(), 0.05)?;
    for _ in 0..200 {
        let logits = model.forward(&x)?;
        let loss = cross_entropy(&logits, &y)?;
        opt.backward_step(&loss)?;
    }
    let path = std::env::temp_dir().join(format!("runly_test_{}.safetensors", std::process::id()));
    model.save(&path)?;
    let loaded = Model::load(&path, &device)?;
    let a = model.forward(&x)?;
    let b = loaded.forward(&x)?;
    let diff = (a - &b)?.abs()?.sum_all()?.to_scalar::<f32>()?;
    assert!(diff < 1e-4, "predictions diff {diff}");
    let acc = accuracy(&b, &y)?;
    assert!(acc >= 0.95, "loaded accuracy {acc}");
    std::fs::remove_file(&path)?;
    Ok(())
}

#[test]
fn batch_norm_trains_and_eval_matches() -> Result<()> {
    let device = Device::Cpu;
    let data = xor(64, 0.15, 42);
    let (x, y) = data.tensors(&device)?;
    let mut model = Model::new(&device);
    model.linear(2, 16)?;
    model.batch_norm(16)?;
    model.tanh();
    model.linear(16, 2)?;
    let mut opt = Adam::new_lr(model.vars(), 0.02)?;
    for _ in 0..300 {
        let logits = model.forward(&x)?;
        let loss = cross_entropy(&logits, &y)?;
        opt.backward_step(&loss)?;
    }
    let acc = accuracy(&model.forward(&x)?, &y)?;
    assert!(acc >= 0.97, "train accuracy {acc}");
    let eval_acc = accuracy(&model.forward_eval(&x)?, &y)?;
    assert!(eval_acc >= 0.9, "eval accuracy {eval_acc}");
    let path = std::env::temp_dir().join(format!("runly_bn_{}.safetensors", std::process::id()));
    model.save(&path)?;
    let loaded = Model::load(&path, &device)?;
    let diff = (model.forward_eval(&x)? - loaded.forward_eval(&x)?)?
        .abs()?
        .sum_all()?
        .to_scalar::<f32>()?;
    assert!(diff < 1e-4, "eval predictions diff {diff}");
    std::fs::remove_file(&path)?;
    Ok(())
}

fn block_images() -> Result<(Tensor, Tensor)> {
    let device = Device::Cpu;
    let n = 60usize;
    let mut x = Vec::with_capacity(n * 64);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let class = (i % 2) as u32;
        for r in 0..8usize {
            for c in 0..8usize {
                let block = if (r < 3 && c < 3 && class == 0) || (r >= 5 && c >= 5 && class == 1) {
                    1.0f32
                } else {
                    0.0
                };
                let noise = ((i * 89 + r * 31 + c * 17) % 13) as f32 - 6.0;
                x.push(block + noise / 30.0);
            }
        }
        y.push(class);
    }
    let x = Tensor::from_vec(x, (n, 1, 8, 8), &device)?;
    let y = Tensor::from_vec(y, n, &device)?;
    Ok((x, y))
}

#[test]
fn conv_model_learns_and_roundtrips() -> Result<()> {
    let device = Device::Cpu;
    let (x, y) = block_images()?;
    let mut model = Model::new(&device);
    model.conv2d(1, 4, 3, 1, 0)?;
    model.batch_norm(4)?;
    model.relu();
    model.max_pool(2);
    model.flatten();
    model.linear(4 * 3 * 3, 2)?;
    assert_eq!(model.num_params(), 4 * 3 * 3 + 4 + 16 + 2 * 4 * 3 * 3 + 2);
    let mut opt = Adam::new_lr(model.vars(), 0.01)?;
    for _ in 0..200 {
        let logits = model.forward(&x)?;
        let loss = cross_entropy(&logits, &y)?;
        opt.backward_step(&loss)?;
    }
    let acc = accuracy(&model.forward(&x)?, &y)?;
    assert!(acc >= 0.95, "accuracy {acc}");
    let path = std::env::temp_dir().join(format!("runly_conv_{}.safetensors", std::process::id()));
    model.save(&path)?;
    let loaded = Model::load(&path, &device)?;
    let diff = (model.forward(&x)? - loaded.forward(&x)?)?
        .abs()?
        .sum_all()?
        .to_scalar::<f32>()?;
    assert!(diff < 1e-4, "predictions diff {diff}");
    let acc = accuracy(&loaded.forward(&x)?, &y)?;
    assert!(acc >= 0.95, "loaded accuracy {acc}");
    std::fs::remove_file(&path)?;
    Ok(())
}

#[test]
fn half_precision_roundtrip_keeps_predictions() -> Result<()> {
    let device = Device::Cpu;
    let data = xor(16, 0.1, 2);
    let (x, y) = data.tensors(&device)?;
    let mut model = Model::new(&device);
    model.linear(2, 128)?;
    model.tanh();
    model.linear(128, 2)?;
    let mut opt = Adam::new_lr(model.vars(), 0.05)?;
    for _ in 0..200 {
        let logits = model.forward(&x)?;
        let loss = cross_entropy(&logits, &y)?;
        opt.backward_step(&loss)?;
    }
    let full = std::env::temp_dir().join(format!("runly_f32_{}.safetensors", std::process::id()));
    let half = std::env::temp_dir().join(format!("runly_f16_{}.safetensors", std::process::id()));
    model.save(&full)?;
    model.save_half(&half)?;
    let full_size = std::fs::metadata(&full)?.len();
    let half_size = std::fs::metadata(&half)?.len();
    assert!(
        half_size * 4 < full_size * 3,
        "f16 file {half_size} bytes vs f32 file {full_size} bytes"
    );
    let loaded = Model::load(&half, &device)?;
    let diff = (model.forward(&x)? - loaded.forward(&x)?)?
        .abs()?
        .sum_all()?
        .to_scalar::<f32>()?;
    assert!(diff < 5.0, "predictions diff {diff}");
    let acc = accuracy(&loaded.forward(&x)?, &y)?;
    assert!(acc >= 0.95, "loaded accuracy {acc}");
    std::fs::remove_file(&full)?;
    std::fs::remove_file(&half)?;
    Ok(())
}

#[test]
fn dataset_helpers_behave() {
    let d = xor(8, 0.1, 1);
    assert_eq!(d.len(), 32);
    assert_eq!(d.dim, 2);
    assert_eq!(d.num_classes, 2);
    let s = spirals(10, 1.75, 0.0, 1);
    assert_eq!(s.len(), 20);
    let (train, test) = d.train_test_split(0.25, 1);
    assert_eq!(train.len() + test.len(), 32);
    let batches = batch_indices(10, 3, 1);
    assert_eq!(batches.iter().map(|b| b.len()).sum::<usize>(), 10);
    assert_eq!(batches.len(), 4);
}

#[test]
fn adam_weight_decay_matches_reference() -> Result<()> {
    let device = Device::Cpu;
    let w = Var::new(1.0f32, &device)?;
    let config = AdamConfig {
        lr: 0.05,
        weight_decay: 0.1,
        ..Default::default()
    };
    let mut opt = Adam::new(vec![w.clone()], config)?;
    let loss = (w.as_tensor() - 3f64)?.sqr()?;
    let grads = loss.backward()?;
    opt.step(&grads)?;
    let got = w.as_tensor().to_scalar::<f32>()? as f64;
    let g: f64 = -4.0 + 0.1;
    let m_hat = 0.1 * g / (1.0 - 0.9);
    let v_hat = 0.001 * g * g / (1.0 - 0.999);
    let expected = 1.0 - 0.05 * (m_hat / (v_hat.sqrt() + 1e-8));
    assert!((got - expected).abs() < 1e-5, "{got} vs {expected}");
    Ok(())
}

#[test]
fn adam_weight_decay_pulls_weight_to_zero() -> Result<()> {
    let device = Device::Cpu;
    let w = Var::new(0.5f32, &device)?;
    let config = AdamConfig {
        lr: 0.01,
        weight_decay: 1.0,
        ..Default::default()
    };
    let mut opt = Adam::new(vec![w.clone()], config)?;
    let loss = w.as_tensor().sqr()?;
    for _ in 0..50 {
        let grads = loss.backward()?;
        opt.step(&grads)?;
    }
    let got = w.as_tensor().to_scalar::<f32>()? as f64;
    assert!(got < 0.5, "weight {got} did not shrink");
    Ok(())
}
