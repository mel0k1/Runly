use runly::prelude::*;

#[test]
fn adam_state_round_trips() -> Result<()> {
    let device = Device::Cpu;
    let loss = |w: &Var| (w.as_tensor() - 3f64)?.sqr();
    let w_ref = Var::new(1.0f32, &device)?;
    let mut a = Adam::new_lr(vec![w_ref.clone()], 0.05)?;
    for _ in 0..4 {
        let grads = loss(&w_ref)?.backward()?;
        a.step(&grads)?;
    }
    let target = w_ref.as_tensor().to_scalar::<f32>()? as f64;

    let w = Var::new(1.0f32, &device)?;
    let mut b = Adam::new_lr(vec![w.clone()], 0.05)?;
    for _ in 0..3 {
        let grads = loss(&w)?.backward()?;
        b.step(&grads)?;
    }
    let path = std::env::temp_dir().join("runly_adam_state.safetensors");
    b.save_state(&path)?;
    let mut c = Adam::new_lr(vec![w.clone()], 0.05)?;
    c.load_state(&path)?;
    let grads = loss(&w)?.backward()?;
    c.step(&grads)?;
    let got = w.as_tensor().to_scalar::<f32>()? as f64;
    assert!((got - target).abs() < 1e-6, "{got} vs {target}");
    std::fs::remove_file(&path).ok();
    Ok(())
}

#[test]
fn adam_state_restores_config() -> Result<()> {
    let device = Device::Cpu;
    let w = Var::new(1.0f32, &device)?;
    let config = AdamConfig {
        lr: 0.05,
        weight_decay: 0.1,
        max_grad_norm: Some(0.5),
        ..Default::default()
    };
    let a = Adam::new(vec![w.clone()], config)?;
    let path = std::env::temp_dir().join("runly_adam_config.safetensors");
    a.save_state(&path)?;
    let mut b = Adam::new(vec![w.clone()], AdamConfig::default())?;
    b.load_state(&path)?;
    assert!((b.learning_rate() - 0.05).abs() < 1e-12);
    let c = b.config();
    assert!((c.weight_decay - 0.1).abs() < 1e-12);
    assert!((c.beta1 - 0.9).abs() < 1e-12);
    assert_eq!(c.max_grad_norm, Some(0.5));
    std::fs::remove_file(&path).ok();
    Ok(())
}

#[test]
fn adam_state_restores_none_clipping() -> Result<()> {
    let device = Device::Cpu;
    let w = Var::new(1.0f32, &device)?;
    let a = Adam::new_lr(vec![w.clone()], 0.05)?;
    let path = std::env::temp_dir().join("runly_adam_none.safetensors");
    a.save_state(&path)?;
    let mut b = Adam::new(
        vec![w.clone()],
        AdamConfig {
            max_grad_norm: Some(1.0),
            ..Default::default()
        },
    )?;
    b.load_state(&path)?;
    assert_eq!(b.config().max_grad_norm, None);
    std::fs::remove_file(&path).ok();
    Ok(())
}

#[test]
fn sgd_state_round_trips() -> Result<()> {
    let device = Device::Cpu;
    let loss = |w: &Var| (w.as_tensor() - 3f64)?.sqr();
    let w_ref = Var::new(1.0f32, &device)?;
    let mut a = SGD::new_lr(vec![w_ref.clone()], 0.1)?;
    for _ in 0..3 {
        let grads = loss(&w_ref)?.backward()?;
        a.step(&grads)?;
    }
    let target = w_ref.as_tensor().to_scalar::<f32>()? as f64;

    let w = Var::new(1.0f32, &device)?;
    let mut b = SGD::new_lr(vec![w.clone()], 0.1)?;
    for _ in 0..2 {
        let grads = loss(&w)?.backward()?;
        b.step(&grads)?;
    }
    let path = std::env::temp_dir().join("runly_sgd_state.safetensors");
    b.save_state(&path)?;
    let mut c = SGD::new_lr(vec![w.clone()], 0.1)?;
    c.load_state(&path)?;
    let grads = loss(&w)?.backward()?;
    c.step(&grads)?;
    let got = w.as_tensor().to_scalar::<f32>()? as f64;
    assert!((got - target).abs() < 1e-5, "{got} vs {target}");
    std::fs::remove_file(&path).ok();
    Ok(())
}

#[test]
fn sgd_state_restores_config() -> Result<()> {
    let device = Device::Cpu;
    let w = Var::new(1.0f32, &device)?;
    let config = SgdConfig {
        lr: 0.1,
        momentum: 0.5,
        max_grad_norm: Some(2.0),
        ..Default::default()
    };
    let a = SGD::new(vec![w.clone()], config)?;
    let path = std::env::temp_dir().join("runly_sgd_config.safetensors");
    a.save_state(&path)?;
    let mut b = SGD::new(vec![w.clone()], SgdConfig::default())?;
    b.load_state(&path)?;
    let c = b.config();
    assert!((c.momentum - 0.5).abs() < 1e-12);
    assert!((c.dampening - 0.0).abs() < 1e-12);
    assert_eq!(c.max_grad_norm, Some(2.0));
    std::fs::remove_file(&path).ok();
    Ok(())
}

#[test]
fn optimizer_state_rejects_mismatched_size() -> Result<()> {
    let device = Device::Cpu;
    let w = Var::new(1.0f32, &device)?;
    let a = Adam::new_lr(vec![w.clone()], 0.01)?;
    let path = std::env::temp_dir().join("runly_adam_mismatch.safetensors");
    a.save_state(&path)?;
    let mut b = Adam::new_lr(vec![w.clone(), w.clone()], 0.01)?;
    assert!(b.load_state(&path).is_err());
    std::fs::remove_file(&path).ok();
    Ok(())
}

#[test]
fn checkpointer_saves_model_and_optimizer() -> Result<()> {
    let device = Device::Cpu;
    let data = xor(8, 0.1, 3);
    let (x, y) = data.tensors(&device)?;
    let mut model = Model::new(&device);
    model.linear(2, 8)?;
    model.tanh();
    model.linear(8, 2)?;
    let mut opt = Adam::new_lr(model.vars(), 0.02)?;
    opt.set_max_grad_norm(Some(1.0));
    let logits = model.forward(&x)?;
    let loss = cross_entropy(&logits, &y)?;
    opt.backward_step(&loss)?;
    let path = std::env::temp_dir().join("runly_checkpoint_opt.safetensors");
    std::fs::remove_file(&path).ok();
    let mut cp = Checkpointer::new(&path);
    assert!(cp.update_with_optimizer(&model, &opt, 0.7)?);
    let tensors = runly::candle::safetensors::load(&path, &device)?;
    assert!(tensors.contains_key("adam.0.m"));
    assert!(tensors.contains_key("layers.0.weight"));
    let restored = Model::load(&path, &device)?;
    let mut resumed = Adam::new_lr(restored.vars(), 0.0)?;
    resumed.load_state(&path)?;
    assert!((resumed.learning_rate() - 0.02).abs() < 1e-12);
    assert_eq!(resumed.config().max_grad_norm, Some(1.0));
    std::fs::remove_file(&path).ok();
    Ok(())
}

#[test]
fn early_stopping_saves_optimizer() -> Result<()> {
    let device = Device::Cpu;
    let mut model = Model::new(&device);
    model.linear(2, 2)?;
    let opt = Adam::new_lr(model.vars(), 0.02)?;
    let path = std::env::temp_dir().join("runly_es_opt.safetensors");
    std::fs::remove_file(&path).ok();
    let mut es = EarlyStopping::new(&path, 2, 0.0)?;
    assert!(!es.update_with_optimizer(&model, &opt, 0.5)?);
    assert!(!es.update_with_optimizer(&model, &opt, 0.6)?);
    assert!(es.update_with_optimizer(&model, &opt, 0.7)?);
    let tensors = runly::candle::safetensors::load(&path, &device)?;
    assert!(tensors.contains_key("adam.0.m"));
    std::fs::remove_file(&path).ok();
    Ok(())
}
