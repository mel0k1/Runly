use runly::prelude::*;

#[test]
fn early_stopping_stops_after_patience() -> Result<()> {
    let device = Device::Cpu;
    let mut model = Model::new(&device);
    model.linear(2, 2)?;
    let path = std::env::temp_dir().join("runly_early_stopping_patience.safetensors");
    std::fs::remove_file(&path).ok();
    let mut es = EarlyStopping::new(&path, 2, 0.0)?;
    assert!(!es.update(&model, 0.5)?);
    assert!(!es.update(&model, 0.6)?);
    assert_eq!(es.bad_epochs(), 1);
    assert!(es.update(&model, 0.7)?);
    assert!(es.stopped());
    assert!(es.update(&model, 0.1)?);
    assert_eq!(es.best(), Some(0.5));
    std::fs::remove_file(&path).ok();
    Ok(())
}

#[test]
fn early_stopping_respects_min_delta() -> Result<()> {
    let device = Device::Cpu;
    let mut model = Model::new(&device);
    model.linear(2, 2)?;
    let path = std::env::temp_dir().join("runly_early_stopping_min_delta.safetensors");
    std::fs::remove_file(&path).ok();
    let mut es = EarlyStopping::new(&path, 2, 0.2)?;
    assert!(!es.update(&model, 0.5)?);
    assert!(!es.update(&model, 0.4)?);
    assert_eq!(es.bad_epochs(), 1);
    assert!(es.update(&model, 0.45)?);
    assert_eq!(es.best(), Some(0.5));
    std::fs::remove_file(&path).ok();
    Ok(())
}

#[test]
fn early_stopping_resets_on_improvement() -> Result<()> {
    let device = Device::Cpu;
    let mut model = Model::new(&device);
    model.linear(2, 2)?;
    let path = std::env::temp_dir().join("runly_early_stopping_reset.safetensors");
    std::fs::remove_file(&path).ok();
    let mut es = EarlyStopping::new(&path, 2, 0.1)?;
    assert!(!es.update(&model, 0.5)?);
    assert!(!es.update(&model, 0.48)?);
    assert_eq!(es.bad_epochs(), 1);
    assert!(!es.update(&model, 0.3)?);
    assert_eq!(es.bad_epochs(), 0);
    assert_eq!(es.best(), Some(0.3));
    assert!(!es.stopped());
    std::fs::remove_file(&path).ok();
    Ok(())
}

#[test]
fn early_stopping_saves_best_weights() -> Result<()> {
    let device = Device::Cpu;
    let mut a = Model::new(&device);
    a.linear(2, 2)?;
    let mut b = Model::new(&device);
    b.linear(2, 2)?;
    let x = Tensor::randn(0f32, 1f32, (4, 2), &device)?;
    let path = std::env::temp_dir().join("runly_early_stopping_best.safetensors");
    std::fs::remove_file(&path).ok();
    let mut es = EarlyStopping::new(&path, 5, 0.0)?;
    assert!(!es.update(&a, 0.9)?);
    assert!(!es.update(&b, 0.1)?);
    assert!(!es.update(&a, 0.5)?);
    assert_eq!(es.best(), Some(0.1));
    let restored = Model::load(&path, &device)?;
    let diff = (restored.forward_eval(&x)? - b.forward_eval(&x)?)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    assert!(diff < 1e-6, "max diff {diff}");
    std::fs::remove_file(&path).ok();
    Ok(())
}

#[test]
fn early_stopping_validates_args() -> Result<()> {
    assert!(EarlyStopping::new("unused.safetensors", 0, 0.0).is_err());
    assert!(EarlyStopping::new("unused.safetensors", 1, -0.1).is_err());
    Ok(())
}

#[test]
fn early_stopping_exposes_path_and_checkpointer() -> Result<()> {
    let dir = std::env::temp_dir();
    let es = EarlyStopping::new(dir.join("unused_stopping.safetensors"), 3, 0.0)?;
    assert_eq!(es.path(), dir.join("unused_stopping.safetensors"));
    assert!(es.checkpointer().best().is_none());
    assert_eq!(es.bad_epochs(), 0);
    Ok(())
}
