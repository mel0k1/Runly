use runly::prelude::*;

#[test]
fn checkpointer_writes_file_and_tracks_metric() -> Result<()> {
    let device = Device::Cpu;
    let mut model = Model::new(&device);
    model.linear(3, 2)?;
    let path = std::env::temp_dir().join("runly_checkpointer_metric.safetensors");
    std::fs::remove_file(&path).ok();
    let mut cp = Checkpointer::new(&path);
    assert!(cp.best().is_none());
    assert!(cp.update(&model, 0.7)?);
    assert!(path.exists());
    assert_eq!(cp.best(), Some(0.7));
    assert!(!cp.update(&model, 0.9)?);
    assert_eq!(cp.best(), Some(0.7));
    assert!(cp.update(&model, 0.2)?);
    assert_eq!(cp.best(), Some(0.2));
    std::fs::remove_file(&path).ok();
    Ok(())
}

#[test]
fn checkpointer_keeps_best_weights() -> Result<()> {
    let device = Device::Cpu;
    let mut a = Model::new(&device);
    a.linear(2, 2)?;
    let mut b = Model::new(&device);
    b.linear(2, 2)?;
    let x = Tensor::randn(0f32, 1f32, (4, 2), &device)?;
    let path = std::env::temp_dir().join("runly_checkpointer_best.safetensors");
    std::fs::remove_file(&path).ok();
    let mut cp = Checkpointer::new(&path);
    assert!(cp.update(&a, 1.0)?);
    assert!(cp.update(&b, 0.5)?);
    assert!(!cp.update(&a, 0.8)?);
    assert_eq!(cp.best(), Some(0.5));
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
fn checkpointer_path_is_exposed() -> Result<()> {
    let device = Device::Cpu;
    let mut model = Model::new(&device);
    model.linear(2, 2)?;
    let dir = std::env::temp_dir();
    let cp = Checkpointer::new(dir.join("unused.safetensors"));
    assert_eq!(cp.path(), dir.join("unused.safetensors"));
    let _ = model;
    Ok(())
}
