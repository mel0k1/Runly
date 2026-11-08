use runly::prelude::*;

fn xor_tensors(device: &Device) -> Result<(Tensor, Tensor)> {
    let (x, y) = xor(32, 0.12, 11).tensors(device)?;
    Ok((x, y))
}

fn xor_model(device: &Device) -> Result<Model> {
    let mut model = Model::new(device);
    model.linear(2, 8)?;
    model.tanh();
    model.linear(8, 2)?;
    Ok(model)
}

#[test]
fn fit_learns_xor() -> Result<()> {
    let device = Device::Cpu;
    let (x, y) = xor_tensors(&device)?;
    let model = xor_model(&device)?;
    let mut opt = Adam::new_lr(model.vars(), 0.05)?;
    let history = fit(
        &mut opt,
        |xb, yb| cross_entropy(&model.forward(xb)?, yb),
        &x,
        &y,
        FitConfig {
            epochs: 300,
            batch_size: 16,
            shuffle: true,
            seed: 42,
        },
    )?;
    assert_eq!(history.len(), 300);
    assert!(
        *history.last().expect("empty history") < 0.1,
        "final loss {}",
        history.last().expect("empty history")
    );
    let acc = accuracy(&model.forward(&x)?, &y)?;
    assert!(acc >= 0.95, "accuracy {acc}");
    Ok(())
}

#[test]
fn fit_history_shrinks() -> Result<()> {
    let device = Device::Cpu;
    let (x, y) = xor_tensors(&device)?;
    let model = xor_model(&device)?;
    let mut opt = Adam::new_lr(model.vars(), 0.02)?;
    let history = fit(
        &mut opt,
        |xb, yb| cross_entropy(&model.forward(xb)?, yb),
        &x,
        &y,
        FitConfig {
            epochs: 100,
            batch_size: 32,
            shuffle: false,
            seed: 1,
        },
    )?;
    assert!(
        history[100 - 1] < history[0],
        "{} vs {}",
        history[99],
        history[0]
    );
    Ok(())
}

#[test]
fn fit_is_deterministic_for_a_seed() -> Result<()> {
    let device = Device::Cpu;
    let (x, y) = xor_tensors(&device)?;
    let model = xor_model(&device)?;
    let path = std::env::temp_dir().join("runly_fit_seed.safetensors");
    model.save(&path)?;
    let run = |shuffle: bool| -> Result<Vec<f64>> {
        let model = Model::load(&path, &device)?;
        let mut opt = Adam::new_lr(model.vars(), 0.02)?;
        fit(
            &mut opt,
            |xb, yb| cross_entropy(&model.forward(xb)?, yb),
            &x,
            &y,
            FitConfig {
                epochs: 20,
                batch_size: 16,
                shuffle,
                seed: 5,
            },
        )
    };
    let a = run(true)?;
    let b = run(true)?;
    assert_eq!(a, b);
    let c = run(false)?;
    let d = run(false)?;
    assert_eq!(c, d);
    std::fs::remove_file(&path).ok();
    Ok(())
}

#[test]
fn fit_rejects_zero_batch_size() -> Result<()> {
    let device = Device::Cpu;
    let (x, y) = xor_tensors(&device)?;
    let model = xor_model(&device)?;
    let mut opt = Adam::new_lr(model.vars(), 0.02)?;
    let result = fit(
        &mut opt,
        |xb, yb| cross_entropy(&model.forward(xb)?, yb),
        &x,
        &y,
        FitConfig {
            epochs: 1,
            batch_size: 0,
            shuffle: true,
            seed: 1,
        },
    );
    assert!(result.is_err());
    Ok(())
}
