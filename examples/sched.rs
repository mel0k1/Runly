use runly::prelude::*;

fn run(device: &Device, mut sched: impl LrScheduler, tag: &str) -> Result<()> {
    let data = xor(64, 0.15, 42);
    let (x, y) = data.tensors(device)?;
    let mut model = Model::new(device);
    model.linear(2, 16)?;
    model.tanh();
    model.linear(16, 2)?;
    let mut opt = Adam::new_lr(model.vars(), 0.02)?;
    for epoch in 1..=300 {
        sched.apply(&mut opt);
        let logits = model.forward(&x)?;
        let loss = cross_entropy(&logits, &y)?;
        opt.backward_step(&loss)?;
        if epoch == 1 || epoch % 100 == 0 {
            let value = loss.to_dtype(DType::F64)?.to_scalar::<f64>()?;
            let acc = accuracy(&logits, &y)?;
            println!(
                "{tag}  epoch {epoch}  lr {:.5}  loss {value:.4}  accuracy {acc:.3}",
                opt.learning_rate()
            );
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let device = device()?;
    let cosine = CosineAnnealingLR::new(0.02, 280, 1e-4)?;
    run(&device, Warmup::new(cosine, 20, 0.02)?, "warmup+cosine")?;
    run(
        &device,
        OneCycleLR::new(0.02, 300, 0.3, 0.001, 1e-4)?,
        "one cycle",
    )?;
    Ok(())
}
