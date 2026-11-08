use runly::prelude::*;

fn main() -> Result<()> {
    let device = device()?;
    let data = xor(64, 0.15, 42);
    let (x, y) = data.tensors(&device)?;
    let mut model = Model::new(&device);
    model.linear(2, 16)?;
    model.tanh();
    model.linear(16, 2)?;
    let mut opt = Adam::new_lr(model.vars(), 0.02)?;
    let mut sched = CosineAnnealingLR::new(0.02, 300, 1e-4)?;
    for epoch in 1..=300 {
        sched.apply(&mut opt);
        let logits = model.forward(&x)?;
        let loss = cross_entropy(&logits, &y)?;
        opt.backward_step(&loss)?;
        if epoch == 1 || epoch % 100 == 0 {
            let value = loss.to_dtype(DType::F64)?.to_scalar::<f64>()?;
            let acc = accuracy(&logits, &y)?;
            println!(
                "epoch {epoch}  lr {:.5}  loss {value:.4}  accuracy {acc:.3}",
                opt.learning_rate()
            );
        }
    }
    Ok(())
}
