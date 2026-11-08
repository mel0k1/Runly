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
    for epoch in 1..=300 {
        let logits = model.forward(&x)?;
        let loss = cross_entropy(&logits, &y)?;
        opt.backward_step(&loss)?;
        if epoch == 1 || epoch % 50 == 0 {
            println!(
                "epoch {epoch:3}  loss {:.4}  accuracy {:.3}",
                loss.to_dtype(DType::F64)?.to_scalar::<f64>()?,
                accuracy(&logits, &y)?
            );
        }
    }
    let logits = model.forward(&x)?;
    println!("final accuracy {:.3}", accuracy(&logits, &y)?);
    model.save("xor.safetensors")?;
    println!("weights saved to xor.safetensors");
    Ok(())
}
