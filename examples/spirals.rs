use runly::prelude::*;

fn main() -> Result<()> {
    let device = Device::Cpu;
    let data = spirals(200, 1.75, 0.2, 7);
    let (train, test) = data.train_test_split(0.2, 1);
    let (x, y) = train.tensors(&device)?;
    let (xt, yt) = test.tensors(&device)?;
    let mut model = Model::new(&device);
    model.linear(2, 64)?;
    model.relu();
    model.linear(64, 64)?;
    model.relu();
    model.linear(64, 2)?;
    let mut opt = Adam::new_lr(model.vars(), 0.005)?;
    for epoch in 1..=800 {
        let logits = model.forward(&x)?;
        let loss = cross_entropy(&logits, &y)?;
        opt.backward_step(&loss)?;
        if epoch == 1 || epoch % 100 == 0 {
            let acc = accuracy(&model.forward(&xt)?, &yt)?;
            println!(
                "epoch {epoch:3}  loss {:.4}  test accuracy {:.3}",
                loss.to_dtype(DType::F64)?.to_scalar::<f64>()?,
                acc
            );
        }
    }
    println!("train accuracy {:.3}", accuracy(&model.forward(&x)?, &y)?);
    println!("test accuracy {:.3}", accuracy(&model.forward(&xt)?, &yt)?);
    Ok(())
}
