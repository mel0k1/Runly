use runly::prelude::*;

fn main() -> Result<()> {
    let device = device()?;
    let data = spirals(250, 1.75, 0.2, 7);
    let mut csv = String::from("x1,x2,label\n");
    for i in 0..data.len() {
        csv.push_str(&format!(
            "{:.5},{:.5},{}\n",
            data.x[2 * i],
            data.x[2 * i + 1],
            data.y[i]
        ));
    }
    std::fs::write("spirals.csv", csv)?;
    println!("wrote spirals.csv, {} rows", data.len());

    let loaded = load_csv_named("spirals.csv", "label")?;
    println!(
        "loaded back: {} rows, {} features, {} classes",
        loaded.len(),
        loaded.dim,
        loaded.num_classes
    );
    let (train, test) = loaded.train_test_split(0.2, 1);
    let (x, y) = train.tensors(&device)?;
    let (xt, yt) = test.tensors(&device)?;
    let mut model = Model::new(&device);
    model.linear(loaded.dim, 64)?;
    model.relu();
    model.linear(64, 64)?;
    model.relu();
    model.linear(64, loaded.num_classes)?;
    let mut opt = Adam::new_lr(model.vars(), 0.005)?;
    for epoch in 1..=800 {
        let logits = model.forward(&x)?;
        let loss = cross_entropy(&logits, &y)?;
        opt.backward_step(&loss)?;
        if epoch == 1 || epoch % 200 == 0 {
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
