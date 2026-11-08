#[cfg(feature = "mnist")]
use runly::prelude::*;

#[cfg(feature = "mnist")]
fn main() -> Result<()> {
    let device = device()?;
    let mnist = runly::mnist::load(&device)?;
    let n = mnist.train_labels.elem_count();
    let mut model = Model::new(&device);
    model.linear(784, 128)?;
    model.relu();
    model.linear(128, 10)?;
    let mut opt = Adam::new_lr(model.vars(), 1e-3)?;
    for epoch in 1..=5u32 {
        let batches = batch_indices(n, 100, epoch as u64);
        let mut total = 0.0f64;
        for batch in &batches {
            let idx = Tensor::from_vec(
                batch.iter().map(|i| *i as u32).collect::<Vec<u32>>(),
                batch.len(),
                &device,
            )?;
            let bx = mnist.train_images.index_select(&idx, 0)?;
            let by = mnist.train_labels.index_select(&idx, 0)?;
            let logits = model.forward(&bx)?;
            let loss = cross_entropy(&logits, &by)?;
            opt.backward_step(&loss)?;
            total += loss.to_dtype(DType::F64)?.to_scalar::<f64>()?;
        }
        let acc = accuracy(&model.forward(&mnist.test_images)?, &mnist.test_labels)?;
        println!(
            "epoch {epoch}  train loss {:.4}  test accuracy {:.4}",
            total / batches.len() as f64,
            acc
        );
    }
    Ok(())
}

#[cfg(not(feature = "mnist"))]
fn main() {
    println!("run with --features mnist");
}
