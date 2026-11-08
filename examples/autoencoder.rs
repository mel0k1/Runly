use runly::prelude::*;

fn main() -> Result<()> {
    let device = device()?;
    let n = 128usize;
    let mut data = Vec::with_capacity(n * 64);
    for i in 0..n {
        let corner = i % 4;
        let top = corner < 2;
        let left = corner % 2 == 0;
        for r in 0..8usize {
            for c in 0..8usize {
                let in_block =
                    (if top { r < 4 } else { r >= 4 }) && (if left { c < 4 } else { c >= 4 });
                data.push(if in_block { 1.0f32 } else { -1.0 });
            }
        }
    }
    let x = Tensor::from_vec(data, (n, 1, 8, 8), &device)?;
    let mut model = Model::new(&device);
    model.conv2d(1, 2, 3, 2, 1)?;
    model.tanh();
    model.conv2d(2, 4, 3, 2, 1)?;
    model.tanh();
    model.conv_transpose2d(4, 2, 2, 2, 0)?;
    model.tanh();
    model.conv_transpose2d(2, 1, 2, 2, 0)?;
    model.tanh();
    println!(
        "encoder: 8x8 -> 2x4x4 -> 4x2x2, decoder back to 1x8x8, {} params",
        model.num_params()
    );
    let mut opt = Adam::new_lr(model.vars(), 0.01)?;
    for epoch in 1..=500 {
        let out = model.forward(&x)?;
        let loss = mse(&out, &x)?;
        opt.backward_step(&loss)?;
        if epoch == 1 || epoch % 100 == 0 {
            println!(
                "epoch {epoch:3}  mse {:.4}",
                loss.to_dtype(DType::F64)?.to_scalar::<f64>()?
            );
        }
    }
    let recon = model.forward_eval(&x)?;
    let err = (recon - &x)?.abs()?.max_all()?.to_scalar::<f32>()?;
    println!("max reconstruction error {:.4}", err);
    Ok(())
}
