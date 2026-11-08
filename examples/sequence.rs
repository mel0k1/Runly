use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use runly::prelude::*;

fn main() -> Result<()> {
    let device = device()?;
    let vocab = 10usize;
    let marker = 7usize;
    let len = 8usize;
    let n = 512usize;
    let mut rng = StdRng::seed_from_u64(7);
    let mut tokens = Vec::with_capacity(n * len);
    let mut labels = Vec::with_capacity(n);
    for _ in 0..n {
        let positive = rng.random_bool(0.5);
        let count = if positive {
            rng.random_range(2..=3)
        } else {
            rng.random_range(0..=1)
        };
        let mut positions = Vec::with_capacity(count);
        while positions.len() < count {
            let pos = rng.random_range(0..len);
            if !positions.contains(&pos) {
                positions.push(pos);
            }
        }
        for pos in 0..len {
            let token = if positions.contains(&pos) {
                marker
            } else {
                rng.random_range(0..marker)
            };
            tokens.push(token as u32);
        }
        labels.push(positive as u32);
    }
    let cut = n / 4 * 3;
    let x = Tensor::from_vec(tokens, (n, len), &device)?;
    let y = Tensor::from_vec(labels, n, &device)?;
    let (train_x, test_x) = (x.narrow(0, 0, cut)?, x.narrow(0, cut, n - cut)?);
    let (train_y, test_y) = (y.narrow(0, 0, cut)?, y.narrow(0, cut, n - cut)?);

    let emb = Var::from_tensor(&Tensor::randn(0f32, 0.1f32, (vocab, 8), &device)?)?;
    let mut model = Model::new(&device);
    model.gru(8, 24)?;
    model.select_last_step();
    model.linear(24, 2)?;

    let mut vars = vec![emb.clone()];
    vars.extend(model.vars());
    let mut opt = Adam::new(
        vars,
        AdamConfig {
            lr: 0.005,
            max_grad_norm: Some(1.0),
            ..Default::default()
        },
    )?;

    let history = fit(
        &mut opt,
        |tokens, targets| {
            let embedded = embedding(emb.as_tensor(), tokens)?;
            let logits = model.forward(&embedded)?;
            cross_entropy(&logits, targets)
        },
        &train_x,
        &train_y,
        FitConfig {
            epochs: 200,
            batch_size: 64,
            shuffle: true,
            seed: 42,
        },
    )?;
    for (i, loss) in history.iter().enumerate() {
        if (i + 1) % 25 == 0 {
            println!("epoch {:3}  loss {:.4}", i + 1, loss);
        }
    }

    let embedded = embedding(emb.as_tensor(), &test_x)?;
    let logits = model.forward_eval(&embedded)?;
    println!("test accuracy {:.3}", accuracy(&logits, &test_y)?);
    let matrix = confusion_matrix(&logits, &test_y, 2)?;
    for (class, m) in per_class_metrics(&matrix).iter().enumerate() {
        println!(
            "class {class}: precision {:.3} recall {:.3} f1 {:.3}",
            m.precision, m.recall, m.f1
        );
    }
    println!("macro f1 {:.3}", macro_f1(&matrix));
    Ok(())
}
