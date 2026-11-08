use runly::prelude::*;

fn pattern_inputs(n: usize, steps: usize, features: usize, seed: u64) -> Tensor {
    let mut values = Vec::with_capacity(n * steps * features);
    for i in 0..n {
        for t in 0..steps {
            for f in 0..features {
                let bit = ((i * 7919 + t * 104729 + f * 1299709 + seed as usize) % 97) % 2;
                values.push(if bit == 0 { -1.0f32 } else { 1.0 });
            }
        }
    }
    Tensor::from_vec(values, (n, steps, features), &Device::Cpu).unwrap()
}

#[test]
fn rnn_shapes_and_states_evolve() -> Result<()> {
    let device = Device::Cpu;
    let mut model = Model::new(&device);
    model.rnn(3, 5)?;
    let x = pattern_inputs(4, 6, 3, 1);
    let out = model.forward(&x)?;
    assert_eq!(out.dims(), &[4, 6, 5]);
    let drift = (out.i((.., 0, ..))? - out.i((.., 5, ..))?)?
        .abs()?
        .sum_all()?
        .to_scalar::<f32>()?;
    assert!(
        drift > 1e-4,
        "hidden states must evolve over time, drift {drift}"
    );
    let path = std::env::temp_dir().join(format!(
        "runly_rnn_split_{}.safetensors",
        std::process::id()
    ));
    model.save(&path)?;
    let mut split = Model::load(&path, &device)?;
    split.select_last_step();
    let last = split.forward(&x)?;
    assert_eq!(last.dims(), &[4, 5]);
    let direct = select_last_step(&out)?;
    let diff = (direct - &last)?.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(diff < 1e-6, "select_last_step mismatch {diff}");
    std::fs::remove_file(&path)?;
    Ok(())
}

#[test]
fn gru_shapes_and_states_evolve() -> Result<()> {
    let device = Device::Cpu;
    let mut model = Model::new(&device);
    model.gru(2, 7)?;
    let x = pattern_inputs(3, 5, 2, 2);
    let out = model.forward(&x)?;
    assert_eq!(out.dims(), &[3, 5, 7]);
    let drift = (out.i((.., 0, ..))? - out.i((.., 4, ..))?)?
        .abs()?
        .sum_all()?
        .to_scalar::<f32>()?;
    assert!(
        drift > 1e-4,
        "hidden states must evolve over time, drift {drift}"
    );
    Ok(())
}

#[test]
fn rnn_and_gru_save_load_roundtrip() -> Result<()> {
    let device = Device::Cpu;
    let mut model = Model::new(&device);
    model.rnn(2, 6)?;
    model.gru(6, 4)?;
    model.select_last_step();
    model.linear(4, 2)?;
    assert_eq!(
        model.num_params(),
        (2 * 6 + 6) + (6 * 6 + 6) + 3 * (6 * 4 + 4) + 3 * (4 * 4 + 4) + (4 * 2 + 2)
    );
    let x = pattern_inputs(5, 4, 2, 3);
    let path = std::env::temp_dir().join(format!("runly_rnn_{}.safetensors", std::process::id()));
    model.save(&path)?;
    let loaded = Model::load(&path, &device)?;
    let diff = (model.forward(&x)? - loaded.forward(&x)?)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    assert!(diff < 1e-6, "roundtrip diff {diff}");
    let eval_diff = (model.forward_eval(&x)? - loaded.forward_eval(&x)?)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    assert!(eval_diff < 1e-6, "eval roundtrip diff {eval_diff}");
    std::fs::remove_file(&path)?;
    Ok(())
}

#[test]
fn rnn_gradients_match_numerical() -> Result<()> {
    let device = Device::Cpu;
    let mut model = Model::new(&device);
    model.rnn(2, 3)?;
    model.select_last_step();
    model.linear(3, 2)?;
    let x = pattern_inputs(3, 4, 2, 5);
    let target = Tensor::from_vec(vec![0.0f32, 1.0, 1.0, 0.0, 0.0, 0.0], (3, 2), &device)?;
    let reports = check_model(
        &model,
        &x,
        |out| mse(out, &target),
        GradCheckConfig::default(),
    )?;
    for report in &reports {
        assert!(report.ok, "{} max_err {:.3e}", report.name, report.max_err);
    }
    Ok(())
}

#[test]
fn gru_gradients_match_numerical() -> Result<()> {
    let device = Device::Cpu;
    let mut model = Model::new(&device);
    model.gru(2, 3)?;
    model.select_last_step();
    model.linear(3, 2)?;
    let x = pattern_inputs(3, 3, 2, 6);
    let target = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 0.0], (3, 2), &device)?;
    let reports = check_model(
        &model,
        &x,
        |out| mse(out, &target),
        GradCheckConfig::default(),
    )?;
    for report in &reports {
        assert!(report.ok, "{} max_err {:.3e}", report.name, report.max_err);
    }
    Ok(())
}

#[test]
fn gru_learns_to_remember_first_symbol() -> Result<()> {
    let device = Device::Cpu;
    let n = 96usize;
    let steps = 5usize;
    let mut values = Vec::with_capacity(n * steps);
    let mut labels = Vec::with_capacity(n);
    for i in 0..n {
        let class = (i % 2) as u32;
        values.push(if class == 1 { 1.0f32 } else { -1.0 });
        for t in 1..steps {
            let bit = ((i * 31 + t * 17) % 7) % 2;
            values.push(if bit == 0 { -1.0 } else { 1.0 });
        }
        labels.push(class);
    }
    let x = Tensor::from_vec(values, (n, steps, 1), &device)?;
    let y = Tensor::from_vec(labels, n, &device)?;
    let mut model = Model::new(&device);
    model.gru(1, 10)?;
    model.select_last_step();
    model.linear(10, 2)?;
    let mut opt = Adam::new_lr(model.vars(), 0.04)?;
    let mut final_loss = 0.0;
    for _ in 0..400 {
        let logits = model.forward(&x)?;
        let loss = cross_entropy(&logits, &y)?;
        opt.backward_step(&loss)?;
        final_loss = loss.to_scalar::<f32>()?;
    }
    assert!(
        final_loss < 0.15,
        "gru did not learn to remember the first symbol, loss {final_loss}"
    );
    let acc = accuracy(&model.forward(&x)?, &y)?;
    assert!(acc >= 0.97, "gru accuracy {acc}");
    Ok(())
}

#[test]
fn rnn_learns_last_step_rule() -> Result<()> {
    let device = Device::Cpu;
    let n = 96usize;
    let steps = 3usize;
    let mut values = Vec::with_capacity(n * steps);
    let mut labels = Vec::with_capacity(n);
    for i in 0..n {
        for t in 0..steps {
            let bit = ((i * 13 + t * 29) % 11) % 2;
            values.push(if bit == 0 { -1.0f32 } else { 1.0 });
        }
        let last_bit = ((i * 13 + (steps - 1) * 29) % 11) % 2;
        labels.push(last_bit as u32);
    }
    let x = Tensor::from_vec(values, (n, steps, 1), &device)?;
    let y = Tensor::from_vec(labels, n, &device)?;
    let mut model = Model::new(&device);
    model.rnn(1, 10)?;
    model.select_last_step();
    model.linear(10, 2)?;
    let mut opt = Adam::new_lr(model.vars(), 0.05)?;
    let mut final_loss = 0.0;
    for _ in 0..300 {
        let logits = model.forward(&x)?;
        let loss = cross_entropy(&logits, &y)?;
        opt.backward_step(&loss)?;
        final_loss = loss.to_scalar::<f32>()?;
    }
    assert!(
        final_loss < 0.15,
        "rnn did not learn the last step rule, loss {final_loss}"
    );
    let acc = accuracy(&model.forward(&x)?, &y)?;
    assert!(acc >= 0.97, "rnn accuracy {acc}");
    Ok(())
}
