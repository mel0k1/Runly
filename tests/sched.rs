use runly::prelude::*;

#[test]
fn exponential_lr_halves_every_call() -> Result<()> {
    let mut sched = ExponentialLR::new(1.0, 0.5)?;
    for expected in [1.0, 0.5, 0.25, 0.125] {
        let got = sched.lr();
        assert!((got - expected).abs() < 1e-12, "{got} vs {expected}");
    }
    Ok(())
}

#[test]
fn step_lr_drops_after_step_size_calls() -> Result<()> {
    let mut sched = StepLR::new(1.0, 2, 0.1)?;
    for expected in [1.0, 1.0, 0.1, 0.1, 0.01, 0.01] {
        let got = sched.lr();
        assert!((got - expected).abs() < 1e-12, "{got} vs {expected}");
    }
    Ok(())
}

#[test]
fn cosine_lr_matches_reference_values() -> Result<()> {
    let mut sched = CosineAnnealingLR::new(1.0, 4, 0.0)?;
    let expected = [1.0, 0.8535533905932737, 0.5, 0.1464466094067262, 0.0];
    for value in expected {
        let got = sched.lr();
        assert!((got - value).abs() < 1e-12, "{got} vs {value}");
    }
    Ok(())
}

#[test]
fn cosine_lr_stops_at_eta_min() -> Result<()> {
    let mut sched = CosineAnnealingLR::new(0.02, 300, 1e-4)?;
    let mut last = 0.0;
    for _ in 0..301 {
        last = sched.lr();
    }
    assert!((last - 1e-4).abs() < 1e-9, "final lr {last}");
    Ok(())
}

#[test]
fn scheduler_updates_optimizer_learning_rate() -> Result<()> {
    let device = Device::Cpu;
    let w = Var::new(1.0f32, &device)?;
    let mut opt = Adam::new_lr(vec![w.clone()], 1.0)?;
    let mut sched = StepLR::new(1.0, 3, 0.5)?;
    for _ in 0..3 {
        sched.apply(&mut opt);
    }
    assert!((opt.learning_rate() - 1.0).abs() < 1e-12);
    sched.apply(&mut opt);
    assert!((opt.learning_rate() - 0.5).abs() < 1e-12);
    Ok(())
}

#[test]
fn invalid_scheduler_arguments_are_rejected() {
    assert!(StepLR::new(1.0, 0, 0.5).is_err());
    assert!(ExponentialLR::new(1.0, 1.5).is_err());
    assert!(CosineAnnealingLR::new(1.0, 0, 0.0).is_err());
    assert!(CosineAnnealingLR::new(1.0, 10, 2.0).is_err());
}

#[test]
fn xor_trains_with_cosine_schedule() -> Result<()> {
    let device = Device::Cpu;
    let data = xor(64, 0.15, 42);
    let (x, y) = data.tensors(&device)?;
    let mut model = Model::new(&device);
    model.linear(2, 16)?;
    model.tanh();
    model.linear(16, 2)?;
    let mut opt = Adam::new_lr(model.vars(), 0.02)?;
    let mut sched = CosineAnnealingLR::new(0.02, 300, 1e-4)?;
    let mut loss = f64::INFINITY;
    for _ in 0..300 {
        sched.apply(&mut opt);
        let logits = model.forward(&x)?;
        let loss_t = cross_entropy(&logits, &y)?;
        loss = loss_t.to_dtype(DType::F64)?.to_scalar::<f64>()?;
        opt.backward_step(&loss_t)?;
    }
    let acc = accuracy(&model.forward(&x)?, &y)?;
    assert!(acc >= 0.95, "accuracy {acc}");
    assert!(loss < 0.1, "final loss {loss}");
    Ok(())
}

#[test]
fn warmup_ramps_then_hands_over() -> Result<()> {
    let inner = StepLR::new(1.0, 2, 0.1)?;
    let mut sched = Warmup::new(inner, 3, 0.6)?;
    let expected = [0.2, 0.4, 0.6, 1.0, 1.0, 0.1];
    for value in expected {
        let got = sched.lr();
        assert!((got - value).abs() < 1e-12, "{got} vs {value}");
    }
    Ok(())
}

#[test]
fn one_cycle_matches_reference_values() -> Result<()> {
    let mut sched = OneCycleLR::new(1.0, 100, 0.3, 0.0, 0.0)?;
    let mut values = Vec::new();
    for _ in 0..101 {
        values.push(sched.lr());
    }
    let checks = [(0, 1.0 / 30.0), (29, 1.0), (30, 1.0), (65, 0.5), (100, 0.0)];
    for (index, expected) in checks {
        let got = values[index];
        assert!(
            (got - expected).abs() < 1e-9,
            "step {index}: {got} vs {expected}"
        );
    }
    Ok(())
}

#[test]
fn one_cycle_zero_warmup_starts_at_peak() -> Result<()> {
    let mut sched = OneCycleLR::new(0.02, 50, 0.0, 0.01, 1e-4)?;
    let first = sched.lr();
    assert!((first - 0.02).abs() < 1e-12, "first lr {first}");
    Ok(())
}

#[test]
fn invalid_warmup_and_one_cycle_arguments_are_rejected() {
    let inner = StepLR::new(1.0, 1, 0.5).unwrap();
    assert!(Warmup::new(inner, 0, 1.0).is_err());
    assert!(Warmup::new(inner, 5, -1.0).is_err());
    assert!(OneCycleLR::new(1.0, 1, 0.3, 0.0, 0.0).is_err());
    assert!(OneCycleLR::new(1.0, 100, 1.0, 0.0, 0.0).is_err());
    assert!(OneCycleLR::new(1.0, 100, 0.3, 2.0, 0.0).is_err());
    assert!(OneCycleLR::new(1.0, 100, 0.3, 0.0, -0.1).is_err());
}

#[test]
fn xor_trains_with_one_cycle() -> Result<()> {
    let device = Device::Cpu;
    let data = xor(64, 0.15, 42);
    let (x, y) = data.tensors(&device)?;
    let mut model = Model::new(&device);
    model.linear(2, 16)?;
    model.tanh();
    model.linear(16, 2)?;
    let mut opt = Adam::new_lr(model.vars(), 0.02)?;
    let mut sched = OneCycleLR::new(0.02, 300, 0.3, 0.001, 1e-4)?;
    for _ in 0..300 {
        sched.apply(&mut opt);
        let logits = model.forward(&x)?;
        let loss = cross_entropy(&logits, &y)?;
        opt.backward_step(&loss)?;
    }
    let acc = accuracy(&model.forward(&x)?, &y)?;
    assert!(acc >= 0.95, "accuracy {acc}");
    Ok(())
}
