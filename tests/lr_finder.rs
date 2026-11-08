use runly::prelude::*;

#[test]
fn lr_finder_sweeps_learning_rates_and_restores_state() -> Result<()> {
    let device = Device::Cpu;
    let data = xor(64, 0.15, 42);
    let (x, y) = data.tensors(&device)?;
    let mut model = Model::new(&device);
    model.linear(2, 16)?;
    model.tanh();
    model.linear(16, 2)?;
    let mut opt = Adam::new_lr(model.vars(), 0.01)?;
    let before = model.forward(&x)?;
    let state_before = opt.state_tensors()?;
    let finder = LrFinder::new(1e-4, 2.0, 8)?;
    let points = finder.run(
        &model,
        &mut opt,
        &x,
        &y,
        |xb, yb| cross_entropy(&model.forward(xb)?, yb),
        64,
    )?;
    assert!(
        points.len() >= 2,
        "expected several points, got {}",
        points.len()
    );
    for pair in points.windows(2) {
        assert!(
            pair[1].lr > pair[0].lr,
            "learning rates must grow: {pair:?}"
        );
        assert!(
            pair[0].loss.is_finite(),
            "loss must be finite: {:?}",
            pair[0]
        );
        assert!(pair[0].grad_norm.is_finite() && pair[0].grad_norm > 0.0);
    }
    assert!((points[0].lr - 1e-4).abs() < 1e-12);
    let after = model.forward(&x)?;
    let diff = (before - &after)?.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(diff < 1e-6, "weights were not restored, diff {diff}");
    let state_after = opt.state_tensors()?;
    for (name, tensor) in &state_before {
        let restored = state_after
            .get(name)
            .unwrap_or_else(|| panic!("missing {name} after restore"));
        let delta = (restored - tensor)?
            .abs()?
            .max_all()?
            .to_dtype(DType::F64)?
            .to_scalar::<f64>()?;
        assert!(
            delta < 1e-9,
            "optimizer state {name} was not restored, delta {delta}"
        );
    }
    assert!(
        (opt.config().lr - 0.01).abs() < 1e-12,
        "base lr not restored"
    );
    let lr = suggested_lr(&points).expect("suggested lr");
    assert!((1e-4..2.0).contains(&lr), "suggested lr {lr} out of range");
    Ok(())
}

#[test]
fn lr_finder_stops_on_loss_explosion() -> Result<()> {
    let device = Device::Cpu;
    let data = xor(32, 0.15, 7);
    let (x, y) = data.tensors(&device)?;
    let mut model = Model::new(&device);
    model.linear(2, 16)?;
    model.tanh();
    model.linear(16, 2)?;
    let mut opt = Adam::new_lr(model.vars(), 1e-3)?;
    let finder = LrFinder::new(1e-2, 1e5, 40)?;
    let points = finder.run(
        &model,
        &mut opt,
        &x,
        &y,
        |xb, yb| cross_entropy(&model.forward(xb)?, yb),
        32,
    )?;
    assert!(points.len() < 40, "finder should stop early on explosion");
    let best = points.iter().map(|p| p.loss).fold(f64::INFINITY, f64::min);
    assert!(points[points.len() - 1].loss > 4.0 * best || points.len() == 2);
    Ok(())
}

#[test]
fn lr_finder_validates_arguments() -> Result<()> {
    assert!(LrFinder::new(0.0, 1.0, 10).is_err());
    assert!(LrFinder::new(-1.0, 1.0, 10).is_err());
    assert!(LrFinder::new(2.0, 1.0, 10).is_err());
    assert!(LrFinder::new(1e-3, 1e-3, 10).is_err());
    assert!(LrFinder::new(1e-3, 1.0, 1).is_err());
    assert!(LrFinder::new(1e-3, 1.0, 2)?.with_stop_div(1.0).is_err());
    let device = Device::Cpu;
    let mut model = Model::new(&device);
    model.linear(1, 1)?;
    let x = Tensor::zeros((1, 1), DType::F32, &device)?;
    let y = Tensor::zeros(1, DType::U32, &device)?;
    let mut opt = Adam::new_lr(model.vars(), 0.01)?;
    let finder = LrFinder::new(1e-3, 1.0, 4)?;
    assert!(
        finder
            .run(
                &model,
                &mut opt,
                &x,
                &y,
                |xb, yb| cross_entropy(&model.forward(xb)?, yb),
                0
            )
            .is_err()
    );
    Ok(())
}

#[test]
fn suggested_lr_picks_steepest_descent() {
    let points = |lrs: &[f64], losses: &[f64]| -> Vec<LrPoint> {
        lrs.iter()
            .zip(losses.iter())
            .map(|(&lr, &loss)| LrPoint {
                lr,
                loss,
                grad_norm: 1.0,
            })
            .collect()
    };
    let flat = points(&[0.1, 0.2, 0.4, 0.8], &[1.0, 0.9, 0.9, 0.9]);
    assert!((suggested_lr(&flat).unwrap() - 0.1).abs() < 1e-12);
    assert!(suggested_lr(&[]).is_none());
    assert!(suggested_lr(&points(&[0.1], &[1.0])).is_none());
}
