use runly::prelude::*;

#[test]
fn conv_transpose2d_one_by_one_matches_reference() -> Result<()> {
    let device = Device::Cpu;
    let mut model = Model::new(&device);
    model.conv_transpose2d(1, 1, 1, 1, 0)?;
    let weight = Tensor::from_vec(vec![3.0f32], (1, 1, 1, 1), &device)?;
    let bias = Tensor::from_vec(vec![0.5f32], 1, &device)?;
    {
        let data = model.varmap().data().lock().expect("varmap poisoned");
        data.get("layers.0.weight")
            .expect("weight missing")
            .set(&weight)?;
        data.get("layers.0.bias")
            .expect("bias missing")
            .set(&bias)?;
    }
    let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 1, 2, 2), &device)?;
    let out = model.forward_eval(&x)?;
    assert_eq!(out.dims(), [1, 1, 2, 2]);
    let got = out.flatten_all()?.to_vec1::<f32>()?;
    let expected = [3.5, 6.5, 9.5, 12.5];
    for (g, e) in got.iter().zip(expected) {
        assert!((g - e).abs() < 1e-6, "{g} vs {e}");
    }
    Ok(())
}

#[test]
fn conv_transpose2d_shapes_follow_the_formula() -> Result<()> {
    let device = Device::Cpu;
    let mut model = Model::new(&device);
    model.conv_transpose2d(2, 3, 2, 2, 0)?;
    let x = Tensor::randn(0f32, 1f32, (1, 2, 4, 4), &device)?;
    assert_eq!(model.forward_eval(&x)?.dims(), [1, 3, 8, 8]);
    let mut model = Model::new(&device);
    model.conv_transpose2d(2, 3, 3, 2, 1)?;
    assert_eq!(model.forward_eval(&x)?.dims(), [1, 3, 7, 7]);
    Ok(())
}

#[test]
fn model_with_conv_transpose_round_trips() -> Result<()> {
    let device = Device::Cpu;
    let mut model = Model::new(&device);
    model.conv2d(1, 2, 3, 1, 1)?;
    model.relu();
    model.conv_transpose2d(2, 1, 2, 2, 0)?;
    let path = std::env::temp_dir().join("runly_conv_transpose_roundtrip.safetensors");
    model.save(&path)?;
    let restored = Model::load(&path, &device)?;
    let x = Tensor::randn(0f32, 1f32, (2, 1, 4, 4), &device)?;
    let diff = (model.forward_eval(&x)? - restored.forward_eval(&x)?)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    assert!(diff < 1e-5, "max diff {diff}");
    std::fs::remove_file(&path).ok();
    Ok(())
}

#[test]
fn conv_transpose_network_learns_upsampling() -> Result<()> {
    let device = Device::Cpu;
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for bits in 0..16u32 {
        let value = |i: usize| if (bits >> i) & 1 == 1 { 1.0f32 } else { -1.0 };
        for i in 0..4 {
            xs.push(value(i));
        }
        for row in 0..4 {
            for col in 0..4 {
                ys.push(value((row / 2) * 2 + col / 2));
            }
        }
    }
    let x = Tensor::from_vec(xs, (16, 1, 2, 2), &device)?;
    let target = Tensor::from_vec(ys, (16, 1, 4, 4), &device)?;
    let mut model = Model::new(&device);
    model.conv_transpose2d(1, 1, 2, 2, 0)?;
    let mut opt = Adam::new_lr(model.vars(), 0.02)?;
    for _ in 0..400 {
        let out = model.forward(&x)?;
        let loss = mse(&out, &target)?;
        opt.backward_step(&loss)?;
    }
    let err = (model.forward(&x)? - target)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    assert!(err < 0.25, "max abs error {err}");
    Ok(())
}
