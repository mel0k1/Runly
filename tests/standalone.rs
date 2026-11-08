use runly::prelude::*;

#[test]
fn conv_transpose2d_matches_reference() -> Result<()> {
    let device = Device::Cpu;
    let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 1, 2, 2), &device)?;
    let weight = Tensor::from_vec(vec![3.0f32], (1, 1, 1, 1), &device)?;
    let bias = Tensor::from_vec(vec![0.5f32], 1, &device)?;
    let out = conv_transpose2d(&x, &weight, Some(&bias), 1, 0)?;
    assert_eq!(out.dims(), [1, 1, 2, 2]);
    let got = out.flatten_all()?.to_vec1::<f32>()?;
    let expected = [3.5, 6.5, 9.5, 12.5];
    for (g, e) in got.iter().zip(expected) {
        assert!((g - e).abs() < 1e-6, "{g} vs {e}");
    }
    Ok(())
}

#[test]
fn conv_transpose2d_upsamples_with_identity_kernel() -> Result<()> {
    let device = Device::Cpu;
    let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 1, 2, 2), &device)?;
    let weight = Tensor::from_vec(vec![1.0f32, 1.0, 1.0, 1.0], (1, 1, 2, 2), &device)?;
    let out = conv_transpose2d(&x, &weight, None, 2, 0)?;
    assert_eq!(out.dims(), [1, 1, 4, 4]);
    let got = out.flatten_all()?.to_vec1::<f32>()?;
    let expected = [
        1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0,
    ];
    for (g, e) in got.iter().zip(expected) {
        assert!((g - e).abs() < 1e-6, "{g} vs {e}");
    }
    Ok(())
}

#[test]
fn conv_transpose2d_validates_arguments() -> Result<()> {
    let device = Device::Cpu;
    let x = Tensor::randn(0f32, 1f32, (1, 2, 3, 3), &device)?;
    let weight = Tensor::randn(0f32, 1f32, (1, 1, 2, 2), &device)?;
    assert!(conv_transpose2d(&x, &weight, None, 2, 0).is_err());
    let matching = Tensor::randn(0f32, 1f32, (2, 1, 2, 2), &device)?;
    let bias = Tensor::randn(0f32, 1f32, 5, &device)?;
    assert!(conv_transpose2d(&x, &matching, Some(&bias), 2, 0).is_err());
    Ok(())
}

#[test]
fn embedding_selects_rows() -> Result<()> {
    let device = Device::Cpu;
    let weight = Tensor::from_vec(
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        (4, 2),
        &device,
    )?;
    let indices = Tensor::from_vec(vec![3u32, 0], 2, &device)?;
    let out = embedding(&weight, &indices)?;
    assert_eq!(out.dims(), [2, 2]);
    let got = out.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(got, vec![7.0, 8.0, 1.0, 2.0]);
    let grid = Tensor::from_vec(vec![0u32, 1, 2, 3], (2, 2), &device)?;
    let out = embedding(&weight, &grid)?;
    assert_eq!(out.dims(), [2, 2, 2]);
    Ok(())
}

#[test]
fn embedding_validates_arguments() -> Result<()> {
    let device = Device::Cpu;
    let weight = Tensor::randn(0f32, 1f32, (4, 2), &device)?;
    let out_of_range = Tensor::from_vec(vec![9u32], 1, &device)?;
    assert!(embedding(&weight, &out_of_range).is_err());
    let rank_three = Tensor::randn(0f32, 1f32, (4, 2, 2), &device)?;
    let indices = Tensor::from_vec(vec![0u32], 1, &device)?;
    assert!(embedding(&rank_three, &indices).is_err());
    Ok(())
}
