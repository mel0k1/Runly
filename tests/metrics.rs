use runly::prelude::*;

#[test]
fn confusion_matrix_counts_predictions() -> Result<()> {
    let device = Device::Cpu;
    let logits = Tensor::from_vec(
        vec![
            10.0f32, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0, 10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0,
            10.0, 0.0,
        ],
        (6, 3),
        &device,
    )?;
    let labels = Tensor::from_vec(vec![0u32, 1, 2, 1, 0, 1], 6, &device)?;
    let matrix = confusion_matrix(&logits, &labels, 3)?;
    assert_eq!(matrix, vec![vec![1, 1, 0], vec![1, 2, 0], vec![0, 0, 1]]);
    let acc = accuracy(&logits, &labels)?;
    let diagonal: usize = matrix.iter().enumerate().map(|(k, row)| row[k]).sum();
    let total: usize = matrix.iter().map(|row| row.iter().sum::<usize>()).sum();
    assert!((acc - diagonal as f64 / total as f64).abs() < 1e-9);
    Ok(())
}

#[test]
fn per_class_metrics_match_reference() -> Result<()> {
    let matrix = vec![vec![1, 1, 0], vec![1, 2, 0], vec![0, 0, 1]];
    let metrics = per_class_metrics(&matrix);
    assert_eq!(metrics.len(), 3);
    assert!((metrics[0].precision - 0.5).abs() < 1e-9);
    assert!((metrics[0].recall - 0.5).abs() < 1e-9);
    assert!((metrics[0].f1 - 0.5).abs() < 1e-9);
    assert_eq!(metrics[0].support, 2);
    assert!((metrics[1].precision - 2.0 / 3.0).abs() < 1e-9);
    assert!((metrics[1].recall - 2.0 / 3.0).abs() < 1e-9);
    assert!((metrics[1].f1 - 2.0 / 3.0).abs() < 1e-9);
    assert_eq!(metrics[1].support, 3);
    assert!((metrics[2].precision - 1.0).abs() < 1e-9);
    assert_eq!(metrics[2].support, 1);
    let expected_macro = (0.5 + 2.0 / 3.0 + 1.0) / 3.0;
    assert!((macro_f1(&matrix) - expected_macro).abs() < 1e-9);
    Ok(())
}

#[test]
fn per_class_metrics_handles_empty_class() -> Result<()> {
    let matrix = vec![vec![2, 0], vec![0, 0]];
    let metrics = per_class_metrics(&matrix);
    assert!((metrics[0].f1 - 1.0).abs() < 1e-9);
    assert_eq!(metrics[0].support, 2);
    assert!((metrics[1].precision - 0.0).abs() < 1e-9);
    assert!((metrics[1].recall - 0.0).abs() < 1e-9);
    assert!((metrics[1].f1 - 0.0).abs() < 1e-9);
    assert_eq!(metrics[1].support, 0);
    assert!((macro_f1(&matrix) - 0.5).abs() < 1e-9);
    Ok(())
}

#[test]
fn confusion_matrix_rejects_bad_input() -> Result<()> {
    let device = Device::Cpu;
    let logits = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (2, 2), &device)?;
    let labels = Tensor::from_vec(vec![0u32, 5], 2, &device)?;
    assert!(confusion_matrix(&logits, &labels, 2).is_err());
    let short_labels = Tensor::from_vec(vec![0u32], 1, &device)?;
    assert!(confusion_matrix(&logits, &short_labels, 2).is_err());
    assert!(confusion_matrix(&logits, &short_labels, 0).is_err());
    Ok(())
}

#[test]
fn metrics_work_on_trained_model() -> Result<()> {
    let device = Device::Cpu;
    let data = xor(32, 0.12, 11);
    let (x, y) = data.tensors(&device)?;
    let mut model = Model::new(&device);
    model.linear(2, 8)?;
    model.tanh();
    model.linear(8, 2)?;
    let mut opt = Adam::new_lr(model.vars(), 0.05)?;
    for _ in 0..500 {
        let logits = model.forward(&x)?;
        let loss = cross_entropy(&logits, &y)?;
        opt.backward_step(&loss)?;
    }
    let logits = model.forward(&x)?;
    let matrix = confusion_matrix(&logits, &y, 2)?;
    let diagonal: usize = matrix.iter().enumerate().map(|(k, row)| row[k]).sum();
    let total: usize = matrix.iter().map(|row| row.iter().sum::<usize>()).sum();
    let acc = accuracy(&logits, &y)?;
    assert!((acc - diagonal as f64 / total as f64).abs() < 1e-9);
    assert!(acc >= 0.95, "accuracy {acc}");
    for m in per_class_metrics(&matrix) {
        assert!(m.precision >= 0.9, "precision {}", m.precision);
        assert!(m.recall >= 0.9, "recall {}", m.recall);
    }
    Ok(())
}
