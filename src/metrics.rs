use candle::{D, DType, Result, Tensor};

pub fn argmax(logits: &Tensor) -> Result<Tensor> {
    logits.argmax(D::Minus1)
}

pub fn accuracy(logits: &Tensor, targets: &Tensor) -> Result<f64> {
    let hits = logits
        .argmax(D::Minus1)?
        .eq(targets)?
        .to_dtype(DType::F64)?
        .sum_all()?
        .to_scalar::<f64>()?;
    Ok(hits / targets.dims()[0] as f64)
}

pub fn confusion_matrix(
    logits: &Tensor,
    labels: &Tensor,
    num_classes: usize,
) -> Result<Vec<Vec<usize>>> {
    if num_classes == 0 {
        candle::bail!("num_classes must be positive");
    }
    let preds = logits.argmax(D::Minus1)?.to_vec1::<u32>()?;
    let labels = labels.to_vec1::<u32>()?;
    if preds.len() != labels.len() {
        candle::bail!(
            "{} predictions do not match {} labels",
            preds.len(),
            labels.len()
        );
    }
    let mut matrix = vec![vec![0usize; num_classes]; num_classes];
    for (pred, label) in preds.iter().zip(&labels) {
        let pred = *pred as usize;
        let label = *label as usize;
        if pred >= num_classes || label >= num_classes {
            candle::bail!("class {label} or {pred} is outside of {num_classes} classes");
        }
        matrix[label][pred] += 1;
    }
    Ok(matrix)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClassMetrics {
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub support: usize,
}

fn cell(matrix: &[Vec<usize>], row: usize, col: usize) -> usize {
    matrix
        .get(row)
        .and_then(|r| r.get(col))
        .copied()
        .unwrap_or(0)
}

pub fn per_class_metrics(matrix: &[Vec<usize>]) -> Vec<ClassMetrics> {
    let num_classes = matrix.len();
    let mut metrics = Vec::with_capacity(num_classes);
    for k in 0..num_classes {
        let tp = cell(matrix, k, k);
        let support: usize = matrix[k].iter().sum();
        let predicted: usize = (0..num_classes).map(|row| cell(matrix, row, k)).sum();
        let precision = if predicted > 0 {
            tp as f64 / predicted as f64
        } else {
            0.0
        };
        let recall = if support > 0 {
            tp as f64 / support as f64
        } else {
            0.0
        };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        metrics.push(ClassMetrics {
            precision,
            recall,
            f1,
            support,
        });
    }
    metrics
}

pub fn macro_f1(matrix: &[Vec<usize>]) -> f64 {
    if matrix.is_empty() {
        return 0.0;
    }
    per_class_metrics(matrix).iter().map(|m| m.f1).sum::<f64>() / matrix.len() as f64
}
