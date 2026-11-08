use candle::{Device, Result, Tensor};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use std::fs;
use std::path::Path;

#[derive(Debug)]
pub struct Dataset {
    pub x: Vec<f32>,
    pub y: Vec<u32>,
    pub dim: usize,
    pub num_classes: usize,
}

impl Dataset {
    pub fn len(&self) -> usize {
        self.y.len()
    }

    pub fn is_empty(&self) -> bool {
        self.y.is_empty()
    }

    pub fn tensors(&self, device: &Device) -> Result<(Tensor, Tensor)> {
        let x = Tensor::from_vec(self.x.clone(), (self.len(), self.dim), device)?;
        let y = Tensor::from_vec(self.y.clone(), self.len(), device)?;
        Ok((x, y))
    }

    pub fn train_test_split(&self, test_frac: f64, seed: u64) -> (Dataset, Dataset) {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut indices: Vec<usize> = (0..self.len()).collect();
        indices.shuffle(&mut rng);
        let cut = ((self.len() as f64 * (1.0 - test_frac)).round() as usize)
            .clamp(1, self.len().saturating_sub(1));
        (self.gather(&indices[..cut]), self.gather(&indices[cut..]))
    }

    fn gather(&self, indices: &[usize]) -> Dataset {
        let mut x = Vec::with_capacity(indices.len() * self.dim);
        for &i in indices {
            x.extend_from_slice(&self.x[i * self.dim..(i + 1) * self.dim]);
        }
        Dataset {
            x,
            y: indices.iter().map(|&i| self.y[i]).collect(),
            dim: self.dim,
            num_classes: self.num_classes,
        }
    }
}

fn normal(rng: &mut impl Rng) -> f64 {
    let u = 1.0 - rng.random::<f64>();
    let v = rng.random::<f64>();
    (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
}

pub fn xor(n_per_center: usize, noise: f64, seed: u64) -> Dataset {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let centers = [[0.0f64, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let labels = [0u32, 1, 1, 0];
    let mut x = Vec::new();
    let mut y = Vec::new();
    for (center, &label) in centers.iter().zip(labels.iter()) {
        for _ in 0..n_per_center {
            x.push((center[0] + noise * normal(&mut rng)) as f32);
            x.push((center[1] + noise * normal(&mut rng)) as f32);
            y.push(label);
        }
    }
    Dataset {
        x,
        y,
        dim: 2,
        num_classes: 2,
    }
}

pub fn spirals(n_per_class: usize, turns: f32, noise: f32, seed: u64) -> Dataset {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut x = Vec::new();
    let mut y = Vec::new();
    for class in 0..2u32 {
        for i in 0..n_per_class {
            let r = (i + 1) as f32 / n_per_class as f32;
            let t = turns * std::f32::consts::TAU * r
                + class as f32 * std::f32::consts::PI
                + normal(&mut rng) as f32 * noise;
            x.push(r * t.sin());
            x.push(r * t.cos());
            y.push(class);
        }
    }
    Dataset {
        x,
        y,
        dim: 2,
        num_classes: 2,
    }
}

pub fn batch_indices(total: usize, batch_size: usize, seed: u64) -> Vec<Vec<usize>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut indices: Vec<usize> = (0..total).collect();
    indices.shuffle(&mut rng);
    indices
        .chunks(batch_size.max(1))
        .map(<[usize]>::to_vec)
        .collect()
}

#[derive(Debug, Clone)]
pub struct CsvConfig {
    pub delimiter: char,
    pub header: bool,
    pub label: Option<usize>,
}

impl Default for CsvConfig {
    fn default() -> Self {
        Self {
            delimiter: ',',
            header: true,
            label: None,
        }
    }
}

fn parse_csv(text: &str, delimiter: char) -> Vec<Vec<String>> {
    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut row: Vec<String> = Vec::new();
    let mut field = String::new();
    let mut in_quotes = false;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if in_quotes {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    field.push('"');
                } else {
                    in_quotes = false;
                }
            } else {
                field.push(c);
            }
        } else if c == '"' && field.is_empty() {
            in_quotes = true;
        } else if c == delimiter {
            row.push(field.trim().to_string());
            field.clear();
        } else if c == '\r' {
            continue;
        } else if c == '\n' {
            row.push(field.trim().to_string());
            field.clear();
            rows.push(std::mem::take(&mut row));
        } else {
            field.push(c);
        }
    }
    if !field.is_empty() || !row.is_empty() {
        row.push(field.trim().to_string());
        rows.push(row);
    }
    rows.into_iter()
        .filter(|r| !(r.len() == 1 && r[0].is_empty()))
        .collect()
}

fn parse_number(cell: &str, row: usize, col: usize) -> Result<f32> {
    cell.parse::<f32>().map_err(|_| {
        candle::Error::Msg(format!(
            "row {} column {col}: cannot parse {cell:?} as number",
            row + 1
        ))
    })
}

fn parse_label(cell: &str, row: usize, col: usize) -> Result<u32> {
    if let Ok(v) = cell.parse::<u32>() {
        return Ok(v);
    }
    match cell.parse::<f32>() {
        Ok(v) if v >= 0.0 && v.fract() == 0.0 && v <= u32::MAX as f32 => Ok(v as u32),
        _ => candle::bail!(
            "row {} column {col}: cannot parse {cell:?} as label",
            row + 1
        ),
    }
}

fn rows_to_dataset(
    path: &Path,
    rows: &[Vec<String>],
    header: bool,
    label: Option<usize>,
) -> Result<Dataset> {
    if rows.is_empty() {
        candle::bail!("{}: no data rows", path.display());
    }
    let data = match header {
        true => &rows[1..],
        false => rows,
    };
    if data.is_empty() {
        candle::bail!("{}: header without data rows", path.display());
    }
    let width = data[0].len();
    if width < 2 {
        candle::bail!(
            "{}: expected at least 2 columns, got {width}",
            path.display()
        );
    }
    for (i, row) in data.iter().enumerate() {
        if row.len() != width {
            candle::bail!(
                "{}: row {} has {} cells, expected {width}",
                path.display(),
                i + 1,
                row.len()
            );
        }
    }
    let label_col = match label {
        Some(i) if i < width => i,
        Some(i) => {
            candle::bail!(
                "{}: label column {i} out of range, table has {width} columns",
                path.display()
            );
        }
        None => width - 1,
    };
    let dim = width - 1;
    let mut x = Vec::with_capacity(data.len() * dim);
    let mut y = Vec::with_capacity(data.len());
    let mut max_label = 0u32;
    for (i, row) in data.iter().enumerate() {
        for (j, cell) in row.iter().enumerate() {
            if j == label_col {
                let label = parse_label(cell, i, j)?;
                max_label = max_label.max(label);
                y.push(label);
            } else {
                x.push(parse_number(cell, i, j)?);
            }
        }
    }
    Ok(Dataset {
        x,
        y,
        dim,
        num_classes: max_label as usize + 1,
    })
}

pub fn load_csv_with<P: AsRef<Path>>(path: P, config: CsvConfig) -> Result<Dataset> {
    let path = path.as_ref();
    let text = fs::read_to_string(path)
        .map_err(|e| candle::Error::Msg(format!("{}: {e}", path.display())))?;
    let rows = parse_csv(&text, config.delimiter);
    rows_to_dataset(path, &rows, config.header, config.label)
}

pub fn load_csv<P: AsRef<Path>>(path: P) -> Result<Dataset> {
    load_csv_with(path, CsvConfig::default())
}

pub fn load_csv_named<P: AsRef<Path>>(path: P, label: &str) -> Result<Dataset> {
    let path = path.as_ref();
    let text = fs::read_to_string(path)
        .map_err(|e| candle::Error::Msg(format!("{}: {e}", path.display())))?;
    let rows = parse_csv(&text, ',');
    let header = rows
        .first()
        .ok_or_else(|| candle::Error::Msg(format!("{}: file is empty", path.display())))?;
    let index = header.iter().position(|h| h == label).ok_or_else(|| {
        candle::Error::Msg(format!(
            "{}: column {label:?} not found in header {:?}",
            path.display(),
            header
        ))
    })?;
    rows_to_dataset(path, &rows, true, Some(index))
}
