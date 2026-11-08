use candle::{Device, Result, Tensor};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

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
