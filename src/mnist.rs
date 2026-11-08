use candle::{Device, Result, Tensor};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

const BASE_URL: &str = "https://ossci-datasets.s3.amazonaws.com/mnist";

pub struct Mnist {
    pub train_images: Tensor,
    pub train_labels: Tensor,
    pub test_images: Tensor,
    pub test_labels: Tensor,
}

pub fn load(device: &Device) -> Result<Mnist> {
    load_dir(Path::new("data"), device)
}

pub fn load_dir(data_dir: &Path, device: &Device) -> Result<Mnist> {
    let train_images = prepare(
        data_dir,
        "train-images-idx3-ubyte.gz",
        "train-images-idx3-ubyte",
    )?;
    let train_labels = prepare(
        data_dir,
        "train-labels-idx1-ubyte.gz",
        "train-labels-idx1-ubyte",
    )?;
    let test_images = prepare(
        data_dir,
        "t10k-images-idx3-ubyte.gz",
        "t10k-images-idx3-ubyte",
    )?;
    let test_labels = prepare(
        data_dir,
        "t10k-labels-idx1-ubyte.gz",
        "t10k-labels-idx1-ubyte",
    )?;
    Ok(Mnist {
        train_images: read_images(&train_images, device)?,
        train_labels: read_labels(&train_labels, device)?,
        test_images: read_images(&test_images, device)?,
        test_labels: read_labels(&test_labels, device)?,
    })
}

fn prepare(data_dir: &Path, gz_name: &str, raw_name: &str) -> Result<PathBuf> {
    fs::create_dir_all(data_dir)?;
    let raw = data_dir.join(raw_name);
    if raw.exists() {
        return Ok(raw);
    }
    let gz = data_dir.join(gz_name);
    if !gz.exists() {
        let url = format!("{BASE_URL}/{gz_name}");
        let response = ureq::get(&url)
            .call()
            .map_err(|e| candle::Error::Msg(format!("download {url}: {e}")))?;
        let mut reader = response.into_reader();
        let mut file = fs::File::create(&gz)?;
        std::io::copy(&mut reader, &mut file)?;
    }
    let mut decoder = flate2::read::GzDecoder::new(fs::File::open(&gz)?);
    let mut bytes = Vec::new();
    decoder.read_to_end(&mut bytes)?;
    fs::write(&raw, bytes)?;
    Ok(raw)
}

fn be32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_be_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

fn read_images(path: &Path, device: &Device) -> Result<Tensor> {
    let bytes = fs::read(path)?;
    if be32(&bytes, 0) != 2051 {
        candle::bail!("bad idx3 magic in {}", path.display());
    }
    let n = be32(&bytes, 4) as usize;
    let rows = be32(&bytes, 8) as usize;
    let cols = be32(&bytes, 12) as usize;
    let data: Vec<f32> = bytes[16..].iter().map(|b| f32::from(*b) / 255.0).collect();
    if data.len() != n * rows * cols {
        candle::bail!("size mismatch in {}", path.display());
    }
    Tensor::from_vec(data, (n, rows * cols), device)
}

fn read_labels(path: &Path, device: &Device) -> Result<Tensor> {
    let bytes = fs::read(path)?;
    if be32(&bytes, 0) != 2049 {
        candle::bail!("bad idx1 magic in {}", path.display());
    }
    let n = be32(&bytes, 4) as usize;
    if bytes.len() < 8 + n {
        candle::bail!("truncated labels in {}", path.display());
    }
    let data: Vec<u32> = bytes[8..8 + n].iter().map(|b| u32::from(*b)).collect();
    Tensor::from_vec(data, n, device)
}
