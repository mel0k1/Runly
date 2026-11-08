use runly::prelude::*;

fn gflops(m: usize, k: usize, n: usize, iters: usize) -> Result<f64> {
    let device = Device::Cpu;
    let a = Tensor::randn(0f32, 1f32, (m, k), &device)?;
    let b = Tensor::randn(0f32, 1f32, (k, n), &device)?;
    let _ = a.matmul(&b)?;
    let start = std::time::Instant::now();
    for _ in 0..iters {
        let c = a.matmul(&b)?;
        std::hint::black_box(c);
    }
    let secs = start.elapsed().as_secs_f64();
    Ok(2.0 * (m * k * n) as f64 * iters as f64 / secs / 1e9)
}

fn main() -> Result<()> {
    println!("matmul f32, cpu");
    for (m, k, n) in [
        (256usize, 256usize, 256usize),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
    ] {
        let iters = if m <= 512 { 20 } else { 5 };
        println!(
            "{m:>5} x {k:<5} x {n:<5}  {:8.2} gflops",
            gflops(m, k, n, iters)?
        );
    }
    Ok(())
}
