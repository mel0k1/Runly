# Runly

[![CI](https://github.com/mel0k1/Runly/actions/workflows/ci.yml/badge.svg)](https://github.com/mel0k1/Runly/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Neural networks in Rust. Runly assembles a multilayer perceptron from linear layers and activations, computes gradients through autograd, and stores trained weights in safetensors. The tensor math runs on candle by Hugging Face; on top of it sits a small API with few moving parts.

Russian version of this README: [README.md](README.md)

## Why another library

Python wrappers are everywhere, but building a neural network from scratch in Rust is harder than it should be: the candle API is low level, and the docs assume you already know what you want. Runly provides the minimum needed to train an MLP, and you can read all of it in one evening.

It also works as a first project if you want to see how autograd works inside. The whole library is a few hundred lines, and every training step is visible in three lines of an example.

## Quick start

You only need Rust:

```bash
git clone https://github.com/mel0k1/Runly.git
cd Runly
cargo run --release --example spirals
```

The example trains a 2-64-64-2 network on the two spirals dataset. The output looks like this:

```text
epoch   1  loss 1.0616  test accuracy 0.650
epoch 100  loss 0.2502  test accuracy 0.950
epoch 300  loss 0.0147  test accuracy 0.988
epoch 800  loss 0.0033  test accuracy 1.000
train accuracy 1.000
test accuracy 1.000
```

## Code example

```rust
use runly::prelude::*;

fn main() -> candle::Result<()> {
    let device = Device::Cpu;
    let data = xor(64, 0.15, 42);
    let (x, y) = data.tensors(&device)?;

    let mut model = Model::new(&device);
    model.linear(2, 16)?;
    model.tanh();
    model.linear(16, 2)?;

    let mut opt = Adam::new_lr(model.vars(), 0.02)?;
    for _ in 0..300 {
        let logits = model.forward(&x)?;
        let loss = cross_entropy(&logits, &y)?;
        opt.backward_step(&loss)?;
    }
    println!("accuracy {:.3}", accuracy(&model.forward(&x)?, &y)?);
    Ok(())
}
```

forward builds the computation graph, cross_entropy returns a scalar, and backward_step computes gradients and updates the parameters. The library has no sessions, no compilation stages, and no separate train function.

## What is inside

| Module | What it does |
|--------|--------------|
| `model` | a sequence of Linear layers and activations: relu, tanh, sigmoid, gelu, softmax |
| `optim` | Adam and SGD with momentum, plus AdamW from candle |
| `loss` | cross entropy, MSE, Huber, binary cross entropy with logits |
| `data` | xor and two spirals generators, train/test split, mini batches |
| `gradcheck` | analytical gradients checked against numerical ones |
| `metrics` | argmax and accuracy |
| `mnist` | MNIST loader behind the `mnist` feature |

## Tasks used to verify the library

| Task | Network | Result |
|------|---------|--------|
| XOR | 2-16-2, tanh | accuracy 1.000, loss 0.002 in 300 epochs |
| two spirals | 2-64-64-2, relu | test accuracy 1.000 in 800 epochs |
| MNIST | 784-128-10, relu | test accuracy 0.972 in 5 epochs |

MNIST trains in about a minute on two CPU cores.

## Your own data

A dataset is a flat feature vector and a label vector:

```rust
let ds = Dataset {
    x: features,
    y: labels,
    dim: 4,
    num_classes: 3,
};
let (x, y) = ds.tensors(&device)?;
```

Features are f32, labels are u32. After that, the training loop from the example above works as is.

## Saving and loading

```rust
model.save("model.safetensors")?;
let restored = Model::load("model.safetensors", &device)?;
```

The file stores the weights and the architecture description, so load returns a model you can keep training.

## Gradient checking

Autograd is easy to get wrong, so the tests compare it against numerical differentiation: every parameter gradient is computed two ways, and the difference must stay within tolerances. You can run the same check on your own model:

```rust
let reports = check_model(
    &model,
    &x,
    |logits| cross_entropy(logits, &y),
    GradCheckConfig::default(),
)?;
```

## Performance

examples/bench.rs measures f32 matmul. On two Xeon cores it reaches 140-160 GFLOPS; most of that speed comes from the gemm crate with hand written vectorization inside candle.

## Tests and CI

```bash
cargo test
```

Eight tests: Adam and SGD steps are checked against hand computed values, the gradient check passes on two architectures, xor and spirals actually train, and saved weights round trip without distortion. GitHub Actions runs fmt, clippy, and the tests on every push.

## MNIST

```bash
cargo run --release --features mnist --example mnist
```

The `mnist` feature adds a loader: on the first run it downloads four files (about 11 MB) into the data folder, unpacks them, and builds tensors.

## Roadmap

Convolutional layers, CSV loading, GPU support through the candle cuda feature, and a compact weight format. If something here matters to you, open an issue.

## License

Apache-2.0, see the LICENSE file.
