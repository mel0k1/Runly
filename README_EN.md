# Runly

[![CI](https://github.com/mel0k1/Runly/actions/workflows/ci.yml/badge.svg)](https://github.com/mel0k1/Runly/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Neural networks in Rust. Runly assembles networks from linear and convolutional layers, activations, and pooling, computes gradients through autograd, and stores trained weights in safetensors. The tensor math runs on candle by Hugging Face; on top of it sits a small API with few moving parts.

Russian version of this README: [README.md](README.md)

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
    let device = device()?;
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

## Convolutional networks

conv2d adds a convolutional layer, max_pool shrinks the feature maps, and flatten unfolds them into a vector before the linear layers:

```rust
let mut model = Model::new(&device);
model.conv2d(1, 8, 5, 1, 2)?;
model.relu();
model.max_pool(2);
model.conv2d(8, 16, 5, 1, 2)?;
model.relu();
model.max_pool(2);
model.flatten();
model.linear(16 * 7 * 7, 10)?;
```

The conv2d arguments are input channels, output channels, kernel size, stride, and padding. The input has the shape (batch, channels, height, width), and the size must be divisible by the pooling kernel. max_pool also works as a standalone function on a tensor when you need pooling without a model.

The save/load pair stores the architecture description together with the stride and padding, so a convolutional model is restored from the file in full and is ready for further training.

## CSV

The loader reads numeric tables: every column except the label becomes an f32 feature, the label is u32. By default the delimiter is a comma, the first row is the header, and the label sits in the last column:

```rust
let ds = load_csv("iris.csv")?;
let ds = load_csv_named("iris.csv", "species")?;
```

CsvConfig lets you pick the delimiter, disable the header, and select the label column by index. Quotes and commas inside quotes are parsed according to the CSV rules, and a parsing error names the row and the column. Labels must be integers starting from zero. The examples/csv.rs example writes spirals to a CSV file, reads it back, and trains a network on it.

## GPU

device() returns a CUDA device when the library is built with the cuda feature and CPU otherwise, so the same code works in both modes:

```bash
cargo run --release --features cuda --example cnn
```

A CUDA Toolkit installation is required: candle compiles its kernels during the build, so the first compilation takes noticeably longer than usual.

## What is inside

| Module | What it does |
|--------|--------------|
| `model` | Linear, Conv2d, max_pool, flatten, and activations: relu, tanh, sigmoid, gelu, softmax |
| `optim` | Adam and SGD with momentum, plus AdamW from candle |
| `loss` | cross entropy, MSE, Huber, binary cross entropy with logits |
| `data` | xor and two spirals generators, a CSV loader, train/test split, mini batches |
| `gradcheck` | analytical gradients checked against numerical ones |
| `metrics` | argmax and accuracy |
| `mnist` | MNIST loader behind the `mnist` feature |

## Tasks used to verify the library

| Task | Network | Result |
|------|---------|--------|
| XOR | 2-16-2, tanh | accuracy 1.000, loss 0.002 in 300 epochs |
| two spirals | 2-64-64-2, relu | test accuracy 1.000 in 800 epochs |
| MNIST | 784-128-10, relu | test accuracy 0.972 in 5 epochs |
| MNIST CNN | 8-5x5 and 16-5x5 with maxpool | test accuracy 0.985 in 5 epochs |

The MLP trains on MNIST in about a minute on two CPU cores, the convolutional network takes about seven on the same cores.

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

Features are f32, labels are u32. After that, the training loop from the example above works as is. If your data lives in a table, the CSV loader from the section above is the easier path.

## Saving and loading

```rust
model.save("model.safetensors")?;
let restored = Model::load("model.safetensors", &device)?;
```

The file stores the weights and the architecture description, including the convolution parameters, so load returns a model you can keep training.

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

Fifteen tests: Adam and SGD steps are checked against hand computed values, the gradient check passes on linear and convolutional architectures plus pooling on its own, xor, spirals, and a convolutional network on images actually train, saved weights round trip without distortion, and the CSV loader parses quoted fields and reports broken rows. GitHub Actions runs fmt, clippy, and the tests on every push.

## MNIST

```bash
cargo run --release --features mnist --example mnist
cargo run --release --features mnist --example cnn
```

The `mnist` feature adds a loader: on the first run it downloads four files (about 11 MB) into the data folder, unpacks them, and builds tensors. mnist trains a perceptron, cnn solves the same task with a convolutional network.

## Roadmap

A compact weight format, batchnorm, and dropout. If something here matters to you, open an issue.

## License

Apache-2.0, see the LICENSE file.
