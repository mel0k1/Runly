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

## Learning rate schedulers

A scheduler returns the learning rate for the current step and moves its internal counter; apply also sets it on the optimizer:

```rust
let mut opt = Adam::new_lr(model.vars(), 0.02)?;
let mut sched = CosineAnnealingLR::new(0.02, 300, 1e-4)?;
for epoch in 0..300 {
    sched.apply(&mut opt);
    let logits = model.forward(&x)?;
    let loss = cross_entropy(&logits, &y)?;
    opt.backward_step(&loss)?;
}
```

Inside are StepLR, ExponentialLR, and CosineAnnealingLR: the first cuts the lr by gamma every step_size steps, the second multiplies it by gamma on every step, and the third follows a cosine from the base value down to eta_min over t_max steps. The current value comes from lr(), and a scheduler works with any optimizer that implements the Optimizer trait.

Warmup wraps any scheduler: for the first warmup_steps steps the lr grows linearly up to peak, and then the inner schedule takes over:

```rust
let cosine = CosineAnnealingLR::new(0.02, 280, 1e-4)?;
let mut sched = Warmup::new(cosine, 20, 0.02)?;
```

OneCycleLR raises the lr from initial_lr to the peak over the warmup fraction of total_steps and follows a cosine down to final_lr, one full cycle for a single training run:

```rust
let mut sched = OneCycleLR::new(0.02, 300, 0.3, 0.001, 1e-4)?;
```

examples/sched.rs trains xor with warmup plus cosine and with one cycle, printing the lr as it goes.

## Gradient clipping

Adam and SGD have max_grad_norm: before each step the optimizer computes the norm of all gradients and, when it exceeds the limit, shrinks them proportionally. Clipping is set in the config or switched on at runtime and pairs well with warmup, when the first steps are not yet settled:

```rust
let mut opt = Adam::new(model.vars(), AdamConfig {
    lr: 0.02,
    max_grad_norm: Some(1.0),
    ..Default::default()
})?;
opt.set_max_grad_norm(Some(0.5));
```

last_grad_norm returns the gradient norm of the latest step when clipping is on — a handy value to watch for spikes in the training log.

## Convolutional networks

conv2d adds a convolutional layer, max_pool and avg_pool shrink the feature maps, and flatten unfolds them into a vector before the linear layers:

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

The conv2d arguments are input channels, output channels, kernel size, stride, and padding. The input has the shape (batch, channels, height, width), and the size must be divisible by the pooling kernel. max_pool takes the maximum in each window, avg_pool takes the average, and both work as standalone functions on a tensor when you need pooling without a model. global_avg_pool averages each feature map as a whole, turns any input into the shape (N, C, 1, 1), and does not require the size to be divisible; add flatten after it before a linear layer.

conv_transpose2d adds a transposed convolution, it grows the feature maps and shows up in decoders, autoencoders, and segmentation. The arguments are the same: input channels, output channels, kernel size, stride, and padding. The output size is (input − 1) · stride − 2 · padding + kernel, so a kernel of 2 with stride 2 and no padding doubles the map, and a kernel of 4 with stride 2 and padding 1 does the same:

```rust
model.conv2d(16, 8, 3, 1, 1)?;
model.relu();
model.conv_transpose2d(8, 4, 2, 2, 0)?;
model.tanh();
```

The convolution shrinks, the transposed one expands, together they form an encoder and a decoder. In save/load the layer is stored under its architecture code with its stride and padding, so the model is restored in full. examples/autoencoder.rs trains such an autoencoder: a conv2d encoder squeezes an 8×8 image down to 4×2×2 and a conv_transpose2d decoder restores it, MSE drops from about 0.76 to 0.001 in 500 epochs.

adaptive_avg_pool and adaptive_max_pool bring a feature map to a given output size regardless of the input: the window and the step are chosen automatically, and the input can be of any size, even smaller than the output. This is handy in front of a classifier for images of arbitrary sizes:

```rust
model.conv2d(3, 16, 3, 1, 1)?;
model.relu();
model.adaptive_avg_pool(4);
model.flatten();
model.linear(16 * 4 * 4, 10)?;
```

## BatchNorm and dropout

batch_norm goes after a linear or convolutional layer and normalizes activations across the batch; the input can be (N, C) or (N, C, H, W). dropout turns off a random fraction of neurons during training:

```rust
model.conv2d(1, 8, 5, 1, 2)?;
model.batch_norm(8)?;
model.relu();
model.max_pool(2);
model.dropout(0.25);
```

The batch_norm layer keeps running mean and variance: forward computes batch statistics and updates the running ones. For inference and test evaluation there is forward_eval — it works on the accumulated statistics, does not update them, and dropout simply switches off:

```rust
let acc = accuracy(&model.forward_eval(&test_x)?, &test_y)?;
```

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
| `model` | Linear, Conv2d, ConvTranspose2d, batch_norm, max_pool, avg_pool, adaptive_avg_pool, adaptive_max_pool, global_avg_pool, dropout, flatten, and activations: relu, tanh, sigmoid, gelu, softmax |
| `optim` | Adam and SGD with momentum, weight decay, and gradient clipping, plus AdamW from candle |
| `sched` | learning rate schedulers: StepLR, ExponentialLR, CosineAnnealingLR, Warmup, OneCycleLR |
| `checkpoint` | best model tracking and early stopping by metric |
| `loss` | cross entropy, MSE, Huber, binary cross entropy with logits |
| `data` | xor and two spirals generators, a CSV loader, train/test split, mini batches |
| `gradcheck` | analytical gradients checked against numerical ones |
| `metrics` | argmax, accuracy, confusion matrix, precision, recall, F1 |
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

The file stores the weights and the architecture description, including the convolution parameters, so load returns a model you can keep training. save_half writes the weights in f16, the file comes out roughly half the size, and load upcasts the precision back to f32 on its own:

```rust
model.save_half("model.safetensors")?;
let restored = Model::load("model.safetensors", &device)?;
```

During training it is better to keep the best model rather than the latest one. Checkpointer compares the metric with the previous one and rewrites the file only when the metric improved; the best value is available through best:

```rust
let mut best = Checkpointer::new("best.safetensors");
for epoch in 0..300 {
    sched.apply(&mut opt);
    let logits = model.forward(&x)?;
    let loss = cross_entropy(&logits, &y)?;
    opt.backward_step(&loss)?;
    best.update(&model, loss.to_dtype(DType::F64)?.to_scalar::<f64>()?)?;
}
let model = Model::load("best.safetensors", &device)?;
```

EarlyStopping sits on top of Checkpointer: it watches the metric, writes the best model into the same file, and stops the training when there have been no improvements for a while. An improvement means the metric dropped by more than min_delta; once patience epochs pass without one, update returns true and the loop can break:

```rust
let mut stop = EarlyStopping::new("best.safetensors", 10, 1e-4)?;
for epoch in 0..300 {
    sched.apply(&mut opt);
    let logits = model.forward(&x)?;
    let loss = cross_entropy(&logits, &y)?;
    opt.backward_step(&loss)?;
    let val = mse(&model.forward_eval(&val_x)?, &val_y)?;
    let val = val.to_dtype(DType::F64)?.to_scalar::<f64>()?;
    if stop.update(&model, val)? {
        break;
    }
}
let model = Model::load("best.safetensors", &device)?;
```

best returns the best metric, bad_epochs counts the epochs without improvement in a row, and stopped tells whether the training was stopped.

## Classification metrics

confusion_matrix builds the confusion matrix: rows are the true classes, columns are the predicted ones. per_class_metrics breaks it down into precision, recall, F1, and support per class, and macro_f1 averages F1 across classes:

```rust
let matrix = confusion_matrix(&logits, &labels, 10)?;
for (class, m) in per_class_metrics(&matrix).iter().enumerate() {
    println!(
        "class {class}: precision {:.3} recall {:.3} f1 {:.3}",
        m.precision, m.recall, m.f1
    );
}
println!("macro f1 {:.3}", macro_f1(&matrix));
```

A class with no predictions and no examples yields zero precision, recall, and F1 instead of a division by zero.

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

Eighty tests: Adam and SGD steps with weight decay and gradient clipping are checked against hand computed values, clipping cuts by the global norm across all parameters, the schedulers produce exact schedules and really move the optimizer lr, the gradient check passes on linear and convolutional architectures, including the transposed convolution, batch_norm, and every kind of pooling, avg_pool, global_avg_pool, adaptive_avg_pool, and adaptive_max_pool are checked against hand computed values, including inputs that are not evenly divisible, a network with a transposed convolution learns upsampling, xor, spirals, and a convolutional network with batch_norm actually train, saved weights round trip in f32 and f16 without distortion, batch_norm switches between training and inference correctly, dropout zeroes out the expected fraction of neurons, the CSV loader parses quoted fields and reports broken rows, Checkpointer and EarlyStopping keep the best model, early stopping fires on patience and min_delta, and the confusion matrix and per-class metrics match hand computed values. GitHub Actions runs fmt, clippy, and the tests on every push.

## MNIST

```bash
cargo run --release --features mnist --example mnist
cargo run --release --features mnist --example cnn
```

The `mnist` feature adds a loader: on the first run it downloads four files (about 11 MB) into the data folder, unpacks them, and builds tensors. mnist trains a perceptron, cnn solves the same task with a convolutional network.

## License

Apache-2.0, see the LICENSE file.
