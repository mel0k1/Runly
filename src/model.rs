use candle::{D, DType, Device, IndexOp, Module, Result, Tensor, Var};
use candle_nn::{
    Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig, VarBuilder, VarMap, linear,
};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Act {
    Relu,
    Tanh,
    Sigmoid,
    Gelu,
    Softmax,
}

impl Act {
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Act::Relu => xs.relu(),
            Act::Tanh => xs.tanh(),
            Act::Sigmoid => candle_nn::ops::sigmoid(xs),
            Act::Gelu => xs.gelu_erf(),
            Act::Softmax => candle_nn::ops::softmax(xs, D::Minus1),
        }
    }

    fn code(self) -> u32 {
        match self {
            Act::Relu => 1,
            Act::Tanh => 2,
            Act::Sigmoid => 3,
            Act::Gelu => 4,
            Act::Softmax => 5,
        }
    }

    fn from_code(code: u32) -> Result<Self> {
        match code {
            1 => Ok(Act::Relu),
            2 => Ok(Act::Tanh),
            3 => Ok(Act::Sigmoid),
            4 => Ok(Act::Gelu),
            5 => Ok(Act::Softmax),
            _ => candle::bail!("unknown activation code {code}"),
        }
    }
}

const CODE_LINEAR: u32 = 0;
const CODE_CONV2D: u32 = 6;
const CODE_MAX_POOL: u32 = 7;
const CODE_FLATTEN: u32 = 8;
const CODE_BATCH_NORM: u32 = 9;
const CODE_DROPOUT: u32 = 10;
const CODE_AVG_POOL: u32 = 11;
const CODE_GLOBAL_AVG_POOL: u32 = 12;
const CODE_CONV_TRANSPOSE2D: u32 = 13;
const CODE_ADAPTIVE_AVG_POOL: u32 = 14;
const CODE_ADAPTIVE_MAX_POOL: u32 = 15;
const CODE_RNN: u32 = 16;
const CODE_GRU: u32 = 17;
const CODE_SELECT_LAST: u32 = 18;

pub fn avg_pool(xs: &Tensor, kernel: usize) -> Result<Tensor> {
    let (n, c, h, w) = xs.dims4()?;
    if h % kernel != 0 || w % kernel != 0 {
        candle::bail!("input {h}x{w} is not divisible by pool size {kernel}");
    }
    let oh = h / kernel;
    let ow = w / kernel;
    xs.reshape((n, c, oh, kernel, ow, kernel))?
        .transpose(3, 4)?
        .contiguous()?
        .reshape((n, c, oh, ow, kernel * kernel))?
        .mean(D::Minus1)
}

pub fn global_avg_pool(xs: &Tensor) -> Result<Tensor> {
    if xs.dims().len() != 4 {
        candle::bail!("global avg pool expects a (n, c, h, w) tensor");
    }
    xs.mean_keepdim((2, 3))
}

pub fn adaptive_avg_pool(xs: &Tensor, output: usize) -> Result<Tensor> {
    if output == 0 {
        candle::bail!("pool output size must be positive");
    }
    let (_, _, h, w) = xs.dims4()?;
    let mut rows = Vec::new();
    for i in 0..output {
        let h0 = i * h / output;
        let h1 = ((i + 1) * h).div_ceil(output);
        let mut cols = Vec::new();
        for j in 0..output {
            let w0 = j * w / output;
            let w1 = ((j + 1) * w).div_ceil(output);
            let window = xs.i((.., .., h0..h1, w0..w1))?;
            cols.push(window.mean((2, 3))?);
        }
        rows.push(Tensor::stack(&cols, 2)?);
    }
    Tensor::stack(&rows, 2)
}

pub fn adaptive_max_pool(xs: &Tensor, output: usize) -> Result<Tensor> {
    if output == 0 {
        candle::bail!("pool output size must be positive");
    }
    let (_, _, h, w) = xs.dims4()?;
    let mut rows = Vec::new();
    for i in 0..output {
        let h0 = i * h / output;
        let h1 = ((i + 1) * h).div_ceil(output);
        let mut cols = Vec::new();
        for j in 0..output {
            let w0 = j * w / output;
            let w1 = ((j + 1) * w).div_ceil(output);
            let window = xs.i((.., .., h0..h1, w0..w1))?;
            cols.push(window.max(3)?.max(2)?);
        }
        rows.push(Tensor::stack(&cols, 2)?);
    }
    Tensor::stack(&rows, 2)
}

pub fn max_pool(xs: &Tensor, kernel: usize) -> Result<Tensor> {
    let (n, c, h, w) = xs.dims4()?;
    if h % kernel != 0 || w % kernel != 0 {
        candle::bail!("input {h}x{w} is not divisible by pool size {kernel}");
    }
    let oh = h / kernel;
    let ow = w / kernel;
    xs.reshape((n, c, oh, kernel, ow, kernel))?
        .transpose(3, 4)?
        .contiguous()?
        .reshape((n, c, oh, ow, kernel * kernel))?
        .max(D::Minus1)
}

pub fn conv_transpose2d(
    xs: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    padding: usize,
) -> Result<Tensor> {
    let (_, in_c, _, _) = xs.dims4()?;
    let (w_in, w_out, k_h, k_w) = weight.dims4()?;
    if w_in != in_c {
        candle::bail!("weight expects {w_in} input channels, input has {in_c}");
    }
    if k_h != k_w {
        candle::bail!("rectangular kernel {k_h}x{k_w} is not supported");
    }
    let out = xs.conv_transpose2d(weight, padding, 0, stride, 1)?;
    match bias {
        None => Ok(out),
        Some(b) => {
            let channels = b.dims1()?;
            if channels != w_out {
                candle::bail!("bias expects {w_out} channels, got {channels}");
            }
            out.broadcast_add(&b.reshape((1, channels, 1, 1))?)
        }
    }
}

pub fn select_last_step(xs: &Tensor) -> Result<Tensor> {
    let (_, steps, _) = xs.dims3()?;
    xs.narrow(1, steps - 1, 1)?.squeeze(1)?.contiguous()
}

pub fn embedding(weight: &Tensor, indices: &Tensor) -> Result<Tensor> {
    if weight.dims().len() != 2 {
        candle::bail!("embedding weight must be a (vocab, dim) matrix");
    }
    let mut shape = indices.dims().to_vec();
    shape.push(weight.dim(D::Minus1)?);
    weight
        .index_select(&indices.flatten_all()?, 0)?
        .reshape(shape)
}

#[derive(Debug)]
struct BatchNorm {
    running_mean: Var,
    running_var: Var,
    weight: Tensor,
    bias: Tensor,
    momentum: f64,
    eps: f64,
}

impl BatchNorm {
    fn forward_train(&self, xs: &Tensor) -> Result<Tensor> {
        let mut inner = xs.dims().to_vec();
        inner.swap(0, 1);
        let x = xs.transpose(0, 1)?.flatten_from(1)?.contiguous()?;
        let mean = x.mean_keepdim(1)?;
        let centered = x.broadcast_sub(&mean)?;
        let var = centered.sqr()?.mean_keepdim(1)?;
        let n = x.dim(1)? as f64;
        let correction = if n > 1.0 { n / (n - 1.0) } else { 1.0 };
        let batch_mean = mean.flatten_all()?;
        let batch_var = (var.flatten_all()? * correction)?;
        let next_mean = ((self.running_mean.as_tensor() * (1.0 - self.momentum))?
            + (&batch_mean * self.momentum)?)?;
        let next_var = ((self.running_var.as_tensor() * (1.0 - self.momentum))?
            + (&batch_var * self.momentum)?)?;
        self.running_mean.set(&next_mean)?;
        self.running_var.set(&next_var)?;
        let out = centered.broadcast_div(&((var + self.eps)?.sqrt()?))?;
        let out = out
            .broadcast_mul(&self.weight.reshape(((), 1))?)?
            .broadcast_add(&self.bias.reshape(((), 1))?)?;
        out.reshape(inner.as_slice())?.transpose(0, 1)
    }

    fn forward_eval(&self, xs: &Tensor) -> Result<Tensor> {
        let shape: Vec<usize> = xs
            .dims()
            .iter()
            .enumerate()
            .map(|(i, d)| if i == 1 { *d } else { 1 })
            .collect();
        let mean = self.running_mean.as_tensor().reshape(shape.as_slice())?;
        let std = (self.running_var.as_tensor().reshape(shape.as_slice())? + self.eps)?.sqrt()?;
        xs.broadcast_sub(&mean)?
            .broadcast_div(&std)?
            .broadcast_mul(&self.weight.reshape(shape.as_slice())?)?
            .broadcast_add(&self.bias.reshape(shape.as_slice())?)
    }
}

#[derive(Debug)]
struct Rnn {
    ih: candle_nn::Linear,
    hh: candle_nn::Linear,
}

impl Rnn {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch, steps, _) = xs.dims3()?;
        let (hidden, _) = self.ih.weight().dims2()?;
        let mut state = Tensor::zeros((batch, hidden), DType::F32, xs.device())?;
        let mut states = Vec::with_capacity(steps);
        for step in 0..steps {
            let x = xs.narrow(1, step, 1)?.squeeze(1)?.contiguous()?;
            state = (self.ih.forward(&x)? + self.hh.forward(&state)?)?.tanh()?;
            states.push(state.clone());
        }
        Tensor::stack(&states, 1)
    }
}

#[derive(Debug)]
struct Gru {
    ir: candle_nn::Linear,
    hr: candle_nn::Linear,
    iz: candle_nn::Linear,
    hz: candle_nn::Linear,
    ni: candle_nn::Linear,
    nh: candle_nn::Linear,
}

impl Gru {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch, steps, _) = xs.dims3()?;
        let (hidden, _) = self.ir.weight().dims2()?;
        let mut state = Tensor::zeros((batch, hidden), DType::F32, xs.device())?;
        let mut states = Vec::with_capacity(steps);
        for step in 0..steps {
            let x = xs.narrow(1, step, 1)?.squeeze(1)?.contiguous()?;
            let r = candle_nn::ops::sigmoid(&(&self.ir.forward(&x)? + self.hr.forward(&state)?)?)?;
            let z = candle_nn::ops::sigmoid(&(&self.iz.forward(&x)? + self.hz.forward(&state)?)?)?;
            let n =
                (&self.ni.forward(&x)? + r.broadcast_mul(&self.nh.forward(&state)?)?)?.tanh()?;
            state = z
                .affine(-1f64, 1f64)?
                .broadcast_mul(&n)?
                .broadcast_add(&z.broadcast_mul(&state)?)?;
            states.push(state.clone());
        }
        Tensor::stack(&states, 1)
    }
}

#[derive(Debug)]
enum Layer {
    Linear(candle_nn::Linear),
    Conv2d(Conv2d),
    ConvTranspose2d(ConvTranspose2d),
    BatchNorm(BatchNorm),
    Rnn(Rnn),
    Gru(Gru),
    MaxPool(usize),
    AvgPool(usize),
    AdaptiveAvgPool(usize),
    AdaptiveMaxPool(usize),
    GlobalAvgPool,
    Flatten,
    SelectLast,
    Dropout(f64),
    Activation(Act),
}

pub struct Model {
    varmap: VarMap,
    device: Device,
    layers: Vec<Layer>,
}

impl Model {
    pub fn new(device: &Device) -> Self {
        Self {
            varmap: VarMap::new(),
            device: device.clone(),
            layers: Vec::new(),
        }
    }

    fn builder(&self, index: usize) -> VarBuilder<'_> {
        VarBuilder::from_varmap(&self.varmap, DType::F32, &self.device)
            .pp(format!("layers.{index}"))
    }

    fn param_index(&self) -> usize {
        self.layers
            .iter()
            .filter(|layer| {
                matches!(
                    layer,
                    Layer::Linear(_)
                        | Layer::Conv2d(_)
                        | Layer::ConvTranspose2d(_)
                        | Layer::BatchNorm(_)
                        | Layer::Rnn(_)
                        | Layer::Gru(_)
                )
            })
            .count()
    }

    fn var_handle(&self, name: &str) -> Result<Var> {
        let data = self.varmap.data().lock().expect("varmap poisoned");
        data.get(name)
            .cloned()
            .ok_or_else(|| candle::Error::Msg(format!("missing variable {name} in varmap")))
    }

    fn batch_norm_at(&self, index: usize, channels: usize, momentum: f64) -> Result<BatchNorm> {
        if !(0.0..=1.0).contains(&momentum) {
            candle::bail!("batch norm momentum must be in [0, 1], got {momentum}");
        }
        let vb = self.builder(index);
        let _ = vb.get_with_hints(channels, "running_mean", candle_nn::Init::Const(0.0))?;
        let _ = vb.get_with_hints(channels, "running_var", candle_nn::Init::Const(1.0))?;
        let weight = vb.get_with_hints(channels, "weight", candle_nn::Init::Const(1.0))?;
        let bias = vb.get_with_hints(channels, "bias", candle_nn::Init::Const(0.0))?;
        Ok(BatchNorm {
            running_mean: self.var_handle(&format!("layers.{index}.running_mean"))?,
            running_var: self.var_handle(&format!("layers.{index}.running_var"))?,
            weight,
            bias,
            momentum,
            eps: 1e-5,
        })
    }

    pub fn linear(&mut self, in_dim: usize, out_dim: usize) -> Result<()> {
        let index = self.param_index();
        let layer = linear(in_dim, out_dim, self.builder(index))?;
        self.layers.push(Layer::Linear(layer));
        Ok(())
    }

    pub fn conv2d(
        &mut self,
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
    ) -> Result<()> {
        let index = self.param_index();
        let config = Conv2dConfig {
            stride,
            padding,
            ..Default::default()
        };
        let layer = candle_nn::conv2d(
            in_channels,
            out_channels,
            kernel,
            config,
            self.builder(index),
        )?;
        self.layers.push(Layer::Conv2d(layer));
        Ok(())
    }

    pub fn conv_transpose2d(
        &mut self,
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
    ) -> Result<()> {
        let index = self.param_index();
        let config = ConvTranspose2dConfig {
            stride,
            padding,
            ..Default::default()
        };
        let layer = candle_nn::conv_transpose2d(
            in_channels,
            out_channels,
            kernel,
            config,
            self.builder(index),
        )?;
        self.layers.push(Layer::ConvTranspose2d(layer));
        Ok(())
    }

    pub fn max_pool(&mut self, kernel: usize) {
        self.layers.push(Layer::MaxPool(kernel));
    }

    pub fn avg_pool(&mut self, kernel: usize) {
        self.layers.push(Layer::AvgPool(kernel));
    }

    pub fn global_avg_pool(&mut self) {
        self.layers.push(Layer::GlobalAvgPool);
    }

    pub fn adaptive_avg_pool(&mut self, output: usize) {
        self.layers.push(Layer::AdaptiveAvgPool(output));
    }

    pub fn adaptive_max_pool(&mut self, output: usize) {
        self.layers.push(Layer::AdaptiveMaxPool(output));
    }

    pub fn rnn(&mut self, input_size: usize, hidden_size: usize) -> Result<()> {
        if input_size == 0 || hidden_size == 0 {
            candle::bail!(
                "rnn sizes must be positive, got input {input_size}, hidden {hidden_size}"
            );
        }
        let index = self.param_index();
        let layer = {
            let vb = self.builder(index);
            Rnn {
                ih: linear(input_size, hidden_size, vb.pp("ih"))?,
                hh: linear(hidden_size, hidden_size, vb.pp("hh"))?,
            }
        };
        self.layers.push(Layer::Rnn(layer));
        Ok(())
    }

    pub fn gru(&mut self, input_size: usize, hidden_size: usize) -> Result<()> {
        if input_size == 0 || hidden_size == 0 {
            candle::bail!(
                "gru sizes must be positive, got input {input_size}, hidden {hidden_size}"
            );
        }
        let index = self.param_index();
        let layer = {
            let vb = self.builder(index);
            Gru {
                ir: linear(input_size, hidden_size, vb.pp("ir"))?,
                hr: linear(hidden_size, hidden_size, vb.pp("hr"))?,
                iz: linear(input_size, hidden_size, vb.pp("iz"))?,
                hz: linear(hidden_size, hidden_size, vb.pp("hz"))?,
                ni: linear(input_size, hidden_size, vb.pp("ni"))?,
                nh: linear(hidden_size, hidden_size, vb.pp("nh"))?,
            }
        };
        self.layers.push(Layer::Gru(layer));
        Ok(())
    }

    pub fn select_last_step(&mut self) {
        self.layers.push(Layer::SelectLast);
    }

    pub fn batch_norm(&mut self, channels: usize) -> Result<()> {
        let index = self.param_index();
        let layer = self.batch_norm_at(index, channels, 0.1)?;
        self.layers.push(Layer::BatchNorm(layer));
        Ok(())
    }

    pub fn dropout(&mut self, p: f64) -> Result<()> {
        if !(0.0..1.0).contains(&p) {
            candle::bail!("dropout probability must be in [0, 1), got {p}");
        }
        self.layers.push(Layer::Dropout(p));
        Ok(())
    }

    pub fn flatten(&mut self) {
        self.layers.push(Layer::Flatten);
    }

    pub fn relu(&mut self) {
        self.layers.push(Layer::Activation(Act::Relu));
    }

    pub fn tanh(&mut self) {
        self.layers.push(Layer::Activation(Act::Tanh));
    }

    pub fn sigmoid(&mut self) {
        self.layers.push(Layer::Activation(Act::Sigmoid));
    }

    pub fn gelu(&mut self) {
        self.layers.push(Layer::Activation(Act::Gelu));
    }

    pub fn softmax(&mut self) {
        self.layers.push(Layer::Activation(Act::Softmax));
    }

    fn pass(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let mut out = xs.clone();
        for layer in &self.layers {
            out = match layer {
                Layer::Linear(l) => l.forward(&out)?,
                Layer::Conv2d(l) => l.forward(&out)?,
                Layer::ConvTranspose2d(l) => l.forward(&out)?,
                Layer::BatchNorm(b) => {
                    if train {
                        b.forward_train(&out)?
                    } else {
                        b.forward_eval(&out)?
                    }
                }
                Layer::Rnn(l) => l.forward(&out)?,
                Layer::Gru(l) => l.forward(&out)?,
                Layer::MaxPool(k) => max_pool(&out, *k)?,
                Layer::AvgPool(k) => avg_pool(&out, *k)?,
                Layer::AdaptiveAvgPool(k) => adaptive_avg_pool(&out, *k)?,
                Layer::AdaptiveMaxPool(k) => adaptive_max_pool(&out, *k)?,
                Layer::GlobalAvgPool => global_avg_pool(&out)?,
                Layer::Flatten => out.flatten_from(1)?,
                Layer::SelectLast => select_last_step(&out)?,
                Layer::Dropout(p) => {
                    if train {
                        candle_nn::ops::dropout(&out, *p as f32)?
                    } else {
                        out
                    }
                }
                Layer::Activation(a) => a.forward(&out)?,
            };
        }
        Ok(out)
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.pass(xs, true)
    }

    pub fn forward_eval(&self, xs: &Tensor) -> Result<Tensor> {
        self.pass(xs, false)
    }

    pub fn vars(&self) -> Vec<Var> {
        let data = self.varmap.data().lock().expect("varmap poisoned");
        let mut named: Vec<(String, Var)> =
            data.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        named.sort_by(|a, b| a.0.cmp(&b.0));
        named.into_iter().map(|(_, v)| v).collect()
    }

    pub fn varmap(&self) -> &VarMap {
        &self.varmap
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn num_params(&self) -> usize {
        self.varmap.all_vars().iter().map(|v| v.elem_count()).sum()
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        self.save_as(path, DType::F32)
    }

    pub fn save_as<P: AsRef<Path>>(&self, path: P, dtype: DType) -> Result<()> {
        let mut map: HashMap<String, Tensor> = HashMap::new();
        {
            let data = self.varmap.data().lock().expect("varmap poisoned");
            for (name, var) in data.iter() {
                map.insert(name.clone(), var.as_tensor().to_dtype(dtype)?);
            }
        }
        let mut arch: Vec<u32> = Vec::with_capacity(self.layers.len());
        let mut args: Vec<u32> = Vec::new();
        for layer in &self.layers {
            match layer {
                Layer::Linear(_) => arch.push(CODE_LINEAR),
                Layer::Activation(a) => arch.push(a.code()),
                Layer::Conv2d(c) => {
                    arch.push(CODE_CONV2D);
                    let config = c.config();
                    args.push(config.stride as u32);
                    args.push(config.padding as u32);
                }
                Layer::ConvTranspose2d(c) => {
                    arch.push(CODE_CONV_TRANSPOSE2D);
                    let config = c.config();
                    args.push(config.stride as u32);
                    args.push(config.padding as u32);
                }
                Layer::MaxPool(k) => {
                    arch.push(CODE_MAX_POOL);
                    args.push(*k as u32);
                }
                Layer::AvgPool(k) => {
                    arch.push(CODE_AVG_POOL);
                    args.push(*k as u32);
                }
                Layer::AdaptiveAvgPool(k) => {
                    arch.push(CODE_ADAPTIVE_AVG_POOL);
                    args.push(*k as u32);
                }
                Layer::AdaptiveMaxPool(k) => {
                    arch.push(CODE_ADAPTIVE_MAX_POOL);
                    args.push(*k as u32);
                }
                Layer::GlobalAvgPool => arch.push(CODE_GLOBAL_AVG_POOL),
                Layer::Rnn(_) => arch.push(CODE_RNN),
                Layer::Gru(_) => arch.push(CODE_GRU),
                Layer::SelectLast => arch.push(CODE_SELECT_LAST),
                Layer::BatchNorm(b) => {
                    arch.push(CODE_BATCH_NORM);
                    args.push((b.momentum * 10000.0).round() as u32);
                }
                Layer::Dropout(p) => {
                    arch.push(CODE_DROPOUT);
                    args.push((p * 10000.0).round() as u32);
                }
                Layer::Flatten => arch.push(CODE_FLATTEN),
            }
        }
        let n_layers = arch.len();
        map.insert(
            "meta.arch".to_string(),
            Tensor::from_vec(arch, n_layers, &self.device)?,
        );
        if !args.is_empty() {
            let n_args = args.len();
            map.insert(
                "meta.args".to_string(),
                Tensor::from_vec(args, n_args, &self.device)?,
            );
        }
        candle::safetensors::save(&map, path)
    }

    pub fn save_half<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        self.save_as(path, DType::F16)
    }

    pub fn load<P: AsRef<Path>>(path: P, device: &Device) -> Result<Self> {
        let mut tensors = candle::safetensors::load(path.as_ref(), device)?;
        for tensor in tensors.values_mut() {
            if tensor.dtype().is_float() && tensor.dtype() != DType::F32 {
                *tensor = tensor.to_dtype(DType::F32)?;
            }
        }
        let mut indices: Vec<usize> = tensors
            .keys()
            .filter_map(|key| {
                key.strip_prefix("layers.")?
                    .split('.')
                    .next()?
                    .parse::<usize>()
                    .ok()
            })
            .collect();
        indices.sort_unstable();
        indices.dedup();
        let arch: Vec<u32> = match tensors.get("meta.arch") {
            Some(t) => t.to_vec1::<u32>()?,
            None => vec![CODE_LINEAR; indices.len()],
        };
        let args: Vec<u32> = match tensors.get("meta.args") {
            Some(t) => t.to_vec1::<u32>()?,
            None => vec![],
        };
        let mut model = Self::new(device);
        let mut param_index = 0usize;
        let mut arg_index = 0usize;
        for &code in &arch {
            match code {
                CODE_LINEAR => {
                    let index = param_index;
                    param_index += 1;
                    if index >= indices.len() {
                        candle::bail!("architecture refers to missing layers.{index}");
                    }
                    if indices[index] != index {
                        candle::bail!("missing layers.{index} in {}", path.as_ref().display());
                    }
                    let weight =
                        tensors
                            .get(&format!("layers.{index}.weight"))
                            .ok_or_else(|| {
                                candle::Error::Msg(format!("missing layers.{index}.weight"))
                            })?;
                    let bias = tensors.get(&format!("layers.{index}.bias")).cloned();
                    let (out_dim, in_dim) = weight.dims2()?;
                    let vb = VarBuilder::from_varmap(&model.varmap, DType::F32, device)
                        .pp(format!("layers.{index}"));
                    let w = vb.get((out_dim, in_dim), "weight")?;
                    model
                        .varmap
                        .set_one(format!("layers.{index}.weight"), weight)?;
                    let b = match &bias {
                        Some(bias) => {
                            let tensor = vb.get(out_dim, "bias")?;
                            model.varmap.set_one(format!("layers.{index}.bias"), bias)?;
                            Some(tensor)
                        }
                        None => None,
                    };
                    model
                        .layers
                        .push(Layer::Linear(candle_nn::Linear::new(w, b)));
                }
                CODE_CONV2D => {
                    if arg_index + 2 > args.len() {
                        candle::bail!(
                            "conv2d layer {} has no stride and padding in meta.args",
                            param_index
                        );
                    }
                    let stride = args[arg_index] as usize;
                    let padding = args[arg_index + 1] as usize;
                    arg_index += 2;
                    let index = param_index;
                    param_index += 1;
                    if index >= indices.len() {
                        candle::bail!("architecture refers to missing layers.{index}");
                    }
                    if indices[index] != index {
                        candle::bail!("missing layers.{index} in {}", path.as_ref().display());
                    }
                    let weight =
                        tensors
                            .get(&format!("layers.{index}.weight"))
                            .ok_or_else(|| {
                                candle::Error::Msg(format!("missing layers.{index}.weight"))
                            })?;
                    let (out_c, in_c, k_h, k_w) = weight.dims4()?;
                    if k_h != k_w {
                        candle::bail!("rectangular kernel {k_h}x{k_w} is not supported");
                    }
                    let config = Conv2dConfig {
                        stride,
                        padding,
                        ..Default::default()
                    };
                    let vb = VarBuilder::from_varmap(&model.varmap, DType::F32, device)
                        .pp(format!("layers.{index}"));
                    let bias_key = format!("layers.{index}.bias");
                    let layer = match tensors.get(&bias_key) {
                        Some(bias) => {
                            let layer = candle_nn::conv2d(in_c, out_c, k_h, config, vb)?;
                            model
                                .varmap
                                .set_one(format!("layers.{index}.weight"), weight)?;
                            model.varmap.set_one(bias_key, bias)?;
                            layer
                        }
                        None => {
                            let layer = candle_nn::conv2d_no_bias(in_c, out_c, k_h, config, vb)?;
                            model
                                .varmap
                                .set_one(format!("layers.{index}.weight"), weight)?;
                            layer
                        }
                    };
                    model.layers.push(Layer::Conv2d(layer));
                }
                CODE_CONV_TRANSPOSE2D => {
                    if arg_index + 2 > args.len() {
                        candle::bail!(
                            "conv_transpose2d layer {} has no stride and padding in meta.args",
                            param_index
                        );
                    }
                    let stride = args[arg_index] as usize;
                    let padding = args[arg_index + 1] as usize;
                    arg_index += 2;
                    let index = param_index;
                    param_index += 1;
                    if index >= indices.len() {
                        candle::bail!("architecture refers to missing layers.{index}");
                    }
                    if indices[index] != index {
                        candle::bail!("missing layers.{index} in {}", path.as_ref().display());
                    }
                    let weight =
                        tensors
                            .get(&format!("layers.{index}.weight"))
                            .ok_or_else(|| {
                                candle::Error::Msg(format!("missing layers.{index}.weight"))
                            })?;
                    let (in_c, out_c, k_h, k_w) = weight.dims4()?;
                    if k_h != k_w {
                        candle::bail!("rectangular kernel {k_h}x{k_w} is not supported");
                    }
                    let config = ConvTranspose2dConfig {
                        stride,
                        padding,
                        ..Default::default()
                    };
                    let vb = VarBuilder::from_varmap(&model.varmap, DType::F32, device)
                        .pp(format!("layers.{index}"));
                    let bias_key = format!("layers.{index}.bias");
                    let layer = match tensors.get(&bias_key) {
                        Some(bias) => {
                            let layer = candle_nn::conv_transpose2d(in_c, out_c, k_h, config, vb)?;
                            model
                                .varmap
                                .set_one(format!("layers.{index}.weight"), weight)?;
                            model.varmap.set_one(bias_key, bias)?;
                            layer
                        }
                        None => {
                            let layer =
                                candle_nn::conv_transpose2d_no_bias(in_c, out_c, k_h, config, vb)?;
                            model
                                .varmap
                                .set_one(format!("layers.{index}.weight"), weight)?;
                            layer
                        }
                    };
                    model.layers.push(Layer::ConvTranspose2d(layer));
                }
                CODE_MAX_POOL => {
                    if arg_index >= args.len() {
                        candle::bail!("max_pool layer has no kernel size in meta.args");
                    }
                    let kernel = args[arg_index] as usize;
                    arg_index += 1;
                    model.layers.push(Layer::MaxPool(kernel));
                }
                CODE_RNN => {
                    let index = param_index;
                    param_index += 1;
                    if index >= indices.len() {
                        candle::bail!("architecture refers to missing layers.{index}");
                    }
                    if indices[index] != index {
                        candle::bail!("missing layers.{index} in {}", path.as_ref().display());
                    }
                    let weight = tensors
                        .get(&format!("layers.{index}.ih.weight"))
                        .ok_or_else(|| {
                            candle::Error::Msg(format!("missing layers.{index}.ih.weight"))
                        })?;
                    let (hidden, input) = weight.dims2()?;
                    let vb = VarBuilder::from_varmap(&model.varmap, DType::F32, device)
                        .pp(format!("layers.{index}"));
                    let layer = Rnn {
                        ih: linear(input, hidden, vb.pp("ih"))?,
                        hh: linear(hidden, hidden, vb.pp("hh"))?,
                    };
                    for name in ["ih.weight", "ih.bias", "hh.weight", "hh.bias"] {
                        let key = format!("layers.{index}.{name}");
                        let tensor = tensors
                            .get(&key)
                            .ok_or_else(|| candle::Error::Msg(format!("missing {key}")))?;
                        model.varmap.set_one(&key, tensor)?;
                    }
                    model.layers.push(Layer::Rnn(layer));
                }
                CODE_GRU => {
                    let index = param_index;
                    param_index += 1;
                    if index >= indices.len() {
                        candle::bail!("architecture refers to missing layers.{index}");
                    }
                    if indices[index] != index {
                        candle::bail!("missing layers.{index} in {}", path.as_ref().display());
                    }
                    let weight = tensors
                        .get(&format!("layers.{index}.ir.weight"))
                        .ok_or_else(|| {
                            candle::Error::Msg(format!("missing layers.{index}.ir.weight"))
                        })?;
                    let (hidden, input) = weight.dims2()?;
                    let vb = VarBuilder::from_varmap(&model.varmap, DType::F32, device)
                        .pp(format!("layers.{index}"));
                    let layer = Gru {
                        ir: linear(input, hidden, vb.pp("ir"))?,
                        hr: linear(hidden, hidden, vb.pp("hr"))?,
                        iz: linear(input, hidden, vb.pp("iz"))?,
                        hz: linear(hidden, hidden, vb.pp("hz"))?,
                        ni: linear(input, hidden, vb.pp("ni"))?,
                        nh: linear(hidden, hidden, vb.pp("nh"))?,
                    };
                    for name in [
                        "ir.weight",
                        "ir.bias",
                        "hr.weight",
                        "hr.bias",
                        "iz.weight",
                        "iz.bias",
                        "hz.weight",
                        "hz.bias",
                        "ni.weight",
                        "ni.bias",
                        "nh.weight",
                        "nh.bias",
                    ] {
                        let key = format!("layers.{index}.{name}");
                        let tensor = tensors
                            .get(&key)
                            .ok_or_else(|| candle::Error::Msg(format!("missing {key}")))?;
                        model.varmap.set_one(&key, tensor)?;
                    }
                    model.layers.push(Layer::Gru(layer));
                }
                CODE_AVG_POOL => {
                    if arg_index >= args.len() {
                        candle::bail!("avg_pool layer has no kernel size in meta.args");
                    }
                    let kernel = args[arg_index] as usize;
                    arg_index += 1;
                    model.layers.push(Layer::AvgPool(kernel));
                }
                CODE_ADAPTIVE_AVG_POOL => {
                    if arg_index >= args.len() {
                        candle::bail!("adaptive avg pool has no output size in meta.args");
                    }
                    let output = args[arg_index] as usize;
                    arg_index += 1;
                    model.layers.push(Layer::AdaptiveAvgPool(output));
                }
                CODE_ADAPTIVE_MAX_POOL => {
                    if arg_index >= args.len() {
                        candle::bail!("adaptive max pool has no output size in meta.args");
                    }
                    let output = args[arg_index] as usize;
                    arg_index += 1;
                    model.layers.push(Layer::AdaptiveMaxPool(output));
                }
                CODE_GLOBAL_AVG_POOL => model.layers.push(Layer::GlobalAvgPool),
                CODE_BATCH_NORM => {
                    if arg_index >= args.len() {
                        candle::bail!("batch_norm layer has no momentum in meta.args");
                    }
                    let momentum = args[arg_index] as f64 / 10000.0;
                    arg_index += 1;
                    let index = param_index;
                    param_index += 1;
                    if index >= indices.len() {
                        candle::bail!("architecture refers to missing layers.{index}");
                    }
                    if indices[index] != index {
                        candle::bail!("missing layers.{index} in {}", path.as_ref().display());
                    }
                    let mean = tensors
                        .get(&format!("layers.{index}.running_mean"))
                        .ok_or_else(|| {
                            candle::Error::Msg(format!("missing layers.{index}.running_mean"))
                        })?;
                    let channels = mean.dims1()?;
                    let layer = model.batch_norm_at(index, channels, momentum)?;
                    for key in [
                        format!("layers.{index}.running_mean"),
                        format!("layers.{index}.running_var"),
                        format!("layers.{index}.weight"),
                        format!("layers.{index}.bias"),
                    ] {
                        let tensor = tensors
                            .get(&key)
                            .ok_or_else(|| candle::Error::Msg(format!("missing {key}")))?;
                        model.varmap.set_one(&key, tensor)?;
                    }
                    model.layers.push(Layer::BatchNorm(layer));
                }
                CODE_DROPOUT => {
                    if arg_index >= args.len() {
                        candle::bail!("dropout layer has no probability in meta.args");
                    }
                    let p = args[arg_index] as f64 / 10000.0;
                    arg_index += 1;
                    model.dropout(p)?;
                }
                CODE_FLATTEN => model.layers.push(Layer::Flatten),
                CODE_SELECT_LAST => model.layers.push(Layer::SelectLast),
                other => {
                    if other <= 5 {
                        model.layers.push(Layer::Activation(Act::from_code(other)?));
                    } else {
                        candle::bail!("unknown architecture code {other}");
                    }
                }
            }
        }
        if arg_index != args.len() {
            candle::bail!("unmatched meta.args in {}", path.as_ref().display());
        }
        if param_index != indices.len() {
            candle::bail!(
                "unmatched parametrized layers in {}",
                path.as_ref().display()
            );
        }
        if model.layers.is_empty() {
            candle::bail!("no layers.* weights found in {}", path.as_ref().display());
        }
        Ok(model)
    }
}
