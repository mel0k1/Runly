use candle::{D, DType, Device, Module, Result, Tensor, Var};
use candle_nn::{Conv2d, Conv2dConfig, VarBuilder, VarMap, linear};
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

#[derive(Debug)]
enum Layer {
    Linear(candle_nn::Linear),
    Conv2d(Conv2d),
    MaxPool(usize),
    Flatten,
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
            .filter(|layer| matches!(layer, Layer::Linear(_) | Layer::Conv2d(_)))
            .count()
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

    pub fn max_pool(&mut self, kernel: usize) {
        self.layers.push(Layer::MaxPool(kernel));
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

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut out = xs.clone();
        for layer in &self.layers {
            out = match layer {
                Layer::Linear(l) => l.forward(&out)?,
                Layer::Conv2d(l) => l.forward(&out)?,
                Layer::MaxPool(k) => max_pool(&out, *k)?,
                Layer::Flatten => out.flatten_from(1)?,
                Layer::Activation(a) => a.forward(&out)?,
            };
        }
        Ok(out)
    }

    pub fn vars(&self) -> Vec<Var> {
        self.varmap.all_vars()
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
        let mut map: HashMap<String, Tensor> = HashMap::new();
        {
            let data = self.varmap.data().lock().expect("varmap poisoned");
            for (name, var) in data.iter() {
                map.insert(name.clone(), var.as_tensor().clone());
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
                Layer::MaxPool(k) => {
                    arch.push(CODE_MAX_POOL);
                    args.push(*k as u32);
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

    pub fn load<P: AsRef<Path>>(path: P, device: &Device) -> Result<Self> {
        let tensors = candle::safetensors::load(path.as_ref(), device)?;
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
                CODE_MAX_POOL => {
                    if arg_index >= args.len() {
                        candle::bail!("max_pool layer has no kernel size in meta.args");
                    }
                    let kernel = args[arg_index] as usize;
                    arg_index += 1;
                    model.layers.push(Layer::MaxPool(kernel));
                }
                CODE_FLATTEN => model.layers.push(Layer::Flatten),
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
