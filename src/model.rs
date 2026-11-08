use candle::{D, DType, Device, Module, Result, Tensor, Var};
use candle_nn::{VarBuilder, VarMap, linear};
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

#[derive(Debug)]
enum Layer {
    Linear(candle_nn::Linear),
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

    pub fn linear(&mut self, in_dim: usize, out_dim: usize) -> Result<()> {
        let index = self
            .layers
            .iter()
            .filter(|layer| matches!(layer, Layer::Linear(_)))
            .count();
        let layer = linear(in_dim, out_dim, self.builder(index))?;
        self.layers.push(Layer::Linear(layer));
        Ok(())
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
        let arch: Vec<u32> = self
            .layers
            .iter()
            .map(|layer| match layer {
                Layer::Linear(_) => 0,
                Layer::Activation(a) => a.code(),
            })
            .collect();
        map.insert(
            "meta.arch".to_string(),
            Tensor::from_vec(arch, self.layers.len(), &self.device)?,
        );
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
            None => vec![0; indices.len()],
        };
        let mut model = Self::new(device);
        let mut linear_index = 0usize;
        for &code in &arch {
            if code != 0 {
                model.layers.push(Layer::Activation(Act::from_code(code)?));
                continue;
            }
            let index = linear_index;
            linear_index += 1;
            if index >= indices.len() {
                candle::bail!("architecture refers to missing layers.{index}");
            }
            if indices[index] != index {
                candle::bail!("missing layers.{index} in {}", path.as_ref().display());
            }
            let weight = tensors
                .get(&format!("layers.{index}.weight"))
                .ok_or_else(|| candle::Error::Msg(format!("missing layers.{index}.weight")))?;
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
        if linear_index != indices.len() {
            candle::bail!("unmatched linear layers in {}", path.as_ref().display());
        }
        if model.layers.is_empty() {
            candle::bail!("no layers.* weights found in {}", path.as_ref().display());
        }
        Ok(model)
    }
}
