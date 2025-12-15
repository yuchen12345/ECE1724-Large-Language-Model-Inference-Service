// src/model.rs
use anyhow::{Error as E, Result};
use candle_core::Device;
use candle_core::quantized::gguf_file::Content; 

// Import model architectures
use candle_transformers::models::quantized_phi::ModelWeights as QPhiModel;
use candle_transformers::models::quantized_llama::ModelWeights as QMistralModel;

use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

#[path = "config.rs"]
mod config;
use config::Settings;

pub enum ModelEnum {
    Phi(QPhiModel),
    Mistral(QMistralModel),
    Llama3(QMistralModel),
}

pub struct LoadedModel {
    pub model: ModelEnum,
    pub tokenizer: Tokenizer,
    pub device: Device,
}

fn pick_device() -> Device {
    // macOS
    #[cfg(target_os = "macos")]
    {
        match Device::new_metal(0) {
            Ok(d) => {
                println!("Using Metal: {:?}", d);
                return d;
            }
            Err(e) => {
                println!("Metal init failed: {:?}", e);
                return Device::Cpu;
            }
        }
    }
    // Linux / Windows
    #[cfg(not(target_os = "macos"))]
    {
        match Device::new_cuda(0) {
            Ok(d) => {
                println!("Using CUDA: {:?}", d);
                return d;
            }
            Err(e) => {
                println!("CUDA init failed: {:?}", e);
                return Device::Cpu;
            }
        }
    }
}

impl LoadedModel {
    pub fn load(name: &str) -> Result<Self> {
        // Select available computing device
    let device = pick_device();
    println!("Loading model '{}' on {:?}...", name, device);

        // Load Configuration
        let settings = Settings::new()?;
        
        // Find specific model config by name
        let model_conf = settings.models.get(name)
            .ok_or_else(|| E::msg(format!("Model '{}' not found in config.toml", name)))?;
        println!("Config found: Arch={}, Repo={}", model_conf.arch, model_conf.repo);

        // Download Files using Config
        let api = Api::new()?;
        
        // Fetch Tokenizer
        let tokenizer_repo = api.repo(Repo::new(model_conf.tokenizer_repo.clone(), RepoType::Model));
        let tokenizer_filename = tokenizer_repo.get(&model_conf.tokenizer_file)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        // Fetch Weights
        let model_repo = api.repo(Repo::new(model_conf.repo.clone(), RepoType::Model));
        let model_filename = model_repo.get(&model_conf.file)?;
        let mut file = std::fs::File::open(&model_filename)?;
        let content = Content::read(&mut file)?;

        // Load Model based on Architecture defined in Config
        let model_enum = match model_conf.arch.as_str() {
            "phi" => {
                let model = QPhiModel::from_gguf(content, &mut file, &device)?;
                ModelEnum::Phi(model)
            },
            "mistral" => {
                let model = QMistralModel::from_gguf(content, &mut file, &device)?;
                ModelEnum::Mistral(model)
            },
            "llama3" => {
                let model = QMistralModel::from_gguf(content, &mut file, &device)?;
                ModelEnum::Llama3(model)
            },
            _ => return Err(E::msg(format!("Architecture '{}' not supported", model_conf.arch))),
        };

        Ok(Self {
            model: model_enum,
            tokenizer,
            device,
        })
    }
}