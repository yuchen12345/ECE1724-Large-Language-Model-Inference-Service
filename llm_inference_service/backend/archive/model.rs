use anyhow::Result;
use std::path::PathBuf;

use candle_core::Device;
use candle_transformers::models::phi::Config;
use tokenizers::Tokenizer;

use hf_hub::{api::sync::Api, Repo, RepoType};


pub struct LLM {
    pub tokenizer: Tokenizer,
    pub config: Config,
    pub model_file: PathBuf,
    pub device: Device,
}

pub fn load_model() -> Result<LLM> {
    let api = Api::new()?;
    let repo = api.repo(Repo::new("microsoft/phi-1_5".to_string(), RepoType::Model));

    let tokenizer_file = repo.get("tokenizer.json")?;
    let config_file = repo.get("config.json")?;
    let model_file = repo.get("model.safetensors")?;

    let tokenizer = Tokenizer::from_file(tokenizer_file)
        .map_err(anyhow::Error::msg)?;
    let config: Config = serde_json::from_str(&std::fs::read_to_string(config_file)?)?;

    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);

    Ok(LLM {
        tokenizer,
        config,
        model_file,
        device,
    })
}
