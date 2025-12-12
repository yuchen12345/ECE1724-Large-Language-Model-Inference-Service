// src/config.rs
use serde::Deserialize;
use std::collections::HashMap;
use config::Config;
use anyhow::Result;

#[derive(Debug, Deserialize)]
pub struct ModelConfig {
    pub arch: String,           
    pub repo: String,           // HuggingFace Repo for Weights
    pub file: String,           // GGUF Filename
    pub tokenizer_repo: String, // HuggingFace Repo for Tokenizer
    pub tokenizer_file: String, // Tokenizer Filename
}

#[derive(Debug, Deserialize)]
pub struct Settings {
    pub models: HashMap<String, ModelConfig>,
}

impl Settings {
    pub fn new() -> Result<Self> {
        let settings = Config::builder()
            .add_source(config::File::with_name("config"))
            .build()?;

        settings.try_deserialize().map_err(|e| anyhow::Error::msg(e.to_string()))
    }
}