// src/config.rs
use anyhow::{Context, Result};
use config::Config;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Deserialize, Clone)]
#[allow(dead_code)]
pub struct ModelConfig {
    pub arch: String,
    pub repo: String,           // HuggingFace Repo for Weights
    pub file: String,           // GGUF Filename
    pub tokenizer_repo: String, // HuggingFace Repo for Tokenizer
    pub tokenizer_file: String, // Tokenizer Filename
}

#[derive(Debug, Deserialize, Clone)]
#[allow(dead_code)]
pub struct Settings {
    pub models: HashMap<String, ModelConfig>,
}

#[allow(dead_code)]
impl Settings {
    // Load settings from config
    pub fn new() -> Result<Self> {
        let built = Config::builder()
            .add_source(config::File::with_name("config"))
            .build()
            .context("failed to build config (expected config.{toml|yaml|json} in CWD)")?;

        let settings: Self = built
            .try_deserialize()
            .map_err(|e| anyhow::Error::msg(e.to_string()))
            .context("failed to deserialize config into Settings")?;

        debug_assert!(
            !settings.models.is_empty(),
            "settings.models is empty; did you forget to define [models]?"
        );

        Ok(settings)
    }
    pub fn get_model(&self, name: &str) -> Result<&ModelConfig> {
        self.models
            .get(name)
            .with_context(|| format!("model `{}` not found in settings.models", name))
    }
    // list model keys
    pub fn model_names(&self) -> Vec<String> {
        // deterministic ordering helps tests and logs
        let mut keys: Vec<String> = self.models.keys().cloned().collect();
        keys.sort();
        keys
    }
}