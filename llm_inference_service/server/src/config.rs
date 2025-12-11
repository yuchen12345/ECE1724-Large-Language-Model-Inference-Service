use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::sync::{Arc, RwLock};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub path: String,
    pub architecture: String, // "llama" for most small models
    pub max_context: usize,
    pub temperature: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub models: Vec<ModelConfig>,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            models: vec![
                ModelConfig {
                    name: "qwen-1.5b".to_string(),
                    path: "models/qwen2.5-1.5b-instruct-q4_k_m.gguf".to_string(),
                    architecture: "llama".to_string(),
                    max_context: 2048,
                    temperature: 0.7,
                },
                ModelConfig {
                    name: "tinyllama-1.1b".to_string(),
                    path: "models/tinyllama-1.1b-chat-v1.0.Q4_K_M".to_string(),
                    architecture: "llama".to_string(),
                    max_context: 4096,
                    temperature: 0.7,
                },
                // ModelConfig {
                //     name: "smollm-1.7b".to_string(),
                //     path: "models/SmolLM-1.7B-Instruct-Q5_K_M.gguf".to_string(),
                //     architecture: "llama".to_string(),
                //     max_context: 2048,
                //     temperature: 0.7,
                // },
            ],
        }
    }
}

pub struct ConfigManager {
    config_path: String,
    config: Arc<RwLock<AppConfig>>,
}

impl ConfigManager {
    pub fn new(config_path: String) -> Result<Self> {
        let config = if std::path::Path::new(&config_path).exists() {
            let content = fs::read_to_string(&config_path)?;
            serde_json::from_str(&content)?
        } else {
            let default_config = AppConfig::default();
            let json = serde_json::to_string_pretty(&default_config)?;
            fs::write(&config_path, json)?;
            tracing::info!("Created default config at {}", config_path);
            default_config
        };

        Ok(Self {
            config_path,
            config: Arc::new(RwLock::new(config)),
        })
    }

    pub fn reload(&self) -> Result<()> {
        let content = fs::read_to_string(&self.config_path)?;
        let new_config: AppConfig = serde_json::from_str(&content)?;
        
        let mut config = self.config.write().unwrap();
        *config = new_config;
        tracing::info!("Config reloaded successfully");
        Ok(())
    }

    pub fn get_config(&self) -> AppConfig {
        self.config.read().unwrap().clone()
    }

    pub fn find_model(&self, name: &str) -> Option<ModelConfig> {
        let config = self.config.read().unwrap();
        config.models.iter().find(|m| m.name == name).cloned()
    }

    pub fn list_models(&self) -> Vec<ModelConfig> {
        self.config.read().unwrap().models.clone()
    }
}