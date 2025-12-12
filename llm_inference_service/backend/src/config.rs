// src/config.rs
use anyhow::{Context, Result};
use config::Config;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Deserialize, Clone)]
pub struct ModelConfig {
    pub arch: String,
    pub repo: String,           // HuggingFace Repo for Weights
    pub file: String,           // GGUF Filename
    pub tokenizer_repo: String, // HuggingFace Repo for Tokenizer
    pub tokenizer_file: String, // Tokenizer Filename
}

#[derive(Debug, Deserialize, Clone)]
pub struct Settings {
    pub models: HashMap<String, ModelConfig>,
}

impl Settings {
    /// Load settings from `config.*` in the current working directory.
    /// Additions:
    /// - richer error context (so debugging is easier)
    /// - small debug assertions (no runtime change in release builds)
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

    /// Optional helper for callers: list model keys (useful for /models endpoint).
    pub fn model_names(&self) -> Vec<String> {
        // deterministic ordering helps tests and logs
        let mut keys: Vec<String> = self.models.keys().cloned().collect();
        keys.sort();
        keys
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    /// Write a temporary `config.toml` into a temp directory, set CWD there,
    /// then call `Settings::new()` which reads "config" from current working dir.
    fn with_temp_cwd<F: FnOnce()>(toml: &str, f: F) {
        let tmp = std::env::temp_dir().join(format!(
            "settings_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&tmp).expect("create temp dir failed");

        let config_path: PathBuf = tmp.join("config.toml");
        fs::write(&config_path, toml).expect("write config.toml failed");

        let old = std::env::current_dir().expect("get current_dir failed");
        std::env::set_current_dir(&tmp).expect("set_current_dir failed");

        // Run test body
        f();

        // Restore and clean up best-effort
        std::env::set_current_dir(&old).expect("restore current_dir failed");
        let _ = fs::remove_file(&config_path);
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn loads_valid_config_toml() {
        let toml = r#"
            [models]

            [models.phi]
            arch = "phi"
            repo = "microsoft/phi-2"
            file = "phi-2.Q4_K_M.gguf"
            tokenizer_repo = "microsoft/phi-2"
            tokenizer_file = "tokenizer.json"

            [models.mistral]
            arch = "mistral"
            repo = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
            file = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
            tokenizer_repo = "mistralai/Mistral-7B-Instruct-v0.2"
            tokenizer_file = "tokenizer.json"
        "#;

        with_temp_cwd(toml, || {
            let s = Settings::new().expect("Settings::new should succeed");
            assert_eq!(s.model_names(), vec!["mistral".to_string(), "phi".to_string()]);
            let phi = s.get_model("phi").expect("phi should exist");
            assert_eq!(phi.arch, "phi");
        });
    }

    #[test]
    fn missing_model_returns_error() {
        let toml = r#"
            [models]
            [models.onlyone]
            arch = "llama3"
            repo = "meta-llama/Meta-Llama-3-8B-Instruct"
            file = "llama3.Q4_K_M.gguf"
            tokenizer_repo = "meta-llama/Meta-Llama-3-8B-Instruct"
            tokenizer_file = "tokenizer.json"
        "#;

        with_temp_cwd(toml, || {
            let s = Settings::new().expect("Settings::new should succeed");
            let err = s.get_model("does_not_exist").unwrap_err();
            let msg = format!("{:#}", err);
            assert!(msg.contains("model `does_not_exist` not found"));
        });
    }

    #[test]
    fn invalid_config_fails_to_deserialize() {
        // Missing required fields should fail.
        let toml = r#"
            [models]
            [models.bad]
            arch = "phi"
            repo = "x/y"
            # file missing
            tokenizer_repo = "x/y"
            tokenizer_file = "tokenizer.json"
        "#;

        with_temp_cwd(toml, || {
            let err = Settings::new().unwrap_err();
            let msg = format!("{:#}", err);
           
            assert!(msg.contains("deserialize") || msg.contains("deserial"));
        });
    }
}
