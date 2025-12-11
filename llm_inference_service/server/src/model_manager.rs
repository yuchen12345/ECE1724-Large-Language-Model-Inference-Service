use anyhow::{anyhow, Result};
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_llama as llama;
use candle_core::quantized::gguf_file::Content; // <--- Added this import
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;

use crate::config::ModelConfig;

const MAX_MODELS: usize = 2;

pub struct ModelSlot {
    pub config: ModelConfig,
    pub model: llama::ModelWeights,
    pub tokenizer: Tokenizer,
    pub device: Device,
}
#[derive(Clone)] // <--- ADD THIS LINE
pub struct ModelManager {
    slots: Arc<Mutex<HashMap<String, ModelSlot>>>,
}

impl ModelManager {
    pub fn new() -> Self {
        Self {
            slots: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn load_model(&self, config: ModelConfig) -> Result<()> {
        let mut slots = self.slots.lock().unwrap();

        if slots.contains_key(&config.name) {
            return Err(anyhow!("Model '{}' is already loaded", config.name));
        }

        if slots.len() >= MAX_MODELS {
            return Err(anyhow!(
                "Maximum {} models reached. Unload a model first. Currently loaded: {:?}",
                MAX_MODELS,
                slots.keys().collect::<Vec<_>>()
            ));
        }

        tracing::info!("Loading model: {} from {}", config.name, config.path);

        // Initialize CUDA device
        let device = Device::cuda_if_available(0)?;
        
        // Load tokenizer
        let tokenizer_path = config.path.replace(".gguf", "-tokenizer.json");
        let tokenizer = if std::path::Path::new(&tokenizer_path).exists() {
            Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?
        } else {
            // Fallback to default tokenizer
            tracing::warn!("Tokenizer not found at {}, using default", tokenizer_path);
            Tokenizer::from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", None)
                .map_err(|e| anyhow!("Failed to load default tokenizer: {}", e))?
        };

        // Load GGUF model
        let mut file = std::fs::File::open(&config.path)?;


        // --- FIX STARTS HERE ---
        // 1. Read the metadata/content from the file first
        let content = Content::read(&mut file)?;

        // 2. Pass the content + file + device to the loader
        let model = llama::ModelWeights::from_gguf(content, &mut file, &device)?;
        // --- FIX ENDS HERE ---


        tracing::info!("Model '{}' loaded successfully", config.name);

        slots.insert(
            config.name.clone(),
            ModelSlot {
                config,
                model,
                tokenizer,
                device,
            },
        );

        Ok(())
    }

    pub fn unload_model(&self, name: &str) -> Result<()> {
        let mut slots = self.slots.lock().unwrap();

        if slots.remove(name).is_some() {
            tracing::info!("Model '{}' unloaded successfully", name);
            Ok(())
        } else {
            Err(anyhow!("Model '{}' is not loaded", name))
        }
    }

    pub fn get_model(&self, name: &str) -> Result<Arc<Mutex<HashMap<String, ModelSlot>>>> {
        let slots = self.slots.lock().unwrap();
        if slots.contains_key(name) {
            Ok(self.slots.clone())
        } else {
            Err(anyhow!("Model '{}' is not loaded", name))
        }
    }

    pub fn list_loaded(&self) -> Vec<String> {
        let slots = self.slots.lock().unwrap();
        slots.keys().cloned().collect()
    }

    pub fn is_loaded(&self, name: &str) -> bool {
        let slots = self.slots.lock().unwrap();
        slots.contains_key(name)
    }
}

// Logits processor for sampling
pub struct LogitsProcessor {
    temperature: f64,
}

impl LogitsProcessor {
    pub fn new(temperature: f64) -> Self {
        Self { temperature }
    }

    pub fn sample(&self, logits: &Tensor) -> Result<u32> {
        let logits = logits.to_vec1::<f32>()?;
        let logits = if self.temperature > 0.0 {
            logits.iter().map(|l| l / self.temperature as f32).collect()
        } else {
            logits
        };

        // Simple argmax sampling
        let token = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as u32)
            .unwrap();

        Ok(token)
    }
}