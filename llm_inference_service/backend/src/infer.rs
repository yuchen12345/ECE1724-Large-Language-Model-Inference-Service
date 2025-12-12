// src/infer.rs
use crate::model::{LoadedModel, ModelEnum};
use anyhow::Result;
use candle_core::{Tensor, DType};
use candle_transformers::generation::LogitsProcessor;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub struct InferenceParams {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub max_tokens: Option<usize>,
    pub seed: Option<u64>,
}

// Inference loop for a given prompt
pub fn run_inference(
    loaded_model: &mut LoadedModel, 
    prompt: &str,
    params: InferenceParams, 
    mut callback: impl FnMut(String)
) -> Result<()> {
    let temp = params.temperature.unwrap_or(0.7);
    let top_p = params.top_p.unwrap_or(0.9);
    let max_new_tokens = params.max_tokens.unwrap_or(1024);
    let seed = params.seed.unwrap_or_else(|| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    });
    let tokenizer = &loaded_model.tokenizer;
    let device = &loaded_model.device;
    // Encode prompt into Token Ids
    let tokens = tokenizer.encode(prompt, true).map_err(anyhow::Error::msg)?;
    let mut input_ids = tokens.get_ids().to_vec();
    // Initialize sampler: 
    // temperature for randomness
    // top-p for diversity
    let mut logits_processor = LogitsProcessor::new(seed, Some(temp), Some(top_p));

    // track length of text generated so far
    let mut prev_text_len = 0;
    // Decode prompt for length
    let initial_text = tokenizer.decode(&input_ids, true).map_err(anyhow::Error::msg)?;
    prev_text_len = initial_text.len();

    for index in 0..max_new_tokens {
        // Calculate input context
        let context_size = if index > 0 { 1 } else { input_ids.len() };
        let start_at = input_ids.len() - context_size;
        let input_tensor = Tensor::new(&input_ids[start_at..], device)?.unsqueeze(0)?;
        
        // Match enum to call the modle's formward function
        let logits = match &mut loaded_model.model {
            ModelEnum::Phi(m) => m.forward(&input_tensor, start_at)?,
            ModelEnum::Mistral(m) => m.forward(&input_tensor, start_at)?,
            ModelEnum::Llama3(m) => m.forward(&input_tensor, start_at)?,
        };
        // Extract logits for the last token
        let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
        let next_token = logits_processor.sample(&logits)?;
        input_ids.push(next_token);

        // Incremental decoding
        let current_text = tokenizer.decode(&input_ids, true).map_err(anyhow::Error::msg)?;
        if current_text.len() > prev_text_len {
            // Calculate the newly added text and send via callback
            let new_text = &current_text[prev_text_len..];
            callback(new_text.to_string());
            prev_text_len = current_text.len();
        }

        // Check for stop tokens
        if next_token == tokenizer.token_to_id("</s>").unwrap_or(2) 
            || next_token == 50256
            || next_token == 128001
            || next_token == 128009 { 
            break; 
        }
    }
    
    Ok(())
}