// src/infer.rs
use crate::model::{LoadedModel, ModelEnum};
use anyhow::{Context, Result};
use candle_core::{DType, Tensor};
use candle_transformers::generation::LogitsProcessor;
use std::time::{SystemTime, UNIX_EPOCH};

// Parameters that control model generation behavior
#[derive(Debug, Clone)]
pub struct InferenceParams {
    // Softmax temperature. Higher => more random
    pub temperature: Option<f64>,
    // Nucleus sampling (top-p). Lower => more conservative
    pub top_p: Option<f64>,
    // Maximum number of new tokens to generate
    pub max_tokens: Option<usize>,
    // RNG seed for sampling. If None, seed is derived from current time
    pub seed: Option<u64>,
}

#[inline]
fn derive_seed_from_time() -> u64 {
    // Fetch system current time
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[inline]
fn decode_ids(tokenizer: &tokenizers::Tokenizer, ids: &[u32]) -> Result<String> {
    tokenizer
        .decode(ids, true)
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("tokenizer.decode failed (ids_len={})", ids.len()))
}

#[inline]
fn encode_prompt(tokenizer: &tokenizers::Tokenizer, prompt: &str) -> Result<Vec<u32>> {
    let enc = tokenizer
        .encode(prompt, true)
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("tokenizer.encode failed (prompt_len={})", prompt.len()))?;

    Ok(enc.get_ids().to_vec())
}

// stop token ids for models
#[inline]
fn stop_token_ids(tokenizer: &tokenizers::Tokenizer) -> (u32, u32, u32, u32) {
    let eos = tokenizer.token_to_id("</s>").unwrap_or(2);
    let gpt2_eos = 50256;
    let llama3_eot = 128001;
    let llama3_eom = 128009;
    (eos, gpt2_eos, llama3_eot, llama3_eom)
}

// Inference loop for a given prompt.
pub fn run_inference(
    loaded_model: &mut LoadedModel,
    prompt: &str,
    params: InferenceParams,
    mut callback: impl FnMut(String),
) -> Result<()> {
    // Parameter defaults
    let temp = params.temperature.unwrap_or(0.7);
    let top_p = params.top_p.unwrap_or(0.9);
    let max_new_tokens = params.max_tokens.unwrap_or(1024);
    let seed = params.seed.unwrap_or_else(derive_seed_from_time);

    let tokenizer = &loaded_model.tokenizer;
    let device = &loaded_model.device;

    // Encode prompt into Token Ids
    let mut input_ids = encode_prompt(tokenizer, prompt)
        .with_context(|| "failed to encode prompt into token ids")?;

    // Initialize sampler
    // temperature for randomness
    // top-p for diversity
    let mut logits_processor = LogitsProcessor::new(seed, Some(temp), Some(top_p));

    // Track length of generated text so far
    //let mut prev_text_len = 0usize;

    let initial_text = decode_ids(tokenizer, &input_ids)
        .with_context(|| "failed to decode initial prompt tokens")?;
    let mut prev_text_len = initial_text.len();

    // Precompute stop token ids (same checks as before).
    let (stop_0, stop_1, stop_2, stop_3) = stop_token_ids(tokenizer);

    // Generation loop
    for index in 0..max_new_tokens {
        // Context sizing:
        // - First step uses full prompt context
        // - Later steps feed only the last token
        let context_size = if index > 0 { 1 } else { input_ids.len() };

        debug_assert!(context_size >= 1, "context_size must be >= 1");
        debug_assert!(
            input_ids.len() >= context_size,
            "input_ids.len() must be >= context_size"
        );

        let start_at = input_ids.len() - context_size;

        // Build input tensor: shape [1, context_size]
        let input_slice = &input_ids[start_at..];
        let input_tensor = Tensor::new(input_slice, device)
            .with_context(|| format!("Tensor::new failed (slice_len={})", input_slice.len()))?
            .unsqueeze(0)
            .context("unsqueeze(0) failed for input_tensor")?;

        // Forward pass: call correct model variant (unchanged)
        let logits = match &mut loaded_model.model {
            ModelEnum::Phi(m) => m
                .forward(&input_tensor, start_at)
                .with_context(|| format!("Phi.forward failed (start_at={})", start_at))?,
            ModelEnum::Mistral(m) => m
                .forward(&input_tensor, start_at)
                .with_context(|| format!("Mistral.forward failed (start_at={})", start_at))?,
            ModelEnum::Llama3(m) => m
                .forward(&input_tensor, start_at)
                .with_context(|| format!("Llama3.forward failed (start_at={})", start_at))?,
        };

        // Extract logits for the last token:
        let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
        // Sample next token
        let next_token = logits_processor
            .sample(&logits)
            .context("logits_processor.sample failed")?;

        // Append token to running sequence
        input_ids.push(next_token);

        // Incremental decoding
        // Decode full text each step, then only emit the newly added suffix.
        let current_text = decode_ids(tokenizer, &input_ids)
            .with_context(|| format!("failed to decode at step index={}", index))?;

        if current_text.len() > prev_text_len {
            let new_text = &current_text[prev_text_len..];
            callback(new_text.to_string());
            prev_text_len = current_text.len();
        }
        // Stop tokens
        if next_token == stop_0 || next_token == stop_1 || next_token == stop_2 || next_token == stop_3
        {
            break;
        }
    }
    Ok(())
}


