use anyhow::Result;
use candle_core::Tensor;
use tokio::sync::mpsc;

use crate::model_manager::{LogitsProcessor, ModelManager};

pub struct InferenceRequest {
    pub model: String,
    pub prompt: String,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f64>,
}

pub struct InferenceResponse {
    pub text: String,
    pub tokens_generated: usize,
}

#[derive(Debug, Clone)]
pub enum StreamToken {
    Token(String),
    Done,
    Error(String),
}

pub struct InferenceEngine {
    model_manager: ModelManager,
}

impl InferenceEngine {
    pub fn new(model_manager: ModelManager) -> Self {
        Self { model_manager }
    }

    pub async fn generate(&self, req: InferenceRequest) -> Result<InferenceResponse> {
        let (tx, mut rx) = mpsc::channel(100);
        
        let model_name = req.model.clone();
        let max_tokens = req.max_tokens.unwrap_or(512);
        let temperature = req.temperature.unwrap_or(0.7);

        // Spawn blocking task for inference
        let model_manager = self.model_manager.clone();
        tokio::task::spawn_blocking(move || {
            Self::generate_blocking(&model_manager, &model_name, &req.prompt, max_tokens, temperature, Some(tx))
        })
        .await??;

        // Collect all tokens
        let mut full_text = String::new();
        let mut token_count = 0;

        while let Some(token) = rx.recv().await {
            match token {
                StreamToken::Token(t) => {
                    full_text.push_str(&t);
                    token_count += 1;
                }
                StreamToken::Done => break,
                StreamToken::Error(e) => return Err(anyhow::anyhow!(e)),
            }
        }

        Ok(InferenceResponse {
            text: full_text,
            tokens_generated: token_count,
        })
    }

    pub async fn generate_stream(
        &self,
        req: InferenceRequest,
        tx: mpsc::Sender<StreamToken>,
    ) -> Result<()> {
        let model_name = req.model.clone();
        let max_tokens = req.max_tokens.unwrap_or(512);
        let temperature = req.temperature.unwrap_or(0.7);

        let model_manager = self.model_manager.clone();
        tokio::task::spawn_blocking(move || {
            Self::generate_blocking(&model_manager, &model_name, &req.prompt, max_tokens, temperature, Some(tx))
        })
        .await??;

        Ok(())
    }

    // fn generate_blocking(
    //     model_manager: &ModelManager,
    //     model_name: &str,
    //     prompt: &str,
    //     max_tokens: usize,
    //     temperature: f64,
    //     tx: Option<mpsc::Sender<StreamToken>>,
    // ) -> Result<()> {
    //     let slots = model_manager.get_model(model_name)?;
    //     let mut slots = slots.lock().unwrap();
        
    //     let slot = slots
    //         .get_mut(model_name)
    //         .ok_or_else(|| anyhow::anyhow!("Model not found"))?;

    //     let tokenizer = &slot.tokenizer;
    //     let model = &mut slot.model;
    //     let device = &slot.device;

    //     // Tokenize input
    //     let tokens = tokenizer
    //         .encode(prompt, true)
    //         .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
    //     let input_tokens = tokens.get_ids();

    //     tracing::info!(
    //         "Generating {} tokens for prompt with {} input tokens",
    //         max_tokens,
    //         input_tokens.len()
    //     );

    //     let mut all_tokens = input_tokens.to_vec();
    //     let logits_processor = LogitsProcessor::new(temperature);

    //     let eos_token = tokenizer
    //         .token_to_id("</s>")
    //         .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
    //         .unwrap_or(2);

    //     for idx in 0..max_tokens {
    //         let context_size = if idx > 0 { 1 } else { all_tokens.len() };
    //         let start_pos = all_tokens.len().saturating_sub(context_size);
    //         let input = Tensor::new(&all_tokens[start_pos..], device)?;

    //         let logits = model.forward(&input, start_pos)?;
    //         let logits = logits.squeeze(0)?;

    //         let next_token = logits_processor.sample(&logits)?;

    //         if next_token == eos_token {
    //             if let Some(ref tx) = tx {
    //                 let _ = tx.blocking_send(StreamToken::Done);
    //             }
    //             break;
    //         }

    //         all_tokens.push(next_token);

    //         // Decode and send token
    //         if let Some(ref tx) = tx {
    //             if let Ok(text) = tokenizer.decode(&[next_token], true) {
    //                 if tx.blocking_send(StreamToken::Token(text)).is_err() {
    //                     break;
    //                 }
    //             }
    //         }
    //     }

    //     if tx.is_some() {
    //         if let Some(ref tx) = tx {
    //             let _ = tx.blocking_send(StreamToken::Done);
    //         }
    //     }

    //     Ok(())
    // }

    fn generate_blocking(
    model_manager: &ModelManager,
    model_name: &str,
    prompt: &str,
    max_tokens: usize,
    temperature: f64,
    tx: Option<mpsc::Sender<StreamToken>>,
) -> Result<()> {
    let slots = model_manager.get_model(model_name)?;
    let mut slots = slots.lock().unwrap();

    let slot = slots
        .get_mut(model_name)
        .ok_or_else(|| anyhow::anyhow!("Model not found"))?;

    let tokenizer = &slot.tokenizer;
    let model = &mut slot.model;
    let device = &slot.device;

    // 1. tokenize prompt
    let tokens = tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
    let input_tokens = tokens.get_ids(); // Vec<u32>

    tracing::info!(
        "Generating {} tokens for prompt with {} input tokens",
        max_tokens,
        input_tokens.len()
    );

    let mut all_tokens = input_tokens.to_vec();
    let logits_processor = LogitsProcessor::new(temperature);

    let eos_token = tokenizer
        .token_to_id("</s>")
        .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
        .unwrap_or(2);

    for idx in 0..max_tokens {
        // 2. 只在第一步喂完整 prompt，后面每次只喂最新 1 个 token
        let context_size = if idx > 0 { 1 } else { all_tokens.len() };
        let start_pos = all_tokens.len().saturating_sub(context_size);

        // [seq] -> [1, seq]，满足 quantized_llama::ModelWeights 的输入要求
        let input = Tensor::new(&all_tokens[start_pos..], device)?
            .unsqueeze(0)?; // [1, seq_len]

        // 3. llama 的 forward 需要 index_pos（当前上下文的起始位置）做 kv-cache 与 rope
        let logits = model.forward(&input, start_pos)?;   // [1, vocab]
        let logits = logits.squeeze(0)?;                  // [vocab]

        let next_token = logits_processor.sample(&logits)?;

        if next_token == eos_token {
            if let Some(ref tx) = tx {
                let _ = tx.blocking_send(StreamToken::Done);
            }
            break;
        }

        all_tokens.push(next_token);

        // 4. 增量解码并通过 channel 推出去（用于 /inference/stream SSE）
        if let Some(ref tx) = tx {
            if let Ok(text) = tokenizer.decode(&[next_token], true) {
                if tx.blocking_send(StreamToken::Token(text)).is_err() {
                    // 客户端断开了，结束生成
                    break;
                }
            }
        }
    }

    // 如果循环自然结束，还没发送 Done，这里兜底发一次
    if let Some(ref tx) = tx {
        let _ = tx.blocking_send(StreamToken::Done);
    }

    Ok(())
}


    
}