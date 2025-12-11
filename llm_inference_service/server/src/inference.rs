use anyhow::Result;
use candle_core::{Tensor, IndexOp};


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

pub struct InferenceEngine {
    model_manager: ModelManager,
}

impl InferenceEngine {
    pub fn new(model_manager: ModelManager) -> Self {
        Self { model_manager }
    }

    /// 一次性返回完整结果的推理接口（给 /inference 用）
    pub async fn generate(&self, req: InferenceRequest) -> Result<InferenceResponse> {
        let model_name = req.model.clone();
        let prompt = req.prompt.clone();
        let max_tokens = req.max_tokens.unwrap_or(512);
        let temperature = req.temperature.unwrap_or(0.7);

        let model_manager = self.model_manager.clone();

        // 在阻塞线程里跑 Candle 推理，避免卡住 tokio runtime
        let resp = tokio::task::spawn_blocking(move || {
            Self::generate_blocking(
                &model_manager,
                &model_name,
                &prompt,
                max_tokens,
                temperature,
            )
        })
        .await??;

        Ok(resp)
    }

    /// 真正的同步推理逻辑：返回完整文本和 token 数
    fn generate_blocking(
        model_manager: &ModelManager,
        model_name: &str,
        prompt: &str,
        max_tokens: usize,
        temperature: f64,
    ) -> Result<InferenceResponse> {
        // 拿到目标模型 slot
        let slots = model_manager.get_model(model_name)?;
        let mut slots = slots.lock().unwrap();

        let slot = slots
            .get_mut(model_name)
            .ok_or_else(|| anyhow::anyhow!("Model not found"))?;

        let tokenizer = &slot.tokenizer;
        let model = &mut slot.model;
        let device = &slot.device;

        // 1. Tokenize prompt
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
            let context_size = if idx > 0 { 1 } else { all_tokens.len() };
            let start_pos = all_tokens.len().saturating_sub(context_size);

            let input = Tensor::new(&all_tokens[start_pos..], device)?
                .unsqueeze(0)?; // [1, seq_len]

            let mut logits = model.forward(&input, start_pos)?;
            logits = logits.squeeze(0)?;

            if logits.rank() == 2 {
                let dims = logits.dims();
                let seq_len = dims[0];
                logits = logits.i((seq_len - 1, ..))?; // 取最后一个 time step
            }

            let next_token = logits_processor.sample(&logits)?;

            if next_token == eos_token {
                break;
            }

            all_tokens.push(next_token);
        }

        //  生成完之后，一次性 decode「新生成」的部分
        let gen_tokens = &all_tokens[input_tokens.len()..]; // 只去掉 prompt 部分
        let text = tokenizer
            .decode(gen_tokens, true)
            .map_err(|e| anyhow::anyhow!("Decode failed: {}", e))?;

        let tokens_generated = gen_tokens.len();

        Ok(InferenceResponse {
            text,
            tokens_generated,
        })

        // let mut all_tokens = input_tokens.to_vec();
        // let logits_processor = LogitsProcessor::new(temperature);

        // let eos_token = tokenizer
        //     .token_to_id("</s>")
        //     .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
        //     .unwrap_or(2);

        // // 用来累积最终文本
        // let mut full_text = String::new();
        // let mut token_count = 0;

        // for idx in 0..max_tokens {
        //     // 2. 首次喂完整 prompt，后续每次只喂 1 个新 token
        //     let context_size = if idx > 0 { 1 } else { all_tokens.len() };
        //     let start_pos = all_tokens.len().saturating_sub(context_size);

        //     // [seq] -> [1, seq]，符合 quantized_llama 的输入
        //     let input = Tensor::new(&all_tokens[start_pos..], device)?
        //         .unsqueeze(0)?; // [1, seq_len]

        //     // 3. 调用模型，注意只取“最后一个 time step”的 logits
        //     let mut logits = model.forward(&input, start_pos)?; // 通常是 [1, seq_len, vocab] 或 [1, vocab]
        //     logits = logits.squeeze(0)?; // 变成 [seq_len, vocab] 或 [vocab]

        //     if logits.rank() == 2 {
        //         let dims = logits.dims();
        //         let seq_len = dims[0];
        //         // 只取最后一个 token 的 logits -> [vocab]
        //         logits = logits.i((seq_len - 1, ..))?;
        //     }

        //     let next_token = logits_processor.sample(&logits)?;

        //     // 遇到 eos，结束生成
        //     if next_token == eos_token {
        //         break;
        //     }

        //     all_tokens.push(next_token);

        //     // 4. decode token 并拼接到最终字符串中
        //     if let Ok(text) = tokenizer.decode(&[next_token], true) {
        //         full_text.push_str(&text);
        //         token_count += 1;
        //     }
        // }

        // Ok(InferenceResponse {
        //     text: full_text,
        //     tokens_generated: token_count,
        // })
    }


}
