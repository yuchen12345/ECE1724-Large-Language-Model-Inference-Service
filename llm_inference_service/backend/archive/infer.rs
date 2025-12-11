use anyhow::Result;
use candle_core::Tensor;
use crate::model::LLM;
use candle_nn::VarBuilder;
use candle_transformers::models::phi::Model;
use candle_core::DType;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;


pub fn infer(llm: &LLM, prompt: String) -> Result<String> {
    let tokens = llm.tokenizer.encode(prompt, true).unwrap();
    let mut tokens = tokens.get_ids().to_vec();

    let mut input = Tensor::new(tokens.as_slice(), &llm.device)?.unsqueeze(0)?;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[llm.model_file.clone()],
            DType::F32,
            &llm.device,
        )?
    };

    let mut model = Model::new(&llm.config, vb)?;
    
    let mut logits = model.forward(&input)?;
    let mut next_token_logits = logits.squeeze(0)?;

    let max_gen_tokens = 16;
    let mut output = String::new();

    for _ in 0..max_gen_tokens {
        let next_token_id = next_token_logits.argmax(0)?.to_scalar::<u32>()?;
        let next_token = llm
            .tokenizer
            .decode(&[next_token_id], true)
            .unwrap();

        output.push_str(&next_token);

        if next_token_id == 50256 {
            break;
        }

        tokens.push(next_token_id);
        input = Tensor::new(&[next_token_id], &llm.device)?.unsqueeze(0)?;
        logits = model.forward(&input)?;
        next_token_logits = logits.squeeze(0)?;
    }

    Ok(output.trim().to_string())
}

pub fn infer_stream(
    llm: std::sync::Arc<LLM>,
    prompt: String,
) -> ReceiverStream<String> {
    let (tx, rx) = mpsc::channel(32);

    std::thread::spawn(move || {
        let _ = (|| -> Result<()> {
            let tokens = llm.tokenizer.encode(prompt, true).unwrap();
            let mut tokens = tokens.get_ids().to_vec();

            let mut input =
                Tensor::new(tokens.as_slice(), &llm.device)?.unsqueeze(0)?;

            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(
                    &[llm.model_file.clone()],
                    DType::F32,
                    &llm.device,
                )?
            };
            let mut model = Model::new(&llm.config, vb)?;

            let mut logits = model.forward(&input)?;
            let mut next_token_logits = logits.squeeze(0)?;

            let max_gen_tokens = 64;

            for _ in 0..max_gen_tokens {
                let next_token_id =
                    next_token_logits.argmax(0)?.to_scalar::<u32>()?;
                let next_token = llm
                    .tokenizer
                    .decode(&[next_token_id], true)
                    .unwrap();

                if tx.blocking_send(next_token.clone()).is_err() {
                    break;
                }

                if next_token_id == 50256 {
                    break;
                }

                tokens.push(next_token_id);
                input = Tensor::new(&[next_token_id], &llm.device)?.unsqueeze(0)?;
                logits = model.forward(&input)?;
                next_token_logits = logits.squeeze(0)?;
            }
            tx.blocking_send("[DONE]".to_string()).ok();
            Ok(())
        })();
    });

    ReceiverStream::new(rx)
}

