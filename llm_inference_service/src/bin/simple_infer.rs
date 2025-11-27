use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::phi::{Config, Model};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;
use std::io::Write;

fn main() -> Result<()> {
    // Create a CUDA device, fall back to CPU if CUDA fails
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    println!("Running on device: {:?}", device);

    // Create reference to model repo microsoft/phi-1_5
    let api = Api::new()?;
    let repo = api.repo(Repo::new("microsoft/phi-1_5".to_string(), RepoType::Model));

    // Load model files
    println!("Loading model files...");
    let tokenizer_file = repo.get("tokenizer.json")?;
    let config_file = repo.get("config.json")?;
    let model_file = repo.get("model.safetensors")?;

    // Load tokenizerm, model configuration, and weights.
    let tokenizer = Tokenizer::from_file(tokenizer_file).map_err(E::msg)?;
    let config: Config = serde_json::from_str(&std::fs::read_to_string(config_file)?)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)? };
    //Construct the model
    let mut model = Model::new(&config, vb)?;

    // Define and tokenize prompt
    let prompt = "Hello, explain C++ to me in one sentence.";
    let tokens = tokenizer.encode(prompt, true).map_err(E::msg)?;
    let mut tokens = tokens.get_ids().to_vec();
    print!("Prompt: {}", prompt);
    std::io::stdout().flush()?;

    let mut input = Tensor::new(tokens.as_slice(), &device)?.unsqueeze(0)?;
    let mut logits = model.forward(&input)?;
    let mut next_token_logits = logits.squeeze(0)?;
    let max_gen_tokens = 50;

    // At each step, model predicts one token
    for _ in 0..max_gen_tokens {
        // Select token with highest probability
        let next_token_id = next_token_logits.argmax(0)?.to_scalar::<u32>()?;     
        let next_token = tokenizer.decode(&[next_token_id], true).map_err(E::msg)?;
        print!("{}", next_token);
        std::io::stdout().flush()?;
        // End of sentence
        if next_token_id == 50256{
            break;
        }
        // Apeend new token to sequence and prepare for the next step
        tokens.push(next_token_id);
        input = Tensor::new(&[next_token_id], &device)?.unsqueeze(0)?;
        logits = model.forward(&input)?;
        next_token_logits = logits.squeeze(0)?;
    }

    println!("\n--- Finish ---");
    Ok(())
}