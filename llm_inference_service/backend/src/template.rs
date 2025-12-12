// src/templates.rs
pub fn apply_chat_template(model_name: &str, raw_prompt: &str) -> String {
    match model_name {
        "llama3" => format!(
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            raw_prompt
        ),
        "mistral" => format!(
            "<s>[INST] {} [/INST]", 
            raw_prompt
        ),
        "phi" => format!(
            "Instruct: {}\nOutput:", 
            raw_prompt
        ),
        _ => raw_prompt.to_string(),
    }
}