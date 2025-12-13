// src/templates.rs
// Different input template for each models
pub fn apply_chat_template(model_name: &str, raw_prompt: &str, system_prompt: Option<String>) -> String {
    let sys_msg = system_prompt.unwrap_or("".to_string());

    match model_name {
        "llama3" => {
            let sys_block = if !sys_msg.is_empty() {
                format!("<|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>", sys_msg)
            } else {
                "".to_string()
            };
            format!(
                "<|begin_of_text|>{}<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                sys_block, raw_prompt
            )
        },
        "mistral" => {
            let final_prompt = if !sys_msg.is_empty() {
                format!("System: {}\n\nUser: {}", sys_msg, raw_prompt)
            } else {
                raw_prompt.to_string()
            };
            format!("<s>[INST] {} [/INST]", final_prompt)
        },
        "phi" => {
            let final_prompt = if !sys_msg.is_empty() {
                format!("{} {}", sys_msg, raw_prompt)
            } else {
                raw_prompt.to_string()
            };
            format!("Instruct: {}\nOutput:", final_prompt)
        },
        _ => raw_prompt.to_string(),
    }
}

