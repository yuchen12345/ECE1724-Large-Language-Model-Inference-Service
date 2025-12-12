use leptos::*;
use serde::{Deserialize, Serialize};
use gloo_net::http::Request;
use futures::StreamExt;
use wasm_streams::ReadableStream;
use wasm_bindgen::JsCast;

const API_BASE: &str = "http://127.0.0.1:8081";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct ChatMessage {
    id: u64, // id for each chat message
    role: String, // User or AI
    content: String,
}

#[derive(Deserialize)]
struct ModelListResponse {
    // model list, key as model name, value as model settings
    models: std::collections::HashMap<String, serde_json::Value>,
    active: String,
}

#[derive(Serialize)]
// load model request
struct LoadModelRequest { name: String }

#[derive(Deserialize)]
// api response
struct ApiResponse { status: String, message: Option<String> }

#[derive(Serialize)]
struct InferRequest {
    // inference request parameters
    prompt: String,
    temperature: f64,
    top_p: f64,
    max_tokens: usize,
    seed: Option<u64>,
}

#[component]
// Add instruction for each model parameters
fn HelpTooltip(text: &'static str) -> impl IntoView {
    view! {
        <span class="tooltip-container">
            <span class="icon">"?"</span>
            <span class="tooltip-text">{text}</span>
        </span>
    }
}

#[component]
fn App() -> impl IntoView {
    let (status_text, set_status_text) = create_signal("Checking server...".to_string()); // show check server
    let (is_online, set_is_online) = create_signal(false); // check if server online
    let (models, set_models) = create_signal::<Vec<String>>(vec![]); // check list of models
    let (active_model, set_active_model) = create_signal("".to_string()); // check model that is selected
    
    // chat history box
    let (chat_history, set_chat_history) = create_signal::<Vec<ChatMessage>>(
        vec![
            ChatMessage { 
                id: js_sys::Date::now() as u64,
                role: "AI".into(), 
                content: "Hello! I am your local AI.".into(), 
            }
        ]
    ); 
    
    let (user_input_text, set_user_input_text) = create_signal("".to_string()); // user input
    // show if is generating, to disable/enable send button
    let (is_generating, set_is_generating) = create_signal(false); 
    let (loading_overlay, set_loading_overlay) = create_signal::<Option<String>>(None); // add overlay when model is loading
    // Handle the streaming text separately
    let (streaming_content, set_streaming_content) = create_signal("".to_string());

    // Model inference parameters
    let (temperature, set_temperature) = create_signal(0.7);
    let (top_p, set_top_p) = create_signal(0.9);
    let (max_tokens, set_max_tokens) = create_signal(200);
    let (seed, set_seed) = create_signal::<Option<u64>>(None);
    // control chat history window
    let chat_history_ref = create_node_ref::<html::Div>();

    // Init
    create_effect(move |_| {
        spawn_local(async move {
            // Health Check to set if server online
            if Request::get(&format!("{}/health", API_BASE)).send().await.is_ok() {
                set_is_online.set(true);
                set_status_text.set("Server Online".to_string());
            } else {
                set_status_text.set("Server Offline".to_string());
            }

            // Fetch Model list
            if let Ok(res) = Request::get(&format!("{}/models", API_BASE)).send().await {
                if let Ok(data) = res.json::<ModelListResponse>().await {
                    let mut model_names: Vec<String> = data.models.into_keys().collect();
                    model_names.sort();
                    set_models.set(model_names);
                    set_active_model.set(data.active); // set current active model
                }
            }
        });
    });

    // Auto-scroll the chat window to bottom
    let scroll_to_bottom = move || {
        // Check if chat_history_ref is currently attached to a real DOM element
        if let Some(div) = chat_history_ref.get() {
            let _ = div.set_scroll_top(div.scroll_height()); // Scroll
        }
    };

    // Load Model
    let load_model = move |model_name: String| {
        if model_name.is_empty() { 
            return; 
        }
        spawn_local(async move {
            // show overlay if model is loading
            set_loading_overlay.set(Some(format!("Loading {}...", model_name)));
            // load model
            let res = Request::post(&format!("{}/load_model", API_BASE))
                .json(&LoadModelRequest { name: model_name.clone() })
                .unwrap()
                .send()
                .await;
            match res {
                Ok(r) => {
                    // Request success
                     if let Ok(data) = r.json::<ApiResponse>().await {
                        if data.status == "ok" {
                            // Set active model
                            set_active_model.set(model_name.clone());
                            set_chat_history.update(|h| h.push(ChatMessage {
                                id: js_sys::Date::now() as u64,
                                role: "AI".into(),
                                content: format!("System: Model loaded: {}", model_name),
                            }));
                            scroll_to_bottom();
                        } else {
                            logging::error!("Error loading model: {:?}", data.message);
                        }
                     }
                }
                Err(e) => logging::error!("Failed to connect: {}", e),
            }
            // hide overlay when model loading done
            set_loading_overlay.set(None);
        });
    };

    // Send Message
    let send_message = move || {
        // fetch user input and remove space
        let text = user_input_text.get_untracked().trim().to_string();
        if text.is_empty() || is_generating.get_untracked() { 
            return; 
        }
        // Check if there is active model selected
        let current_model = active_model.get_untracked();
        if current_model.is_empty() {
             logging::warn!("No active model selected");
             return;
        }
        // Clean user input after user send the message
        set_user_input_text.set("".into());
        set_is_generating.set(true);
        set_streaming_content.set("".to_string()); // Clear stream buffer

        // Push user input to chat history
        set_chat_history.update(|h| {
            h.push(ChatMessage { 
                    id: js_sys::Date::now() as u64,
                    role: "User".into(), 
                    content: text.clone(), 
                }
            )
        });
        scroll_to_bottom();
        
        spawn_local(async move {
            // inference parameters
            let payload = InferRequest {
                prompt: text,
                temperature: temperature.get_untracked(),
                top_p: top_p.get_untracked(),
                max_tokens: max_tokens.get_untracked(),
                seed: seed.get_untracked(),
            };
            // send request for inference
            let response = Request::post(&format!("{}/infer_stream", API_BASE))
                .json(&payload)
                .unwrap()
                .send()
                .await;

            if let Ok(resp) = response {
                if let Some(body) = resp.body() {
                    // Convert the Web ReadableStream(JavaScript) into a Rust Stream
                    let mut stream = ReadableStream::from_raw(body.dyn_into().unwrap()).into_stream();
                    let mut buffer = String::new();
                    // Loop through each incoming data chunk
                    while let Some(Ok(chunk_js_value)) = stream.next().await {
                        // Convert raw js value into rust vec, convert raw bytes to string
                        let chunk = js_sys::Uint8Array::new(&chunk_js_value).to_vec();
                        let chunk_str = String::from_utf8_lossy(&chunk);
                        buffer.push_str(&chunk_str);

                        // Parse SSE data
                        // Split buffer by newline, keep last line
                        let lines: Vec<&str> = buffer.split('\n').collect();
                        let last_line = lines.last().cloned().unwrap_or("").to_string();
                        
                        for line in lines.iter().take(lines.len() - 1) {
                            let line = line.trim();
                            if line.is_empty() { continue; }

                            if line.starts_with("data:") {
                                let raw_content = &line[5..]; 
                                let content_str = if raw_content.starts_with(' '){ 
                                    &raw_content[1..] 
                                } else { 
                                    raw_content 
                                };
                                // Done marker, inference finished
                                if content_str == "[DONE]" { 
                                    break; 
                                } 
                                if content_str.starts_with("[MODEL:"){ 
                                    continue; 
                                }
                                if content_str.starts_with("[ERROR]"){ 
                                    continue; 
                                }

                                // Try parse JSON
                                let text_to_append = match serde_json::from_str::<serde_json::Value>(content_str) {
                                    Ok(json) => json["text"].as_str().unwrap_or("").to_string(),
                                    Err(_) => content_str.to_string(),
                                };

                                // Update separate signal instead of history
                                set_streaming_content.update(|s| s.push_str(&text_to_append));
                                scroll_to_bottom();
                            }
                        }
                        buffer = last_line;
                    }
                }
            } else {
                logging::error!("Network error");
            }

            // When done, push the full message to history
            let final_content = streaming_content.get_untracked();
            if !final_content.is_empty() {
                set_chat_history.update(|h| h.push(ChatMessage {
                    id: js_sys::Date::now() as u64,
                    role: "AI".into(),
                    content: final_content,
                }));
                set_streaming_content.set("".to_string());
            }

            set_is_generating.set(false);
        });
    };

    view! {
        <div id="sidebar">
            <h2>"LLM chat"</h2>
            <div class="control-group">
                <label>"Models"</label>
                // Model selection
                <select 
                    // Bind value directly to active_model signal
                    prop:value=move || active_model.get()
                    on:change=move |ev| {
                    let new_val = event_target_value(&ev);
                    if new_val != active_model.get_untracked() {
                        load_model(new_val);
                    }
                }>
                    // When no model selected
                    <Show when=move || active_model.get().is_empty()>
                        <option value="" disabled selected>"Select a model to start"</option>
                    </Show>
                    <For
                        each=move || models.get() 
                        key=|name| name.clone()
                        children=move |name| {
                            let is_selected = name == active_model.get();
                            view! { 
                                <option value=name.clone() selected=is_selected>
                                    {name.to_uppercase()}
                                </option> }
                        }
                    />
                </select>
            </div>

             <hr style="border-color: #4d4d4f; width: 100%;" />

            // Temperature slide
            <div class="control-group">
                <label style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="display: flex; align-items: center;">
                        "Temperature"
                        <HelpTooltip text="Controls randomness. Higher values (e.g., 1.0) make output more creative but less precise. Lower values (e.g., 0.2) make it deterministic and focused."/>
                    </div>
                    <span class="value-display">{move || temperature.get()}</span>
                </label>
                <input type="range" min="0.1" max="2.0" step="0.1" 
                    prop:value=move || temperature.get()
                    on:input=move |ev| set_temperature.set(event_target_value(&ev).parse().unwrap_or(0.7))
                />
            </div>

            // Top P slide
            <div class="control-group">
                <label style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="display: flex; align-items: center;">
                        "Top P " 
                        <HelpTooltip text="Nucleus sampling. Restricts token selection to the top % of probability mass. 0.9 means considering the top 90%. Lower values reduce diversity."/>
                    </div>
                    <span class="value-display">{move || top_p.get()}</span></label>
                <input type="range" min="0.0" max="1.0" step="0.05" 
                    prop:value=move || top_p.get()
                    on:input=move |ev| set_top_p.set(event_target_value(&ev).parse().unwrap_or(0.9))
                />
            </div>

            // Max Tokens
            <div class="control-group">
                <label style="display: flex; align-items: center;">
                    "Max Tokens"
                    <HelpTooltip text="The maximum number of tokens to generate. Increase this if your answers are getting cut off."/>
                </label>
                <input type="number"
                    prop:value=move || max_tokens.get()
                    on:input=move |ev| set_max_tokens.set(event_target_value(&ev).parse().unwrap_or(200))
                />
            </div>

            // Seed
            <div class="control-group">
                <label style="display: flex; align-items: center;">
                    "Seed (Optional)"
                    <HelpTooltip text="A number to reproduce specific results. Using the same seed with the same settings will generate the exact same response."/>
                </label>
                <input type="number" placeholder="Random"
                    // If no input
                    on:input=move |ev| {
                        let val = event_target_value(&ev);
                        if val.is_empty() {
                            set_seed.set(None);
                        } else {
                            set_seed.set(val.parse().ok());
                        }
                    }
                />
            </div>
            // show if server online
            <div id="server-status">
                <div class={move || format!("status-dot {}", if is_online.get() { "online" } else { "" })}></div>
                <span>{move || status_text.get()}</span>
            </div>
        </div>

        <div id="main-chat">
            // Chat history box
            <div id="chat-history" node_ref=chat_history_ref>
                <For
                    each=move || chat_history.get()
                    // use unique ID
                    key=|msg| msg.id 
                    children=move |msg| {
                        let msg_type = if msg.role == "User" { "user" } else { "ai" };
                        let avatar_text = if msg.role == "User" { "U" } else { "AI" };
                        view! {
                            <div class={format!("message {}", msg_type)}>
                                <div class="avatar">{avatar_text}</div>
                                <div class="content">{msg.content}</div>
                            </div>
                        }
                    }
                />
                <Show when=move || !streaming_content.get().is_empty() || is_generating.get()>
                    <div class="message ai">
                        <div class="avatar">"AI"</div>
                        <div class="content">{move || streaming_content.get()}</div>
                    </div>
                </Show>
            </div>
            // User input box
            <div id="input-area">
                <div class="input-container">
                    <textarea 
                        placeholder="Send a message..."
                        prop:value=move || user_input_text.get()
                        on:input=move |ev| set_user_input_text.set(event_target_value(&ev))
                        // Enter to send message, Shift + enter to start a new line
                        on:keydown=move |ev| {
                            if ev.key() == "Enter" && !ev.shift_key() {
                                ev.prevent_default();
                                send_message();
                            }
                        }
                    ></textarea>
                    // Send button
                    <button id="send-btn" on:click=move |_| send_message() disabled=move || is_generating.get()>
                        "Send"
                    </button>
                </div>
            </div>
        </div>
        
        <Show when=move || loading_overlay.get().is_some()>
             <div id="loading-overlay" style="display: flex;">
                <div class="spinner"></div>
                <h3 style="margin-top: 20px;">{move || loading_overlay.get().unwrap()}</h3>
            </div>
        </Show>
    }
}

fn main() {
    leptos::mount_to_body(|| view! { <App/> })
}