mod config;
mod infer;
mod model;
mod template;

// import standard library
use std::{
    collections::HashMap,
    net::SocketAddr,
    path::PathBuf,
    process::Command,
    sync::{Arc, Mutex as StdMutex},
};
// import Axum
use axum::{
    Json, 
    Router,
    extract::State,
    response::sse::{Event, KeepAlive, Sse},
    routing::{get, post},
};
// import serde for serializing and deserializing
use serde::{
    Deserialize, 
    Serialize
};
use serde_json::json;
// import tokio for asynchronous runtime handling
use tokio::{
    sync::{Mutex as TokioMutex, Semaphore, mpsc},
    task,
};
// import tokio_stream for SSE
use hf_hub::{
    Repo, 
    RepoType, 
    api::sync::Api
};
use tokio_stream::{StreamExt, wrappers::ReceiverStream};
use tower_http::cors::{Any, CorsLayer}; // CORS // Hugging face

// Internal modules
use config::Settings;
use infer::{InferenceParams, run_inference};
use model::LoadedModel;
use template::apply_chat_template;

// Calculate how much VRAM the GPU has (in order to determine if unload model)
fn detect_vram_mb() -> usize {
    // Run 'nvidia-smi' command
    let output_result = Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .output();

    match output_result {
        Ok(o) => {
            if o.status.success() {
                // Parse output string
                let stdout: std::borrow::Cow<'_, str> = String::from_utf8_lossy(&o.stdout);
                
                let first_line = stdout.lines().next();
                if let Some(line) = first_line {
                    let parsed_result = line.trim().parse::<usize>();
                    if let Ok(total_mb) = parsed_result {
                        // Leave 1GB for system overhead
                        let safe_limit: usize = total_mb.saturating_sub(1024); 
                        println!(
                            "GPU VRAM: {} MB. Using safe limit: {} MB",
                            total_mb, 
                            safe_limit
                        ); 
                        return safe_limit;
                    }
                }
            }
        }
        // For machine without NVIDIA drivers
        Err(_) | Ok(_) => {
            println!("VRAM detection failed. Using default.");
        }
    }
    #[cfg(target_os = "macos")]{
        return 6976;
    } // Default for Mac
    // Default: 8000 - 1024(1G)
    return 6976; 
}

// Check the model file's actual size on disk so that we know
// if we can actually load it
// Return (file path, size in mb)
fn get_model_file_info(name: &str, conf: &config::ModelConfig) -> anyhow::Result<(PathBuf, usize)> {
    let api = Api::new()?;
    let repo_id = conf.repo.clone();
    let repo = api.repo(Repo::new(repo_id, RepoType::Model));
    // This .get() call will download the file if not present, or return path if cached.
    println!("Checking file for '{}'", name);
    let path = repo.get(&conf.file)?;

    // Read file size
    let metadata = std::fs::metadata(&path)?;
    let size_bytes = metadata.len();
    let size_mb = (size_bytes / 1024 / 1024) as usize;

    // Add 500MB buffer for overhead
    let effective_mb = size_mb + 500;
    Ok((path, effective_mb))
}

// --- App State ---
#[derive(Clone)]
struct AppState {
    models: Arc<TokioMutex<HashMap<String, Option<Arc<StdMutex<LoadedModel>>>>>>,
    active_model: Arc<TokioMutex<String>>,
    semaphore: Arc<Semaphore>,
    model_sizes: Arc<TokioMutex<HashMap<String, usize>>>, // Track VRAM size of each model
    vram_limit: usize,
    settings: Arc<Settings>, // Global settings
}
// Response structures in JSON
#[derive(Serialize)]
struct ModelStatus {
    loaded: bool,
    size_mb: usize,
}
#[derive(Serialize)]
struct ModelList {
    models: HashMap<String, ModelStatus>,
    active: String,
    vram_usage: String,
}
#[derive(Deserialize)]
struct SetModelRequest {
    name: String,
}
#[derive(Deserialize)]
struct LoadModelRequest {
    name: String,
}
#[derive(Deserialize)]
struct UnloadModelRequest {
    name: String,
}
#[derive(Deserialize)]
struct InferRequest {
    prompt: String,
    temperature: Option<f64>,
    top_p: Option<f64>,
    max_tokens: Option<usize>,
    seed: Option<u64>,
    system_prompt: Option<String>,
}
// Standardized API response
#[derive(Serialize)]
struct ApiResponse<T> {
    status: String,
    data: Option<T>,
    message: Option<String>,
}
impl<T> ApiResponse<T> {
    fn ok(data: T) -> Json<Self> {
        Json(Self {
            status: "ok".to_string(),
            data: Some(data),
            message: None,
        })
    }
    fn error(msg: impl Into<String>) -> Json<Self> {
        Json(Self {
            status: "error".to_string(),
            data: None,
            message: Some(msg.into()),
        })
    }
}

// POST /load_model
async fn load_model_handler(
    State(state): State<AppState>,
    Json(req): Json<LoadModelRequest>,
) -> Json<ApiResponse<String>> {
    // Check if model exists in config
    let model_conf = {
        let models_map = &state.settings.models;
        match models_map.get(&req.name) {
            Some(c) => c.clone(),
            None => {
                let error_msg = format!("Model '{}' not found in config.", req.name);
                return ApiResponse::error(error_msg);
            }
        }
    };
    // Check if model already loaded
    let models_guard = state.models.lock().await;
    let model_entry = models_guard.get(&req.name).unwrap();
    if model_entry.is_some() {
        let mut active = state.active_model.lock().await;
        *active = req.name.clone();
        let msg = format!("Model '{}' is already loaded.", req.name);
        return ApiResponse::ok(msg);
    }
    drop(models_guard); // Release lock so other requests are not blocked

    // Download and measure, run in a blocking task to avoid block other requests
    let name_clone = req.name.clone();
    let file_info_result =
        task::spawn_blocking(move || get_model_file_info(&name_clone, &model_conf))
            .await
            .unwrap();

    let (_path, required_mb) = match file_info_result {
        Ok(info) => info,
        Err(e) => {
            let error_msg = format!("Failed to fetch model info: {}", e);
            return ApiResponse::error(error_msg);
        }
    };

    // VRAM Check
    let mut models = state.models.lock().await;
    let mut sizes = state.model_sizes.lock().await;

    // Update the size record with actual data
    sizes.insert(req.name.clone(), required_mb);
    // Calculate current total VRAM usage
    let mut current_usage_mb: usize = 0;
    for (name, instance) in models.iter() {
        if instance.is_some() {
            current_usage_mb += sizes.get(name).unwrap_or(&0);
        }
    }
    println!(
        "VRAM Check: Current={}MB, Needed={}MB, Limit={}MB",
        current_usage_mb, 
        required_mb, 
        state.vram_limit
    );

    // Auto unload old models if no enough VRAM
    while current_usage_mb + required_mb > state.vram_limit {
        let mut victim = String::new();
        for (name, instance) in models.iter() {
            if instance.is_some() {
                victim = name.clone();
                break;
            }
        }
        // No enough VRAM space for model to be load
        if victim.is_empty() {
            let error_msg = format!(
                "Model {} ({}MB) is too large for VRAM limit",
                req.name, 
                required_mb
            );
            return ApiResponse::error(error_msg);
        }

        println!("Auto-unloading: {} to free space", victim);
        if let Some(slot) = models.get_mut(&victim) {
            *slot = None; // Free VRAM
        }
        current_usage_mb -= sizes.get(&victim).unwrap_or(&0);
    }

    // Release locks before the heavy loading to keep the server responsive
    drop(models);
    drop(sizes);

    let name_final = req.name.clone();
    //println!("Loading weights for {}", name_final);
    // Actual loading
    let load_task = task::spawn_blocking(move || {
        LoadedModel::load(&name_final)
    });
    let load_result = load_task.await.unwrap();
    match load_result {
        Ok(model) => {
            // Re-acquire lock for newly loaded model.
            let mut models = state.models.lock().await;
            models.insert(req.name.clone(), Some(Arc::new(StdMutex::new(model))));
            // Set as active model
            let mut active = state.active_model.lock().await;
            *active = req.name.clone();
            println!("Model {} loaded successfully.", req.name);
            ApiResponse::ok(format!("Model '{}' loaded.", req.name))
        }
        Err(e) => ApiResponse::error(format!("Failed to load: {}", e)),
    }
}

// GET /models
// Return a list with all models, including status and VRAM usage
async fn list_models(State(state): State<AppState>) -> Json<ModelList> {
    let models = state.models.lock().await;
    let sizes = state.model_sizes.lock().await;
    let active = state.active_model.lock().await;
    let mut result = HashMap::new();
    let mut used = 0;
    for (name, instance) in models.iter() {
        let is_loaded = instance.is_some();
        let size = *sizes.get(name).unwrap_or(&0);
        if is_loaded {
            used += size;
        }
        result.insert(
            name.clone(),
            ModelStatus {
                loaded: is_loaded,
                size_mb: size,
            },
        );
    }
    Json(ModelList {
        models: result,
        active: active.clone(),
        vram_usage: format!("{}/{} MB", used, state.vram_limit),
    })
}

// POST /infer
// Return full response at once
async fn infer_handler(
    State(state): State<AppState>,
    Json(req): Json<InferRequest>,
) -> Json<ApiResponse<String>> {
    // Concurrency Control
    let _permit = state.semaphore.acquire().await.unwrap();
    // Check if there is active model
    let active = state.active_model.lock().await.clone();
    if active.is_empty() {
        return ApiResponse::error("Active model not selected.");
    }
    let models = state.models.lock().await;
    // Clone the Arc to the model
    let model_arc = match models.get(&active) {
        Some(Some(m)) => m.clone(),
        _ => return ApiResponse::error("Model not found or not loaded."),
    };
    drop(models); // Release lock
    // Apply template to input so that it match model's standard input
    let prompt = apply_chat_template(&active, &req.prompt, req.system_prompt.clone());
    let params = InferenceParams {
        temperature: req.temperature,
        top_p: req.top_p,
        max_tokens: req.max_tokens,
        seed: req.seed,
    };
    // Run inference
    let result = task::spawn_blocking(move || {
        let mut model = model_arc.lock().unwrap();
        let mut output = String::new();
        // The callback appends token to string buffer
        let _ = run_inference(
            &mut *model, 
            &prompt, 
            params, 
            |t| output.push_str(&t)
        );
        output
    })
    .await
    .unwrap();
    ApiResponse::ok(format!("[Model: {}] {}", active, result))
}

// POST /infer_stream
// Return response using SSE which means token by token
async fn infer_stream_handler(State(state): State<AppState>, Json(req): Json<InferRequest>) -> Sse<impl tokio_stream::Stream<Item = Result<Event, std::convert::Infallible>>> {
    // Channel for tokens
    let (tx, rx) = mpsc::channel(100);
    task::spawn(async move {
        // Concurrency Control
        let permit = state.semaphore.clone().acquire_owned().await.unwrap();
        let active_guard = state.active_model.lock().await;
        let active = active_guard.clone();
        drop(active_guard);
        
        // Check if there is active model
        if active.is_empty() {
            let error_msg = "Active model not selected.";
            let _ = tx.send(error_msg.into()).await;
            return;
        }
        let models_guard = state.models.lock().await;
        let model_arc_option = models_guard.get(&active);
        let model_arc = match model_arc_option {
            Some(Some(m)) => m.clone(),
            _ => {
                let error_msg = "Model not found or not loaded.";
                let _ = tx.send(error_msg.into()).await;
                return;
            }
        };
        drop(models_guard);// Release lock
        
        let _permit = permit;
        let prompt = apply_chat_template(&active, &req.prompt, req.system_prompt.clone());
        let params = InferenceParams { 
            temperature: req.temperature, 
            top_p: req.top_p, 
            max_tokens: req.max_tokens, 
            seed: req.seed 
        };
        let tx_clone = tx.clone();
        
        // Run inference
        let handle = task::spawn_blocking(move || {
            let _ = tx_clone.blocking_send(format!("[MODEL: {}]", active));   
            // when there is a stop signal from frontend,
            // the mutex becomes poisoned. Ignore the poison state and forcibly acquire lock
            let mut model = model_arc.lock().unwrap_or_else(|e| e.into_inner());

            let res = run_inference(
                &mut *model, 
                &prompt, 
                params, 
                |t| { 
                    let json_msg = json!({ "text": t }).to_string();
                    
                    // if client disconnect, stop inference
                    let send_result = tx_clone.blocking_send(json_msg);
                    if send_result.is_err() {
                        panic!("Client disconnected, stopping inference.");
                    }
                }
            );
            if let Err(e) = res {
                let error_msg = format!("[ERROR] {}", e);
                let _ = tx_clone.blocking_send(error_msg);
            }
            let _ = tx_clone.blocking_send("[DONE]".to_string());
        });
        match handle.await {
            Ok(_) => {}, // task complete
            Err(e) => {
                if e.is_panic() {
                    println!("Inference stopped by user.");
                } else {
                    println!("Inference task failed: {:?}", e);
                }
            }
        }
    });
    
    // Convert the channel receiver into a Stream compatible with Axum SSE
    Sse::new(ReceiverStream::new(rx).map(|m| Ok(Event::default().data(m))))
        .keep_alive(KeepAlive::default())
}

//POST /set_model
// Set active model for one of loaded models
async fn set_model(
    State(state): State<AppState>,
    Json(req): Json<SetModelRequest>,
) -> Json<ApiResponse<String>> {
    let models = state.models.lock().await;
    if !models.contains_key(&req.name) {
        return ApiResponse::error("Model not found.");
    }
    if models.get(&req.name).unwrap().is_some() {
        let mut active = state.active_model.lock().await;
        *active = req.name.clone();
        return ApiResponse::ok(format!("Active model switched to {}", req.name));
    }
    ApiResponse::error(format!("Model {} not loaded.", req.name))
}

//POST /unload_model
// Drop model to free VRAM
async fn unload_model_handler(
    State(state): State<AppState>,
    Json(req): Json<UnloadModelRequest>,
) -> Json<ApiResponse<String>> {
    let mut models = state.models.lock().await;
    if let Some(slot) = models.get_mut(&req.name) {
        if slot.is_some() {
            *slot = None;
            let mut active = state.active_model.lock().await;
            if *active == req.name {
                *active = "".into();
            }
            return ApiResponse::ok(format!("Unload model {}", req.name));
        }
    }
    ApiResponse::error(format!("Model {} not loaded.", req.name))
}

#[tokio::main]
async fn main() {
    // Load settings from config.toml
    let settings = Settings::new().expect("Failed to load config.toml");
    let settings_arc = Arc::new(settings.clone());
    // Initialize state maps
    let mut model_map = HashMap::new();
    let mut size_map = HashMap::new();

    for (name, _) in settings.models {
        model_map.insert(name.clone(), None);
        // Initial size is 0 until we download/measure it
        size_map.insert(name, 0);
    }
    //println!("Loaded config: {:?} models found.", model_map.len());

    // Auto-detect VRAM
    let auto_vram_limit = detect_vram_mb();
    // Create shared application state
    let state = AppState {
        models: Arc::new(TokioMutex::new(model_map)),
        active_model: Arc::new(TokioMutex::new("".to_string())),
        semaphore: Arc::new(Semaphore::new(1)), // Only one allowed for enough VRAM space
        model_sizes: Arc::new(TokioMutex::new(size_map)),
        vram_limit: auto_vram_limit,
        settings: settings_arc,
    };

    // Configure CORS
    let cors_layer = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    // Routers
    let app = Router::new()
        .route("/health", get(|| async { "OK" }))
        .route("/models", get(list_models))
        .route("/set_model", post(set_model))
        .route("/load_model", post(load_model_handler))
        .route("/unload_model", post(unload_model_handler))
        .route("/infer", post(infer_handler))
        .route("/infer_stream", post(infer_stream_handler))
        .with_state(state)
        .layer(cors_layer); // Enable CORS

    // Start server
    let addr = SocketAddr::from(([127, 0, 0, 1], 8081));
    println!("Server running at http://{}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app)
        .await
        .unwrap();
}
