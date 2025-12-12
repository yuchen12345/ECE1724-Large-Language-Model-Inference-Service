mod model;
mod infer;
mod template;
// Import config module
#[path = "config.rs"]
mod config;

use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::{Arc, Mutex as StdMutex}, 
    process::Command,
    path::PathBuf,
};
use axum::{
    extract::State,
    routing::{get, post},
    response::sse::{Event, Sse},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tokio::{
    sync::{mpsc, Mutex as TokioMutex, Semaphore},
    task,
};
use tokio_stream::{
    wrappers::ReceiverStream,
    StreamExt,
};
use tower_http::cors::{Any, CorsLayer};
use hf_hub::{api::sync::Api, Repo, RepoType}; // Need hf_hub to resolve paths

use model::LoadedModel;
use infer::{run_inference, InferenceParams};
use template::apply_chat_template;
use config::Settings;

// --- Helper: Auto-detect VRAM ---
fn detect_vram_mb() -> usize {
    println!("Detecting GPU VRAM...");
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .output();

    match output {
        Ok(o) if o.status.success() => {
            let stdout = String::from_utf8_lossy(&o.stdout);
            if let Some(line) = stdout.lines().next() {
                if let Ok(total_mb) = line.trim().parse::<usize>() {
                    // Safety Buffer: Leave 1GB for system/desktop overhead
                    let safe_limit = total_mb.saturating_sub(1024); 
                    println!("Detected GPU VRAM: {} MB. Using safe limit: {} MB", total_mb, safe_limit);
                    return safe_limit;
                }
            }
        }
        _ => { println!("VRAM detection failed. Using default."); }
    }

    #[cfg(target_os = "macos")]
    { return 12000; } // Default for Mac
    
    8000 // Default fallback
}

// --- Helper: Fetch File Size without loading model ---
// Returns (file_path, size_in_mb)
fn get_model_file_info(name: &str, conf: &config::ModelConfig) -> anyhow::Result<(PathBuf, usize)> {
    let api = Api::new()?;
    let repo = api.repo(Repo::new(conf.repo.clone(), RepoType::Model));
    // This .get() call will download the file if not present, or return path if cached.
    // We do this in spawn_blocking, so it's fine.
    println!("Checking file for '{}'... (This may download if not cached)", name);
    let path = repo.get(&conf.file)?;
    
    let metadata = std::fs::metadata(&path)?;
    let size_bytes = metadata.len();
    let size_mb = (size_bytes / 1024 / 1024) as usize;
    
    // Add 500MB buffer for context window (KV Cache) overhead
    let effective_mb = size_mb + 500; 
    
    Ok((path, effective_mb))
}

// --- App State ---
#[derive(Clone)]
struct AppState {
    models: Arc<TokioMutex<HashMap<String, Option<Arc<StdMutex<LoadedModel>>>>>>,
    active_model: Arc<TokioMutex<String>>,
    semaphore: Arc<Semaphore>,
    // Mutex now, because we update sizes dynamically after download
    model_sizes: Arc<TokioMutex<HashMap<String, usize>>>,
    vram_limit_mb: usize,
    // Store config to look up repo info
    settings: Arc<Settings>,
}

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
struct SetModelRequest { name: String }
#[derive(Deserialize)]
struct LoadModelRequest { name: String }
#[derive(Deserialize)]
struct UnloadModelRequest { name: String }
#[derive(Deserialize)]
struct InferRequest {
    prompt: String,
    temperature: Option<f64>,
    top_p: Option<f64>,
    max_tokens: Option<usize>,
    seed: Option<u64>,
}
#[derive(Serialize)]
struct ApiResponse<T> {
    status: String,
    data: Option<T>,
    message: Option<String>,
}
impl<T> ApiResponse<T> {
    fn ok(data: T) -> Json<Self> {
        Json(Self { status: "ok".to_string(), data: Some(data), message: None })
    }
    fn error(msg: impl Into<String>) -> Json<Self> {
        Json(Self { status: "error".to_string(), data: None, message: Some(msg.into()) })
    }
}

// --- Handlers ---

async fn load_model_handler(
    State(state): State<AppState>,
    Json(req): Json<LoadModelRequest>,
) -> Json<ApiResponse<String>> {
    // 1. Check if model exists in config
    let model_conf = match state.settings.models.get(&req.name) {
        Some(c) => c.clone(),
        None => return ApiResponse::error(format!("Model '{}' not found in config.", req.name)),
    };

    // 2. Check if already loaded
    let models_guard = state.models.lock().await;
    if models_guard.get(&req.name).unwrap().is_some() {
        let mut active = state.active_model.lock().await;
        *active = req.name.clone();
        return ApiResponse::ok(format!("Model '{}' is already loaded.", req.name));
    }
    drop(models_guard); // Release lock while downloading

    // 3. Pre-flight: Download & Measure Size (Heavy I/O)
    // We do this BEFORE locking the global model map to avoid blocking other requests
    let name_clone = req.name.clone();
    let file_info_result = task::spawn_blocking(move || {
        get_model_file_info(&name_clone, &model_conf)
    }).await.unwrap();

    let (_path, required_mb) = match file_info_result {
        Ok(info) => info,
        Err(e) => return ApiResponse::error(format!("Failed to fetch model info: {}", e)),
    };

    // 4. VRAM Budget Check (Critical Section)
    let mut models = state.models.lock().await;
    let mut sizes = state.model_sizes.lock().await;
    
    // Update the size record with actual data
    sizes.insert(req.name.clone(), required_mb);

    let mut current_usage_mb: usize = 0;
    for (name, instance) in models.iter() {
        if instance.is_some() {
            current_usage_mb += sizes.get(name).unwrap_or(&0);
        }
    }

    println!("VRAM Check: Current={}MB, Needed={}MB, Limit={}MB", current_usage_mb, required_mb, state.vram_limit_mb);

    // Auto-unload loop
    while current_usage_mb + required_mb > state.vram_limit_mb {
        let mut victim = String::new();
        for (name, instance) in models.iter() {
            if instance.is_some() {
                victim = name.clone();
                break;
            }
        }
        if victim.is_empty() {
             return ApiResponse::error(format!("Model {} ({}MB) is too large for VRAM limit", req.name, required_mb));
        }
        
        println!("Auto-unloading: {} to free space", victim);
        if let Some(slot) = models.get_mut(&victim) {
            *slot = None; // Drop (Free VRAM)
        }
        current_usage_mb -= sizes.get(&victim).unwrap_or(&0);
    }

    // 5. Load the Model (Heavy Compute)
    drop(models); // Release map lock
    drop(sizes);
    
    let name_final = req.name.clone();
    println!("Loading weights for {}...", name_final);
    
    let load_result = task::spawn_blocking(move || {
        LoadedModel::load(&name_final)
    }).await.unwrap();

    match load_result {
        Ok(model) => {
            let mut models = state.models.lock().await;
            models.insert(req.name.clone(), Some(Arc::new(StdMutex::new(model))));
            
            let mut active = state.active_model.lock().await;
            *active = req.name.clone();
            
            println!("Model {} loaded successfully.", req.name);
            ApiResponse::ok(format!("Model '{}' loaded.", req.name))
        },
        Err(e) => ApiResponse::error(format!("Failed to load: {}", e))
    }
}

async fn list_models(State(state): State<AppState>) -> Json<ModelList> {
    let models = state.models.lock().await;
    let sizes = state.model_sizes.lock().await;
    let active = state.active_model.lock().await;
    
    let mut result = HashMap::new();
    let mut used = 0;

    for (name, instance) in models.iter() {
        let is_loaded = instance.is_some();
        let size = *sizes.get(name).unwrap_or(&0);
        if is_loaded { used += size; }
        
        result.insert(name.clone(), ModelStatus { 
            loaded: is_loaded,
            size_mb: size 
        });
    }

    Json(ModelList {
        models: result,
        active: active.clone(),
        vram_usage: format!("{}/{} MB", used, state.vram_limit_mb),
    })
}

async fn infer_handler(State(state): State<AppState>, Json(req): Json<InferRequest>) -> Json<ApiResponse<String>> {
    let _permit = state.semaphore.acquire().await.unwrap();
    let active = state.active_model.lock().await.clone();
    if active.is_empty() { return ApiResponse::error("Active model not selected."); }
    let models = state.models.lock().await;
    let model_arc = match models.get(&active) {
        Some(Some(m)) => m.clone(),
        _ => return ApiResponse::error("Active model not found or not loaded."),
    };
    drop(models);
    let prompt = apply_chat_template(&active, &req.prompt);
    let params = InferenceParams { temperature: req.temperature, top_p: req.top_p, max_tokens: req.max_tokens, seed: req.seed };
    let result = task::spawn_blocking(move || {
        let mut model = model_arc.lock().unwrap();
        let mut output = String::new();
        let _ = run_inference(&mut *model, &prompt, params, |t| output.push_str(&t));
        output
    }).await.unwrap();
    ApiResponse::ok(format!("[Model: {}] {}", active, result))
}

async fn infer_stream_handler(State(state): State<AppState>, Json(req): Json<InferRequest>) -> Sse<impl tokio_stream::Stream<Item = Result<Event, std::convert::Infallible>>> {
    let (tx, rx) = mpsc::channel(100);
    task::spawn(async move {
        let permit = state.semaphore.clone().acquire_owned().await.unwrap();
        let active = state.active_model.lock().await.clone();
        if active.is_empty() { let _ = tx.send("[ERROR] No active model".into()).await; return; }
        let models = state.models.lock().await;
        let model_arc = match models.get(&active) {
            Some(Some(m)) => m.clone(),
            _ => { let _ = tx.send("[ERROR] Model not loaded".into()).await; return; }
        };
        drop(models);
        let _permit = permit;
        let prompt = apply_chat_template(&active, &req.prompt);
        let params = InferenceParams { temperature: req.temperature, top_p: req.top_p, max_tokens: req.max_tokens, seed: req.seed };
        let tx_clone = tx.clone();
        task::spawn_blocking(move || {
            let _ = tx_clone.blocking_send(format!("[MODEL: {}]", active));
            let mut model = model_arc.lock().unwrap();
            let res = run_inference(&mut *model, &prompt, params, |t| { let _ = tx_clone.blocking_send(t); });
            if let Err(e) = res { let _ = tx_clone.blocking_send(format!("[ERROR] {}", e)); }
            let _ = tx_clone.blocking_send("[DONE]".to_string());
        }).await.unwrap();
    });
    Sse::new(ReceiverStream::new(rx).map(|m| Ok(Event::default().data(m))))
}

async fn set_model(State(state): State<AppState>, Json(req): Json<SetModelRequest>) -> Json<ApiResponse<String>> {
    let models = state.models.lock().await;
    if !models.contains_key(&req.name) { return ApiResponse::error("Model not found."); }
    if models.get(&req.name).unwrap().is_some() {
        let mut active = state.active_model.lock().await;
        *active = req.name.clone();
        return ApiResponse::ok(format!("Switched to {}", req.name));
    }
    ApiResponse::error(format!("Model {} not loaded. Call /load_model", req.name))
}

async fn unload_model_handler(State(state): State<AppState>, Json(req): Json<UnloadModelRequest>) -> Json<ApiResponse<String>> {
    let mut models = state.models.lock().await;
    if let Some(slot) = models.get_mut(&req.name) {
        if slot.is_some() {
            *slot = None;
            let mut active = state.active_model.lock().await;
            if *active == req.name { *active = "".into(); }
            return ApiResponse::ok(format!("Unloaded {}", req.name));
        }
    }
    ApiResponse::error("Model not loaded")
}

#[tokio::main]
async fn main() {
    // 1. Load settings (URLs only)
    let settings = Settings::new().expect("Failed to load config.toml");
    let settings_arc = Arc::new(settings.clone());

    // 2. Initialize Maps
    let mut model_map = HashMap::new();
    let mut size_map = HashMap::new();

    for (name, _) in settings.models {
        model_map.insert(name.clone(), None);
        // Initial size is 0 until we download/measure it
        size_map.insert(name, 0); 
    }

    println!("Loaded config: {:?} models found.", model_map.len());

    // 3. Auto-detect VRAM
    let auto_vram_limit = detect_vram_mb();

    let state = AppState {
        models: Arc::new(TokioMutex::new(model_map)),
        active_model: Arc::new(TokioMutex::new("".to_string())),
        semaphore: Arc::new(Semaphore::new(1)),
        model_sizes: Arc::new(TokioMutex::new(size_map)), // Now dynamic
        vram_limit_mb: auto_vram_limit,
        settings: settings_arc,
    };

    let app = Router::new()
        .route("/health", get(|| async { "OK" }))
        .route("/models", get(list_models))         
        .route("/set_model", post(set_model))
        .route("/load_model", post(load_model_handler))
        .route("/unload_model", post(unload_model_handler))
        .route("/infer", post(infer_handler))
        .route("/infer_stream", post(infer_stream_handler))
        .with_state(state)
        .layer(CorsLayer::new().allow_origin(Any).allow_methods(Any).allow_headers(Any));

    let addr = SocketAddr::from(([127, 0, 0, 1], 8081));
    println!("Server running at http://{}", addr);
    axum::serve(tokio::net::TcpListener::bind(addr).await.unwrap(), app).await.unwrap();
}