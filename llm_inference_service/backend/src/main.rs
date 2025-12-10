mod model;
mod infer;

use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::Arc,
};
use axum::{
    extract::State,
    routing::{get, post},
    response::sse::{Event, Sse},
    Json, Router,
};
// Serde
use serde::{Deserialize, Serialize};
// Tokio Concurrency
use tokio::{
    sync::{mpsc, Mutex, Semaphore},
    task,
};
// Stream
use tokio_stream::{
    wrappers::ReceiverStream,
    StreamExt,
};
use tower_http::cors::{Any, CorsLayer}; //CORS
// use model::{load_model, LLM};
// use infer::infer;

// Application global state that stores: 
// 1. multi-model registry 2. currently active model 3. semaphore for controlling concurrency
#[derive(Clone)]
struct AppState {
    models: Arc<Mutex<HashMap<String, DummyModel>>>,
    active_model: Arc<Mutex<String>>,
    semaphore: Arc<Semaphore>,
}
// Placeholder for real LLM model, will be replaced by real LLM objects
#[derive(Clone)]
struct DummyModel {
    name: String,
    loaded: bool,
}
#[derive(Serialize)]
struct ModelStatus {
    loaded: bool,
}
//Response format for GET /models
#[derive(Serialize)]
struct ModelList {
    models: HashMap<String, ModelStatus>,
    active: String,
}
// Request format for POST /set_model
#[derive(Deserialize)]
struct SetModelRequest {
    name: String,
}
// Request format for POST /load_model_handler
#[derive(Deserialize)]
struct LoadModelRequest {
    name: String,
}
// Request format for POST /unload_model_handler
#[derive(Deserialize)]
struct UnloadModelRequest {
    name: String,
}
// Request format for POST /infer and POST /infer_stream
#[derive(Deserialize)]
struct InferRequest {
    prompt: String,
}

// Run a normal inference request without streaming
async fn infer_handler(
    State(state): State<AppState>,
    Json(req): Json<InferRequest>,
) -> Json<String> {
    let _permit = state.semaphore.acquire().await.unwrap();
    // Read the currently active model name
    let active = state.active_model.lock().await.clone();
    if active.is_empty() {
        return Json("No active model selected.".to_string());
    }
    let models = state.models.lock().await;
    let model = match models.get(&active) {
        Some(m) => m,
        None => return Json("Active model not found.".to_string()),
    };
    if !model.loaded {
        return Json("Active model is not loaded.".to_string());
    }
    drop(models);
    // Placeholder for real LLM inference
    let result = format!(
        "[Model: {}] Answer to prompt: {}",
        active, req.prompt
    );
    Json(result)
}

// Run streaming inference (token by token output)
async fn infer_stream_handler(
    State(state): State<AppState>,
    Json(req): Json<InferRequest>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, std::convert::Infallible>>> {
    let _permit = state.semaphore.acquire().await.unwrap();
    // Read the currently active model name
    let active = state.active_model.lock().await.clone();
    let prompt = req.prompt.clone();
    // Create channel for streaming tokens
    let (tx, rx) = mpsc::channel(32);
    tokio::spawn(async move {
        // Send model name as first event
        let _ = tx.send(format!("[MODEL {}]", active)).await;
        // Stream characters with an async delay
        for ch in prompt.chars() {
            let _ = tx.send(ch.to_string()).await;
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
        // Send completion marker
        let _ = tx.send("[DONE]".to_string()).await;
    });
    let stream = ReceiverStream::new(rx).map(|msg| {
        Ok(Event::default().data(msg))
    });

    Sse::new(stream)
}

// List all registered model and currently active model
async fn list_models(State(state): State<AppState>) -> Json<ModelList> {
    let models = state.models.lock().await;
    let active = state.active_model.lock().await;
    let mut result = HashMap::new();
    for (name, model) in models.iter() {
        result.insert(name.clone(), ModelStatus { loaded: model.loaded });
    }
    Json(ModelList {
        models: result,
        active: active.clone(),
    })
}

// Set current active model to specific one
async fn set_model(
    State(state): State<AppState>,
    Json(req): Json<SetModelRequest>,
) -> Json<String> {
    let models = state.models.lock().await;
    // Validate if the requested model exists
    let model = match models.get(&req.name) {
        Some(m) => m,
        None => return Json("Model not found.".to_string()),
    };
    // Check whether the model is loaded
    if !model.loaded {
        return Json("Model is not loaded.".to_string());
    }
    drop(models);
    // Update active model
    let mut active = state.active_model.lock().await;
    *active = req.name.clone();
    Json(format!("Active model set to {}", req.name))
}

// Load model into memory
async fn load_model_handler(
    State(state): State<AppState>,
    Json(req): Json<LoadModelRequest>,
) -> Json<String> {
    let mut models = state.models.lock().await;
    // Check whether the requested model is supported
    if !models.contains_key(&req.name) {
        return Json(format!("Model '{}' is not supported.", req.name));
    }
    let model = models.get_mut(&req.name).unwrap();
    // Check if duplicate loading
    if model.loaded {
        return Json(format!("Model '{}' is already loaded.", req.name));
    }
    // Placeholder for real LLM loading
    println!("[Dummy] Loading model {}...", req.name);
    tokio::time::sleep(std::time::Duration::from_millis(600)).await;
    model.loaded = true;
    println!("[Dummy] Model {} loaded.", req.name);
    // If no active model is selected, set this model to default
    let mut active = state.active_model.lock().await;
    if active.is_empty() {
        *active = req.name.clone();
    }
    Json(format!("Model '{}' loaded successfully.", req.name))
}

// Unload model from memory
async fn unload_model_handler(
    State(state): State<AppState>,
    Json(req): Json<UnloadModelRequest>,
) -> Json<String> {
    let mut models = state.models.lock().await;
    // Check whether the requested model is supported
    if !models.contains_key(&req.name) {
        return Json(format!("Model '{}' not found.", req.name));
    }
    let model = models.get_mut(&req.name).unwrap();
    // Prevent unloading not loaded model
    if !model.loaded {
        return Json(format!("Model '{}' is not loaded.", req.name));
    }
    // If unloading model is active, reset active model
    let mut active = state.active_model.lock().await;
    if *active == req.name {
        *active = "".to_string();
    }
    // Placeholder
    println!("[Dummy] Unloading model {}...", req.name);
    tokio::time::sleep(std::time::Duration::from_millis(600)).await;
    model.loaded = false;
    Json(format!("Model '{}' unloaded.", req.name))
}

#[tokio::main]
async fn main() {
    // Initialize the dummy models, to be replaced by real models
    let mut model_map = HashMap::new();
    model_map.insert(
        "phi".to_string(),
        DummyModel {
            name: "phi".to_string(),
            loaded: false,
        },
    );
    model_map.insert(
        "mistral".to_string(),
        DummyModel {
            name: "mistral".to_string(),
            loaded: false,
        },
    );

    // Initialize the shared state
    let state = AppState {
        models: Arc::new(Mutex::new(model_map)),
        active_model: Arc::new(Mutex::new("".to_string())),
        semaphore: Arc::new(Semaphore::new(1)),
    };

    // Router with APIs
    let app = Router::new()
        .route("/health", get(|| async { "OK" }))
        .route("/models", get(list_models))         
        .route("/set_model", post(set_model))
        .route("/load_model", post(load_model_handler))
        .route("/unload_model", post(unload_model_handler))
        .route("/infer", post(infer_handler))
        .route("/infer_stream", post(infer_stream_handler))
        .with_state(state)
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        );
    // Start server
    let addr = SocketAddr::from(([127, 0, 0, 1], 8081));
    println!("Server running at http://{}", addr);
    axum::serve(
        tokio::net::TcpListener::bind(addr).await.unwrap(),
        app,
    )
    .await
    .unwrap();
}
