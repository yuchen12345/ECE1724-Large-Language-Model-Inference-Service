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
use model::{load_model, LLM};
use infer::infer;

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
}
//Response format for GET /models
#[derive(Serialize)]
struct ModelList {
    models: Vec<String>,
    active: String,
}
// Request format for POST /set_model
#[derive(Deserialize)]
struct SetModelRequest {
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
    std::thread::spawn(move || {
        tx.blocking_send(format!("[MODEL {}]", active)).ok();
        // Simulate token streaming
        for ch in prompt.chars() {
            tx.blocking_send(ch.to_string()).ok();
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        // Completion marker [DONE]
        tx.blocking_send("[DONE]".to_string()).ok();
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
    let model_names = models.keys().cloned().collect::<Vec<_>>();
    Json(ModelList {
        models: model_names,
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
    if !models.contains_key(&req.name) {
        return Json("Model not found".to_string());
    }
    drop(models);
    // Update active model
    let mut active = state.active_model.lock().await;
    *active = req.name.clone();
    Json(format!("Active model set to {}", req.name))
}


#[tokio::main]
async fn main() {
    // Initialize the dummy models, to be replaced by real models
    let mut model_map = HashMap::new();
    model_map.insert(
        "phi".to_string(),
        DummyModel { name: "phi".to_string() },
    );
    model_map.insert(
        "mistral".to_string(),
        DummyModel { name: "mistral".to_string() },
    );

    // Initialize the shared state
    let state = AppState {
        models: Arc::new(Mutex::new(model_map)),
        active_model: Arc::new(Mutex::new("phi".to_string())),
        semaphore: Arc::new(Semaphore::new(1)),
    };

    // Router with APIs
    let app = Router::new()
        .route("/health", get(|| async { "OK" }))
        .route("/models", get(list_models))         
        .route("/set_model", post(set_model))
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
