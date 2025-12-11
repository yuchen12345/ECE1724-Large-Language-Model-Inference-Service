mod api;
mod config;
mod inference;
mod model_manager;

use anyhow::Result;
use axum::{
    routing::{get, post},
    Router,
};
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use api::AppState;
use config::ConfigManager;
use inference::InferenceEngine;
use model_manager::ModelManager;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("Starting LLM Inference Server");

    // Initialize components
    let config_manager = Arc::new(ConfigManager::new("models.json".to_string())?);
    let model_manager = Arc::new(ModelManager::new());
    let inference_engine = Arc::new(InferenceEngine::new((*model_manager).clone()));

    let state = Arc::new(AppState {
        config_manager,
        model_manager,
        inference_engine,
    });

    // Build router
    let app = Router::new()
        // Model management
        .route("/models", get(api::list_models))
        .route("/models/loaded", get(api::list_loaded))
        .route("/models/:name/load", post(api::load_model))
        .route("/models/:name/unload", post(api::unload_model))
        // Config management
        .route("/config/reload", post(api::reload_config))
        // Inference
        .route("/inference", post(api::inference))
        // .route("/inference/stream", post(api::inference_stream))
        // Health check
        .route("/health", get(|| async { "OK" }))
        .layer(CorsLayer::permissive())
        .with_state(state);

    // Start server
    let addr = "127.0.0.1:8080";
    tracing::info!("Server listening on http://{}", addr);
    tracing::info!("API endpoints:");
    tracing::info!("  GET  /models              - List available models");
    tracing::info!("  GET  /models/loaded       - List loaded models");
    tracing::info!("  POST /models/:name/load   - Load a model");
    tracing::info!("  POST /models/:name/unload - Unload a model");
    tracing::info!("  POST /config/reload       - Reload configuration");
    tracing::info!("  POST /inference           - Non-streaming inference");
    // tracing::info!("  POST /inference/stream    - Streaming inference");
    tracing::info!("  GET  /health              - Health check");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}