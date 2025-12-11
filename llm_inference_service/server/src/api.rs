use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response, Sse},
    Json,
};
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

use crate::config::ConfigManager;
use crate::inference::{InferenceEngine, InferenceRequest, StreamToken};
use crate::model_manager::ModelManager;

pub struct AppState {
    pub config_manager: Arc<ConfigManager>,
    pub model_manager: Arc<ModelManager>,
    pub inference_engine: Arc<InferenceEngine>,
}

#[derive(Serialize)]
pub struct ApiResponse<T> {
    success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<T>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

impl<T> ApiResponse<T> {
    fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
        }
    }

    fn error(msg: String) -> ApiResponse<()> {
        ApiResponse {
            success: false,
            data: None,
            error: Some(msg),
        }
    }
}

// List all available models
pub async fn list_models(
    State(state): State<Arc<AppState>>,
) -> Json<ApiResponse<Vec<crate::config::ModelConfig>>> {
    let models = state.config_manager.list_models();
    Json(ApiResponse::success(models))
}

// List loaded models
#[derive(Serialize)]
pub struct LoadedModelsResponse {
    models: Vec<String>,
}

pub async fn list_loaded(
    State(state): State<Arc<AppState>>,
) -> Json<ApiResponse<LoadedModelsResponse>> {
    let models = state.model_manager.list_loaded();
    Json(ApiResponse::success(LoadedModelsResponse { models }))
}

// Load a model
pub async fn load_model(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Response {
    match state.config_manager.find_model(&name) {
        Some(config) => match state.model_manager.load_model(config) {
            Ok(_) => (
                StatusCode::OK,
                Json(ApiResponse::success(format!("Model '{}' loaded", name))),
            )
                .into_response(),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiResponse::<()>::error(e.to_string())),
            )
                .into_response(),
        },
        None => (
            StatusCode::NOT_FOUND,
            Json(ApiResponse::<()>::error(format!(
                "Model '{}' not found in config",
                name
            ))),
        )
            .into_response(),
    }
}

// Unload a model
pub async fn unload_model(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Response {
    match state.model_manager.unload_model(&name) {
        Ok(_) => (
            StatusCode::OK,
            Json(ApiResponse::success(format!("Model '{}' unloaded", name))),
        )
            .into_response(),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(ApiResponse::<()>::error(e.to_string())),
        )
            .into_response(),
    }
}

// Reload config
pub async fn reload_config(State(state): State<Arc<AppState>>) -> Response {
    match state.config_manager.reload() {
        Ok(_) => (
            StatusCode::OK,
            Json(ApiResponse::success("Config reloaded successfully")),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse::<()>::error(e.to_string())),
        )
            .into_response(),
    }
}

// Non-streaming inference
#[derive(Deserialize)]
pub struct InferenceApiRequest {
    model: String,
    prompt: String,
    max_tokens: Option<usize>,
    temperature: Option<f64>,
}

#[derive(Serialize)]
pub struct InferenceApiResponse {
    text: String,
    tokens_generated: usize,
}

pub async fn inference(
    State(state): State<Arc<AppState>>,
    Json(req): Json<InferenceApiRequest>,
) -> Response {
    if !state.model_manager.is_loaded(&req.model) {
        return (
            StatusCode::BAD_REQUEST,
            Json(ApiResponse::<()>::error(format!(
                "Model '{}' is not loaded",
                req.model
            ))),
        )
            .into_response();
    }

    let inference_req = InferenceRequest {
        model: req.model,
        prompt: req.prompt,
        max_tokens: req.max_tokens,
        temperature: req.temperature,
    };

    match state.inference_engine.generate(inference_req).await {
        Ok(response) => (
            StatusCode::OK,
            Json(ApiResponse::success(InferenceApiResponse {
                text: response.text,
                tokens_generated: response.tokens_generated,
            })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse::<()>::error(e.to_string())),
        )
            .into_response(),
    }
}

// Streaming inference
#[derive(Serialize)]
struct SseToken {
    token: Option<String>,
    done: bool,
    error: Option<String>,
}

pub async fn inference_stream(
    State(state): State<Arc<AppState>>,
    Json(req): Json<InferenceApiRequest>,
) -> Response {
    if !state.model_manager.is_loaded(&req.model) {
        return (
            StatusCode::BAD_REQUEST,
            Json(ApiResponse::<()>::error(format!(
                "Model '{}' is not loaded",
                req.model
            ))),
        )
            .into_response();
    }

    let (tx, rx) = mpsc::channel(100);

    let inference_req = InferenceRequest {
        model: req.model,
        prompt: req.prompt,
        max_tokens: req.max_tokens,
        temperature: req.temperature,
    };

    let engine = state.inference_engine.clone();
    tokio::spawn(async move {
        if let Err(e) = engine.generate_stream(inference_req, tx.clone()).await {
            let _ = tx.send(StreamToken::Error(e.to_string())).await;
        }
    });

    let stream = ReceiverStream::new(rx).map(|token| {
        let event = match token {
            StreamToken::Token(t) => SseToken {
                token: Some(t),
                done: false,
                error: None,
            },
            StreamToken::Done => SseToken {
                token: None,
                done: true,
                error: None,
            },
            StreamToken::Error(e) => SseToken {
                token: None,
                done: true,
                error: Some(e),
            },
        };
        Ok::<_, Infallible>(axum::response::sse::Event::default().json_data(event).unwrap())
    });

    Sse::new(stream).into_response()
}