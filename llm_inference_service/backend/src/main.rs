mod model;
mod infer;

use axum::{
    extract::State,
    routing::{get, post},
    Router, Json,
};
use serde::Deserialize;
use std::{net::SocketAddr, sync::Arc};
use tokio::task;

use model::{load_model, LLM};
use infer::infer;
use tokio::sync::Semaphore;
use axum::response::sse::{Event, Sse};
use tokio_stream::StreamExt;


#[derive(Clone)]
struct AppState {
    llm: Arc<LLM>,
    semaphore: Arc<Semaphore>,
}

#[derive(Deserialize)]
struct InferRequest {
    prompt: String,
}

async fn infer_handler(
    State(state): State<AppState>,
    Json(req): Json<InferRequest>,
) -> Json<String> {
    let llm = state.llm.clone();
    let prompt = req.prompt.clone();
   
    let permit = state.semaphore.acquire().await.unwrap();

    let result = match task::spawn_blocking(move || infer(&llm, prompt)).await {
        Ok(Ok(res)) => res,
        Ok(Err(e)) => {
            eprintln!("Inference error: {:?}", e);
            "Inference failed".to_string()
        }
        Err(e) => {
            eprintln!("Task join error: {:?}", e);
            "Inference failed".to_string()
        }
    };

    drop(permit);

    Json(result)
}

async fn infer_stream_handler(
    State(state): State<AppState>,
    Json(req): Json<InferRequest>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, std::convert::Infallible>>> {
    let llm = state.llm.clone();

    let permit = state.semaphore.clone().acquire_owned().await.unwrap();

    let stream = infer::infer_stream(llm, req.prompt);

    let mapped = stream.map(move |token| {
        let _permit = &permit;
        Ok(Event::default().data(token))
    });

    Sse::new(mapped)
}


#[tokio::main]
async fn main() {
    let llm = load_model().expect("Model loading failed");

    let state = AppState {
        llm: Arc::new(llm),
        semaphore: Arc::new(Semaphore::new(1)),
    };

    let app = Router::new()
        .route("/health", get(|| async { "OK" }))
        .route("/infer", post(infer_handler))
        .route("/infer_stream", post(infer_stream_handler))
        .with_state(state);

    let addr = SocketAddr::from(([127, 0, 0, 1], 8081));
    println!("Server running at http://{}", addr);

    axum::serve(
        tokio::net::TcpListener::bind(addr).await.unwrap(),
        app,
    )
    .await
    .unwrap();
}
