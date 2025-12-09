mod simple_infer;
use simple_infer::simple_infer;

use axum::{extract::State,routing::{get, post},Router, Json,};
use serde::Deserialize;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::task;

#[derive(Clone)]
struct AppState;

#[derive(Deserialize)]
struct InferRequest {
    prompt: String,
}

async fn infer(
    State(_state): State<Arc<AppState>>, 
    Json(req): Json<InferRequest>,
) -> Json<String> {

    let prompt = req.prompt.clone();

    let result = match task::spawn_blocking(move || {
        simple_infer(prompt)
    }).await {
        Ok(Ok(res)) => res,
        Ok(Err(e)) => {
            eprintln!("Inference error: {:?}", e);  
            return Json("Inference failed".to_string());
        }
        Err(e) => {
            eprintln!("Task join error: {:?}", e); 
            return Json("Task failed".to_string());
        }
    };

    Json(result)
}

#[tokio::main]
async fn main() {
    let shared_state = Arc::new(AppState);

    let app = Router::new()
        .route("/health", get(|| async { "OK" }))
        .route("/infer", post(infer))
        .with_state(shared_state);

    let addr = SocketAddr::from(([127, 0, 0, 1], 8081));
    println!("Server running at http://{}", addr);

    axum::serve(
        tokio::net::TcpListener::bind(addr).await.unwrap(),
        app,
    )
    .await
    .unwrap();
}
