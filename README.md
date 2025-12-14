# ECE1724-Large-Language-Model-Inference-Service
## Team Information

| Name           | Student ID  | Email                         |
|----------------|-------------|-------------------------------|
| Yingchen Jiang | 1010814610  |                               |
| Yuchen Zoe Xu  | 1006708779  | yuchenzoe.xu@mail.utoronto.ca |


## Video Presentation
https://drive.google.com/file/d/18GvJWGg6Cde0rkAtcILo_ZcqBuS4fiHF/view?usp=sharing

## Video Demo
https://drive.google.com/file/d/18hCu2wLkswSUNpgBWB5x0ySXun6VhEZH/view?usp=sharing

## 1. Motivation
Large Language Model (LLM) inference services, such as Google's Gemini 2.5, OpenAI's API (ChatGPT, GPT-5, etc.), LLaMA, and models from Mistral AI, are now core building blocks for modern AI systems. These services take user input and run trained LLMs to produce useful outputs, making advanced AI widely accessible. People use them every day as chatbots, code assistants, and study tools without needing high-end hardware or training expertise. 

Most production-grade inference systems today are built using Python-based frameworks. While these technologies are powerful and flexible, they still face limitations in scalability, latency, and memory efficiency.

Our motivation for this project comes from two main observations. Firstly, many backends such as vLLM depend heavily on Python. While Python provides rich machine learning libraries and efficient development capabilities, there exist constraints in concurrent and multithread performance, which are crucial for LLM inference services. One critical constraint arises from Python's Global Interpreter Lock (GIL), which allows only one thread to execute bytecode at a time, preventing multiple threads from executing bytecode simultaneously [1]. Before the Python 3.13 and 3.14 version, there was no supported "no-GIL" option, and even in 3.13, the GIL removal is experimental and is not set as the default. Most deployed services still run builds where the GIL is enabled [2]. Additionally, there are measurements that show CPython generally runs slower than other languages, such as Rust, due to interpreter and runtime overhead [3]. As a result, Python-based inference stacks often face slower execution speed and concurrency performance issues, which can affect the latency of the system.

Secondly, Python's memory management can be a problem in long-running services. Objects are freed only when nothing references them, which makes it easy to keep them alive unintentionally—issues often arise in global caches or descriptors. For example, an empirical study of 671 open-source Python projects identified eight common memory leak patterns, showing that Python is susceptible to memory leaks [4]. In practice, services that run continuously may accumulate leaked memory, eventually leading to out-of-memory crashes. Also, Python's dynamic typing adds another risk: many mistakes (such as wrong attribute names, mixing types) only show up at runtime. This increases the chance of bugs being deployed to production, making the framework error-prone. Overall, managing stability and memory efficiency in large, continuously running Python services can become a great challenge.

To address these two limitations, we propose building an LLM inference service with Rust. Rust works without GIL, so CPU-bound tasks can run in parallel across cores; Its async ecosystem, such as Tokio, gives efficient, non-blocking I/O for token-streaming workloads. Moreover, Rust's strict ownership and lifetime system ensures static and safe memory management, preventing memory leaks or race conditions. It enforces deterministic deallocation of space once an object goes out of scope, reducing the vulnerability of memory leaks and late runtime bugs in long-running services. In addition, there is no widely adopted, production-grade Rust system for LLM inference compared to Python/C++; this gap, combined with Rust's strengths in concurrency, latency, and memory safety, motivates us to build a lightweight, reliable LLM inference service that supports real-time streaming outputs, manages multiple models efficiently, and keeps memory usage safe.

## 2. Objectives
The goal of this project is to build a local Rust-based LLM inference service that supports multiple open-source LLMs, handle requests through a RESTful API for easy integration, and deliver responses to frontend interface via real-time token streaming. 

The backend focuses on performance and safety, while the frontend demonstrates a modern Rust-based full-stack approach.

## 3. Core Features
Our project implement the following core features:
### (1) Core Inference Backend
We use a Rust framework Candle to run open-source models, including Phi-2, Mistral-7B-Instruct, and LLaMA-3-8B-Instruct (GGUF). The backend supports:
- Model loading, unloading, and execution;
- Token-by-token response generation;
- Memory/VRAM management to avoid system overload.
### (2) Multi-model support and runtime switching:
Model options are listed in a config file, loaded into a lookup table (map) on startup, and one of the models is marked as the “active” model. Users can switch the active model at runtime through the API, without restarting the whole service.

### (3) REST API for Inference Access
The backend exposes a set of REST endpoints using Axum so the frontend (or any client) can control the service easily. The following are all the APIs we have:

`/health` checks if the server is running

`/models` returns the available model list

`/load_model`, `/set_model`, `/unload_model` handle model loading, unloading and switching

`/infer` runs normal (non-streaming) generation

`/infer_stream` runs streaming generation

### (4) Real-time token streaming
To achieve a chat-like experience, our backend streams tokens instead of waiting for the full response. It uses Server-Sent Events (SSE) to keep a persistent connection to the client, and a Tokio mpsc channel to pass tokens from the inference loop to the HTTP streaming response.

### (5) Web chat UI
We build a rust-based frontend which is built with Leptos and compiled to WebAssembly, so the whole project stays Rust-based end-to-end. The frontend includes a model selector, some parameter controllers, and a chat window that displays streaming responses in real time.

## 4. Additional Features
In addition to the core features, we have several additional features to improve the system's robustness, usability and safety:
### (1) VRAM safety (NVIDIA GPU)
In order to avoid crashes when loading large models, our backend checks available GPU VRAM using `nvidia-smi`. It estimates the model memory cost (with an extra buffer), and only loads models when it is safe.

### (2) Configurable generation
From the frontend interface, users can add generation prompts and tune generation behavior per request using temperature, top_p, max_tokens, and optional seed, which let the users control style of generated response and control output length.

### (3) Request generation cancellation
We supports stopping generation using AbortController, so the users can cancel an ongoing output safely when they are not satisfied with the output.

### (4) Chat export and File import
Users can export the full chat history as a .md file, saving their chat history. Also, they can import a text file or code files to the chat interface. The frontend reads the file content, includes it in the prompt and then sent to the backend, so the model can answer questions using the attached text.

## 5. User’s Guide
In this section, we will explain how to run and use the project, including how to start the backend, interact with its REST APIs, and how to use the web-based frontend interface. 

### Backend Usage
The backend provides a local LLM inference service implemented in Rust. It is responsible for model loading/unloading, inference generation, and real-time streaming of generated tokens.

To start the backend server, first make sure Rust is installed. Then from the project root directory, run the following:
```bash
cd llm_inference_service/backend
cargo run
```
This command will launch an Axum-based REST API server. By default, the server listens on http://localhost:8081 and is ready to accept requests from the frontend.

### REST APIs
We have seven REST endpoints that allow users to manage models and run inference. These APIs can be tested using `curl`.
#### Health check
> This endpoint verifies that the backend server is running correctly and responding to requests.
```bash
curl http://localhost:8081/health
```
#### List models
> This endpoint returns the list of LLM models that can be loaded and used for inference.
```bash
curl http://localhost:8081/models
```
#### Load a model
> This endpoint downloads (if it is first time loading) and loads the specified model into memory. Before loading, the backend performs NVIDIA-based GPU VRAM checks to reduce the risk of out-of-memory errors.
```bash
curl -X POST http://localhost:8081/load_model \
  -H "Content-Type: application/json" \
  -d '{"name": "mistral"}'
```
#### Unload a model
> This endpoint unloads a model from memory and frees GPU VRAM.
```bash
curl -X POST http://localhost:8081/unload_model \
  -H "Content-Type: application/json" \
  -d '{"name": "mistral"}'
```
#### Set a model as active
> This endpoint sets the active model used for subsequent inference requests.
```bash
curl -X POST http://localhost:8081/set_model \
  -H "Content-Type: application/json" \
  -d '{"name": "mistral"}'
```
#### Run inference without streaming
> This endpoint runs a standard inference request and returns the fully generated response. The users can set the generation parameters if they want. 
```bash
curl -X POST http://localhost:8081/infer \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello, how are you?",
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 200,
	"seed": 200
  }'
```
#### Run inference with streaming
> This endpoint runs a inference request with real-time token streaming and returns the generated response token by token. The users can set the generation parameters if they want. 
```bash
curl -X POST http://localhost:8081/infer_stream \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello, how are you?",
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 200,
	"seed": 200
  }'
```
### Frontend Usage
The frontend provides a web-based interface that interacts with all backend features through API endpoints. It is implemented using Leptos and compiled to WebAssembly.

To start the frontend, run the following command (the first two commands are only for first time setup):
```bash
rustup target add wasm32-unknown-unknown #if it is the first time running frontend
cargo install trunk #if it is the first time running frontend
cd llm_inference_service/frontend 
trunk serve --open
```
This command builds the frontend and opens the application in `http://127.0.0.1:8080//`.

Now, we will introduce how to use the frontend interface. The frontend contains the following features:

#### Server status
On the left-side corner, there is the server status showing. Green means server is running while red means not running.
![server_online](/screenshots/server_online.png)

#### Model selection
Users can select which model to use from the dropdown menu. The selected model is set as the active model in the backend.
![model_selection](/screenshots/model_selection.png)

#### System prompt
Users can also set a system prompt to control model's behavior, e.g. speak in a sarcastic way in no more than 30 words.
![system_prompt](/screenshots/system_prompt.png)

#### Generation parameters
The interface provides slider and input fields for generation parameter settings including `temperature`, `top_p`, `max_token`, and optional `seed`, allowing users to control the model generation behaviour. There are also tooltips for these parameters provided so users can check what each parameter does.
![parameters](/screenshots/parameters.png)

#### Request cancellation
If users are not satisfied with the current generating output, they can stop an ongoing generation request using the stop button, which safely aborts the streaming connection.
![stop_generation](/screenshots/stop_generation.png)

#### Chat export
The full chat history can be exported as a Markdown (`.md`) file.
![export_chat](/screenshots/export_chat.png)

#### File attachment support
Users can attach text or source code files that are within certain size limit. The frontend reads the file content and includes it in the prompt sent to the backend, allowing the model to answer questions based on the attached files.
![import_file](/screenshots/import_file.png)

#### Real-time streaming responses
During inference, tokens are received from the backend via streaming and displayed incrementally in the chat window.

## 6. Reproducibility Guide

## 7. Contributions by each team member
| Task | Yuchen | Yingchen |
|------|-----|----------|
| REST API implementation                   | ✓ |   |
| Model loading                             |   | ✓ |
| LLM inference integration (Candle)        |   | ✓ |
| Real-time streaming (SSE + tokio mpsc)    |   | ✓ |
| Frontend UI design (Leptos)               | ✓ |   |
| Frontend–backend integration              | ✓ |   |
| Documentation                             | ✓ | ✓ |
| Presentation & Demo                       | ✓ |   |
## 8. Lessons learned and concluding remarks


## References
[1] Abhinav Ajitsaria, “What Is the Python Global Interpreter Lock (GIL)?,” Realpython.com, Mar. 06, 2018. Accessed: Oct. 06, 2025. [Online]. Available: https://realpython.com/python-gil/?utm_source  
[2] “What’s New In Python 3.13,” Python documentation, 2024. Accessed: Oct. 06, 2025. [Online]. Available: https://docs.python.org/3/whatsnew/3.13.html?utm_source  
[3] Lukas Beierlieb, A. Bauer, R. Leppich, Lukas Iffländer, and S. Kounev, “Efficient Data Processing: Assessing the Performance of Different Programming Languages,” Apr. 2023, doi: https://doi.org/10.1145/3578245.3584691.  
[4] J. Chen, D. Yu, and H. Hu, “Towards an understanding of memory leak patterns: an empirical study in Python,” Software Quality Journal, vol. 31, no. 4, pp. 1303–1330, Jun. 2023, doi: https://doi.org/10.1007/s11219-023-09641-5.  
