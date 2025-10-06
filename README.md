# ECE1779-Large-Language-Model-Inference-Service
# Project Proposal
## 1. Motivation
Large Language Model (LLM) inference services, such as Google's Gemini 2.5, OpenAI's API (ChatGPT, GPT-5, etc.), LLaMA, and models from Mistral AI, are now core building blocks for modern AI systems. These services take user input and run trained LLMs to produce useful outputs, making advanced AI widely accessible. People use them every day as chatbots, code assistants, and study tools without needing high-end hardware or training expertise. 

Most production-grade inference systems today are built using Python-based frameworks, often alongside C++ and CUDA for GPU acceleration. While these technologies are powerful and flexible, they still face limitations in scalability, latency, and memory efficiency.

Our motivation for this project comes from two main observations. Firstly, backends such as vLLM depend heavily on Python. While Python provides rich machine learning libraries and efficient development capabilities, there exist constraints in concurrent and multithread performance, which are crucial for LLM inference services. One critical constraint arises from Python's Global Interpreter Lock (GIL), which allows only one thread to execute bytecode at a time, preventing multiple threads from executing bytecode simultaneously [1]. Before the Python 3.13 version, there was no supported "no-GIL" option, and even in 3.13, the GIL removal is experimental and is not set as the default. Most deployed services still run builds where the GIL is enabled [2]. While CUDA kernels and C++ extensions can release the GIL during heavy compute, the surrounding Python environment can still become a bottleneck under multi-user, real-time situation. Additionally, there are measurements that show CPython generally runs slower than other languages, such as Rust, due to interpreter and runtime overhead [3]. Overall, Python-based inference stacks often face slower execution speed and concurrency performance issue, which can affect the latency of the system.

Secondly, Python's memory management can be a problem in long-running services. Objects are freed only when nothing references them, which makes it easy to keep them alive by mistake—common cases include global caches or descriptors. For example, an empirical study of 671 open-source Python projects identified eight common memory leak patterns [4]. In practice, services that run continuously may accumulate leaked memory until out-of-memory crashes. Also, Python's dynamic typing adds another risk: many mistakes (such as wrong attribute names, mixing types) only show up at runtime. This increases the chance of bugs slipping into production, making the framework error-prone. As a result, managing stability and memory efficiency in large, continuously running Python services can become a great challenge.


To address these two limitations, we propose building an LLM inference service with Rust. Rust works without GIL, so CPU-bound tasks can run in parallel across cores; Its async ecosystem, such as Tokio, gives efficient, non-blocking I/O for token-streaming workloads. Moreover, Rust's strict ownership and lifetime system ensures static and safe memory management, preventing memory leaks or race conditions. Rust enforces deterministic deallocation of space once an object goes out of scope, reducing the risk of slow leaks and late runtime bugs in long-running services. Rust also interoperates well with existing kernels and libraries such as CUDA or crates like Candle, allowing us to reuse proven model implementations while keeping the serving layer fast and memory-safe. Moreover, there is no widely adopted, production-grade Rust system for LLM inference compared to Python/C++; this gap, combined with Rust's strengths in concurrency, latency, and memory safety, motivates us to build a lightweight, reliable LLM inference service that supports real-time streaming outputs, manages multiple models efficiently, and keeps memory usage safe.

---

## 2. Objective and Key Features
The goal of this project is to build a local LLM (Large Language Model) inference system that can host several lightweight open-source models, handle inference requests through a REST API, and send back responses in real time using streaming. Everything will run on local hardware to show that efficient multi-model LLM serving is possible without using cloud or commercial APIs.
Specifically, the project aims to implement the following components:
### (1) Core Inference Backend
We will integrate a Rust-based framework such as Candle to run quantized open-source models (e.g., Mistral and LLaMA variants, each between 1–3B parameters). The backend will support:
- Model loading, unloading, and inference execution;
- Tokenization and incremental text generation;
- Basic concurrency control to handle up to 10 simultaneous inference requests;
- Resource management to prevent system overload.
### (2) Multi-Model Management
The service will maintain a registry of available models and allow users to switch between them dynamically. At least two models will be loaded during demonstration to show that the system can handle multiple configurations (e.g., Mistral-7B for general tasks and LLaMA-3B for lightweight inference).
Endpoints will be provided to list, load, and unload models on demand.
### (3) REST API for Inference Access
Using Rocket, the backend will expose a set of REST endpoints:
- GET /models — list all available models and their status.
- POST /load — load a specific model into memory.
- POST /infer — run inference and return a full response.
- POST /infer?stream=true — stream the model’s output token by token.
    The API will use structured JSON formats for both requests and responses to ensure clarity and compatibility with common tools like curl and Postman.
### (4) Streaming Output for Real-Time Interaction
The system will implement Server-Sent Events (SSE) to support real-time token streaming. This allows the client to receive model outputs incrementally, simulating a conversational typing experience. The implementation will showcase asynchronous request handling and concurrency in Rust.
### (5) Minimal Chat Frontend
A simple web-based interface (HTML + JavaScript) will demonstrate the service’s usability. Users can send prompts, view streaming responses in real time, and switch between loaded models. This frontend will serve as a functional demo rather than a production-grade UI.
### (6) Evaluation and Deliverables
We will evaluate:
- Response latency and memory usage under different workloads;
- Concurrency performance with multiple simultaneous clients;
- Accuracy and stability of model outputs.
The final deliverables will include:
- Rust source code for the backend;
- Frontend chat demo;
- Setup and API documentation;
- Performance summary and a short demo video.

By the end of the project, we will have a complete local LLM inference service that supports multiple models, provides real-time streaming, and manages concurrent users effectively — all running on local machines.



---

## 3. Tentative Plan
- Member A 
    - Focuses on integrating the Candle or Burn library for running LLMs.
    - Implements the model loading, unloading, and inference execution logic.
    - Tunes performance parameters such as batch size and memory mapping.
    - Ensures that inference works both synchronously and with token streaming.
- Member B
    - Backend Infrastructure: 
        - Responsible for the overall server design and implementation using Axum.
        - Builds REST endpoints and ensures concurrent handling of inference requests.
        - Implements Server-Sent Events streaming pipeline.
        - Works closely with the model integration lead to expose inference APIs.
    - Frontend and Testing
	    - Develops the minimal web-based chat interface for demonstration.
	    - Implements client-side streaming logic.
	    - Designs test cases to validate all API endpoints.
	    - Conducts performance evaluations and documents results.
We plan to organize the work into five phases—including setup, vertical slice, feature expansion, test, and documentation/delivery phases:

### Phase 1—Setup (10-06 to 10-12)
In this phase, we will research more on building an LLM inference service to build a foundational understanding of the Rust ML ecosystem. We will define our system architecture and stack for the project: Candle, Burn, or Mistral.rs for inference, Axum/Tokio for the server, and SSE or WebSockets for streaming. 

### Phase 2—Vertical slice (10-13 to 10-26)
Our goal in this phase will be to implement a basic backend framework that will serve as the backbone of the inference service. We will put one model online end-to-end, while the tasks include: 
- Integrate one inference backend (Candle or Mistral.rs).
- Build a small Axum/Tokio server with one HTTP endpoint and SSE streaming.
- Add a minimal API endpoint to verify streaming.
  
### Phase 3—Feature expansion (10-27 to 11-16)
In this stage, we will make our service functional, which stays responsive under load and supports multiple models safely. Here are the tasks in this stage:
- Add a model manager to list, load, unload, and select multiple models at runtime.
- Ensure the service can handle many requests without letting the server overload. Also, keep a limit on how many run at the same time.
- Keep a stable API that supports streaming. Check the input and return useful error messages.
  
### Phase 4—Test (11-17 to 12-01)
This phase will mainly be for testing and improving as we need to make sure the systen works as expected and can be used normally and stably. There are several tasks:
- Try common paths end-to-end, such as send a prompt, get responses, switch between models, and handle invalid input.
- Check that the tokens are sent continuously without random stops.
- Check the response time and performance of the service when multiple users send requests at the same time.
  
### Phase 5-Documentation and delivery (12-01 to 12-15)
In this part, we will finish a clear final documentation, prepare presentation slides, and demo video. Once everything is working normally, we will package the code so others can run it easily.

---

## 4. References
[1] Abhinav Ajitsaria, “What Is the Python Global Interpreter Lock (GIL)?,” Realpython.com, Mar. 06, 2018. Accessed: Oct. 06, 2025. [Online]. Available: https://realpython.com/python-gil/?utm_source  
[2] “What’s New In Python 3.13,” Python documentation, 2024. Accessed: Oct. 06, 2025. [Online]. Available: https://docs.python.org/3/whatsnew/3.13.html?utm_source  
[3] Lukas Beierlieb, A. Bauer, R. Leppich, Lukas Iffländer, and S. Kounev, “Efficient Data Processing: Assessing the Performance of Different Programming Languages,” Apr. 2023, doi: https://doi.org/10.1145/3578245.3584691.  
[4] J. Chen, D. Yu, and H. Hu, “Towards an understanding of memory leak patterns: an empirical study in Python,” Software Quality Journal, vol. 31, no. 4, pp. 1303–1330, Jun. 2023, doi: https://doi.org/10.1007/s11219-023-09641-5.  
‌
‌
‌
‌


‌
‌
