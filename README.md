# ECE1779-Large-Language-Model-Inference-Service
# Project Proposal
## 1. Motivation
Large Language Model (LLM) inference services such as Google's Gemini 2.5, OpenAI's API (ChatGPT, GPT-5, etc.), LLaMA, and models from Mistral AI have become essential building blocks for modern AI systems. These services provide interfaces that take user input and generate predictions or outputs using trained LLMs, making advanced AI capabilities widely accessible. People use them every day as chatbots, code assistants, and study tools without needing high-end hardware or expertise in training large models.

Today, most production-grade inference systems today are built using Python-based frameworks, combined with C++ and CUDA for GPU acceleration. While these pre-existing technologies are powerful and flexible, they still face limitations in scalability, performance, and memory management.

Our motivation for this project comes from two main observations. First, current LLM inference backends, such as vLLM, depend heavily on Python. While Python provides rich machine learning libraries and efficient development capabilities, there exist constraints in concurrent and multithread performance, which are crucial for LLM inference services. One critical bottleneck arises from Python's Global Interpreter Lock (GIL), which allows only one thread executing bytecode at a time, preventing multiple threads from executing Python bytecode simultaneously [1]. A study on Python Concurrency for High-load Multicore Processing further demonstrates that Python's multithreading exhibits poor scalability and degraded performance in CPU-bound tasks, even when using asynchronous frameworks [2]. As a result, real-time or multi-user inference services using Python can experience concurrency limitations and performance bottlenecks.

Second, one of the major challenges in LLM serving systems is the inefficient management of GPU memory. Large models such as LLaMA-2 or Mistral-7B can occupy 10-20 GB of GPU memory per instance, leaving little space for other processes. Many current frameworks, including Python-based vLLM or TensorRT, keep models loaded in GPU memory at all times—even when they are idle—leading to inefficient GPU usage and limited scalability when serving multiple models simultaneously. A recent work by Choi et al. on GPU memory management highlights that existing systems often lack dynamic GPU memory reuse mechanisms, which results in wasted capacity and inflexibility [3]. Furthermore, another article on Dynamic Memory Management for Serving LLMs identifies that GPU memory fragmentation has become a persistent issue in long-running LLM services [4]. When using key-value caches, the system continuously allocates and frees memory. Over time, this process fragments available GPU memory, leaving many small unused gaps that cannot be reused efficiently. These findings indicate that current Python/C++-based inference systems are prone to memory management issues like fragmentation and potential leaks, especially in continuous service environments like LLM serving services.

To address these limitations, our project aims to explore Rust as the foundation language for building a high-performance, streaming-capable LLM inference service. Rust works without the GIL and allows effective parallelism across multiple CPU cores, enabling better concurrency and performance. It has asynchronous runtime libraries such as Tokio that support lightweight, non-blocking I/O-bound tasks, allowing high concurrency performance. Moreover, Rust's strict ownership and lifetime system ensures static and safe memory management, preventing issues like memory leaks or fragmentation. Unlike Python, where memory management depends on runtime garbage collection, Rust enforces deterministic deallocation of space once an object goes out of scope, providing a promising design for an efficient dynamic GPU memory reuse mechanism. Furthermore, Rust's low-level control allows developers to build custom memory allocators or pooling strategies that reduce fragmentation and improve memory management. It can prevent memory corruption or crashes and supports asynchronous streaming, using WebSockets or SSE, for a real-time streaming-supported inference service.


---

## 2. Objective and Key Features
### Objective
### Key Features

---

## 3. Tentative Plan

---

## 4. References
[1] Abhinav Ajitsaria, “What Is the Python Global Interpreter Lock (GIL)?,” Realpython.com, Mar. 06, 2018. Accessed: Oct. 06, 2025. [Online]. Available: https://realpython.com/python-gil/?utm_source  
[2] Aslan Zholdybay and Askar Aituov, “PYTHON CONCURRENCY FOR HIGH-LOAD MULTICORE PROCESSING,” Universum Technical sciences, vol. 134, no. 5, May 2025, doi: https://doi.org/10.32743/UniTech.2025.134.5.20073.  
[3] X. Piao and J.-K. Kim, “GMM: An Efficient GPU Memory Management-based Model Serving System for Multiple DNN Inference Models,” Proceedings of the 53rd International Conference on Parallel Processing, pp. 660–668, Aug. 2024, doi: https://doi.org/10.1145/3673038.3673122.  
[4] R. Prabhu, A. Nayak, J. Mohan, R. Ramjee, and A. Panwar, “vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention,” arXiv.org, 2024. Accessed: Oct. 06, 2025. [Online]. Available: https://arxiv.org/abs/2405.04437?utm_source  
‌
‌


‌
‌
