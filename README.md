# ECE1779-Large-Language-Model-Inference-Service
# Project Proposal
## 1. Motivation
Large Language Model (LLM) inference services， such as Google's Gemini 2.5, OpenAI's API (ChatGPT, GPT-5, etc.), LLaMA, and models from Mistral AI， are now core building blocks for modern AI systems. These services  take user input and run trained LLMs to produce useful outputs, making advanced AI widely accessible. People use them every day as chatbots, code assistants, and study tools without needing high-end hardware or training expertise.

Most production-grade inference systems today are built using Python-based frameworks, often alongside C++ and CUDA for GPU acceleration. While these technologies are powerful and flexible, they still face limitations in scalability, latency, and memory efficiency.

Our motivation for this project comes from two main observations. Firstly, backends like vLLM depend heavily on Python,. While Python provides rich machine learning libraries and efficient development capabilities, there exist constraints in concurrent and multithread performance, which are crucial for LLM inference services. One critical constraint arises from Python's Global Interpreter Lock (GIL), which allows only one thread executing bytecode at a time, preventing multiple threads from executing bytecode simultaneously [1]. A study on Python Concurrency for High-load Multicore Processing further demonstrates that Python's multithreading exhibits degraded performance in CPU-bound tasks, even when using asynchronous frameworks [2]. While CUDA kernels and C++ extensions can release the GIL during heavy compute, the surrounding Python environment can still become a bottleneck under multi-user, real-time situation. 

Secondly, one of the major challenges in LLM serving systems is the management of GPU memory. Large models such as LLaMA-2 or Mistral-7B can occupy 10-20 GB of GPU memory per instance, leaving little space for other processes. Many serving frameworks keep models resident even when idle, which wastes capacity and limits multi-model deployments. Recent work on GPU memory management highlights that existing systems often lack dynamic GPU memory reuse mechanisms, leading to underutilized memory [3]. Long-running services also face GPU memory fragmentation, especially when managing key-value (KV) cashes [4]. When using KV caches, the system continuously allocates and frees memory. Over time, this process fragments available GPU memory, creating many small gaps that cannot be reused efficiently. Together, these issues indicate that current Python/C++-based inference systems are prone to memory management issues like fragmentation and memory waste, especially in continuous service operation.

To address these limitations, we propose building the LLM inference service in Rust to improve concurrency and memory safety. Rust works without the GIL and allows true parallelism across multcore CPUs, enabling better concurrency and performance. It has asynchronous runtime libraries such as Tokio that provide lightweight, non-blocking I/O, allowing high concurrency performance and real-time streaming. Moreover, Rust's strict ownership and lifetime system ensures static and safe memory management, preventing issues like memory leaks or fragmentation. Rust enforces deterministic deallocation of space once an object goes out of scope, makes it easier to design dynamic GPU memory reuse policies. Furthermore, Rust's low-level control allows developers to build custom memory allocators or pooling strategies to limit fragmentation. It support asynchronous streaming, using WebSockets or SSE, for a real-time streaming-supported inference service.


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
