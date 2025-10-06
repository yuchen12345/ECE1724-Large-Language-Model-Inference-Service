# ECE1779-Large-Language-Model-Inference-Service
# Project Proposal
## 1. Motivation
Large Language Model (LLM) inference services, such as Google's Gemini 2.5, OpenAI's API (ChatGPT, GPT-5, etc.), LLaMA, and models from Mistral AI, are now core building blocks for modern AI systems. These services take user input and run trained LLMs to produce useful outputs, making advanced AI widely accessible. People use them every day as chatbots, code assistants, and study tools without needing high-end hardware or training expertise. 

Most production-grade inference systems today are built using Python-based frameworks, often alongside C++ and CUDA for GPU acceleration. While these technologies are powerful and flexible, they still face limitations in scalability, latency, and memory efficiency.

Our motivation for this project comes from two main observations. Firstly, backends such as vLLM depend heavily on Python. While Python provides rich machine learning libraries and efficient development capabilities, there exist constraints in concurrent and multithread performance, which are crucial for LLM inference services. One critical constraint arises from Python's Global Interpreter Lock (GIL), which allows only one thread to execute bytecode at a time, preventing multiple threads from executing bytecode simultaneously [1]. Before the Python 3.13 version, there was no supported "no-GIL" option, and even in 3.13, the GIL removal is experimental and is not set as the default. Most deployed services still run builds where the GIL is enabled [2]. While CUDA kernels and C++ extensions can release the GIL during heavy compute, the surrounding Python environment can still become a bottleneck under multi-user, real-time situation. Additionally, there are measurements that show CPython generally runs slower than other languages, such as Rust, due to interpreter and runtime overhead [3]. Overall, Python-based inference stacks often face slower execution speed and concurrency performance issue, which can affect the latency of the system.

Secondly, Python's memory management can be a problem in long-running services. Objects are freed only when nothing references them, which makes it easy to keep them alive by mistake—common cases include global caches or descriptors. For example, an empirical study of 671 open-source Python projects identified eight common memory leak patterns [4]. In practice, services that run continuously may accumulate leaked memory until out-of-memory crashes. Also, Python's dynamic typing adds another risk: many mistakes, including wrong attribute names, mixing types. only show up at runtime. This increases the chance of bugs slipping into production, making the framework error-prone. As a result, managing stability and memory efficiency in large, continuously running Python services can become a great challenge.


To address these two limitations, we propose building an LLM inference service with Rust. Rust works without GIL, so CPU-bound tasks can run in parallel across cores, and its async ecosystem, such as Tokio, gives efficient, non-blocking I/O for token-streaming workloads. Moreover, Rust's strict ownership and lifetime system ensures static and safe memory management, preventing memory leaks or race conditions. Rust enforces deterministic deallocation of space once an object goes out of scope, reducing the risk of slow leaks and late runtime bugs in long-running services. Rust also interoperates well with existing kernels and libraries such as CUDA or crates like candl, so we can reuse proven models while keeping the serving layer fast and safe. Moreover, there is no widely adopted, production-grade Rust system for LLM inference compared to Python/C++; this gap, combined with Rust's strengths in concurrency, latency, and memory safety, motivates us to build a lightweight, reliable LLM inference service that supports real-time streaming outputs, manages multiple models efficiently, and keeps memory usage safe.

---

## 2. Objective and Key Features
### Objective
### Key Features

---

## 3. Tentative Plan

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
