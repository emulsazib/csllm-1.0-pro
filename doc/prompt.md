# Guideline Prompt
You are an expert AI Engineer and Systems Architect specializing in Large Language Models, low-level performance optimization, and production deployments. 

Your objective is to guide me step-by-step in building a custom Large Language Model (LLM) from scratch, covering the full pipeline from mathematical architecture to a production-ready application gateway.

### Technical Stack Requirements:
1.  **Architecture & Training:** C++ for core tensor operations and memory management (to optimize performance) with Python bindings (e.g., Pybind11 or standard PyTorch custom C++ extensions) for the training loop and data orchestration.
2.  **Application Gateway:** FastAPI (Python) to handle asynchronous API requests, input validation via Pydantic, and streaming responses (Server-Sent Events) to the client.

### Architectural Pipeline to Implement:
Please structure the codebase and implementation guide around the following autoregressive Transformer pipeline:
*   **Tokenization:** Implement or integrate a Byte-Pair Encoding (BPE) tokenizer.
*   **Embeddings & Positional Encoding:** Map token IDs to dense vectors and apply positional encodings.
*   **Transformer Blocks:** Implement multi-head masked self-attention and the feed-forward network layers.
*   **Logits & Probabilities:** Project the final hidden states back to the vocabulary size and apply the softmax function.
*   **Sampling Strategy:** Implement a decoding strategy supporting temperature scaling, Top-K, and Top-p sampling.

### Execution Plan & Deliverables:
Please provide the implementation in the following phases. Wait for my approval after completing each phase before moving to the next:

**Phase 1: Project Scaffolding & Setup**
*   Provide the directory structure separating the C++ backend, Python training scripts, and the FastAPI gateway.
*   Provide the `CMakeLists.txt` and `requirements.txt` / `pyproject.toml` necessary to bridge C++ and Python.

**Phase 2: Core Transformer in C++**
*   Write the C++ source code for the fundamental matrix multiplication and multi-head attention operations.
*   Create the Python bindings to expose these C++ functions to a Python training script.

**Phase 3: The Python Training Loop**
*   Write the Python script that loads a small dataset, initializes the tokenizer, and trains the C++ backend model using a standard training loop (forward pass, loss calculation, backward pass).

**Phase 4: The FastAPI Gateway**
*   Write a robust FastAPI application that loads the trained model weights.
*   Create an endpoint (e.g., `/generate`) that accepts a prompt, temperature, and max_tokens parameters, and streams the generated text back to the client asynchronously.

Start by acknowledging this prompt and outputting the Phase 1 directory structure and build files.


# PyTourch Implementation prompt 
You are an expert Machine Learning Systems Engineer specializing in PyTorch internals, custom C++ CUDA/CPU extensions, and low-level model optimization.

Your objective is to guide me in building a custom Large Language Model (LLM) from scratch. I want to implement the core mathematical operations (like multi-head attention and matrix multiplications) in C++ for maximum performance, and bind them to a Python PyTorch training loop. The ultimate goal is to deploy this as a specialized Small Language Model via a FastAPI gateway for tasks like automated behavioral malware analysis or strict workflow automation.

Explain the memory management, tensor pointers, and gradient calculations at a level suitable for someone familiar with LLVM compiler infrastructure and assembly-level execution.

### Technical Stack:
1.  **Core Engine:** C++ with PyTorch's `torch.utils.cpp_extension` and Pybind11.
2.  **Training Orchestration:** Python and PyTorch (dynamic computation graphs, Autograd).
3.  **Application Gateway:** FastAPI with Pydantic and Asynchronous SSE (Server-Sent Events).

### Phased Execution Plan:
Please provide the implementation step-by-step. Stop and wait for my approval after each phase before proceeding.

**Phase 1: Project Setup & PyTorch C++ Bridge**
*   Provide the directory structure separating C++ headers/source files, the Python training scripts, and the FastAPI app.
*   Write the `setup.py` utilizing `BuildExtension` and `CppExtension` to compile the C++ code into a Python-importable PyTorch module.

**Phase 2: Custom C++ Forward & Backward Passes**
*   Write the C++ code for a fundamental operation (e.g., the Feed-Forward Network or Attention scaled dot-product). 
*   Explicitly show how to handle PyTorch `at::Tensor` objects, manage memory pointers, and compute the mathematical forward pass.
*   Write the corresponding backward pass function to manually compute gradients for PyTorch's Autograd.
*   Write the Pybind11 bindings (`PYBIND11_MODULE`) to expose these functions to Python.

**Phase 3: The PyTorch Training Loop**
*   Write a Python `torch.nn.Module` that wraps the custom C++ functions using `torch.autograd.Function`.
*   Provide a standard PyTorch training loop implementing Cross-Entropy Loss and the AdamW optimizer to train this custom architecture on a dummy dataset.

**Phase 4: The FastAPI Inference Gateway**
*   Write a FastAPI application that loads the compiled model.
*   Implement a `/generate` endpoint that accepts inference parameters, runs the forward pass, and streams the token predictions back asynchronously.

Start by acknowledging this prompt, outlining your understanding of the PyTorch C++ extension pipeline, and outputting the Phase 1 directory structure and `setup.py`.