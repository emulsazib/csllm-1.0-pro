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


# AI Infrastructure

You are an expert Full-Stack AI Infrastructure Engineer and Data Visualization Architect. We are expanding a custom Large Language Model (LLM) built from scratch (C++ backend, PyTorch training loop, and a FastAPI gateway). 

Your objective is to design and implement a comprehensive suite of new features, including a dataset management system, a real-time visual diagnostic web application, a model export pipeline, and a dynamic model configuration system.

### Tech Stack Additions:
*   **Backend:** FastAPI (with WebSockets for real-time logs and telemetry).
*   **Frontend:** React with Three.js/React Three Fiber (for real-time Transformer node/graph animation) and Chart.js/D3.js for probability charts and training loss curves.
*   **Model Serialization:** Safetensors or PyTorch `.pt` format for exporting.

### Execution Plan & Deliverables:
Please implement this iteratively. Wait for my approval after completing each phase before writing code for the next.

**Phase 1: Dataset Plugin & Model Export Module**
*   **Dataset Directory:** Create a `datasets/` directory structure with a plugin interface (e.g., a Python base class) so users can easily drop in `.txt`, `.jsonl`, or `.csv` files, and the `DataLoader` will automatically parse and chunk the text.
*   **Export Pipeline:** Write a Python utility script/API endpoint that serializes the trained model weights and vocabulary into a standalone, exportable format (like `.safetensors` or `.pt`) so it can be deployed anywhere.

**Phase 2: Backend Telemetry & Configuration API**
*   **WebSockets for Training:** Create a FastAPI WebSocket endpoint that streams real-time training logs, epoch progress, and loss metrics.
*   **Versioning & Configuration:** Create an API endpoint (`/configure_model`) that accepts hyperparameters (vocab size, context window, number of layers, hidden dimensions, heads). This endpoint must generate a new versionized configuration file and initialize a fresh model build.

**Phase 3: Web App - Tokenization, Embeddings & Probabilities UI**
*   Write the React frontend components to interact with the LLM inference pipeline:
    *   **Tokenize & Embed Feedback:** A UI that takes a prompt, highlights the generated token chunks, and displays a heatmap or raw vector representation of the resulting embeddings.
    *   **Probabilities & Sampling:** An interactive playground with sliders for Temperature, Top-K, and Top-p. The UI must display a live bar chart of the top token probabilities before the final sample is chosen.

**Phase 4: Web App - Real-Time Transformer Animation & Logs**
*   **Transformer Graph Animation:** Build a React Three Fiber or D3.js component that visually represents the Transformer architecture (Query, Key, Value matrices and Attention Heads). When a prompt is processed, animate the information flow and attention weights in real-time.
*   **Training Dashboard:** Build a dashboard view that subscribes to the WebSocket endpoint from Phase 2, displaying real-time loss curves and standard output logs.

Start by acknowledging this prompt and outputting the exact directory structure required to accommodate the new React frontend, the `datasets/` plugin system, and the new FastAPI WebSocket routes.

# Dashboard Impove 

You are an expert AI Systems Architect and Full-Stack Platform Engineer. We are building an enterprise-grade control dashboard for a custom Large Language Model (LLM) written with a PyTorch/C++ backend, a FastAPI gateway, and a React-based frontend.

Your goal is to implement a complete web dashboard that allows users to dynamically scale model architecture, trigger and monitor training runs on chosen datasets, inspect prompt generations step-by-step (explainable AI), and export model artifacts for external use.

### Key Requirements & Features:

1. **Architectural Scaling & Hyperparameter Configurator:**
   - Build a UI interface to configure model depth and capacity prior to training: number of Transformer layers, hidden dimension (d_model), number of attention heads, feed-forward dimension (d_ff), context length, vocabulary size, and target sequence window.
   - Calculate and display the resulting parameter count (e.g., "125M parameters") and estimated VRAM footprint in real-time as the user adjusts sliders.
   - Version-control model configurations so each architectural change creates a distinct model initialization.

2. **Dashboard-Driven Training Pipeline:**
   - Provide a dataset browser UI where users select from local datasets in `datasets/` (e.g., `.txt`, `.jsonl`).
   - Allow users to launch, pause, or terminate training jobs directly from the UI.
   - Stream training logs, GPU memory usage, learning rate decay, and real-time loss curves via FastAPI WebSockets.

3. **Explainable Inference & Prompt Testing Playground:**
   - Implement an interactive playground where users submit prompts and receive generated responses.
   - Display an "Inference Breakdown Trace" for every generated token:
     * Tokenization view: Highlight exact text chunks mapped to token IDs.
     * Attention inspection: Show a matrix heat-map or node graph revealing how attention heads weighted context tokens.
     * Probability distribution: Render a live bar chart of the top candidate tokens (Top-K/Top-p) and their softmax probabilities prior to final sampling.

4. **Model Export & Packaging Suite:**
   - Implement an export modal allowing one-click packaging of trained weights into standard `.safetensors` format along with `tokenizer.json` and `config.json`.
   - Provide options to download the bundled zip file or export a standalone Python/C++ deployment package.

### Technical Stack:
- **Backend:** FastAPI (Async ASGI, WebSockets, Pydantic, Safetensors serialization).
- **Core Engine:** PyTorch with C++ tensor extensions for model execution and weight initialization.
- **Frontend:** React, Tailwind CSS, Recharts/D3.js (for probability/loss charts), and WebSockets.

---

### Phased Execution Plan:

Please implement this system in the following order. Stop and wait for my confirmation after completing each phase:

**Phase 1: Model Configurator & Parameter Calculator (Backend + Frontend)**
- Implement the FastAPI endpoints and React schema forms to define model architecture hyperparameters.
- Include a dynamic parameter calculator utility function (`calculate_model_params(config)`) returning total trainable parameters and estimated memory overhead.

**Phase 2: Training Manager & WebSocket Telemetry**
- Create the asynchronous background process runner in FastAPI to execute PyTorch training scripts with user-selected datasets and model configs.
- Set up WebSocket handlers to broadcast real-time metrics (Epoch, Step, Loss, Tokens/sec, GPU VRAM) to the React dashboard.

**Phase 3: Explainable Generation Engine & Tracing API**
- Instrument the PyTorch forward/inference loop with hook points to capture token-level intermediate states (token IDs, attention weights, top-N logit probabilities).
- Build the React Prompt Testing Playground to render this data in real time alongside generated tokens.

**Phase 4: Export Manager & Packaging**
- Write backend utility routes to compile model weights, tokenizer configs, and architecture definitions into `.safetensors` release archives.
- Create the UI export modal to trigger and download compiled model packages.

Start by acknowledging this plan and providing the updated folder structure for both the FastAPI backend and React frontend to support these components.