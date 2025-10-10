# Chatbot on AI PC Technical Notes

## 1. Repository Layout
```
chatbot_on_AIPC/
|-- app/
|   |-- main.py            # FastAPI entry point
|   |-- api/chat.py        # REST endpoints
|   |-- models/model_manager.py
|   |-- models/ov_inference.py
|   `-- utils/             # download and conversion helpers
|-- static/                # index.html, chat.js, style.css
|-- models/                # OpenVINO IR files live here
|-- config.json            # runtime settings
|-- run.py                 # convenience launcher
`-- requirements.txt       # Python dependencies
```

## 2. Startup Flow
1. `run.py` loads `config.json`, resolves the target device (NPU > GPU > CPU), and starts Uvicorn.
2. `app/main.py` creates the FastAPI app, mounts static files, and registers the chat routes.
3. `model_manager.py` checks for model files in `models/`; if absent it raises an informative error for the user.
4. `ov_inference.py` loads the OpenVINO model, keeps it in memory, and exposes a `generate` helper that returns a single string response.

## 3. Backend Components
- **main.py**: builds the FastAPI application, sets up CORS, serves `static/index.html`, and wires the REST router.
- **api/chat.py**: exposes `POST /api/chat` and `GET /health`. The chat endpoint validates the request, calls the inference helper, and returns the final text plus timing info.
- **models/model_manager.py**: handles model path resolution, basic caching, and exposes a singleton-like accessor so the model loads only once.
- **models/ov_inference.py**: wraps OpenVINO Runtime. It prepares the compiled model, runs generation with max tokens, temperature, top-p, and repetition penalty values taken from the config.
- **utils/download.py** (optional use): helper functions to fetch model artifacts from Hugging Face when needed.

## 4. Frontend Components
- **static/index.html**: minimal chat page with an input box, send button, and message list.
- **static/chat.js**: submits prompts via `fetch` to `/api/chat`, appends the response, and maintains session memory in the browser only.
- **static/style.css**: light styling for desktop use; responsive tweaks keep the layout usable on smaller screens.

## 5. API Contract
### 5.1 `GET /health`
Returns application status and device information.
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "NPU"
}
```

### 5.2 `POST /api/chat`
Accepts a single prompt and returns the completed answer (no token streaming).
```json
// Request
{
  "message": "Hello!",
  "max_tokens": 512,
  "temperature": 0.7,
  "top_p": 0.9
}

// Response
{
  "response": "Hi there, how can I help you?",
  "inference_time": 2.14,
  "tokens_generated": 120
}
```
Validation keeps the payload small (e.g., prompt length, positive numeric settings) and falls back to defaults when optional fields are omitted.

## 6. Configuration
`config.json` drives model metadata and inference defaults.
```json
{
  "model": {
    "name": "OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-int4-cw-ov",
    "local_dir": "models",
    "max_context_length": 8192
  },
  "inference": {
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.05
  },
  "server": {
    "host": "127.0.0.1",
    "port": 8000,
    "log_level": "info"
  },
  "hardware": {
    "preferred_device": "NPU",
    "fallback_order": ["GPU", "CPU"],
    "precision": "FP16"
  }
}
```
Environment variables or CLI flags can override the server host/port if desired.

## 7. Model Handling
- Place `openvino_model.bin` and `openvino_model.xml` inside `models/`.
- Large model snapshots from Hugging Face should be downloaded once and reused.
- Loading happens lazily on the first request; subsequent requests reuse the same compiled model.

## 8. Dependency Highlights
- FastAPI 0.110+
- OpenVINO Runtime with NPU support
- optimum-intel for model conversions (if needed)
- transformers for tokenizer utilities

## 9. Testing Checklist
- Unit test model loading and generation helpers (e.g., mock the runtime for quick feedback).
- Hit `GET /health` to verify the service starts and the model is ready.
- Send a sample prompt to `/api/chat` and confirm the response arrives as a single JSON payload.
