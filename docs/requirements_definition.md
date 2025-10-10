# Chatbot on AI PC Requirements

## 1. Goal
- Provide a local chat app that runs DeepSeek-R1-Distill-Qwen-1.5B-int4-cw-ov through OpenVINO on an Intel AI PC
- Prefer the NPU, but allow GPU or CPU fallback
- Offer a safe, minimal browser UI for local use

## 2. Target Users
- Individuals who want to run an LLM on their own machine
- Developers or creators evaluating Intel AI PC capabilities
- People who do not want to send chat data to the cloud

## 3. Functional Requirements
### 3.1 Chat
- Serve a simple web UI at `http://localhost:8000`
- Accept text prompts and return the full response after inference (no streaming)
- Keep the conversation history in the current browser session and allow clearing it

### 3.2 Model Management
- When the OpenVINO model files are missing, guide the user to place them under `models/`
- Reuse locally cached models to avoid repeated downloads

### 3.3 Configuration
- Allow adjusting inference settings (`max_tokens`, `temperature`, etc.) and device selection via `config.json`
- Optional: accept overrides for host and port through CLI flags or environment variables

## 4. Non-functional Requirements
- Aim for responses within a few seconds per prompt (depends on hardware)
- Keep memory usage around 8 GB or less on standard hardware
- Run fully offline without sending any data outside the local machine

## 5. Technical Constraints
- Backend: Python 3.9-3.12 with FastAPI and Uvicorn
- Inference stack: OpenVINO Runtime, optimum-intel, transformers
- Frontend: HTML, CSS, vanilla JavaScript
- API surface: REST only

## 6. Setup Flow
1. Clone the repository and create a Python virtual environment
2. Install dependencies with `pip install -r requirements.txt`
3. Download the OpenVINO model (e.g., via Hugging Face CLI) and place it in `models/`
4. Run `python run.py` and open the site in a browser

## 7. Possible Future Enhancements
- Persist conversation history (e.g., SQLite)
- Add optional voice input or text-to-speech
- Support switching between multiple models
