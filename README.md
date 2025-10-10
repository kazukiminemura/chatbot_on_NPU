# Chatbot on AI PC

This repository provides a small local chatbot that runs the DeepSeek-R1-Distill-Qwen-1.5B-int4-cw-ov model with OpenVINO on an Intel AI PC (NPU). The app stays entirely on your machine and serves a simple browser UI.

## Features
- Local inference accelerated by OpenVINO with automatic fallback to GPU or CPU when no NPU is available
- Browser-based chat interface that returns the full answer after each request (no streaming requirement)
- Works fully offline once the model files are downloaded

## Requirements
- Windows 10 or 11
- Python 3.9-3.12
- Intel AI PC with NPU (recommended), GPU or CPU fallback is supported
- At least 8 GB RAM and 8 GB of free disk space

## Quick Start
1. Clone and enter the repository:
   ```powershell
   git clone https://github.com/kazukiminemura/chatbot_on_AIPC.git
   cd chatbot_on_AIPC
   ```
2. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate
   python -m pip install --upgrade pip wheel
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
4. Start the application:
   ```powershell
   python run.py
   ```
5. Open `http://localhost:8000` in your browser to start chatting.

## Model Files
When the server runs for the first time it expects the OpenVINO model files (`openvino_model.bin` and `openvino_model.xml`) under the `models/` directory. If they are missing, download them manually with Hugging Face CLI:
```powershell
pip install "huggingface_hub[cli]"
huggingface-cli download OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-int4-cw-ov ^
  --local-dir models/models--OpenVINO--DeepSeek-R1-Distill-Qwen-1.5B-int4-cw-ov
```
After the download, copy the `openvino_model.bin` and `openvino_model.xml` files from the snapshot folder into `models/`.

> Windows Note: If you see errors about long paths, run `git config --system core.longpaths true` or clone the repository into a shorter directory name.

## Troubleshooting
- **Package conflicts**: Remove old packages inside the virtual environment (`pip uninstall starlette`) and retry `pip install -r requirements.txt`.
- **Missing `huggingface_hub.errors`**: Update the hub client with `pip install --upgrade "huggingface-hub>=0.23.0"`.
- **Model cannot be opened**: Ensure the OpenVINO model files are present in `models/` and not locked by another process.

## Project Layout
```
chatbot_on_AIPC/
|-- app/                # FastAPI backend and model helpers
|-- static/             # HTML, CSS, JavaScript for the web client
|-- models/             # Place OpenVINO model files here
|-- docs/               # Simplified documentation
|-- config.json         # Runtime configuration
`-- requirements.txt    # Python dependencies
```

## Documentation
- `docs/requirements_definition.md`
- `docs/technical_specification.md`
