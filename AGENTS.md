# Repository Guidelines

### Project Structure & Module Organization
- `onnxocr/`: Core OCR modules (predict_*.py, postprocess, utils), ONNX models under `onnxocr/models/` and test images in `onnxocr/test_images/`.
- `app-service.py`: Minimal Flask JSON API (`/ocr`).
- `webui.py`: Batch-friendly Web UI using `templates/` + `static/`.
- `templates/`, `static/`: Frontend assets for the Web UI.
- `result_img/`: Sample recognition outputs shown in the README.
- `Dockerfile`, `docker-compose.yml`: Containerized runtime.
- `.github/workflows/`: CI and automation.

### Build, Test, and Development Commands
- Create env and install: `pip install -r requirements.txt` (Python >=3.6).
- Quick smoke test: `python test_ocr.py` (runs OCR on a sample image).
- Run JSON API (CPU): `python app-service.py` (serves on `:5005`).
- Run Web UI: `python webui.py` then open `http://localhost:5005/`.
- Docker build/run:
  - `docker build -t onnxocr-service .`
  - `docker run -p 5006:5005 onnxocr-service`

### Coding Style & Naming Conventions
- Python, PEP 8, 4-space indentation.
- Names: `snake_case` for functions/vars, `PascalCase` for classes, module names lowercase.
- Keep functions focused; prefer small utilities in `onnxocr/utils.py`.
- Avoid committing large binaries; place ONNX models under `onnxocr/models/` and reference paths.

### Testing Guidelines
- Current tests are script-based: run `python test_ocr.py` and inspect console output and saved images.
- Add new checks near usage sites (e.g., small verifier scripts per module). If you add `pytest`, name tests `test_*.py` and place under `tests/`.
- Validate both API (`/ocr`) and Web UI flows when changing inference or postprocess code.

### Commit & Pull Request Guidelines
- Commits in this repo are concise and imperative (English or Chinese acceptable). Example: `Add ppocrv5 model and update docs`.
- Reference related files/modules in the body; group logically related changes.
- PRs: include a clear summary, reproduction steps, before/after screenshots for UI changes, and sample request/response for API changes. Link issues when applicable.

### Security & Configuration Tips
- Default runs on CPU (`use_gpu=False`). Match `onnxruntime-gpu` version if enabling GPU.
- Large server models (PP-OCRv5) belong under `onnxocr/models/ppocrv5/` (do not commit to Git).
