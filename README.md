
# AI Watercolor Backend (FastAPI)

A ready-to-run FastAPI backend that converts photos into watercolor-style images
using Stable Diffusion (img2img) and optional ControlNet canny hints, plus
simple post-processing (paper texture & bleed).

## Quick Start (VS Code)

1. Open this folder in Visual Studio Code.
2. Create venv and select interpreter:
   ```bash
   python -m venv .venv
   # Activate:
   # Windows: .venv\Scripts\activate
   # macOS/Linux: source .venv/bin/activate
   ```
3. Install deps:
   ```bash
   pip install -r requirements.txt
   ```
4. Run:
   - Terminal: `uvicorn main:app --reload --port 8000`
   - or VS Code: `Run and Debug` -> **FastAPI: Run server**
5. Open Swagger UI: http://127.0.0.1:8000/docs

## API
- `POST /api/watercolor`
  - form-data:
    - `file`: image file
    - `strength` (float, default 0.65)
    - `cfg` (float, default 7.0)
    - `steps` (int, default 32)
    - `controlnet` (str, optional: "canny")
    - `paper_texture` (float 0~0.5, default 0.25)
    - `bleed` (int 0~6, default 3)
  - response:
    ```json
    {"result_url": "/api/watercolor/result/<id>.png", "seed": 1234567}
    ```
- `GET /api/watercolor/result/{name}` -> returns the PNG result.

## Notes
- Requires a GPU (CUDA) for best performance; CPU works but slower.
- Models are cached under `~/.cache/huggingface/`.
- Results saved under `media/results/`.
- For production, replace synthetic paper texture with a real paper texture multiply blend.
# OldPicture_BackEnd_New
