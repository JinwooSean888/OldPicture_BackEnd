
from fastapi import APIRouter, UploadFile, Form
from fastapi.responses import FileResponse
from PIL import Image
import io, os, uuid

from config import RESULT_DIR
from pipelines.watercolor import watercolor
from pipelines.hints import canny_hint
from pipelines.post_watercolor import add_paper_texture, soft_bleed
from typing import Optional

router = APIRouter()

@router.post("/watercolor")
async def watercolor_api(
    file: UploadFile,
    strength: float = Form(0.55),
    cfg: float = Form(7.0),
    steps: int = Form(10),
    controlnet: Optional[str] = Form(None),  # "canny" | "lineart"
    paper_texture: float = Form(0),
    bleed: int = Form(0)
):
    # Read image
    init = Image.open(io.BytesIO(await file.read())).convert("RGB")

    # Optional ControlNet hint
    control_img = canny_hint(init) if controlnet == "canny" else None

    # Run watercolor conversion
    img, seed = watercolor(
        init_img=init,
        strength=strength,
        cfg=cfg,
        steps=steps,
        controlnet_kind=controlnet,
        control_image=control_img
    )

    # Post-processing: paper texture + soft bleed
    if paper_texture > 0:
        img = add_paper_texture(img, paper_texture)
    if bleed > 0:
        img = soft_bleed(img, bleed)

    # Save and return
    out_id = f"{uuid.uuid4().hex}.png"
    out_path = os.path.join(RESULT_DIR, out_id)
    os.makedirs(RESULT_DIR, exist_ok=True)
    img.save(out_path)

    return {"result_url": f"/api/watercolor/result/{out_id}", "seed": seed}

@router.get("/watercolor/result/{name}")
def get_result(name: str):
    path = os.path.join(RESULT_DIR, name)
    return FileResponse(path, media_type="image/png")
