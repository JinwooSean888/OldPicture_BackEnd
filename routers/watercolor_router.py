from tensorflow.keras.models import load_model

# FastAPI 파일 업로드용
from fastapi import File

from fastapi import APIRouter, UploadFile, Form
from fastapi.responses import FileResponse
from PIL import Image
import io, os, uuid

from config import RESULT_DIR
from pipelines.watercolor import watercolor
from pipelines.hints import canny_hint
from pipelines.post_watercolor import add_paper_texture, soft_bleed
from typing import Optional
import numpy as np
from skimage import color

router = APIRouter()


from network import unet
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


# --------------------------------------------------------------------
# [2️⃣ colorize API: colorization_model_1031.keras 사용]
# --------------------------------------------------------------------
# 실제 모델 파일 경로
MODEL_PATH = r"C:/Advance_Project/OldPicture_Backend/test.keras"
print("[INFO] 모델 절대 경로:", MODEL_PATH)

try:
    color_model = unet()
    model_weight = r"C:\Advance_Project\OldPicture_AiModel/colorization_model_1031_weights.h5"
    color_model.load_weights(model_weight)

    # color_model = load_model(MODEL_PATH)
    print(f"[✅] Loaded pretrained colorization model: {MODEL_PATH}")
except Exception as e:
    print(f"[❌] Failed to load model: {e}")
    color_model = None


def predict_and_lab2rgb(lab_image):
    pred_ab = color_model.predict(np.expand_dims(lab_image, (0, -1)))  # 차원을 추가
    pred_img = np.zeros((100, 75, 3))

    pred_img[:, :, 0] = lab_image.reshape((100, 75))
    pred_img[:, :, 1:] = pred_ab[0]

    pred_lab = (pred_img * [100, 255, 255]) - [0, 128, 128]
    rgb_img = color.lab2rgb(pred_lab.astype(np.uint8))
    return rgb_img

def colorize_image(pil_img: Image.Image) -> Image.Image:
    """흑백 이미지를 컬러 이미지로 변환"""
    if color_model is None:
        raise RuntimeError("Colorization model not loaded.")

    # 입력 전처리: (256,256) 사이즈로 resize 후 0~1 정규화
    img = np.array(pil_img.resize((75, 100)).convert("L"))
    y = np.array(predict_and_lab2rgb(img))
    print(y.shape, y)
    # x = np.array(img) / 255.0
    # x = np.expand_dims(x, axis=0)
    # print(x.shape)
    # # 예측 수행
    # y = color_model.predict(x)
    # y = np.clip(y[0] * 255, 0, 255).astype(np.uint8)

    return Image.fromarray(y)

@router.post("/colorize")
async def colorize_api(file: UploadFile = File(...)):
    """
    📸 사전학습된 colorization_model_1031.keras 모델을 이용해
    흑백 이미지를 자동으로 컬러화하는 API
    """
    # 업로드 이미지 읽기
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")

    # 모델 실행
    result_img = colorize_image(img)

    # 결과 저장
    out_id = f"{uuid.uuid4().hex}.png"
    out_path = os.path.join(RESULT_DIR, out_id)
    result_img.save(out_path)

    return {"result_url": f"/api/colorize/result/{out_id}"}

@router.get("/colorize/result/{name}")
def get_colorize_result(name: str):
    path = os.path.join(RESULT_DIR, name)
    if not os.path.exists(path):
        return {"error": "File not found"}
    return FileResponse(path, media_type="image/png")