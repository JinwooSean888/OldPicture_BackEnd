import numpy as np
import cv2
from PIL import Image

import numpy as np
from PIL import Image

def add_paper_texture(pil_img: Image.Image, intensity: float = 0.2, tex=None):
    """
    pil_img: PIL.Image (RGB/RGBA 허용)
    intensity: 0~1
    tex: (H,W) 또는 (H,W,1)/(H,W,3) numpy, 선택 (없으면 내부 생성 예시 사용 가능)
    """
    # 1) 이미지 준비
    img = np.asarray(pil_img.convert("RGB"), dtype=np.float32) / 255.0  # (H,W,3)
    H, W, _ = img.shape
    intensity = float(np.clip(intensity, 0.0, 1.0))

    # 2) 텍스처 준비 (예: 외부에서 (H,W)로 들어오는 경우가 대부분)
    if tex is None:
        # 간단 노이즈(종이질감 대체): 필요시 자체 텍스처로 교체 가능
        rng = np.random.default_rng(123)
        tex = rng.normal(0.5, 0.15, size=(H, W)).astype(np.float32)
    else:
        tex = np.asarray(tex)
        # 크기 안 맞으면 리사이즈
        if tex.ndim == 2 and (tex.shape[0] != H or tex.shape[1] != W):
            from PIL import Image as _PIL
            tex = np.array(_PIL.fromarray((tex*255).astype(np.uint8)).resize((W, H), _PIL.BICUBIC), dtype=np.float32) / 255.0
        elif tex.ndim == 3 and (tex.shape[0] != H or tex.shape[1] != W):
            from PIL import Image as _PIL
            tex = np.array(_PIL.fromarray((tex[...,0]*255).astype(np.uint8)).resize((W, H), _PIL.BICUBIC), dtype=np.float32) / 255.0

    # 3) 스케일 정리: 0~1
    if tex.dtype != np.float32 and tex.dtype != np.float64:
        tex = tex.astype(np.float32)
    if tex.max() > 1.0 or tex.min() < 0.0:
        # 0~255 범위로 들어왔을 가능성
        tex = np.clip(tex, 0, 255) / 255.0
    else:
        tex = np.clip(tex, 0.0, 1.0)

    # 4) 채널 맞추기
    if tex.ndim == 2:
        tex = tex[..., None]         # (H,W) -> (H,W,1)
    if tex.shape[-1] == 1:
        tex = np.repeat(tex, 3, axis=-1)   # (H,W,1) -> (H,W,3)

    # 5) 블렌딩
    out = img * (1.0 - intensity) + (img * tex) * intensity
    out = np.clip(out, 0.0, 1.0)
    out = (out * 255.0).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")

def soft_bleed(pil_img, radius=3, edge_dark=0.06):
    """Soft bilateral blur + slight edge darkening to mimic watercolor bleed."""
    img = np.array(pil_img.convert("RGB"))
    blurred = cv2.bilateralFilter(img, d=9, sigmaColor=50, sigmaSpace=radius)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 140)
    edges = cv2.GaussianBlur(edges, (5,5), 0)
    mask = (edges.astype(np.float32) / 255.0)[..., None]
    out = blurred.astype(np.float32) - edge_dark * 255 * mask
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


