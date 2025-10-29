
import numpy as np
import cv2
from PIL import Image

def canny_hint(pil_img, low=120, high=240):  # ← 임계치 살짝 상향
    # RGB로 강제 변환
    arr = np.array(pil_img.convert("RGB"))

    # ✅ 노이즈 줄이고 엣지만 남기기
    arr = cv2.bilateralFilter(arr, d=7, sigmaColor=50, sigmaSpace=7)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low, high)

    # 가장 얇은 엣지는 살짝 굵게
    edges = cv2.dilate(edges, np.ones((2,2), np.uint8), iterations=1)

    # ✅ 3채널 RGB, 0~255 범위(PIL이 좋아함)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb)  # 흰 선(255) / 검은 배경(0)
# For lineart/openpose, consider using controlnet-aux detectors (optional).
# from controlnet_aux import LineartDetector, OpenposeDetector
