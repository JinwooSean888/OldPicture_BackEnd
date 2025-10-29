
import random
import torch
import time   # ⬅ 추가
from PIL import Image
from .stylize_core import WCService, DEVICE
from .presets import WATERCOLOR_PRESET
from typing import Optional


def watercolor(
    init_img: Image.Image,
    width: int = None,
    height: int = None,
    strength: float = None,
    cfg: float = None,
    steps: int = None,
    controlnet_kind: str = None,
    control_image: Image.Image = None,
    seed: int = None
):
    p = WATERCOLOR_PRESET.copy()
    if strength is not None: p["strength"] = float(strength)
    if cfg is not None: p["cfg"] = float(cfg)
    if steps is not None: p["steps"] = int(steps)

    if width and height:
        w = int(round(width / 64) * 64)
        h = int(round(height / 64) * 64)
        init_img = init_img.resize((w, h), Image.LANCZOS)

    if seed is None:
        seed = random.randint(1, 2**31 - 1)

    gen = torch.Generator(device=DEVICE).manual_seed(seed)

    # ✅ 로그 추가
    print(f"[watercolor] start - device: {DEVICE}, controlnet: {controlnet_kind}")
    t0 = time.time()

    pipe = WCService.img2img()
    pipe.controlnet = WCService.controlnet(controlnet_kind) if controlnet_kind else None
    t1 = time.time()
    print(f"[watercolor] pipeline ready in {t1 - t0:.2f}s")

    # ✅ 추론 구간 로그
    print("[watercolor] inference start")
    use_autocast = (DEVICE == "cuda")
    if use_autocast:
        with torch.inference_mode(), torch.autocast("cuda"):
            result = pipe(
                prompt=p["prompt"],
                image=init_img,
                negative_prompt=p["negative"],
                guidance_scale=p["cfg"],
                num_inference_steps=p["steps"],
                strength=p["strength"],
                control_image=control_image if pipe.controlnet else None,
                generator=gen
            ).images[0]
    else:
        with torch.inference_mode():
            result = pipe(
                prompt=p["prompt"],
                image=init_img,
                negative_prompt=p["negative"],
                guidance_scale=p["cfg"],
                num_inference_steps=p["steps"],
                strength=p["strength"],
                control_image=control_image if pipe.controlnet else None,
                generator=gen
            ).images[0]

    t2 = time.time()
    print(f"[watercolor] inference done in {t2 - t1:.2f}s (total {t2 - t0:.2f}s)")

    return result, seed