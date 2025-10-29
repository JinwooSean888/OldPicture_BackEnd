
import torch
from diffusers import StableDiffusionImg2ImgPipeline, ControlNetModel
from config import SD15_REPO, CONTROLNETS

import torch

# ✅ GPU 자동 감지 (CUDA > MPS > CPU)
##if torch.cuda.is_available():
##    DEVICE = "cuda"
#elif torch.backends.mps.is_available():
 ##   DEVICE = "mps"
#else:
DEVICE = "cpu"

print(f"[init] Using device: {DEVICE}")

class WCService:
    _img2img = None
    _controlnets = {}

    @classmethod
    def img2img(cls):
        if cls._img2img is None:
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    SD15_REPO,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
).to(DEVICE)
            torch.save(pipe.unet.state_dict(), "unet_sd15.pth")

            # 예: VAE만 저장
            torch.save(pipe.vae.state_dict(), "vae_sd15.pth")

            # 예: 텍스트 인코더 저장
            torch.save(pipe.text_encoder.state_dict(), "text_encoder_sd15.pth")
            
            # Memory optimizations
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
            cls._img2img = pipe
        return cls._img2img

    @classmethod
    def controlnet(cls, kind: str):
        if kind not in CONTROLNETS:
            raise ValueError(f"Unknown controlnet kind: {kind}")
        if kind not in cls._controlnets:
            repo = CONTROLNETS[kind]
            cn = ControlNetModel.from_pretrained(
    repo,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
).to(DEVICE)
            cls._controlnets[kind] = cn
        return cls._controlnets[kind]
