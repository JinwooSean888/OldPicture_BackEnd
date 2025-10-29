# ─────────────────────────────────────────────────────────────────────
# File: training/sd_eval_img2img.py  (학습 결과 빠른 샘플 확인)
# ─────────────────────────────────────────────────────────────────────

import os, torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, EulerAncestralDiscreteScheduler

MODEL_ID = "runwayml/stable-diffusion-v1-5"
LORA_DIR = "./export/watercolor_lora/final"   # 학습 산출물
INPUT    = "./test_inputs/test.jpg"           # 테스트 이미지 경로
OUTPUT   = "./test_outputs/result.jpg"
PROMPT   = "soft watercolor, pastel tones, <wtr> style"
NEG      = "dark, low contrast, blurry, artifacts"
STEPS    = 20
CFG      = 6.5
STRENGTH = 0.6

if torch.cuda.is_available():
    DEVICE, DTYPE = "cuda", torch.float16
elif torch.backends.mps.is_available():
    DEVICE, DTYPE = "mps", torch.float32
else:
    DEVICE, DTYPE = "cpu", torch.float32

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.safety_checker = None
pipe.to(DEVICE)

# LoRA 로드 (폴더 또는 단일 safetensors 모두 지원)
pipe.load_attn_procs(LORA_DIR)

init = Image.open(INPUT).convert("RGB")
use_autocast = (DEVICE == "cuda")

if use_autocast:
    with torch.inference_mode(), torch.autocast("cuda"):
        out = pipe(prompt=PROMPT, image=init, negative_prompt=NEG,
                   num_inference_steps=STEPS, guidance_scale=CFG, strength=STRENGTH).images[0]
else:
    with torch.inference_mode():
        out = pipe(prompt=PROMPT, image=init, negative_prompt=NEG,
                   num_inference_steps=STEPS, guidance_scale=CFG, strength=STRENGTH).images[0]

out.save(OUTPUT, quality=92)
print("[saved]", OUTPUT)