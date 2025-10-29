# 📦 Watercolor LoRA Training Pack
# ─────────────────────────────────────────────────────────────────────
# 이 캔버스에는 "어제 만든 백엔드"와 호환되는, 바로 학습 가능한 코드가
# 파일별로 포함돼 있습니다. 아래 파일 블록들을 각 경로대로 저장하면 됩니다.
#
# 포함물
# 1) training/train_watercolor_lora.py    ← LoRA 학습 스크립트 (SD1.5 기반)
# 2) training/prepare_dataset.py          ← 데이터 정리/증강(선택)
# 3) training/sd_eval_img2img.py          ← 학습 결과 빠른 샘플 확인(img2img)
# 4) training/requirements.txt            ← 학습용 의존성
# 5) training/README.md                   ← 사용법 요약
#
# 주의
# - torch_dtype / dtype 관련 호환 이슈 때문에, 여기서는 "torch_dtype"를 사용합니다.
# - CUDA가 아니면 자동으로 float32로 전환합니다.
# - Apple Silicon(M1/M2)은 MPS로 자동 선택됩니다.


# ─────────────────────────────────────────────────────────────────────
# File: training/train_watercolor_lora.py
# ─────────────────────────────────────────────────────────────────────

import os, math, random
from dataclasses import dataclass
from typing import Optional, List
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.optimization import get_scheduler
from transformers import CLIPTokenizer

# ====== 설정 ======
MODEL_ID   = "runwayml/stable-diffusion-v1-5"  # 사전학습 모델
DATA_DIR   = os.environ.get("DATA_DIR", "./dataset/images")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./export/watercolor_lora")
TRIGGER    = os.environ.get("TRIGGER", "<wtr>")  # 스타일 토큰

# 디바이스/정밀도 자동 선택
if torch.cuda.is_available():
    DEVICE, TORCH_DTYPE = "cuda", torch.float16
elif torch.backends.mps.is_available():
    DEVICE, TORCH_DTYPE = "mps", torch.float32
else:
    DEVICE, TORCH_DTYPE = "cpu", torch.float32

# 하이퍼파라미터 (필요시 환경변수로 오버라이드)
RES       = int(os.environ.get("RES", 512))
BATCH     = int(os.environ.get("BATCH", 2))
ACCUM     = int(os.environ.get("ACCUM", 4))              # 유효 배치 = BATCH*ACCUM
EPOCHS    = int(os.environ.get("EPOCHS", 4))
LR_UNET   = float(os.environ.get("LR_UNET", 1e-4))
LR_TXT    = float(os.environ.get("LR_TXT", 5e-5))
SAVE_EVERY= int(os.environ.get("SAVE_EVERY", 400))
NUM_WORK  = int(os.environ.get("NUM_WORK", 2))
SEED      = int(os.environ.get("SEED", 42))

PROMPTS = [
    f"a {TRIGGER} style landscape, soft wash, delicate edges",
    f"portrait, {TRIGGER} style, watercolor texture, soft bleeding",
    f"city street, {TRIGGER} style illustration, hand-painted look",
    f"still life, {TRIGGER} style, subtle color wash",
]

random.seed(SEED)

torch.manual_seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)

@dataclass
class Item:
    path: str
    prompt: str

class ImgSet(Dataset):
    def __init__(self, root: str, res: int):
        assert os.path.isdir(root), f"Dataset dir not found: {root}"
        self.paths = [
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ]
        assert self.paths, f"No images found in {root}"
        self.t = transforms.Compose([
            transforms.Resize(res, transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(res),
            transforms.ToTensor(),                      # [0,1]
            transforms.Normalize([0.5],[0.5])           # -> [-1,1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        img = Image.open(p).convert("RGB")
        img = self.t(img)
        return Item(p, random.choice(PROMPTS)), img


def inject_lora(unet, rank: int = 8):
    """
    diffusers 버전에 따라 LoRA 주입 시그니처가 다른 문제를 흡수.
    - 신버전: LoRAAttnProcessor(hidden_size=..., cross_attention_dim=..., rank=...)
    - 구버전: 인자 없는 LoRAAttnProcessor()만 허용 (rank 조정 불가)
    """
    attn_procs = {}
    new_api = version.parse(diffusers.__version__) >= version.parse("0.24.0")  # 대략적 경계

    for name in unet.attn_processors.keys():
        if "mid_block" in name:
            hidden_size = unet.config.block_out_channels[-1]
        elif "up_blocks" in name:
            i = int(name.split(".")[1])
            hidden_size = list(reversed(unet.config.block_out_channels))[i]
        elif "down_blocks" in name:
            i = int(name.split(".")[1])
            hidden_size = unet.config.block_out_channels[i]
        else:
            hidden_size = unet.config.block_out_channels[0]

        cross_dim = unet.config.cross_attention_dim

        if new_api:
            # 신버전 API
            attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_dim,
                rank=rank,
            )
        else:
            # 구버전 호환: 인자 없는 생성자 시도
            try:
                attn_procs[name] = LoRAAttnProcessor()
            except TypeError:
                # 일부 아주 구버전 대비 완충
                attn_procs[name] = LoRAAttnProcessor

    unet.set_attn_processor(attn_procs)


def save(pipe: StableDiffusionPipeline, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    # UNet의 LoRA 가중치 저장 (safetensors)
    pipe.unet.save_attn_procs(out_dir)
    # 토크나이저 저장 (스타일 토큰 포함)
    pipe.tokenizer.save_pretrained(out_dir)
    # 텍스트 임베딩 전체 백업(선택)
    torch.save({
        "token": TRIGGER,
        "emb": pipe.text_encoder.get_input_embeddings().weight.detach().cpu()
    }, os.path.join(out_dir, "text_encoder_full.pt"))
    print(f"[save] {out_dir}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ds = ImgSet(DATA_DIR, RES)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=NUM_WORK, drop_last=True)

    # 사전학습 파이프라인 로드
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=TORCH_DTYPE,   # ← 구버전 호환을 위해 torch_dtype 사용
        safety_checker=None
    )
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe.to(DEVICE)

    tok: CLIPTokenizer = pipe.tokenizer

    # 새 스타일 토큰 추가 및 초기화
    if TRIGGER not in tok.get_vocab():
        tok.add_tokens(TRIGGER)
        pipe.text_encoder.resize_token_embeddings(len(tok))
        with torch.no_grad():
            base_tok = "watercolor"
            base_id = tok.convert_tokens_to_ids(base_tok) if base_tok in tok.get_vocab() else tok.eos_token_id
            new_id  = tok.convert_tokens_to_ids(TRIGGER)
            base_emb = pipe.text_encoder.get_input_embeddings().weight[base_id].clone()
            pipe.text_encoder.get_input_embeddings().weight.data[new_id] = base_emb

    # LoRA 주입 (UNet)
    inject_lora(pipe.unet, rank=8)

    # 옵티마이저/스케줄러 (UNet LoRA + 텍스트 임베딩만 학습)
    params = [
        {"params": [p for p in pipe.unet.parameters() if p.requires_grad], "lr": LR_UNET},
        {"params": [pipe.text_encoder.get_input_embeddings().weight], "lr": LR_TXT}
    ]
    opt = torch.optim.AdamW(params, weight_decay=1e-2)
    steps_total = EPOCHS * math.ceil(len(ds) / BATCH)
    sch = get_scheduler(
        name="cosine",
        optimizer=opt,
        num_warmup_steps=int(0.03 * steps_total),
        num_training_steps=steps_total
    )

    pipe.unet.train(); pipe.text_encoder.train()

    step = 0
    accum = 0
    opt.zero_grad(set_to_none=True)

    for epoch in range(EPOCHS):
        for (items, imgs) in dl:
            imgs = imgs.to(DEVICE, dtype=TORCH_DTYPE)

            # 텍스트 인코딩
            prompts = [it.prompt for it in items]
            tokens = tok(
                prompts,
                padding="max_length",
                max_length=tok.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).to(DEVICE)
            enc = pipe.text_encoder(**tokens).last_hidden_state

            # 노이즈 주입(DDPM)
            with torch.no_grad():
                noise = torch.randn_like(imgs)
                timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (imgs.size(0),), device=imgs.device).long()
                noisy = pipe.scheduler.add_noise(imgs, noise, timesteps)

            # 예측 및 손실
            pred = pipe.unet(noisy, timesteps, encoder_hidden_states=enc).sample
            loss = torch.nn.functional.mse_loss(pred.float(), noise.float()) / ACCUM
            loss.backward(); accum += 1

            if accum == ACCUM:
                torch.nn.utils.clip_grad_norm_(pipe.unet.parameters(), 1.0)
                opt.step(); sch.step(); opt.zero_grad(set_to_none=True); accum = 0

            if step % 50 == 0:
                print(f"[step {step}] loss={loss.item() * ACCUM:.4f}")
            if step and step % SAVE_EVERY == 0:
                save(pipe, os.path.join(OUTPUT_DIR, f"step{step}"))

            step += 1

    save(pipe, os.path.join(OUTPUT_DIR, "final"))


if __name__ == "__main__":
    # Mac(M1/M2) 안정성: 일부 op fallback 허용
    if DEVICE == "mps":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    main()
