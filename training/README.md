# ─────────────────────────────────────────────────────────────────────
# File: training/README.md
# ─────────────────────────────────────────────────────────────────────

# Watercolor LoRA Training

## 0) 의존성 설치
```bash
cd training
python -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
```

## 1) 데이터 준비
- 원천 이미지를 `training/raw_images/`에 넣고, 아래 실행으로 정리/증강해서 `training/dataset/images/`로 보내세요.
```bash
python prepare_dataset.py
```

## 2) 학습 실행
```bash
python train_watercolor_lora.py
```
환경변수로 하이퍼파라미터를 바꿀 수 있습니다. 예:
```bash
RES=512 BATCH=2 ACCUM=4 EPOCHS=4 LR_UNET=1e-4 LR_TXT=5e-5 python train_watercolor_lora.py
```

완료 후 산출물은 `training/export/watercolor_lora/final/`에 생성됩니다.

## 3) 샘플 확인 (img2img)
```bash
python sd_eval_img2img.py
```
- `LORA_DIR` 를 산출물 폴더로 맞추세요.
- `INPUT` 테스트 이미지를 지정하세요.

## 4) 다른 백엔드에서 사용
- 산출물 폴더(예: `training/export/watercolor_lora/final/`)를 서비스 백엔드의 `models/lora/watercolor_final/`로 복사
- 백엔드의 img2img 호출 전에 `pipe.load_attn_procs("models/lora/watercolor_final")` 호출 또는 제가 어제 드린 `lora_dir` 파라미터로 전달

## 5) 팁
- 데이터 수량: 50~100장 권장(스타일 LoRA)
- 해상도: 512~768 권장
- Apple Silicon: `export PYTORCH_ENABLE_MPS_FALLBACK=1`
- 구버전 diffusers: `dtype` 대신 `torch_dtype` 사용 (본 스크립트는 이미 적용)
