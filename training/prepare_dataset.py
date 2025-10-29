# ─────────────────────────────────────────────────────────────────────
# File: training/prepare_dataset.py  (선택: 폴더 정리/증강)
# ─────────────────────────────────────────────────────────────────────

import os, shutil, random
from PIL import Image, ImageOps, ImageEnhance

SRC = "./raw_images"       # 원천 폴더
DST = "./dataset/images"   # 학습 입력 폴더
SIZE = 768                  # 리사이즈 기준(긴 변)
AUG_FLIP = True
AUG_COLOR = True
SEED = 42

random.seed(SEED)

os.makedirs(DST, exist_ok=True)

for fn in os.listdir(SRC):
    if not fn.lower().endswith((".png",".jpg",".jpeg",".webp")):
        continue
    p = os.path.join(SRC, fn)
    try:
        im = Image.open(p).convert("RGB")
        w, h = im.size
        scale = SIZE / max(w, h)
        if scale < 1:
            im = im.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
        base_name = os.path.splitext(fn)[0]
        out = os.path.join(DST, base_name + ".jpg")
        im.save(out, quality=92)

        # 간단 증강
        if AUG_FLIP and random.random() < 0.5:
            im2 = ImageOps.mirror(im)
            im2.save(os.path.join(DST, base_name + "_flip.jpg"), quality=92)
        if AUG_COLOR and random.random() < 0.5:
            im3 = ImageEnhance.Color(im).enhance(0.9 + 0.2*random.random())
            im3.save(os.path.join(DST, base_name + "_color.jpg"), quality=92)
    except Exception as e:
        print("skip:", p, e)

print("[done] prepared ->", DST)
