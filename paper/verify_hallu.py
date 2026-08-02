"""환각 감사 결과 시각 검증 시트.

검출기가 원본(=학습 데이터)에 대해 여분 검출 0을 낸 것은 암기 효과일 수 있어,
지목된 검출이 실제 항공기 형상인지 눈으로 확인해야 한다.
초록=GT(보호 영역), 빨강=GT 밖 검출(환각 후보).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

DRIVE = Path("/Users/daehyunyoo/Library/CloudStorage/GoogleDrive-dhyoo970111@gmail.com/내 드라이브/Military_OD")
sys.path.insert(0, str(DRIVE))
from src.augment.masks import labels_to_pixel_boxes  # noqa: E402
from src.eval.audit_hallucination import CONF, CONTAINMENT, _max_containment  # noqa: E402
from src.utils.yolo import read_yolo_labels  # noqa: E402

RAW = Path(__file__).parent / "raw"
audit = pd.read_csv(DRIVE / "outputs_full/analysis/hallucination_audit.csv")
top = audit[audit.delta > 0].sort_values("delta", ascending=False).head(6)

from ultralytics import YOLO  # noqa: E402

model = YOLO(str(DRIVE / "outputs_full/runs/basic_aug_yolov8n_seed42_20260704_0154/weights/best.pt"))

fig, axes = plt.subplots(2, 6, figsize=(15, 5.4))
for col, (_, row) in enumerate(top.iterrows()):
    # 감사 실행 시 cwd가 DRIVE였으므로 생성물은 상대경로로 기록돼 있다.
    src, gen = Path(row.source_image), Path(row.output_image)
    if not gen.is_absolute():
        gen = DRIVE / gen
    label_path = next(RAW.rglob(f"{src.stem}.txt"), None)
    labels = read_yolo_labels(label_path) if label_path else []
    for r, path in enumerate([src, gen]):
        img = Image.open(path).convert("RGB")
        ax = axes[r, col]
        ax.imshow(img)
        gts = labels_to_pixel_boxes(labels, img.size, padding_ratio=0.10)
        for x0, y0, x1, y1 in gts:
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#00c000", lw=1.2))
        res = model.predict(str(path), conf=CONF, verbose=False)[0]
        n_extra = 0
        for b in (res.boxes.xyxy.cpu().numpy() if res.boxes is not None else []):
            if _max_containment(tuple(b), gts) < CONTAINMENT:
                n_extra += 1
                ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                                           fill=False, ec="#e00000", lw=1.6))
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(f"{row.plan} · {row.class_name}\nsource: {n_extra} extra", fontsize=7.5)
        else:
            ax.set_xlabel(f"generated: {n_extra} extra", fontsize=7.5)
for r, lab in enumerate(["Source (real)", "Generated"]):
    axes[r, 0].set_ylabel(lab, fontsize=9)
fig.suptitle("Green = protected ground truth   |   Red = detection outside protection (hallucination candidate)",
             fontsize=9)
fig.tight_layout()
out = DRIVE / "paper" / "figures" / "fig7_hallucination_examples.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
print("saved", out)
