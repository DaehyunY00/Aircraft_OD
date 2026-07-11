"""Tail-class Copy-Paste augmentation (Ghiasi et al., CVPR 2021) baseline.

Tail instances are cropped by their bbox and pasted onto random train images
with scale/flip jitter; labels are updated with the pasted boxes. The budget is
driven by the same augmentation plan as the selective/uniform inpaint variants
so all tail techniques are compared at an equal image budget.

LIMITATION: the dataset has no segmentation masks, so the paste is a
**rectangular bbox crop** — the pasted patch carries a rectangle of its source
background instead of an object-only cutout (Ghiasi et al. paste mask-level
cutouts). This is documented in the README and must be stated in the paper.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

for _parent in Path(__file__).resolve().parents:
    if (_parent / "src").is_dir():
        sys.path.insert(0, str(_parent))
        break

import pandas as pd
from PIL import Image

from src.utils.image import load_rgb
from src.utils.io import ensure_dir, load_config
from src.utils.yolo import (
    label_path_for_image,
    list_images,
    read_yolo_labels,
    write_yolo_labels,
    xyxy_to_yolo,
    yolo_to_xyxy,
)


def copy_paste_config(config: dict[str, Any]) -> dict[str, Any]:
    cp = config.get("copy_paste", {}) or {}
    return {
        "scale_jitter": [float(v) for v in cp.get("scale_jitter", [0.5, 1.5])],
        "horizontal_flip_prob": float(cp.get("horizontal_flip_prob", 0.5)),
        "max_paste_attempts": int(cp.get("max_paste_attempts", 10)),
        "max_overlap_iou": float(cp.get("max_overlap_iou", 0.3)),
    }


def collect_instance_pool(
    images_dir: Path,
    labels_dir: Path,
    class_ids: set[int],
) -> dict[int, list[tuple[Path, tuple[int, int, int, int]]]]:
    """Per-class pool of (source image, pixel bbox) tail instances."""
    pool: dict[int, list[tuple[Path, tuple[int, int, int, int]]]] = {c: [] for c in class_ids}
    for image_path in list_images(images_dir):
        if image_path.stem.startswith("copypaste_"):
            continue  # keep re-runs idempotent: never source from earlier outputs
        labels = read_yolo_labels(label_path_for_image(image_path, images_dir, labels_dir))
        if not labels:
            continue
        wanted = [label for label in labels if int(label["class_id"]) in class_ids]
        if not wanted:
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        for label in wanted:
            box = yolo_to_xyxy(
                float(label["x_center"]),
                float(label["y_center"]),
                float(label["width"]),
                float(label["height"]),
                width,
                height,
            )
            if box[2] - box[0] >= 4 and box[3] - box[1] >= 4:
                pool[int(label["class_id"])].append((image_path, box))
    return pool


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(area_a + area_b - inter)


def _find_placement(
    patch_size: tuple[int, int],
    target_size: tuple[int, int],
    existing_boxes: list[tuple[int, int, int, int]],
    rng: random.Random,
    max_attempts: int,
    max_overlap_iou: float,
) -> tuple[int, int, int, int] | None:
    """Random fully-inside placement, preferring low IoU with existing boxes."""
    patch_w, patch_h = patch_size
    width, height = target_size
    if patch_w >= width or patch_h >= height:
        return None
    best: tuple[float, tuple[int, int, int, int]] | None = None
    for _ in range(max(1, max_attempts)):
        x1 = rng.randint(0, width - patch_w - 1)
        y1 = rng.randint(0, height - patch_h - 1)
        box = (x1, y1, x1 + patch_w, y1 + patch_h)
        worst = max((_iou(box, other) for other in existing_boxes), default=0.0)
        if worst <= max_overlap_iou:
            return box
        if best is None or worst < best[0]:
            best = (worst, box)
    return best[1] if best is not None else None


def copy_paste_from_plan(
    images_dir: str | Path,
    labels_dir: str | Path,
    plan_csv: str | Path,
    out_images_dir: str | Path,
    out_labels_dir: str | Path,
    config: dict[str, Any] | None = None,
    seed: int = 42,
    overwrite: bool = False,
) -> int:
    """Create plan-budgeted copy-paste images. Returns the number created."""
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    out_images_dir = ensure_dir(out_images_dir)
    out_labels_dir = ensure_dir(out_labels_dir)
    cp_cfg = copy_paste_config(config or {})
    plan = pd.read_csv(plan_csv)
    class_ids = set(plan["class_id"].astype(int))
    pool = collect_instance_pool(images_dir, labels_dir, class_ids)
    # freeze the target list before we start writing new files into the dir
    targets = [p for p in list_images(images_dir) if not p.stem.startswith("copypaste_")]
    rng = random.Random(seed)
    created = 0
    for row in plan.itertuples():
        class_id = int(row.class_id)
        budget = int(row.num_synthetic_images)
        instances = pool.get(class_id, [])
        if budget <= 0 or not instances or not targets:
            continue
        for idx in range(budget):
            out_name = f"copypaste_c{class_id}_{idx:04d}.jpg"
            out_image_path = out_images_dir / out_name
            out_label_path = out_labels_dir / Path(out_name).with_suffix(".txt")
            if out_image_path.exists() and out_label_path.exists() and not overwrite:
                created += 1
                continue
            source_path, box = instances[idx % len(instances)]
            target_path = targets[rng.randrange(len(targets))]
            target = load_rgb(target_path)
            target_labels = read_yolo_labels(label_path_for_image(target_path, images_dir, labels_dir))
            existing_boxes = [
                yolo_to_xyxy(
                    float(l["x_center"]), float(l["y_center"]), float(l["width"]), float(l["height"]),
                    target.width, target.height,
                )
                for l in target_labels
            ]
            patch = load_rgb(source_path).crop(box)
            scale = rng.uniform(*cp_cfg["scale_jitter"])
            new_w = max(4, int(patch.width * scale))
            new_h = max(4, int(patch.height * scale))
            # keep the patch pasteable on small targets
            fit = min(1.0, (target.width - 2) / new_w, (target.height - 2) / new_h)
            if fit < 1.0:
                new_w = max(4, int(new_w * fit))
                new_h = max(4, int(new_h * fit))
            patch = patch.resize((new_w, new_h), Image.Resampling.LANCZOS)
            if rng.random() < cp_cfg["horizontal_flip_prob"]:
                patch = patch.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            placement = _find_placement(
                (patch.width, patch.height),
                target.size,
                existing_boxes,
                rng,
                cp_cfg["max_paste_attempts"],
                cp_cfg["max_overlap_iou"],
            )
            if placement is None:
                continue
            pasted = target.copy()
            pasted.paste(patch, (placement[0], placement[1]))
            new_labels = list(target_labels)
            x_c, y_c, w, h = xyxy_to_yolo(*placement, target.width, target.height)
            new_labels.append(
                {"class_id": class_id, "x_center": x_c, "y_center": y_c, "width": w, "height": h}
            )
            pasted.save(out_image_path, quality=95)
            write_yolo_labels(new_labels, out_label_path)
            created += 1
    return created


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create tail copy-paste images from an augmentation plan.")
    parser.add_argument("--images", required=True, help="Source train images directory")
    parser.add_argument("--labels", required=True, help="Source train labels directory")
    parser.add_argument("--plan", required=True, help="Augmentation plan CSV")
    parser.add_argument("--out-images", required=True)
    parser.add_argument("--out-labels", required=True)
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else int(cfg.get("detector", {}).get("seeds", [42])[0])
    created = copy_paste_from_plan(
        args.images, args.labels, args.plan, args.out_images, args.out_labels,
        config=cfg, seed=seed, overwrite=args.overwrite,
    )
    print(f"[INFO] copy-paste 추가 샘플 수: {created}")


if __name__ == "__main__":
    main()
