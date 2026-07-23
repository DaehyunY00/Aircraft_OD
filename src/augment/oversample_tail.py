from __future__ import annotations

import argparse
import sys
from pathlib import Path

for _parent in Path(__file__).resolve().parents:
    if (_parent / "src").is_dir():
        sys.path.insert(0, str(_parent))
        break

import pandas as pd

from src.utils.io import copy_or_symlink, ensure_dir
from src.utils.yolo import label_path_for_image, list_images, read_yolo_labels


def collect_tail_sources(images_dir: Path, labels_dir: Path, tail_ids: set[int]) -> list[tuple[Path, Path, int]]:
    sources: list[tuple[Path, Path, int]] = []
    for image_path in list_images(images_dir):
        # In-place output dir: skip our own copies or every rerun/resume grows
        # the source pool and appends new duplicates beyond the plan budget.
        if "_oversample_c" in image_path.stem:
            continue
        label_path = label_path_for_image(image_path, images_dir, labels_dir)
        labels = read_yolo_labels(label_path)
        present = sorted({int(label["class_id"]) for label in labels if int(label["class_id"]) in tail_ids})
        for class_id in present:
            sources.append((image_path, label_path, class_id))
    return sources


def oversample_from_plan(
    images_dir: str | Path,
    labels_dir: str | Path,
    plan_csv: str | Path,
    out_images_dir: str | Path,
    out_labels_dir: str | Path,
    overwrite: bool = False,
) -> int:
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    out_images_dir = ensure_dir(out_images_dir)
    out_labels_dir = ensure_dir(out_labels_dir)
    plan = pd.read_csv(plan_csv)
    tail_ids = set(plan["class_id"].astype(int).tolist())
    budgets = {int(row.class_id): int(row.num_synthetic_images) for row in plan.itertuples()}
    sources = collect_tail_sources(images_dir, labels_dir, tail_ids)
    by_class: dict[int, list[tuple[Path, Path]]] = {}
    for image_path, label_path, class_id in sources:
        by_class.setdefault(class_id, []).append((image_path, label_path))
    created = 0
    for class_id, budget in budgets.items():
        class_sources = by_class.get(class_id, [])
        if not class_sources:
            continue
        for idx in range(budget):
            image_path, label_path = class_sources[idx % len(class_sources)]
            out_name = f"{image_path.stem}_oversample_c{class_id}_{idx:04d}{image_path.suffix.lower()}"
            copy_or_symlink(image_path, out_images_dir / out_name, overwrite=overwrite)
            copy_or_symlink(label_path, out_labels_dir / Path(out_name).with_suffix(".txt"), overwrite=overwrite)
            created += 1
    return created


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create tail oversampling copies/symlinks from an augmentation plan.")
    parser.add_argument("--images", required=True, help="Source train images directory")
    parser.add_argument("--labels", required=True, help="Source train labels directory")
    parser.add_argument("--plan", required=True, help="Augmentation plan CSV")
    parser.add_argument("--out-images", required=True, help="Output images directory")
    parser.add_argument("--out-labels", required=True, help="Output labels directory")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    created = oversample_from_plan(args.images, args.labels, args.plan, args.out_images, args.out_labels, overwrite=args.overwrite)
    print(f"[INFO] oversampling 추가 샘플 수: {created}")


if __name__ == "__main__":
    main()
