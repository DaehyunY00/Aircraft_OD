"""Repeat Factor Sampling (Gupta et al., LVIS CVPR 2019) — dataset-level.

Ultralytics does not expose sampler injection, so RFS is materialized at the
dataset level: the per-image repeat factor r(I) = max_{c in I} r(c) with
r(c) = max(1, sqrt(t / f(c))) (f(c) = fraction of train images containing c)
is turned into physical image/label duplicates. The fractional part of r(I) is
resolved stochastically with a seeded RNG (standard LVIS practice).
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path
from typing import Any

for _parent in Path(__file__).resolve().parents:
    if (_parent / "src").is_dir():
        sys.path.insert(0, str(_parent))
        break

from src.utils.io import copy_or_symlink, ensure_dir, load_config
from src.utils.yolo import label_path_for_image, list_images, read_yolo_labels


def rfs_config(config: dict[str, Any]) -> dict[str, Any]:
    rfs = config.get("rfs", {}) or {}
    return {
        # LVIS uses t=0.001 over 1203 classes; small datasets need a larger t
        # for the tail to receive any repetition at all.
        "threshold": float(rfs.get("threshold", 0.01)),
    }


def compute_class_frequencies(images_dir: Path, labels_dir: Path) -> tuple[dict[int, float], dict[Path, set[int]]]:
    """f(c) = fraction of train images containing class c, plus per-image class sets."""
    image_classes: dict[Path, set[int]] = {}
    counts: dict[int, int] = {}
    # exclude copies from a previous RFS pass so re-runs (resume) are idempotent
    images = [p for p in list_images(images_dir) if "_rfs_" not in p.stem]
    for image_path in images:
        labels = read_yolo_labels(label_path_for_image(image_path, images_dir, labels_dir))
        present = {int(label["class_id"]) for label in labels}
        image_classes[image_path] = present
        for class_id in present:
            counts[class_id] = counts.get(class_id, 0) + 1
    total = max(1, len(images))
    frequencies = {class_id: count / total for class_id, count in counts.items()}
    return frequencies, image_classes


def repeat_factor(frequency: float, threshold: float) -> float:
    """r(c) = max(1, sqrt(t / f(c)))."""
    if frequency <= 0:
        return 1.0
    return max(1.0, math.sqrt(threshold / frequency))


def compute_image_repeat_factors(
    images_dir: str | Path,
    labels_dir: str | Path,
    threshold: float,
) -> dict[Path, float]:
    """r(I) = max over classes present in the image."""
    frequencies, image_classes = compute_class_frequencies(Path(images_dir), Path(labels_dir))
    class_factors = {c: repeat_factor(f, threshold) for c, f in frequencies.items()}
    return {
        image_path: max((class_factors.get(c, 1.0) for c in classes), default=1.0)
        for image_path, classes in image_classes.items()
    }


def apply_rfs(
    images_dir: str | Path,
    labels_dir: str | Path,
    out_images_dir: str | Path,
    out_labels_dir: str | Path,
    threshold: float,
    seed: int = 42,
    overwrite: bool = False,
) -> int:
    """Materialize repeat factors as extra image/label copies. Returns copies created."""
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    out_images_dir = ensure_dir(out_images_dir)
    out_labels_dir = ensure_dir(out_labels_dir)
    factors = compute_image_repeat_factors(images_dir, labels_dir, threshold)
    rng = random.Random(seed)
    created = 0
    for image_path, factor in sorted(factors.items()):
        extra = int(math.floor(factor)) - 1
        fraction = factor - math.floor(factor)
        if rng.random() < fraction:
            extra += 1
        label_path = label_path_for_image(image_path, images_dir, labels_dir)
        for copy_idx in range(extra):
            out_name = f"{image_path.stem}_rfs_{copy_idx:02d}{image_path.suffix.lower()}"
            copy_or_symlink(image_path, out_images_dir / out_name, overwrite=overwrite)
            if label_path.exists():
                copy_or_symlink(label_path, out_labels_dir / Path(out_name).with_suffix(".txt"), overwrite=overwrite)
            created += 1
    return created


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply dataset-level Repeat Factor Sampling.")
    parser.add_argument("--images", required=True, help="Source train images directory")
    parser.add_argument("--labels", required=True, help="Source train labels directory")
    parser.add_argument("--out-images", required=True)
    parser.add_argument("--out-labels", required=True)
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    threshold = args.threshold if args.threshold is not None else rfs_config(cfg)["threshold"]
    seed = args.seed if args.seed is not None else int(cfg.get("detector", {}).get("seeds", [42])[0])
    created = apply_rfs(
        args.images, args.labels, args.out_images, args.out_labels,
        threshold=threshold, seed=seed, overwrite=args.overwrite,
    )
    print(f"[INFO] RFS 추가 복제 수: {created}")


if __name__ == "__main__":
    main()
