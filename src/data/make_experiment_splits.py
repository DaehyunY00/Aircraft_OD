from __future__ import annotations

import argparse
import csv
from collections import Counter
import sys
from pathlib import Path
from typing import Sequence

for _parent in Path(__file__).resolve().parents:
    if (_parent / "src").is_dir():
        sys.path.insert(0, str(_parent))
        break

from sklearn.model_selection import train_test_split

from src.utils.yolo import read_yolo_labels


def primary_class(label_path: str | Path) -> int:
    labels = read_yolo_labels(label_path)
    if not labels:
        return -1
    counts = Counter(int(item["class_id"]) for item in labels)
    return counts.most_common(1)[0][0]


def stratified_partition(
    items: Sequence[tuple[Path, Path]],
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> dict[str, list[tuple[Path, Path]]]:
    """Split image/label pairs into train/val/test when no official split exists."""
    if len(items) < 10:
        return {"train": list(items), "val": list(items), "test": list(items)}
    labels = [primary_class(label_path) for _, label_path in items]
    stratify = labels if min(Counter(labels).values()) >= 2 else None
    train_items, tmp_items, _, tmp_y = train_test_split(
        list(items),
        labels,
        test_size=val_fraction + test_fraction,
        random_state=seed,
        stratify=stratify,
    )
    rel_test = test_fraction / (val_fraction + test_fraction)
    tmp_counts = Counter(tmp_y)
    tmp_stratify = tmp_y if min(tmp_counts.values()) >= 2 else None
    val_items, test_items = train_test_split(
        tmp_items,
        test_size=rel_test,
        random_state=seed,
        stratify=tmp_stratify,
    )
    return {"train": train_items, "val": val_items, "test": test_items}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a stratified train/val/test split manifest for YOLO image-label pairs.")
    parser.add_argument("--images", required=True, help="Directory containing images")
    parser.add_argument("--labels", required=True, help="Directory containing YOLO txt labels")
    parser.add_argument("--out", required=True, help="Output CSV manifest path")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    from src.utils.io import ensure_dir
    from src.utils.yolo import label_path_for_image, list_images

    args = parse_args()
    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    items = []
    for image_path in list_images(images_dir):
        label_path = label_path_for_image(image_path, images_dir, labels_dir)
        if label_path.exists():
            items.append((image_path, label_path))
    splits = stratified_partition(items, val_fraction=args.val_fraction, test_fraction=args.test_fraction, seed=args.seed)
    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "image_path", "label_path", "primary_class"])
        writer.writeheader()
        for split, split_items in splits.items():
            for image_path, label_path in split_items:
                writer.writerow(
                    {
                        "split": split,
                        "image_path": str(image_path),
                        "label_path": str(label_path),
                        "primary_class": primary_class(label_path),
                    }
                )
    print(f"[INFO] split manifest 저장: {out_path}")


if __name__ == "__main__":
    main()
