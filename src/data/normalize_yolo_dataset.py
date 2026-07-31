from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

for _parent in Path(__file__).resolve().parents:
    if (_parent / "src").is_dir():
        sys.path.insert(0, str(_parent))
        break

from src.data.inspect_dataset import inspect_dataset
from src.data.make_experiment_splits import stratified_partition
from src.utils.io import copy_or_symlink, ensure_dir
from src.utils.yolo import create_data_yaml, label_path_for_image, list_images, read_yolo_labels, write_yolo_labels


def _image_label_pairs(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for image_path in list_images(images_dir):
        try:
            label_path = label_path_for_image(image_path, images_dir, labels_dir)
        except ValueError:
            label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            pairs.append((image_path, label_path))
    return pairs


def _limit_pairs(
    pairs: list[tuple[Path, Path]],
    max_images: int | None,
    max_classes: int | None,
) -> list[tuple[Path, Path]]:
    filtered: list[tuple[Path, Path]] = []
    for image_path, label_path in pairs:
        if max_classes is not None:
            labels = [label for label in read_yolo_labels(label_path) if int(label["class_id"]) < max_classes]
            if not labels:
                continue
        filtered.append((image_path, label_path))
        if max_images is not None and len(filtered) >= max_images:
            break
    return filtered


def _materialize_split(
    split: str,
    pairs: list[tuple[Path, Path]],
    out_root: Path,
    source_images_root: Path,
    source_labels_root: Path,
    max_classes: int | None,
    overwrite: bool,
) -> int:
    ensure_dir(out_root / "images" / split)
    ensure_dir(out_root / "labels" / split)
    count = 0
    for image_path, label_path in tqdm(pairs, desc=f"normalize {split}"):
        try:
            rel = image_path.relative_to(source_images_root)
        except ValueError:
            rel = Path(image_path.name)
        out_image = out_root / "images" / split / rel
        out_label = out_root / "labels" / split / rel.with_suffix(".txt")
        copy_or_symlink(image_path, out_image, overwrite=overwrite)
        labels = read_yolo_labels(label_path)
        if max_classes is not None:
            labels = [label for label in labels if int(label["class_id"]) < max_classes]
            write_yolo_labels(labels, out_label)
        else:
            copy_or_symlink(label_path, out_label, overwrite=overwrite)
        count += 1
    return count


def normalize_dataset(
    raw_root: str | Path,
    out_root: str | Path,
    max_images_per_split: int | None = None,
    max_classes: int | None = None,
    overwrite: bool = False,
    seed: int = 42,
) -> Path:
    raw_root = Path(raw_root)
    out_root = ensure_dir(out_root)
    inspection = inspect_dataset(raw_root)
    if not inspection["splits"]:
        raise FileNotFoundError(f"YOLO image/label split을 찾지 못했습니다: {raw_root}")

    class_names = inspection["class_names"]
    if max_classes is not None:
        class_names = class_names[:max_classes]
    if not class_names:
        raise ValueError("클래스 이름을 추론하지 못했습니다. data.yaml 또는 YOLO label을 확인하세요.")

    split_pairs: dict[str, list[tuple[Path, Path]]] = {}
    split_roots: dict[str, tuple[Path, Path]] = {}
    for split, info in inspection["splits"].items():
        images_dir = Path(info["images"])
        labels_dir = Path(info["labels"])
        pairs = _image_label_pairs(images_dir, labels_dir)
        split_pairs[split] = _limit_pairs(pairs, max_images_per_split, max_classes)
        split_roots[split] = (images_dir, labels_dir)

    if "val" not in split_pairs and "train" in split_pairs:
        partitions = stratified_partition(split_pairs["train"], seed=seed)
        split_pairs = partitions
        split_roots = {split: split_roots["train"] for split in partitions}
    elif "test" not in split_pairs:
        if "val" in split_pairs and len(split_pairs["val"]) >= 4:
            val_pairs = split_pairs["val"]
            mid = max(1, len(val_pairs) // 2)
            split_pairs["val"] = val_pairs[:mid]
            split_pairs["test"] = val_pairs[mid:] or val_pairs[:mid]
            split_roots["test"] = split_roots["val"]
        elif "val" in split_pairs:
            split_pairs["test"] = list(split_pairs["val"])
            split_roots["test"] = split_roots["val"]
        else:
            split_pairs["val"] = list(split_pairs["train"])
            split_pairs["test"] = list(split_pairs["train"])
            split_roots["val"] = split_roots["train"]
            split_roots["test"] = split_roots["train"]

    summary: dict[str, int] = {}
    for split in ("train", "val", "test"):
        pairs = split_pairs.get(split, [])
        images_dir, labels_dir = split_roots.get(split, split_roots.get("train", (raw_root, raw_root)))
        summary[split] = _materialize_split(split, pairs, out_root, images_dir, labels_dir, max_classes, overwrite)

    data_yaml = create_data_yaml(out_root, class_names, out_root / "data.yaml")
    ensure_dir(out_root / "metadata")
    (out_root / "metadata" / "normalization_summary.json").write_text(
        json.dumps({"source": inspection, "counts": summary}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[INFO] 정규화 완료: {data_yaml}")
    return data_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize a YOLO dataset to Ultralytics layout.")
    parser.add_argument("--raw", required=True, help="Raw dataset root")
    parser.add_argument("--out", required=True, help="Normalized dataset root")
    parser.add_argument("--max-images-per-split", type=int, default=None)
    parser.add_argument("--max-classes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    normalize_dataset(
        args.raw,
        args.out,
        max_images_per_split=args.max_images_per_split,
        max_classes=args.max_classes,
        overwrite=args.overwrite,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
