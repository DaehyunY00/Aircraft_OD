from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

for _parent in Path(__file__).resolve().parents:
    if (_parent / "src").is_dir():
        sys.path.insert(0, str(_parent))
        break

import yaml
from tqdm import tqdm

from src.augment.oversample_tail import oversample_from_plan
from src.utils.io import copy_or_symlink, ensure_dir, load_config
from src.utils.yolo import create_data_yaml, label_path_for_image, list_images, normalize_class_names


def read_data_yaml(data_yaml: str | Path) -> dict[str, Any]:
    data_yaml = Path(data_yaml)
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(data.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    data["_root"] = root
    return data


def split_dirs(data_yaml: str | Path, split: str) -> tuple[Path, Path]:
    data = read_data_yaml(data_yaml)
    split_value = data.get(split)
    if split_value is None and split == "val":
        split_value = data.get("valid", "images/val")
    if split_value is None and split == "test":
        split_value = data.get("val", "images/val")
    images_dir = Path(split_value)
    if not images_dir.is_absolute():
        images_dir = data["_root"] / images_dir
    labels_dir = Path(str(images_dir).replace("/images/", "/labels/"))
    if not labels_dir.exists():
        labels_dir = data["_root"] / "labels" / split
    return images_dir, labels_dir


def copy_split(
    source_images: Path,
    source_labels: Path,
    dest_root: Path,
    split: str,
    overwrite: bool = False,
) -> int:
    ensure_dir(dest_root / "images" / split)
    ensure_dir(dest_root / "labels" / split)
    count = 0
    for image_path in tqdm(list_images(source_images), desc=f"copy {dest_root.name}/{split}"):
        label_path = label_path_for_image(image_path, source_images, source_labels)
        rel = image_path.relative_to(source_images)
        copy_or_symlink(image_path, dest_root / "images" / split / rel, overwrite=overwrite)
        if label_path.exists():
            copy_or_symlink(label_path, dest_root / "labels" / split / rel.with_suffix(".txt"), overwrite=overwrite)
        count += 1
    return count


def add_synthetic_split(
    synthetic_root: Path,
    dest_root: Path,
    overwrite: bool = False,
) -> int:
    images_dir = synthetic_root / "images" / "train"
    labels_dir = synthetic_root / "labels" / "train"
    if not images_dir.exists():
        return 0
    count = 0
    for image_path in tqdm(list_images(images_dir), desc=f"add synthetic {synthetic_root.name}"):
        label_path = label_path_for_image(image_path, images_dir, labels_dir)
        copy_or_symlink(image_path, dest_root / "images" / "train" / image_path.name, overwrite=overwrite)
        if label_path.exists():
            copy_or_symlink(label_path, dest_root / "labels" / "train" / label_path.name, overwrite=overwrite)
        count += 1
    return count


def build_experiment_datasets(
    base_data_yaml: str | Path,
    experiments_root: str | Path,
    class_names: list[str] | None = None,
    uniform_plan: str | Path | None = None,
    selective_plan: str | Path | None = None,
    synthetic_root: str | Path | None = None,
    variants: list[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    data = read_data_yaml(base_data_yaml)
    class_names = class_names or normalize_class_names(data.get("names"), data.get("nc"))
    experiments_root = ensure_dir(experiments_root)
    synthetic_root = Path(synthetic_root) if synthetic_root else data["_root"].parent / "synthetic_inpaint"
    variants = variants or ["real_only", "basic_aug", "tail_oversampling", "uniform_tail_inpaint", "selective_tail_inpaint"]

    train_images, train_labels = split_dirs(base_data_yaml, "train")
    val_images, val_labels = split_dirs(base_data_yaml, "val")
    test_images, test_labels = split_dirs(base_data_yaml, "test")
    outputs: dict[str, Path] = {}

    for variant in variants:
        variant_root = ensure_dir(experiments_root / variant)
        copy_split(train_images, train_labels, variant_root, "train", overwrite=overwrite)
        copy_split(val_images, val_labels, variant_root, "val", overwrite=overwrite)
        copy_split(test_images, test_labels, variant_root, "test", overwrite=overwrite)

        if variant == "tail_oversampling" and selective_plan:
            created = oversample_from_plan(
                variant_root / "images" / "train",
                variant_root / "labels" / "train",
                selective_plan,
                variant_root / "images" / "train",
                variant_root / "labels" / "train",
                overwrite=overwrite,
            )
            print(f"[INFO] tail_oversampling 추가 샘플 수: {created}")
        elif variant == "uniform_tail_inpaint":
            added = add_synthetic_split(synthetic_root / "uniform", variant_root, overwrite=overwrite)
            print(f"[INFO] uniform_tail_inpaint synthetic 추가 수: {added}")
        elif variant == "selective_tail_inpaint":
            added = add_synthetic_split(synthetic_root / "selective", variant_root, overwrite=overwrite)
            print(f"[INFO] selective_tail_inpaint synthetic 추가 수: {added}")

        outputs[variant] = create_data_yaml(variant_root, class_names, variant_root / "data.yaml")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build experiment dataset variants.")
    parser.add_argument("--base-data", required=True)
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--experiments-root", default=None)
    parser.add_argument("--uniform-plan", default=None)
    parser.add_argument("--selective-plan", default=None)
    parser.add_argument("--synthetic-root", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    build_experiment_datasets(
        args.base_data,
        args.experiments_root or cfg["paths"]["experiments_data"],
        uniform_plan=args.uniform_plan,
        selective_plan=args.selective_plan,
        synthetic_root=args.synthetic_root,
        variants=cfg.get("experiments", {}).get("variants"),
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
