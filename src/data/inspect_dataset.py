from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

for _parent in Path(__file__).resolve().parents:
    if (_parent / "src").is_dir():
        sys.path.insert(0, str(_parent))
        break

from src.utils.io import ensure_dir, load_yaml, save_json
from src.utils.yolo import IMAGE_EXTS, list_images, normalize_class_names, read_yolo_labels

SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "valid": "val",
    "val": "val",
    "validation": "val",
    "test": "test",
    "testing": "test",
}


def normalize_split_name(name: str) -> str | None:
    return SPLIT_ALIASES.get(name.lower())


def find_yaml_files(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted([*root.rglob("*.yaml"), *root.rglob("*.yml")], key=lambda p: (p.name != "data.yaml", len(p.parts)))


def _resolve_yaml_path(yaml_path: Path, value: Any, dataset_path: Path | None) -> Path | None:
    if value is None:
        return None
    if isinstance(value, list):
        value = value[0] if value else None
    if value is None:
        return None
    value_path = Path(str(value))
    if value_path.is_absolute():
        return value_path
    if dataset_path is not None:
        candidate = dataset_path / value_path
        if candidate.exists():
            return candidate
    return yaml_path.parent / value_path


def _labels_for_images_dir(images_dir: Path) -> Path | None:
    parts = list(images_dir.parts)
    candidates: list[Path] = []
    for idx, part in enumerate(parts):
        if part.lower() == "images":
            repl = parts.copy()
            repl[idx] = "labels"
            candidates.append(Path(*repl))
    candidates.append(images_dir.parent / "labels")
    if images_dir.name.lower() in SPLIT_ALIASES:
        candidates.append(images_dir.parent.parent / "labels" / images_dir.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else None


def inspect_from_yaml(yaml_path: Path) -> dict[str, Any]:
    data = load_yaml(yaml_path)
    dataset_path = data.get("path")
    dataset_root = Path(dataset_path) if dataset_path else yaml_path.parent
    if dataset_root and not dataset_root.is_absolute():
        dataset_root = (yaml_path.parent / dataset_root).resolve()
    class_names = normalize_class_names(data.get("names"), data.get("nc"))
    splits: dict[str, dict[str, str]] = {}
    for raw_name in ("train", "val", "valid", "test"):
        split = normalize_split_name(raw_name)
        if split is None or split in splits:
            continue
        images_dir = _resolve_yaml_path(yaml_path, data.get(raw_name), dataset_root)
        if images_dir is None:
            continue
        labels_dir = _labels_for_images_dir(images_dir)
        splits[split] = {"images": str(images_dir), "labels": str(labels_dir) if labels_dir else ""}
    return {
        "root": str(yaml_path.parent),
        "yaml": str(yaml_path),
        "class_names": class_names,
        "splits": splits,
    }


def discover_splits(root: str | Path) -> dict[str, dict[str, str]]:
    root = Path(root)
    splits: dict[str, dict[str, str]] = {}
    image_dirs = sorted({p.parent for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS})
    for images_dir in image_dirs:
        split: str | None = None
        if images_dir.name.lower() in SPLIT_ALIASES and images_dir.parent.name.lower() == "images":
            split = normalize_split_name(images_dir.name)
        elif images_dir.name.lower() == "images":
            split = normalize_split_name(images_dir.parent.name)
        elif images_dir.parent.name.lower() == "images":
            split = normalize_split_name(images_dir.name)
        elif images_dir.name.lower() in SPLIT_ALIASES:
            split = normalize_split_name(images_dir.name)
        if split is None or split in splits:
            continue
        labels_dir = _labels_for_images_dir(images_dir)
        splits[split] = {"images": str(images_dir), "labels": str(labels_dir) if labels_dir else ""}
    if not splits and image_dirs:
        images_dir = image_dirs[0]
        labels_dir = _labels_for_images_dir(images_dir)
        splits["train"] = {"images": str(images_dir), "labels": str(labels_dir) if labels_dir else ""}
    return splits


def infer_class_names(splits: dict[str, dict[str, str]]) -> list[str]:
    max_class = -1
    for split_info in splits.values():
        labels_dir = Path(split_info.get("labels", ""))
        if not labels_dir.exists():
            continue
        for label_path in labels_dir.rglob("*.txt"):
            try:
                for label in read_yolo_labels(label_path):
                    max_class = max(max_class, int(label["class_id"]))
            except ValueError:
                continue
    return [f"class_{idx}" for idx in range(max_class + 1)]


def inspect_dataset(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    yaml_files = find_yaml_files(root)
    if yaml_files:
        inspection = inspect_from_yaml(yaml_files[0])
        if inspection["splits"]:
            if not inspection["class_names"]:
                inspection["class_names"] = infer_class_names(inspection["splits"])
            return inspection
    splits = discover_splits(root)
    return {
        "root": str(root),
        "yaml": str(yaml_files[0]) if yaml_files else "",
        "class_names": infer_class_names(splits),
        "splits": splits,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a YOLO-format dataset.")
    parser.add_argument("--root", required=True, help="Raw dataset root")
    parser.add_argument("--out", default=None, help="Optional JSON output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = inspect_dataset(args.root)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.out:
        ensure_dir(Path(args.out).parent)
        save_json(result, args.out)


if __name__ == "__main__":
    main()
