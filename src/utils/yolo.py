from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .io import ensure_dir, save_yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def yolo_to_xyxy(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
    clip: bool = True,
) -> tuple[int, int, int, int]:
    """Convert normalized YOLO xywh to integer pixel xyxy."""
    x1 = (x_center - width / 2.0) * image_width
    y1 = (y_center - height / 2.0) * image_height
    x2 = (x_center + width / 2.0) * image_width
    y2 = (y_center + height / 2.0) * image_height
    if clip:
        x1 = max(0, min(image_width, x1))
        y1 = max(0, min(image_height, y1))
        x2 = max(0, min(image_width, x2))
        y2 = max(0, min(image_height, y2))
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def xyxy_to_yolo(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_width: int,
    image_height: int,
    clip: bool = True,
) -> tuple[float, float, float, float]:
    """Convert pixel xyxy to normalized YOLO xywh."""
    if clip:
        x1 = max(0.0, min(float(image_width), x1))
        x2 = max(0.0, min(float(image_width), x2))
        y1 = max(0.0, min(float(image_height), y1))
        y2 = max(0.0, min(float(image_height), y2))
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    x_center = x1 + width / 2.0
    y_center = y1 + height / 2.0
    return (
        x_center / image_width,
        y_center / image_height,
        width / image_width,
        height / image_height,
    )


def read_yolo_labels(path: str | Path) -> list[dict[str, float | int]]:
    path = Path(path)
    if not path.exists():
        return []
    labels: list[dict[str, float | int]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"Invalid YOLO label line {line_no} in {path}: {line}")
        class_id = int(float(parts[0]))
        coords = [float(v) for v in parts[1:5]]
        if any(v < -1e-6 or v > 1.0 + 1e-6 for v in coords):
            raise ValueError(f"YOLO coordinates must be normalized in {path}:{line_no}: {line}")
        labels.append(
            {
                "class_id": class_id,
                "x_center": coords[0],
                "y_center": coords[1],
                "width": coords[2],
                "height": coords[3],
            }
        )
    return labels


def write_yolo_labels(labels: Iterable[dict[str, float | int]], path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    lines = []
    for item in labels:
        lines.append(
            f"{int(item['class_id'])} "
            f"{float(item['x_center']):.6f} {float(item['y_center']):.6f} "
            f"{float(item['width']):.6f} {float(item['height']):.6f}"
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def label_path_for_image(image_path: str | Path, images_root: str | Path, labels_root: str | Path) -> Path:
    image_path = Path(image_path)
    rel = image_path.relative_to(images_root)
    return Path(labels_root) / rel.with_suffix(".txt")


def normalize_class_names(names: object, nc: int | None = None) -> list[str]:
    if isinstance(names, dict):
        items = sorted(((int(k), str(v)) for k, v in names.items()), key=lambda x: x[0])
        return [name for _, name in items]
    if isinstance(names, list):
        return [str(v) for v in names]
    if nc is None:
        return []
    return [f"class_{idx}" for idx in range(nc)]


def create_data_yaml(dataset_root: str | Path, class_names: list[str], yaml_path: str | Path | None = None) -> Path:
    dataset_root = Path(dataset_root)
    yaml_path = Path(yaml_path) if yaml_path else dataset_root / "data.yaml"
    data = {
        "path": str(dataset_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test" if (dataset_root / "images/test").exists() else "images/val",
        "nc": len(class_names),
        "names": class_names,
    }
    return save_yaml(data, yaml_path)
