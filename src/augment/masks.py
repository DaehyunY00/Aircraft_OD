from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

for _parent in Path(__file__).resolve().parents:
    if (_parent / "src").is_dir():
        sys.path.insert(0, str(_parent))
        break

from PIL import Image, ImageDraw, ImageFilter

from src.utils.io import ensure_dir
from src.utils.yolo import read_yolo_labels, yolo_to_xyxy


def expand_box(
    box: tuple[int, int, int, int],
    image_size: tuple[int, int],
    padding_ratio: float = 0.1,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    width, height = image_size
    pad_x = int(round((x2 - x1) * padding_ratio))
    pad_y = int(round((y2 - y1) * padding_ratio))
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )


def labels_to_pixel_boxes(
    labels: Sequence[dict[str, float | int]],
    image_size: tuple[int, int],
    padding_ratio: float = 0.0,
) -> list[tuple[int, int, int, int]]:
    width, height = image_size
    boxes: list[tuple[int, int, int, int]] = []
    for label in labels:
        box = yolo_to_xyxy(
            float(label["x_center"]),
            float(label["y_center"]),
            float(label["width"]),
            float(label["height"]),
            width,
            height,
        )
        if padding_ratio > 0:
            box = expand_box(box, image_size, padding_ratio)
        boxes.append(box)
    return boxes


def create_inpainting_mask(
    image_size: tuple[int, int],
    labels: Sequence[dict[str, float | int]],
    padding_ratio: float = 0.1,
    blur_radius: int = 8,
) -> tuple[Image.Image, list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    """Create a Stable Diffusion inpainting mask.

    White pixels are editable background. Black pixels are protected object boxes.
    Returns the mask, padded protected boxes, and original boxes.
    """
    padded_boxes = labels_to_pixel_boxes(labels, image_size, padding_ratio=padding_ratio)
    original_boxes = labels_to_pixel_boxes(labels, image_size, padding_ratio=0.0)
    mask = Image.new("L", image_size, 255)
    draw = ImageDraw.Draw(mask)
    for box in padded_boxes:
        draw.rectangle(box, fill=0)
    if blur_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        draw = ImageDraw.Draw(mask)
        for box in original_boxes:
            draw.rectangle(box, fill=0)
    return mask, padded_boxes, original_boxes


def mask_from_label_file(
    image_path: str | Path,
    label_path: str | Path,
    padding_ratio: float = 0.1,
    blur_radius: int = 8,
) -> tuple[Image.Image, list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    with Image.open(image_path) as image:
        image_size = image.size
    labels = read_yolo_labels(label_path)
    return create_inpainting_mask(image_size, labels, padding_ratio=padding_ratio, blur_radius=blur_radius)


def save_mask_debug(mask: Image.Image, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    mask.save(path)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a bbox-protected inpainting mask from a YOLO label file.")
    parser.add_argument("--image", required=True, help="Input image")
    parser.add_argument("--label", required=True, help="YOLO label txt")
    parser.add_argument("--out", required=True, help="Output mask PNG")
    parser.add_argument("--padding-ratio", type=float, default=0.1)
    parser.add_argument("--blur-radius", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mask, _, _ = mask_from_label_file(args.image, args.label, padding_ratio=args.padding_ratio, blur_radius=args.blur_radius)
    save_mask_debug(mask, args.out)
    print(f"[INFO] mask 저장: {args.out}")


if __name__ == "__main__":
    main()
