from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw

from .io import ensure_dir


def load_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def is_nontrivial_image(image: Image.Image, min_std: float = 2.0) -> bool:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    if arr.size == 0:
        return False
    return float(arr.std()) >= min_std


def mean_abs_diff(a: Image.Image, b: Image.Image) -> float:
    arr_a = np.asarray(a.convert("RGB"), dtype=np.float32)
    arr_b = np.asarray(b.convert("RGB"), dtype=np.float32)
    if arr_a.shape != arr_b.shape:
        b = b.resize(a.size)
        arr_b = np.asarray(b.convert("RGB"), dtype=np.float32)
    return float(np.mean(np.abs(arr_a - arr_b)))


def boxes_mask(image_size: tuple[int, int], boxes: Sequence[tuple[int, int, int, int]]) -> np.ndarray:
    """Boolean (H, W) mask that is True inside any of the given pixel boxes."""
    width, height = image_size
    mask = np.zeros((height, width), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        x1 = max(0, min(width, int(x1)))
        x2 = max(0, min(width, int(x2)))
        y1 = max(0, min(height, int(y1)))
        y2 = max(0, min(height, int(y2)))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True
    return mask


def _aligned_arrays(a: Image.Image, b: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    if b.size != a.size:
        b = b.resize(a.size)
    arr_a = np.asarray(a.convert("RGB"), dtype=np.float32)
    arr_b = np.asarray(b.convert("RGB"), dtype=np.float32)
    return arr_a, arr_b


def region_mean_abs_diff(
    a: Image.Image,
    b: Image.Image,
    boxes: Sequence[tuple[int, int, int, int]],
    inside: bool,
) -> float:
    """Mean absolute pixel diff restricted to the union of boxes (inside) or its complement."""
    arr_a, arr_b = _aligned_arrays(a, b)
    region = boxes_mask(a.size, boxes)
    if not inside:
        region = ~region
    if not region.any():
        return 0.0
    return float(np.abs(arr_a - arr_b)[region].mean())


def background_mean_abs_diff(
    original: Image.Image,
    generated: Image.Image,
    protected_boxes: Sequence[tuple[int, int, int, int]],
) -> float:
    """Mean absolute pixel diff outside all protected boxes (the inpainted background)."""
    return region_mean_abs_diff(original, generated, protected_boxes, inside=False)


def bbox_interior_mean_abs_diff(
    original: Image.Image,
    generated: Image.Image,
    boxes: Sequence[tuple[int, int, int, int]],
) -> float:
    """Mean absolute pixel diff inside the union of boxes (protected-object violation monitor)."""
    return region_mean_abs_diff(original, generated, boxes, inside=True)


def editable_background_ratio(
    image_size: tuple[int, int],
    protected_boxes: Sequence[tuple[int, int, int, int]],
) -> float:
    """Fraction of pixels outside all protected boxes. 0.0 means nothing can be inpainted."""
    protected = boxes_mask(image_size, protected_boxes)
    total = protected.size
    if total == 0:
        return 0.0
    return float((~protected).sum() / total)


def max_bbox_diff(original: Image.Image, generated: Image.Image, boxes: Sequence[tuple[int, int, int, int]]) -> float:
    if not boxes:
        return 0.0
    diffs: list[float] = []
    for x1, y1, x2, y2 in boxes:
        if x2 <= x1 or y2 <= y1:
            continue
        diffs.append(mean_abs_diff(original.crop((x1, y1, x2, y2)), generated.crop((x1, y1, x2, y2))))
    return max(diffs) if diffs else 0.0


def paste_protected_regions(
    original: Image.Image,
    generated: Image.Image,
    boxes: Sequence[tuple[int, int, int, int]],
) -> Image.Image:
    output = generated.copy()
    for box in boxes:
        if box[2] > box[0] and box[3] > box[1]:
            output.paste(original.crop(box), box)
    return output


def save_contact_sheet(
    rows: Sequence[Sequence[Image.Image]],
    path: str | Path,
    cell_size: tuple[int, int] = (256, 256),
    labels: Sequence[str] | None = None,
) -> Path:
    if not rows:
        raise ValueError("No rows provided for contact sheet.")
    path = Path(path)
    ensure_dir(path.parent)
    width = cell_size[0] * max(len(row) for row in rows)
    label_h = 24 if labels else 0
    height = (cell_size[1] + label_h) * len(rows)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    for r, row in enumerate(rows):
        top = r * (cell_size[1] + label_h)
        for c, image in enumerate(row):
            resized = image.convert("RGB").resize(cell_size)
            sheet.paste(resized, (c * cell_size[0], top + label_h))
            if labels and r == 0 and c < len(labels):
                draw.text((c * cell_size[0] + 6, top + 4), labels[c], fill=(0, 0, 0))
    sheet.save(path)
    return path
