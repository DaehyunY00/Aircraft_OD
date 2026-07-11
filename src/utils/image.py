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
