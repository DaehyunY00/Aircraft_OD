from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.augment.copy_paste_tail import collect_instance_pool, copy_paste_from_plan
from src.utils.yolo import read_yolo_labels
from tests.test_pipeline_smoke_components import _write_tiny_yolo_dataset


def _setup(tmp_path: Path) -> tuple[Path, Path, Path]:
    raw = tmp_path / "raw"
    _write_tiny_yolo_dataset(raw)
    plan = tmp_path / "plan.csv"
    pd.DataFrame(
        [{"class_id": 2, "class_name": "tail_jet", "num_synthetic_images": 3}]
    ).to_csv(plan, index=False)
    return raw / "images" / "train", raw / "labels" / "train", plan


def test_collect_instance_pool_only_requested_classes(tmp_path: Path) -> None:
    images, labels, _ = _setup(tmp_path)
    pool = collect_instance_pool(images, labels, {2})
    assert set(pool) == {2}
    assert len(pool[2]) == 2  # tiny dataset has 2 train images of class 2


def test_copy_paste_creates_budgeted_images_with_updated_labels(tmp_path: Path) -> None:
    images, labels, plan = _setup(tmp_path)
    out_images = tmp_path / "out" / "images"
    out_labels = tmp_path / "out" / "labels"

    created = copy_paste_from_plan(images, labels, plan, out_images, out_labels, config={}, seed=42)

    assert created == 3
    created_images = sorted(out_images.glob("copypaste_c2_*.jpg"))
    assert len(created_images) == 3
    for image_path in created_images:
        parsed = read_yolo_labels(out_labels / image_path.with_suffix(".txt").name)
        # target image had 1 original label; one pasted class-2 label was added
        assert len(parsed) == 2
        assert int(parsed[-1]["class_id"]) == 2
        for label in parsed:
            for key in ("x_center", "y_center", "width", "height"):
                assert 0.0 <= float(label[key]) <= 1.0
        assert float(parsed[-1]["width"]) > 0 and float(parsed[-1]["height"]) > 0


def test_copy_paste_is_deterministic_for_a_seed(tmp_path: Path) -> None:
    images, labels, plan = _setup(tmp_path)
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    copy_paste_from_plan(images, labels, plan, out_a / "img", out_a / "lbl", config={}, seed=7)
    copy_paste_from_plan(images, labels, plan, out_b / "img", out_b / "lbl", config={}, seed=7)
    labels_a = sorted((out_a / "lbl").glob("*.txt"))
    labels_b = sorted((out_b / "lbl").glob("*.txt"))
    assert [p.read_text() for p in labels_a] == [p.read_text() for p in labels_b]
