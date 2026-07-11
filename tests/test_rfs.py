from __future__ import annotations

import math
from pathlib import Path

import pytest

from src.augment.repeat_factor_sampling import (
    apply_rfs,
    compute_class_frequencies,
    compute_image_repeat_factors,
    repeat_factor,
)
from tests.test_pipeline_smoke_components import _write_tiny_yolo_dataset


def test_repeat_factor_formula() -> None:
    assert repeat_factor(0.5, 0.5) == 1.0  # frequent class clamps to 1
    assert repeat_factor(0.125, 0.5) == pytest.approx(2.0)  # sqrt(0.5/0.125)
    assert repeat_factor(0.0, 0.5) == 1.0


def test_class_frequencies_on_tiny_dataset(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_tiny_yolo_dataset(raw)  # train: 6/3/2 single-label images (11 total)
    freqs, image_classes = compute_class_frequencies(raw / "images" / "train", raw / "labels" / "train")
    assert len(image_classes) == 11
    assert freqs[0] == pytest.approx(6 / 11)
    assert freqs[1] == pytest.approx(3 / 11)
    assert freqs[2] == pytest.approx(2 / 11)


def test_image_repeat_factors_are_max_over_classes(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_tiny_yolo_dataset(raw)
    threshold = 0.5
    factors = compute_image_repeat_factors(raw / "images" / "train", raw / "labels" / "train", threshold)
    for image_path, factor in factors.items():
        if "_c2_" in image_path.name:
            assert factor == pytest.approx(math.sqrt(0.5 / (2 / 11)))
        elif "_c0_" in image_path.name:
            assert factor == 1.0  # head class: clamped


def test_apply_rfs_creates_expected_copy_counts(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_tiny_yolo_dataset(raw)
    images = raw / "images" / "train"
    labels = raw / "labels" / "train"
    threshold = 0.5
    factors = compute_image_repeat_factors(images, labels, threshold)

    created = apply_rfs(images, labels, images, labels, threshold=threshold, seed=42)

    # deterministic floor part: every image gets at least floor(r)-1 extra copies
    min_expected = sum(int(math.floor(f)) - 1 for f in factors.values())
    max_expected = sum(int(math.floor(f)) for f in factors.values())
    assert min_expected <= created <= max_expected
    rfs_copies = list(images.glob("*_rfs_*"))
    assert len(rfs_copies) == created
    for copy_path in rfs_copies:
        assert (labels / copy_path.with_suffix(".txt").name).exists()


def test_apply_rfs_rerun_is_idempotent(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_tiny_yolo_dataset(raw)
    images = raw / "images" / "train"
    labels = raw / "labels" / "train"
    first = apply_rfs(images, labels, images, labels, threshold=0.5, seed=42)
    second = apply_rfs(images, labels, images, labels, threshold=0.5, seed=42)
    # previous _rfs_ copies are excluded from frequency computation and expansion
    assert second == first
    assert len(list(images.glob("*_rfs_*"))) == first
