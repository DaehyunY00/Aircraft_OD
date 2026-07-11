from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.eval.verify_generation import (
    background_ssim,
    enforce_failure_rate,
    update_verification_report,
    verification_config,
    verify_pair,
)

BOXES = [(40, 40, 60, 60)]


def _image(base: int = 120) -> Image.Image:
    rng = np.random.default_rng(0)
    arr = rng.integers(base - 40, base + 40, size=(100, 100, 3), dtype=np.int64).astype(np.uint8)
    return Image.fromarray(arr)


def _cfg() -> dict:
    return verification_config(
        {"verification": {"min_background_change": 10.0, "max_bbox_protected_change": 5.0, "compute_lpips": False}}
    )


def test_original_copy_fails_verification() -> None:
    original = _image()
    passed, metrics = verify_pair(original, original.copy(), BOXES, _cfg())
    assert not passed
    assert metrics["background_pixel_diff"] == 0.0
    assert "background_unchanged" in metrics["verification_fail_reason"]
    assert metrics["background_ssim"] == pytest.approx(1.0, abs=1e-6)


def test_background_changed_image_passes() -> None:
    original = _image()
    arr = np.asarray(original).astype(np.int16)
    background = np.ones((100, 100), dtype=bool)
    background[40:60, 40:60] = False
    arr[background] = np.clip(arr[background] + 30, 0, 255)
    generated = Image.fromarray(arr.astype(np.uint8))

    passed, metrics = verify_pair(original, generated, BOXES, _cfg())

    assert passed
    assert metrics["background_pixel_diff"] >= 10.0
    assert metrics["bbox_interior_pixel_diff"] == 0.0
    assert metrics["background_ssim"] < 1.0
    assert metrics["verification_fail_reason"] == ""


def test_changed_bbox_interior_is_protection_violation() -> None:
    original = _image()
    arr = np.asarray(original).astype(np.int16)
    background = np.ones((100, 100), dtype=bool)
    background[40:60, 40:60] = False
    arr[background] = np.clip(arr[background] + 30, 0, 255)
    arr[40:60, 40:60] = np.clip(arr[40:60, 40:60] + 50, 0, 255)
    generated = Image.fromarray(arr.astype(np.uint8))

    passed, metrics = verify_pair(original, generated, BOXES, _cfg())

    assert not passed
    assert "protected_region_changed" in metrics["verification_fail_reason"]
    assert metrics["bbox_interior_pixel_diff"] > 5.0


def test_background_ssim_ignores_protected_region() -> None:
    original = _image()
    arr = np.asarray(original).copy()
    arr[40:60, 40:60] = 255  # change only the protected interior
    generated = Image.fromarray(arr)
    # windows overlapping the bbox edge still see the change, so allow a margin
    assert background_ssim(original, generated, BOXES) > 0.9


def test_enforce_failure_rate_raises_and_skips_dry_run() -> None:
    failing = pd.DataFrame(
        [
            {"accepted": False, "dry_run": False},
            {"accepted": True, "dry_run": False},
        ]
    )
    with pytest.raises(RuntimeError):
        enforce_failure_rate(failing, max_failure_rate=0.05, plan_name="uniform")

    dry = pd.DataFrame([{"accepted": False, "dry_run": True}])
    assert enforce_failure_rate(dry, max_failure_rate=0.05, plan_name="uniform") == 0.0

    ok = pd.DataFrame([{"accepted": True, "dry_run": False}] * 20 + [{"accepted": False, "dry_run": False}] * 1)
    assert enforce_failure_rate(ok, max_failure_rate=0.05, plan_name="uniform") == pytest.approx(1 / 21)


def test_update_verification_report_merges_plans(tmp_path: Path) -> None:
    rows_a = [{"output_image": "a.jpg", "accepted": True}]
    rows_b = [{"output_image": "b.jpg", "accepted": False}]
    update_verification_report(tmp_path, "uniform", rows_a)
    path = update_verification_report(tmp_path, "selective", rows_b)
    report = pd.read_csv(path)
    assert set(report["plan"]) == {"uniform", "selective"}
    # re-running one plan replaces its rows instead of duplicating them
    update_verification_report(tmp_path, "uniform", rows_a)
    report = pd.read_csv(path)
    assert len(report) == 2
