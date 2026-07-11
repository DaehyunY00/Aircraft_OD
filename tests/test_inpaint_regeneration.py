from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import src.augment.inpaint_background as ib
from tests.test_pipeline_smoke_components import _write_tiny_yolo_dataset


def _config() -> dict:
    return {
        "detector": {"seeds": [42]},
        "diffusion": {
            "mask_padding_ratio": 0.08,
            "mask_blur_radius": 2,
            "num_inference_steps": 1,
            "bbox_diff_threshold": 18.0,
            "prompts": ["sky"],
            "negative_prompt": "bad",
        },
        "verification": {
            "min_background_change": 10.0,
            "max_bbox_protected_change": 5.0,
            "min_editable_background_ratio": 0.05,
            "max_retries_per_image": 1,
            "max_failure_rate": 1.0,
        },
    }


def _write_plan(path: Path) -> Path:
    pd.DataFrame(
        [{"class_id": 2, "class_name": "tail_jet", "num_synthetic_images": 2}]
    ).to_csv(path, index=False)
    return path


def _fake_background_inpaint(pipe, device, image, mask, prompt, negative_prompt, diffusion_cfg, seed):
    """Change only the editable (white) mask region, like a real inpainting model."""
    arr = np.asarray(image.convert("RGB"), dtype=np.int16)
    editable = np.asarray(mask.convert("L"), dtype=np.uint8) > 127
    arr[editable] = np.clip(arr[editable] + 40, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def _noop_inpaint(pipe, device, image, mask, prompt, negative_prompt, diffusion_cfg, seed):
    return image.copy()


@pytest.fixture()
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    raw = tmp_path / "raw"
    _write_tiny_yolo_dataset(raw)
    monkeypatch.setattr(ib, "load_inpaint_pipeline", lambda cfg: (None, "cpu"))
    return {
        "data_yaml": raw / "data.yaml",
        "plan": _write_plan(tmp_path / "plan.csv"),
        "synthetic": tmp_path / "synthetic",
        "outputs": tmp_path / "outputs",
        "config": _config(),
    }


def _log(env: dict, plan_name: str = "uniform") -> pd.DataFrame:
    return pd.read_csv(env["outputs"] / "synthetic" / f"generation_log_{plan_name}.csv")


def test_dry_run_writes_marker_and_is_flagged(env: dict) -> None:
    ib.generate_from_plan(
        env["data_yaml"], env["plan"], env["synthetic"], env["outputs"], env["config"],
        plan_name="uniform", dry_run=True,
    )
    marker = env["synthetic"] / "uniform" / ib.DRY_RUN_MARKER_NAME
    assert marker.exists()
    log = _log(env)
    assert bool(log["dry_run"].all())
    assert not bool(log["verification_passed"].any())
    assert (log["reject_reason"] == "dry_run_copy").all()


def test_real_run_after_dry_run_regenerates_everything(env: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    ib.generate_from_plan(
        env["data_yaml"], env["plan"], env["synthetic"], env["outputs"], env["config"],
        plan_name="uniform", dry_run=True,
    )
    monkeypatch.setattr(ib, "_run_inpaint", _fake_background_inpaint)
    ib.generate_from_plan(
        env["data_yaml"], env["plan"], env["synthetic"], env["outputs"], env["config"],
        plan_name="uniform", dry_run=False,
    )
    log = _log(env)
    assert not bool(log["dry_run"].any())
    assert bool(log["accepted"].all())
    assert bool(log["verification_passed"].all())
    assert not (log["reject_reason"] == "already_exists_verified").any()
    assert (log["background_pixel_diff"] >= 10.0).all()
    assert not (env["synthetic"] / "uniform" / ib.DRY_RUN_MARKER_NAME).exists()


def test_resume_verifies_existing_files(env: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ib, "_run_inpaint", _fake_background_inpaint)
    ib.generate_from_plan(
        env["data_yaml"], env["plan"], env["synthetic"], env["outputs"], env["config"],
        plan_name="uniform", dry_run=False,
    )
    ib.generate_from_plan(
        env["data_yaml"], env["plan"], env["synthetic"], env["outputs"], env["config"],
        plan_name="uniform", dry_run=False,
    )
    log = _log(env)
    assert (log["reject_reason"] == "already_exists_verified").all()
    assert bool(log["verification_passed"].all())


def test_stale_original_copies_are_not_reused(env: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate a stale state where synthetic files are byte copies of the
    # originals but no dry-run marker exists (e.g. produced by the old code).
    monkeypatch.setattr(ib, "_run_inpaint", _noop_inpaint)
    ib.generate_from_plan(
        env["data_yaml"], env["plan"], env["synthetic"], env["outputs"], env["config"],
        plan_name="uniform", dry_run=False,
    )
    log = _log(env)
    assert not bool(log["accepted"].any())
    assert log["reject_reason"].str.startswith("verification_failed").all()
    assert list((env["synthetic"] / "uniform" / "images" / "train").glob("*.jpg")) == []
    assert len(list((env["synthetic"] / "uniform" / "rejected").glob("*.jpg"))) > 0


def test_noop_generation_is_rejected_and_removed_from_train(env: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ib, "_run_inpaint", _fake_background_inpaint)
    ib.generate_from_plan(
        env["data_yaml"], env["plan"], env["synthetic"], env["outputs"], env["config"],
        plan_name="uniform", dry_run=False,
    )
    assert len(list((env["synthetic"] / "uniform" / "images" / "train").glob("*.jpg"))) == 2
    # Regenerate with force: the no-op generator must fail verification and
    # previously accepted files must be removed from the train split.
    monkeypatch.setattr(ib, "_run_inpaint", _noop_inpaint)
    ib.generate_from_plan(
        env["data_yaml"], env["plan"], env["synthetic"], env["outputs"], env["config"],
        plan_name="uniform", dry_run=False, force=True,
    )
    log = _log(env)
    assert not bool(log["accepted"].any())
    assert list((env["synthetic"] / "uniform" / "images" / "train").glob("*.jpg")) == []
