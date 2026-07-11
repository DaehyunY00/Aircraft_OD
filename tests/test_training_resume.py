from __future__ import annotations

from pathlib import Path

from src.train.train_yolo import find_reusable_run, matching_run_dirs, run_name_prefix


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def test_find_reusable_run_prefers_completed_run(tmp_path: Path) -> None:
    model = "yolov8n.pt"
    prefix = run_name_prefix("real_only", model, 42)
    interrupted = tmp_path / f"{prefix}20260701_1200"
    completed = tmp_path / f"{prefix}20260701_1300"
    _touch(interrupted / "weights" / "last.pt")
    _touch(completed / "weights" / "best.pt")
    _touch(completed / "training_meta.yaml")

    run_dir, checkpoint = find_reusable_run(tmp_path, "real_only", model, 42)

    assert run_dir == completed
    assert checkpoint is None


def test_find_reusable_run_returns_last_checkpoint_for_interrupted_run(tmp_path: Path) -> None:
    model = "yolov8n.pt"
    prefix = run_name_prefix("aug_selective_inpaint", model, 43)
    interrupted = tmp_path / f"{prefix}20260701_1400"
    _touch(interrupted / "weights" / "last.pt")

    run_dir, checkpoint = find_reusable_run(tmp_path, "aug_selective_inpaint", model, 43)

    assert run_dir == interrupted
    assert checkpoint == interrupted / "weights" / "last.pt"


def test_matching_run_dirs_ignores_other_variants_and_seeds(tmp_path: Path) -> None:
    model = "yolov8n.pt"
    wanted = tmp_path / f"{run_name_prefix('basic_aug', model, 42)}20260701_1200"
    other_seed = tmp_path / f"{run_name_prefix('basic_aug', model, 43)}20260701_1200"
    other_variant = tmp_path / f"{run_name_prefix('real_only', model, 42)}20260701_1200"
    wanted.mkdir(parents=True)
    other_seed.mkdir(parents=True)
    other_variant.mkdir(parents=True)

    matches = matching_run_dirs(tmp_path, "basic_aug", model, 42)

    assert matches == [wanted]
