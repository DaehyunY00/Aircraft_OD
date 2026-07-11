from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.eval.compute_long_tail_metrics import compute_long_tail_metrics
from src.run_pipeline import _baseline_ap_for_planning, _metric_available


def test_compute_long_tail_metrics_keeps_eval_split_separate(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "metrics"
    analysis_dir = tmp_path / "analysis"
    metrics_dir.mkdir(parents=True)
    analysis_dir.mkdir(parents=True)

    raw = pd.DataFrame(
        [
            {"experiment": "real_only", "seed": 42, "eval_split": "val", "mAP50": 0.1, "mAP50_95": 0.1, "training_seconds": 10},
            {"experiment": "aug", "seed": 42, "eval_split": "val", "mAP50": 0.2, "mAP50_95": 0.2, "training_seconds": 10},
            {"experiment": "real_only", "seed": 42, "eval_split": "test", "mAP50": 0.3, "mAP50_95": 0.3, "training_seconds": 10},
            {"experiment": "aug", "seed": 42, "eval_split": "test", "mAP50": 0.4, "mAP50_95": 0.4, "training_seconds": 10},
        ]
    )
    per_class = pd.DataFrame(
        [
            {"experiment": "real_only", "seed": 42, "eval_split": "val", "class_id": 0, "class_name": "head", "ap50_95": 0.10},
            {"experiment": "real_only", "seed": 42, "eval_split": "val", "class_id": 1, "class_name": "tail", "ap50_95": 0.20},
            {"experiment": "aug", "seed": 42, "eval_split": "val", "class_id": 0, "class_name": "head", "ap50_95": 0.20},
            {"experiment": "aug", "seed": 42, "eval_split": "val", "class_id": 1, "class_name": "tail", "ap50_95": 0.30},
            {"experiment": "real_only", "seed": 42, "eval_split": "test", "class_id": 0, "class_name": "head", "ap50_95": 0.40},
            {"experiment": "real_only", "seed": 42, "eval_split": "test", "class_id": 1, "class_name": "tail", "ap50_95": 0.50},
            {"experiment": "aug", "seed": 42, "eval_split": "test", "class_id": 0, "class_name": "head", "ap50_95": 0.60},
            {"experiment": "aug", "seed": 42, "eval_split": "test", "class_id": 1, "class_name": "tail", "ap50_95": 0.80},
        ]
    )
    groups = pd.DataFrame(
        [
            {"class_id": 0, "class_name": "head", "group": "head"},
            {"class_id": 1, "class_name": "tail", "group": "tail"},
        ]
    )

    raw_path = metrics_dir / "raw_yolo_metrics.csv"
    per_class_path = metrics_dir / "per_class_ap.csv"
    groups_path = analysis_dir / "class_groups.csv"
    raw.to_csv(raw_path, index=False)
    per_class.to_csv(per_class_path, index=False)
    groups.to_csv(groups_path, index=False)

    summary_path, _ = compute_long_tail_metrics(raw_path, per_class_path, groups_path, tmp_path)
    summary = pd.read_csv(summary_path)
    aug_val = summary[(summary["experiment"] == "aug") & (summary["eval_split"] == "val")].iloc[0]
    aug_test = summary[(summary["experiment"] == "aug") & (summary["eval_split"] == "test")].iloc[0]

    assert round(float(aug_val["tail_ap_gain_vs_real_only"]), 6) == 0.1
    assert round(float(aug_test["tail_ap_gain_vs_real_only"]), 6) == 0.3


def test_baseline_ap_for_planning_uses_baseline_variant_and_split_only() -> None:
    per_class = pd.DataFrame(
        [
            {"experiment": "basic_aug", "seed": 42, "eval_split": "val", "class_id": 1, "ap50_95": 0.20},
            {"experiment": "basic_aug", "seed": 42, "eval_split": "test", "class_id": 1, "ap50_95": 0.80},
            {"experiment": "real_only", "seed": 42, "eval_split": "val", "class_id": 1, "ap50_95": 0.10},
            {"experiment": "aug_selective_inpaint", "seed": 42, "eval_split": "val", "class_id": 1, "ap50_95": 0.40},
        ]
    )

    planning_ap = _baseline_ap_for_planning(per_class, "val")

    assert planning_ap is not None
    assert planning_ap["experiment"].unique().tolist() == ["basic_aug"]
    assert planning_ap["eval_split"].unique().tolist() == ["val"]
    assert planning_ap["ap50_95"].tolist() == [0.20]


def test_metric_available_requires_raw_and_per_class_rows(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    pd.DataFrame(
        [{"experiment": "real_only", "seed": 42, "eval_split": "val", "mAP50": 0.2}]
    ).to_csv(metrics / "raw_yolo_metrics.csv", index=False)
    pd.DataFrame(
        [{"experiment": "real_only", "seed": 42, "eval_split": "val", "class_id": 1, "ap50_95": 0.3}]
    ).to_csv(metrics / "per_class_ap.csv", index=False)

    assert _metric_available(tmp_path, "real_only", 42, "val")
    assert not _metric_available(tmp_path, "real_only", 42, "test")
