"""Regression tests for the 2026-07 code-review fixes.

Covers: per-class AP row alignment (ap_class_index), dry-run rows excluded from
accepted generation-log names, oversampling idempotency across reruns, quality
filter NaN handling and dropped-name exclusion semantics, and metric-cache
invalidation when the training run changes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.augment.build_experiment_datasets import (
    accepted_names_from_logs,
    quality_dropped_names,
)
from src.augment.oversample_tail import oversample_from_plan
from src.eval.collect_yolo_metrics import per_class_ap_table
from src.eval.synthetic_quality import plan_quality_filter
from src.run_pipeline import _metric_available


def test_per_class_ap_table_aligns_rows_by_ap_class_index() -> None:
    names = ["a", "b", "c", "d"]
    # Class 1 is absent from the eval split: all_ap has 3 rows for classes 0/2/3.
    all_ap = np.array(
        [
            [0.10] * 10,
            [0.30] * 10,
            [0.40] * 10,
        ]
    )
    ap_class_index = np.array([0, 2, 3])
    df = per_class_ap_table(names, maps=None, all_ap=all_ap, ap_class_index=ap_class_index)
    by_id = df.set_index("class_id")
    assert by_id.loc[0, "ap50"] == 0.10
    # The old code would have written class 2's AP into class 1's row here.
    assert pd.isna(by_id.loc[1, "ap50"]) and pd.isna(by_id.loc[1, "ap50_95"])
    assert by_id.loc[2, "ap50"] == 0.30
    assert by_id.loc[3, "ap50"] == 0.40


def test_per_class_ap_table_legacy_maps_fallback() -> None:
    names = ["a", "b"]
    df = per_class_ap_table(names, maps=[0.5, 0.7], all_ap=None, ap_class_index=None)
    by_id = df.set_index("class_id")
    assert by_id.loc[0, "ap50_95"] == 0.5
    assert by_id.loc[1, "ap50_95"] == 0.7


def test_accepted_names_from_logs_excludes_dry_run_rows(tmp_path: Path) -> None:
    log = pd.DataFrame(
        {
            "output_image": ["/x/real_1.jpg", "/x/dry_1.jpg"],
            "accepted": [True, True],
            "dry_run": [False, True],
        }
    )
    log.to_csv(tmp_path / "generation_log_uniform.csv", index=False)
    names = accepted_names_from_logs(tmp_path, "uniform")
    assert names == {"real_1.jpg"}


def test_oversample_from_plan_is_idempotent_across_reruns(tmp_path: Path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    for stem in ("img_a", "img_b"):
        (images / f"{stem}.jpg").write_bytes(b"fake")
        (labels / f"{stem}.txt").write_text("2 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    plan = tmp_path / "plan.csv"
    pd.DataFrame([{"class_id": 2, "num_synthetic_images": 3}]).to_csv(plan, index=False)

    created_first = oversample_from_plan(images, labels, plan, images, labels)
    files_after_first = sorted(p.name for p in images.glob("*.jpg"))
    created_second = oversample_from_plan(images, labels, plan, images, labels)
    files_after_second = sorted(p.name for p in images.glob("*.jpg"))

    assert created_first == 3
    # Rerun must reproduce the same file set, not append new copies of copies.
    assert created_second == 3
    assert files_after_first == files_after_second
    assert len(files_after_second) == 2 + 3


def test_plan_quality_filter_keeps_nan_scores() -> None:
    report = pd.DataFrame(
        {
            "plan": ["uniform"] * 4,
            "image": [f"/x/im{i}.jpg" for i in range(4)],
            "class_id": [1, 1, 2, 2],
            "class_name": ["a", "a", "b", "b"],
            "clip_score": [0.1, 0.9, np.nan, 0.8],
        }
    )
    filter_df, refill = plan_quality_filter(report, "uniform", clip_score_percentile=50)
    by_image = filter_df.set_index("image")["kept"]
    assert not by_image["/x/im0.jpg"]  # below cutoff -> dropped
    assert by_image["/x/im1.jpg"]
    assert by_image["/x/im2.jpg"]  # NaN = scoring failed, must be kept
    assert by_image["/x/im3.jpg"]
    assert int(refill["num_synthetic_images"].sum()) == 1


def test_quality_dropped_names_returns_only_dropped(tmp_path: Path) -> None:
    csv = tmp_path / "quality_filter_uniform.csv"
    pd.DataFrame(
        {"image": ["/x/keep.jpg", "/x/drop.jpg"], "kept": [True, False]}
    ).to_csv(csv, index=False)
    assert quality_dropped_names(csv) == {"drop.jpg"}


def test_metric_available_requires_matching_run_dir(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    old_run = "/runs/basic_aug_seed42_old"
    raw = pd.DataFrame(
        [{"experiment": "basic_aug", "seed": 42, "eval_split": "val", "run_dir": old_run, "mAP50": 0.5}]
    )
    per_class = pd.DataFrame(
        [
            {
                "experiment": "basic_aug",
                "seed": 42,
                "eval_split": "val",
                "run_dir": old_run,
                "class_id": 0,
                "ap50": 0.5,
            }
        ]
    )
    raw.to_csv(metrics_dir / "raw_yolo_metrics.csv", index=False)
    per_class.to_csv(metrics_dir / "per_class_ap.csv", index=False)

    assert _metric_available(tmp_path, "basic_aug", 42, "val", run_dir=Path(old_run))
    # A retrained model (new run_dir) must NOT reuse the stale cached row.
    assert not _metric_available(tmp_path, "basic_aug", 42, "val", run_dir=Path("/runs/basic_aug_seed42_new"))
    # Without a run_dir the legacy behaviour (existence check) still holds.
    assert _metric_available(tmp_path, "basic_aug", 42, "val")
