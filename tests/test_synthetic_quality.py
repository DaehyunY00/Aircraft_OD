from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.augment.build_experiment_datasets import (
    accepted_names_from_logs,
    add_synthetic_split,
    quality_kept_names,
)
from src.eval.synthetic_quality import (
    compute_quality_report,
    plan_quality_filter,
    quality_filter_config,
    synthetic_quality_config,
)


def _write_dummy_images(root: Path, names: list[str]) -> None:
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for name in names:
        arr = rng.integers(0, 255, size=(32, 32, 3), dtype=np.int64).astype(np.uint8)
        Image.fromarray(arr).save(root / "images" / "train" / name)
        (root / "labels" / "train" / Path(name).with_suffix(".txt").name).write_text(
            "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
        )


def _generation_log(tmp_path: Path, plan_root: Path, names: list[str]) -> Path:
    rows = []
    for i, name in enumerate(names):
        rows.append(
            {
                "source_image": str(plan_root / "images" / "train" / name),  # self as source: exists
                "output_image": str(plan_root / "images" / "train" / name),
                "class_id": i % 2,
                "class_name": f"c{i % 2}",
                "prompt": f"prompt {i}",
                "seed": 100 + i,
                "accepted": True,
                "dry_run": False,
            }
        )
    log_path = tmp_path / "generation_log_selective.csv"
    pd.DataFrame(rows).to_csv(log_path, index=False)
    return log_path


def test_quality_report_schema_and_merge(tmp_path: Path) -> None:
    plan_root = tmp_path / "synthetic" / "selective"
    names = [f"img_{i:02d}.jpg" for i in range(4)]
    _write_dummy_images(plan_root, names)
    log_path = _generation_log(tmp_path, plan_root, names)
    outputs = tmp_path / "outputs"

    fake_clip = lambda image, prompt: float(len(prompt))  # noqa: E731
    fake_lpips = lambda a, b: 0.25  # noqa: E731

    report_path = compute_quality_report(
        log_path, outputs, "selective", config={}, clip_scorer=fake_clip, lpips_fn=fake_lpips
    )
    report = pd.read_csv(report_path)
    assert {"plan", "image", "class_id", "class_name", "prompt", "clip_score", "lpips"}.issubset(report.columns)
    assert len(report) == 4
    assert (report["lpips"] == 0.25).all()

    # re-scoring the same plan replaces rows instead of duplicating them
    compute_quality_report(log_path, outputs, "selective", config={}, clip_scorer=fake_clip, lpips_fn=fake_lpips)
    assert len(pd.read_csv(report_path)) == 4


def test_quality_report_tolerates_unavailable_scorers(tmp_path: Path) -> None:
    # When a metric backend is unavailable/incompatible, its scorer yields None
    # (or is None); the report is still written with empty metric columns and no
    # exception — quality scoring never blocks the experiment.
    plan_root = tmp_path / "synthetic" / "selective"
    names = [f"img_{i:02d}.jpg" for i in range(3)]
    _write_dummy_images(plan_root, names)
    log_path = _generation_log(tmp_path, plan_root, names)
    outputs = tmp_path / "outputs"

    report_path = compute_quality_report(
        log_path, outputs, "selective", config={},
        clip_scorer=lambda image, prompt: None,  # simulates version-incompatible CLIPScore
        lpips_fn=None,                            # simulates missing LPIPS backend
    )
    report = pd.read_csv(report_path)
    assert len(report) == 3
    assert report["clip_score"].isna().all()
    assert report["lpips"].isna().all()


def test_plan_quality_filter_drops_bottom_percentile_and_builds_refill() -> None:
    report = pd.DataFrame(
        [
            {"plan": "selective", "image": f"/x/img_{i}.jpg", "class_id": i % 2, "class_name": f"c{i % 2}",
             "prompt": "p", "clip_score": float(i), "lpips": 0.1}
            for i in range(8)
        ]
    )
    filter_df, refill = plan_quality_filter(report, "selective", clip_score_percentile=50.0)

    assert int(filter_df["kept"].sum()) == 4
    assert set(filter_df.loc[filter_df["kept"], "clip_score"]) == {4.0, 5.0, 6.0, 7.0}
    # 4 dropped images across 2 classes -> refill continues numbering per class
    assert int(refill["num_synthetic_images"].sum()) == 4
    assert set(refill["start_index"]) == {4}


def test_plan_quality_filter_keeps_all_without_scores() -> None:
    report = pd.DataFrame(
        [{"plan": "selective", "image": "/x/a.jpg", "class_id": 0, "class_name": "c0",
          "prompt": "p", "clip_score": None, "lpips": None}]
    )
    filter_df, refill = plan_quality_filter(report, "selective", 50.0)
    assert bool(filter_df["kept"].all())
    assert refill.empty


def test_quality_kept_names_and_filtered_dataset_build(tmp_path: Path) -> None:
    plan_root = tmp_path / "synthetic" / "selective"
    names = [f"img_{i:02d}.jpg" for i in range(4)]
    _write_dummy_images(plan_root, names)
    refill_root = tmp_path / "synthetic" / "selective_refill"
    _write_dummy_images(refill_root, ["refill_00.jpg"])

    qf_csv = tmp_path / "quality_filter_selective.csv"
    pd.DataFrame(
        [{"image": str(plan_root / "images" / "train" / n), "kept": i >= 2} for i, n in enumerate(names)]
    ).to_csv(qf_csv, index=False)

    kept = quality_kept_names(qf_csv)
    assert kept == {"img_02.jpg", "img_03.jpg"}

    dest = tmp_path / "variant"
    added = add_synthetic_split(plan_root, dest, include_names=kept)
    added += add_synthetic_split(refill_root, dest)
    assert added == 3
    assert sorted(p.name for p in (dest / "images" / "train").glob("*.jpg")) == [
        "img_02.jpg",
        "img_03.jpg",
        "refill_00.jpg",
    ]


def test_accepted_names_from_logs_excludes_stale_leftovers(tmp_path: Path) -> None:
    # Plan dir holds 3 files but the current log only accepted 2 (e.g. the plan
    # shrank after a max_per_class change): only logged-accepted names count.
    plan_root = tmp_path / "synthetic" / "selective"
    names = ["keep_a.jpg", "keep_b.jpg", "stale_c.jpg"]
    _write_dummy_images(plan_root, names)
    log_dir = tmp_path / "outputs" / "synthetic"
    log_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"output_image": str(plan_root / "images" / "train" / "keep_a.jpg"), "accepted": True},
            {"output_image": str(plan_root / "images" / "train" / "keep_b.jpg"), "accepted": True},
            {"output_image": str(plan_root / "images" / "train" / "rejected_d.jpg"), "accepted": False},
        ]
    ).to_csv(log_dir / "generation_log_selective.csv", index=False)
    pd.DataFrame(
        [{"output_image": str(plan_root / "images" / "train" / "refill_e.jpg"), "accepted": True}]
    ).to_csv(log_dir / "generation_log_selective_refill.csv", index=False)

    include = accepted_names_from_logs(log_dir, "selective")
    assert include == {"keep_a.jpg", "keep_b.jpg", "refill_e.jpg"}

    dest = tmp_path / "variant"
    added = add_synthetic_split(plan_root, dest, include_names=include)
    assert added == 2  # stale_c excluded, rejected_d/refill_e not in this dir
    assert accepted_names_from_logs(log_dir, "uniform") is None  # no log -> None


def test_config_defaults() -> None:
    qf = quality_filter_config({})
    assert not qf["enabled"] and qf["clip_score_percentile"] == 50.0
    sq = synthetic_quality_config({})
    assert not sq["enabled"] and sq["fid_feature_dim"] == 2048
