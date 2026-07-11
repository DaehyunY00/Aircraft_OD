from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t as t_dist
from scipy.stats import wilcoxon

from src.eval.compute_long_tail_metrics import assign_threshold_groups
from src.eval.statistics import (
    confidence_interval,
    paired_class_ap,
    run_statistical_tests,
    wilcoxon_test,
)


def _per_class_frame() -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(7)
    for class_id in range(8):
        for seed in (42, 43):
            base = 0.4 + 0.02 * class_id + rng.normal(0, 0.001)
            rows.append(
                {"experiment": "aug_uniform_inpaint", "seed": seed, "eval_split": "test",
                 "class_id": class_id, "class_name": f"c{class_id}", "ap50_95": base}
            )
            rows.append(
                {"experiment": "aug_selective_inpaint", "seed": seed, "eval_split": "test",
                 "class_id": class_id, "class_name": f"c{class_id}", "ap50_95": base + 0.05}
            )
    return pd.DataFrame(rows)


def test_wilcoxon_matches_scipy_on_known_arrays() -> None:
    per_class = _per_class_frame()
    paired = paired_class_ap(per_class, "aug_selective_inpaint", "aug_uniform_inpaint")
    result = wilcoxon_test(paired, "aug_selective_inpaint", "aug_uniform_inpaint")

    expected = wilcoxon(paired["aug_selective_inpaint"], paired["aug_uniform_inpaint"])
    assert result["n_classes"] == 8
    assert result["p_value"] == pytest.approx(float(expected.pvalue))
    assert result["statistic"] == pytest.approx(float(expected.statistic))
    assert result["mean_diff"] == pytest.approx(0.05, abs=1e-9)
    assert result["p_value"] < 0.05


def test_wilcoxon_degenerate_when_identical() -> None:
    paired = pd.DataFrame({"a": [0.1, 0.2, 0.3], "b": [0.1, 0.2, 0.3]})
    result = wilcoxon_test(paired, "a", "b")
    assert math.isnan(result["p_value"])


def test_confidence_interval_matches_manual_t_interval() -> None:
    values = np.array([0.50, 0.54, 0.58])
    mean, std, ci_low, ci_high = confidence_interval(values, confidence=0.95)
    assert mean == pytest.approx(0.54)
    assert std == pytest.approx(float(values.std(ddof=1)))
    half = t_dist.ppf(0.975, df=2) * std / math.sqrt(3)
    assert ci_low == pytest.approx(mean - half)
    assert ci_high == pytest.approx(mean + half)
    # single seed: no CI
    _, _, low, high = confidence_interval(np.array([0.5]))
    assert math.isnan(low) and math.isnan(high)


def test_run_statistical_tests_writes_outputs(tmp_path: Path) -> None:
    per_class = _per_class_frame()
    per_class_path = tmp_path / "per_class_ap.csv"
    per_class.to_csv(per_class_path, index=False)
    groups = pd.DataFrame(
        [{"class_id": i, "class_name": f"c{i}", "group": "tail" if i >= 5 else "head", "instance_count": 10 + i * 100}
         for i in range(8)]
    )
    groups_path = tmp_path / "class_groups.csv"
    groups.to_csv(groups_path, index=False)

    tests_path, summary_path, md_path = run_statistical_tests(
        per_class_path, groups_path, tmp_path, config={}, eval_split="test"
    )

    tests = pd.read_csv(tests_path)
    assert set(tests["scope"]) == {"all", "tail"}
    selective_vs_uniform = tests[
        (tests["variant_a"] == "aug_selective_inpaint")
        & (tests["variant_b"] == "aug_uniform_inpaint")
        & (tests["scope"] == "all")
    ].iloc[0]
    assert selective_vs_uniform["p_value"] < 0.05
    summary = pd.read_csv(summary_path)
    assert {"macro_ap_mean", "macro_ap_std", "ci_low", "ci_high"}.issubset(summary.columns)
    assert md_path.exists()


def test_assign_threshold_groups() -> None:
    groups = pd.DataFrame(
        [
            {"class_id": 0, "instance_count": 30},
            {"class_id": 1, "instance_count": 120},
            {"class_id": 2, "instance_count": 500},
        ]
    )
    thr = assign_threshold_groups(groups, {"tail_max_instances": 50, "medium_max_instances": 200})
    assert thr["group_threshold"].tolist() == ["tail", "medium", "head"]
    with pytest.raises(ValueError):
        assign_threshold_groups(groups, {"tail_max_instances": 200, "medium_max_instances": 200})
