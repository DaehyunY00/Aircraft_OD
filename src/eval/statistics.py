"""Paired significance tests and seed-variability statistics for variant comparisons.

- Wilcoxon signed-rank on per-class AP (classes as paired samples, seed-averaged)
  for configured variant pairs, on the full class set and the tail subset.
- Per-seed macro AP mean +/- std with a t-distribution confidence interval.

Outputs: outputs*/analysis/statistical_tests.csv, variant_summary_stats.csv and a
markdown summary statistical_tests.md.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

for _parent in Path(__file__).resolve().parents:
    if (_parent / "src").is_dir():
        sys.path.insert(0, str(_parent))
        break

import numpy as np
import pandas as pd

from src.utils.io import ensure_dir, load_config

DEFAULT_COMPARISON_PAIRS = [
    ["aug_selective_inpaint", "aug_uniform_inpaint"],
    ["aug_selective_inpaint", "basic_aug"],
]


def statistics_settings(config: dict[str, Any]) -> dict[str, Any]:
    stats_cfg = config.get("statistics", {}) or {}
    return {
        "comparison_pairs": stats_cfg.get("comparison_pairs", DEFAULT_COMPARISON_PAIRS),
        "confidence": float(stats_cfg.get("confidence", 0.95)),
    }


def _ap_column(df: pd.DataFrame) -> str:
    return "ap50_95" if "ap50_95" in df.columns else "ap50"


def paired_class_ap(
    per_class: pd.DataFrame,
    variant_a: str,
    variant_b: str,
    class_ids: set[int] | None = None,
) -> pd.DataFrame:
    """Seed-averaged per-class AP for two variants, aligned on class_id."""
    ap_col = _ap_column(per_class)
    df = per_class[per_class["experiment"].isin([variant_a, variant_b])].copy()
    if class_ids is not None:
        df = df[df["class_id"].astype(int).isin(class_ids)]
    df = df.dropna(subset=[ap_col])
    pivot = (
        df.groupby(["experiment", "class_id"], as_index=False)[ap_col]
        .mean()
        .pivot(index="class_id", columns="experiment", values=ap_col)
    )
    if variant_a not in pivot.columns or variant_b not in pivot.columns:
        return pd.DataFrame()
    return pivot[[variant_a, variant_b]].dropna()


def wilcoxon_test(paired: pd.DataFrame, variant_a: str, variant_b: str) -> dict[str, Any]:
    """Wilcoxon signed-rank on paired per-class AP. NaN p-value when degenerate."""
    from scipy.stats import wilcoxon

    n = len(paired)
    result: dict[str, Any] = {
        "variant_a": variant_a,
        "variant_b": variant_b,
        "n_classes": n,
        "mean_diff": float((paired[variant_a] - paired[variant_b]).mean()) if n else float("nan"),
        "median_diff": float((paired[variant_a] - paired[variant_b]).median()) if n else float("nan"),
        "statistic": float("nan"),
        "p_value": float("nan"),
    }
    diffs = (paired[variant_a] - paired[variant_b]).to_numpy(dtype=float) if n else np.array([])
    if n >= 2 and np.any(diffs != 0.0):
        try:
            stat = wilcoxon(paired[variant_a], paired[variant_b])
            result["statistic"] = float(stat.statistic)
            result["p_value"] = float(stat.pvalue)
        except ValueError:
            pass
    return result


def confidence_interval(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float, float, float]:
    """Return (mean, std, ci_low, ci_high) with a t-distribution CI over seeds."""
    from scipy.stats import t

    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mean = float(values.mean())
    if n == 1:
        return mean, 0.0, float("nan"), float("nan")
    std = float(values.std(ddof=1))
    half_width = float(t.ppf(0.5 + confidence / 2.0, df=n - 1) * std / math.sqrt(n))
    return mean, std, mean - half_width, mean + half_width


def run_statistical_tests(
    per_class_ap_csv: str | Path,
    class_groups_csv: str | Path,
    outputs: str | Path,
    config: dict[str, Any],
    eval_split: str | None = None,
) -> tuple[Path, Path, Path]:
    settings = statistics_settings(config)
    per_class = pd.read_csv(per_class_ap_csv)
    groups = pd.read_csv(class_groups_csv)
    if eval_split and "eval_split" in per_class.columns:
        filtered = per_class[per_class["eval_split"] == eval_split]
        if filtered.empty:
            # Falling back to the unfiltered frame would silently mix planning
            # (val) rows with final-eval rows in the same test.
            raise ValueError(
                f"per_class_ap에 eval_split={eval_split} 행이 없습니다. "
                "해당 split의 metric 수집을 먼저 실행하세요."
            )
        per_class = filtered
    ap_col = _ap_column(per_class)
    tail_ids = set(groups.loc[groups["group"] == "tail", "class_id"].astype(int))
    scopes: dict[str, set[int] | None] = {"all": None, "tail": tail_ids}
    # The weakness arm targets a different class set than the frequency-defined
    # tail, so reporting it only under 'tail' would score it on classes it never
    # augmented. Scope it to the classes its own plan selected.
    weakness_plan_csv = Path(outputs) / "analysis" / "augmentation_plan_weakness.csv"
    if weakness_plan_csv.exists():
        weak_ids = set(pd.read_csv(weakness_plan_csv)["class_id"].astype(int))
        if weak_ids:
            scopes["weak"] = weak_ids

    test_rows: list[dict[str, Any]] = []
    for variant_a, variant_b in settings["comparison_pairs"]:
        for scope, class_ids in scopes.items():
            paired = paired_class_ap(per_class, variant_a, variant_b, class_ids)
            row = wilcoxon_test(paired, variant_a, variant_b)
            row["scope"] = scope
            row["eval_split"] = eval_split or "all"
            test_rows.append(row)
    tests = pd.DataFrame(test_rows)

    summary_rows: list[dict[str, Any]] = []
    for scope, class_ids in scopes.items():
        df = per_class.dropna(subset=[ap_col])
        if class_ids is not None:
            df = df[df["class_id"].astype(int).isin(class_ids)]
        macro = df.groupby(["experiment", "seed"], as_index=False)[ap_col].mean()
        for variant, group_df in macro.groupby("experiment"):
            mean, std, ci_low, ci_high = confidence_interval(
                group_df[ap_col].to_numpy(), settings["confidence"]
            )
            summary_rows.append(
                {
                    "experiment": variant,
                    "scope": scope,
                    "eval_split": eval_split or "all",
                    "n_seeds": int(group_df["seed"].nunique()),
                    "macro_ap_mean": mean,
                    "macro_ap_std": std,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "confidence": settings["confidence"],
                }
            )
    summaries = pd.DataFrame(summary_rows)

    analysis_dir = ensure_dir(Path(outputs) / "analysis")
    tests_path = analysis_dir / "statistical_tests.csv"
    summary_path = analysis_dir / "variant_summary_stats.csv"
    md_path = analysis_dir / "statistical_tests.md"
    tests.to_csv(tests_path, index=False)
    summaries.to_csv(summary_path, index=False)
    md_path.write_text(_render_markdown(tests, summaries, eval_split), encoding="utf-8")
    print(f"[INFO] 통계 검정 저장: {tests_path}, {summary_path}, {md_path}")
    return tests_path, summary_path, md_path


def _render_markdown(tests: pd.DataFrame, summaries: pd.DataFrame, eval_split: str | None) -> str:
    lines = [f"# Statistical tests (eval_split={eval_split or 'all'})", ""]
    lines.append("## Wilcoxon signed-rank on paired per-class AP (seed-averaged)")
    lines.append("")
    if tests.empty:
        lines.append("_no comparisons available_")
    else:
        lines.append(tests.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## Macro AP over seeds: mean ± std with t-distribution CI")
    lines.append("")
    if summaries.empty:
        lines.append("_no variants available_")
    else:
        lines.append(summaries.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Wilcoxon tests and seed statistics on per-class AP.")
    parser.add_argument("--per-class", required=True, help="metrics/per_class_ap.csv")
    parser.add_argument("--groups", required=True, help="analysis/class_groups.csv")
    parser.add_argument("--outputs", required=True)
    parser.add_argument("--config", default="configs/full.yaml")
    parser.add_argument("--eval-split", default=None, choices=["train", "val", "test"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    split = args.eval_split or cfg.get("eval", {}).get("split", "test")
    run_statistical_tests(args.per_class, args.groups, args.outputs, cfg, eval_split=split)


if __name__ == "__main__":
    main()
