from __future__ import annotations

import argparse
import sys
from pathlib import Path

for _parent in Path(__file__).resolve().parents:
    if (_parent / "src").is_dir():
        sys.path.insert(0, str(_parent))
        break

import pandas as pd

from src.utils.io import ensure_dir


def compute_long_tail_metrics(
    raw_metrics_csv: str | Path,
    per_class_ap_csv: str | Path,
    class_groups_csv: str | Path,
    outputs: str | Path,
) -> tuple[Path, Path]:
    outputs = Path(outputs)
    metrics_dir = ensure_dir(outputs / "metrics")
    raw = pd.read_csv(raw_metrics_csv) if Path(raw_metrics_csv).exists() else pd.DataFrame()
    per_class = pd.read_csv(per_class_ap_csv) if Path(per_class_ap_csv).exists() else pd.DataFrame()
    groups = pd.read_csv(class_groups_csv)
    if per_class.empty:
        summary_path = metrics_dir / "summary_by_experiment.csv"
        raw.to_csv(summary_path, index=False)
        long_path = metrics_dir / "long_tail_metrics.csv"
        pd.DataFrame().to_csv(long_path, index=False)
        return summary_path, long_path

    if "eval_split" not in per_class.columns:
        per_class["eval_split"] = "test"
    if "eval_split" not in raw.columns and not raw.empty:
        raw["eval_split"] = "test"
    merged = per_class.merge(groups[["class_id", "group"]], on="class_id", how="left")
    ap_col = "ap50_95" if "ap50_95" in merged.columns else "ap50"
    keys = ["experiment", "seed", "eval_split"]
    group_scores = (
        merged.groupby([*keys, "group"], as_index=False)[ap_col]
        .mean()
        .pivot(index=keys, columns="group", values=ap_col)
        .reset_index()
    )
    for group in ("head", "medium", "tail"):
        if group not in group_scores.columns:
            group_scores[group] = pd.NA
    macro = merged.groupby(keys, as_index=False)[ap_col].mean().rename(columns={ap_col: "macro_ap"})
    summary = raw.merge(group_scores, on=keys, how="left").merge(macro, on=keys, how="left")
    summary = summary.rename(columns={"head": "head_ap", "medium": "medium_ap", "tail": "tail_ap"})
    summary["head_tail_ap_gap"] = summary["head_ap"] - summary["tail_ap"]

    baseline = summary[summary["experiment"] == "real_only"][["seed", "eval_split", "tail_ap", "macro_ap"]].rename(
        columns={"tail_ap": "baseline_tail_ap", "macro_ap": "baseline_macro_ap"}
    )
    summary = summary.merge(baseline, on=["seed", "eval_split"], how="left")
    summary["tail_ap_gain_vs_real_only"] = summary["tail_ap"] - summary["baseline_tail_ap"]
    summary["macro_ap_gain_vs_real_only"] = summary["macro_ap"] - summary["baseline_macro_ap"]
    analysis_dir = outputs / "analysis"
    synthetic_counts = {"real_only": 0, "basic_aug": 0}
    uniform_plan = analysis_dir / "augmentation_plan_uniform.csv"
    selective_plan = analysis_dir / "augmentation_plan_selective.csv"
    if uniform_plan.exists():
        synthetic_counts["uniform_tail_inpaint"] = int(pd.read_csv(uniform_plan)["num_synthetic_images"].sum())
    if selective_plan.exists():
        selective_count = int(pd.read_csv(selective_plan)["num_synthetic_images"].sum())
        synthetic_counts["selective_tail_inpaint"] = selective_count
        synthetic_counts["tail_oversampling"] = selective_count
    summary["synthetic_images"] = summary["experiment"].map(synthetic_counts).fillna(0).astype(int)
    denom = summary["synthetic_images"].astype(float).where(summary["synthetic_images"] > 0)
    summary["ap_gain_per_100_synthetic_images"] = summary["tail_ap_gain_vs_real_only"] / denom * 100.0
    summary["ap_gain_per_generated_image"] = summary["tail_ap_gain_vs_real_only"] / denom
    training_hours = (summary["training_seconds"].astype(float) / 3600.0).where(summary["training_seconds"].astype(float) > 0)
    summary["ap_gain_per_training_hour"] = summary["tail_ap_gain_vs_real_only"] / training_hours

    summary_path = metrics_dir / "summary_by_experiment.csv"
    long_path = metrics_dir / "long_tail_metrics.csv"
    summary.to_csv(summary_path, index=False)
    merged.to_csv(long_path, index=False)
    print(f"[INFO] long-tail metric 저장: {summary_path}, {long_path}")
    return summary_path, long_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute long-tail metrics from per-class AP.")
    parser.add_argument("--raw", required=True)
    parser.add_argument("--per-class", required=True)
    parser.add_argument("--groups", required=True)
    parser.add_argument("--outputs", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compute_long_tail_metrics(args.raw, args.per_class, args.groups, args.outputs)


if __name__ == "__main__":
    main()
