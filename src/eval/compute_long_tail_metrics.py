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


def assign_threshold_groups(groups: pd.DataFrame, group_thresholds: dict) -> pd.DataFrame:
    """LVIS-style explicit count-threshold grouping on train instance counts.

    tail: instance_count <= tail_max_instances; medium: <= medium_max_instances;
    head: the rest. Reported alongside the bottom_percent grouping so reviewers
    can see both conventions.
    """
    tail_max = int(group_thresholds.get("tail_max_instances", 50))
    medium_max = int(group_thresholds.get("medium_max_instances", 200))
    if medium_max <= tail_max:
        raise ValueError(
            f"tail.group_thresholds: medium_max_instances({medium_max})는 "
            f"tail_max_instances({tail_max})보다 커야 합니다."
        )
    df = groups.copy()
    counts = df["instance_count"].astype(int)
    df["group_threshold"] = "head"
    df.loc[counts <= medium_max, "group_threshold"] = "medium"
    df.loc[counts <= tail_max, "group_threshold"] = "tail"
    return df


def compute_long_tail_metrics(
    raw_metrics_csv: str | Path,
    per_class_ap_csv: str | Path,
    class_groups_csv: str | Path,
    outputs: str | Path,
    tail_cfg: dict | None = None,
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

    # Gains against both references: real_only (historic lower bound) and
    # basic_aug (the primary baseline; RQ1 is the marginal gain over it).
    for ref, suffix in (("real_only", "real_only"), ("basic_aug", "basic_aug")):
        baseline = summary[summary["experiment"] == ref][["seed", "eval_split", "tail_ap", "macro_ap"]].rename(
            columns={"tail_ap": f"baseline_tail_ap_{suffix}", "macro_ap": f"baseline_macro_ap_{suffix}"}
        )
        summary = summary.merge(baseline, on=["seed", "eval_split"], how="left")
        summary[f"tail_ap_gain_vs_{suffix}"] = summary["tail_ap"] - summary[f"baseline_tail_ap_{suffix}"]
        summary[f"macro_ap_gain_vs_{suffix}"] = summary["macro_ap"] - summary[f"baseline_macro_ap_{suffix}"]
    analysis_dir = ensure_dir(outputs / "analysis")
    threshold_cfg = (tail_cfg or {}).get("group_thresholds")
    if threshold_cfg and "instance_count" in groups.columns:
        thr_groups = assign_threshold_groups(groups, threshold_cfg)
        thr_groups.to_csv(analysis_dir / "class_groups_threshold.csv", index=False)
        thr_merged = per_class.merge(thr_groups[["class_id", "group_threshold"]], on="class_id", how="left")
        thr_scores = (
            thr_merged.groupby([*keys, "group_threshold"], as_index=False)[ap_col]
            .mean()
            .pivot(index=keys, columns="group_threshold", values=ap_col)
            .reset_index()
        )
        for group in ("head", "medium", "tail"):
            if group not in thr_scores.columns:
                thr_scores[group] = pd.NA
        thr_scores = thr_scores.rename(
            columns={group: f"{group}_ap_threshold" for group in ("head", "medium", "tail")}
        )
        summary = summary.merge(
            thr_scores[[*keys, "head_ap_threshold", "medium_ap_threshold", "tail_ap_threshold"]],
            on=keys,
            how="left",
        )
    synthetic_dir = outputs / "synthetic"

    def _plan_budget(plan_csv: Path) -> int:
        return int(pd.read_csv(plan_csv)["num_synthetic_images"].sum()) if plan_csv.exists() else 0

    def _realized_inpaint(plan_name: str, include_refill: bool = False) -> int | None:
        """Actual accepted synthetic images for an inpaint plan.

        Reported instead of the plan budget so the same-budget comparison and the
        per-image efficiency metrics use the images that truly reached training.
        Refill images are only linked into the _qf variants by the dataset
        builder, so the base variant count must not include the refill log.
        """
        log_names = [f"generation_log_{plan_name}.csv"]
        if include_refill:
            log_names.append(f"generation_log_{plan_name}_refill.csv")
        total = None
        for log_name in log_names:
            log_path = synthetic_dir / log_name
            if not log_path.exists():
                continue
            log = pd.read_csv(log_path)
            if "accepted" not in log.columns:
                continue
            realized = log[~log.get("dry_run", False).astype(bool)] if "dry_run" in log.columns else log
            total = (total or 0) + int(realized["accepted"].astype(bool).sum())
        return total

    def _quality_dropped(plan_name: str) -> int:
        filter_path = synthetic_dir / f"quality_filter_{plan_name}.csv"
        if not filter_path.exists():
            return 0
        df = pd.read_csv(filter_path)
        if "kept" not in df.columns:
            return 0
        return int((~df["kept"].astype(bool)).sum())

    from src.utils.variants import KNOWN_BASE_VARIANTS, uses_synthetic_plan

    synthetic_counts = {"real_only": 0, "basic_aug": 0}
    selective_plan = analysis_dir / "augmentation_plan_selective.csv"
    # Inpaint variants: realized accepted count (falls back to plan budget).
    # Derived from the variant registry rather than hardcoded per variant —
    # aug_weakness_inpaint was added to the pipeline but missed here, so its
    # synthetic_images reported 0 while the dataset actually held 1000 images.
    for base in KNOWN_BASE_VARIANTS:
        plan_name = uses_synthetic_plan(base)
        if plan_name:
            synthetic_counts[base] = _realized_inpaint(plan_name) or _plan_budget(
                analysis_dir / f"augmentation_plan_{plan_name}.csv"
            )
    # oversample/copy_paste hit their budget deterministically (no verification).
    if selective_plan.exists():
        selective_count = _plan_budget(selective_plan)
        synthetic_counts["aug_oversample"] = selective_count
        synthetic_counts["aug_copy_paste"] = selective_count
    def _synthetic_count(experiment: str) -> int:
        if experiment in synthetic_counts:
            return synthetic_counts[experiment]
        try:
            from src.utils.variants import parse_variant, uses_synthetic_plan

            spec = parse_variant(str(experiment))
            if spec.quality_filter:
                plan = uses_synthetic_plan(spec.base)
                if plan:
                    # _qf dataset = plan accepted - quality-dropped + refill accepted
                    realized = _realized_inpaint(plan, include_refill=True)
                    if realized is not None:
                        return max(0, realized - _quality_dropped(plan))
            return synthetic_counts.get(spec.base, 0)
        except ValueError:
            return 0

    summary["synthetic_images"] = summary["experiment"].map(_synthetic_count).astype(int)
    denom = summary["synthetic_images"].astype(float).where(summary["synthetic_images"] > 0)
    # Budget-efficiency metrics measure the marginal tail gain over basic_aug.
    summary["ap_gain_per_100_synthetic_images"] = summary["tail_ap_gain_vs_basic_aug"] / denom * 100.0
    summary["ap_gain_per_generated_image"] = summary["tail_ap_gain_vs_basic_aug"] / denom
    training_hours = (summary["training_seconds"].astype(float) / 3600.0).where(summary["training_seconds"].astype(float) > 0)
    summary["ap_gain_per_training_hour"] = summary["tail_ap_gain_vs_basic_aug"] / training_hours

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
    parser.add_argument("--config", default=None, help="Config with tail.group_thresholds for threshold-group metrics")
    return parser.parse_args()


def main() -> None:
    from src.utils.io import load_config

    args = parse_args()
    tail_cfg = load_config(args.config).get("tail") if args.config else None
    compute_long_tail_metrics(args.raw, args.per_class, args.groups, args.outputs, tail_cfg=tail_cfg)


if __name__ == "__main__":
    main()
