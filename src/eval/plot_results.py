from __future__ import annotations

import argparse
import sys
from pathlib import Path

for _parent in Path(__file__).resolve().parents:
    if (_parent / "src").is_dir():
        sys.path.insert(0, str(_parent))
        break

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.io import ensure_dir


def _bar(df: pd.DataFrame, x: str, y: str, path: Path, title: str, ylabel: str) -> None:
    if df.empty or x not in df or y not in df:
        return
    plt.figure(figsize=(9, 4.5))
    plt.bar(df[x].astype(str), df[y])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


GROUP_COLORS = {"head": "#2f6f8f", "medium": "#6a8f3a", "tail": "#b54c3f"}


def _plot_delta_vs_baseline(
    per_class: pd.DataFrame,
    groups: pd.DataFrame,
    figures: Path,
    baseline: str = "basic_aug",
) -> None:
    """RQ3 plots: per-class AP delta vs the basic_aug baseline (Gen2Det-style).

    Scatter (delta vs train instance count, colored by head/medium/tail group)
    per variant plus one violin comparing delta distributions across variants —
    shows whether tail gains come without head damage.
    """
    if per_class.empty or baseline not in set(per_class["experiment"]):
        return
    ap_col = "ap50_95" if "ap50_95" in per_class.columns else "ap50"
    class_mean = per_class.groupby(["experiment", "class_id"], as_index=False)[ap_col].mean()
    base = class_mean[class_mean["experiment"] == baseline].set_index("class_id")[ap_col]
    meta_cols = ["class_id", "group"] + (["instance_count"] if "instance_count" in groups.columns else [])
    meta = groups[meta_cols].set_index("class_id")
    deltas: dict[str, pd.DataFrame] = {}
    for variant, df in class_mean.groupby("experiment"):
        if variant in (baseline, "real_only"):
            continue
        merged = df.set_index("class_id").join(meta, how="left")
        merged["delta"] = merged[ap_col] - base
        merged = merged.dropna(subset=["delta"])
        if not merged.empty:
            deltas[variant] = merged

    for variant, merged in deltas.items():
        if "instance_count" not in merged.columns:
            continue
        plt.figure(figsize=(7, 4.5))
        for group_name, group_df in merged.groupby("group"):
            plt.scatter(
                group_df["instance_count"],
                group_df["delta"],
                label=str(group_name),
                color=GROUP_COLORS.get(str(group_name), "#777777"),
                alpha=0.8,
            )
        plt.axhline(0.0, color="black", linewidth=0.8)
        plt.xscale("log")
        plt.xlabel("Train instances (log)")
        plt.ylabel(f"Δ{ap_col} vs {baseline}")
        plt.title(f"Per-class AP delta: {variant}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures / f"delta_ap_scatter_{variant}.png", dpi=180)
        plt.close()

    if deltas:
        plt.figure(figsize=(max(6, 2 * len(deltas)), 4.5))
        names = list(deltas)
        plt.violinplot([deltas[name]["delta"].to_numpy() for name in names], showmedians=True)
        plt.axhline(0.0, color="black", linewidth=0.8)
        plt.xticks(range(1, len(names) + 1), names, rotation=20, ha="right")
        plt.ylabel(f"Δ{ap_col} vs {baseline}")
        plt.title("Per-class AP delta distribution by variant")
        plt.tight_layout()
        plt.savefig(figures / "delta_ap_violin_by_variant.png", dpi=180)
        plt.close()


def _filter_eval_split(df: pd.DataFrame, eval_split: str | None) -> pd.DataFrame:
    if eval_split and "eval_split" in df.columns:
        filtered = df[df["eval_split"] == eval_split].copy()
        if not filtered.empty:
            return filtered
    return df


def plot_results(outputs: str | Path, eval_split: str | None = None) -> None:
    outputs = Path(outputs)
    figures = ensure_dir(outputs / "figures")
    analysis = outputs / "analysis"
    metrics = outputs / "metrics"

    class_dist = analysis / "class_distribution.csv"
    if class_dist.exists():
        df = pd.read_csv(class_dist).sort_values("instance_count", ascending=False)
        plt.figure(figsize=(12, 5))
        plt.bar(df["class_name"].astype(str), df["instance_count"])
        plt.ylabel("Train instances")
        plt.xticks(rotation=90, fontsize=7)
        plt.tight_layout()
        plt.savefig(figures / "class_distribution_bar.png", dpi=180)
        plt.close()

    group_summary = analysis / "head_medium_tail_summary.csv"
    if group_summary.exists():
        df = pd.read_csv(group_summary)
        _bar(df, "group", "instances", figures / "head_medium_tail_summary.png", "Head / Medium / Tail instances", "Instances")

    summary_path = metrics / "summary_by_experiment.csv"
    if summary_path.exists():
        summary = _filter_eval_split(pd.read_csv(summary_path), eval_split)
        agg = summary.groupby("experiment", as_index=False).mean(numeric_only=True)
        _bar(agg, "experiment", "tail_ap", figures / "tail_ap_comparison.png", "Tail AP by experiment", "Tail AP")
        _bar(agg, "experiment", "macro_ap", figures / "macro_ap_comparison.png", "Macro AP by experiment", "Macro AP")
        _bar(
            agg,
            "experiment",
            "head_tail_ap_gap",
            figures / "head_tail_ap_gap_comparison.png",
            "Head-Tail AP Gap",
            "AP gap",
        )
        if "ap_gain_per_100_synthetic_images" in agg:
            _bar(
                agg.dropna(subset=["ap_gain_per_100_synthetic_images"]),
                "experiment",
                "ap_gain_per_100_synthetic_images",
                figures / "ap_gain_per_100_synthetic_images.png",
                "AP gain per 100 synthetic images",
                "Tail AP gain",
            )

    per_class_path = metrics / "per_class_ap.csv"
    if per_class_path.exists():
        per_class = _filter_eval_split(pd.read_csv(per_class_path), eval_split)
        if not per_class.empty and {"basic_aug", "aug_selective_inpaint"}.issubset(set(per_class["experiment"])):
            ap_col = "ap50_95" if "ap50_95" in per_class.columns else "ap50"
            pivot = per_class.pivot_table(index=["class_id", "class_name"], columns="experiment", values=ap_col, aggfunc="mean")
            pivot = pivot.reset_index()
            plt.figure(figsize=(12, 5))
            x = range(len(pivot))
            plt.bar([i - 0.2 for i in x], pivot.get("basic_aug", 0), width=0.4, label="basic_aug")
            plt.bar([i + 0.2 for i in x], pivot.get("aug_selective_inpaint", 0), width=0.4, label="selective inpaint")
            plt.xticks(list(x), pivot["class_name"].astype(str), rotation=90, fontsize=7)
            plt.ylabel(ap_col)
            plt.legend()
            plt.tight_layout()
            plt.savefig(figures / "per_class_ap_before_after_selective.png", dpi=180)
            plt.close()

    groups_path = analysis / "class_groups.csv"
    if per_class_path.exists() and groups_path.exists():
        per_class = _filter_eval_split(pd.read_csv(per_class_path), eval_split)
        _plot_delta_vs_baseline(per_class, pd.read_csv(groups_path), figures)

    for plan_name in ("augmentation_plan_uniform.csv", "augmentation_plan_selective.csv"):
        plan_path = analysis / plan_name
        if plan_path.exists():
            df = pd.read_csv(plan_path)
            _bar(
                df,
                "class_name",
                "num_synthetic_images",
                figures / plan_name.replace(".csv", ".png"),
                plan_name.replace("_", " ").replace(".csv", ""),
                "Synthetic images",
            )
    print(f"[INFO] figure 저장 완료: {figures}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot experiment results with matplotlib.")
    parser.add_argument("--outputs", required=True)
    parser.add_argument("--eval-split", default=None, choices=["train", "val", "test"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_results(args.outputs, eval_split=args.eval_split)


if __name__ == "__main__":
    main()
