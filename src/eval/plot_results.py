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
        if not per_class.empty and {"real_only", "selective_tail_inpaint"}.issubset(set(per_class["experiment"])):
            ap_col = "ap50_95" if "ap50_95" in per_class.columns else "ap50"
            pivot = per_class.pivot_table(index=["class_id", "class_name"], columns="experiment", values=ap_col, aggfunc="mean")
            pivot = pivot.reset_index()
            plt.figure(figsize=(12, 5))
            x = range(len(pivot))
            plt.bar([i - 0.2 for i in x], pivot.get("real_only", 0), width=0.4, label="real_only")
            plt.bar([i + 0.2 for i in x], pivot.get("selective_tail_inpaint", 0), width=0.4, label="selective")
            plt.xticks(list(x), pivot["class_name"].astype(str), rotation=90, fontsize=7)
            plt.ylabel(ap_col)
            plt.legend()
            plt.tight_layout()
            plt.savefig(figures / "per_class_ap_before_after_selective.png", dpi=180)
            plt.close()

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
