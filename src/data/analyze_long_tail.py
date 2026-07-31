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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.utils.io import ensure_dir, load_config, load_yaml
from src.utils.yolo import label_path_for_image, list_images, normalize_class_names, read_yolo_labels


def _split_dirs(data_yaml: str | Path, split: str) -> tuple[Path, Path]:
    data_yaml = Path(data_yaml)
    data = load_yaml(data_yaml)
    root = Path(data.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    split_value = data.get(split)
    if split_value is None and split == "val":
        split_value = data.get("valid")
    if split_value is None:
        split_value = f"images/{split}"
    images_dir = Path(split_value)
    if not images_dir.is_absolute():
        images_dir = root / images_dir
    labels_dir = Path(str(images_dir).replace("/images/", "/labels/"))
    if not labels_dir.exists():
        labels_dir = root / "labels" / split
    return images_dir, labels_dir


def _class_names(data_yaml: str | Path) -> list[str]:
    data = load_yaml(data_yaml)
    return normalize_class_names(data.get("names"), data.get("nc"))


def collect_class_statistics(data_yaml: str | Path) -> pd.DataFrame:
    class_names = _class_names(data_yaml)
    rows = {
        class_id: {
            "class_id": class_id,
            "class_name": class_names[class_id] if class_id < len(class_names) else f"class_{class_id}",
            "instance_count": 0,
            "image_count": 0,
            "val_instance_count": 0,
            "avg_bbox_area": 0.0,
        }
        for class_id in range(len(class_names))
    }
    bbox_areas: dict[int, list[float]] = {class_id: [] for class_id in rows}

    for split in ("train", "val"):
        images_dir, labels_dir = _split_dirs(data_yaml, split)
        for image_path in list_images(images_dir):
            label_path = label_path_for_image(image_path, images_dir, labels_dir)
            labels = read_yolo_labels(label_path)
            if not labels:
                continue
            seen: set[int] = set()
            try:
                with Image.open(image_path) as img:
                    img_w, img_h = img.size
            except Exception:
                img_w, img_h = 1, 1
            for label in labels:
                class_id = int(label["class_id"])
                if class_id not in rows:
                    rows[class_id] = {
                        "class_id": class_id,
                        "class_name": f"class_{class_id}",
                        "instance_count": 0,
                        "image_count": 0,
                        "val_instance_count": 0,
                        "avg_bbox_area": 0.0,
                    }
                    bbox_areas[class_id] = []
                if split == "train":
                    rows[class_id]["instance_count"] += 1
                    bbox_areas[class_id].append(float(label["width"]) * img_w * float(label["height"]) * img_h)
                elif split == "val":
                    rows[class_id]["val_instance_count"] += 1
                seen.add(class_id)
            if split == "train":
                for class_id in seen:
                    rows[class_id]["image_count"] += 1

    for class_id, areas in bbox_areas.items():
        rows[class_id]["avg_bbox_area"] = float(np.mean(areas)) if areas else 0.0
    df = pd.DataFrame(rows.values()).sort_values("instance_count", ascending=False).reset_index(drop=True)
    return df


def collect_dataset_summary(data_yaml: str | Path, stats: pd.DataFrame) -> pd.DataFrame:
    """Create a compact dataset-level summary for reproducibility logs."""
    split_image_counts: dict[str, int] = {}
    split_box_counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        images_dir, labels_dir = _split_dirs(data_yaml, split)
        image_paths = list_images(images_dir)
        split_image_counts[split] = len(image_paths)
        box_count = 0
        for image_path in image_paths:
            label_path = label_path_for_image(image_path, images_dir, labels_dir)
            box_count += len(read_yolo_labels(label_path))
        split_box_counts[split] = box_count
    nonzero = stats.loc[stats["instance_count"] > 0, "instance_count"]
    imbalance_ratio = float(nonzero.max() / max(1, nonzero.min())) if not nonzero.empty else 0.0
    return pd.DataFrame(
        [
            {
                "num_images_train": split_image_counts.get("train", 0),
                "num_images_val": split_image_counts.get("val", 0),
                "num_images_test": split_image_counts.get("test", 0),
                "num_images_total": sum(split_image_counts.values()),
                "num_classes": int(len(stats)),
                "total_bboxes_train": int(split_box_counts.get("train", 0)),
                "total_bboxes_val": int(split_box_counts.get("val", 0)),
                "total_bboxes_test": int(split_box_counts.get("test", 0)),
                "total_bboxes": int(sum(split_box_counts.values())),
                "imbalance_ratio_train": imbalance_ratio,
            }
        ]
    )


def assign_class_groups(stats: pd.DataFrame, tail_cfg: dict[str, Any]) -> pd.DataFrame:
    df = stats.copy()
    df["group"] = "medium"
    nonzero = df[df["instance_count"] > 0].copy()
    if nonzero.empty:
        return df
    method = tail_cfg.get("method", "bottom_percent")
    if method == "count_threshold":
        max_instances = int(tail_cfg.get("max_instances", 50))
        tail_ids = set(nonzero.loc[nonzero["instance_count"] <= max_instances, "class_id"].astype(int))
        head_cut = max_instances * 3
        head_ids = set(nonzero.loc[nonzero["instance_count"] >= head_cut, "class_id"].astype(int))
    else:
        pct = float(tail_cfg.get("bottom_percent", 0.3))
        group_n = max(1, int(math.ceil(len(nonzero) * pct)))
        eligible = nonzero[nonzero["val_instance_count"] >= int(tail_cfg.get("min_val_instances", 0))]
        tail_pool = eligible if len(eligible) >= group_n else nonzero
        tail_ids = set(tail_pool.sort_values("instance_count", ascending=True).head(group_n)["class_id"].astype(int))
        head_ids = set(nonzero.sort_values("instance_count", ascending=False).head(group_n)["class_id"].astype(int))
    df.loc[df["class_id"].isin(head_ids), "group"] = "head"
    df.loc[df["class_id"].isin(tail_ids), "group"] = "tail"
    return df


def save_long_tail_outputs(stats: pd.DataFrame, grouped: pd.DataFrame, dataset_summary: pd.DataFrame, outputs: str | Path) -> None:
    outputs = Path(outputs)
    analysis_dir = ensure_dir(outputs / "analysis")
    stats.to_csv(analysis_dir / "class_distribution.csv", index=False)
    dataset_summary.to_csv(analysis_dir / "dataset_summary.csv", index=False)
    grouped[["class_id", "class_name", "group", "instance_count", "image_count", "val_instance_count"]].to_csv(
        analysis_dir / "class_groups.csv", index=False
    )
    summary = (
        grouped.groupby("group", as_index=False)
        .agg(classes=("class_id", "count"), instances=("instance_count", "sum"), images=("image_count", "sum"))
        .sort_values("group")
    )
    summary.to_csv(analysis_dir / "head_medium_tail_summary.csv", index=False)

    plt.figure(figsize=(12, 5))
    plot_df = grouped.sort_values("instance_count", ascending=False)
    colors = plot_df["group"].map({"head": "#2f6f8f", "medium": "#6a8f3a", "tail": "#b54c3f"}).fillna("#777777")
    plt.bar(plot_df["class_name"], plot_df["instance_count"], color=colors)
    plt.xticks(rotation=90, fontsize=7)
    plt.ylabel("Train instances")
    plt.tight_layout()
    plt.savefig(analysis_dir / "class_distribution.png", dpi=180)
    plt.close()


def normalize_log_count(counts: pd.Series) -> pd.Series:
    values = np.log1p(counts.astype(float))
    denom = values.max() - values.min()
    if denom <= 1e-12:
        return pd.Series(np.ones(len(values)), index=counts.index)
    return (values - values.min()) / denom


def normalize_ap(ap: pd.Series) -> pd.Series:
    values = ap.astype(float).fillna(0.0)
    if values.max() > 1.0:
        values = values / 100.0
    values = values.clip(0.0, 1.0)
    denom = values.max() - values.min()
    if denom <= 1e-12:
        return values
    return (values - values.min()) / denom


def compute_priority_scores(
    class_stats: pd.DataFrame,
    baseline_ap: pd.DataFrame | None,
    alpha: float = 0.6,
) -> pd.DataFrame:
    df = class_stats.copy()
    if baseline_ap is not None and not baseline_ap.empty:
        ap_cols = [c for c in ("ap50_95", "ap50", "baseline_ap") if c in baseline_ap.columns]
        ap_col = ap_cols[0] if ap_cols else None
        if ap_col:
            ap_df = (
                baseline_ap[["class_id", ap_col]]
                .rename(columns={ap_col: "baseline_ap"})
                .groupby("class_id", as_index=False)["baseline_ap"]
                .mean()
            )
            df = df.merge(ap_df, on="class_id", how="left")
    if "baseline_ap" not in df.columns:
        df["baseline_ap"] = 0.0
    df["baseline_ap"] = df["baseline_ap"].fillna(0.0)
    rarity_score = 1.0 - normalize_log_count(df["instance_count"])
    weakness_score = 1.0 - normalize_ap(df["baseline_ap"])
    df["rarity_score"] = rarity_score
    df["weakness_score"] = weakness_score
    df["priority_score"] = alpha * rarity_score + (1.0 - alpha) * weakness_score
    return df


def _allocate_by_weights(
    weights: np.ndarray,
    total_budget: int,
    min_per_class: int,
    max_per_class: int,
) -> list[int]:
    n = len(weights)
    if n == 0 or total_budget <= 0:
        return []
    allocation = np.zeros(n, dtype=int)
    if total_budget >= min_per_class * n:
        allocation[:] = min_per_class
    remaining = max(0, total_budget - int(allocation.sum()))
    weights = weights.astype(float)
    if weights.sum() <= 1e-12:
        weights = np.ones(n, dtype=float)
    while remaining > 0:
        caps = np.maximum(0, max_per_class - allocation)
        active = caps > 0
        if not active.any():
            break
        probs = weights * active
        probs = probs / probs.sum()
        raw = probs * remaining
        add = np.minimum(np.floor(raw).astype(int), caps)
        if add.sum() == 0:
            remainders = np.where(active, raw - np.floor(raw), -1.0)
            idx = int(np.argmax(remainders))
            add[idx] = 1
        allocation += add
        remaining -= int(add.sum())
    return allocation.tolist()


def build_augmentation_plans(
    grouped_stats: pd.DataFrame,
    baseline_ap: pd.DataFrame | None,
    selective_cfg: dict[str, Any],
    outputs: str | Path,
) -> tuple[Path, Path, Path]:
    analysis_dir = ensure_dir(Path(outputs) / "analysis")
    tail_df = grouped_stats[grouped_stats["group"] == "tail"].copy()
    scored = compute_priority_scores(tail_df, baseline_ap, alpha=float(selective_cfg.get("alpha", 0.6)))
    budget = int(selective_cfg.get("total_synthetic_budget", 0))
    min_per_class = int(selective_cfg.get("min_per_class", 0))
    max_per_class = int(selective_cfg.get("max_per_class", max(1, budget)))

    uniform = scored.copy()
    uniform_weights = np.ones(len(uniform), dtype=float)
    uniform["num_synthetic_images"] = _allocate_by_weights(uniform_weights, budget, min_per_class, max_per_class)

    selective = scored.copy()
    selective["num_synthetic_images"] = _allocate_by_weights(
        selective["priority_score"].to_numpy(dtype=float),
        budget,
        min_per_class,
        max_per_class,
    )

    # Weakness plan: the same budget and the same class count as the other two
    # plans, but the class set is chosen by measured baseline AP over *every*
    # class instead of by frequency. On this dataset instance_count and AP50
    # correlate at -0.33, so the frequency-defined tail is not the weak set —
    # holding K and budget fixed makes the allocation signal the only variable.
    num_classes = int(selective_cfg.get("weakness_num_classes", 0) or len(tail_df))
    all_scored = compute_priority_scores(
        grouped_stats.copy(), baseline_ap, alpha=float(selective_cfg.get("alpha", 0.6))
    )
    # class_id as the secondary key keeps the selection deterministic when the
    # baselines have not been trained yet and every baseline_ap is still 0.0.
    weakness = (
        all_scored.sort_values(["baseline_ap", "class_id"], ascending=[True, True]).head(num_classes).copy()
    )
    weakness["num_synthetic_images"] = _allocate_by_weights(
        weakness["weakness_score"].to_numpy(dtype=float),
        budget,
        min_per_class,
        max_per_class,
    )
    weakness = weakness.sort_values("class_id")

    columns = [
        "class_id",
        "class_name",
        "instance_count",
        "image_count",
        "baseline_ap",
        "rarity_score",
        "weakness_score",
        "priority_score",
        "num_synthetic_images",
    ]
    uniform_path = analysis_dir / "augmentation_plan_uniform.csv"
    selective_path = analysis_dir / "augmentation_plan_selective.csv"
    weakness_path = analysis_dir / "augmentation_plan_weakness.csv"
    uniform[columns].to_csv(uniform_path, index=False)
    selective[columns].to_csv(selective_path, index=False)
    weakness[columns].to_csv(weakness_path, index=False)
    return uniform_path, selective_path, weakness_path


def analyze_long_tail(data_yaml: str | Path, config: dict[str, Any], outputs: str | Path) -> pd.DataFrame:
    stats = collect_class_statistics(data_yaml)
    grouped = assign_class_groups(stats, config.get("tail", {}))
    dataset_summary = collect_dataset_summary(data_yaml, stats)
    save_long_tail_outputs(stats, grouped, dataset_summary, outputs)
    counts = stats.loc[stats["instance_count"] > 0, "instance_count"]
    if not counts.empty:
        ratio = counts.max() / max(1, counts.min())
        total_images = int(dataset_summary.loc[0, "num_images_total"])
        total_bboxes = int(dataset_summary.loc[0, "total_bboxes"])
        print(f"[INFO] 이미지 수: {total_images}, 클래스 수: {len(stats)}, 총 bbox: {total_bboxes}, imbalance ratio: {ratio:.2f}")
    return grouped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze long-tailed class distribution.")
    parser.add_argument("--data", required=True, help="Ultralytics data.yaml")
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--outputs", default=None)
    parser.add_argument("--baseline-ap", default=None, help="Optional per_class_ap.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    outputs = args.outputs or cfg["paths"]["outputs"]
    grouped = analyze_long_tail(args.data, cfg, outputs)
    baseline_ap = pd.read_csv(args.baseline_ap) if args.baseline_ap and Path(args.baseline_ap).exists() else None
    build_augmentation_plans(grouped, baseline_ap, cfg.get("selective_generation", {}), outputs)


if __name__ == "__main__":
    main()
