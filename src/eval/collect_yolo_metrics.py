from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

for _parent in Path(__file__).resolve().parents:
    if (_parent / "src").is_dir():
        sys.path.insert(0, str(_parent))
        break

import pandas as pd

from src.utils.detector import load_detector
from src.utils.io import ensure_dir, load_config, load_yaml
from src.utils.yolo import normalize_class_names


def _last_results_row(run_dir: Path) -> dict[str, Any]:
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        return {}
    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]
    if df.empty:
        return {}
    row = df.iloc[-1].to_dict()
    return {
        "mAP50": row.get("metrics/mAP50(B)", row.get("metrics/mAP50", None)),
        "mAP50_95": row.get("metrics/mAP50-95(B)", row.get("metrics/mAP50-95", None)),
        "precision": row.get("metrics/precision(B)", row.get("metrics/precision", None)),
        "recall": row.get("metrics/recall(B)", row.get("metrics/recall", None)),
    }


def _training_seconds(run_dir: Path) -> float | None:
    meta = run_dir / "training_meta.yaml"
    if meta.exists():
        data = load_yaml(meta)
        return data.get("training_seconds")
    return None


def _class_names(data_yaml: str | Path) -> list[str]:
    data = load_yaml(data_yaml)
    return normalize_class_names(data.get("names"), data.get("nc"))


def per_class_ap_table(
    names: list[str],
    maps: Any = None,
    all_ap: Any = None,
    ap_class_index: Any = None,
) -> pd.DataFrame:
    """Per-class AP rows aligned by class id.

    box.all_ap has one row per class PRESENT in the eval split, ordered by
    box.ap_class_index — not by class id. Classes absent from the split stay
    None (NaN in the CSV) so group means skip them; Ultralytics' maps fills
    absent classes with the overall mAP, which would pollute group averages.
    """
    ap_rows_by_class: dict[int, Any] = {}
    if all_ap is not None and ap_class_index is not None and len(ap_class_index) == len(all_ap):
        for row_idx, present_class_id in enumerate(ap_class_index):
            ap_rows_by_class[int(present_class_id)] = all_ap[row_idx]
    per_class_rows: list[dict[str, Any]] = []
    for class_id, class_name in enumerate(names):
        ap50_95 = None
        ap50 = None
        ap_row = ap_rows_by_class.get(class_id)
        if ap_row is not None:
            ap50 = float(ap_row[0])
            ap50_95 = float(sum(ap_row) / len(ap_row))
        elif not ap_rows_by_class and maps is not None and class_id < len(maps):
            # Legacy fallback when ap_class_index is unavailable in this
            # Ultralytics version (maps is class-id aligned but fills absent
            # classes with the overall mAP).
            ap50_95 = float(maps[class_id])
        per_class_rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "ap50": ap50,
                "ap50_95": ap50_95,
            }
        )
    return pd.DataFrame(per_class_rows)


def validate_and_collect_per_class(
    weights: str | Path,
    data_yaml: str | Path,
    imgsz: int = 640,
    split: str = "test",
    model_name: str | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    # model_name으로 계열을 판정해 RT-DETR 체크포인트는 RTDETR 클래스(전용
    # validator)로 연다 — YOLO 클래스로 열면 NMS 전제 후처리가 잘못 적용된다.
    model = load_detector(weights, model_name=model_name)
    metrics = model.val(data=str(data_yaml), imgsz=imgsz, split=split, plots=True, verbose=False)
    names = _class_names(data_yaml)
    box = getattr(metrics, "box", None)
    overall = {
        "mAP50": getattr(box, "map50", None),
        "mAP50_95": getattr(box, "map", None),
        "precision": getattr(box, "mp", None),
        "recall": getattr(box, "mr", None),
    }
    per_class = per_class_ap_table(
        names,
        maps=getattr(box, "maps", None),
        all_ap=getattr(box, "all_ap", None),
        ap_class_index=getattr(box, "ap_class_index", None),
    )
    return overall, per_class


def collect_metrics(
    run_dir: str | Path,
    outputs: str | Path,
    experiment: str,
    seed: int,
    model_name: str,
    data_yaml: str | Path | None = None,
    weights: str | Path | None = None,
    imgsz: int = 640,
    split: str = "test",
) -> tuple[Path, Path]:
    run_dir = Path(run_dir)
    outputs = Path(outputs)
    metrics_dir = ensure_dir(outputs / "metrics")
    overall = _last_results_row(run_dir)
    per_class = pd.DataFrame()
    if data_yaml and weights and Path(weights).exists():
        try:
            val_overall, per_class = validate_and_collect_per_class(
                weights, data_yaml, imgsz=imgsz, split=split, model_name=model_name
            )
            overall.update({k: v for k, v in val_overall.items() if v is not None})
        except Exception as exc:
            print(f"[WARN] class-wise validation metric 수집 실패. results.csv 파싱만 사용합니다: {exc}")
    confusion_paths = list(run_dir.rglob("confusion_matrix*.png"))
    raw_row = {
        "experiment": experiment,
        "seed": seed,
        "model_name": model_name,
        "run_dir": str(run_dir),
        "weights": str(weights) if weights else "",
        "mAP50": overall.get("mAP50"),
        "mAP50_95": overall.get("mAP50_95"),
        "precision": overall.get("precision"),
        "recall": overall.get("recall"),
        "eval_split": split,
        "confusion_matrix_path": str(confusion_paths[0]) if confusion_paths else "",
        "training_seconds": _training_seconds(run_dir),
    }
    raw_path = metrics_dir / "raw_yolo_metrics.csv"
    raw_df = pd.DataFrame([raw_row])
    if raw_path.exists():
        raw_df = pd.concat([pd.read_csv(raw_path), raw_df], ignore_index=True)
    raw_df = raw_df.drop_duplicates(subset=["experiment", "seed", "eval_split"], keep="last")
    raw_df.to_csv(raw_path, index=False)

    per_class_path = metrics_dir / "per_class_ap.csv"
    if not per_class.empty:
        per_class.insert(0, "seed", seed)
        per_class.insert(0, "experiment", experiment)
        per_class["run_dir"] = str(run_dir)
        per_class["model_name"] = model_name
        per_class["eval_split"] = split
        if per_class_path.exists():
            per_class = pd.concat([pd.read_csv(per_class_path), per_class], ignore_index=True)
        per_class = per_class.drop_duplicates(subset=["experiment", "seed", "eval_split", "class_id"], keep="last")
        per_class.to_csv(per_class_path, index=False)
    elif not per_class_path.exists():
        pd.DataFrame(
            columns=["experiment", "seed", "class_id", "class_name", "ap50", "ap50_95", "run_dir", "model_name", "eval_split"]
        ).to_csv(per_class_path, index=False)
    print(f"[INFO] metric 저장: {raw_path}, {per_class_path}")
    return raw_path, per_class_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect YOLO metrics.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--outputs", default=None)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model-name", default="yolo")
    parser.add_argument("--data", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--split", default=None, choices=["train", "val", "test"], help="Evaluation split. Defaults to config eval.split or test.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    outputs = args.outputs or cfg["paths"]["outputs"]
    imgsz = int(cfg.get("detector", {}).get("imgsz", 640))
    split = args.split or cfg.get("eval", {}).get("split", "test")
    weights = args.weights
    if weights is None:
        candidate = Path(args.run_dir) / "weights" / "best.pt"
        weights = candidate if candidate.exists() else None
    collect_metrics(
        args.run_dir,
        outputs,
        args.experiment,
        args.seed,
        args.model_name,
        data_yaml=args.data,
        weights=weights,
        imgsz=imgsz,
        split=split,
    )


if __name__ == "__main__":
    main()
