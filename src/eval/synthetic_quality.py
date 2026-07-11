"""Quantitative quality metrics for generated synthetic images.

- Per-class FID between the synthetic set and the real train images of the same
  class (torchmetrics FrechetInceptionDistance), plus an overall FID.
- Per-image CLIPScore (generation prompt vs generated image) and LPIPS
  (source vs generated), written to outputs*/synthetic/quality_report.csv.
- Quality-filter planning: drop the bottom CLIPScore percentile and emit a
  refill plan (with start_index offsets) so the synthetic budget stays constant.

All metrics run on CPU when no GPU is available; --max-images samples the set.
Scorers are injectable so tests run without the heavy model dependencies.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

for _parent in Path(__file__).resolve().parents:
    if (_parent / "src").is_dir():
        sys.path.insert(0, str(_parent))
        break

import numpy as np
import pandas as pd
from PIL import Image

from src.eval.verify_generation import make_lpips_scorer
from src.utils.image import load_rgb
from src.utils.io import ensure_dir, load_config

ClipScorer = Callable[[Image.Image, str], float | None]

QUALITY_REPORT_NAME = "quality_report.csv"
FID_REPORT_NAME = "fid_by_class.csv"


def quality_filter_config(config: dict[str, Any]) -> dict[str, Any]:
    qf = config.get("quality_filter", {}) or {}
    return {
        "enabled": bool(qf.get("enabled", False)),
        "clip_score_percentile": float(qf.get("clip_score_percentile", 50.0)),
        # P2 extension point: >1 rounds would re-score refills and iterate.
        "max_refill_rounds": int(qf.get("max_refill_rounds", 1)),
    }


def synthetic_quality_config(config: dict[str, Any]) -> dict[str, Any]:
    sq = config.get("synthetic_quality", {}) or {}
    return {
        "enabled": bool(sq.get("enabled", False)),
        "max_images": sq.get("max_images"),
        "fid_feature_dim": int(sq.get("fid_feature_dim", 2048)),
    }


def make_clip_scorer(model_name: str = "openai/clip-vit-base-patch16") -> ClipScorer | None:
    """CLIPScore(prompt, image) scorer via torchmetrics; None when unavailable."""
    try:
        import torch
        from torchmetrics.multimodal.clip_score import CLIPScore

        metric = CLIPScore(model_name_or_path=model_name)
        metric.eval()

        def scorer(image: Image.Image, prompt: str) -> float | None:
            arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
            tensor = torch.from_numpy(arr).permute(2, 0, 1)
            with torch.no_grad():
                return float(metric(tensor, prompt))

        return scorer
    except Exception:
        print("[WARN] torchmetrics CLIPScore를 사용할 수 없어 clip_score는 기록하지 않습니다.")
        return None


def compute_quality_report(
    generation_log_csv: str | Path,
    outputs: str | Path,
    plan_name: str,
    config: dict[str, Any],
    max_images: int | None = None,
    clip_scorer: ClipScorer | None = None,
    lpips_fn: Callable | None = None,
) -> Path:
    """Score accepted synthetic images (CLIPScore + LPIPS) into quality_report.csv."""
    log = pd.read_csv(generation_log_csv)
    if "dry_run" in log.columns:
        log = log[~log["dry_run"].astype(bool)]
    log = log[log["accepted"].astype(bool)]
    if max_images is not None and len(log) > max_images:
        log = log.sample(n=max_images, random_state=int(config.get("detector", {}).get("seeds", [42])[0]))
    if clip_scorer is None:
        clip_scorer = make_clip_scorer()
    if lpips_fn is None:
        ver = config.get("verification", {}) or {}
        lpips_fn = make_lpips_scorer(int(ver.get("lpips_max_side", 512)))

    rows: list[dict[str, Any]] = []
    for record in log.to_dict("records"):
        output_image = Path(str(record.get("output_image", "")))
        source_image = Path(str(record.get("source_image", "")))
        if not output_image.exists():
            continue
        generated = load_rgb(output_image)
        prompt = str(record.get("prompt", ""))
        clip_score = clip_scorer(generated, prompt) if clip_scorer is not None else None
        lpips_value = None
        if lpips_fn is not None and source_image.exists():
            lpips_value = lpips_fn(load_rgb(source_image), generated)
        rows.append(
            {
                "image": str(output_image),
                "class_id": record.get("class_id"),
                "class_name": record.get("class_name"),
                "prompt": prompt,
                "clip_score": clip_score,
                "lpips": lpips_value,
            }
        )

    synthetic_dir = ensure_dir(Path(outputs) / "synthetic")
    report_path = synthetic_dir / QUALITY_REPORT_NAME
    new_df = pd.DataFrame(rows)
    if not new_df.empty:
        new_df.insert(0, "plan", plan_name)
    if report_path.exists():
        old = pd.read_csv(report_path)
        if "plan" in old.columns:
            old = old[old["plan"] != plan_name]
        # keep previously scored images of this plan that we did not re-score
        new_df = pd.concat([old, new_df], ignore_index=True)
        new_df = new_df.drop_duplicates(subset=["plan", "image"], keep="last")
    new_df.to_csv(report_path, index=False)
    print(f"[INFO] quality report 저장: {report_path} ({len(rows)}장 채점)")
    return report_path


def compute_class_fid(
    generation_log_csv: str | Path,
    data_yaml: str | Path,
    outputs: str | Path,
    plan_name: str,
    config: dict[str, Any],
    max_images: int | None = None,
) -> Path:
    """Per-class and overall FID between synthetic images and real train images."""
    from src.augment.inpaint_background import collect_source_images_by_class

    sq_cfg = synthetic_quality_config(config)
    synthetic_dir = ensure_dir(Path(outputs) / "synthetic")
    fid_path = synthetic_dir / f"fid_by_class_{plan_name}.csv"
    try:
        import torch
        from torchmetrics.image.fid import FrechetInceptionDistance
    except Exception:
        print("[WARN] torchmetrics FrechetInceptionDistance를 사용할 수 없어 FID를 건너뜁니다.")
        pd.DataFrame(columns=["class_id", "class_name", "n_real", "n_synthetic", "fid"]).to_csv(fid_path, index=False)
        return fid_path

    log = pd.read_csv(generation_log_csv)
    if "dry_run" in log.columns:
        log = log[~log["dry_run"].astype(bool)]
    log = log[log["accepted"].astype(bool)]
    real_by_class = collect_source_images_by_class(data_yaml)

    def _tensor_batch(paths: list[Path], limit: int | None):
        selected = paths[:limit] if limit else paths
        batch = []
        for path in selected:
            image = load_rgb(path).resize((299, 299))
            batch.append(torch.from_numpy(np.asarray(image, dtype=np.uint8)).permute(2, 0, 1))
        return torch.stack(batch) if batch else None

    rows: list[dict[str, Any]] = []
    all_real: list[Path] = []
    all_fake: list[Path] = []
    for class_id, group in log.groupby("class_id"):
        fake_paths = [Path(p) for p in group["output_image"].astype(str) if Path(p).exists()]
        real_paths = [img for img, _ in real_by_class.get(int(class_id), [])]
        all_real.extend(real_paths)
        all_fake.extend(fake_paths)
        fid_value = None
        if len(fake_paths) >= 2 and len(real_paths) >= 2:
            metric = FrechetInceptionDistance(feature=sq_cfg["fid_feature_dim"], normalize=False)
            real_batch = _tensor_batch(real_paths, max_images)
            fake_batch = _tensor_batch(fake_paths, max_images)
            metric.update(real_batch, real=True)
            metric.update(fake_batch, real=False)
            fid_value = float(metric.compute())
        rows.append(
            {
                "class_id": int(class_id),
                "class_name": group["class_name"].iloc[0],
                "n_real": len(real_paths),
                "n_synthetic": len(fake_paths),
                "fid": fid_value,
            }
        )
    if len(all_fake) >= 2 and len(all_real) >= 2:
        metric = FrechetInceptionDistance(feature=sq_cfg["fid_feature_dim"], normalize=False)
        metric.update(_tensor_batch(sorted(set(all_real)), max_images), real=True)
        metric.update(_tensor_batch(all_fake, max_images), real=False)
        rows.append(
            {
                "class_id": -1,
                "class_name": "__overall__",
                "n_real": len(set(all_real)),
                "n_synthetic": len(all_fake),
                "fid": float(metric.compute()),
            }
        )
    pd.DataFrame(rows).to_csv(fid_path, index=False)
    print(f"[INFO] FID report 저장: {fid_path}")
    return fid_path


def plan_quality_filter(
    quality_report: pd.DataFrame,
    plan_name: str,
    clip_score_percentile: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a plan's scored images into kept/dropped by CLIPScore percentile.

    Returns (filter_df with `kept` flag, refill_plan with per-class counts and
    start_index offsets). When no CLIP scores exist, everything is kept.
    """
    df = quality_report[quality_report.get("plan", plan_name) == plan_name].copy()
    if df.empty:
        return df.assign(kept=True), pd.DataFrame(columns=["class_id", "class_name", "num_synthetic_images", "start_index"])
    scores = pd.to_numeric(df["clip_score"], errors="coerce")
    if scores.notna().sum() == 0:
        print("[WARN] clip_score가 없어 품질 필터링을 건너뜁니다 (전체 유지).")
        df["kept"] = True
        return df, pd.DataFrame(columns=["class_id", "class_name", "num_synthetic_images", "start_index"])
    cutoff = float(np.nanpercentile(scores.to_numpy(dtype=float), clip_score_percentile))
    df["kept"] = scores >= cutoff
    dropped = df[~df["kept"]]
    refill_rows = []
    for class_id, group in dropped.groupby("class_id"):
        total_in_class = int((df["class_id"] == class_id).sum())
        refill_rows.append(
            {
                "class_id": int(class_id),
                "class_name": group["class_name"].iloc[0],
                "num_synthetic_images": int(len(group)),
                # continue numbering after every already-generated image
                "start_index": total_in_class,
            }
        )
    return df, pd.DataFrame(refill_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score synthetic image quality (CLIPScore/LPIPS/FID).")
    parser.add_argument("--log", required=True, help="generation_log_<plan>.csv")
    parser.add_argument("--data", default=None, help="Base data.yaml (required for FID)")
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--outputs", default=None)
    parser.add_argument("--plan-name", default="selective")
    parser.add_argument("--max-images", type=int, default=None, help="Sample size cap for CPU runs")
    parser.add_argument("--skip-fid", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    outputs = args.outputs or cfg["paths"]["outputs"]
    compute_quality_report(args.log, outputs, args.plan_name, cfg, max_images=args.max_images)
    if not args.skip_fid and args.data:
        compute_class_fid(args.log, args.data, outputs, args.plan_name, cfg, max_images=args.max_images)


if __name__ == "__main__":
    main()
