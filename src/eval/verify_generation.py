"""Automatic verification that inpainted synthetic images are real generations.

For every generated image we quantify, against its source image:
- mean absolute pixel diff OUTSIDE all protected bboxes (the inpainted background),
- SSIM restricted to the background region,
- LPIPS on the full image (optional dependency, CPU-capable),
- mean absolute pixel diff INSIDE the protected bboxes (protection-violation monitor).

An image counts as a failed generation when the background barely changed
(diff < verification.min_background_change) or the protected region changed
(diff > verification.max_bbox_protected_change). The inpainting loop retries
failed images and the pipeline aborts when the overall failure rate exceeds
verification.max_failure_rate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

for _parent in Path(__file__).resolve().parents:
    if (_parent / "src").is_dir():
        sys.path.insert(0, str(_parent))
        break

import numpy as np
import pandas as pd
from PIL import Image

from src.augment.masks import labels_to_pixel_boxes
from src.utils.image import (
    background_mean_abs_diff,
    bbox_interior_mean_abs_diff,
    boxes_mask,
    is_nontrivial_image,
    load_rgb,
)
from src.utils.io import ensure_dir, load_config
from src.utils.yolo import read_yolo_labels

Boxes = Sequence[tuple[int, int, int, int]]
LpipsFn = Callable[[Image.Image, Image.Image], float | None]

VERIFICATION_REPORT_NAME = "verification_report.csv"


def verification_config(config: dict[str, Any]) -> dict[str, Any]:
    """Read generation-verification thresholds from config with safe defaults."""
    ver = config.get("verification", {}) or {}
    diffusion = config.get("diffusion", {}) or {}
    return {
        "min_background_change": float(ver.get("min_background_change", 10.0)),
        # JPEG(quality 95) round-trips alone produce ~1-3 mean abs diff inside the
        # pasted bbox, so the protection monitor must sit above that noise floor.
        "max_bbox_protected_change": float(ver.get("max_bbox_protected_change", 5.0)),
        "min_editable_background_ratio": float(ver.get("min_editable_background_ratio", 0.05)),
        "max_retries_per_image": int(ver.get("max_retries_per_image", diffusion.get("max_retries_per_image", 2))),
        "max_failure_rate": float(ver.get("max_failure_rate", 0.05)),
        # Refill rejected images from other sources until the planned budget is met,
        # capped at needed * budget_attempt_multiplier attempts per class, so every
        # tail variant is compared at the same realized budget.
        "budget_attempt_multiplier": float(ver.get("budget_attempt_multiplier", 2.0)),
        "compute_lpips": bool(ver.get("compute_lpips", True)),
        "lpips_max_side": int(ver.get("lpips_max_side", 512)),
    }


def background_ssim(
    original: Image.Image,
    generated: Image.Image,
    protected_boxes: Boxes,
    win: int = 7,
) -> float:
    """SSIM between original and generated restricted to the background region.

    Windowed SSIM (uniform filter) on grayscale; the mean is taken only over
    pixels outside every protected bbox. Returns 1.0 when there is no background.
    """
    from scipy.ndimage import uniform_filter

    if generated.size != original.size:
        generated = generated.resize(original.size)
    a = np.asarray(original.convert("L"), dtype=np.float64)
    b = np.asarray(generated.convert("L"), dtype=np.float64)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mu_a = uniform_filter(a, win)
    mu_b = uniform_filter(b, win)
    var_a = uniform_filter(a * a, win) - mu_a**2
    var_b = uniform_filter(b * b, win) - mu_b**2
    cov = uniform_filter(a * b, win) - mu_a * mu_b
    ssim_map = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2))
    background = ~boxes_mask(original.size, protected_boxes)
    if not background.any():
        return 1.0
    return float(ssim_map[background].mean())


def make_lpips_scorer(max_side: int = 512) -> LpipsFn | None:
    """Build a CPU-capable LPIPS scorer, or None when no backend is installed.

    Tries torchmetrics first, then the standalone `lpips` package.
    """
    raw_scorer = None
    try:
        import torch
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True)
        metric.eval()

        def raw_scorer(a: Image.Image, b: Image.Image) -> float:  # type: ignore[misc]
            ta, tb = (_to_lpips_tensor(img, max_side) for img in (a, b))
            with torch.no_grad():
                return float(metric(ta, tb))

    except Exception:
        try:
            import lpips as lpips_pkg
            import torch

            net = lpips_pkg.LPIPS(net="alex")
            net.eval()

            def raw_scorer(a: Image.Image, b: Image.Image) -> float:  # type: ignore[misc]
                ta, tb = (_to_lpips_tensor(img, max_side) * 2.0 - 1.0 for img in (a, b))
                with torch.no_grad():
                    return float(net(ta, tb))

        except Exception:
            print("[WARN] LPIPS backend(torchmetrics/lpips)가 없어 LPIPS는 기록하지 않습니다.")
            return None

    # Per-call failures degrade to None (quality metric, never a pipeline gate).
    warned = {"done": False}

    def scorer(a: Image.Image, b: Image.Image) -> float | None:
        try:
            return raw_scorer(a, b)
        except Exception as exc:
            if not warned["done"]:
                print(f"[WARN] LPIPS 계산 실패 — lpips는 비워둡니다: {exc}")
                warned["done"] = True
            return None

    return scorer


def _to_lpips_tensor(image: Image.Image, max_side: int):
    import torch

    if max(image.size) > max_side:
        scale = max_side / max(image.size)
        image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))
    arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0  # writable copy
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def verify_pair(
    original: Image.Image,
    generated: Image.Image,
    protected_boxes: Boxes,
    ver_cfg: dict[str, Any],
    lpips_fn: LpipsFn | None = None,
    compute_ssim: bool = True,
) -> tuple[bool, dict[str, Any]]:
    """Judge one generated image against its source. Returns (passed, metrics)."""
    background_diff = background_mean_abs_diff(original, generated, protected_boxes)
    interior_diff = bbox_interior_mean_abs_diff(original, generated, protected_boxes)
    size_ok = generated.size == original.size
    nontrivial = is_nontrivial_image(generated)
    background_ok = background_diff >= ver_cfg["min_background_change"]
    protected_ok = interior_diff <= ver_cfg["max_bbox_protected_change"]
    passed = bool(size_ok and nontrivial and background_ok and protected_ok)
    reasons = []
    if not background_ok:
        reasons.append("background_unchanged")
    if not protected_ok:
        reasons.append("protected_region_changed")
    if not size_ok:
        reasons.append("size_mismatch")
    if not nontrivial:
        reasons.append("trivial_image")
    ssim_value = None
    if compute_ssim:
        try:
            ssim_value = background_ssim(original, generated, protected_boxes)
        except Exception as exc:
            print(f"[WARN] SSIM 계산 실패: {exc}")
    lpips_value = lpips_fn(original, generated) if lpips_fn is not None else None
    return passed, {
        "background_pixel_diff": background_diff,
        "background_ssim": ssim_value,
        "lpips": lpips_value,
        "bbox_interior_pixel_diff": interior_diff,
        "output_size_matches": size_ok,
        "nontrivial_image": nontrivial,
        "verification_passed": passed,
        "verification_fail_reason": ";".join(reasons),
    }


def update_verification_report(outputs: str | Path, plan_name: str, rows: list[dict[str, Any]]) -> Path:
    """Merge one plan's verification rows into outputs/synthetic/verification_report.csv."""
    synthetic_dir = ensure_dir(Path(outputs) / "synthetic")
    report_path = synthetic_dir / VERIFICATION_REPORT_NAME
    new_df = pd.DataFrame(rows)
    if not new_df.empty:
        new_df.insert(0, "plan", plan_name)
    if report_path.exists():
        old = pd.read_csv(report_path)
        old = old[old.get("plan", "") != plan_name] if "plan" in old.columns else old
        new_df = pd.concat([old, new_df], ignore_index=True)
    new_df.to_csv(report_path, index=False)
    return report_path


def enforce_failure_rate(log_df: pd.DataFrame, max_failure_rate: float, plan_name: str) -> float:
    """Abort the pipeline when too many generations failed verification.

    Dry-run rows are structure checks and excluded. Returns the failure rate.
    """
    if log_df.empty:
        return 0.0
    real = log_df[~log_df["dry_run"].astype(bool)] if "dry_run" in log_df.columns else log_df
    if real.empty:
        return 0.0
    failure_rate = float(1.0 - real["accepted"].astype(bool).mean())
    if failure_rate > max_failure_rate:
        raise RuntimeError(
            f"synthetic 생성 실패율 {failure_rate:.1%}가 verification.max_failure_rate="
            f"{max_failure_rate:.1%}를 초과했습니다 (plan={plan_name}). "
            "diffusion 파라미터/프롬프트/마스크 설정을 점검한 뒤 재실행하세요."
        )
    return failure_rate


def reverify_from_log(
    generation_log_csv: str | Path,
    config: dict[str, Any],
    outputs: str | Path,
    plan_name: str,
    padding_ratio: float | None = None,
) -> Path:
    """Re-verify already generated images listed in a generation log (offline QC)."""
    ver_cfg = verification_config(config)
    diffusion_cfg = config.get("diffusion", {}) or {}
    padding = float(padding_ratio if padding_ratio is not None else diffusion_cfg.get("mask_padding_ratio", 0.1))
    lpips_fn = make_lpips_scorer(ver_cfg["lpips_max_side"]) if ver_cfg["compute_lpips"] else None
    log = pd.read_csv(generation_log_csv)
    rows: list[dict[str, Any]] = []
    for record in log.to_dict("records"):
        output_image = Path(str(record.get("output_image", "")))
        source_image = Path(str(record.get("source_image", "")))
        if not output_image.exists() or not source_image.exists():
            continue
        original = load_rgb(source_image)
        generated = load_rgb(output_image)
        label_path = source_image.parent.parent.parent / "labels" / "train" / source_image.with_suffix(".txt").name
        labels = read_yolo_labels(label_path)
        boxes = labels_to_pixel_boxes(labels, original.size, padding_ratio=padding)
        passed, metrics = verify_pair(original, generated, boxes, ver_cfg, lpips_fn=lpips_fn)
        rows.append(
            {
                "source_image": str(source_image),
                "output_image": str(output_image),
                "class_id": record.get("class_id"),
                "class_name": record.get("class_name"),
                "prompt": record.get("prompt"),
                "seed": record.get("seed"),
                "generation_seed": record.get("generation_seed"),
                "retry_index": record.get("retry_index"),
                "accepted": passed,
                **metrics,
            }
        )
    return update_verification_report(outputs, plan_name, rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-verify generated synthetic images from a generation log.")
    parser.add_argument("--log", required=True, help="generation_log_<plan>.csv")
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--outputs", default=None)
    parser.add_argument("--plan-name", default="selective")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    outputs = args.outputs or cfg["paths"]["outputs"]
    report = reverify_from_log(args.log, cfg, outputs, args.plan_name)
    print(f"[INFO] verification report 저장: {report}")


if __name__ == "__main__":
    main()
