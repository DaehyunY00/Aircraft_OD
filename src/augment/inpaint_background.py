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
from PIL import Image
from tqdm import tqdm

from src.augment.masks import create_inpainting_mask, labels_to_pixel_boxes
from src.utils.image import (
    editable_background_ratio,
    is_nontrivial_image,
    load_rgb,
    max_bbox_diff,
    paste_protected_regions,
    save_contact_sheet,
)
from src.eval.verify_generation import (
    enforce_failure_rate,
    make_lpips_scorer,
    update_verification_report,
    verification_config,
    verify_pair,
)
from src.utils.io import copy_file, ensure_dir, load_config
from src.utils.seed import set_seed
from src.utils.timing import ProgressTimer
from src.utils.yolo import label_path_for_image, list_images, read_yolo_labels

DRY_RUN_MARKER_NAME = "DRY_RUN_MARKER.txt"


def _torch_dtype(name: str):
    import torch

    return {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(str(name).lower(), torch.float16)


def load_inpaint_pipeline(diffusion_cfg: dict[str, Any]):
    import torch
    from diffusers import StableDiffusionInpaintPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = _torch_dtype(diffusion_cfg.get("torch_dtype", "float16")) if device == "cuda" else torch.float32
    model_id = diffusion_cfg.get("model_id", "runwayml/stable-diffusion-inpainting")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    if diffusion_cfg.get("enable_attention_slicing", True):
        pipe.enable_attention_slicing()
    if diffusion_cfg.get("enable_xformers_if_available", True):
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    return pipe, device


def _split_dirs(data_yaml: str | Path) -> tuple[Path, Path]:
    import yaml

    data_yaml = Path(data_yaml)
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(data.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    train = Path(data.get("train", "images/train"))
    if not train.is_absolute():
        train = root / train
    labels = Path(str(train).replace("/images/", "/labels/"))
    if not labels.exists():
        labels = root / "labels" / "train"
    return train, labels


def collect_source_images_by_class(data_yaml: str | Path) -> dict[int, list[tuple[Path, Path]]]:
    images_dir, labels_dir = _split_dirs(data_yaml)
    by_class: dict[int, list[tuple[Path, Path]]] = {}
    for image_path in list_images(images_dir):
        label_path = label_path_for_image(image_path, images_dir, labels_dir)
        labels = read_yolo_labels(label_path)
        present = sorted({int(label["class_id"]) for label in labels})
        for class_id in present:
            by_class.setdefault(class_id, []).append((image_path, label_path))
    return by_class


def filter_sources_by_editable_ratio(
    sources: list[tuple[Path, Path]],
    padding_ratio: float,
    min_editable_ratio: float,
) -> tuple[list[tuple[Path, Path]], int]:
    """Drop source images whose padded bboxes cover (almost) the whole frame.

    For such images background inpainting is a geometric no-op: the mask protects
    everything, diffusion cannot change anything, and the "synthetic" output is a
    copy of the original. They must not consume generation budget.
    """
    eligible: list[tuple[Path, Path]] = []
    skipped = 0
    for image_path, label_path in sources:
        try:
            with Image.open(image_path) as image:
                size = image.size
            boxes = labels_to_pixel_boxes(read_yolo_labels(label_path), size, padding_ratio=padding_ratio)
        except Exception:
            skipped += 1
            continue
        if editable_background_ratio(size, boxes) >= min_editable_ratio:
            eligible.append((image_path, label_path))
        else:
            skipped += 1
    return eligible, skipped


def _run_inpaint(
    pipe,
    device: str,
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    negative_prompt: str,
    diffusion_cfg: dict[str, Any],
    seed: int,
) -> Image.Image:
    import torch

    resolution = int(diffusion_cfg.get("resolution", 512))
    work_size = (resolution, resolution)
    work_image = image.resize(work_size, Image.Resampling.LANCZOS)
    # NEAREST keeps the mask binary: LANCZOS ringing plus the pipeline's 0.5
    # binarization would erode the editable background band around each bbox.
    work_mask = mask.resize(work_size, Image.Resampling.NEAREST)
    generator = torch.Generator(device=device).manual_seed(seed)
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=work_image,
        mask_image=work_mask,
        num_inference_steps=int(diffusion_cfg.get("num_inference_steps", 20)),
        guidance_scale=float(diffusion_cfg.get("guidance_scale", 7.5)),
        strength=float(diffusion_cfg.get("strength", 0.85)),
        generator=generator,
    ).images[0]
    return result.resize(image.size, Image.Resampling.LANCZOS)


def generate_from_plan(
    data_yaml: str | Path,
    plan_csv: str | Path,
    out_root: str | Path,
    outputs: str | Path,
    config: dict[str, Any],
    plan_name: str = "selective",
    force: bool = False,
    dry_run: bool = False,
) -> Path:
    diffusion_cfg = config.get("diffusion", {})
    ver_cfg = verification_config(config)
    seed = int(config.get("detector", {}).get("seeds", [42])[0])
    set_seed(seed)
    plan = pd.read_csv(plan_csv)
    source_by_class = collect_source_images_by_class(data_yaml)
    padding_ratio = float(diffusion_cfg.get("mask_padding_ratio", 0.1))
    for class_id, sources in list(source_by_class.items()):
        eligible, skipped = filter_sources_by_editable_ratio(
            sources, padding_ratio, ver_cfg["min_editable_background_ratio"]
        )
        if skipped:
            print(
                f"[WARN] class {class_id}: 배경 편집 가능 영역이 "
                f"{ver_cfg['min_editable_background_ratio']:.0%} 미만인 source {skipped}장 제외"
            )
        source_by_class[class_id] = eligible
    out_root = Path(out_root)
    image_out_dir = ensure_dir(out_root / plan_name / "images" / "train")
    label_out_dir = ensure_dir(out_root / plan_name / "labels" / "train")
    rejected_dir = ensure_dir(out_root / plan_name / "rejected")
    canonical_image_dir = ensure_dir(out_root / "images" / "train") if plan_name == "selective" else None
    canonical_label_dir = ensure_dir(out_root / "labels" / "train") if plan_name == "selective" else None
    canonical_rejected_dir = ensure_dir(out_root / "rejected") if plan_name == "selective" else None
    synthetic_dir = ensure_dir(Path(outputs) / "synthetic")
    debug_dir = ensure_dir(synthetic_dir / "debug_masks" / plan_name)

    dry_run_marker = out_root / plan_name / DRY_RUN_MARKER_NAME
    stale_from_dry_run = False
    if dry_run:
        print(
            "[WARN] --dry-run-inpaint는 파이프라인 구조 점검 전용입니다. "
            "synthetic 이미지는 원본 사본이며 학습/논문 실험에 사용하면 안 됩니다."
        )
        dry_run_marker.parent.mkdir(parents=True, exist_ok=True)
        dry_run_marker.write_text(
            "This plan directory was produced by --dry-run-inpaint (original copies, no diffusion).\n"
            "A subsequent real run regenerates every file in this directory.\n",
            encoding="utf-8",
        )
    elif dry_run_marker.exists():
        stale_from_dry_run = True
        print(
            f"[WARN] {dry_run_marker}가 존재합니다. 이전 dry-run이 만든 원본 사본을 "
            "정상 생성물로 재사용하지 않고 전체 재생성합니다."
        )

    pipe = None
    device = "cpu"
    lpips_fn = None
    if not dry_run:
        pipe, device = load_inpaint_pipeline(diffusion_cfg)
        if ver_cfg["compute_lpips"]:
            lpips_fn = make_lpips_scorer(ver_cfg["lpips_max_side"])

    prompts = diffusion_cfg.get("prompts") or ["realistic sky background, aviation photography"]
    negative_prompt = diffusion_cfg.get("negative_prompt", "")
    logs: list[dict[str, Any]] = []
    contact_rows: list[list[Image.Image]] = []
    total_images = int(
        sum(
            int(row.get("num_synthetic_images", 0))
            for _, row in plan.iterrows()
            if source_by_class.get(int(row["class_id"]), [])
        )
    )
    timer = ProgressTimer(total_images)
    image_bar = tqdm(total=total_images, desc=f"inpaint {plan_name}", disable=total_images == 0)

    for _, row in plan.iterrows():
        class_id = int(row["class_id"])
        needed = int(row.get("num_synthetic_images", 0))
        # Refill plans (quality filtering) continue numbering after the initial
        # budget so new images never collide with (or reuse the seeds of) the
        # originally generated ones.
        raw_start = row.get("start_index", 0)
        start_index = int(raw_start) if pd.notna(raw_start) else 0
        sources = source_by_class.get(class_id, [])
        if needed <= 0 or not sources:
            continue
        # Budget-based generation: keep advancing the index (new source + seed)
        # until `needed` images pass verification, so verification rejects are
        # refilled and every tail variant lands at the same realized budget.
        accepted_count = 0
        max_attempts = max(needed, int(round(needed * ver_cfg["budget_attempt_multiplier"])))
        for attempt in range(max_attempts):
            if accepted_count >= needed:
                break
            idx = start_index + attempt
            source_image, source_label = sources[idx % len(sources)]
            prompt = prompts[idx % len(prompts)]
            attempt_seed = seed + class_id * 100_000 + idx
            output_name = f"{source_image.stem}_{plan_name}_c{class_id}_{idx:04d}_s{attempt_seed}.jpg"
            output_image = image_out_dir / output_name
            output_label = label_out_dir / output_image.with_suffix(".txt").name
            canonical_output_image = canonical_image_dir / output_name if canonical_image_dir else output_image
            canonical_output_label = canonical_label_dir / output_label.name if canonical_label_dir else output_label
            original = load_rgb(source_image)
            labels = read_yolo_labels(source_label)
            if output_image.exists() and output_label.exists() and not force and not dry_run and not stale_from_dry_run:
                # Resume path: never trust an existing file blindly — a stale
                # dry-run copy or corrupt file must be regenerated, not laundered
                # into the synthetic set.
                existing = load_rgb(output_image)
                existing_boxes = labels_to_pixel_boxes(labels, original.size, padding_ratio=padding_ratio)
                verified, metrics = verify_pair(original, existing, existing_boxes, ver_cfg, lpips_fn=lpips_fn)
                if verified:
                    if canonical_image_dir and not canonical_output_image.exists():
                        copy_file(output_image, canonical_output_image, overwrite=True)
                    if canonical_label_dir and not canonical_output_label.exists():
                        copy_file(output_label, canonical_output_label, overwrite=True)
                    label_count_output = len(read_yolo_labels(canonical_output_label)) if canonical_output_label.exists() else 0
                    logs.append(
                        {
                            "source_image": str(source_image),
                            "output_image": str(canonical_output_image),
                            "class_id": class_id,
                            "class_name": row.get("class_name", f"class_{class_id}"),
                            "prompt": prompt,
                            "seed": attempt_seed,
                            "accepted": True,
                            "reject_reason": "already_exists_verified",
                            "dry_run": False,
                            "bbox_pixel_diff": metrics["bbox_interior_pixel_diff"],
                            "background_pixel_diff": metrics["background_pixel_diff"],
                            "background_ssim": metrics["background_ssim"],
                            "lpips": metrics["lpips"],
                            "verification_passed": True,
                            "verification_fail_reason": "",
                            "mask_padding": padding_ratio,
                            "inference_steps": diffusion_cfg.get("num_inference_steps", 20),
                            "label_count_original": len(labels),
                            "label_count_output": label_count_output,
                            "label_sanity_ok": len(labels) == label_count_output,
                            "output_exists": canonical_output_image.exists(),
                            "output_size_matches": metrics["output_size_matches"],
                            "nontrivial_image": metrics["nontrivial_image"],
                        }
                    )
                    accepted_count += 1
                    timer.update()
                    image_bar.update(1)
                    image_bar.set_postfix_str(timer.status())
                    continue
                print(
                    f"[WARN] 기존 synthetic 파일이 검증에 실패해 재생성합니다: {output_image.name} "
                    f"({metrics['verification_fail_reason']})"
                )
            mask, padded_boxes, original_boxes = create_inpainting_mask(
                original.size,
                labels,
                padding_ratio=float(diffusion_cfg.get("mask_padding_ratio", 0.1)),
                blur_radius=int(diffusion_cfg.get("mask_blur_radius", 8)),
            )
            if len(contact_rows) < 12:
                mask.save(debug_dir / f"{output_image.stem}_mask.png")
            accepted = False
            reject_reason = ""
            bbox_diff = 999.0
            background_diff = 0.0
            background_ssim_value = None
            lpips_value = None
            verification_passed = False
            verification_fail_reason = ""
            generated = original.copy()
            max_retries = int(ver_cfg["max_retries_per_image"])
            if dry_run:
                # Structure-check only: the copy is written so downstream dataset
                # wiring can be exercised, but it is explicitly marked as a
                # non-verified dry-run artifact (and the plan dir carries a marker).
                generated = original.copy()
                accepted = True
                reject_reason = "dry_run_copy"
                bbox_diff = 0.0
                background_diff = 0.0
                verification_fail_reason = "dry_run"
            else:
                for retry in range(max_retries + 1):
                    current_seed = attempt_seed + retry
                    try:
                        raw_generated = _run_inpaint(
                            pipe,
                            device,
                            original,
                            mask,
                            prompt,
                            negative_prompt,
                            diffusion_cfg,
                            current_seed,
                        )
                        # Protection violations must be measured before the paste:
                        # after paste_protected_regions the interior is original by
                        # construction and any check on it is vacuous.
                        bbox_diff = max_bbox_diff(original, raw_generated, original_boxes)
                        generated = paste_protected_regions(original, raw_generated, padded_boxes)
                        verification_passed, metrics = verify_pair(
                            original, generated, padded_boxes, ver_cfg, lpips_fn=lpips_fn
                        )
                        background_diff = metrics["background_pixel_diff"]
                        background_ssim_value = metrics["background_ssim"]
                        lpips_value = metrics["lpips"]
                        verification_fail_reason = metrics["verification_fail_reason"]
                        # Acceptance depends only on the final image: background
                        # actually changed AND the protected region is preserved
                        # (post-paste interior check inside verify_pair). The
                        # pre-paste bbox diff is a diagnostic only — paste_protected_regions
                        # restores the object + padding halo to the original, so a
                        # high value never harms the output. It is logged, not gated.
                        accepted = verification_passed
                        if bbox_diff > float(diffusion_cfg.get("bbox_diff_threshold", 18.0)):
                            print(
                                f"[INFO] 높은 pre-paste bbox diff({bbox_diff:.1f}) — 배경만 최종 반영되고 "
                                f"객체는 원본으로 복원되므로 승인에는 영향 없음: {output_image.name}"
                            )
                        reject_reason = "" if accepted else f"verification_failed:{verification_fail_reason}"
                        if accepted:
                            break
                    except Exception as exc:
                        reject_reason = f"inpaint_error:{type(exc).__name__}:{exc}"
                        accepted = False
            if accepted:
                accepted_count += 1
                generated.save(output_image, quality=95)
                copy_file(source_label, output_label, overwrite=True)
                if canonical_image_dir:
                    copy_file(output_image, canonical_output_image, overwrite=True)
                if canonical_label_dir:
                    copy_file(output_label, canonical_output_label, overwrite=True)
                if len(contact_rows) < 12:
                    contact_rows.append([original, mask.convert("RGB"), generated])
            else:
                rejected_path = rejected_dir / output_name
                generated.save(rejected_path, quality=90)
                if canonical_rejected_dir:
                    copy_file(rejected_path, canonical_rejected_dir / output_name, overwrite=True)
                # A rejected regeneration must also remove any stale file from the
                # train split (e.g. a dry-run copy that failed re-verification).
                for stale in (output_image, output_label, canonical_output_image, canonical_output_label):
                    if stale is not None and stale.exists():
                        stale.unlink()
            log_image = canonical_output_image if accepted else (canonical_rejected_dir / output_name if canonical_rejected_dir else rejected_dir / output_name)
            log_label = canonical_output_label if accepted else output_label
            label_count_output = len(read_yolo_labels(log_label)) if accepted and log_label.exists() else 0
            output_exists = log_image.exists()
            output_size_matches = False
            nontrivial = False
            if output_exists:
                try:
                    checked_image = load_rgb(log_image)
                    output_size_matches = checked_image.size == original.size
                    nontrivial = is_nontrivial_image(checked_image)
                except Exception:
                    output_size_matches = False
                    nontrivial = False
            logs.append(
                {
                    "source_image": str(source_image),
                    "output_image": str(log_image),
                    "class_id": class_id,
                    "class_name": row.get("class_name", f"class_{class_id}"),
                    "prompt": prompt,
                    "seed": attempt_seed,
                    "accepted": accepted,
                    "reject_reason": reject_reason,
                    "dry_run": dry_run,
                    "bbox_pixel_diff": bbox_diff,
                    "background_pixel_diff": background_diff,
                    "background_ssim": background_ssim_value,
                    "lpips": lpips_value,
                    "verification_passed": verification_passed,
                    "verification_fail_reason": verification_fail_reason,
                    "mask_padding": padding_ratio,
                    "inference_steps": diffusion_cfg.get("num_inference_steps", 20),
                    "label_count_original": len(labels),
                    "label_count_output": label_count_output,
                    "label_sanity_ok": accepted and len(labels) == label_count_output,
                    "output_exists": output_exists,
                    "output_size_matches": output_size_matches,
                    "nontrivial_image": nontrivial,
                }
            )
            timer.update()
            image_bar.update(1)
            image_bar.set_postfix_str(timer.status())
        if accepted_count < needed:
            print(
                f"[WARN] class {class_id}: 목표 budget {needed}장 중 {accepted_count}장만 검증 통과 "
                f"({max_attempts}회 시도 소진). 소스 다양성 부족/배경 여지 부족 가능. "
                "verification.budget_attempt_multiplier를 올리거나 소스를 늘리세요."
            )
    image_bar.close()
    if not dry_run and dry_run_marker.exists():
        dry_run_marker.unlink()
    log_df = pd.DataFrame(logs)
    log_path = synthetic_dir / f"generation_log_{plan_name}.csv"
    log_df.to_csv(log_path, index=False)
    if plan_name == "selective":
        log_df.to_csv(synthetic_dir / "generation_log.csv", index=False)
    if contact_rows:
        save_contact_sheet(contact_rows, synthetic_dir / f"review_sheet_{plan_name}.jpg", labels=["original", "mask", "generated"])
    print(f"[INFO] 생성 로그 저장: {log_path}")
    report_columns = [
        "source_image",
        "output_image",
        "class_id",
        "class_name",
        "prompt",
        "seed",
        "accepted",
        "background_pixel_diff",
        "background_ssim",
        "lpips",
        "bbox_pixel_diff",
        "verification_passed",
        "verification_fail_reason",
        "dry_run",
    ]
    if not log_df.empty:
        report_rows = log_df[[c for c in report_columns if c in log_df.columns]].to_dict("records")
        report_path = update_verification_report(outputs, plan_name, report_rows)
        print(f"[INFO] verification report 저장: {report_path}")
    if not dry_run:
        failure_rate = enforce_failure_rate(log_df, ver_cfg["max_failure_rate"], plan_name)
        print(f"[INFO] 생성 검증 실패율: {failure_rate:.1%} (허용 한도 {ver_cfg['max_failure_rate']:.1%})")
    return log_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate bbox-protected background inpainting images.")
    parser.add_argument("--data", required=True, help="Base data.yaml")
    parser.add_argument("--plan", required=True, help="Augmentation plan CSV")
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--out", default=None, help="Synthetic output root")
    parser.add_argument("--outputs", default=None, help="Experiment outputs root")
    parser.add_argument("--plan-name", default="selective")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Copy originals instead of running diffusion")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    processed = Path(cfg["paths"]["processed_data"])
    out = args.out or processed / "synthetic_inpaint"
    outputs = args.outputs or cfg["paths"]["outputs"]
    generate_from_plan(args.data, args.plan, out, outputs, cfg, plan_name=args.plan_name, force=args.force, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
