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

from src.augment.masks import create_inpainting_mask
from src.utils.image import is_nontrivial_image, load_rgb, max_bbox_diff, paste_protected_regions, save_contact_sheet
from src.utils.io import copy_file, ensure_dir, load_config
from src.utils.seed import set_seed
from src.utils.timing import ProgressTimer
from src.utils.yolo import label_path_for_image, list_images, read_yolo_labels


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
    work_mask = mask.resize(work_size, Image.Resampling.LANCZOS)
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
    seed = int(config.get("detector", {}).get("seeds", [42])[0])
    set_seed(seed)
    plan = pd.read_csv(plan_csv)
    source_by_class = collect_source_images_by_class(data_yaml)
    out_root = Path(out_root)
    image_out_dir = ensure_dir(out_root / plan_name / "images" / "train")
    label_out_dir = ensure_dir(out_root / plan_name / "labels" / "train")
    rejected_dir = ensure_dir(out_root / plan_name / "rejected")
    canonical_image_dir = ensure_dir(out_root / "images" / "train") if plan_name == "selective" else None
    canonical_label_dir = ensure_dir(out_root / "labels" / "train") if plan_name == "selective" else None
    canonical_rejected_dir = ensure_dir(out_root / "rejected") if plan_name == "selective" else None
    synthetic_dir = ensure_dir(Path(outputs) / "synthetic")
    debug_dir = ensure_dir(synthetic_dir / "debug_masks" / plan_name)

    pipe = None
    device = "cpu"
    if not dry_run:
        pipe, device = load_inpaint_pipeline(diffusion_cfg)

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
        sources = source_by_class.get(class_id, [])
        if needed <= 0 or not sources:
            continue
        for idx in range(needed):
            source_image, source_label = sources[idx % len(sources)]
            prompt = prompts[idx % len(prompts)]
            attempt_seed = seed + class_id * 100_000 + idx
            output_name = f"{source_image.stem}_{plan_name}_c{class_id}_{idx:04d}_s{attempt_seed}.jpg"
            output_image = image_out_dir / output_name
            output_label = label_out_dir / output_image.with_suffix(".txt").name
            canonical_output_image = canonical_image_dir / output_name if canonical_image_dir else output_image
            canonical_output_label = canonical_label_dir / output_label.name if canonical_label_dir else output_label
            if output_image.exists() and output_label.exists() and not force:
                labels = read_yolo_labels(source_label)
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
                        "reject_reason": "already_exists",
                        "bbox_pixel_diff": 0.0,
                        "mask_padding": diffusion_cfg.get("mask_padding_ratio", 0.1),
                        "inference_steps": diffusion_cfg.get("num_inference_steps", 20),
                        "label_count_original": len(labels),
                        "label_count_output": label_count_output,
                        "label_sanity_ok": len(labels) == label_count_output,
                        "output_exists": canonical_output_image.exists(),
                        "output_size_matches": True,
                        "nontrivial_image": True,
                    }
                )
                timer.update()
                image_bar.update(1)
                image_bar.set_postfix_str(timer.status())
                continue
            original = load_rgb(source_image)
            labels = read_yolo_labels(source_label)
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
            generated = original.copy()
            max_retries = int(diffusion_cfg.get("max_retries_per_image", 2))
            for retry in range(max_retries + 1):
                current_seed = attempt_seed + retry
                try:
                    if dry_run:
                        generated = original.copy()
                    else:
                        generated = _run_inpaint(
                            pipe,
                            device,
                            original,
                            mask,
                            prompt,
                            negative_prompt,
                            diffusion_cfg,
                            current_seed,
                        )
                    generated = paste_protected_regions(original, generated, padded_boxes)
                    bbox_diff = max_bbox_diff(original, generated, original_boxes)
                    valid = generated.size == original.size and is_nontrivial_image(generated)
                    accepted = valid and bbox_diff <= float(diffusion_cfg.get("bbox_diff_threshold", 18.0))
                    reject_reason = "" if accepted else "bbox_or_image_quality_failed"
                    if accepted:
                        break
                except Exception as exc:
                    reject_reason = f"inpaint_error:{type(exc).__name__}:{exc}"
                    accepted = False
            if accepted:
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
                    "bbox_pixel_diff": bbox_diff,
                    "mask_padding": diffusion_cfg.get("mask_padding_ratio", 0.1),
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
    image_bar.close()
    log_path = synthetic_dir / f"generation_log_{plan_name}.csv"
    pd.DataFrame(logs).to_csv(log_path, index=False)
    if plan_name == "selective":
        pd.DataFrame(logs).to_csv(synthetic_dir / "generation_log.csv", index=False)
    if contact_rows:
        save_contact_sheet(contact_rows, synthetic_dir / f"review_sheet_{plan_name}.jpg", labels=["original", "mask", "generated"])
    print(f"[INFO] 생성 로그 저장: {log_path}")
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
