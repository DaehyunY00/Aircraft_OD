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
import yaml
from tqdm import tqdm

from src.augment.copy_paste_tail import copy_paste_from_plan
from src.augment.oversample_tail import oversample_from_plan
from src.augment.repeat_factor_sampling import apply_rfs, rfs_config
from src.utils.io import copy_or_symlink, ensure_dir, load_config
from src.utils.variants import parse_variant, uses_synthetic_plan
from src.utils.yolo import create_data_yaml, label_path_for_image, list_images, normalize_class_names


def read_data_yaml(data_yaml: str | Path) -> dict[str, Any]:
    data_yaml = Path(data_yaml)
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(data.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    data["_root"] = root
    return data


def split_dirs(data_yaml: str | Path, split: str) -> tuple[Path, Path]:
    data = read_data_yaml(data_yaml)
    split_value = data.get(split)
    if split_value is None and split == "val":
        split_value = data.get("valid", "images/val")
    if split_value is None and split == "test":
        split_value = data.get("val", "images/val")
    images_dir = Path(split_value)
    if not images_dir.is_absolute():
        images_dir = data["_root"] / images_dir
    labels_dir = Path(str(images_dir).replace("/images/", "/labels/"))
    if not labels_dir.exists():
        labels_dir = data["_root"] / "labels" / split
    return images_dir, labels_dir


def copy_split(
    source_images: Path,
    source_labels: Path,
    dest_root: Path,
    split: str,
    overwrite: bool = False,
) -> int:
    ensure_dir(dest_root / "images" / split)
    ensure_dir(dest_root / "labels" / split)
    count = 0
    for image_path in tqdm(list_images(source_images), desc=f"copy {dest_root.name}/{split}"):
        label_path = label_path_for_image(image_path, source_images, source_labels)
        rel = image_path.relative_to(source_images)
        copy_or_symlink(image_path, dest_root / "images" / split / rel, overwrite=overwrite)
        if label_path.exists():
            copy_or_symlink(label_path, dest_root / "labels" / split / rel.with_suffix(".txt"), overwrite=overwrite)
        count += 1
    return count


def add_synthetic_split(
    synthetic_root: Path,
    dest_root: Path,
    overwrite: bool = False,
    include_names: set[str] | None = None,
    exclude_names: set[str] | None = None,
) -> int:
    """Link a synthetic plan's train images into a variant dataset.

    include_names restricts to the given file names (generation-log filtering);
    exclude_names removes specific files on top of that (quality filtering).
    """
    images_dir = synthetic_root / "images" / "train"
    labels_dir = synthetic_root / "labels" / "train"
    if not images_dir.exists():
        return 0
    count = 0
    for image_path in tqdm(list_images(images_dir), desc=f"add synthetic {synthetic_root.name}"):
        if include_names is not None and image_path.name not in include_names:
            continue
        if exclude_names is not None and image_path.name in exclude_names:
            continue
        label_path = label_path_for_image(image_path, images_dir, labels_dir)
        copy_or_symlink(image_path, dest_root / "images" / "train" / image_path.name, overwrite=overwrite)
        if label_path.exists():
            copy_or_symlink(label_path, dest_root / "labels" / "train" / label_path.name, overwrite=overwrite)
        count += 1
    return count


def quality_kept_names(quality_filter_csv: Path) -> set[str] | None:
    """File names kept by the CLIPScore quality filter, or None when absent."""
    if not quality_filter_csv.exists():
        return None
    df = pd.read_csv(quality_filter_csv)
    if "kept" not in df.columns or "image" not in df.columns:
        return None
    return {Path(str(p)).name for p in df.loc[df["kept"].astype(bool), "image"]}


def object_gate_dropped_names(gate_csv: Path) -> set[str] | None:
    """Object-level gate가 기각한 파일 이름. 없으면 None.

    게이트는 생성물 전량을 검사하므로(품질 필터와 달리 표본이 아님) 기각 목록이
    곧 전체 판정이다. 그래도 kept==False 만 뽑아 제외하는 방식은 동일하게 쓴다 —
    새 파일이 나중에 추가돼도 목록에 없다는 이유로 통째로 빠지지 않게 하기 위함.
    """
    if not gate_csv.exists():
        return None
    df = pd.read_csv(gate_csv)
    if "kept" not in df.columns or "image" not in df.columns:
        return None
    return {Path(str(p)).name for p in df.loc[~df["kept"].astype(bool), "image"]}


def quality_dropped_names(quality_filter_csv: Path) -> set[str] | None:
    """File names explicitly DROPPED by the quality filter, or None when absent.

    The filter only scores a sample of the accepted images (synthetic_quality.
    max_images). Excluding the dropped ones — instead of intersecting with the
    kept ones — keeps every unscored image in the _qf dataset, so the variant's
    budget does not collapse to the scored sample.
    """
    if not quality_filter_csv.exists():
        return None
    df = pd.read_csv(quality_filter_csv)
    if "kept" not in df.columns or "image" not in df.columns:
        return None
    return {Path(str(p)).name for p in df.loc[~df["kept"].astype(bool), "image"]}


def accepted_names_from_logs(generation_log_dir: Path, plan_name: str) -> set[str] | None:
    """Accepted output-image names from the current generation logs (plan + refill).

    Plan allocations can shrink between runs (e.g. a max_per_class change): files
    from an older, larger allocation then linger in the plan directory. Copying
    only what the current log accepted keeps the realized budget equal to the
    plan and immune to stale leftovers. Returns None when no log exists
    (dataset built outside the pipeline).
    """
    names: set[str] = set()
    found = False
    for log_name in (f"generation_log_{plan_name}.csv", f"generation_log_{plan_name}_refill.csv"):
        log_path = generation_log_dir / log_name
        if not log_path.exists():
            continue
        log = pd.read_csv(log_path)
        if "accepted" not in log.columns or "output_image" not in log.columns:
            continue
        found = True
        # dry-run rows are accepted=True structural placeholders (original
        # copies) — training on them silently invalidates the experiment.
        if "dry_run" in log.columns:
            log = log[~log["dry_run"].astype(bool)]
        accepted = log[log["accepted"].astype(bool)]
        names |= {Path(str(p)).name for p in accepted["output_image"]}
    return names if found else None


def build_experiment_datasets(
    base_data_yaml: str | Path,
    experiments_root: str | Path,
    class_names: list[str] | None = None,
    uniform_plan: str | Path | None = None,
    selective_plan: str | Path | None = None,
    synthetic_root: str | Path | None = None,
    variants: list[str] | None = None,
    overwrite: bool = False,
    quality_filter_dir: str | Path | None = None,
    config: dict[str, Any] | None = None,
    generation_log_dir: str | Path | None = None,
) -> dict[str, Path]:
    data = read_data_yaml(base_data_yaml)
    class_names = class_names or normalize_class_names(data.get("names"), data.get("nc"))
    experiments_root = ensure_dir(experiments_root)
    synthetic_root = Path(synthetic_root) if synthetic_root else data["_root"].parent / "synthetic_inpaint"
    variants = variants or ["real_only", "basic_aug", "aug_oversample", "aug_uniform_inpaint", "aug_selective_inpaint"]

    train_images, train_labels = split_dirs(base_data_yaml, "train")
    val_images, val_labels = split_dirs(base_data_yaml, "val")
    test_images, test_labels = split_dirs(base_data_yaml, "test")
    outputs: dict[str, Path] = {}

    for variant in variants:
        spec = parse_variant(variant)
        variant_root = ensure_dir(experiments_root / variant)
        copy_split(train_images, train_labels, variant_root, "train", overwrite=overwrite)
        copy_split(val_images, val_labels, variant_root, "val", overwrite=overwrite)
        copy_split(test_images, test_labels, variant_root, "test", overwrite=overwrite)

        plan_name = uses_synthetic_plan(variant)
        seed = int((config or {}).get("detector", {}).get("seeds", [42])[0])
        if spec.base in ("aug_copy_paste", "aug_oversample") and not selective_plan:
            # Without a plan these branches would fall through and train a
            # silent basic_aug clone under a tail-technique name.
            raise ValueError(f"{variant}: augmentation plan(selective_plan)이 없어 데이터셋을 만들 수 없습니다.")
        if spec.base == "aug_copy_paste":
            created = copy_paste_from_plan(
                variant_root / "images" / "train",
                variant_root / "labels" / "train",
                selective_plan,
                variant_root / "images" / "train",
                variant_root / "labels" / "train",
                config=config,
                seed=seed,
                overwrite=overwrite,
            )
            if created == 0:
                raise RuntimeError(f"{variant}: copy-paste 샘플이 0장 생성됐습니다 — basic_aug와 동일한 무효 실험이 됩니다.")
            print(f"[INFO] {variant} 추가 샘플 수: {created}")
        elif spec.base == "aug_rfs":
            created = apply_rfs(
                variant_root / "images" / "train",
                variant_root / "labels" / "train",
                variant_root / "images" / "train",
                variant_root / "labels" / "train",
                threshold=rfs_config(config or {})["threshold"],
                seed=seed,
                overwrite=overwrite,
            )
            if created == 0:
                raise RuntimeError(
                    f"{variant}: RFS 복제가 0장입니다 (threshold가 모든 클래스 빈도보다 낮음) — "
                    "basic_aug와 동일한 무효 실험이 됩니다. rfs.threshold를 올리거나 variant를 제외하세요."
                )
            print(f"[INFO] {variant} RFS 복제 수: {created}")
        elif spec.base == "aug_oversample":
            created = oversample_from_plan(
                variant_root / "images" / "train",
                variant_root / "labels" / "train",
                selective_plan,
                variant_root / "images" / "train",
                variant_root / "labels" / "train",
                overwrite=overwrite,
            )
            if created == 0:
                raise RuntimeError(f"{variant}: oversampling 샘플이 0장 생성됐습니다 — basic_aug와 동일한 무효 실험이 됩니다.")
            print(f"[INFO] {variant} 추가 샘플 수: {created}")
        elif plan_name:
            dry_run_marker = synthetic_root / plan_name / "DRY_RUN_MARKER.txt"
            if dry_run_marker.exists():
                raise RuntimeError(
                    f"{variant}: {dry_run_marker}가 존재합니다 — 이 plan 디렉터리는 --dry-run-inpaint가 "
                    "만든 원본 사본입니다. --skip-inpaint/--only-train 없이 실제 생성을 먼저 실행하세요."
                )
            log_names = (
                accepted_names_from_logs(Path(generation_log_dir), plan_name)
                if generation_log_dir is not None
                else None
            )
            exclude: set[str] | None = None
            if spec.quality_filter:
                if quality_filter_dir is None:
                    raise ValueError(
                        f"{variant}: quality-filtered variant인데 quality_filter_dir가 없습니다. "
                        "먼저 synthetic quality 채점(quality_filter.enabled)을 실행하세요."
                    )
                exclude = quality_dropped_names(Path(quality_filter_dir) / f"quality_filter_{plan_name}.csv")
                if exclude is None:
                    raise FileNotFoundError(
                        f"{variant}: {quality_filter_dir}/quality_filter_{plan_name}.csv가 없거나 "
                        "kept 컬럼이 없습니다. 품질 채점/필터링 단계를 먼저 실행하세요."
                    )
            if spec.object_gate:
                gate_csv = Path(generation_log_dir or synthetic_root) / f"object_gate_{plan_name}.csv"
                gate_drop = object_gate_dropped_names(gate_csv)
                if gate_drop is None:
                    raise FileNotFoundError(
                        f"{variant}: {gate_csv}가 없거나 kept/image 컬럼이 없습니다. "
                        "먼저 src/eval/audit_hallucination.py --gate 를 실행하세요."
                    )
                # 두 필터가 함께 걸리면(_qf_og) 합집합을 제외한다.
                exclude = gate_drop if exclude is None else (exclude | gate_drop)
                print(f"[INFO] {variant}: object gate가 {len(gate_drop)}장 제외")
            added = add_synthetic_split(
                synthetic_root / plan_name, variant_root, overwrite=overwrite,
                include_names=log_names, exclude_names=exclude,
            )
            if spec.quality_filter:
                # budget refill: images regenerated to replace the filtered-out ones
                added += add_synthetic_split(
                    synthetic_root / f"{plan_name}_refill", variant_root, overwrite=overwrite,
                    include_names=log_names,
                )
            if added == 0:
                # Training an inpaint variant on base-only data silently equals
                # basic_aug and corrupts the comparison — refuse instead.
                raise RuntimeError(
                    f"{variant}: synthetic 이미지가 0장 추가됐습니다 ({synthetic_root / plan_name}). "
                    "세션 재시작으로 생성물이 사라졌을 수 있습니다. --skip-inpaint 없이 재실행해 "
                    "생성부터 다시 하거나, paths.synthetic_data를 Drive 경로로 설정해 보존하세요."
                )
            print(f"[INFO] {variant} synthetic 추가 수: {added}")

        outputs[variant] = create_data_yaml(variant_root, class_names, variant_root / "data.yaml")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build experiment dataset variants.")
    parser.add_argument("--base-data", required=True)
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--experiments-root", default=None)
    parser.add_argument("--uniform-plan", default=None)
    parser.add_argument("--selective-plan", default=None)
    parser.add_argument("--synthetic-root", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    build_experiment_datasets(
        args.base_data,
        args.experiments_root or cfg["paths"]["experiments_data"],
        uniform_plan=args.uniform_plan,
        selective_plan=args.selective_plan,
        synthetic_root=args.synthetic_root,
        variants=cfg.get("experiments", {}).get("variants"),
        overwrite=args.overwrite,
        config=cfg,
    )


if __name__ == "__main__":
    main()
