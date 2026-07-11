from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw
from src.augment.build_experiment_datasets import build_experiment_datasets
from src.augment.inpaint_background import generate_from_plan
from src.data.analyze_long_tail import analyze_long_tail, build_augmentation_plans
from src.data.normalize_yolo_dataset import normalize_dataset


def _write_tiny_yolo_dataset(root: Path) -> None:
    classes = ["head_jet", "mid_jet", "tail_jet"]
    counts = {"train": [6, 3, 2], "valid": [2, 1, 1], "test": [2, 1, 1]}
    for split in counts:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    for split, split_counts in counts.items():
        idx = 0
        for class_id, count in enumerate(split_counts):
            for _ in range(count):
                image = Image.new("RGB", (96, 64), (120 + class_id * 20, 160, 210))
                draw = ImageDraw.Draw(image)
                draw.rectangle((36, 24, 60, 40), fill=(30, 30 + class_id * 50, 30))
                name = f"{split}_c{class_id}_{idx}.jpg"
                image.save(root / "images" / split / name)
                (root / "labels" / split / name.replace(".jpg", ".txt")).write_text(
                    f"{class_id} 0.5 0.5 0.25 0.25\n",
                    encoding="utf-8",
                )
                idx += 1
    (root / "data.yaml").write_text(
        "\n".join(
            [
                f"path: {root}",
                "train: images/train",
                "val: images/valid",
                "test: images/test",
                f"nc: {len(classes)}",
                "names:",
                *[f"  - {name}" for name in classes],
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_analysis_dry_run_inpaint_and_dataset_build(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_tiny_yolo_dataset(raw)
    outputs = tmp_path / "outputs"
    processed = tmp_path / "processed"
    experiments = tmp_path / "experiments"
    config = {
        "paths": {
            "processed_data": str(processed),
            "experiments_data": str(experiments),
            "outputs": str(outputs),
        },
        "detector": {"seeds": [42]},
        "tail": {"method": "bottom_percent", "bottom_percent": 0.34, "min_val_instances": 1},
        "selective_generation": {"alpha": 0.6, "total_synthetic_budget": 4, "min_per_class": 1, "max_per_class": 3},
        "diffusion": {
            "mask_padding_ratio": 0.08,
            "mask_blur_radius": 2,
            "num_inference_steps": 1,
            "prompts": ["sky"],
            "negative_prompt": "bad",
        },
        "rfs": {"threshold": 0.5},
        "experiments": {
            "variants": [
                "real_only",
                "basic_aug",
                "aug_oversample",
                "aug_rfs",
                "aug_copy_paste",
                "aug_uniform_inpaint",
                "aug_selective_inpaint",
            ]
        },
    }

    data_yaml = normalize_dataset(raw, processed / "base", seed=42)
    grouped = analyze_long_tail(data_yaml, config, outputs)
    uniform_plan, selective_plan = build_augmentation_plans(grouped, None, config["selective_generation"], outputs)

    synthetic_root = processed / "synthetic_inpaint"
    generate_from_plan(data_yaml, uniform_plan, synthetic_root, outputs, config, plan_name="uniform", dry_run=True)
    generate_from_plan(data_yaml, selective_plan, synthetic_root, outputs, config, plan_name="selective", dry_run=True)
    experiment_yamls = build_experiment_datasets(
        data_yaml,
        experiments,
        uniform_plan=uniform_plan,
        selective_plan=selective_plan,
        synthetic_root=synthetic_root,
        variants=config["experiments"]["variants"],
        config=config,
    )

    assert (outputs / "analysis" / "dataset_summary.csv").exists()
    assert (outputs / "synthetic" / "generation_log.csv").exists()
    log = pd.read_csv(outputs / "synthetic" / "generation_log.csv")
    assert set(["label_sanity_ok", "output_exists", "output_size_matches", "nontrivial_image"]).issubset(log.columns)
    assert bool(log["label_sanity_ok"].all())
    assert len(list((synthetic_root / "images" / "train").glob("*.jpg"))) == int(
        pd.read_csv(selective_plan)["num_synthetic_images"].sum()
    )
    assert set(experiment_yamls) == set(config["experiments"]["variants"])
    for yaml_path in experiment_yamls.values():
        assert yaml_path.exists()
