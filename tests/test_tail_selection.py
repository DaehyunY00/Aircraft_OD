import pandas as pd

from src.data.analyze_long_tail import assign_class_groups, build_augmentation_plans, compute_priority_scores


def test_rare_low_ap_class_gets_highest_priority() -> None:
    stats = pd.DataFrame(
        [
            {"class_id": 0, "class_name": "head", "instance_count": 1000, "image_count": 500, "group": "tail"},
            {"class_id": 1, "class_name": "rare_good", "instance_count": 20, "image_count": 18, "group": "tail"},
            {"class_id": 2, "class_name": "rare_bad", "instance_count": 18, "image_count": 16, "group": "tail"},
        ]
    )
    ap = pd.DataFrame(
        [
            {"class_id": 0, "ap50_95": 0.8},
            {"class_id": 1, "ap50_95": 0.7},
            {"class_id": 2, "ap50_95": 0.1},
        ]
    )
    scored = compute_priority_scores(stats, ap, alpha=0.6)
    top = scored.sort_values("priority_score", ascending=False).iloc[0]
    assert int(top["class_id"]) == 2


def test_tail_group_bottom_percent() -> None:
    stats = pd.DataFrame(
        [
            {"class_id": 0, "class_name": "c0", "instance_count": 100, "image_count": 50, "val_instance_count": 10},
            {"class_id": 1, "class_name": "c1", "instance_count": 50, "image_count": 25, "val_instance_count": 10},
            {"class_id": 2, "class_name": "c2", "instance_count": 5, "image_count": 5, "val_instance_count": 10},
        ]
    )
    grouped = assign_class_groups(stats, {"method": "bottom_percent", "bottom_percent": 0.34, "min_val_instances": 1})
    assert grouped.loc[grouped["class_id"] == 2, "group"].item() == "tail"


def test_build_augmentation_plans(tmp_path) -> None:
    grouped = pd.DataFrame(
        [
            {
                "class_id": 1,
                "class_name": "tail_a",
                "instance_count": 10,
                "image_count": 8,
                "val_instance_count": 3,
                "group": "tail",
            },
            {
                "class_id": 2,
                "class_name": "tail_b",
                "instance_count": 20,
                "image_count": 12,
                "val_instance_count": 3,
                "group": "tail",
            },
        ]
    )
    uniform, selective = build_augmentation_plans(
        grouped,
        None,
        {"alpha": 0.6, "total_synthetic_budget": 6, "min_per_class": 1, "max_per_class": 4},
        tmp_path,
    )
    uniform_df = pd.read_csv(uniform)
    selective_df = pd.read_csv(selective)
    assert uniform_df["num_synthetic_images"].sum() == 6
    assert selective_df["num_synthetic_images"].sum() == 6
