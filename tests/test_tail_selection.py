import numpy as np
import pandas as pd

from src.data.analyze_long_tail import (
    _allocate_by_weights,
    assign_class_groups,
    build_augmentation_plans,
    compute_priority_scores,
)


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
    plans = build_augmentation_plans(
        grouped,
        None,
        {"alpha": 0.6, "total_synthetic_budget": 6, "min_per_class": 1, "max_per_class": 4},
        tmp_path,
    )
    assert set(plans) == {"uniform", "selective", "weakness", "weakness_uniform"}
    # Every plan spends the same budget: only the class set and the allocation
    # weights differ, so budget can never confound the comparison.
    for name, path in plans.items():
        df = pd.read_csv(path)
        assert df["num_synthetic_images"].sum() == 6, name
        assert (df["num_synthetic_images"] >= 1).all(), name
        assert (df["num_synthetic_images"] <= 4).all(), name


def test_weak_plans_share_class_set(tmp_path) -> None:
    grouped = pd.DataFrame(
        [
            {"class_id": cid, "class_name": f"c{cid}", "instance_count": 100 + cid,
             "image_count": 50, "val_instance_count": 5, "group": "tail" if cid < 3 else "medium"}
            for cid in range(6)
        ]
    )
    ap = pd.DataFrame([{"class_id": cid, "ap50_95": 0.9 - 0.1 * cid} for cid in range(6)])
    plans = build_augmentation_plans(
        grouped,
        ap,
        {"alpha": 0.6, "total_synthetic_budget": 9, "min_per_class": 1, "max_per_class": 9,
         "weakness_num_classes": 3},
        tmp_path,
    )
    weak = pd.read_csv(plans["weakness"])
    weak_uniform = pd.read_csv(plans["weakness_uniform"])
    assert set(weak["class_id"]) == set(weak_uniform["class_id"])
    # weakest AP classes are 5, 4, 3 with the descending AP above
    assert set(weak["class_id"].astype(int)) == {3, 4, 5}


def test_pinned_weakness_class_ids_override_ranking(tmp_path) -> None:
    grouped = pd.DataFrame(
        [
            {"class_id": cid, "class_name": f"c{cid}", "instance_count": 100 + cid,
             "image_count": 50, "val_instance_count": 5, "group": "tail" if cid < 2 else "medium"}
            for cid in range(6)
        ]
    )
    ap = pd.DataFrame([{"class_id": cid, "ap50_95": 0.9 - 0.1 * cid} for cid in range(6)])
    plans = build_augmentation_plans(
        grouped,
        ap,
        {"alpha": 0.6, "total_synthetic_budget": 9, "min_per_class": 1, "max_per_class": 9,
         "weakness_num_classes": 3, "weakness_class_ids": [1, 2, 4]},
        tmp_path,
    )
    for plan in ("weakness", "weakness_uniform"):
        df = pd.read_csv(plans[plan])
        assert set(df["class_id"].astype(int)) == {1, 2, 4}  # pinned, not the AP-ranked {3,4,5}
        assert df["num_synthetic_images"].sum() == 9
    import pytest

    with pytest.raises(ValueError):
        build_augmentation_plans(
            grouped, ap,
            {"alpha": 0.6, "total_synthetic_budget": 9, "min_per_class": 1, "max_per_class": 9,
             "weakness_num_classes": 3, "weakness_class_ids": [1, 2, 99]},
            tmp_path,
        )


def test_allocator_uniform_largest_remainder() -> None:
    # The confirmatory-design invariant: uniform, B=1000, K=13, min=5, max=200
    # must give twelve classes 77 images and one class 76 — never 88/76.
    allocation = _allocate_by_weights(np.ones(13), 1000, 5, 200)
    assert sum(allocation) == 1000
    assert sorted(allocation) == [76] + [77] * 12
    # Equal remainders break toward the lower index, so the single 76 sits last.
    assert allocation == [77] * 12 + [76]


def test_allocator_sum_bounds_and_determinism() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = int(rng.integers(1, 20))
        weights = rng.random(n)
        budget = int(rng.integers(1, 500))
        min_pc = int(rng.integers(0, 4))
        max_pc = int(rng.integers(max(1, min_pc), 60))
        allocation = _allocate_by_weights(weights.copy(), budget, min_pc, max_pc)
        again = _allocate_by_weights(weights.copy(), budget, min_pc, max_pc)
        assert allocation == again  # deterministic
        assert all(a <= max_pc for a in allocation)
        assert sum(allocation) <= budget
        if budget >= min_pc * n:
            assert all(a >= min(min_pc, max_pc) for a in allocation)
        if budget <= max_pc * n:
            assert sum(allocation) == budget  # spends the full budget when caps allow


def test_allocator_caps_redistribute() -> None:
    # One dominant weight hits its cap; the surplus must flow to the others
    # instead of being dropped.
    allocation = _allocate_by_weights(np.array([100.0, 1.0, 1.0]), 30, 0, 20)
    assert sum(allocation) == 30
    assert allocation[0] == 20
    assert allocation[1] + allocation[2] == 10
    # Deterministic tie-break: equal weights and equal remainders favor index 1.
    assert allocation[1] >= allocation[2]
