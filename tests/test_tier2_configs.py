"""Tier 2 확장(cell C2/C3/C4) config·설계 invariant 테스트.

- 세 config가 로드되고 variant 이름이 전부 파서에 등록되어 있는지
- MAR20 budget 500 / K=6 allocator invariant (84×2 + 83×4, cap 100 미적용)
- 축소 설계(weak arms 없음)의 confirmatory contrast family가 그대로 동작하는지
- RT-DETR cell의 batch 명시 / baseline_variants 축소가 설계 문서와 일치하는지
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.analyze_long_tail import _allocate_by_weights
from src.eval.confirmatory_stats import confirmatory_settings, evaluate_contrasts, seed_macro_table
from src.utils.detector import is_rtdetr_model
from src.utils.io import load_config
from src.utils.variants import parse_variant, uses_synthetic_plan

ROOT = Path(__file__).resolve().parents[1]
TIER2_CONFIGS = {
    "mad_rtdetr": ROOT / "configs" / "mad_rtdetr.yaml",
    "mar20_yolo": ROOT / "configs" / "mar20_yolo.yaml",
    "mar20_rtdetr": ROOT / "configs" / "mar20_rtdetr.yaml",
}


def _load(name: str) -> dict:
    return load_config(TIER2_CONFIGS[name], default_path=ROOT / "configs" / "default.yaml")


@pytest.mark.parametrize("name", sorted(TIER2_CONFIGS))
def test_config_loads_and_variants_known(name: str) -> None:
    cfg = _load(name)
    for variant in cfg["experiments"]["variants"]:
        parse_variant(variant)  # 미등록 이름이면 raise
    for variant in cfg["experiments"]["baseline_variants"]:
        parse_variant(variant)
    assert cfg["eval"]["split"] == "test"
    assert cfg["planning"]["split"] == "val"
    assert cfg["detector"]["seeds"] == [42, 43, 44]


@pytest.mark.parametrize("name", ["mad_rtdetr", "mar20_rtdetr"])
def test_rtdetr_cells_use_rtdetr_with_explicit_batch(name: str) -> None:
    cfg = _load(name)
    assert is_rtdetr_model(cfg["detector"]["model"])
    assert int(cfg["detector"]["batch"]) > 0, "RT-DETR은 auto-batch(-1) 금지"
    # 축소 설계: weak arms 없음 + real_only 재학습 없음 (하한선은 C1에 존재)
    assert cfg["experiments"]["baseline_variants"] == ["basic_aug"]
    assert set(cfg["experiments"]["variants"]) == {"aug_rfs", "aug_uniform_inpaint", "aug_selective_inpaint"}
    arms = cfg["confirmatory"]["arms"]
    assert set(arms) == {"tail_uniform", "tail_weighted"}
    assert not any(c.get("interaction") for c in cfg["confirmatory"]["primary_contrasts"])


def test_mar20_yolo_is_full_2x2() -> None:
    cfg = _load("mar20_yolo")
    assert set(cfg["confirmatory"]["arms"]) == {"tail_uniform", "tail_weighted", "weak_uniform", "weak_weighted"}
    assert any(c.get("interaction") for c in cfg["confirmatory"]["primary_contrasts"])
    assert len(cfg["confirmatory"]["primary_contrasts"]) == 5
    plans = {uses_synthetic_plan(v) for v in cfg["experiments"]["variants"]} - {None}
    assert plans == {"uniform", "selective", "weakness", "weakness_uniform"}


def test_mar20_budget_allocator_invariant() -> None:
    """B=500, K=6, cap 100 → uniform 배분은 84×2 + 83×4 (결정적)."""
    cfg = _load("mar20_yolo")
    sel = cfg["selective_generation"]
    assert int(sel["total_synthetic_budget"]) == 500
    assert int(sel["weakness_num_classes"]) == 6
    allocation = _allocate_by_weights(
        np.ones(6), int(sel["total_synthetic_budget"]), int(sel["min_per_class"]), int(sel["max_per_class"])
    )
    assert sum(allocation) == 500
    assert sorted(allocation, reverse=True) == [84, 84, 83, 83, 83, 83]
    # largest-remainder 동률은 낮은 인덱스 우선
    assert allocation == [84, 84, 83, 83, 83, 83]


def test_mar20_cells_share_pools_and_splits() -> None:
    yolo = _load("mar20_yolo")
    rtdetr = _load("mar20_rtdetr")
    # 변환/정규화/합성 pool 공유, 실험 데이터셋·outputs 는 분리
    assert yolo["paths"]["raw_data"] == rtdetr["paths"]["raw_data"]
    assert yolo["paths"]["processed_data"] == rtdetr["paths"]["processed_data"]
    assert yolo["paths"]["synthetic_data"] == rtdetr["paths"]["synthetic_data"]
    assert yolo["paths"]["experiments_data"] != rtdetr["paths"]["experiments_data"]
    assert yolo["paths"]["outputs"] != rtdetr["paths"]["outputs"]
    # split 규칙과 diffusion 설정(프롬프트 포함)은 문자 그대로 동일해야 pool 공유가 성립
    assert yolo["dataset"] == rtdetr["dataset"]
    assert yolo["diffusion"] == rtdetr["diffusion"]
    assert yolo["verification"] == rtdetr["verification"]
    assert yolo["detector"]["seeds"][0] == rtdetr["detector"]["seeds"][0]  # 생성 seed 공식의 기준


def test_mad_rtdetr_reuses_confirmatory_design_constants() -> None:
    """C2는 C1(confirmatory)과 tail/diffusion 설계가 동일해야 pool 재사용이 성립."""
    c1 = load_config(ROOT / "configs" / "confirmatory.yaml", default_path=ROOT / "configs" / "default.yaml")
    c2 = _load("mad_rtdetr")
    assert c1["diffusion"] == c2["diffusion"]
    assert c1["verification"] == c2["verification"]
    assert c1["tail"] == c2["tail"]
    assert c1["selective_generation"]["total_synthetic_budget"] == c2["selective_generation"]["total_synthetic_budget"]
    assert c1["selective_generation"]["weakness_class_ids"] == c2["selective_generation"]["weakness_class_ids"]
    assert c1["paths"]["synthetic_data"] != c2["paths"]["synthetic_data"], "C1 산출물 디렉터리 오염 금지 — 시딩 복사로 재사용"


def test_reduced_contrast_family_evaluates_without_weak_arms() -> None:
    """C2/C4 축소 설계(weak arms 없음)에서 confirmatory 통계가 그대로 동작한다."""
    cfg = _load("mad_rtdetr")
    settings = confirmatory_settings(cfg)
    arms = ["basic_aug", "aug_uniform_inpaint", "aug_selective_inpaint"]
    rows = []
    for arm in arms:
        for i, seed in enumerate((42, 43, 44)):
            for class_id, scope_ids in ((0, "tail"), (1, "tail"), (2, "other")):
                gain = 0.10 if arm != "basic_aug" and scope_ids == "tail" else 0.0
                rows.append(
                    {"experiment": arm, "seed": seed, "class_id": class_id,
                     "ap50_95": 0.5 + gain + 0.001 * i, "eval_split": "test"}
                )
    per_class = pd.DataFrame(rows)
    scopes = {"all": None, "tail": {0, 1}}
    macro = seed_macro_table(per_class, scopes, arms)
    contrasts = evaluate_contrasts(macro, settings)
    assert list(contrasts["contrast"]) == [
        "tail_arms_vs_baseline_in_tail",
        "uniform_vs_baseline_all",
        "weighted_vs_uniform_in_tail",
    ]
    assert (contrasts["n_seeds"] == 3).all()
    assert "p_holm" in contrasts.columns
    tail_row = contrasts.set_index("contrast").loc["tail_arms_vs_baseline_in_tail"]
    assert tail_row["estimate"] == pytest.approx(0.10, abs=1e-9)
