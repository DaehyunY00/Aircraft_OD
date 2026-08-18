import json
import math

import numpy as np
import pandas as pd
import pytest

from src.eval.confirmatory_stats import (
    confirmatory_settings,
    evaluate_contrasts,
    freeze_analysis_plan,
    holm_adjust,
    run_confirmatory_stats,
    seed_macro_table,
)

CONFIG = {
    "confirmatory": {
        "baseline": "basic_aug",
        "arms": {
            "tail_uniform": "aug_uniform_inpaint",
            "tail_weighted": "aug_selective_inpaint",
            "weak_uniform": "aug_weakuniform_inpaint",
            "weak_weighted": "aug_weakness_inpaint",
        },
        "confidence": 0.95,
        "primary_contrasts": [
            {"name": "tail_vs_base_tail", "scope": "tail",
             "plus": ["aug_uniform_inpaint", "aug_selective_inpaint"], "minus": ["basic_aug"]},
            {"name": "interaction", "interaction": True},
            {"name": "weighted_vs_uniform_weak", "scope": "weak",
             "plus": ["aug_weakness_inpaint"], "minus": ["aug_weakuniform_inpaint"]},
        ],
    }
}

ARMS = ["basic_aug", "aug_uniform_inpaint", "aug_selective_inpaint", "aug_weakuniform_inpaint", "aug_weakness_inpaint"]
TAIL_IDS = [0, 1]
WEAK_IDS = [2, 3]


def _per_class(effects: dict[str, dict[int, float]], seeds=(42, 43, 44)) -> pd.DataFrame:
    """Per-class AP frame: base 0.5 + arm/class effect + tiny seed offset."""
    rows = []
    for arm in ARMS:
        for seed_index, seed in enumerate(seeds):
            for class_id in TAIL_IDS + WEAK_IDS:
                ap = 0.5 + effects.get(arm, {}).get(class_id, 0.0) + 0.001 * seed_index
                rows.append(
                    {"experiment": arm, "seed": seed, "class_id": class_id,
                     "ap50_95": ap, "eval_split": "test"}
                )
    return pd.DataFrame(rows)


def _scopes():
    return {"all": None, "tail": set(TAIL_IDS), "weak": set(WEAK_IDS)}


def test_interaction_detects_targeted_gains() -> None:
    # tail arms help only tail classes (+0.10), weak arms help only weak classes (+0.10)
    effects = {
        "aug_uniform_inpaint": {0: 0.10, 1: 0.10},
        "aug_selective_inpaint": {0: 0.10, 1: 0.10},
        "aug_weakuniform_inpaint": {2: 0.10, 3: 0.10},
        "aug_weakness_inpaint": {2: 0.10, 3: 0.10},
    }
    macro = seed_macro_table(_per_class(effects), _scopes(), ARMS)
    contrasts = evaluate_contrasts(macro, confirmatory_settings(CONFIG))
    interaction = contrasts[contrasts["contrast"] == "interaction"].iloc[0]
    # (weak−tail arms in weak scope)=+0.10−(−0.10)? tail arms don't touch weak → 0.
    # I = (0.10 − 0) − (0 − 0.10) = 0.20
    assert interaction["n_seeds"] == 3
    assert interaction["estimate"] == pytest.approx(0.20, abs=1e-9)
    tail_contrast = contrasts[contrasts["contrast"] == "tail_vs_base_tail"].iloc[0]
    assert tail_contrast["estimate"] == pytest.approx(0.10, abs=1e-9)


def test_single_seed_arm_never_compared() -> None:
    effects: dict[str, dict[int, float]] = {}
    per_class = _per_class(effects)
    # weakness arm only has seed 42
    per_class = per_class[~((per_class.experiment == "aug_weakness_inpaint") & (per_class.seed != 42))]
    macro = seed_macro_table(per_class, _scopes(), ARMS)
    contrasts = evaluate_contrasts(macro, confirmatory_settings(CONFIG))
    weighted = contrasts[contrasts["contrast"] == "weighted_vs_uniform_weak"].iloc[0]
    assert weighted["n_seeds"] == 1          # seed-matched to the single common seed
    assert math.isnan(weighted["p_value"])   # never tested against a 3-seed mean
    assert "note" in contrasts.columns and isinstance(weighted["note"], str)
    # contrasts not involving the single-seed arm keep all three seeds
    tail_contrast = contrasts[contrasts["contrast"] == "tail_vs_base_tail"].iloc[0]
    assert tail_contrast["n_seeds"] == 3


def test_holm_adjustment() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03, float("nan")])
    # m=3: sorted 0.01→×3=0.03, 0.03→×2=0.06, 0.04→×1=0.04→monotone max 0.06
    assert adjusted[0] == pytest.approx(0.03)
    assert adjusted[2] == pytest.approx(0.06)
    assert adjusted[1] == pytest.approx(0.06)
    assert math.isnan(adjusted[3])


def test_freeze_blocks_changed_plan(tmp_path) -> None:
    freeze_analysis_plan(CONFIG, tmp_path)
    freeze_analysis_plan(CONFIG, tmp_path)  # identical refreeze is fine
    changed = json.loads(json.dumps(CONFIG))
    changed["confirmatory"]["primary_contrasts"][0]["scope"] = "all"
    with pytest.raises(RuntimeError):
        freeze_analysis_plan(changed, tmp_path)


def test_run_confirmatory_stats_end_to_end(tmp_path) -> None:
    effects = {
        "aug_uniform_inpaint": {0: 0.08, 1: 0.09},
        "aug_selective_inpaint": {0: 0.09, 1: 0.08},
        "aug_weakuniform_inpaint": {2: 0.05, 3: 0.06},
        "aug_weakness_inpaint": {2: 0.06, 3: 0.05},
    }
    per_class = _per_class(effects)
    per_class_csv = tmp_path / "per_class_ap.csv"
    per_class.to_csv(per_class_csv, index=False)
    groups_csv = tmp_path / "class_groups.csv"
    pd.DataFrame(
        [{"class_id": cid, "class_name": f"c{cid}", "group": "tail" if cid in TAIL_IDS else "medium"}
         for cid in TAIL_IDS + WEAK_IDS]
    ).to_csv(groups_csv, index=False)
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    pd.DataFrame([{"class_id": cid} for cid in WEAK_IDS]).to_csv(
        analysis_dir / "augmentation_plan_weakness.csv", index=False
    )
    macro_path, summary_path, contrasts_path, md_path = run_confirmatory_stats(
        per_class_csv, groups_csv, tmp_path, CONFIG, eval_split="test"
    )
    contrasts = pd.read_csv(contrasts_path)
    assert set(contrasts["contrast"]) == {"tail_vs_base_tail", "interaction", "weighted_vs_uniform_weak"}
    assert "p_holm" in contrasts.columns
    summary = pd.read_csv(summary_path)
    assert (summary["n_seeds"] == 3).all()
    assert md_path.exists() and (tmp_path / "analysis" / "confirmatory_plan_freeze.json").exists()
