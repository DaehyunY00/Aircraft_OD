"""Experiment-variant name parsing.

Variant naming convention (marginal-gain design, see README Research Questions):

- ``real_only``             : no augmentation at all (reference lower bound)
- ``basic_aug``             : Ultralytics default augmentation (primary baseline)
- ``aug_oversample``        : basic_aug + tail oversampling
- ``aug_rfs``               : basic_aug + repeat factor sampling
- ``aug_copy_paste``        : basic_aug + tail copy-paste
- ``aug_uniform_inpaint``     : basic_aug + uniform tail inpainting (Li et al. ECCV 2024)
- ``aug_selective_inpaint``   : basic_aug + selective tail inpainting (proposed)
- ``aug_weakness_inpaint``    : basic_aug + weakness-driven inpainting (proposed)
- ``aug_weakuniform_inpaint`` : basic_aug + uniform inpainting over the weak set

The four inpainting variants share one generation budget and one class count,
and form a 2×2 design over {class set} × {within-set weighting}:

  set \\ weighting | uniform                  | weighted
  ----------------|--------------------------|------------------------
  frequency tail  | aug_uniform_inpaint      | aug_selective_inpaint
  measured weak   | aug_weakuniform_inpaint  | aug_weakness_inpaint

Uniform splits the budget evenly, selective weights the frequency-defined tail
by rarity+weakness, and the weak-set plans rank *all* classes by measured
baseline AP. This dataset's instance_count/AP50 correlation is -0.33 (rarer
classes score higher), so frequency and weakness select disjoint class sets.

Suffix axes are appended with ``_``:

- ``_qf``: quality-filtered synthetic set (CLIPScore percentile filter).

P2 extension point: add future ablation axes (budget scale ``_x2``, prompt
diversity ``_pdiv`` ...) as additional recognized suffixes here so every module
keeps working from a single parser.
"""

from __future__ import annotations

from dataclasses import dataclass

KNOWN_BASE_VARIANTS = (
    "real_only",
    "basic_aug",
    "aug_oversample",
    "aug_rfs",
    "aug_copy_paste",
    "aug_uniform_inpaint",
    "aug_selective_inpaint",
    "aug_weakness_inpaint",
    "aug_weakuniform_inpaint",
)

_SUFFIX_FLAGS = {
    "qf": "quality_filter",
    # 생성 배경에 만들어진 라벨 없는 객체를 검출기로 걸러낸 세트. CLIPScore 기반
    # _qf 와는 거르는 대상이 다르다 — _qf 는 '못 만든' 이미지를, _og 는 '잘 만들었지만
    # 라벨이 없는 객체가 들어간' 이미지를 뺀다. 후자가 학습에 거짓 음성으로 작용한다.
    "og": "object_gate",
}


@dataclass(frozen=True)
class VariantSpec:
    name: str
    base: str
    quality_filter: bool = False
    object_gate: bool = False


def parse_variant(name: str) -> VariantSpec:
    """Split a variant name into its base variant and suffix flags."""
    remaining = name
    flags: dict[str, bool] = {flag: False for flag in _SUFFIX_FLAGS.values()}
    changed = True
    while changed and remaining not in KNOWN_BASE_VARIANTS:
        changed = False
        for suffix, flag in _SUFFIX_FLAGS.items():
            token = f"_{suffix}"
            if remaining.endswith(token):
                remaining = remaining[: -len(token)]
                flags[flag] = True
                changed = True
    if remaining not in KNOWN_BASE_VARIANTS:
        raise ValueError(
            f"Unknown experiment variant: {name!r}. "
            f"Known base variants: {', '.join(KNOWN_BASE_VARIANTS)}; suffixes: "
            + ", ".join(f"_{s}" for s in _SUFFIX_FLAGS)
        )
    return VariantSpec(name=name, base=remaining, **flags)


def uses_basic_aug(variant: str) -> bool:
    """All tail-technique variants sit on top of the strong basic_aug baseline.

    Only real_only trains with Ultralytics augmentation disabled (2403.07113 /
    X-Paste / Gen2Det reporting practice: measure marginal gain over a strong
    baseline, not against a weak no-augmentation run).
    """
    return parse_variant(variant).base != "real_only"


SYNTHETIC_PLAN_NAMES = ("uniform", "selective", "weakness", "weakness_uniform")

_PLAN_BY_BASE = {
    "aug_uniform_inpaint": "uniform",
    "aug_selective_inpaint": "selective",
    "aug_weakness_inpaint": "weakness",
    "aug_weakuniform_inpaint": "weakness_uniform",
}


def uses_synthetic_plan(variant: str) -> str | None:
    """Return which augmentation plan ('uniform'/'selective'/'weakness') a variant consumes."""
    return _PLAN_BY_BASE.get(parse_variant(variant).base)
