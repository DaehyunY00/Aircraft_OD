import pytest

from src.utils.variants import parse_variant, uses_basic_aug, uses_synthetic_plan


def test_parse_base_variants() -> None:
    assert parse_variant("real_only").base == "real_only"
    assert parse_variant("aug_selective_inpaint").base == "aug_selective_inpaint"
    assert not parse_variant("aug_selective_inpaint").quality_filter


def test_parse_quality_filter_suffix() -> None:
    spec = parse_variant("aug_selective_inpaint_qf")
    assert spec.base == "aug_selective_inpaint"
    assert spec.quality_filter


def test_unknown_variant_raises() -> None:
    with pytest.raises(ValueError):
        parse_variant("mystery_variant")


def test_only_real_only_disables_basic_aug() -> None:
    assert not uses_basic_aug("real_only")
    for name in ("basic_aug", "aug_oversample", "aug_rfs", "aug_copy_paste", "aug_uniform_inpaint", "aug_selective_inpaint", "aug_selective_inpaint_qf"):
        assert uses_basic_aug(name), name


def test_uses_synthetic_plan() -> None:
    assert uses_synthetic_plan("aug_uniform_inpaint") == "uniform"
    assert uses_synthetic_plan("aug_selective_inpaint_qf") == "selective"
    assert uses_synthetic_plan("aug_oversample") is None
