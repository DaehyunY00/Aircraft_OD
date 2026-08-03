import pandas as pd

from src.augment.build_experiment_datasets import object_gate_dropped_names
from src.utils.variants import parse_variant, uses_synthetic_plan


def test_og_suffix_sets_object_gate_flag() -> None:
    spec = parse_variant("aug_weakness_inpaint_og")
    assert spec.base == "aug_weakness_inpaint"
    assert spec.object_gate is True
    assert spec.quality_filter is False


def test_gated_variant_still_resolves_its_plan() -> None:
    # 게이트는 어느 이미지를 빼느냐만 바꾼다. 어느 plan을 쓰는지는 그대로여야 한다.
    assert uses_synthetic_plan("aug_uniform_inpaint_og") == "uniform"
    assert uses_synthetic_plan("aug_selective_inpaint_og") == "selective"


def test_both_suffixes_compose() -> None:
    spec = parse_variant("aug_selective_inpaint_qf_og")
    assert spec.base == "aug_selective_inpaint"
    assert spec.quality_filter is True
    assert spec.object_gate is True


def test_gate_reader_returns_dropped_not_kept(tmp_path) -> None:
    csv = tmp_path / "object_gate_uniform.csv"
    pd.DataFrame(
        [
            {"image": "/x/keep_a.jpg", "kept": True},
            {"image": "/x/drop_b.jpg", "kept": False},
            {"image": "/x/keep_c.jpg", "kept": True},
        ]
    ).to_csv(csv, index=False)
    assert object_gate_dropped_names(csv) == {"drop_b.jpg"}


def test_gate_reader_absent_file_is_none_not_empty(tmp_path) -> None:
    # None과 빈 집합을 구분해야 한다. 빈 집합이면 '아무것도 제외 안 함'으로
    # 조용히 통과해 게이트가 걸리지 않은 데이터셋이 만들어진다.
    assert object_gate_dropped_names(tmp_path / "missing.csv") is None
