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


def test_gated_variant_without_log_dir_fails_loudly(tmp_path) -> None:
    """게이트 CSV 위치를 모르면 빌드를 멈춰야 한다.

    이전 구현은 generation_log_dir 가 없으면 synthetic_root 로 조용히 대체했는데,
    거기엔 CSV가 없어 엉뚱한 경로를 가리켰다. 같은 누락이 accepted-names 필터도
    꺼버려 기각된 이미지가 학습에 섞일 수 있었다.
    """
    import pytest

    from src.augment.build_experiment_datasets import build_experiment_datasets

    base = tmp_path / "base"
    (base / "images" / "train").mkdir(parents=True)
    (base / "labels" / "train").mkdir(parents=True)
    (tmp_path / "data.yaml").write_text(
        f"path: {base}\ntrain: images/train\nval: images/train\ntest: images/train\nnc: 1\nnames:\n  0: a\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="generation_log_dir"):
        build_experiment_datasets(
            tmp_path / "data.yaml",
            tmp_path / "exp",
            variants=["aug_uniform_inpaint_og"],
            synthetic_root=tmp_path / "syn",
            generation_log_dir=None,
        )
