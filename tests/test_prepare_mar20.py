from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from PIL import Image

from src.data.prepare_mar20 import (
    MAR20_CLASS_CODES,
    carve_val_split,
    parse_hbb_xml,
    prepare_mar20,
)
from src.utils.io import load_yaml
from src.utils.yolo import read_yolo_labels


def _write_xml(path: Path, width: int, height: int, boxes: list[tuple[str, int, int, int, int]]) -> None:
    root = ET.Element("annotation")
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    for code, xmin, ymin, xmax, ymax in boxes:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = code
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(xmin)
        ET.SubElement(bndbox, "ymin").text = str(ymin)
        ET.SubElement(bndbox, "xmax").text = str(xmax)
        ET.SubElement(bndbox, "ymax").text = str(ymax)
    path.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")


def _make_mar20_raw(root: Path, train_ids: list[str], test_ids: list[str], code_by_id: dict[str, str]) -> None:
    images = root / "JPEGImages"
    hbb = root / "Annotations" / "Horizontal Bounding Boxes"
    imagesets = root / "ImageSets" / "Main"
    for d in (images, hbb, imagesets):
        d.mkdir(parents=True, exist_ok=True)
    for image_id in train_ids + test_ids:
        Image.new("RGB", (100, 80), color=(90, 90, 90)).save(images / f"{image_id}.jpg")
        _write_xml(hbb / f"{image_id}.xml", 100, 80, [(code_by_id[image_id], 10, 20, 50, 60)])
    imagesets.joinpath("train.txt").write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    imagesets.joinpath("test.txt").write_text("\n".join(test_ids) + "\n", encoding="utf-8")


@pytest.fixture()
def mar20_raw(tmp_path: Path) -> tuple[Path, list[str], list[str]]:
    raw = tmp_path / "mar20_raw"
    # 클래스당 5장씩 두 클래스 → stratify 가능 (각 계층 >= 2)
    train_ids = [f"tr{i:03d}" for i in range(10)]
    test_ids = ["te000", "te001"]
    code_by_id = {i: ("A1" if int(i[2:]) % 2 == 0 else "A2") for i in train_ids}
    code_by_id.update({"te000": "A1", "te001": "A20"})
    _make_mar20_raw(raw, train_ids, test_ids, code_by_id)
    return raw, train_ids, test_ids


def test_prepare_mar20_layout_and_splits(mar20_raw, tmp_path: Path) -> None:
    raw, train_ids, test_ids = mar20_raw
    out = tmp_path / "mar20_yolo"
    data_yaml = prepare_mar20(raw, out, val_fraction=0.2, seed=42)
    data = load_yaml(data_yaml)
    assert data["names"] == MAR20_CLASS_CODES
    assert data["nc"] == 20

    train_images = sorted(p.stem for p in (out / "images" / "train").iterdir())
    val_images = sorted(p.stem for p in (out / "images" / "val").iterdir())
    test_images = sorted(p.stem for p in (out / "images" / "test").iterdir())
    # 공식 test는 그대로 보존, val은 공식 train의 20%
    assert test_images == sorted(test_ids)
    assert len(val_images) == 2
    assert len(train_images) == 8
    assert set(train_images) | set(val_images) == set(train_ids)
    assert not (set(train_images) & set(val_images))


def test_prepare_mar20_label_normalization(mar20_raw, tmp_path: Path) -> None:
    raw, _, test_ids = mar20_raw
    out = tmp_path / "mar20_yolo"
    prepare_mar20(raw, out, val_fraction=0.2, seed=42)
    labels = read_yolo_labels(out / "labels" / "test" / f"{test_ids[0]}.txt")
    assert len(labels) == 1
    label = labels[0]
    # (10,20,50,60) @ 100x80 → cx=0.30 cy=0.50 w=0.40 h=0.50, A1 → class 0
    assert label["class_id"] == 0
    assert label["x_center"] == pytest.approx(0.30, abs=1e-6)
    assert label["y_center"] == pytest.approx(0.50, abs=1e-6)
    assert label["width"] == pytest.approx(0.40, abs=1e-6)
    assert label["height"] == pytest.approx(0.50, abs=1e-6)
    # A20 → class 19
    labels_last = read_yolo_labels(out / "labels" / "test" / f"{test_ids[1]}.txt")
    assert labels_last[0]["class_id"] == 19


def test_prepare_mar20_idempotent_and_deterministic(mar20_raw, tmp_path: Path) -> None:
    raw, _, _ = mar20_raw
    out = tmp_path / "mar20_yolo"
    prepare_mar20(raw, out, val_fraction=0.2, seed=42)
    first_val = sorted(p.name for p in (out / "images" / "val").iterdir())
    # 재호출은 기존 변환을 재사용 (overwrite 없이 재분할 금지)
    prepare_mar20(raw, out, val_fraction=0.2, seed=42)
    assert sorted(p.name for p in (out / "images" / "val").iterdir()) == first_val
    # 같은 seed로 새로 변환해도 동일한 val 집합 (결정성)
    out2 = tmp_path / "mar20_yolo_2"
    prepare_mar20(raw, out2, val_fraction=0.2, seed=42)
    assert sorted(p.name for p in (out2 / "images" / "val").iterdir()) == first_val


def test_prepare_mar20_rejects_unknown_class(tmp_path: Path) -> None:
    raw = tmp_path / "raw_bad"
    _make_mar20_raw(raw, ["tr000", "tr001"], ["te000"], {"tr000": "A1", "tr001": "A2", "te000": "B7"})
    with pytest.raises(ValueError, match="B7"):
        prepare_mar20(raw, tmp_path / "out_bad", val_fraction=0.5, seed=42)


def test_prepare_mar20_rejects_overlapping_splits(tmp_path: Path) -> None:
    raw = tmp_path / "raw_overlap"
    _make_mar20_raw(raw, ["tr000", "tr001"], ["tr000"], {"tr000": "A1", "tr001": "A2"})
    with pytest.raises(ValueError, match="겹칩니다"):
        prepare_mar20(raw, tmp_path / "out_overlap", val_fraction=0.5, seed=42)


def test_parse_hbb_xml_reads_size_and_boxes(tmp_path: Path) -> None:
    xml_path = tmp_path / "sample.xml"
    _write_xml(xml_path, 800, 600, [("A3", 1, 2, 3, 4), ("A19", 10, 10, 20, 20)])
    width, height, boxes = parse_hbb_xml(xml_path)
    assert (width, height) == (800, 600)
    assert boxes[0] == ("A3", 1.0, 2.0, 3.0, 4.0)
    assert boxes[1][0] == "A19"


def test_carve_val_split_stratified_deterministic() -> None:
    ids = [f"i{i:02d}" for i in range(20)]
    primary = {i: (0 if idx < 10 else 1) for idx, i in enumerate(ids)}
    train_a, val_a = carve_val_split(ids, primary, val_fraction=0.2, seed=42)
    train_b, val_b = carve_val_split(list(reversed(ids)), primary, val_fraction=0.2, seed=42)
    assert (train_a, val_a) == (train_b, val_b)  # 입력 순서 무관
    assert len(val_a) == 4
    # stratified: 두 계층에서 각각 2장씩
    assert sum(1 for i in val_a if primary[i] == 0) == 2
