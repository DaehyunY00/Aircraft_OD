from pathlib import Path

from src.utils.io import load_yaml
from src.utils.yolo import create_data_yaml, labels_dir_for_images_dir, xyxy_to_yolo, yolo_to_xyxy


def test_yolo_to_xyxy_round_trip() -> None:
    box = yolo_to_xyxy(0.5, 0.5, 0.2, 0.4, image_width=100, image_height=200)
    assert box == (40, 60, 60, 140)
    yolo = xyxy_to_yolo(*box, image_width=100, image_height=200)
    assert yolo == (0.5, 0.5, 0.2, 0.4)


def test_labels_dir_for_images_dir_is_separator_agnostic(tmp_path: Path) -> None:
    """str.replace('/images/', ...)의 Windows 역슬래시 무력화 버그 회귀 테스트."""
    # 네이티브 Path로 구성 — Windows에서는 역슬래시, Linux에서는 슬래시
    images_dir = tmp_path / "base" / "images" / "train"
    expected = tmp_path / "base" / "labels" / "train"
    assert labels_dir_for_images_dir(images_dir) == expected
    # 마지막 컴포넌트가 'images'면 split 이름으로 간주해 치환하지 않는다
    flat = tmp_path / "dataset" / "images"
    assert labels_dir_for_images_dir(flat) == flat
    # 여러 'images' 컴포넌트면 뒤쪽(마지막 제외)을 치환
    nested = Path("a") / "images" / "x" / "images" / "train"
    assert labels_dir_for_images_dir(nested) == Path("a") / "images" / "x" / "labels" / "train"


def test_create_data_yaml(tmp_path: Path) -> None:
    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "images" / "val").mkdir(parents=True)
    yaml_path = create_data_yaml(tmp_path, ["a10", "f16"])
    data = load_yaml(yaml_path)
    assert data["nc"] == 2
    assert data["names"] == ["a10", "f16"]
    assert data["train"] == "images/train"
