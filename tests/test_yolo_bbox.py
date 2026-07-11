from pathlib import Path

from src.utils.io import load_yaml
from src.utils.yolo import create_data_yaml, xyxy_to_yolo, yolo_to_xyxy


def test_yolo_to_xyxy_round_trip() -> None:
    box = yolo_to_xyxy(0.5, 0.5, 0.2, 0.4, image_width=100, image_height=200)
    assert box == (40, 60, 60, 140)
    yolo = xyxy_to_yolo(*box, image_width=100, image_height=200)
    assert yolo == (0.5, 0.5, 0.2, 0.4)


def test_create_data_yaml(tmp_path: Path) -> None:
    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "images" / "val").mkdir(parents=True)
    yaml_path = create_data_yaml(tmp_path, ["a10", "f16"])
    data = load_yaml(yaml_path)
    assert data["nc"] == 2
    assert data["names"] == ["a10", "f16"]
    assert data["train"] == "images/train"
