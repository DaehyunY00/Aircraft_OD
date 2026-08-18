from __future__ import annotations

import sys
import types
from pathlib import Path

from src.utils.detector import is_rtdetr_model, load_detector


def test_is_rtdetr_model_names() -> None:
    assert is_rtdetr_model("rtdetr-l.pt")
    assert is_rtdetr_model("rtdetr-x.pt")
    assert is_rtdetr_model("RTDETR-L.pt")
    assert is_rtdetr_model(Path("/models/rtdetr-l.pt"))
    assert not is_rtdetr_model("yolov8n.pt")
    assert not is_rtdetr_model("yolo11s.pt")
    # 학습 산출 체크포인트 이름만으로는 판정 불가 — model_name 없이 열면 YOLO로 간다
    assert not is_rtdetr_model("best.pt")


def _fake_ultralytics(monkeypatch) -> types.ModuleType:
    module = types.ModuleType("ultralytics")

    class _Recorder:
        def __init__(self, weights: str) -> None:
            self.weights = weights

    class FakeYOLO(_Recorder):
        pass

    class FakeRTDETR(_Recorder):
        pass

    module.YOLO = FakeYOLO
    module.RTDETR = FakeRTDETR
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    return module


def test_load_detector_dispatches_by_name(monkeypatch) -> None:
    module = _fake_ultralytics(monkeypatch)
    assert isinstance(load_detector("rtdetr-l.pt"), module.RTDETR)
    assert isinstance(load_detector("yolov8n.pt"), module.YOLO)


def test_load_detector_checkpoint_uses_model_name(monkeypatch) -> None:
    """RT-DETR run의 best.pt는 model_name으로 판정해 RTDETR 클래스로 열어야 한다."""
    module = _fake_ultralytics(monkeypatch)
    ckpt = "/runs/basic_aug_rtdetr-l_seed42_x/weights/best.pt"
    loaded = load_detector(ckpt, model_name="rtdetr-l.pt")
    assert isinstance(loaded, module.RTDETR)
    assert loaded.weights == ckpt
    # model_name이 YOLO면 같은 경로라도 YOLO 클래스
    assert isinstance(load_detector(ckpt, model_name="yolov8n.pt"), module.YOLO)
    # model_name이 없으면 체크포인트 이름 기준 (best.pt → YOLO)
    assert isinstance(load_detector(ckpt), module.YOLO)
