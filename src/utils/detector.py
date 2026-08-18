"""Ultralytics detector class dispatch.

RT-DETR 체크포인트를 ``YOLO`` 클래스로 열면 task_map이 YOLO용
DetectionTrainer/DetectionValidator를 붙여 NMS 전제의 전/후처리가 적용된다 —
RT-DETR은 NMS-free 디코딩과 전용 validator(RTDETRValidator)를 쓰므로 학습과
검증 모두 반드시 ``RTDETR`` 클래스로 열어야 한다.

학습된 체크포인트 경로(``best.pt``)만으로는 모델 계열을 알 수 없으므로,
호출부는 config의 detector.model 이름을 ``model_name``으로 함께 넘긴다.
"""

from __future__ import annotations

from pathlib import Path


def is_rtdetr_model(model_name: str | Path) -> bool:
    return "rtdetr" in Path(str(model_name)).stem.lower()


def load_detector(weights_or_name: str | Path, model_name: str | Path | None = None):
    """Open weights_or_name with the correct Ultralytics model class.

    model_name: 계열 판정에 쓸 원 모델 이름(예: "rtdetr-l.pt"). 생략하면
    weights_or_name 자체의 파일 이름으로 판정한다.
    """
    ref = model_name if model_name is not None else weights_or_name
    if is_rtdetr_model(ref):
        from ultralytics import RTDETR

        return RTDETR(str(weights_or_name))
    from ultralytics import YOLO

    return YOLO(str(weights_or_name))
