from __future__ import annotations

import argparse
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

for _parent in Path(__file__).resolve().parents:
    if (_parent / "src").is_dir():
        sys.path.insert(0, str(_parent))
        break

from src.utils.detector import is_rtdetr_model, load_detector
from src.utils.io import ensure_dir, load_config, load_yaml, save_yaml
from src.utils.seed import set_seed
from src.utils.timing import format_duration


def run_name_prefix(name: str, model_name: str, seed: int) -> str:
    clean_model_name = Path(model_name).stem
    return f"{name}_{clean_model_name}_seed{seed}_"


def matching_run_dirs(project: str | Path, name: str, model_name: str, seed: int) -> list[Path]:
    project = Path(project)
    prefix = run_name_prefix(name, model_name, seed)
    if not project.exists():
        return []
    return sorted(
        [path for path in project.iterdir() if path.is_dir() and path.name.startswith(prefix)],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def best_or_last_weights(run_dir: str | Path) -> Path | None:
    run_dir = Path(run_dir)
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    if best.exists():
        return best
    if last.exists():
        return last
    return None


def resume_weights(run_dir: str | Path) -> Path | None:
    last = Path(run_dir) / "weights" / "last.pt"
    return last if last.exists() else None


def is_completed_run(run_dir: str | Path) -> bool:
    run_dir = Path(run_dir)
    return (run_dir / "training_meta.yaml").exists() and best_or_last_weights(run_dir) is not None


def run_fingerprint(
    detector: dict[str, Any],
    data_yaml: str | Path,
    use_basic_aug: bool,
) -> dict[str, Any]:
    """Identity of what a training run was trained on/with.

    Stored in training_meta.yaml and compared before reusing a completed run,
    so a config or dataset-path change invalidates the cached run instead of
    silently reporting stale results.
    """
    return {
        "data_yaml": str(data_yaml),
        "model": str(detector.get("model", "yolov8n.pt")),
        "imgsz": int(detector.get("imgsz", 640)),
        "epochs": int(detector.get("epochs", 50)),
        "use_basic_aug": bool(use_basic_aug),
    }


def find_reusable_run(
    project: str | Path,
    name: str,
    model_name: str,
    seed: int,
    expected_fingerprint: dict[str, Any] | None = None,
) -> tuple[Path | None, Path | None]:
    """Return a completed run or an interrupted run with last.pt for resume."""
    for run_dir in matching_run_dirs(project, name, model_name, seed):
        if not is_completed_run(run_dir):
            continue
        if expected_fingerprint is not None:
            stored = (load_yaml(run_dir / "training_meta.yaml") or {}).get("fingerprint")
            if stored is None:
                print(
                    f"[WARN] {run_dir.name}: fingerprint가 없는 구버전 run을 재사용합니다. "
                    "config(epochs/imgsz/model)나 데이터셋이 그때와 같은지 직접 확인하세요."
                )
            elif stored != expected_fingerprint:
                print(
                    f"[INFO] {run_dir.name}: config/데이터 fingerprint 불일치 — 재사용하지 않습니다.\n"
                    f"  저장됨: {stored}\n  현재값: {expected_fingerprint}"
                )
                continue
        return run_dir, None
    for run_dir in matching_run_dirs(project, name, model_name, seed):
        checkpoint = resume_weights(run_dir)
        if checkpoint is not None:
            return run_dir, checkpoint
    return None, None


def disabled_augmentation_args() -> dict[str, float]:
    return {
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "erasing": 0.0,
    }


def core_disabled_augmentation_args() -> dict[str, float]:
    """Conservative no-augmentation args supported by most Ultralytics YOLO versions."""
    return {
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
    }


def train_yolo(
    data_yaml: str | Path,
    config: dict[str, Any],
    name: str,
    seed: int,
    project: str | Path | None = None,
    use_basic_aug: bool = False,
    resume: bool = True,
    force_new_run: bool = False,
) -> Path:
    try:
        import ultralytics  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "ultralytics 패키지가 설치되어 있지 않습니다. Colab에서 먼저 다음 명령을 실행하세요:\n"
            "  pip install -r requirements.txt\n"
            "또는 최소 설치:\n"
            "  pip install ultralytics\n"
            "설치 후 같은 명령을 다시 실행하면 이미 처리된 다운로드/정규화 단계는 대부분 재사용됩니다."
        ) from exc

    detector = config.get("detector", {})
    set_seed(seed)
    model_name = str(detector.get("model", "yolov8n.pt"))
    project = Path(project or Path(config["paths"]["outputs"]) / "runs")
    ensure_dir(project)

    fingerprint = run_fingerprint(detector, data_yaml, use_basic_aug)
    resume_run_dir: Path | None = None
    resume_checkpoint: Path | None = None
    if resume and not force_new_run:
        resume_run_dir, resume_checkpoint = find_reusable_run(
            project, name, model_name, seed, expected_fingerprint=fingerprint
        )
        if resume_run_dir is not None and resume_checkpoint is None:
            print(f"[INFO] 완료된 YOLO run 재사용: {resume_run_dir}")
            return resume_run_dir

    if resume_checkpoint is not None:
        run_name = resume_run_dir.name if resume_run_dir is not None else resume_checkpoint.parent.parent.name
        run_dir = resume_run_dir or resume_checkpoint.parent.parent
        model = load_detector(resume_checkpoint, model_name=model_name)
        print(f"[INFO] 중단된 YOLO 학습 재개: {run_dir} | checkpoint={resume_checkpoint}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        run_name = f"{run_name_prefix(name, model_name, seed)}{timestamp}"
        run_dir = project / run_name
        model = load_detector(model_name)

    epoch_timer = {"start": time.time()}

    def on_train_start(trainer) -> None:  # type: ignore[no-untyped-def]
        epoch_timer["start"] = time.time()
        total_epochs = int(getattr(trainer, "epochs", detector.get("epochs", 0)) or 0)
        print(f"[시간] YOLO 학습 시작: {run_name} | 총 epoch {total_epochs}")

    def on_train_epoch_end(trainer) -> None:  # type: ignore[no-untyped-def]
        current_epoch = int(getattr(trainer, "epoch", 0)) + 1
        total_epochs = int(getattr(trainer, "epochs", detector.get("epochs", current_epoch)) or current_epoch)
        elapsed = time.time() - epoch_timer["start"]
        avg = elapsed / max(1, current_epoch)
        remaining = max(0, total_epochs - current_epoch) * avg
        print(
            f"[시간] YOLO 학습 진행: {run_name} | "
            f"{current_epoch}/{total_epochs} epoch 완료 | "
            f"경과 {format_duration(elapsed)} | 예상 남은 시간 {format_duration(remaining)}"
        )

    try:
        model.add_callback("on_train_start", on_train_start)
        model.add_callback("on_train_epoch_end", on_train_epoch_end)
    except Exception as exc:
        print(f"[WARN] YOLO epoch 시간 콜백을 등록하지 못했습니다. Ultralytics 기본 로그를 사용합니다: {exc}")

    batch = detector.get("batch", -1)
    if is_rtdetr_model(model_name) and (batch is None or int(batch) < 0):
        # Ultralytics auto-batch(-1)는 YOLO 계열 기준으로 검증된 추정이라
        # RT-DETR에서는 OOM/과소 배치가 나올 수 있다. 명시값이 없으면 L4 24GB
        # 기준 보수값으로 고정한다 (config detector.batch로 조정).
        print("[WARN] RT-DETR에 batch=-1(auto)이 설정되어 batch=8로 대체합니다. detector.batch를 명시하세요.")
        batch = 8

    if resume_checkpoint is not None:
        train_args: dict[str, Any] = {"resume": True}
    else:
        train_args = {
            "data": str(data_yaml),
            "imgsz": int(detector.get("imgsz", 640)),
            "epochs": int(detector.get("epochs", 50)),
            "batch": batch,
            "workers": int(detector.get("workers", 2)),
            "seed": seed,
            "project": str(project),
            "name": run_name,
            "patience": int(detector.get("patience", 15)),
            "exist_ok": True,
            "plots": True,
            "verbose": True,
        }
        if not use_basic_aug:
            train_args.update(disabled_augmentation_args())

    started = time.time()
    try:
        model.train(**train_args)
    except Exception as exc:
        if resume_checkpoint is not None:
            raise RuntimeError(
                "YOLO checkpoint resume에 실패했습니다. 데이터 경로가 다시 생성되었는지 확인한 뒤 재실행하세요.\n"
                f"  checkpoint: {resume_checkpoint}\n"
                f"  data.yaml: {data_yaml}"
            ) from exc
        if not use_basic_aug:
            print(f"[WARN] augmentation 비활성화 인자 일부가 현재 Ultralytics와 맞지 않습니다. core 무증강 인자로 재시도합니다: {exc}")
            for key in disabled_augmentation_args():
                train_args.pop(key, None)
            train_args.update(core_disabled_augmentation_args())
            model.train(**train_args)
        else:
            raise
    elapsed = time.time() - started
    save_yaml(
        {
            "run_name": run_name,
            "seed": seed,
            "training_seconds": elapsed,
            # A resumed run only measures the resumed segment, so downstream
            # time-efficiency metrics (ap_gain_per_training_hour) overestimate.
            "training_seconds_resumed_segment_only": resume_checkpoint is not None,
            "args": train_args,
            "fingerprint": fingerprint,
            "resumed_from": str(resume_checkpoint) if resume_checkpoint else "",
        },
        run_dir / "training_meta.yaml",
    )
    save_yaml(config, run_dir / "config.yaml")
    try:
        shutil.copy2(str(data_yaml), run_dir / "data.yaml")
    except Exception:
        pass
    print(f"[INFO] 학습 완료: {run_dir} ({elapsed / 60:.1f}분)")
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO detector with Ultralytics.")
    parser.add_argument("--data", required=True, help="Ultralytics data.yaml")
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--name", required=True, help="Experiment name")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--project", default=None)
    parser.add_argument("--basic-aug", action="store_true", help="Use Ultralytics default augmentation")
    parser.add_argument("--no-resume", action="store_true", help="Do not reuse completed runs or resume from last.pt")
    parser.add_argument("--force-new-run", action="store_true", help="Always create a new timestamped training run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seeds = cfg.get("detector", {}).get("seeds", [42])
    seed = args.seed if args.seed is not None else int(seeds[0])
    train_yolo(
        args.data,
        cfg,
        args.name,
        seed,
        project=args.project,
        use_basic_aug=args.basic_aug,
        resume=not args.no_resume,
        force_new_run=args.force_new_run,
    )


if __name__ == "__main__":
    main()
