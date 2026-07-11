from __future__ import annotations

import argparse
import sys
from pathlib import Path

for _parent in Path(__file__).resolve().parents:
    if (_parent / "src").is_dir():
        sys.path.insert(0, str(_parent))
        break

import pandas as pd

from src.augment.build_experiment_datasets import build_experiment_datasets
from src.augment.inpaint_background import generate_from_plan
from src.data.analyze_long_tail import analyze_long_tail, build_augmentation_plans
from src.data.download_kaggle import dataset_exists, download_kaggle_dataset
from src.data.inspect_dataset import inspect_dataset
from src.data.normalize_yolo_dataset import normalize_dataset
from src.eval.collect_yolo_metrics import collect_metrics
from src.eval.compute_long_tail_metrics import compute_long_tail_metrics
from src.eval.plot_results import plot_results
from src.eval.statistics import run_statistical_tests
from src.eval.synthetic_quality import (
    compute_class_fid,
    compute_quality_report,
    plan_quality_filter,
    quality_filter_config,
    synthetic_quality_config,
)
from src.train.train_yolo import train_yolo
from src.utils.io import ensure_dir, load_config, save_json
from src.utils.timing import ProgressTimer, format_duration
from src.utils.variants import parse_variant, uses_basic_aug, uses_synthetic_plan


def _best_weights(run_dir: Path) -> Path | None:
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    if best.exists():
        return best
    if last.exists():
        return last
    return None


def _train_and_collect(
    data_yaml: Path,
    cfg: dict,
    outputs: Path,
    variant: str,
    seed: int,
    eval_split: str,
    resume_training: bool = True,
    force_new_training: bool = False,
) -> Path:
    use_basic_aug = uses_basic_aug(variant)
    run_dir = train_yolo(
        data_yaml,
        cfg,
        variant,
        seed,
        project=outputs / "runs",
        use_basic_aug=use_basic_aug,
        resume=resume_training,
        force_new_run=force_new_training,
    )
    _collect_for_split(run_dir, data_yaml, cfg, outputs, variant, seed, eval_split)
    return run_dir


def _metric_available(outputs: Path, experiment: str, seed: int, eval_split: str) -> bool:
    raw_path = outputs / "metrics" / "raw_yolo_metrics.csv"
    per_class_path = outputs / "metrics" / "per_class_ap.csv"
    if not raw_path.exists() or not per_class_path.exists():
        return False
    raw = pd.read_csv(raw_path)
    per_class = pd.read_csv(per_class_path)
    if raw.empty or per_class.empty:
        return False
    raw_match = raw[
        (raw["experiment"] == experiment)
        & (raw["seed"].astype(int) == int(seed))
        & (raw.get("eval_split", "test") == eval_split)
    ]
    per_class_match = per_class[
        (per_class["experiment"] == experiment)
        & (per_class["seed"].astype(int) == int(seed))
        & (per_class.get("eval_split", "test") == eval_split)
    ]
    return not raw_match.empty and not per_class_match.empty


def _collect_for_split(
    run_dir: Path,
    data_yaml: Path,
    cfg: dict,
    outputs: Path,
    variant: str,
    seed: int,
    eval_split: str,
) -> None:
    if _metric_available(outputs, variant, seed, eval_split):
        print(f"[INFO] 기존 metric 재사용: {variant}, seed={seed}, split={eval_split}")
        return
    weights = _best_weights(run_dir)
    collect_metrics(
        run_dir,
        outputs,
        experiment=variant,
        seed=seed,
        model_name=str(cfg.get("detector", {}).get("model", "yolo")),
        data_yaml=data_yaml,
        weights=weights,
        imgsz=int(cfg.get("detector", {}).get("imgsz", 640)),
        split=eval_split,
    )


def _baseline_ap_for_planning(
    per_class_ap: pd.DataFrame | None,
    planning_split: str,
    baseline_variant: str = "basic_aug",
) -> pd.DataFrame | None:
    """Per-class AP of the planning baseline on the planning split only.

    The weakness score must reference the strong basic_aug baseline (the run all
    tail techniques are compared against), never a test-split AP.
    """
    if per_class_ap is None or per_class_ap.empty:
        return None
    baseline = per_class_ap[per_class_ap["experiment"] == baseline_variant].copy()
    if "eval_split" in baseline.columns:
        baseline = baseline[baseline["eval_split"] == planning_split]
    return baseline if not baseline.empty else None


def _run_training_jobs(
    jobs: list[tuple[str, int, Path]],
    cfg: dict,
    outputs: Path,
    eval_split: str,
    resume_training: bool = True,
    force_new_training: bool = False,
) -> list[Path]:
    timer = ProgressTimer(len(jobs))
    run_dirs: list[Path] = []
    for variant, seed, data_yaml in jobs:
        print(f"[시간] 학습 작업 시작: {variant}, seed={seed} | {timer.next_status()}")
        run_dirs.append(
            _train_and_collect(
                data_yaml,
                cfg,
                outputs,
                variant,
                seed,
                eval_split,
                resume_training=resume_training,
                force_new_training=force_new_training,
            )
        )
        timer.update()
        print(f"[시간] 학습 작업 완료: {variant}, seed={seed} | {timer.status()}")
    return run_dirs


def run_analysis(cfg: dict, force: bool = False, download: bool = False) -> tuple[Path, Path, Path]:
    paths = cfg["paths"]
    raw_data = Path(paths["raw_data"])
    processed_data = Path(paths["processed_data"])
    outputs = ensure_dir(paths["outputs"])
    if dataset_exists(raw_data):
        print(f"[INFO] 원본 데이터셋 확인 완료: {raw_data}")
    elif download:
        download_kaggle_dataset(cfg["kaggle"]["dataset"], raw_data, force=force)
    else:
        dataset = cfg.get("kaggle", {}).get("dataset", "rookieengg/military-aircraft-detection-dataset-yolo-format")
        raise FileNotFoundError(
            f"YOLO 원본 데이터셋을 찾지 못했습니다: {raw_data}\n"
            "먼저 Kaggle 데이터를 다운로드해야 합니다. Colab에서는 다음 중 하나를 실행하세요.\n\n"
            "1) 파이프라인에서 자동 다운로드:\n"
            f"   python src/run_pipeline.py --config configs/smoke.yaml --download\n\n"
            "2) 다운로드 후 파이프라인 실행:\n"
            f"   python src/data/download_kaggle.py --dataset {dataset} --out {raw_data}\n"
            f"   python src/run_pipeline.py --config configs/smoke.yaml\n\n"
            "Kaggle 인증 파일이 없다면 먼저:\n"
            "   mkdir -p ~/.kaggle\n"
            "   cp /content/kaggle.json ~/.kaggle/kaggle.json\n"
            "   chmod 600 ~/.kaggle/kaggle.json"
        )
    inspection = inspect_dataset(raw_data)
    save_json(inspection, outputs / "analysis" / "dataset_inspection.json")
    base_root = processed_data / "base"
    mode = cfg.get("mode", {})
    data_yaml = normalize_dataset(
        raw_data,
        base_root,
        max_images_per_split=mode.get("max_images_per_split"),
        max_classes=mode.get("max_classes"),
        overwrite=force,
        seed=int(cfg.get("detector", {}).get("seeds", [42])[0]),
    )
    grouped = analyze_long_tail(data_yaml, cfg, outputs)
    uniform_plan, selective_plan = build_augmentation_plans(grouped, None, cfg.get("selective_generation", {}), outputs)
    return Path(data_yaml), uniform_plan, selective_plan


def run_pipeline(args: argparse.Namespace) -> None:
    pipeline_timer = ProgressTimer(1)
    cfg = load_config(args.config)
    eval_split = args.eval_split or cfg.get("eval", {}).get("split", "test")
    planning_split = args.planning_split or cfg.get("planning", {}).get("split", "val")
    resume_training = not args.no_resume
    outputs = ensure_dir(cfg["paths"]["outputs"])
    processed_data = ensure_dir(cfg["paths"]["processed_data"])
    experiments_data = ensure_dir(cfg["paths"]["experiments_data"])
    base_data_yaml = processed_data / "base" / "data.yaml"
    print(f"[시간] 파이프라인 시작 | config={args.config} | 계획 split={planning_split} | 최종 평가 split={eval_split}")

    if args.only_train:
        variants = cfg.get("experiments", {}).get("variants", [])
        jobs: list[tuple[str, int, Path]] = []
        for variant in variants:
            data_yaml = experiments_data / variant / "data.yaml"
            if not data_yaml.exists():
                raise FileNotFoundError(f"실험 데이터셋이 없습니다: {data_yaml}")
            for seed in cfg.get("detector", {}).get("seeds", [42]):
                jobs.append((variant, int(seed), data_yaml))
        _run_training_jobs(
            jobs,
            cfg,
            outputs,
            eval_split,
            resume_training=resume_training,
            force_new_training=args.force_new_training,
        )
    else:
        base_data_yaml, uniform_plan, selective_plan = run_analysis(cfg, force=args.force, download=args.download)
        print(f"[시간] 분석 단계 완료 | 전체 경과 {format_duration(pipeline_timer.elapsed())}")
        if args.only_analysis:
            print(f"[시간] 파이프라인 종료 | 전체 경과 {format_duration(pipeline_timer.elapsed())}")
            return

        # Baseline runs come first: real_only (reference lower bound) and
        # basic_aug (the primary baseline every tail technique is measured
        # against, and the source of the weakness score for selective planning).
        baseline_variant = str(cfg.get("planning", {}).get("baseline_variant", "basic_aug"))
        for variant in ("real_only", "basic_aug"):
            for seed in cfg.get("detector", {}).get("seeds", [42]):
                seed = int(seed)
                print(f"[시간] 학습 작업 시작: {variant}, seed={seed} | 계획 metric 수집 split={planning_split}")
                run_dir = _train_and_collect(
                    base_data_yaml,
                    cfg,
                    outputs,
                    variant,
                    seed,
                    planning_split,
                    resume_training=resume_training,
                    force_new_training=args.force_new_training,
                )
                if eval_split != planning_split:
                    print(f"[시간] {variant} 최종 평가 metric 수집: seed={seed}, split={eval_split}")
                    _collect_for_split(run_dir, base_data_yaml, cfg, outputs, variant, seed, eval_split)

        per_class_path = outputs / "metrics" / "per_class_ap.csv"
        baseline_ap = pd.read_csv(per_class_path) if per_class_path.exists() else None
        grouped = pd.read_csv(outputs / "analysis" / "class_groups.csv")
        planning_baseline_ap = _baseline_ap_for_planning(baseline_ap, planning_split, baseline_variant)
        if planning_baseline_ap is None:
            print(
                f"[WARN] 계획 split={planning_split}의 {baseline_variant} AP를 찾지 못했습니다. "
                "class frequency만으로 synthetic plan을 생성합니다."
            )
        uniform_plan, selective_plan = build_augmentation_plans(
            grouped,
            planning_baseline_ap,
            cfg.get("selective_generation", {}),
            outputs,
        )

        synthetic_root = processed_data / "synthetic_inpaint"
        if not args.skip_inpaint:
            if args.dry_run_inpaint:
                print(
                    "[WARN] --dry-run-inpaint: diffusion 없이 원본 사본을 생성합니다. "
                    "파이프라인 구조 점검 전용이며, 이 상태의 synthetic으로 학습한 결과는 무효입니다."
                )
            print(f"[시간] synthetic 생성 시작 | 전체 경과 {format_duration(pipeline_timer.elapsed())}")
            generate_from_plan(
                base_data_yaml,
                uniform_plan,
                synthetic_root,
                outputs,
                cfg,
                plan_name="uniform",
                force=args.force,
                dry_run=args.dry_run_inpaint,
            )
            generate_from_plan(
                base_data_yaml,
                selective_plan,
                synthetic_root,
                outputs,
                cfg,
                plan_name="selective",
                force=args.force,
                dry_run=args.dry_run_inpaint,
            )
            print(f"[시간] synthetic 생성 완료 | 전체 경과 {format_duration(pipeline_timer.elapsed())}")

        # Synthetic quality scoring (CLIPScore/LPIPS/FID) and optional
        # CLIPScore-percentile filtering with budget refill for *_qf variants.
        qf_cfg = quality_filter_config(cfg)
        sq_cfg = synthetic_quality_config(cfg)
        variants_list = cfg.get("experiments", {}).get("variants") or []
        qf_plans = {uses_synthetic_plan(v) for v in variants_list if parse_variant(v).quality_filter}
        qf_plans.discard(None)
        if not args.skip_inpaint and not args.dry_run_inpaint and (sq_cfg["enabled"] or (qf_cfg["enabled"] and qf_plans)):
            print(f"[시간] synthetic 품질 채점 시작 | 전체 경과 {format_duration(pipeline_timer.elapsed())}")
            for plan_name in ("uniform", "selective"):
                log_csv = outputs / "synthetic" / f"generation_log_{plan_name}.csv"
                if not log_csv.exists():
                    continue
                need_filter = qf_cfg["enabled"] and plan_name in qf_plans
                report_path = None
                if sq_cfg["enabled"] or need_filter:
                    report_path = compute_quality_report(
                        log_csv, outputs, plan_name, cfg, max_images=sq_cfg["max_images"]
                    )
                if sq_cfg["enabled"]:
                    compute_class_fid(log_csv, base_data_yaml, outputs, plan_name, cfg, max_images=sq_cfg["max_images"])
                if need_filter and report_path is not None:
                    filter_df, refill_plan = plan_quality_filter(
                        pd.read_csv(report_path), plan_name, qf_cfg["clip_score_percentile"]
                    )
                    filter_path = outputs / "synthetic" / f"quality_filter_{plan_name}.csv"
                    filter_df.to_csv(filter_path, index=False)
                    kept = int(filter_df["kept"].sum()) if "kept" in filter_df.columns else len(filter_df)
                    print(f"[INFO] 품질 필터 저장: {filter_path} (유지 {kept}장)")
                    if not refill_plan.empty:
                        refill_csv = outputs / "analysis" / f"augmentation_plan_{plan_name}_refill.csv"
                        refill_plan.to_csv(refill_csv, index=False)
                        print(
                            f"[INFO] 품질 필터 budget 재보충 생성: {plan_name}_refill "
                            f"({int(refill_plan['num_synthetic_images'].sum())}장)"
                        )
                        generate_from_plan(
                            base_data_yaml,
                            refill_csv,
                            synthetic_root,
                            outputs,
                            cfg,
                            plan_name=f"{plan_name}_refill",
                            force=args.force,
                        )
                        refill_log = outputs / "synthetic" / f"generation_log_{plan_name}_refill.csv"
                        if refill_log.exists():
                            compute_quality_report(
                                refill_log, outputs, f"{plan_name}_refill", cfg, max_images=sq_cfg["max_images"]
                            )

        experiment_yamls = build_experiment_datasets(
            base_data_yaml,
            experiments_data,
            uniform_plan=uniform_plan,
            selective_plan=selective_plan,
            synthetic_root=synthetic_root,
            variants=cfg.get("experiments", {}).get("variants"),
            overwrite=args.force,
            quality_filter_dir=outputs / "synthetic",
            config=cfg,
        )
        jobs = []
        for variant, data_yaml in experiment_yamls.items():
            if variant in ("real_only", "basic_aug"):
                continue  # already trained on the base dataset above
            for seed in cfg.get("detector", {}).get("seeds", [42]):
                jobs.append((variant, int(seed), data_yaml))
        _run_training_jobs(
            jobs,
            cfg,
            outputs,
            eval_split,
            resume_training=resume_training,
            force_new_training=args.force_new_training,
        )

    raw_path = outputs / "metrics" / "raw_yolo_metrics.csv"
    per_class_path = outputs / "metrics" / "per_class_ap.csv"
    groups_path = outputs / "analysis" / "class_groups.csv"
    if raw_path.exists() and per_class_path.exists() and groups_path.exists():
        compute_long_tail_metrics(raw_path, per_class_path, groups_path, outputs, tail_cfg=cfg.get("tail", {}))
        try:
            run_statistical_tests(per_class_path, groups_path, outputs, cfg, eval_split=eval_split)
        except Exception as exc:
            print(f"[WARN] 통계 검정 단계 실패 (실험 결과 자체는 저장됨): {exc}")
        plot_results(outputs, eval_split=eval_split)
    print(f"[시간] 파이프라인 종료 | 전체 경과 {format_duration(pipeline_timer.elapsed())}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full tail-class inpainting experiment pipeline.")
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--skip-inpaint", action="store_true", help="Skip diffusion generation and use existing synthetic files")
    parser.add_argument("--only-train", action="store_true", help="Train existing experiment datasets only")
    parser.add_argument("--only-analysis", action="store_true", help="Normalize/analyze dataset and create augmentation plans only")
    parser.add_argument("--download", action="store_true", help="Download Kaggle dataset before analysis")
    parser.add_argument("--force", action="store_true", help="Overwrite generated artifacts when supported")
    parser.add_argument("--dry-run-inpaint", action="store_true", help="Create synthetic copies without running diffusion")
    parser.add_argument("--eval-split", default=None, choices=["train", "val", "test"], help="Override config eval.split for metric collection")
    parser.add_argument(
        "--planning-split",
        default=None,
        choices=["train", "val", "test"],
        help="Split used only for baseline AP-based selective generation planning. Defaults to config planning.split or val.",
    )
    parser.add_argument("--no-resume", action="store_true", help="Do not reuse completed YOLO runs or resume interrupted last.pt checkpoints")
    parser.add_argument("--force-new-training", action="store_true", help="Always create new YOLO training runs instead of reusing/resuming")
    return parser.parse_args()


def main() -> None:
    run_pipeline(parse_args())


if __name__ == "__main__":
    main()
