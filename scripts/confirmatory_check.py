"""확인 실험 완료 조건 기계 검증. 하나라도 실패하면 exit 1 (ALL_DONE 금지).

검증 항목 (CODEX_GCLOUD_PROMPT의 완료 조건):
- plan invariant: arm당 1000장, 13개 class, uniform quota 76/77, tail/weak disjoint
- 4 diffusion plan의 생성 로그 accepted 수 == plan 배분 합
- 4 diffusion arm × seeds + aug_rfs × seeds + baseline의 test per-class AP 존재
- confirmatory 통계 산출물 + plan freeze 존재
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

for _parent in Path(__file__).resolve().parents:
    if (_parent / "src").is_dir():
        sys.path.insert(0, str(_parent))
        break

import pandas as pd

from src.utils.io import load_config
from src.utils.variants import SYNTHETIC_PLAN_NAMES, uses_synthetic_plan

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    status = "OK " if condition else "FAIL"
    print(f"[{status}] {message}")
    if not condition:
        FAILURES.append(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/confirmatory.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    outputs = Path(cfg["paths"]["outputs"])
    analysis = outputs / "analysis"
    budget = int(cfg["selective_generation"]["total_synthetic_budget"])
    num_classes = int(cfg["selective_generation"]["weakness_num_classes"])
    seeds = [int(s) for s in cfg["detector"]["seeds"]]
    variants = list(cfg.get("experiments", {}).get("variants", []))

    # --- plan invariants ---
    plans: dict[str, pd.DataFrame] = {}
    for plan in SYNTHETIC_PLAN_NAMES:
        path = analysis / f"augmentation_plan_{plan}.csv"
        check(path.exists(), f"plan CSV 존재: {plan}")
        if path.exists():
            plans[plan] = pd.read_csv(path)
    for plan, df in plans.items():
        check(int(df["num_synthetic_images"].sum()) == budget, f"{plan}: 배분 합 == {budget}")
        check(len(df) == num_classes, f"{plan}: 클래스 수 == {num_classes}")
    for plan in ("uniform", "weakness_uniform"):
        if plan in plans:
            alloc = sorted(plans[plan]["num_synthetic_images"].astype(int))
            check(alloc == [76] + [77] * 12, f"{plan}: uniform quota 77×12 + 76×1")
    if "uniform" in plans and "weakness" in plans:
        tail_ids = set(plans["uniform"]["class_id"].astype(int))
        weak_ids = set(plans["weakness"]["class_id"].astype(int))
        check(not (tail_ids & weak_ids), "tail/weak class set 교집합 0")
    if "weakness" in plans and "weakness_uniform" in plans:
        check(
            set(plans["weakness"]["class_id"].astype(int))
            == set(plans["weakness_uniform"]["class_id"].astype(int)),
            "weak 두 plan의 class set 동일",
        )

    # --- generation logs: accepted == plan ---
    for plan, df in plans.items():
        log_path = outputs / "synthetic" / f"generation_log_{plan}.csv"
        check(log_path.exists(), f"생성 로그 존재: {plan}")
        if log_path.exists():
            log = pd.read_csv(log_path)
            if "dry_run" in log.columns:
                check(not log["dry_run"].astype(bool).any(), f"{plan}: dry-run 행 없음")
            accepted = log[log["accepted"].astype(bool)]
            check(len(accepted) == budget, f"{plan}: accepted {len(accepted)} == {budget}")
            per_class = accepted.groupby("class_id").size()
            plan_alloc = df.set_index("class_id")["num_synthetic_images"]
            check(
                per_class.reindex(plan_alloc.index).fillna(0).astype(int).equals(plan_alloc.astype(int)),
                f"{plan}: 클래스별 accepted == plan 배분",
            )

    # --- metrics: test rows for every variant × seed ---
    per_class_path = outputs / "metrics" / "per_class_ap.csv"
    check(per_class_path.exists(), "per_class_ap.csv 존재")
    if per_class_path.exists():
        pc = pd.read_csv(per_class_path)
        test = pc[pc.get("eval_split", "test") == "test"]
        needed = [v for v in variants] + ["basic_aug", "real_only"]
        for variant in needed:
            for seed in seeds:
                rows = test[(test["experiment"] == variant) & (test["seed"].astype(int) == seed)]
                check(not rows.empty, f"test per-class AP 존재: {variant} seed={seed}")

    # --- confirmatory stats + freeze ---
    for name in (
        "confirmatory_plan_freeze.json",
        "confirmatory_seed_macro.csv",
        "confirmatory_summary.csv",
        "confirmatory_contrasts.csv",
        "confirmatory_stats.md",
    ):
        check((analysis / name).exists(), f"산출물 존재: analysis/{name}")

    contrasts_path = analysis / "confirmatory_contrasts.csv"
    if contrasts_path.exists():
        contrasts = pd.read_csv(contrasts_path)
        check((contrasts["n_seeds"] >= len(seeds)).all(), "모든 primary contrast가 전체 seed로 계산됨")
        check("p_holm" in contrasts.columns, "Holm 보정 컬럼 존재")

    # --- synthetic plans consumed by variants actually exist on disk ---
    synthetic_root = Path(cfg["paths"]["synthetic_data"])
    for variant in variants:
        plan = uses_synthetic_plan(variant)
        if plan:
            img_dir = synthetic_root / plan / "images" / "train"
            count = len(list(img_dir.glob("*.jpg"))) if img_dir.exists() else 0
            check(count >= budget, f"{variant}: pool 이미지 {count} >= {budget}")

    print()
    if FAILURES:
        print(f"[RESULT] 실패 {len(FAILURES)}건:")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print("[RESULT] 완료 조건 전부 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
