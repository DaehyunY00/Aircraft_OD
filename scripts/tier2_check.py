"""Tier 2 cell 완료 조건 기계 검증 (config 주도).

confirmatory_check.py의 cell 버전 — 하드코딩된 2×2 invariant 대신 config에서
설계를 읽어 검증한다:

  1. confirmatory plan freeze 존재
  2. (baseline_variants ∪ variants) × seeds 전부의 test per-class AP 존재
  3. planning baseline의 val per-class AP 존재 (selective plan 누수 방지 증거)
  4. 학습에 소비되는 synthetic plan마다 생성 로그 accepted 수 == plan 합계
  5. confirmatory_contrasts.csv의 모든 primary contrast가 seed-matched n=len(seeds)
  6. statistical_tests / confirmatory_stats 산출물 존재

통과 시 exit 0, 실패 항목을 전부 나열하고 exit 1.
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
from src.utils.variants import uses_synthetic_plan

PLAN_CSV_BY_NAME = {
    "uniform": "augmentation_plan_uniform.csv",
    "selective": "augmentation_plan_selective.csv",
    "weakness": "augmentation_plan_weakness.csv",
    "weakness_uniform": "augmentation_plan_weakness_uniform.csv",
}


def check_cell(config_path: str | Path) -> list[str]:
    cfg = load_config(config_path)
    outputs = Path(cfg["paths"]["outputs"])
    seeds = [int(s) for s in cfg.get("detector", {}).get("seeds", [42])]
    eval_split = str(cfg.get("eval", {}).get("split", "test"))
    planning_split = str(cfg.get("planning", {}).get("split", "val"))
    baseline_variant = str(cfg.get("planning", {}).get("baseline_variant", "basic_aug"))
    experiments = cfg.get("experiments", {}) or {}
    trained = list(dict.fromkeys(list(experiments.get("baseline_variants") or []) + list(experiments.get("variants") or [])))
    if baseline_variant not in trained:
        trained.append(baseline_variant)
    problems: list[str] = []

    freeze = outputs / "analysis" / "confirmatory_plan_freeze.json"
    if cfg.get("confirmatory") and not freeze.exists():
        problems.append(f"freeze 없음: {freeze}")

    per_class_path = outputs / "metrics" / "per_class_ap.csv"
    if not per_class_path.exists():
        problems.append(f"per_class_ap.csv 없음: {per_class_path}")
        return problems
    per_class = pd.read_csv(per_class_path)
    for variant in trained:
        for seed in seeds:
            rows = per_class[
                (per_class["experiment"] == variant)
                & (per_class["seed"].astype(int) == seed)
                & (per_class["eval_split"] == eval_split)
            ]
            if rows.empty:
                problems.append(f"{eval_split} per-class AP 없음: {variant} seed={seed}")
    planning_rows = per_class[
        (per_class["experiment"] == baseline_variant) & (per_class["eval_split"] == planning_split)
    ]
    if planning_rows.empty:
        problems.append(f"planning baseline({baseline_variant})의 {planning_split} AP 없음 — selective plan 근거 부재")

    needed_plans = {uses_synthetic_plan(v) for v in (experiments.get("variants") or [])} - {None}
    for plan_name in sorted(needed_plans):
        plan_csv = outputs / "analysis" / PLAN_CSV_BY_NAME[plan_name]
        log_csv = outputs / "synthetic" / f"generation_log_{plan_name}.csv"
        if not plan_csv.exists():
            problems.append(f"plan 없음: {plan_csv}")
            continue
        if not log_csv.exists():
            problems.append(f"생성 로그 없음: {log_csv}")
            continue
        planned = int(pd.read_csv(plan_csv)["num_synthetic_images"].sum())
        log = pd.read_csv(log_csv)
        if "dry_run" in log.columns and log["dry_run"].astype(bool).any():
            problems.append(f"{plan_name}: dry-run 행이 생성 로그에 존재 — 무효 데이터")
        accepted = int(log["accepted"].astype(bool).sum()) if "accepted" in log.columns else -1
        if accepted != planned:
            problems.append(f"{plan_name}: accepted {accepted} != plan {planned}")

    if cfg.get("confirmatory"):
        contrasts_path = outputs / "analysis" / "confirmatory_contrasts.csv"
        if not contrasts_path.exists():
            problems.append(f"confirmatory_contrasts.csv 없음: {contrasts_path}")
        else:
            contrasts = pd.read_csv(contrasts_path)
            expected = [str(c.get("name")) for c in cfg["confirmatory"].get("primary_contrasts", [])]
            got = [str(v) for v in contrasts.get("contrast", [])]
            if got != expected:
                problems.append(f"contrast 목록 불일치: 기대 {expected} / 실제 {got}")
            bad = contrasts[contrasts["n_seeds"].astype(int) < len(seeds)] if "n_seeds" in contrasts.columns else contrasts
            for _, row in bad.iterrows():
                problems.append(f"contrast {row.get('contrast')}: n_seeds={row.get('n_seeds')} < {len(seeds)}")
        if not (outputs / "analysis" / "confirmatory_stats.md").exists():
            problems.append("confirmatory_stats.md 없음 (tabulate 설치 확인)")

    if not (outputs / "analysis" / "statistical_tests.csv").exists():
        problems.append("statistical_tests.csv 없음")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description="Tier 2 cell completion check")
    parser.add_argument("--config", required=True, action="append", help="cell config (반복 지정 가능)")
    args = parser.parse_args()
    all_problems: list[str] = []
    for config_path in args.config:
        problems = check_cell(config_path)
        tag = Path(config_path).stem
        if problems:
            print(f"[FAIL] {tag}: {len(problems)}건")
            for p in problems:
                print(f"  - {p}")
            all_problems.extend(problems)
        else:
            print(f"[OK] {tag}: 완료 조건 전부 통과")
    sys.exit(1 if all_problems else 0)


if __name__ == "__main__":
    main()
