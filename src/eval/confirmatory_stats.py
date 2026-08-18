"""Seed-blocked confirmatory analysis for the 2×2 allocation design.

Analysis units are per-seed macro APs (mAP50-95 over a class scope), so every
contrast is seed-matched by construction: a contrast only uses seeds for which
*all* participating arms have a run, and a single-seed arm is never compared
against a multi-seed average.

Outputs (outputs*/analysis/):
- confirmatory_seed_macro.csv : experiment × seed × scope macro AP
- confirmatory_summary.csv    : per arm × scope mean, SD, t-CI over seeds
- confirmatory_contrasts.csv  : pre-specified contrasts with per-seed estimates,
                                95% CI, paired t p-value, Holm-adjusted p
- confirmatory_stats.md       : readable summary

The contrast family is defined in config (confirmatory.primary_contrasts) and
must be frozen (JSON + hash) before test metrics are collected — see
freeze_analysis_plan().
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for _parent in Path(__file__).resolve().parents:
    if (_parent / "src").is_dir():
        sys.path.insert(0, str(_parent))
        break

import numpy as np
import pandas as pd

from src.utils.io import ensure_dir, load_config


def confirmatory_settings(config: dict[str, Any]) -> dict[str, Any]:
    cfg = config.get("confirmatory", {}) or {}
    arms = cfg.get("arms", {}) or {}
    return {
        "baseline": str(cfg.get("baseline", "basic_aug")),
        "arms": {str(k): str(v) for k, v in arms.items()},
        "confidence": float(cfg.get("confidence", 0.95)),
        "primary_contrasts": cfg.get("primary_contrasts", []) or [],
        "extra_experiments": [str(v) for v in cfg.get("extra_experiments", ["aug_rfs"])],
    }


def freeze_analysis_plan(config: dict[str, Any], outputs: str | Path) -> Path:
    """Write the pre-specified contrast family to JSON with a content hash.

    Must run before any test-split metric exists so the analysis cannot be
    tuned on the final numbers. Freezing twice is fine only when the content
    is identical; a changed plan after freezing raises.
    """
    settings = confirmatory_settings(config)
    payload = {
        "baseline": settings["baseline"],
        "arms": settings["arms"],
        "confidence": settings["confidence"],
        "primary_contrasts": settings["primary_contrasts"],
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    analysis_dir = ensure_dir(Path(outputs) / "analysis")
    freeze_path = analysis_dir / "confirmatory_plan_freeze.json"
    if freeze_path.exists():
        stored = json.loads(freeze_path.read_text(encoding="utf-8"))
        if stored.get("sha256") != digest:
            raise RuntimeError(
                f"확인 분석 plan이 freeze 이후 변경되었습니다: {freeze_path}\n"
                f"  frozen sha256: {stored.get('sha256')}\n  current sha256: {digest}\n"
                "test 결과를 본 뒤 contrast를 바꾸는 것은 금지됩니다. 의도된 변경이면 "
                "freeze 파일을 별도 이름으로 보존하고 사유를 RESULTS_CONFIRMATORY.md에 기록하세요."
            )
        return freeze_path
    freeze_path.write_text(
        json.dumps(
            {"frozen_at_utc": datetime.now(timezone.utc).isoformat(), "sha256": digest, "plan": payload},
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[INFO] 확인 분석 plan freeze: {freeze_path} (sha256={digest[:12]}…)")
    return freeze_path


def _scopes(outputs: Path, class_groups_csv: str | Path) -> dict[str, set[int] | None]:
    groups = pd.read_csv(class_groups_csv)
    scopes: dict[str, set[int] | None] = {"all": None}
    scopes["tail"] = set(groups.loc[groups["group"] == "tail", "class_id"].astype(int))
    weakness_plan = Path(outputs) / "analysis" / "augmentation_plan_weakness.csv"
    if weakness_plan.exists():
        weak_ids = set(pd.read_csv(weakness_plan)["class_id"].astype(int))
        if weak_ids:
            scopes["weak"] = weak_ids
    return scopes


def seed_macro_table(
    per_class: pd.DataFrame,
    scopes: dict[str, set[int] | None],
    experiments: list[str],
) -> pd.DataFrame:
    ap_col = "ap50_95" if "ap50_95" in per_class.columns else "ap50"
    df = per_class.dropna(subset=[ap_col])
    df = df[df["experiment"].isin(experiments)]
    rows: list[dict[str, Any]] = []
    for scope, class_ids in scopes.items():
        scoped = df if class_ids is None else df[df["class_id"].astype(int).isin(class_ids)]
        macro = scoped.groupby(["experiment", "seed"], as_index=False)[ap_col].mean()
        for _, row in macro.iterrows():
            rows.append(
                {
                    "experiment": row["experiment"],
                    "seed": int(row["seed"]),
                    "scope": scope,
                    "macro_ap": float(row[ap_col]),
                }
            )
    return pd.DataFrame(rows)


def _macro_lookup(seed_macro: pd.DataFrame) -> dict[tuple[str, int, str], float]:
    return {
        (str(r["experiment"]), int(r["seed"]), str(r["scope"])): float(r["macro_ap"])
        for _, r in seed_macro.iterrows()
    }


def _common_seeds(seed_macro: pd.DataFrame, experiments: list[str], scopes: list[str]) -> list[int]:
    seeds: set[int] | None = None
    for experiment in experiments:
        for scope in scopes:
            have = set(
                seed_macro.loc[
                    (seed_macro["experiment"] == experiment) & (seed_macro["scope"] == scope), "seed"
                ].astype(int)
            )
            seeds = have if seeds is None else (seeds & have)
    return sorted(seeds or set())


def _t_stats(values: np.ndarray, confidence: float) -> dict[str, float]:
    from scipy.stats import t, ttest_1samp

    values = np.asarray(values, dtype=float)
    n = len(values)
    out = {
        "n_seeds": n,
        "estimate": float(values.mean()) if n else float("nan"),
        "sd": float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "t_stat": float("nan"),
        "p_value": float("nan"),
    }
    if n >= 2:
        sd = float(values.std(ddof=1))
        out["sd"] = sd
        half = float(t.ppf(0.5 + confidence / 2.0, df=n - 1) * sd / math.sqrt(n))
        out["ci_low"] = out["estimate"] - half
        out["ci_high"] = out["estimate"] + half
        if sd > 0:
            res = ttest_1samp(values, 0.0)
            out["t_stat"] = float(res.statistic)
            out["p_value"] = float(res.pvalue)
    return out


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down adjustment; NaN entries stay NaN and do not count toward m."""
    indexed = [(i, p) for i, p in enumerate(p_values) if not (p is None or (isinstance(p, float) and math.isnan(p)))]
    m = len(indexed)
    adjusted: list[float] = [float("nan")] * len(p_values)
    running_max = 0.0
    for rank, (i, p) in enumerate(sorted(indexed, key=lambda item: item[1])):
        value = min(1.0, (m - rank) * p)
        running_max = max(running_max, value)
        adjusted[i] = running_max
    return adjusted


def evaluate_contrasts(
    seed_macro: pd.DataFrame,
    settings: dict[str, Any],
) -> pd.DataFrame:
    arms = settings["arms"]
    tail_arms = [arms[k] for k in ("tail_uniform", "tail_weighted") if k in arms]
    weak_arms = [arms[k] for k in ("weak_uniform", "weak_weighted") if k in arms]
    lookup = _macro_lookup(seed_macro)
    rows: list[dict[str, Any]] = []
    for contrast in settings["primary_contrasts"]:
        name = str(contrast.get("name", "unnamed"))
        if contrast.get("interaction"):
            experiments = tail_arms + weak_arms
            scopes = ["tail", "weak"]
            seeds = _common_seeds(seed_macro, experiments, scopes)
            values = []
            for seed in seeds:
                weak_in_weak = np.mean([lookup[(a, seed, "weak")] for a in weak_arms])
                tail_in_weak = np.mean([lookup[(a, seed, "weak")] for a in tail_arms])
                weak_in_tail = np.mean([lookup[(a, seed, "tail")] for a in weak_arms])
                tail_in_tail = np.mean([lookup[(a, seed, "tail")] for a in tail_arms])
                values.append((weak_in_weak - tail_in_weak) - (weak_in_tail - tail_in_tail))
            description = (
                f"[mean({'+'.join(weak_arms)}) − mean({'+'.join(tail_arms)})] in weak scope "
                f"minus the same difference in tail scope"
            )
        else:
            plus = [str(v) for v in contrast.get("plus", [])]
            minus = [str(v) for v in contrast.get("minus", [])]
            scope = str(contrast.get("scope", "all"))
            experiments = plus + minus
            scopes = [scope]
            seeds = _common_seeds(seed_macro, experiments, scopes)
            values = [
                float(np.mean([lookup[(a, s, scope)] for a in plus]))
                - float(np.mean([lookup[(a, s, scope)] for a in minus]))
                for s in seeds
            ]
            description = f"mean({'+'.join(plus)}) − mean({'+'.join(minus)}) @ {scope}"
        stats = _t_stats(np.asarray(values), settings["confidence"])
        row: dict[str, Any] = {"contrast": name, "description": description, "seeds_used": ",".join(map(str, seeds))}
        row.update(stats)
        for seed, value in zip(seeds, values):
            row[f"seed{seed}"] = value
        if stats["n_seeds"] < 2:
            row["note"] = "seed-matched 공통 seed가 2개 미만 — 검정 불가 (단일 seed 비교 금지 규칙)"
        rows.append(row)
    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_holm"] = holm_adjust([float(p) for p in result["p_value"]])
    return result


def run_confirmatory_stats(
    per_class_ap_csv: str | Path,
    class_groups_csv: str | Path,
    outputs: str | Path,
    config: dict[str, Any],
    eval_split: str = "test",
) -> tuple[Path, Path, Path, Path]:
    settings = confirmatory_settings(config)
    freeze_analysis_plan(config, outputs)
    per_class = pd.read_csv(per_class_ap_csv)
    if "eval_split" in per_class.columns:
        per_class = per_class[per_class["eval_split"] == eval_split]
        if per_class.empty:
            raise ValueError(f"per_class_ap에 eval_split={eval_split} 행이 없습니다.")
    outputs = Path(outputs)
    scopes = _scopes(outputs, class_groups_csv)
    experiments = sorted(
        {settings["baseline"], *settings["arms"].values(), *settings["extra_experiments"]}
        & set(per_class["experiment"].unique())
    )
    seed_macro = seed_macro_table(per_class, scopes, experiments)

    summary_rows: list[dict[str, Any]] = []
    for (experiment, scope), group in seed_macro.groupby(["experiment", "scope"]):
        stats = _t_stats(group["macro_ap"].to_numpy(), settings["confidence"])
        summary_rows.append(
            {
                "experiment": experiment,
                "scope": scope,
                "n_seeds": stats["n_seeds"],
                "macro_ap_mean": stats["estimate"],
                "macro_ap_sd": stats["sd"],
                "ci_low": stats["ci_low"],
                "ci_high": stats["ci_high"],
                "seeds": ",".join(map(str, sorted(group["seed"].astype(int)))),
            }
        )
    summary = pd.DataFrame(summary_rows)
    contrasts = evaluate_contrasts(seed_macro, settings)

    analysis_dir = ensure_dir(outputs / "analysis")
    macro_path = analysis_dir / "confirmatory_seed_macro.csv"
    summary_path = analysis_dir / "confirmatory_summary.csv"
    contrasts_path = analysis_dir / "confirmatory_contrasts.csv"
    md_path = analysis_dir / "confirmatory_stats.md"
    seed_macro.to_csv(macro_path, index=False)
    summary.to_csv(summary_path, index=False)
    contrasts.to_csv(contrasts_path, index=False)
    md_path.write_text(_render_markdown(seed_macro, summary, contrasts, eval_split), encoding="utf-8")
    print(f"[INFO] 확인 통계 저장: {contrasts_path}")
    return macro_path, summary_path, contrasts_path, md_path


def _render_markdown(
    seed_macro: pd.DataFrame,
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    eval_split: str,
) -> str:
    lines = [f"# Confirmatory seed-blocked analysis (eval_split={eval_split})", ""]
    lines.append("## Per-seed macro AP (mAP50-95)")
    lines.append("")
    if seed_macro.empty:
        lines.append("_empty_")
    else:
        pivot = seed_macro.pivot_table(index=["experiment", "seed"], columns="scope", values="macro_ap")
        lines.append(pivot.reset_index().to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## Arm summaries: mean ± SD, 95% t-CI over seeds")
    lines.append("")
    lines.append(summary.to_markdown(index=False, floatfmt=".4f") if not summary.empty else "_empty_")
    lines.append("")
    lines.append("## Pre-specified contrasts (seed-blocked paired t, Holm-adjusted)")
    lines.append("")
    if contrasts.empty:
        lines.append("_empty_")
    else:
        cols = [c for c in ("contrast", "n_seeds", "estimate", "sd", "ci_low", "ci_high", "p_value", "p_holm", "seeds_used", "note") if c in contrasts.columns]
        lines.append(contrasts[cols].to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed-blocked confirmatory statistics for the 2×2 design.")
    parser.add_argument("--per-class", required=True, help="metrics/per_class_ap.csv")
    parser.add_argument("--groups", required=True, help="analysis/class_groups.csv")
    parser.add_argument("--outputs", required=True)
    parser.add_argument("--config", default="configs/confirmatory.yaml")
    parser.add_argument("--eval-split", default=None, choices=["train", "val", "test"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    split = args.eval_split or cfg.get("eval", {}).get("split", "test")
    run_confirmatory_stats(args.per_class, args.groups, args.outputs, cfg, eval_split=split)


if __name__ == "__main__":
    main()
