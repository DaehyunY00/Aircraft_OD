"""Stage 0 검정: 빈도-난이도 관계가 다른 데이터셋에서도 유지되는가.

주 실험(43클래스, imbalance 10.5배)에서 빈도와 per-class AP는 유의하게 음의
상관(r=-0.375, p=0.013)이었고, 그 덕에 '빈도 하위 K'와 'AP 하위 K'가 교집합
0으로 분리되어 통제 비교가 가능했다. 논문 §VII은 극단적 long-tail에서는 빈도가
예측력을 되찾을 수 있다고 조건부로 적어두었다. 이 스크립트가 그 조건을 측정한다.

판단 기준:
  상관이 음수/0에 가깝고 두 집합이 대체로 분리 → 설계 이전 가능, 본실험 진행
  상관이 양수이고 두 집합이 크게 중첩       → 이전 불가. 그러나 이것도 결과다.
                                              §VII의 추측이 측정으로 바뀐다.

사용:
  python3 src/eval/stage0_probe.py --data <data.yaml> --per-class <per_class_ap.csv> --outputs <dir>
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

for _parent in Path(__file__).resolve().parents:
    if (_parent / "src").is_dir():
        sys.path.insert(0, str(_parent))
        break

import pandas as pd
import yaml
from scipy.stats import pearsonr, spearmanr

from src.utils.io import ensure_dir
from src.utils.yolo import labels_dir_for_images_dir, normalize_class_names

# 주 실험과 동일한 비율: 43클래스 중 하위 13개 = 30%.
BOTTOM_FRACTION = 0.30


def train_instance_counts(data_yaml: Path) -> Counter:
    data = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    root = Path(data.get("path", Path(data_yaml).parent))
    train = Path(data.get("train", "images/train"))
    if not train.is_absolute():
        train = root / train
    labels_dir = labels_dir_for_images_dir(train)
    counts: Counter = Counter()
    for label_path in labels_dir.glob("*.txt"):
        for line in label_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                counts[int(float(line.split()[0]))] += 1
    return counts


def probe(
    data_yaml: Path,
    per_class_csv: Path,
    outputs: Path,
    planning_split: str = "val",
    report_split: str = "test",
    experiment: str = "basic_aug",
) -> pd.DataFrame:
    """빈도-난이도 관계와 두 클래스 집합의 중첩을 측정한다.

    약점 집합은 **계획 split(val)** 의 AP로 정의한다. 주 실험이 그렇게 했고,
    test AP로 대상을 고르면 평가 누출이 되기 때문이다. 이 구분은 실제로 결과를
    바꾼다 — 43클래스 주 실험에서 val 기준 교집합은 0이지만 test 기준으로는
    2가 나온다.
    """
    data = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    names = normalize_class_names(data.get("names"), data.get("nc"))
    counts = train_instance_counts(data_yaml)
    ap_all = pd.read_csv(per_class_csv)
    ap_col = "ap50_95" if "ap50_95" in ap_all.columns else "ap50"
    # 여러 variant가 한 CSV에 쌓이므로 baseline만 골라야 한다. 필터 없이 평균하면
    # 증강 arm의 AP까지 섞여 상관이 조용히 어긋난다.
    if "experiment" in ap_all.columns and experiment in set(ap_all["experiment"]):
        ap_all = ap_all[ap_all["experiment"] == experiment]

    def _split_ap(split: str) -> pd.Series:
        sub = ap_all[ap_all["eval_split"] == split] if "eval_split" in ap_all.columns else ap_all
        return sub.groupby("class_id")[ap_col].mean()

    plan_ap, rep_ap = _split_ap(planning_split), _split_ap(report_split)
    rows = []
    for class_id, name in enumerate(names):
        if counts.get(class_id, 0) == 0 or class_id not in plan_ap.index:
            continue  # 학습 인스턴스나 계획 지표가 없는 클래스는 제외
        rows.append(
            {
                "class_id": class_id,
                "class_name": name,
                "instance_count": counts[class_id],
                f"ap_{planning_split}": float(plan_ap.loc[class_id]),
                f"ap_{report_split}": float(rep_ap.get(class_id, float("nan"))),
            }
        )
    df = pd.DataFrame(rows).dropna(subset=[f"ap_{planning_split}"])
    n = len(df)
    k = max(1, round(n * BOTTOM_FRACTION))

    freq_tail = set(df.nsmallest(k, "instance_count").class_id)
    weak_set = set(df.nsmallest(k, f"ap_{planning_split}").class_id)
    overlap = freq_tail & weak_set

    # 상관은 보고용 split(test)로 낸다. 주 실험 논문 수치와 직접 비교하기 위함.
    corr_col = f"ap_{report_split}" if df[f"ap_{report_split}"].notna().any() else f"ap_{planning_split}"
    valid = df.dropna(subset=[corr_col])
    r, p = pearsonr(valid.instance_count, valid[corr_col])
    rho, ps = spearmanr(valid.instance_count, valid[corr_col])

    analysis_dir = ensure_dir(Path(outputs) / "analysis")
    df.to_csv(analysis_dir / "stage0_per_class.csv", index=False)
    summary = pd.DataFrame(
        [
            {
                "n_classes": n,
                "k": k,
                "ap_column": ap_col,
                "planning_split": planning_split,
                "report_split": report_split,
                "pearson_r": r,
                "pearson_p": p,
                "spearman_rho": rho,
                "spearman_p": ps,
                "overlap_count": len(overlap),
                "overlap_pct": len(overlap) / k * 100,
                "min_instances": int(df.instance_count.min()),
                "max_instances": int(df.instance_count.max()),
                "imbalance_ratio": df.instance_count.max() / max(1, df.instance_count.min()),
            }
        ]
    )
    summary.to_csv(analysis_dir / "stage0_summary.csv", index=False)

    print("=" * 62)
    print(f"클래스 {n}개 | K = {k} (하위 {BOTTOM_FRACTION:.0%}) | 지표 {ap_col}")
    print(f"불균형 비율      : {summary.imbalance_ratio.iloc[0]:.1f}  (주 실험 43클래스는 10.5)")
    print(f"빈도-AP 상관({report_split}) : Pearson r={r:+.3f} (p={p:.4f}) | Spearman rho={rho:+.3f} (p={ps:.4f})")
    print(f"빈도 하위 K ∩ AP 하위 K : {len(overlap)}개 / {k}개 ({len(overlap)/k*100:.1f}% 중첩)")
    print(f"  (약점 집합은 계획 split '{planning_split}' 기준 — 주 실험과 동일 규약)")
    print("-" * 62)
    if r < 0 and p < 0.05 and len(overlap) / k < 0.34:
        print("판정: 역전이 재현되고 두 집합이 대체로 분리 → 본실험 진행 가능")
    elif r > 0 and p < 0.05:
        print("판정: 빈도가 난이도를 예측함(양의 상관). 통제 설계 이전 불가.")
        print("      → 논문 §VII의 '극단적 long-tail에서는 다를 수 있다'가 측정으로 확정됨.")
    else:
        print("판정: 상관이 약하거나 집합이 상당히 중첩. 본실험의 대비가 흐려짐.")
        print("      → 아래 겹치는 클래스를 보고 판단할 것.")
    print("=" * 62)
    print(f"저장: {analysis_dir}/stage0_{{per_class,summary}}.csv")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 0: does the frequency-difficulty inversion transfer?")
    parser.add_argument("--data", required=True, help="data.yaml")
    parser.add_argument("--per-class", required=True, help="per_class_ap.csv")
    parser.add_argument("--outputs", required=True)
    parser.add_argument("--planning-split", default="val", help="약점 집합 정의에 쓸 split (주 실험은 val)")
    parser.add_argument("--report-split", default="test", help="상관 보고에 쓸 split")
    parser.add_argument("--experiment", default="basic_aug", help="baseline variant 이름")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    probe(Path(args.data), Path(args.per_class), Path(args.outputs), args.planning_split, args.report_split, args.experiment)


if __name__ == "__main__":
    main()
