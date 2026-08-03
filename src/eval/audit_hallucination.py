"""배경 환각(hallucination) 정량 감사.

배경 inpainting의 검증 관문은 '보호 영역 불변 + 배경 변화'만 확인하므로,
재생성된 배경 안에 새 항공기가 생겨도 통과한다. 그렇게 들어온 객체는 라벨이
없어 학습에 라벨 노이즈로 작용한다(논문 초안 §IV-B).

측정 설계 — 원본/생성본 짝지어 비교:
  baseline 검출기를 (a) 원본 소스 이미지와 (b) 생성 이미지에 각각 돌리고,
  GT 박스 바깥의 '여분 검출' 수를 센다. 원본 쪽 여분 검출은 검출기 자체의
  오탐 + 데이터셋 미주석 객체이므로, **증가분(생성 − 원본)** 만이 환각에
  귀속된다. 생성본만 재면 검출기 오탐률과 뒤섞여 과대추정된다.
  증가분이 음수일 수도 있다 — inpainting이 배경의 미주석 실제 항공기를
  지워버리는 경우이며, 이것도 보고할 가치가 있는 사실이다.

여분 판정: 검출 박스 면적의 CONTAINMENT 미만만 GT 박스(마스킹과 동일한 padding)에
겹치면 '바깥'으로 본다. IoU 대신 containment를 쓰는 이유는, 큰 GT 박스 안의 작은
검출이 IoU는 낮아도 새 객체가 아니기 때문이다.

박스 계산은 생성 파이프라인과 같은 helper(labels_to_pixel_boxes)를 쓴다.

생성 로그의 source_image는 생성 당시의 절대경로(예: Colab/GCP VM의 /content/...)라
다른 머신에서는 존재하지 않는다. --source-root를 주면 그 경로 아래를 파일명으로
색인해 다시 매핑한다. 정규화가 원본 파일명을 보존하므로(normalize_yolo_dataset의
_materialize_split) 원본 데이터셋을 직접 가리켜도 된다.

사용:
  # 생성 당시 머신에서
  python3 src/eval/audit_hallucination.py \
      --weights outputs_full/runs/basic_aug_.../weights/best.pt \
      --data /content/data/processed/base/data.yaml \
      --outputs outputs_full --per-plan 100

  # 다른 머신에서 (원본 데이터셋만 있으면 됨)
  python3 src/eval/audit_hallucination.py \
      --weights .../best.pt --source-root /path/to/raw_dataset \
      --outputs outputs_full --per-plan 150
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

for _parent in Path(__file__).resolve().parents:
    if (_parent / "src").is_dir():
        sys.path.insert(0, str(_parent))
        break

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.augment.inpaint_background import _split_dirs
from src.augment.masks import labels_to_pixel_boxes
from src.utils.io import ensure_dir
from src.utils.yolo import label_path_for_image, read_yolo_labels

CONF = 0.5
CONTAINMENT = 0.5


def _max_containment(det, gts) -> float:
    """검출 박스가 어떤 GT 박스에 최대 몇 비율로 담기는가."""
    dx0, dy0, dx1, dy1 = det
    det_area = max(1e-6, (dx1 - dx0) * (dy1 - dy0))
    best = 0.0
    for gx0, gy0, gx1, gy1 in gts:
        ix0, iy0 = max(dx0, gx0), max(dy0, gy0)
        ix1, iy1 = min(dx1, gx1), min(dy1, gy1)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        best = max(best, ((ix1 - ix0) * (iy1 - iy0)) / det_area)
    return best


def _extra_detections(model, image_path: Path, gts) -> int:
    res = model.predict(str(image_path), conf=CONF, verbose=False)[0]
    boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else []
    return sum(1 for b in boxes if _max_containment(tuple(b), gts) < CONTAINMENT)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _index_by_name(root: Path, label: str) -> tuple[dict[str, Path], dict[str, Path]]:
    """파일명 → 경로 색인. 이미지와 라벨을 따로 모은다.

    생성 로그의 절대경로는 생성 당시 머신(Colab/GCP VM)의 것이라 다른 곳에서는
    원본도 생성물도 존재하지 않는다. 파일명으로 다시 잇는다. 같은 이름이 여러
    split에 있으면 먼저 만난 것을 쓴다 — 정규화는 원본을 복사만 하므로 내용이 같다.
    """
    images: dict[str, Path] = {}
    labels: dict[str, Path] = {}
    for path in Path(root).rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            images.setdefault(path.name, path)
        elif suffix == ".txt":
            labels.setdefault(path.stem, path)
    print(f"[INFO] {label} 색인: 이미지 {len(images)}장, 라벨 {len(labels)}개")
    return images, labels


def audit(
    weights: Path,
    outputs: Path,
    per_plan: int,
    plans: list[str],
    data_yaml: Path | None = None,
    source_root: Path | None = None,
    synthetic_root: Path | None = None,
) -> pd.DataFrame:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    images_dir = labels_dir = None
    src_index: dict[str, Path] = {}
    label_index: dict[str, Path] = {}
    if source_root is not None:
        src_index, label_index = _index_by_name(source_root, "source-root")
    elif data_yaml is not None:
        images_dir, labels_dir = _split_dirs(data_yaml)
    else:
        raise ValueError("--data 또는 --source-root 중 하나는 필요합니다.")
    gen_index: dict[str, Path] = {}
    if synthetic_root is not None:
        gen_index, _ = _index_by_name(synthetic_root, "synthetic-root")

    def _resolve(src: Path) -> tuple[Path | None, Path | None]:
        if src_index:
            image = src_index.get(src.name)
            return image, (label_index.get(src.stem) if image else None)
        if src.exists():
            return src, label_path_for_image(src, images_dir, labels_dir)
        return None, None

    def _resolve_generated(gen: Path) -> Path | None:
        if gen_index:
            return gen_index.get(gen.name)
        return gen if gen.exists() else None

    synthetic_dir = Path(outputs) / "synthetic"
    rows = []
    for plan in plans:
        log_path = synthetic_dir / f"generation_log_{plan}.csv"
        if not log_path.exists():
            print(f"[WARN] {log_path} 없음 — 건너뜀")
            continue
        log = pd.read_csv(log_path)
        accepted = log[log["accepted"].astype(bool)].sort_values("output_image").reset_index(drop=True)
        # 결정적 표본: 정렬 후 균등 간격이라 클래스가 고르게 섞인다.
        n = min(per_plan, len(accepted))
        sample = accepted.iloc[np.linspace(0, len(accepted) - 1, n).astype(int)]
        missing = 0
        for _, row in sample.iterrows():
            gen = _resolve_generated(Path(row["output_image"]))
            src, label_path = _resolve(Path(row["source_image"]))
            if src is None or label_path is None or gen is None:
                missing += 1
                continue
            labels = read_yolo_labels(label_path)
            gts = labels_to_pixel_boxes(
                labels, Image.open(gen).size, padding_ratio=float(row.get("mask_padding", 0.10) or 0.10)
            )
            rows.append(
                {
                    "plan": plan,
                    "class_name": row.get("class_name", ""),
                    "source_image": str(src),
                    "output_image": str(gen),
                    "n_gt": len(gts),
                    "extra_source": _extra_detections(model, src, gts),
                    "extra_generated": _extra_detections(model, gen, gts),
                }
            )
        print(f"[INFO] {plan}: 표본 {n}장 중 {n - missing}장 감사 (원본/생성물 미발견 {missing})")

    df = pd.DataFrame(rows)
    if df.empty:
        print("[ERROR] 감사할 표본이 없습니다 — --data 경로와 생성물 경로를 확인하세요.")
        return df
    df["delta"] = df["extra_generated"] - df["extra_source"]
    analysis_dir = ensure_dir(Path(outputs) / "analysis")
    df.to_csv(analysis_dir / "hallucination_audit.csv", index=False)

    summary = df.groupby("plan").agg(
        n_images=("delta", "size"),
        extra_src_per_image=("extra_source", "mean"),
        extra_gen_per_image=("extra_generated", "mean"),
        delta_per_image=("delta", "mean"),
        pct_images_gained=("delta", lambda s: float((s > 0).mean() * 100)),
        pct_images_lost=("delta", lambda s: float((s < 0).mean() * 100)),
    )
    summary.round(4).to_csv(analysis_dir / "hallucination_audit_summary.csv")
    print(f"[INFO] 저장: {analysis_dir}/hallucination_audit{{,_summary}}.csv")
    print(summary.round(3).to_string())
    return df


def gate(
    weights: Path,
    outputs: Path,
    plans: list[str],
    source_root: Path | None = None,
    data_yaml: Path | None = None,
    synthetic_root: Path | None = None,
    max_extra: int = 0,
) -> pd.DataFrame:
    """Object-level gate: 생성물 전량을 검사해 보관/기각 목록을 만든다.

    감사(audit)가 원본과 짝지어 '증가분'을 재는 반면, 게이트는 생성물만 보는
    한쪽짜리 규칙이다. 그래도 타당한 이유는 감사에서 원본 쪽 여분 검출이
    0.002/장으로 사실상 0이었기 때문 — 검출기가 실제 이미지의 보호 영역 밖에서
    발화하는 일은 거의 없다. 따라서 생성물에서의 발화는 대부분 생성 산물이다.

    max_extra=0 이면 보호 영역 밖 확신 검출이 하나라도 있으면 기각한다.
    감사에서 확인된 heavy-tail(영향 이미지 46장 중 4장이 전체 130개 중 54개를
    차지) 때문에, 느슨한 임계값으로도 노이즈 대부분을 걷어낼 수 있다.
    """
    from ultralytics import YOLO

    model = YOLO(str(weights))
    images_dir = labels_dir = None
    src_index: dict[str, Path] = {}
    label_index: dict[str, Path] = {}
    if source_root is not None:
        src_index, label_index = _index_by_name(source_root, "source-root")
    elif data_yaml is not None:
        images_dir, labels_dir = _split_dirs(data_yaml)
    else:
        raise ValueError("--data 또는 --source-root 중 하나는 필요합니다.")
    gen_index: dict[str, Path] = {}
    if synthetic_root is not None:
        gen_index, _ = _index_by_name(synthetic_root, "synthetic-root")

    synthetic_dir = Path(outputs) / "synthetic"
    rows = []
    for plan in plans:
        log_path = synthetic_dir / f"generation_log_{plan}.csv"
        if not log_path.exists():
            print(f"[WARN] {log_path} 없음 — 건너뜀")
            continue
        accepted = pd.read_csv(log_path)
        accepted = accepted[accepted["accepted"].astype(bool)]
        skipped = 0
        for _, row in tqdm(list(accepted.iterrows()), desc=f"gate {plan}"):
            gen = Path(row["output_image"])
            gen = gen_index.get(gen.name) if gen_index else (gen if gen.exists() else None)
            src = Path(row["source_image"])
            if src_index:
                src_resolved = src_index.get(src.name)
                label_path = label_index.get(src.stem) if src_resolved else None
            else:
                src_resolved = src if src.exists() else None
                label_path = label_path_for_image(src, images_dir, labels_dir) if src_resolved else None
            if gen is None or label_path is None:
                skipped += 1
                continue
            labels = read_yolo_labels(label_path)
            gts = labels_to_pixel_boxes(
                labels, Image.open(gen).size, padding_ratio=float(row.get("mask_padding", 0.10) or 0.10)
            )
            n_extra = _extra_detections(model, gen, gts)
            rows.append(
                {
                    "plan": plan,
                    "image": str(gen),
                    "class_name": row.get("class_name", ""),
                    "n_extra": n_extra,
                    "kept": n_extra <= max_extra,
                }
            )
        if skipped:
            print(f"[WARN] {plan}: 경로를 찾지 못해 건너뛴 {skipped}장")

    df = pd.DataFrame(rows)
    if df.empty:
        print("[ERROR] 검사할 생성물이 없습니다.")
        return df
    ensure_dir(synthetic_dir)
    for plan, group in df.groupby("plan"):
        # 데이터셋 빌더의 exclude_names 경로가 읽는 형식(image, kept)과 동일하게 쓴다.
        group.to_csv(synthetic_dir / f"object_gate_{plan}.csv", index=False)
    summary = df.groupby("plan").agg(
        n=("kept", "size"),
        kept=("kept", "sum"),
        dropped=("kept", lambda s: int((~s).sum())),
        drop_pct=("kept", lambda s: float((~s).mean() * 100)),
        extra_objects_removed=("n_extra", lambda s: int(s[s > max_extra].sum())),
    )
    summary.to_csv(synthetic_dir / "object_gate_summary.csv")
    print(f"[INFO] 저장: {synthetic_dir}/object_gate_{{<plan>,summary}}.csv")
    print(summary.to_string())
    print(f"\n예산 정렬용 하한(모든 arm 공통으로 맞출 수): {int(summary['kept'].min())}")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit background hallucination in generated images.")
    parser.add_argument("--gate", action="store_true", help="감사 대신 전량 게이트 모드로 실행")
    parser.add_argument("--max-extra", type=int, default=0, help="허용할 보호 영역 밖 검출 수")
    parser.add_argument("--weights", required=True, help="baseline detector weights (best.pt)")
    parser.add_argument("--data", default=None, help="base data.yaml (생성 당시 머신에서)")
    parser.add_argument("--source-root", default=None, help="원본 데이터셋 루트 (파일명으로 재매핑)")
    parser.add_argument("--synthetic-root", default=None, help="생성물 루트 (파일명으로 재매핑)")
    parser.add_argument("--outputs", required=True, help="experiment outputs root")
    parser.add_argument("--per-plan", type=int, default=100, help="sample size per plan")
    parser.add_argument("--plans", default="uniform,selective,weakness")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plans = [p for p in args.plans.split(",") if p]
    common = dict(
        data_yaml=Path(args.data) if args.data else None,
        source_root=Path(args.source_root) if args.source_root else None,
        synthetic_root=Path(args.synthetic_root) if args.synthetic_root else None,
    )
    if args.gate:
        gate(Path(args.weights), Path(args.outputs), plans, max_extra=args.max_extra, **common)
    else:
        audit(Path(args.weights), Path(args.outputs), args.per_plan, plans, **common)


if __name__ == "__main__":
    main()
