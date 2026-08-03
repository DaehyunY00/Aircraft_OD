"""Military Aircraft Detection 103-class dataset(CSV) → 정규화된 YOLO 구조 변환.

우리 주 데이터셋(rookieengg 43종)의 상위 집합이다. 이미지 11,788장 중 99.4%가
여기에 포함되고, 새 이미지 11,904장과 새 클래스 60개가 더 있다. 결정적인 차이는
불균형 정도로, imbalance ratio가 10.5에서 **138.7**로 올라간다. 초안 §VII에서
"극단적 long-tail에서는 빈도가 난이도 예측력을 되찾을 수 있다"고 적어둔 조건을
직접 검정하기 위한 데이터셋이다.

입력: labels_with_split.csv (filename,width,height,class,xmin,ymin,xmax,ymax,split)
      dataset/<filename>.jpg
출력: <out>/{images,labels}/{train,val,test}/ + data.yaml
      → normalize_dataset을 거치지 않고 파이프라인에 바로 물릴 수 있는 형태

사용:
  python3 src/data/convert_mad103.py --raw /path/to/raw102 --out /content/data/processed/mad103
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
from tqdm import tqdm

from src.utils.io import ensure_dir

# CSV의 split 이름을 파이프라인이 쓰는 이름으로.
SPLIT_MAP = {"train": "train", "validation": "val", "test": "test"}


def convert(raw_root: Path, out_root: Path, symlink: bool = True) -> Path:
    csv_path = raw_root / "labels_with_split.csv"
    images_root = raw_root / "dataset"
    if not csv_path.exists():
        raise FileNotFoundError(f"labels_with_split.csv를 찾지 못했습니다: {csv_path}")

    df = pd.read_csv(csv_path)
    # 클래스 id는 이름 정렬 순으로 고정한다. 실행마다 바뀌면 학습된 가중치와
    # 라벨 해석이 어긋나므로 결정적이어야 한다.
    class_names = sorted(df["class"].astype(str).unique())
    class_id = {name: i for i, name in enumerate(class_names)}
    print(f"[INFO] 클래스 {len(class_names)}개, 박스 {len(df):,}개, 이미지 {df.filename.nunique():,}장")

    out_root = ensure_dir(out_root)
    for split in ("train", "val", "test"):
        ensure_dir(out_root / "images" / split)
        ensure_dir(out_root / "labels" / split)

    counts = {"train": 0, "val": 0, "test": 0}
    missing = 0
    for (filename, split_raw), group in tqdm(df.groupby(["filename", "split"]), desc="convert"):
        split = SPLIT_MAP.get(str(split_raw))
        if split is None:
            continue
        src_image = images_root / f"{filename}.jpg"
        if not src_image.exists():
            missing += 1
            continue

        lines = []
        for row in group.itertuples():
            w, h = float(row.width), float(row.height)
            if w <= 0 or h <= 0:
                continue
            # xyxy(픽셀) → YOLO 정규화 cx,cy,w,h. 좌표가 이미지 밖으로 나간
            # 주석이 섞여 있어 클램프한다.
            x0, y0 = max(0.0, float(row.xmin)), max(0.0, float(row.ymin))
            x1, y1 = min(w, float(row.xmax)), min(h, float(row.ymax))
            if x1 <= x0 or y1 <= y0:
                continue
            cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
            bw, bh = (x1 - x0) / w, (y1 - y0) / h
            lines.append(f"{class_id[str(row._4)]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        if not lines:
            continue

        dst_image = out_root / "images" / split / f"{filename}.jpg"
        if not dst_image.exists():
            if symlink:
                dst_image.symlink_to(src_image.resolve())
            else:
                dst_image.write_bytes(src_image.read_bytes())
        (out_root / "labels" / split / f"{filename}.txt").write_text("\n".join(lines), encoding="utf-8")
        counts[split] += 1

    data_yaml = out_root / "data.yaml"
    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(class_names))
    data_yaml.write_text(
        f"path: {out_root}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(class_names)}\nnames:\n{names_block}\n",
        encoding="utf-8",
    )
    print(f"[INFO] 이미지: train {counts['train']:,} / val {counts['val']:,} / test {counts['test']:,}")
    if missing:
        print(f"[WARN] 원본 이미지를 찾지 못해 건너뛴 항목: {missing}")
    print(f"[INFO] data.yaml 저장: {data_yaml}")
    return data_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert the 103-class MAD CSV dataset to YOLO layout.")
    parser.add_argument("--raw", required=True, help="labels_with_split.csv와 dataset/ 이 있는 루트")
    parser.add_argument("--out", required=True, help="출력 루트")
    parser.add_argument("--copy", action="store_true", help="심볼릭 링크 대신 복사")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert(Path(args.raw), Path(args.out), symlink=not args.copy)


if __name__ == "__main__":
    main()
