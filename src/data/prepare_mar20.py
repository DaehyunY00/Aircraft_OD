"""MAR20 (VOC XML) → YOLO 변환 + 공식 test 보존 + train에서 val 분할.

MAR20 (Yu et al., National Remote Sensing Bulletin 2023): 원격탐사 군용기
20기종, 공식 train 1,331 / test 2,511장, HBB+OBB 이중 주석. 이 변환기는
HBB(Horizontal Bounding Boxes)만 사용한다.

원본은 저자 배포 링크에서 수동 다운로드해 아래 구조로 배치한다
(디렉터리 이름의 대소문자·공백 변형은 자동 탐색으로 흡수):

  <mar20_raw>/
    JPEGImages/*.jpg
    Annotations/Horizontal Bounding Boxes/*.xml
    ImageSets/Main/train.txt, test.txt

프로토콜:
- 공식 test는 그대로 test split으로 보존한다 (최종 보고 전용).
- 공식 train은 primary class stratified로 train/val = (1-val_fraction)/val_fraction
  분할한다 (planning은 val만 사용 — 기존 실험과 동일한 누수 방지 규칙).
- 클래스 canonical 이름은 XML의 A1..A20 코드를 그대로 쓴다 (MAR20 논문 표기).
  기종 매핑은 metadata/mar20_class_map.json에 참고용으로만 기록한다.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

for _parent in Path(__file__).resolve().parents:
    if (_parent / "src").is_dir():
        sys.path.insert(0, str(_parent))
        break

from PIL import Image
from sklearn.model_selection import train_test_split

from src.utils.io import copy_or_symlink, ensure_dir, save_json
from src.utils.yolo import create_data_yaml, write_yolo_labels, xyxy_to_yolo

MAR20_CLASS_CODES = [f"A{i}" for i in range(1, 21)]

# MAR20 논문(Yu et al. 2023)의 A1..A20 → 기종 매핑. canonical 이름으로는 쓰지
# 않고(순서 오기재 위험을 결과 산출물에 전파하지 않기 위해) 참고 metadata로만
# 기록한다. 논문 figure/표에는 A-코드를 사용할 것.
MAR20_TYPE_BY_CODE = {
    "A1": "SU-35",
    "A2": "C-130",
    "A3": "C-17",
    "A4": "C-5",
    "A5": "F-16",
    "A6": "TU-160",
    "A7": "E-3",
    "A8": "B-52",
    "A9": "P-3C",
    "A10": "B-1B",
    "A11": "E-8",
    "A12": "TU-22",
    "A13": "F-15",
    "A14": "KC-135",
    "A15": "F-22",
    "A16": "FA-18",
    "A17": "TU-95",
    "A18": "KC-10",
    "A19": "SU-34",
    "A20": "SU-24",
}

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def _dirs_with_suffix(root: Path, suffixes: tuple[str, ...]) -> list[Path]:
    found = {p.parent for p in root.rglob("*") if p.suffix.lower() in suffixes}
    return sorted(found)


def find_images_dir(raw_root: Path) -> Path:
    candidates = _dirs_with_suffix(raw_root, IMAGE_EXTS)
    if not candidates:
        raise FileNotFoundError(f"MAR20 이미지(.jpg)를 찾지 못했습니다: {raw_root}")
    for cand in candidates:
        if "jpegimages" in cand.name.lower().replace(" ", ""):
            return cand
    # 이미지가 가장 많은 디렉터리를 선택 (아카이브가 평평하게 풀린 경우)
    return max(candidates, key=lambda d: sum(1 for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS))


def _is_hbb_xml(xml_path: Path) -> bool:
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return False
    obj = root.find("object")
    if obj is None:
        return False
    bndbox = obj.find("bndbox")
    return bndbox is not None and bndbox.find("xmin") is not None


def find_hbb_annotation_dir(raw_root: Path) -> Path:
    candidates = _dirs_with_suffix(raw_root, (".xml",))
    if not candidates:
        raise FileNotFoundError(f"MAR20 XML 주석을 찾지 못했습니다: {raw_root}")
    # 'Horizontal Bounding Boxes' 우선, 그다음 실제 HBB 형식인지 내용으로 검증
    ranked = sorted(
        candidates,
        key=lambda d: 0 if "horizontal" in d.name.lower() or "hbb" in d.name.lower() else 1,
    )
    for cand in ranked:
        sample = next(iter(sorted(cand.glob("*.xml"))), None)
        if sample is not None and _is_hbb_xml(sample):
            return cand
    raise FileNotFoundError(
        f"HBB(xmin/ymin/xmax/ymax) 형식의 XML 디렉터리를 찾지 못했습니다: {raw_root}\n"
        "MAR20의 'Annotations/Horizontal Bounding Boxes'가 포함되도록 아카이브를 풀었는지 확인하세요."
    )


def find_imageset_file(raw_root: Path, split: str) -> Path:
    matches = [p for p in raw_root.rglob(f"{split}.txt") if "imageset" in str(p.parent).lower().replace(" ", "")]
    if not matches:
        matches = list(raw_root.rglob(f"{split}.txt"))
    if not matches:
        raise FileNotFoundError(
            f"MAR20 공식 split 목록({split}.txt)을 찾지 못했습니다: {raw_root}\n"
            "ImageSets/Main/train.txt·test.txt가 포함된 공식 배포 아카이브가 필요합니다. "
            "공식 split 없이 임의 재분할하면 문헌과 비교 불가능한 결과가 됩니다."
        )
    return sorted(matches, key=lambda p: len(p.parts))[0]


def read_imageset_ids(path: Path) -> list[str]:
    ids = [line.strip().split()[0] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return sorted(set(ids))


def parse_hbb_xml(xml_path: Path) -> tuple[int | None, int | None, list[tuple[str, float, float, float, float]]]:
    """Return (width, height, [(class_code, xmin, ymin, xmax, ymax)])."""
    root = ET.parse(xml_path).getroot()
    width = height = None
    size = root.find("size")
    if size is not None:
        try:
            width = int(float(size.findtext("width", "0")))
            height = int(float(size.findtext("height", "0")))
        except (TypeError, ValueError):
            width = height = None
        if not width or not height:
            width = height = None
    boxes: list[tuple[str, float, float, float, float]] = []
    for obj in root.findall("object"):
        code = (obj.findtext("name") or "").strip()
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        try:
            xmin = float(bndbox.findtext("xmin"))
            ymin = float(bndbox.findtext("ymin"))
            xmax = float(bndbox.findtext("xmax"))
            ymax = float(bndbox.findtext("ymax"))
        except (TypeError, ValueError):
            raise ValueError(f"HBB 좌표를 읽지 못했습니다: {xml_path}")
        boxes.append((code, xmin, ymin, xmax, ymax))
    return width, height, boxes


def _image_path_for_id(images_dir: Path, image_id: str) -> Path:
    for ext in IMAGE_EXTS:
        cand = images_dir / f"{image_id}{ext}"
        if cand.exists():
            return cand
    raise FileNotFoundError(f"이미지 파일이 없습니다: {images_dir / image_id}.*")


def _labels_for_id(
    annotations_dir: Path,
    images_dir: Path,
    image_id: str,
    class_index: dict[str, int],
) -> tuple[Path, list[dict[str, float | int]], int]:
    """Return (image_path, yolo_labels, skipped_degenerate)."""
    image_path = _image_path_for_id(images_dir, image_id)
    xml_path = annotations_dir / f"{image_id}.xml"
    if not xml_path.exists():
        raise FileNotFoundError(f"주석 XML이 없습니다: {xml_path}")
    width, height, boxes = parse_hbb_xml(xml_path)
    if width is None or height is None:
        with Image.open(image_path) as img:
            width, height = img.size
    labels: list[dict[str, float | int]] = []
    skipped = 0
    for code, xmin, ymin, xmax, ymax in boxes:
        if code not in class_index:
            raise ValueError(
                f"{xml_path}: 알 수 없는 클래스 코드 {code!r} — MAR20 HBB 주석은 A1..A20이어야 합니다."
            )
        x_center, y_center, w, h = xyxy_to_yolo(xmin, ymin, xmax, ymax, width, height, clip=True)
        if w <= 0.0 or h <= 0.0:
            skipped += 1
            continue
        labels.append(
            {
                "class_id": class_index[code],
                "x_center": x_center,
                "y_center": y_center,
                "width": w,
                "height": h,
            }
        )
    return image_path, labels, skipped


def _primary_class(labels: list[dict[str, float | int]]) -> int:
    if not labels:
        return -1
    return Counter(int(item["class_id"]) for item in labels).most_common(1)[0][0]


def carve_val_split(
    train_ids: list[str],
    primary_by_id: dict[str, int],
    val_fraction: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """공식 train에서 primary-class stratified val을 결정적으로 떼어낸다."""
    ids = sorted(train_ids)
    strata = [primary_by_id[i] for i in ids]
    stratify = strata if min(Counter(strata).values()) >= 2 else None
    train_part, val_part = train_test_split(
        ids, test_size=val_fraction, random_state=seed, stratify=stratify
    )
    return sorted(train_part), sorted(val_part)


def prepare_mar20(
    raw_root: str | Path,
    out_root: str | Path,
    val_fraction: float = 0.2,
    seed: int = 42,
    overwrite: bool = False,
) -> Path:
    """Convert the official MAR20 archive into an Ultralytics YOLO layout."""
    raw_root = Path(raw_root)
    out_root = Path(out_root)
    data_yaml = out_root / "data.yaml"
    if data_yaml.exists() and not overwrite:
        print(f"[INFO] MAR20 변환 결과가 이미 존재합니다 (재사용): {data_yaml}")
        return data_yaml
    if not raw_root.exists():
        raise FileNotFoundError(
            f"MAR20 원본 디렉터리가 없습니다: {raw_root}\n"
            "MAR20은 자동 다운로드를 지원하지 않습니다. 저자 배포 링크(NWPU, "
            "https://gcheng-nwpu.github.io/ 참조)에서 받아 위 경로에 압축 해제하세요."
        )

    images_dir = find_images_dir(raw_root)
    annotations_dir = find_hbb_annotation_dir(raw_root)
    train_txt = find_imageset_file(raw_root, "train")
    test_txt = find_imageset_file(raw_root, "test")
    official_train_ids = read_imageset_ids(train_txt)
    official_test_ids = read_imageset_ids(test_txt)
    overlap = set(official_train_ids) & set(official_test_ids)
    if overlap:
        raise ValueError(f"공식 train/test 목록이 겹칩니다 ({len(overlap)}건) — 아카이브 손상 여부를 확인하세요.")

    class_index = {code: idx for idx, code in enumerate(MAR20_CLASS_CODES)}
    parsed: dict[str, tuple[Path, list[dict[str, float | int]]]] = {}
    total_skipped = 0
    for image_id in official_train_ids + official_test_ids:
        image_path, labels, skipped = _labels_for_id(annotations_dir, images_dir, image_id, class_index)
        total_skipped += skipped
        parsed[image_id] = (image_path, labels)
    if total_skipped:
        print(f"[WARN] 면적 0 박스 {total_skipped}개를 건너뛰었습니다 (clip 후 degenerate).")

    primary_by_id = {image_id: _primary_class(labels) for image_id, (_, labels) in parsed.items()}
    train_ids, val_ids = carve_val_split(official_train_ids, primary_by_id, val_fraction, seed)

    ensure_dir(out_root)
    counts: dict[str, int] = {}
    for split, ids in (("train", train_ids), ("val", val_ids), ("test", official_test_ids)):
        ensure_dir(out_root / "images" / split)
        ensure_dir(out_root / "labels" / split)
        for image_id in ids:
            image_path, labels = parsed[image_id]
            copy_or_symlink(image_path, out_root / "images" / split / image_path.name, overwrite=overwrite)
            write_yolo_labels(labels, out_root / "labels" / split / f"{image_id}.txt")
        counts[split] = len(ids)

    data_yaml = create_data_yaml(out_root, MAR20_CLASS_CODES, data_yaml)
    save_json(
        {
            "dataset": "MAR20",
            "annotation": "horizontal_bounding_boxes",
            "source_raw_root": str(raw_root),
            "official_train": len(official_train_ids),
            "official_test": len(official_test_ids),
            "val_fraction": val_fraction,
            "val_split_seed": seed,
            "counts": counts,
            "class_codes": MAR20_CLASS_CODES,
            "type_by_code_reference_only": MAR20_TYPE_BY_CODE,
        },
        out_root / "metadata" / "mar20_conversion.json",
    )
    print(
        f"[INFO] MAR20 변환 완료: train {counts['train']} / val {counts['val']} / "
        f"test {counts['test']} → {data_yaml}"
    )
    return data_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert the official MAR20 archive to YOLO layout.")
    parser.add_argument("--raw", required=True, help="MAR20 official archive root (JPEGImages/Annotations/ImageSets)")
    parser.add_argument("--out", required=True, help="Output YOLO dataset root")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_mar20(args.raw, args.out, val_fraction=args.val_fraction, seed=args.seed, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
