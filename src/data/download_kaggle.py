from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

for _parent in Path(__file__).resolve().parents:
    if (_parent / "src").is_dir():
        sys.path.insert(0, str(_parent))
        break

from src.utils.io import ensure_dir


def dataset_exists(out_dir: str | Path) -> bool:
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return False
    has_images = any(p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"} for p in out_dir.rglob("*"))
    has_labels = any(p.suffix.lower() == ".txt" for p in out_dir.rglob("*"))
    has_yaml = any(p.suffix.lower() in {".yaml", ".yml"} for p in out_dir.rglob("*"))
    return has_images and (has_labels or has_yaml)


def download_kaggle_dataset(dataset: str, out_dir: str | Path, force: bool = False) -> Path:
    out_dir = ensure_dir(out_dir)
    if dataset_exists(out_dir) and not force:
        print(f"[INFO] 데이터셋이 이미 존재하여 다운로드를 건너뜁니다: {out_dir}")
        return out_dir
    cmd = ["kaggle", "datasets", "download", "-d", dataset, "-p", str(out_dir), "--unzip"]
    print("[INFO] Kaggle 데이터셋 다운로드 실행:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "kaggle 명령을 찾지 못했습니다. 먼저 `pip install -r requirements.txt`를 실행하세요."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Kaggle 데이터셋 다운로드에 실패했습니다. Colab에서 다음을 먼저 확인하세요:\n"
            "  mkdir -p ~/.kaggle\n"
            "  cp /content/kaggle.json ~/.kaggle/kaggle.json\n"
            "  chmod 600 ~/.kaggle/kaggle.json\n"
            f"실패한 명령: {' '.join(cmd)}"
        ) from exc
    if not dataset_exists(out_dir):
        raise FileNotFoundError(
            f"다운로드 명령은 끝났지만 YOLO 이미지/라벨을 찾지 못했습니다: {out_dir}\n"
            "Kaggle 데이터셋 구조가 바뀌었거나 압축 해제가 실패했을 수 있습니다."
        )
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Kaggle dataset for YOLO experiments.")
    parser.add_argument("--dataset", required=True, help="Kaggle dataset slug, e.g. owner/name")
    parser.add_argument("--out", required=True, help="Output directory for the raw dataset")
    parser.add_argument("--force", action="store_true", help="Download even if files already exist")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_kaggle_dataset(args.dataset, args.out, force=args.force)


if __name__ == "__main__":
    main()
