#!/bin/bash
# 새 Colab 런타임마다 1회 실행: 의존성 설치 + Kaggle 자격증명 + 환경 검증.
# ultralytics 미설치로 파이프라인이 두 번(7/27, 7/29) 죽어서 만든 스크립트.
set -e
cd /content/drive/MyDrive/Military_OD

# Drive의 kaggle.json을 항상 덮어써서 복사한다: 교체된 토큰이 이전에 복사된
# 사본에 가려지는 사고 방지. 형식 검증까지 통과해야 진행.
if [ ! -f kaggle.json ]; then
  echo "[ERROR] Drive의 Military_OD/에 kaggle.json이 없습니다."
  echo "  kaggle.com -> Settings -> API -> 'Create New Token' 버튼으로 파일을 받아 업로드하세요."
  exit 1
fi
python3 - <<'PY'
import json, sys
try:
    d = json.load(open("kaggle.json"))
    assert "username" in d and "key" in d, "username/key 필드 누락"
except Exception as e:
    sys.exit(f"[ERROR] kaggle.json 형식 불량: {e}\n  'Create New Token' 버튼으로 받은 JSON 파일이어야 합니다 (토큰 문자열 복사 X).")
print("kaggle.json 형식 OK")
PY
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# 설치를 인증 검사보다 먼저 한다. Colab은 kaggle CLI가 기본 탑재라 순서가 상관없었지만,
# GCP Deep Learning VM 등 맨 환경에서는 아래 인증 검사가 'kaggle: command not found'로 죽는다.
pip install -q -r requirements.txt

# root가 아닌 환경에서는 pip이 ~/.local 에 설치하는데 이 경로가 PATH에 없는 경우가 많다.
# 콘솔 스크립트(kaggle 등)를 이어지는 단계에서 바로 쓸 수 있도록 여기서 넣어준다.
case ":$PATH:" in
  *":$HOME/.local/bin:"*) ;;
  *) export PATH="$HOME/.local/bin:$PATH" ;;
esac

# ultralytics가 끌어오는 opencv-python은 libGL/libglib을 요구하는데 헤드리스 서버
# 이미지에는 없다(GCP Deep Learning VM에서 'ImportError: libGL.so.1'로 죽음).
# Colab에는 이미 있어서 이 블록이 그냥 통과한다.
if ! python3 -c "import cv2" >/dev/null 2>&1; then
  echo "[INFO] cv2 import 실패 — libGL 등 시스템 라이브러리를 설치합니다."
  sudo apt-get update -qq && sudo apt-get install -y -qq libgl1 libglib2.0-0
fi

# 형식이 맞아도 key 값이 틀리면 다운로드 단계에서야 죽는다(그때까지 수 분 낭비 +
# 로그 동기화 지연으로 원인 파악도 늦어짐). 실제 API 호출로 여기서 판별한다.
if ! kaggle datasets list -s military-aircraft-detection-dataset-yolo-format >/dev/null 2>&1; then
  echo "[ERROR] Kaggle 인증 실패: kaggle.json의 username/key 값이 유효하지 않습니다."
  echo "  kaggle.com -> Settings -> API에서 토큰을 새로 만들어 교체하세요."
  exit 1
fi
echo "Kaggle 인증 OK"

python3 - <<'PY'
import ultralytics, torch_fidelity, diffusers
print("환경 OK: ultralytics", ultralytics.__version__, "| diffusers", diffusers.__version__)
PY
