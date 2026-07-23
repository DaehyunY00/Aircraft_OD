# Tail-Class Selective Background Inpainting Augmentation

SCI급 연구 논문 작성을 목표로 한 long-tailed military aircraft detection 실험 코드베이스입니다. 핵심 가설은 tail class에만 bbox-protected diffusion background inpainting을 적용하면 tail AP가 개선되며, 같은 synthetic image budget에서 class frequency와 baseline AP를 함께 고려한 selective generation이 uniform tail generation보다 효율적이라는 것입니다.

## Research Questions

1. **RQ1 (marginal gain)**: 강한 baseline(`basic_aug`, Ultralytics 기본 증강) 위에 tail-class
   diffusion background inpainting을 얹으면 tail AP의 **추가(marginal) 개선**이 있는가?
   (tail oversampling, RFS, Copy-Paste 등 표준 리밸런싱 대비 포함)
2. **RQ2 (budget 배분)**: 같은 synthetic image budget에서 rarity×weakness priority 기반
   selective generation이 uniform tail generation(= Li et al. ECCV 2024 방식의 tail 적용)보다
   효율적인가?
3. **RQ3 (head 보존)**: Tail AP 개선이 head-class AP 또는 overall mAP 손상 없이 가능한가?

모든 tail 기법 variant는 `basic_aug` 위에서 비교합니다(2403.07113, X-Paste/Gen2Det/DiverGen의
보고 관행). 기본 증강을 끈 단독 비교는 하지 않으며, `real_only`는 참고용 하한선으로만
유지합니다.

## Repository Structure

```text
configs/                 # smoke/full/default experiment configs
notebooks/               # Colab experiment notebook
src/data/                # Kaggle download, inspection, normalization, long-tail analysis
src/augment/             # bbox mask, inpainting, oversampling, experiment dataset builder
src/train/               # Ultralytics YOLO training
src/eval/                # metric collection, long-tail metrics, plots
src/utils/               # IO, YOLO bbox, image, seed utilities
tests/                   # lightweight unit tests
outputs/                 # local output placeholder
```

## Colab Setup

```bash
git clone <repo-url>
cd military-aircraft-tail-inpainting
pip install -r requirements.txt
```

Google Drive에 이미 저장소가 있다면 해당 폴더에서 바로 설치만 실행하면 됩니다.

```bash
cd /content/drive/MyDrive/Military_OD
pip install -r requirements.txt
```

Kaggle credential 설정:

```bash
mkdir -p ~/.kaggle
cp /content/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Dataset download:

```bash
python src/data/download_kaggle.py \
  --dataset rookieengg/military-aircraft-detection-dataset-yolo-format \
  --out /content/data/raw
```

Smoke test:

```bash
python src/run_pipeline.py --config configs/smoke.yaml
```

아직 `/content/data/raw`에 데이터셋이 없다면 다운로드까지 함께 실행합니다.

```bash
python src/run_pipeline.py --config configs/smoke.yaml --download
```

Full experiment:

```bash
python src/run_pipeline.py --config configs/full.yaml
```

## Recommended Experiment Protocol

논문용 최종 test를 보기 전에, 먼저 validation-only pilot으로 가설을 점검하세요.

```bash
# 1. 빠른 구조 검증
python src/run_pipeline.py --config configs/smoke.yaml --download

# 2. 가설 검증용 pilot: validation split만 평가
python src/run_pipeline.py --config configs/pilot.yaml --download

# 3. pilot 결과 확인
python - <<'PY'
import pandas as pd
cols = ["experiment", "eval_split", "mAP50", "mAP50_95", "head_ap", "medium_ap", "tail_ap", "tail_ap_gain_vs_basic_aug", "synthetic_images"]
print(pd.read_csv("outputs_pilot/metrics/summary_by_experiment.csv")[cols].to_markdown(index=False, floatfmt=".4f"))
PY

# 4. 설정 고정 후 최종 full: test split 평가
python src/run_pipeline.py --config configs/full.yaml --download
```

`configs/pilot.yaml`은 `eval.split: "val"`이고, `configs/full.yaml`은 `eval.split: "test"`입니다. Full mode에서도 selective generation 계획은 `planning.split: "val"`의 baseline AP만 사용하므로 test AP가 augmentation 계획에 누수되지 않습니다. 필요하면 명령줄에서 임시로 덮어쓸 수 있습니다.

```bash
python src/run_pipeline.py --config configs/pilot.yaml --eval-split val
python src/run_pipeline.py --config configs/full.yaml --planning-split val --eval-split test
python src/eval/collect_yolo_metrics.py --config configs/full.yaml --split test ...
```

중단된 Colab 세션을 이어갈 때:

```bash
python src/run_pipeline.py --config configs/full.yaml --download
```

기본값은 resume enabled입니다. 이미 정상 종료된 YOLO run은 재사용하고, 중간에 끊긴 run은 `outputs_full/runs/*/weights/last.pt`에서 이어 학습합니다. Synthetic image 생성도 이미 존재하는 파일은 건너뜁니다. 기존 checkpoint를 무시하고 새 학습 run을 만들 때만 다음 옵션을 사용하세요.

```bash
python src/run_pipeline.py --config configs/full.yaml --download --force-new-training
```

이미 synthetic image가 충분히 생성된 뒤 학습만 이어갈 때는:

```bash
python src/run_pipeline.py --config configs/full.yaml --only-train
```

GPU 없이 구조만 빠르게 검증할 때는 diffusion 대신 원본 복사본을 synthetic placeholder로 만드는 dry-run을 사용할 수 있습니다.

```bash
python src/run_pipeline.py --config configs/smoke.yaml --dry-run-inpaint
```

**주의: `--dry-run-inpaint`는 파이프라인 구조 점검 전용입니다.** 생성물은 원본 사본이며
학습/논문 실험에 사용하면 안 됩니다. dry-run은 plan 디렉터리에 `DRY_RUN_MARKER.txt`를
남기고, 이후 실제 실행이 이 marker를 발견하면 해당 디렉터리 전체를 재생성합니다. 실제
실행에서는 모든 생성물이 "배경이 임계 이상 바뀌었는지" 검증(`verification` config 섹션)을
통과해야 train split에 포함됩니다.

> **구버전 결과 폐기 공지**: `outputs_pilot_deprecated/`는 생성 검증 버그(BUGFIX_REPORT.md
> 참고)가 수정되기 전의 pilot 산출물입니다. 증거 보존용으로만 남겨두며, 해당 결과는 전량
> 폐기되었고 논문에 사용하지 않습니다.

실험 실행 중에는 한글 시간 로그가 함께 출력됩니다. 파이프라인은 분석/생성/학습 작업 단위의 경과 시간과 예상 남은 시간을 출력하고, YOLO 학습은 epoch 단위 ETA를, inpainting 생성은 이미지 단위 ETA를 표시합니다.

Colab 런타임이 끊겨도 weight와 metric을 보존하려면 config 파일의 `paths.outputs`를 Google Drive 경로로 바꾸세요. 터미널에 `outputs: ...`를 직접 입력하는 것이 아니라 YAML 파일을 수정해야 합니다. 예:

```bash
python - <<'PY'
from pathlib import Path
path = Path('configs/smoke.yaml')
text = path.read_text()
text = text.replace('outputs: "/content/outputs"', 'outputs: "/content/drive/MyDrive/Military_OD/outputs"')
path.write_text(text)
PY
```


## Experiment Groups

- `real_only`: 기본 증강 OFF. 참고용 하한선.
- `basic_aug`: Ultralytics YOLO 기본 증강 ON (mosaic/mixup 등). **주 baseline.**
- `aug_oversample`: basic_aug + tail oversampling (selective budget과 동일 수량).
- `aug_rfs`: basic_aug + Repeat Factor Sampling (Gupta et al. CVPR 2019, 데이터셋 수준 구현).
- `aug_copy_paste`: basic_aug + tail Copy-Paste (Ghiasi et al. CVPR 2021; bbox 단위 rectangular paste).
- `aug_uniform_inpaint`: basic_aug + uniform tail inpainting (= Li et al. ECCV 2024 재현군).
- `aug_selective_inpaint`: basic_aug + selective tail inpainting (제안 기법, rarity×weakness priority).
- `*_qf` 접미사: CLIPScore 하위 percentile 제거 + budget 재보충한 품질 필터링 ablation
  (예: `aug_selective_inpaint_qf`).

**Copy-Paste 한계**: 이 데이터셋에는 segmentation mask가 없어 `aug_copy_paste`는 Ghiasi et
al.의 mask 단위 cutout이 아니라 **bbox 단위 rectangular patch**를 붙입니다. 붙여넣은 patch가
원본 배경 직사각형을 함께 가져오는 한계가 있으며, 논문에 명시해야 합니다. budget은
selective plan과 동일하게 맞춰 공정 비교합니다. `aug_rfs`는 Ultralytics에 sampler 주입이
어려워 repeat factor `r(c)=max(1, sqrt(t/f(c)))`를 이미지 복제로 물질화한 데이터셋 수준
구현입니다(`rfs.threshold`).

`real_only`를 제외한 모든 variant는 Ultralytics 기본 증강을 켭니다. Validation/test split은
모든 실험군에서 동일하게 유지하고, train split만 변경합니다. 파이프라인은 `real_only`와
`basic_aug`를 먼저 학습한 뒤, `basic_aug`의 planning split(val) per-class AP로 selective plan의
weakness score를 계산하고(`planning.baseline_variant`), 나머지 variant를 학습합니다. test AP는
augmentation 계획에 누수되지 않습니다.

## Background Inpainting Method

YOLO label을 pixel bbox로 변환한 뒤 모든 객체 bbox를 보호 영역으로 설정합니다. Inpainting mask는 white=background edit, black=protected object로 저장되며, bbox padding과 mask blur를 적용합니다. Diffusion 생성 후 원본 bbox crop을 다시 붙이고, bbox crop mean absolute difference, label count, image validity를 검사합니다.

VLM, CLIP, Grounding DINO, SAM 기반 필터링은 사용하지 않습니다. Diffusion model도 fine-tuning하지 않고 pretrained inpainting inference만 사용합니다.

## Metrics

주요 산출 metric은 다음과 같습니다.

- overall mAP50, mAP50-95
- Head AP, Medium AP, Tail AP
- Macro AP
- Head-Tail AP Gap
- Tail/Macro AP gain vs `basic_aug` (주 지표) 및 vs `real_only` (참고)
- AP gain per 100 synthetic images
- AP gain per generated image
- AP gain per training hour

생성품질 지표(`synthetic_quality.enabled`)는 클래스별 FID(합성 vs 해당 클래스 실제 train
이미지), 이미지별 CLIPScore(프롬프트 정합)와 LPIPS를 `outputs*/synthetic/quality_report.csv`,
`fid_by_class_<plan>.csv`로 저장합니다. `quality_filter.enabled`와 `aug_*_inpaint_qf` variant를
함께 켜면 CLIPScore 하위 percentile을 제거하고 제거분만큼 추가 생성(재보충)해 budget을
유지한 ablation을 돌릴 수 있습니다. Wilcoxon 검정과 seed CI는
`outputs*/analysis/statistical_tests.{csv,md}`에 저장됩니다.

## Expected Outputs

```text
/content/outputs/analysis/
/content/outputs/metrics/
/content/outputs/figures/
/content/outputs/synthetic/
/content/data/processed/base/
/content/data/processed/synthetic_inpaint/
/content/data/processed/synthetic_inpaint/images/train/
/content/data/processed/synthetic_inpaint/labels/train/
/content/data/experiments/
```

`outputs/analysis/dataset_summary.csv`에는 split별 이미지 수, bbox 수, 클래스 수, train imbalance ratio가 저장됩니다.

## Tests

실제 데이터셋 없이 빠르게 실행됩니다.

```bash
pytest tests/
```

## Troubleshooting

- `quality_report.csv`의 `clip_score`나 `fid_by_class_*.csv`의 `fid`가 전부 비어 있으면
  품질 지표 백엔드가 로드되지 않은 것입니다. FID는 `torch-fidelity`가 필수라
  `pip install "torchmetrics[image,multimodal]"`로 설치해야 하며(requirements.txt에 반영됨),
  실패 시 로그에 `[ERROR]`와 traceback이 출력됩니다.
- Kaggle download가 실패하면 `~/.kaggle/kaggle.json` 권한이 `600`인지 확인하세요.
- `ModuleNotFoundError: No module named 'ultralytics'`가 나오면 `pip install -r requirements.txt`를 먼저 실행하세요.
- Diffusion inference가 OOM이면 `configs/full.yaml`에서 `resolution`, `num_inference_steps`, synthetic budget을 낮추세요.
- Colab compute unit이 부족하면 먼저 `configs/smoke.yaml`와 `--dry-run-inpaint`로 파이프라인 구조를 검증하세요.
- Ultralytics API 변경으로 per-class AP 수집이 실패하면 `outputs/runs/*/results.csv` 기반 overall metric은 계속 저장되고, 경고 메시지가 출력됩니다.

## Ethical Use Statement

This project is limited to academic analysis of long-tailed object detection and annotation-preserving data augmentation using public datasets. It is not intended for operational surveillance, targeting, weapon-system integration, or real-time military deployment.
