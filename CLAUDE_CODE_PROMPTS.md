# Claude Code 수정 프롬프트 모음

리뷰 문서 `research_review_prior_work.md`의 P0~P2 수정사항을 코드에 반영하기 위한 프롬프트입니다.
**한 번에 하나씩, 순서대로** Claude Code에 입력하세요. 각 프롬프트는 이전 프롬프트의 완료를 전제로 합니다.
각 단계 후 `pytest tests/ -q`가 통과하는지 확인한 뒤 다음으로 넘어가세요.

---

## Prompt 1 — [P0] Inpainting dry-run 버그 수정 + 생성 검증 자동화

```text
이 저장소는 long-tailed military aircraft detection에서 tail class에 bbox-protected diffusion
background inpainting을 적용하는 연구 코드다. research_review_prior_work.md를 먼저 읽어라.

치명적 버그: pilot 실험에서 src/augment/inpaint_background.py가 생성한 이미지들이 원본과
사실상 동일했다(배경이 바뀌지 않음, dry-run 의심). 다음 작업을 수행해라:

1. src/augment/inpaint_background.py와 src/utils/image.py, src/augment/masks.py를 정독하고
   원본이 그대로 저장될 수 있는 모든 경로를 찾아라. 특히 의심 지점:
   - smoke/dry-run 모드 플래그가 full 실행에도 적용되는지
   - pipeline 호출 실패 시 fallback으로 원본을 copy하는 로직
   - paste_protected_regions()가 mask 전체를 덮어써버리는 버그 (mask_padding_ratio,
     blur 처리 포함해 실제로 bbox 외부 픽셀이 diffusion 결과로 교체되는지)
   - strength/steps 파라미터가 실제 pipeline 호출에 전달되는지
   원인을 찾으면 수정하고, 원인과 수정 내용을 BUGFIX_REPORT.md에 기록해라.

2. 생성 직후 자동 검증 모듈 src/eval/verify_generation.py를 새로 만들어라:
   - 각 생성 이미지에 대해: bbox 외부 영역의 mean absolute pixel diff, bbox 외부 SSIM,
     LPIPS(원본 vs 생성, torchmetrics 또는 lpips 패키지) 계산
   - bbox 외부 pixel diff가 임계값(config: verification.min_background_change, 기본 10.0)
     미만이면 "생성 실패"로 판정하고 해당 이미지를 재생성 대상으로 분류
   - 결과를 outputs*/synthetic/verification_report.csv로 저장하고, 실패율이
     verification.max_failure_rate(기본 0.05) 초과 시 RuntimeError로 파이프라인을 중단
   - inpaint_background.py의 메인 루프에 이 검증을 통합해라 (생성 → 즉시 검증 → 실패 시
     최대 max_retries_per_image 재시도)

3. bbox 내부 보호 검증도 강화해라: 기존 bbox_diff_threshold 기반 QC는 유지하되,
   verification_report.csv에 bbox 내부 diff도 함께 기록해라.

4. tests/test_verify_generation.py를 작성해라: 합성 픽셀 배열로 (a) 원본 복사본은 실패 판정,
   (b) 배경이 충분히 바뀐 이미지는 통과, (c) bbox 내부가 바뀐 이미지는 보호 위반 판정을 검증.

5. configs/{smoke,pilot,full,default}.yaml에 verification 섹션을 추가해라.

기존 테스트(pytest tests/)가 모두 통과해야 하고, requirements.txt에 필요 패키지를 추가해라.
```

---

## Prompt 2 — [P0] 실험군 재설계: basic_aug 기반 marginal gain 구조

```text
research_review_prior_work.md의 2.4절과 P0-2를 반영해 실험 설계를 바꾼다.

근거: arXiv 2403.07113은 YOLO 계열에서 mosaic/mixup 기본 증강이 리밸런싱 기법을 압도함을
보였고, X-Paste·Gen2Det·DiverGen 등 합성 증강 논문들은 모두 표준 증강이 켜진 강한 baseline
위에서 marginal gain을 보고한다. 현재의 상호배타적 5개 실험군(real_only, basic_aug,
tail_oversampling, uniform_tail_inpaint, selective_tail_inpaint)은 이 관행과 어긋난다.

작업:
1. src/run_pipeline.py와 src/augment/build_experiment_datasets.py, src/train/train_yolo.py를
   수정해 variant 정의를 다음으로 바꿔라:
   - real_only          : 기본 증강 OFF (참고용 하한선)
   - basic_aug          : YOLO 기본 증강 ON (주 baseline)
   - aug_oversample     : basic_aug + tail oversampling
   - aug_uniform_inpaint: basic_aug + uniform tail inpainting
   - aug_selective_inpaint: basic_aug + selective tail inpainting
   즉 모든 tail 기법 variant에서 YOLO 기본 증강(mosaic, mixup 등 Ultralytics 기본값)을 켠다.
   use_basic_aug 로직을 variant 이름 파싱 기반으로 리팩터링해라.

2. selective plan의 weakness_score가 사용하는 baseline AP는 real_only가 아니라 basic_aug
   run의 planning split AP를 쓰도록 run_pipeline.py의 의존 관계를 수정해라
   (basic_aug를 먼저 학습 → 그 per-class AP로 selective plan 생성 → 나머지 variant 학습).

3. configs/*.yaml의 experiments.variants를 새 이름으로 갱신하고, 기존 outputs 디렉토리와의
   호환성 문제가 있으면 마이그레이션하지 말고 새 실험은 새 이름으로 저장되게만 해라.

4. README.md의 Research Questions과 실험 프로토콜 설명을 새 설계에 맞게 갱신해라.
   RQ1을 "basic_aug 대비 marginal tail AP gain"으로 재정의해라.

5. tests/test_pipeline_smoke_components.py 등 영향받는 테스트를 갱신하고 전체 통과를 확인해라.
```

---

## Prompt 3 — [P1] 통계 검정 + seed 확장 + long-tail 그룹 지표

```text
research_review_prior_work.md P1-3을 반영한다.

1. configs/full.yaml의 detector.seeds를 [42, 43, 44]로 확장해라.

2. src/eval/statistics.py를 새로 만들어라:
   - 입력: outputs*/metrics/의 per-class AP (variant × seed × class)
   - variant 쌍별 비교: 클래스별 AP를 대응 표본으로 Wilcoxon signed-rank test
     (scipy.stats.wilcoxon), tail 클래스 subset과 전체 클래스 각각 수행
   - seed별 macro AP의 mean ± std, 95% CI (t-분포 기반) 계산
   - 결과를 outputs*/analysis/statistical_tests.csv 및 markdown 요약으로 저장

3. src/eval/compute_long_tail_metrics.py를 확장해라:
   - LVIS 관행에 따라 train instance 수 기준 3그룹(head/mid/tail)을 명시적 threshold로 정의
     (config: tail.group_thresholds, 예: tail<=N1, mid<=N2)하고 그룹별 AP(APt/APm/APh)를
     variant × seed로 집계해라. 기존 bottom_percent 방식과 병행 보고.

4. src/eval/plot_results.py에 추가: variant별 클래스 AP 변화(Δ vs basic_aug)의
   scatter/violin 플롯 (RQ3 head 손상 여부 시각화, Gen2Det 방식).

5. tests/test_statistics.py 작성: 알려진 배열로 Wilcoxon 결과와 CI 계산 검증.
   requirements.txt에 scipy 추가.
```

---

## Prompt 4 — [P1] 생성품질 정량 지표(FID/CLIPScore) + 품질 필터링

```text
research_review_prior_work.md P1-5를 반영한다. 근거: Gen2Det의 image/instance-level filtering,
DiverGen의 다양성 분석, arXiv 2406.05184의 CLIPScore 상위 50% 필터링 관행.

1. src/eval/synthetic_quality.py를 새로 만들어라:
   - FID: 합성 이미지 집합 vs 해당 클래스의 실제 train 이미지 집합 (torchmetrics의
     FrechetInceptionDistance, 클래스별 + 전체)
   - CLIPScore: 생성 프롬프트와 생성 이미지 간 정합도 (torchmetrics CLIPScore 또는
     open_clip), 이미지별 기록
   - 결과: outputs*/synthetic/quality_report.csv (image, class, prompt, clip_score, lpips)
     + 클래스별 FID 요약 csv
   GPU 없으면 CPU로 동작하되 --max-images 샘플링 옵션을 둬라.

2. 품질 필터링 옵션을 build_experiment_datasets.py에 추가해라:
   config: quality_filter.enabled, quality_filter.clip_score_percentile (기본 50 = 하위 50% 제거).
   필터링 ON/OFF가 ablation이 되도록 variant 접미사(예: aug_selective_inpaint_qf)로
   실험군을 추가 생성할 수 있게 해라. budget은 필터링 후 남는 수량 기준으로 재보충
   (필터 제거분만큼 추가 생성)하는 로직도 포함해라.

3. tests/test_synthetic_quality.py: 더미 이미지로 report 스키마와 필터링 로직 검증.
```

---

## Prompt 5 — [P1] 표준 long-tail baseline 추가 (Copy-Paste, RFS)

```text
research_review_prior_work.md P1-4를 반영해 baseline 2개를 추가한다.

1. aug_copy_paste variant: Ghiasi et al. CVPR 2021의 Simple Copy-Paste를 tail 클래스에 적용.
   src/augment/copy_paste_tail.py를 새로 만들어라:
   - tail 클래스 인스턴스를 bbox 크롭으로 추출해 풀을 만들고, 랜덤 train 이미지에
     스케일/플립 jitter와 함께 붙여넣고 라벨을 갱신
   - 합성 budget은 selective/uniform inpaint와 동일하게 맞춰 공정 비교
   - segmentation mask가 없으므로 bbox 단위 rectangular paste로 구현하고, 이 한계를
     docstring과 README에 명시해라

2. aug_rfs variant: Repeat Factor Sampling (LVIS, Gupta et al. CVPR 2019).
   Ultralytics YOLO는 sampler 주입이 어려우므로 데이터셋 수준 구현으로 대체해라:
   repeat factor r(c)=max(1, sqrt(t/f(c)))를 이미지 단위로 계산해 train 리스트를 복제하는
   방식(threshold t는 config). src/augment/repeat_factor_sampling.py로 구현.

3. 두 variant를 run_pipeline.py, build_experiment_datasets.py, configs에 통합하고
   tests/test_copy_paste.py, tests/test_rfs.py를 작성해라.
```

---

## Prompt 6 — [P2] Ablation 설정: alpha 스윕, budget 스케일, 프롬프트 다양성

```text
research_review_prior_work.md P2-7, P2-9를 반영한다.

1. run_pipeline.py에 --ablation 모드를 추가해라:
   - alpha 스윕: selective_generation.alpha를 [0.0, 0.25, 0.5, 0.75, 1.0]로 바꿔가며
     aug_selective_inpaint만 재실행 (α=0 → weakness-only, α=1 → rarity-only).
     synthetic 생성은 plan이 같으면 캐시 재사용.
   - budget 스케일: total_synthetic_budget × [0.5, 1, 2, 4] 스윕.
   - 프롬프트 다양성: diffusion.prompts를 첫 1개만 쓰는 single_prompt 모드 vs 전체 5개.
   각 ablation은 configs/ablation_{alpha,budget,prompt}.yaml로 분리해라.

2. 결과 집계가 ablation 축을 인식하도록 collect_yolo_metrics.py와 statistics.py의
   출력 스키마에 ablation 컬럼을 추가해라.

3. ablation은 seed 1개(42)로 돌리되 최종 비교 지점(기본 설정)은 3 seeds 결과를 재사용해라.
```

---

## Prompt 7 — [P2] 2차 데이터셋(MAR20) + 2차 모델 지원

```text
research_review_prior_work.md P2-8을 반영한다.

1. src/data/prepare_mar20.py를 새로 만들어라: MAR20 데이터셋(원격탐사 군용기 20클래스,
   HBB 라벨)을 YOLO 포맷으로 변환하고 train/val/test를 분리. 다운로드는 수동(사용자가
   압축 해제한 경로를 --src로 받음)으로 가정해라.

2. configs/mar20.yaml을 만들어라: full.yaml과 동일 구조, 데이터 경로와 클래스 수만 다름.
   tail 정의(threshold)는 MAR20의 인스턴스 분포를 분석해 자동 산출되도록
   src/data/analyze_long_tail.py를 재사용해라.

3. detector.model을 리스트로 받을 수 있게 확장해라 (예: ["yolov8n.pt", "yolov8s.pt"]).
   run_pipeline이 model × variant × seed 조합을 순회하고 출력 경로에 model 이름을 포함.

4. README에 MAR20 실험 실행 방법을 추가해라.
```

---

## 사용 팁

- 각 프롬프트 실행 후: `git add -A && git commit -m "..."` 으로 단계별 커밋을 남기세요.
- Claude Code가 리뷰 문서를 참조하도록 매 세션 시작 시 "research_review_prior_work.md를 먼저 읽어라"가 포함되어 있습니다. 세션을 이어서 쓸 경우 생략해도 됩니다.
- Prompt 1이 가장 중요합니다. 버그 원인 보고(BUGFIX_REPORT.md)를 직접 읽고 납득한 후에만 Prompt 2로 진행하세요.
