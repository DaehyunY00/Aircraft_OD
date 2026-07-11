# Fable 통합 프롬프트 — SCI Q2 게재를 위한 P0+P1 구현

작성 2026-07-11 | 대상 모델: Claude Fable 5 | 범위: P0+P1 (Q2 안정권) | 구조: 단일 통합 프롬프트

---

## Part 1. 현황 검토 요약 (사용자용, fable에 넣지 않아도 됨)

**코드 실제 상태 (2026-07-11 확인).** `research_review_prior_work.md`, `CLAUDE_CODE_PROMPTS.md`, `USER_ACTION_PLAN.md`는 완성돼 있으나 **코드는 아직 수정 전 원본 상태**다. git 커밋 0개, `verify_generation.py`·`statistics.py`·`synthetic_quality.py`·`copy_paste_tail.py`·`BUGFIX_REPORT.md` 없음, config는 여전히 smoke/pilot/full/default 4개뿐. 즉 리뷰가 지적한 P0~P1 항목이 하나도 반영되지 않았다.

**Inpainting 버그 코드 레벨 진단 (신규).** 리뷰의 "생성 이미지가 원본과 사실상 동일" 현상에 대해 실제 코드에서 유력 원인 3가지를 특정했다. 이 진단을 fable 프롬프트에 이미 반영했다.

1. **dry-run 경로 오염 (1순위 의심).** `src/augment/inpaint_background.py`의 `generate_from_plan(..., dry_run=...)`은 `dry_run=True`일 때 line 229–230에서 `generated = original.copy()`로 원본을 그대로 복사한다. `src/run_pipeline.py --dry-run-inpaint`가 이 경로를 켠다. pilot이 이 플래그로 돌았다면 전 synthetic이 원본 사본이 되어 "배경 안 바뀜"이 정확히 재현된다.
2. **already_exists 분기의 무검증 통과.** line 179–210에서 이미 파일이 있으면 `bbox_pixel_diff=0.0`, `accepted=True`로 재검증 없이 로깅한다. resume 실행 시 dry-run으로 만들어진 stale 사본을 그대로 정상 생성물로 승인한다.
3. **diffusion 파라미터/마스크 극성 확인 필요.** `strength=0.85`+`num_inference_steps=20`이 실제 pipe 호출에 전달되는지, 마스크 극성(white=편집/black=보호)이 `runwayml/stable-diffusion-inpainting`에 맞는지, `_run_inpaint`의 LANCZOS 마스크 resize가 경계를 뭉개는지 검증 필요. `paste_protected_regions`는 bbox만 덮으므로 전체 동일 현상의 원인은 아님(2차 확인용).

**Q2 게재에 추가로 필요한 실험·설계 (P0+P1, 본인 리뷰 기준).**

- **P0-1 생성 신뢰성.** 위 버그를 수정하고, "배경이 실제로 바뀌었다"를 bbox 외부 픽셀 diff·SSIM·LPIPS로 매 이미지 자동 정량 검증. 기존 pilot 결과는 폐기.
- **P0-2 실험 설계 재편.** 현재 상호배타 5군을 basic_aug 공통기반의 marginal-gain 구조로 재설계(real_only / basic_aug / aug_oversample / aug_uniform_inpaint / aug_selective_inpaint). selective의 weakness_score 기준선을 real_only→basic_aug per-class AP로 교체. (근거: arXiv 2403.07113 — YOLO에서 기본 증강이 리밸런싱을 압도, 합성 논문들은 강 baseline 위 marginal gain 보고)
- **P1-1 통계.** seeds ≥3 (42/43/44), variant쌍 per-class AP paired Wilcoxon signed-rank + mean±std + 95% CI.
- **P1-2 그룹 지표.** LVIS식 train 인스턴스 수 기준 명시적 threshold로 head/mid/tail 3군 APh/APm/APt 집계. RQ3(head 손상 여부)는 Δvs basic_aug scatter/violin.
- **P1-3 생성품질 정량화.** 클래스별 FID(합성 vs 실제 배경), CLIPScore, 품질 필터링(CLIPScore 하위 50% 제거) ablation. (근거: Gen2Det filtering, 2406.05184)
- **P1-4 표준 baseline.** Copy-Paste(Ghiasi CVPR2021), Repeat Factor Sampling(LVIS) 추가 — "굳이 diffusion이 필요한가"에 답하기 위한 필수 대조군.
- **필수 프레이밍.** Li et al. ECCV 2024(arXiv 2408.00350)와의 차별화가 논문의 생명선 — uniform 배경증강(=Li et al. 재현)을 D군에 두고 selective(E)가 같은 budget에서 우월함을 통계적으로 입증.

P2(alpha 스윕, budget 스케일, 프롬프트 다양성, MAR20 2차 데이터셋, 2차 모델)는 Q1 도전 시에만. 이 프롬프트에는 포함하지 않되, 확장 가능하도록 코드 구조만 열어둠.

---

## Part 2. Fable에 붙여넣을 통합 프롬프트

> 아래 코드블록 전체를 그대로 fable에 입력하세요. 저장소 루트에서 실행한다고 가정합니다.

```text
너는 이 저장소(Military_OD)의 시니어 ML 엔지니어다. 이 저장소는 long-tailed military
aircraft detection에서 tail class에만 bbox-protected diffusion background inpainting을
적용하는 SCI Q2 목표 연구 코드다. 목표는 리뷰 문서가 지적한 P0+P1 항목을 모두 구현해
논문 실험을 신뢰성 있게 돌릴 수 있는 상태로 만드는 것이다.

=== 먼저 읽어라 (순서 준수) ===
1. research_review_prior_work.md  — 특히 2.4절(basic_aug 압도 현상), 3절 P0/P1 우선순위
2. README.md  — 파이프라인 구조와 실험 프로토콜
3. src/run_pipeline.py, src/augment/inpaint_background.py, src/augment/masks.py,
   src/utils/image.py, src/augment/build_experiment_datasets.py,
   src/augment/oversample_tail.py, src/train/train_yolo.py,
   src/eval/collect_yolo_metrics.py, src/eval/compute_long_tail_metrics.py,
   src/eval/plot_results.py, src/data/analyze_long_tail.py, configs/*.yaml, tests/*

=== 작업 원칙 (전 단계 공통) ===
- 단계(Phase) 순서를 반드시 지켜라. Phase A를 완료하고 BUGFIX_REPORT.md로 근본원인을
  납득시키기 전에는 Phase B 이후로 넘어가지 마라.
- 각 Phase 종료 시: (a) `pytest tests/ -q` 전체 통과, (b) `git add -A && git commit`으로
  단계별 커밋. 커밋 메시지에 Phase 번호와 요약을 남겨라.
- 새 의존성은 requirements.txt에 추가하고, GPU 없이도 최소 동작(CPU fallback + 샘플링
  옵션)하게 만들어라.
- val/test split은 전 variant에서 고정하고 train split만 바꾼다. planning(selective plan
  생성)은 val split AP만 쓰고 test AP가 augmentation 계획에 누수되지 않게 유지해라.
- 하드코딩 금지. 새 파라미터는 전부 configs/*.yaml 로 노출하고 4개 config
  (smoke/pilot/full/default) 모두에 일관되게 반영해라.
- 큰 파일은 한 번에 재작성하지 말고 함수 단위로 편집해라. 기존 통과 테스트를 깨지 마라.

=== 연구 맥락 (구현 판단 기준) ===
- 핵심 메커니즘(bbox 보호 + 배경 diffusion inpainting)은 Li et al. ECCV 2024
  (arXiv 2408.00350)에서 이미 uniform 형태로 제안됨. 이 연구의 novelty는 (1) tail-선택적
  적용, (2) rarity×weakness priority score 배분, (3) 군용기 도메인. 따라서 "uniform vs
  selective 같은 budget 비교"와 "생성이 실제로 유효함의 정량 증명"이 논문의 생명선이다.
  구현은 이 두 축을 흔들리지 않게 뒷받침해야 한다.
- 합성 증강 논문 관행(X-Paste/Gen2Det/DiverGen): 기본 증강을 끈 채 비교하지 말고 강한
  baseline(basic_aug) 위에서 marginal gain을 측정한다. 실험 설계를 여기에 맞춘다.

=====================================================================
Phase A [P0-1] — Inpainting 생성 버그 진단 및 수정
=====================================================================
사전 진단(이미 코드에서 확인된 유력 원인 3가지, 이걸 출발점으로 검증하되 맹신하지 마라):
  (1) dry-run 경로 오염: inpaint_background.py generate_from_plan의 dry_run=True 분기가
      `generated = original.copy()`로 원본을 복사한다. run_pipeline.py --dry-run-inpaint가
      이를 켠다. pilot이 이 플래그로 돌아 전 synthetic이 원본 사본이 됐을 가능성이 가장
      크다. dry-run이 full/pilot 실행에 새어들 수 있는 모든 경로를 추적해라.
  (2) already_exists 무검증 통과: 파일이 이미 있으면 bbox_pixel_diff=0.0, accepted=True로
      재검증 없이 승인한다(약 line 179–210). resume 시 stale 사본을 정상 생성물로 통과시킨다.
  (3) diffusion 파라미터/마스크 극성: strength·num_inference_steps가 실제 pipe 호출에
      전달되는지, 마스크 극성(white=255=편집 배경, black=0=보호 객체)이
      runwayml/stable-diffusion-inpainting 규약과 맞는지, _run_inpaint의 LANCZOS 마스크
      resize가 경계를 뭉개 배경 변화를 억제하는지 확인. paste_protected_regions는 bbox만
      덮으므로 "전체가 원본과 동일" 현상의 직접 원인은 아니다(2차 확인용).

할 일:
  1. 위 3개 파일 + masks.py를 정독해 원본이 그대로 저장될 수 있는 모든 코드 경로를 찾아라.
  2. 근본원인을 수정해라. 특히: (a) 비-dry-run 실행에서는 절대 원본 복사 경로로 빠지지
     않도록 하고, (b) already_exists 분기라도 배경 변화가 임계 미만이면 재생성 대상으로
     돌리도록 고쳐라. dry-run은 "구조 점검 전용"임을 명시적 로그/경고로 남겨라.
  3. BUGFIX_REPORT.md 작성: 재현 조건, 근본원인(코드 위치·라인), 수정 내용, 수정 후
     "동일 입력에서 배경이 실제로 바뀜"을 어떻게 보장하는지 서술.
  4. 기존 pilot 산출물은 삭제하지 말고 outputs_pilot_deprecated/ 로 이름만 바꿔 보존하고,
     README에 "구버전 결과는 폐기됨"을 명시해라.
Phase A gate: BUGFIX_REPORT.md가 근본원인을 코드 라인 수준으로 특정하지 못하면 멈추고
  사용자에게 확인을 요청해라.

=====================================================================
Phase B [P0-2] — 생성 검증 자동화 (verify_generation)
=====================================================================
  1. src/eval/verify_generation.py 신규 작성. 각 생성 이미지에 대해:
     - bbox '외부'(배경) 영역의 mean absolute pixel diff (원본 vs 생성)
     - bbox 외부 SSIM, LPIPS(원본 vs 생성; torchmetrics 또는 lpips 패키지, CPU 동작)
     - bbox '내부'(보호 영역) diff도 함께 기록(보호 위반 감시)
  2. 판정: 배경 diff < config verification.min_background_change(기본 10.0)이면 "생성 실패".
     inpaint_background.py 메인 루프에 통합: 생성→즉시 검증→실패 시
     verification.max_retries_per_image(기본 2) 재시도. already_exists 분기에도 검증 적용.
  3. 결과를 outputs*/synthetic/verification_report.csv로 저장. 전체 실패율이
     verification.max_failure_rate(기본 0.05) 초과 시 RuntimeError로 파이프라인 중단.
  4. tests/test_verify_generation.py: 합성 픽셀 배열로 (a) 원본 사본=실패 판정,
     (b) 배경만 충분히 바뀐 이미지=통과, (c) bbox 내부가 바뀐 이미지=보호 위반 판정 검증.
  5. configs 4종에 verification 섹션 추가.

=====================================================================
Phase C [P0-3] — 실험 설계 재편: basic_aug 기반 marginal gain
=====================================================================
  1. variant 정의를 아래로 바꾸고 use_basic_aug 로직을 variant 이름 파싱 기반으로 리팩터:
     - real_only            : 기본 증강 OFF (참고용 하한선)
     - basic_aug            : YOLO 기본 증강 ON (주 baseline)
     - aug_oversample       : basic_aug + tail oversampling
     - aug_uniform_inpaint  : basic_aug + uniform tail inpainting  (= Li et al. 재현군)
     - aug_selective_inpaint: basic_aug + selective tail inpainting (제안 기법)
     모든 tail 기법 variant에서 Ultralytics 기본 증강(mosaic/mixup 등)을 켠다.
  2. selective plan의 weakness_score 기준선을 real_only가 아니라 basic_aug run의
     planning(val) split per-class AP로 교체. run_pipeline 의존순서를 basic_aug 먼저 학습
     → 그 AP로 selective plan 생성 → 나머지 variant 학습으로 재배열해라.
  3. configs/*.yaml의 experiments.variants를 새 이름으로 갱신. 기존 outputs와 이름 충돌은
     마이그레이션하지 말고 새 이름으로만 저장되게 해라.
  4. README의 Research Questions·프로토콜 갱신. RQ1을 "basic_aug 대비 marginal tail AP
     gain"으로 재정의. RQ2를 "같은 budget에서 selective > uniform(=Li et al.)"로 명시.
  5. tests/test_pipeline_smoke_components.py 등 영향 테스트 갱신 후 전체 통과 확인.

=====================================================================
Phase D [P1-1,2] — 통계 검정 + seed 확장 + long-tail 그룹 지표
=====================================================================
  1. configs/full.yaml의 detector.seeds를 [42, 43, 44]로 확장.
  2. src/eval/statistics.py 신규:
     - 입력: outputs*/metrics/의 per-class AP (variant × seed × class)
     - variant 쌍별(특히 selective vs uniform, selective vs basic_aug) 클래스별 AP를 대응
       표본으로 Wilcoxon signed-rank(scipy.stats.wilcoxon). tail subset·전체 각각 수행.
     - seed별 macro AP의 mean±std, 95% CI(t-분포). 결과를
       outputs*/analysis/statistical_tests.csv + markdown 요약으로 저장.
  3. src/eval/compute_long_tail_metrics.py 확장: LVIS 관행에 따라 train 인스턴스 수 기준
     명시적 threshold 3군(config tail.group_thresholds, 예: tail<=N1, mid<=N2)으로
     APh/APm/APt를 variant×seed 집계. 기존 bottom_percent 방식과 병행 보고.
  4. src/eval/plot_results.py에 RQ3용 플롯 추가: variant별 클래스 AP의 Δ(vs basic_aug)
     scatter/violin (head 손상 없이 tail 개선 여부 시각화, Gen2Det 방식).
  5. tests/test_statistics.py: 알려진 배열로 Wilcoxon·CI 검증. requirements.txt에 scipy 추가.

=====================================================================
Phase E [P1-3] — 생성품질 정량 지표(FID/CLIPScore) + 품질 필터링
=====================================================================
  1. src/eval/synthetic_quality.py 신규:
     - 클래스별 FID: 합성 이미지 집합 vs 해당 클래스 실제 train 이미지
       (torchmetrics FrechetInceptionDistance), 클래스별 + 전체.
     - CLIPScore: 생성 프롬프트 vs 생성 이미지 정합(torchmetrics CLIPScore 또는 open_clip),
       이미지별 기록. LPIPS도 함께.
     - 결과: outputs*/synthetic/quality_report.csv(image,class,prompt,clip_score,lpips) +
       클래스별 FID 요약 csv. GPU 없으면 CPU 동작 + --max-images 샘플링 옵션.
  2. build_experiment_datasets.py에 품질 필터링 추가: config quality_filter.enabled,
     quality_filter.clip_score_percentile(기본 50=하위 50% 제거). 필터 ON/OFF가 ablation이
     되도록 variant 접미사(예: aug_selective_inpaint_qf)로 추가 실험군 생성 가능하게 하고,
     필터로 제거된 만큼 budget을 재보충(추가 생성)하는 로직 포함.
  3. tests/test_synthetic_quality.py: 더미 이미지로 report 스키마·필터링 로직 검증.

=====================================================================
Phase F [P1-4] — 표준 long-tail baseline 추가 (Copy-Paste, RFS)
=====================================================================
  1. aug_copy_paste variant (Ghiasi et al. CVPR 2021, Simple Copy-Paste):
     src/augment/copy_paste_tail.py 신규 — tail 인스턴스를 bbox 크롭으로 추출해 풀 구성,
     랜덤 train 이미지에 scale/flip jitter와 함께 붙이고 라벨 갱신. budget은
     selective/uniform inpaint와 동일하게 맞춰 공정 비교. segmentation mask가 없으므로 bbox
     단위 rectangular paste로 구현하고 이 한계를 docstring과 README에 명시.
  2. aug_rfs variant (Repeat Factor Sampling, Gupta et al. CVPR 2019):
     src/augment/repeat_factor_sampling.py 신규 — Ultralytics는 sampler 주입이 어려우므로
     데이터셋 수준으로 구현. repeat factor r(c)=max(1, sqrt(t/f(c)))(threshold t는 config)를
     이미지 단위로 계산해 train 리스트를 복제.
  3. 두 variant를 run_pipeline.py·build_experiment_datasets.py·configs에 통합하고
     tests/test_copy_paste.py, tests/test_rfs.py 작성.

=====================================================================
최종 산출물 (완료 후 요약 보고)
=====================================================================
- BUGFIX_REPORT.md, 신규 모듈 5개(verify_generation, statistics, synthetic_quality,
  copy_paste_tail, repeat_factor_sampling), 신규 테스트 5개, 갱신된 configs 4종·README.
- 최종 variant 목록: real_only, basic_aug, aug_oversample, aug_rfs, aug_copy_paste,
  aug_uniform_inpaint, aug_selective_inpaint (+선택적 _qf 접미사).
- 완료 시: 변경 파일 목록, 새 config 파라미터 표, 각 Phase 커밋 해시, `pytest tests/ -q`
  최종 결과, 그리고 "smoke 실행 시 verification_report.csv에서 배경 변화가 임계 이상으로
  기록되는지"를 사용자가 GPU에서 확인할 방법을 요약해 보고해라.
- P2(alpha 스윕/budget 스케일/프롬프트 다양성/MAR20/2차 모델)는 구현하지 말되, 각 모듈이
  나중에 축을 추가하기 쉽게(예: variant 접미사, config 축) 구조만 열어두고 그 지점을
  주석으로 표시해라.
```

---

## Part 3. 사용 팁

- **Phase A 게이트를 지키세요.** `BUGFIX_REPORT.md`를 직접 읽고 dry-run 원인이 납득된 뒤에만 다음 단계로. 여기서 막히면 fable에 "재현 로그와 라인 번호를 다시 보여줘"라고 요청.
- **GPU 검증은 별도.** fable은 코드 구현·단위테스트까지만. 실제 diffusion 생성이 배경을 바꾸는지는 Colab GPU에서 `python src/run_pipeline.py --config configs/smoke.yaml` 후 `outputs/synthetic/verification_report.csv`와 review_sheet 육안 확인이 필요합니다.
- **단계별 커밋**이 프롬프트에 포함돼 있으니, 문제가 생기면 특정 Phase 커밋으로 되돌릴 수 있습니다.
- Q1 도전으로 확장할 때는 이 프롬프트 완료 후 별도 P2 프롬프트(alpha 스윕 등)를 요청하세요. 코드 구조는 이미 확장 가능하게 열어두도록 지시돼 있습니다.
