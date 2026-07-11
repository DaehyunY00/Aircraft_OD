# 사용자 작업 순서 가이드 (USER ACTION PLAN)

`research_review_prior_work.md`(리뷰)와 `CLAUDE_CODE_PROMPTS.md`(코드 수정 프롬프트)를 바탕으로,
논문 제출까지 사용자가 직접 수행해야 하는 작업을 순서대로 정리한 문서입니다.

체크박스를 채워가며 진행하세요. 예상 소요는 Colab GPU(T4/A100) 기준입니다.

---

## Phase 0. 준비 (0.5일)

- [ ] 0-1. 현재 코드 상태를 git commit으로 백업 (`git add -A && git commit -m "pre-review baseline"`)
- [ ] 0-2. `research_review_prior_work.md` 정독 — 특히 2.4절(basic_aug 압도 현상)과 P0 항목
- [ ] 0-3. 기존 pilot 결과(outputs_pilot/)는 폐기 대상임을 인지. 삭제하지 말고 `outputs_pilot_deprecated/`로 이름만 변경

## Phase 1. 코드 수정 — Claude Code 실행 (1~2일)

`CLAUDE_CODE_PROMPTS.md`의 프롬프트를 순서대로 실행합니다.

- [ ] 1-1. **Prompt 1** 실행 (dry-run 버그 수정 + 생성 검증 자동화)
- [ ] 1-2. `BUGFIX_REPORT.md`를 직접 읽고 버그 원인이 납득되는지 확인. 납득 안 되면 Claude Code에 추가 질문. **이 단계를 통과하기 전에는 절대 다음으로 진행하지 말 것**
- [ ] 1-3. `pytest tests/ -q` 통과 확인 후 커밋
- [ ] 1-4. **Prompt 2** 실행 (실험군 재설계) → 테스트 통과 확인 후 커밋
- [ ] 1-5. **Prompt 3** 실행 (통계 검정 + seed 3개 + 그룹 지표) → 커밋
- [ ] 1-6. **Prompt 4** 실행 (FID/CLIPScore + 품질 필터링) → 커밋
- [ ] 1-7. **Prompt 5** 실행 (Copy-Paste, RFS baseline) → 커밋
- [ ] 1-8. (Q1 도전 시) **Prompt 6, 7** 실행 → 커밋

## Phase 2. 생성 파이프라인 검증 (0.5일, GPU 필요)

- [ ] 2-1. Colab에서 `python src/run_pipeline.py --config configs/smoke.yaml` 실행
- [ ] 2-2. `outputs/synthetic/verification_report.csv` 확인: bbox 외부 pixel diff가 임계값 이상인지, 실패율 5% 미만인지
- [ ] 2-3. **육안 확인**: contact sheet/review sheet에서 생성 이미지 20장 이상을 원본과 나란히 비교. 배경이 실제로 바뀌었고, 항공기 형상이 보존되었는지 직접 확인
- [ ] 2-4. 프롬프트 5종별로 생성 샘플 품질 확인. 어색한 프롬프트(예: 항공기와 안 맞는 배경)가 있으면 configs의 prompts 수정
- [ ] 2-5. 여기서 문제 발견 시 Phase 1로 돌아가 수정. 통과 시 커밋

## Phase 3. Pilot 재실험 (1~2일, GPU)

- [ ] 3-1. `configs/pilot.yaml`을 새 variant 이름으로 갱신했는지 확인
- [ ] 3-2. pilot 실행 (seed 1개, 16-class subset): 5개 variant 학습
- [ ] 3-3. 확인 포인트: (a) aug_selective_inpaint의 tail AP가 basic_aug 대비 +인지, (b) verification_report와 quality_report에 이상 없는지
- [ ] 3-4. **Go/No-Go 판단**: pilot에서 inpainting 계열이 basic_aug 대비 tail AP 개선을 전혀 못 보이면, full 실험 전에 원인 분석(생성 품질? budget 부족? 프롬프트?)을 먼저 할 것. 개선이 보이면 진행

## Phase 4. Full 실험 (3~7일, GPU 장시간)

- [ ] 4-1. `configs/full.yaml` 확인: seeds [42,43,44], 43 classes, 새 variant 7종(real_only, basic_aug, aug_oversample, aug_rfs, aug_copy_paste, aug_uniform_inpaint, aug_selective_inpaint)
- [ ] 4-2. full 실행. Colab 세션 끊김 대비: variant×seed 단위로 재개 가능한지 사전 확인 (tests/test_training_resume.py 관련 기능)
- [ ] 4-3. 완료 후 `outputs_full/analysis/statistical_tests.csv` 확인: selective vs uniform, selective vs basic_aug의 Wilcoxon p-value
- [ ] 4-4. 그룹별 AP(APt/APm/APh) 표와 head 손상 여부 플롯 확인 (RQ3)

## Phase 5. Ablation (2~4일, GPU) — Q1 도전 시 필수

- [ ] 5-1. alpha 스윕 실행 (`configs/ablation_alpha.yaml`) — priority score가 기여로 인정받는 핵심 실험
- [ ] 5-2. budget 스케일 스윕 실행
- [ ] 5-3. 프롬프트 다양성(single vs multi) 실행
- [ ] 5-4. 품질 필터링 ON/OFF 비교 실행
- [ ] 5-5. (선택) MAR20 데이터셋 다운로드 → `prepare_mar20.py` 변환 → `configs/mar20.yaml`로 주요 variant 재실행
- [ ] 5-6. (선택) YOLOv8s로 주요 variant 재실행

## Phase 6. 분석 및 논문 작성 (2~3주)

- [ ] 6-1. 전체 결과 표 작성: variant × (mAP, APt/APm/APh) mean±std + p-value
- [ ] 6-2. 생성품질 표 작성: 클래스별 FID, CLIPScore 분포
- [ ] 6-3. Related Work 작성 — `research_review_prior_work.md`의 1절 표를 골격으로 사용. **Li et al. ECCV 2024 (arXiv 2408.00350)와의 차별화 문단은 반드시 별도로** (selective vs uniform이 이 논문의 생명선)
- [ ] 6-4. 방법론 서술 시 priority score를 BSGAL(ICML 2024) 대비 "재학습 없는 경량 클래스 수준 배분"으로 포지셔닝
- [ ] 6-5. 한계(limitations) 명시: 단일/이중 데이터셋, bbox 단위 rectangular copy-paste, SD-1.5 고정 등
- [ ] 6-6. 동일 Kaggle 데이터셋 선행 논문(2025, YOLO/RT-DETR 비교)과 프로토콜 차이 명시

## Phase 7. 저널 제출

- [ ] 7-1. 결과 수준에 따라 타겟 결정 (리뷰 4절 참조):
  - Phase 5까지 완료 + 유의한 개선 → **Q1 도전**: Expert Systems with Applications, Engineering Applications of AI, Defence Technology
  - Phase 4까지만 완료 → **Q2 안정권**: Image and Vision Computing, Machine Vision and Applications, IEEE Access
  - MAR20 포함 시 MDPI Remote Sensing도 후보
- [ ] 7-2. 코드 공개 준비 (GitHub public repo + README 정리) — 응용 저널에서도 가산점
- [ ] 7-3. 투고

---

## 요약: 지금 당장 할 일 3가지

1. git 백업 후 Claude Code에서 **Prompt 1** 실행
2. `BUGFIX_REPORT.md` 읽고 dry-run 원인 확인
3. smoke test로 생성 이미지가 실제로 바뀌는지 육안 + verification_report로 이중 확인
