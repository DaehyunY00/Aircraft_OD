# IJASS 확인 실험 — 실행 계획 (CODEX_GCLOUD_PROMPT.md 수정판)

2026-08-04, Claude Code 실행. 원 프롬프트를 저장소 실상에 맞춰 수정한 내용과 근거를 기록한다.

## 원 프롬프트에서 수정한 사항

1. **arm 이름**: 프롬프트의 `aug_tail_uniform_inpaint` 등 새 이름 체계 대신 기존 명명 유지.
   2×2 매핑 — tail×uniform=`aug_uniform_inpaint`, tail×weighted=`aug_selective_inpaint`,
   weak×weighted=`aug_weakness_inpaint`, weak×uniform=**`aug_weakuniform_inpaint`(신규)**.
   신규 구현은 1개 arm뿐이다.
2. **weak 클래스 집합 고정**: val AP 13위/14위 margin이 0.0008이라 재측정 시 집합이 바뀔 수
   있어 `selective_generation.weakness_class_ids`로 설계에 freeze했다
   (`[9,14,16,17,18,19,20,21,22,24,27,30,34]`, 본 실험과 동일, tail과 교집합 0).
3. **RFS seed 42 재사용 포기**: `experiments_data` 경로를 새로 분리(구 할당으로 빌드된
   데이터셋 오염 방지)하면서 fingerprint의 data_yaml이 불일치 → 프롬프트의 "정확히 일치할
   때만 재사용" 조건에 따라 seed 42 포함 3 seed 전부 신규 학습(+~$2).
4. **baseline 재사용**: real_only/basic_aug 42/43/44는 base 데이터셋(내용 불변, Kaggle 원본
   + normalize seed 42 결정적 재현)으로 학습된 기존 run을 복사 재사용. 구버전 run이라
   fingerprint 필드가 없어 warn-reuse 경로로 동작하며, args의 model/imgsz/epochs가 현재
   설정과 일치함을 확인했다.
5. **인프라 현실**: 기존 VM `military-od`와 주 버킷 `gs://military-od-d522190f`는 이미
   삭제됨 → 새 VM `military-od-conf` + 새 버킷 `gs://military-od-confirmatory` 생성.
   합성 pool 3종(3,026장)은 `gs://military-od-gate/synthetic_full/`에 보존돼 있어 재사용.
   baseline weights와 rejected 마커(738장)는 로컬 Drive에서 업로드.
6. **무인 실행 방식**: SSH 키가 passphrase 보호라 비대화형 SSH 불가(검증된 제약) →
   startup-script + GCS 마커 폴링 + 부팅당 11h20m 데드라인 self-stop 패턴. 로컬
   오케스트레이터(`scripts/run_confirmatory_gcp.sh watch`)가 5분 간격으로 재시작/비용
   상한을 관리한다.
7. **allocator**: `_allocate_by_weights`를 결정적 capped largest-remainder로 교체.
   버그 확인됨 — 구 코드는 uniform 1000/13에서 잔여 12장을 첫 클래스에 몰아 88/76×12를
   만들었다. 신 코드는 77×12+76×1. (uniform pool은 11장 신규 생성 + 11장 초과 제외로 수렴)

## 재사용 판정 근거 (프롬프트 5항)

- 생성 파일명이 `{src}_{plan}_c{cid}_{idx}_s{seed}.jpg`로 (모델·프롬프트 순서·시드 공식)에
  결정적이며, confirmatory.yaml의 diffusion/verification 섹션은 full.yaml과 문자 그대로
  동일하다. resume 경로는 기존 파일을 픽셀 검증(verify_pair) 후에만 채택한다.
- 신규 배분 대비 부족분: uniform +11, selective +4, weakness +7 (기존 accepted 1000/1000/1000
  대비). weakness_uniform 1000장은 전량 신규.

## 예상 비용 (hard cap USD 45)

학습 15 runs ≈ 24.9 GPU-h + 생성 ≈ 3h + 셋업/평가/통계 ≈ 3h ≈ 31h × $0.854 ≈ $26.5,
디스크/GCS ≈ $1.1 → **최대 ~$28**. 오케스트레이터가 running 시간을 적산해 $40 경고,
$43에서 정지한다(캡 $45 이전).

## 산출물

- `gs://military-od-confirmatory/outputs_confirmatory/` (+ 로컬 다운로드본): runs(weights,
  training_meta, config), metrics(val/test per-class), analysis(plan 4종, freeze JSON,
  confirmatory_{seed_macro,summary,contrasts}.csv, statistical_tests.*), CHECKSUMS.sha256
- `synthetic_confirmatory/`: pool 4종 + generation log
- 실험 종료 후 `RESULTS_CONFIRMATORY.md` 작성 (재현 명령, 실제 GPU-hour, 비용, 실패/재시도)

## 완료 조건 검증

`scripts/confirmatory_check.py`가 VM에서 기계 검증: plan invariant(1000장/13클래스/77·76
quota/교집합 0), 생성 로그 accepted==plan, 전 variant×seed test AP 존재, freeze/통계 산출물
존재, contrast의 seed-matched n=3. 통과 시에만 ALL_DONE 마커.
