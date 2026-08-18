# Tier 2 확장 실험 — 실행 계획 (2026-08-07 승인)

타겟 논문 Li et al., "A Simple Background Augmentation Method for Object
Detection with Diffusion Model" (ECCV 2024, arXiv:2408.00350) 대비 논문 완결성
강화: **데이터셋 축**(MAR20, 원격탐사 위성 시점)과 **아키텍처 축**(RT-DETR-L,
transformer NMS-free)을 추가한다. 사용자 승인 사항: Tier 2 범위, spot 인스턴스,
MAR20 budget 500장.

## 1. Cell 구성

| cell | 데이터셋 | 검출기 | arm | config |
|---|---|---|---|---|
| C1 (완료) | MAD 43cls | YOLOv8n | 2×2 + RFS ×3 seeds | `configs/confirmatory.yaml` |
| C2 | MAD | RT-DETR-L | basic_aug + {uniform, selective} + RFS ×3 | `configs/mad_rtdetr.yaml` |
| C3 | MAR20 20cls | YOLOv8n | real_only/basic_aug + 2×2 + RFS ×3 | `configs/mar20_yolo.yaml` |
| C4 | MAR20 | RT-DETR-L | basic_aug + {uniform, selective} + RFS ×3 | `configs/mar20_rtdetr.yaml` |

신규 학습 run: C2 12 + C3 18 + C4 12 = **42 runs** (+C3 real_only 3 = 45).

## 2. 설계 결정 (요약)

1. **MAR20 프로토콜**: 공식 test 2,511장 보존(최종 보고 전용), 공식 train
   1,331장을 stratified 80/20 → train/val (split_seed 42). HBB 주석, 클래스
   canonical 이름은 A1..A20 (기종 매핑은 metadata 참고용).
2. **MAR20 tail/weak**: bottom_percent 0.3 → tail 6 / weak 6 클래스. weak는
   basic_aug val AP로 측정 (사전 pin 없음 — margin이 좁으면 학습 후 pin하고 기록).
3. **Budget**: MAR20 B=500, min 5, max 100 (uniform 배분 84×2+83×4 —
   클래스당 quota가 MAD의 77과 유사). MAD cell은 기존 B=1000 유지.
4. **검출기별 selective plan**: weakness score는 해당 검출기 자신의 basic_aug
   val AP에서 계산. uniform plan은 빈도 기반이라 검출기 간 동일.
5. **Pool 재사용**: 생성물이 (source, plan, class, idx, seed42)에 결정적 +
   채택 전 픽셀 재검증이므로 — C2는 confirmatory pool(uniform 전량 / selective
   부분)을 시딩 재사용, C3/C4는 `synthetic_mar20` 공유.
6. **RT-DETR-L**: Ultralytics `rtdetr-l.pt` (COCO pretrained), epochs 50 /
   patience 15 / imgsz 640 / batch 8 명시(auto-batch 금지), seeds 42/43/44.
   학습·검증 모두 RTDETR 클래스로 로드 (`src/utils/detector.py` — YOLO 클래스로
   열면 NMS 전제 후처리가 잘못 적용됨).
7. **위성뷰 프롬프트**: MAR20 전용 5종(top-down apron/runway/grass/stands/desert),
   negative에 aircraft·sky·horizon·oblique 차단. **본 생성 전 50장 파일럿으로 QC
   통과율 확인** (verification.max_failure_rate 0.5 초과 시 자동 중단됨).
8. **real_only**: C3에서만 유지(저렴, 참고 하한선). C2/C4는 생략
   (`experiments.baseline_variants`).
9. **통계**: cell별 primary contrast 사전 freeze + Holm은 cell 내 family 한정.
   C3는 C1과 동일한 5-contrast(2×2 interaction 포함) 재현 검정, C2/C4는 축소
   3-contrast (tail arms vs baseline @tail, uniform vs baseline @all, weighted
   vs uniform @tail). cell 간 비교는 descriptive로만 보고.

## 3. GCP 실행 (spot)

```bash
# 0) 로컬 검증
python -m pytest tests/

# 1) MAR20 수동 다운로드 후 (JPEGImages/Annotations/ImageSets 구조):
bash scripts/run_tier2_gcp.sh prepare /path/to/MAR20

# 2) VM 생성 + 무인 감시 (spot 선점·12h 세션 자동 재시작)
bash scripts/run_tier2_gcp.sh create
bash scripts/run_tier2_gcp.sh watch

# 3) 완료 후
bash scripts/run_tier2_gcp.sh download
python scripts/tier2_check.py --config configs/mad_rtdetr.yaml \
  --config configs/mar20_yolo.yaml --config configs/mar20_rtdetr.yaml
```

- VM: `military-od-tier2`, g2-standard-8 + L4, **SPOT** (선점 시 STOP → watch 재시작),
  us-central1-{a,b,c}, 디스크 150GB, 12h max-run + 부팅당 11h20m self-stop.
- 버킷: `gs://military-od-tier2` (코드/마커/산출물), confirmatory 버킷은 pool
  시딩용 read-only.
- Cell 순서: C2 → C3 → C4. cell마다 GEN(분석+baseline+plan+생성) → 학습/통계 →
  `tier2_check.py` 기계 검증 → CELL_DONE 마커.
- **파일럿 체크포인트**: 첫 부팅에서 RT-DETR basic_aug seed 42를 단독 학습해
  실측 시간을 `PILOT_rtdetr_mad` 마커로 보고 (해당 run은 fingerprint 일치로
  파이프라인이 재사용). 실측이 추정(±30%)을 크게 벗어나면 watch를 멈추고 재견적.

### 무인 재시작 (로컬 PC 독립)

로컬 `watch`는 실행 중인 PC가 절전되면 멈춘다(confirmatory 때 5시간 지연 사례).
이를 피하려면 GCE instance schedule로 매시 VM을 start한다.

```bash
gcloud compute resource-policies create instance-schedule military-od-tier2-sched \
  --region=us-central1 --vm-start-schedule="0 * * * *" --timezone=Asia/Seoul \
  --end-date=<+14d RFC3339>
gcloud compute instances add-resource-policies military-od-tier2 \
  --zone=us-central1-a --resource-policies=military-od-tier2-sched
```

스케줄이 붙으면 watch의 비용 상한 감시가 무력화되므로, `tier2_vm_task.sh`가
부팅 직후 스스로 다음 세 조건을 검사해 정지한다 (§0 재시작 금지 조건):

1. `ALL_DONE` 마커 → 즉시 shutdown
2. `FAILED_*` 마커 → 사람이 원인 확인·마커 삭제 전까지 재시작 금지
   (같은 실패를 11시간씩 반복 과금하는 루프 차단)
3. 누적 가동 시간 > `BUDGET_HOURS`(150h) → `BUDGET_EXCEEDED` 마커 + shutdown.
   phase 마커의 elapsed 합으로 계산하며, 선점으로 finish()를 못 거친 부팅은
   빠지므로 과소 계산된다(backstop 용도).

## 4. 예상 자원·비용 (hard cap USD 60)

| 항목 | 추정 |
|---|---|
| C2 학습 (9×~5.5h + RFS 3×~10h) | ~80 L4-h |
| C2 생성 (selective 부족분) | ~1 L4-h |
| C3 학습 (15×~0.4h + RFS 3×~0.7h) | ~8 L4-h |
| C3 생성 (pool 4종 × 500장) | ~6 L4-h |
| C4 학습 (9×~1.5h + RFS 3×~2.8h) | ~22 L4-h |
| C4 생성 (selective 부족분) | ~1 L4-h |
| 셋업/평가/통계/동기화 | ~7 h |
| **합계** | **~125 h** |

- spot ~$0.30-0.36/h 기준 **~$40-47** + 디스크/GCS ~$4 → **~$45-50**.
  (RT-DETR 시간 ±30% 불확실 → 파일럿으로 보정. on-demand 폴백 시 ~$110.)
- watch가 RUNNING 시간을 적산해 $50 경고, $57 정지 (cap $60 이전).
- GPU quota 1 → 순차 실행. 12h 세션 약 11회, 달력 6~9일 (선점 빈도에 따라 변동).

## 5. 산출물

- `gs://military-od-tier2/outputs_{mad_rtdetr,mar20_yolo,mar20_rtdetr}/` + 로컬
  다운로드본: runs(weights/meta/config), metrics(per_class_ap: variant×seed×
  val/test), analysis(plan, freeze JSON, confirmatory_*.csv, statistical_tests.*),
  CHECKSUMS.sha256
- `synthetic_mad_rtdetr/`, `synthetic_mar20/`: pool + 생성 로그
- 실험 종료 후 `RESULTS_TIER2.md` 작성 (재현 명령, 실제 GPU-hour, 비용, 실패/재시도)

## 6. 리스크와 대응

| 리스크 | 대응 |
|---|---|
| SD 인페인팅 위성뷰 도메인 갭 | C3 GEN 단계 초반 QC 통과율 모니터; max_failure_rate 0.5 초과 시 자동 중단 → 프롬프트/strength 조정 후 재개 |
| RT-DETR-L 시간 추정 ±30% | 첫 부팅 파일럿 마커로 실측 보고 → 재견적 |
| spot 선점 반복 | 검증된 resume 인프라(run/생성물 멱등) + MAX_BOOTS 40 |
| L4 spot 재고 부족 | 3개 zone 폴백; 장기 품절 시 on-demand 전환 판단(비용 재계산 필요) |
| MAR20 소형 객체 @640 | 절대 AP 낮아도 arm 간 상대 비교 유효; imgsz 800 ablation은 P1 |

## 운영 변경 기록

- 2026-08-12 20:15 (사용자 승인): 저녁 시간대 spot 선점·재고 소진 반복으로 `military-od-tier2`를 **SPOT → STANDARD(on-demand) 전환** (`set-scheduling --provisioning-model=STANDARD`). 잔여 ~20 GPU-h 기준 추가 비용 ~$11-14 (spot 대비). 이후 정지는 phase 경계(11h20m)에서만 발생.
