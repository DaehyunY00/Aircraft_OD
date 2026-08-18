# Tier 2 확장 실험 결과 (2026-08-16 완료, ALL_DONE 20:18 KST)

C1(확인 실험, RESULTS_CONFIRMATORY.md)에 **데이터셋 축**(MAR20, 위성 시점)과
**아키텍처 축**(RT-DETR-L)을 추가한 2×2 셀 설계의 결과. 모든 수치는 test split
mAP50-95, 전 run seeds 42/43/44 완비 (총 신규 45 runs).
원본: `gs://military-od-tier2/outputs_{mad_rtdetr,mar20_yolo,mar20_rtdetr}/`.

## 1. 셀 요약

| cell | 데이터셋 | 검출기 | basic_aug (all) | 상태 |
|---|---|---|---|---|
| C1 | MAD 43cls | YOLOv8n | 0.5752 | 완료 (8/6) |
| C2 | MAD | RT-DETR-L | **0.8686** | 완료 (8/14) |
| C3 | MAR20 20cls | YOLOv8n | 0.5720 | 완료 (8/15) |
| C4 | MAR20 | RT-DETR-L | **0.6970** | 완료 (8/16) |

MAR20 합성 생성: 4 pool × 500장, **검증 실패율 0.6%** (MAD ~19% 대비) — 위성
top-down 배경(아프론/활주로)이 inpainting에 유리. 위성뷰 프롬프트 5종 사용.

## 2. 발견 1 — interaction(배분 신호 → 이득 위치)은 약한 검출기에서 도메인 불변

YOLOv8n 셀의 사전 freeze된 interaction contrast (seed-blocked, n=3):

| cell | interaction | seed별 | 원 p | Holm p |
|---|---|---|---|---|
| C1 (MAD) | +0.0598 | +0.063/+0.053/+0.063 | 0.0033 | **0.0167** ✓ |
| C3 (MAR20) | +0.0523 | +0.048/+0.066/+0.042 | 0.0183 | 0.0910 △ |

- **크기와 방향이 두 도메인에서 일치**(+0.052 vs +0.060, 모든 seed 양수).
  C3의 Holm p=0.091은 경계선 — scope가 6클래스(MAD 13)로 줄어 검정력이 낮아진
  것이 주 원인이며, C3 추정치는 C1의 95% CI [0.045, 0.075] **안에 있다**.
- 논문 진술 권고: C1을 주 검정(confirmatory), C3를 "추정치가 원 CI 내에서
  재현되는 독립 replication"으로 기술. "MAR20에서도 유의"라고 쓰지 말 것
  (Holm 후 경계). 해리 패턴 자체는 재현: selective는 tail에서만(+0.044),
  weakness는 weak에서만(+0.034) 이득.
- 주의 각주: MAR20은 tail∩weak 교집합 1/6 (MAD는 0/13) — 완전 분리가 아님.

## 3. 발견 2 — 강한 검출기에서는 모든 증강이 무효 (headroom 조건성)

COCO 사전학습 RT-DETR-L(32M)은 두 데이터셋 모두에서 YOLOv8n(3.2M) 대비
baseline이 크게 높고(+0.29/+0.13), 그 위에서는 어떤 증강도 이득이 없다:

| cell | RFS Δ(all) | uniform Δ(all) | selective Δ(all) | inpaint평균 Δ(tail), p |
|---|---|---|---|---|
| C2 (MAD×RT-DETR) | **−0.0159** | −0.0044 | −0.0008 | −0.0084 (p=0.36) |
| C4 (MAR20×RT-DETR) | +0.0033 | −0.0069 | −0.0045 (n3) | +0.0035 (n3) |

증강 이득은 검출기의 **성능 여유(headroom)에 조건부**다. 데이터가 부족해서가
아니라 모델이 이미 데이터셋의 천장 근처면 재표집도 합성도 보탤 것이 없다.
(C2에서 RFS가 유일하게 뚜렷한 음수인 것은 중복 노출 증가가 강한 모델에는
과적합 쪽으로 작용할 수 있음을 시사 — 단정은 피하고 관찰로 보고.)

## 4. 발견 3 — RFS 우세는 데이터셋 조건부 (MAD 특이적)

| cell (YOLOv8n) | RFS Δ all/tail/weak | 최고 diffusion arm과 비교 |
|---|---|---|
| C1 (MAD) | **+0.086/+0.098/+0.117** | 전 scope에서 diffusion 압도 |
| C3 (MAR20) | +0.003/+0.016/−0.011 | **무효** — diffusion arm(selective all +0.012, weakness weak +0.034)이 우위 |

C1의 "무비용 RFS 우선" 권고는 MAD의 불균형 구조(43cls, ratio 10.5)에 특이적.
MAR20(20cls, 얕은 tail)에서는 반복 노출로 얻을 정보가 없고 배경 다양화가
실제 기여를 한다. 실용 권고를 "먼저 RFS를 시험하되, 이득이 없으면 표적
합성으로 전환"의 2단 규칙으로 수정할 것.

## 5. 통합 조건 지도 (논문의 새 기여)

| | YOLOv8n (여유 큼) | RT-DETR-L (천장 근처) |
|---|---|---|
| 배분 신호 효과 | MAD 유의 + MAR20 재현 | 소멸 |
| RFS | MAD 강함 / MAR20 무효 | 무효~음수 |
| 실용 시사 | 신호 선택이 본질, 재가중 무의미 | 증강 투자 불필요 |

## 6. 투고 판단에 주는 의미

- 시나리오 "재현 성공 + 조건 발견"에 해당 — **TAES/EAAI 상향 정당**.
  headroom 조건성(발견 2)과 RFS 데이터셋 의존성(발견 3)은 단순 재현을 넘는
  새 결과로, "생성 증강은 언제 값을 하는가" 스토리를 완성한다.
- 정직한 한계: ① C3 interaction은 Holm 후 경계(검정력) ② MAR20 tail/weak
  부분 중첩 ③ RT-DETR 셀은 축소 설계(2×2 아닌 uniform/selective만)라
  interaction 검정 불가 — "소멸"은 baseline 대비 Δ≈0으로 뒷받침.

## 7. 완료 기록 (2026-08-16)

- ALL_DONE 20:18 KST (s44는 phase 중단 후 resume으로 14분 만에 완주, 전 run 3-seed 완비)
- 결과 로컬 다운로드(`outputs_mad_rtdetr/`, `outputs_mar20_yolo/`,
  `outputs_mar20_rtdetr/`) + checksum 검증 완료 — 본문 하단 참조
- 인프라 정리: VM `military-od-tier2`+디스크 150GB+policy `military-od-tier2-sched`
  삭제. 버킷 4개(military-od-{confirmatory,tier2,gate,stage0}) 보존.
- 운영 사건: spot 선점 다발(8/9-12) → 8/12 on-demand 전환, zone-a L4 STOCKOUT
  정체 2회(8/13 32h, 8/16 20h). 최종 소요 8/7~8/16 (9.5일), 비용 ~$35
  (spot 구간 ~$17 + on-demand 구간 ~$17).
