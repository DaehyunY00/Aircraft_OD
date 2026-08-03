# 실험 결과 정리 (2026-08-01)

논문 작성용 근거 문서. 모든 수치는 `outputs_full/`의 산출물에서 직접 계산했고,
재현 경로를 각 절 끝에 적어두었다. 주 지표는 **test split의 mAP50-95**(통계 검정이
사용하는 컬럼)이며, 읽기 편한 AP50은 보조로만 병기한다.

## 1. 설정

- 데이터셋: Military Aircraft Detection (YOLO format), 43 클래스
- 이미지 11,788장 (train 9,430 / val 1,179 / test 1,179), bbox 18,832개
- **imbalance ratio 10.49** — LVIS급 long-tail이 아니다. 최소 클래스도 86 instances.
- 검출기 YOLOv8n, 640px, 50 epoch, patience 15
- 계획 split = val, 최종 평가 split = test
- seed: baseline(real_only/basic_aug)은 42/43/44, 증강 arm은 **42 단일** (§7 참조)

## 2. 동기: 이 데이터셋에서 빈도는 난이도의 대리지표가 아니다

basic_aug baseline의 클래스별 AP와 instance_count 상관:

| split | 지표 | Pearson r | Spearman ρ |
|---|---|---|---|
| val | mAP50-95 | −0.377 (p=0.013) | −0.460 (p=0.002) |
| test | mAP50-95 | **−0.375 (p=0.013)** | **−0.408 (p=0.007)** |

**희소할수록 오히려 잘 맞힌다.** 그룹 평균으로도 같은 방향이다(test mAP50-95):
tail 0.6205 > medium/head를 포함한 전체 0.5755.

가장 약한 클래스는 Rafale 0.308, F14 0.365, JAS39 0.489, F18 0.489(AP50 기준)로
대부분 medium·head에 속한 전투기들이다. 반면 tail 최소 빈도인 YF23(97 instances)은
0.928로 전체 최고다. 형상이 독특한 기체는 희소해도 쉽고, 실루엣이 유사한 전투기
군집은 데이터가 많아도 어렵다.

또한 confusion matrix에서 가장 짙은 비대각 성분은 **background 행(미검출)** 이며,
recall(0.52~0.59)이 precision(0.62~0.71)보다 일관되게 낮다. 즉 지배적 오류는
클래스 간 혼동이 아니라 검출 실패이고, 이는 배경 다양성을 늘리는 본 기법의
작동 기제와 부합한다.

> 재현: `per_class_ap.csv` × `class_groups.csv`, `scipy.stats.pearsonr`

## 3. 배분 arm 설계

고정 예산 **1000장**, 고정 클래스 수 **13개**. 배분 신호만 변수로 둔다.

| arm | 클래스 선택 | 클래스당 배분 |
|---|---|---|
| `aug_uniform_inpaint` | 빈도 기반 tail 13개 | 균등 |
| `aug_selective_inpaint` | 빈도 기반 tail 13개 | rarity+weakness 가중 (α=0.6) |
| `aug_weakness_inpaint` | **전체 43개를 baseline AP로 랭킹, 하위 13개** | weakness 가중 |

**두 클래스 집합의 교집합은 0개다.**

- tail: `[2, 7, 13, 26, 28, 29, 31, 32, 35, 36, 37, 41, 42]` (AG600, Be200, E7, Mig31, P3, RQ4, SR71, Su34, Tu160, Tu95, U2, XB70, YF23)
- weak: `[9, 14, 16, 17, 18, 19, 20, 21, 22, 24, 27, 30, 34]` (C17, EF2000, F14, F15, F16, F18, F22, F35, F4, JAS39, Mirage2000, Rafale, Tornado) — head 7 + medium 6

생성 품질 관문 통과율은 세 arm이 고르다: uniform 81.8%, selective 80.8%,
weakness 79.4% (실패율 가드 50% 대비 여유). 성능 차이를 생성 품질 차이로
설명할 수 없다는 뜻이다.

> 재현: `augmentation_plan_*.csv`, `synthetic/generation_log_*.csv`

## 4. 주 결과 (test mAP50-95)

| variant | 합성 | all (43) | tail (13) | weak (13) |
|---|---|---|---|---|
| real_only | 0 | 0.2744 | 0.3258 | 0.1597 |
| basic_aug (baseline) | 0 | 0.5755 | 0.6205 | 0.4561 |
| aug_oversample | 1000 | 0.6008 | 0.6776 | 0.4667 |
| aug_copy_paste | 1000 | 0.5902 | 0.6588 | 0.4513 |
| aug_uniform_inpaint | 1000 | 0.5969 | 0.6737 | 0.4756 |
| aug_selective_inpaint | 1000 | 0.6073 | 0.6767 | 0.4742 |
| aug_weakness_inpaint | 1000 | 0.5938 | 0.6370 | **0.4932** |
| **aug_rfs** | **0** | **0.6576** | **0.7116** | **0.5694** |

basic_aug 대비 이득:

| variant | all | tail | weak |
|---|---|---|---|
| aug_oversample | +0.0253 | +0.0571 | +0.0107 |
| aug_copy_paste | +0.0147 | +0.0383 | −0.0047 |
| aug_uniform_inpaint | +0.0214 | +0.0532 | +0.0195 |
| aug_selective_inpaint | +0.0318 | +0.0562 | +0.0181 |
| aug_weakness_inpaint | +0.0183 | +0.0165 | **+0.0372** |
| aug_rfs | +0.0821 | +0.0911 | +0.1133 |

## 5. 핵심 발견: 이중 해리

Wilcoxon signed-rank, 클래스 단위 대응(seed 평균), vs basic_aug:

| arm | tail scope (n=13) | weak scope (n=13) |
|---|---|---|
| `aug_selective_inpaint` | **+0.0562, p=0.0017** ✓ | +0.0181, p=0.273 ✗ |
| `aug_weakness_inpaint` | +0.0165, p=0.273 ✗ | **+0.0372, p=0.0398** ✓ |

대각선이 교차한다. **각 arm은 자기가 증강한 클래스에서만 유의한 개선을 낸다.**
예산과 클래스 수가 동일하고 대상 집합이 완전히 분리돼 있으므로, 배분 신호가
이득이 떨어지는 위치를 결정한다는 것이 통제된 형태로 확인된다.

이것이 논문의 주 주장이 되어야 한다. "weakness가 selective보다 우수하다"보다
강하고, 실제로 두 arm의 직접 비교는 유의하지 않다(all scope에서 −0.0135, p=0.154).

## 6. 부수 결과: 같은 집합 안의 재가중은 효과가 없다

`aug_selective_inpaint` vs `aug_uniform_inpaint`, tail scope: **+0.0030, p=0.685**.

빈도 기반 tail 안에서 rarity+weakness로 가중치를 준 것이 균등 배분 대비 아무
이득을 주지 못했다. §5와 나란히 놓으면 메시지가 선명해진다 —
**집합 내 재가중은 무효, 집합 자체의 교체는 유효.**

## 7. 한계

**7.1 단일 seed.** 증강 arm 6종은 seed 42만 학습했다. 클래스 단위 Wilcoxon
(n=13/43)은 클래스 난이도를 통제하지만 seed 분산은 반영하지 못한다.
3-seed baseline에서 측정한 노이즈 하한과 비교하면:

| scope | basic_aug seed 표준편차 | 해당 arm의 이득 | 비 |
|---|---|---|---|
| tail | 0.0083 | selective +0.0562 | 6.8× |
| weak | **0.0304** | weakness **+0.0372** | **1.2×** |

**이중 해리의 weak 쪽 팔은 seed 노이즈 대비 여유가 크지 않다.** basic_aug의
weak scope 95% CI [0.3805, 0.5317] 안에 weakness arm의 0.4932가 들어간다.
seed 43/44 추가(3 arm × 2 seed = 6 run, L4 약 9시간)로 해소 가능하며, 향후
과제로 명시한다.

**7.2 RFS baseline이 모든 scope에서 우세하다.** 합성 이미지 0장으로 all +0.0821,
tail +0.0911, weak +0.1133을 얻어 확산 기반 arm들(+0.015~0.032)을 크게 앞선다.
계산 비용은 두 자릿수 배 차이다. 다만 RFS는 *기존 이미지를 더 자주 보여주는*
방식이라 "고정된 생성 예산을 어디에 쓸 것인가"라는 본 연구의 축과 직교한다.
생성 증강의 우월성을 주장하지 않고, 배분 신호의 효과를 분리해 보이는 것으로
논지를 한정해야 한다.

**7.3 데이터셋이 진성 long-tail이 아니다.** imbalance ratio 10.49, 최소 86
instances. LVIS(1000×)급 분포에서는 §2의 음의 상관이 재현되지 않을 수 있다.
결론의 일반화 범위를 이 조건에 한정해야 한다.

**7.4 생성 품질 지표 — 측정 완료 (2026-08-02).** arm당 200장 표본, GCP L4:

| arm | CLIPScore | LPIPS(원본↔생성) | FID overall | FID per-class 중앙값 |
|---|---|---|---|---|
| uniform | 19.6 ± 3.3 | 0.262 | 89.7 | 83.2 |
| selective | 19.1 ± 3.8 | 0.244 | 87.7 | 91.2 |
| weakness | 20.2 ± 3.4 | 0.272 | 100.4 | 103.7 |

CLIP/LPIPS는 arm 간 동등(생성기 동작 동일 확인), uniform·selective FID 동등(통제
확인), weakness FID만 ~12 높음(head/medium 원본의 다중 기체·클러터 장면 때문).
**FID가 가장 나쁜 arm이 자기 scope에서 유의한 이득을 냈으므로 충실도 차이는
이중 해리를 설명할 수 없다** — 오히려 주장을 보강. 산출 방법 주의: VM의
torchmetrics/transformers 비호환(신버전 transformers의 `get_image_features`가
ModelOutput 반환) 때문에 CLIPScore는 transformers CLIP forward의
`logits_per_image / logit_scale` 경로로 직접 계산(정의는 torchmetrics와 동일:
100×cosine, ViT-L/14). FID/LPIPS는 CLI 정상 산출. 원자료
`outputs_full/synthetic/{quality_report.csv, fid_by_class_*.csv}`.

## 7.5 배경 환각 감사 (2026-08-03, 측정 완료)

검증 관문이 못 잡는 실패 유형. baseline 검출기를 원본/생성본에 각각 돌려
보호 박스 밖 검출의 **증가분**을 측정(arm당 150장, 총 450장).

| arm | 이미지당 여분 검출 (원본→생성) | ≥1개 추가된 이미지 | 추가된 미라벨 객체 |
|---|---|---|---|
| uniform | 0.000 → 0.307 | 12.7% | 46 |
| selective | 0.000 → 0.207 | 4.7% | 31 |
| weakness | 0.007 → 0.353 | 13.3% | 52 |
| **전체** | **0.002 → 0.289** | **10.2%** | **130** |

- **생성 이미지 10장 중 1장에 라벨 없는 항공기가 들어 있다.** 분포는 heavy-tail:
  46장 중 27장은 1개, 그러나 4장은 11~17개(하늘이 편대로 채워진 사례).
- 재도색 면적과 단조 관계: LPIPS 0.244/0.262/0.272 ↔ 환각률 4.7/12.7/13.3%.
- **시각 검증 완료**(`paper/figures/fig7_hallucination_examples.png`): 빨간 박스가
  실제 항공기 형상 위에 정확히 얹힘 → 검출기 노이즈 아님. 원본 쪽 0.000은
  검출기가 그 원본으로 학습된 암기 효과가 섞여 있으므로 하한으로 읽어야 하며,
  시각 검증이 이 우려를 해소한다.
- 학습 영향: 미라벨 객체 = "이 항공기는 배경"이라는 거짓 감독 → §2에서 확인한
  지배적 오류(미검출)를 정확히 악화시킨다. 세 arm에 동일하게 걸리므로 이중
  해리는 만들 수 없지만, **inpainting arm 전체의 절대 이득 상한**과 RFS와의
  격차를 설명하는 유력 후보.
- 후속: object-level gate(생성물에 검출기를 돌려 보호 밖 확신 검출이 있으면 기각)로
  재실행하면 "확산 vs 재샘플링 격차 중 얼마가 라벨 노이즈인가"를 직접 검정 가능.
  감사 도구가 곧 그 gate다(`src/eval/audit_hallucination.py`).

원자료: `outputs_full/analysis/hallucination_audit{,_summary}.csv`

## 7.6 Stage 0: 역전은 어디까지 성립하는가 (2026-08-03, 측정 완료)

§7.3에서 "극단적 long-tail에서는 빈도가 예측력을 되찾을 수 있다"고 조건부로
적어둔 것을 검정했다. 같은 사진 풀의 **103클래스판**(우리 43클래스가 99.4%
부분집합, 새 이미지 11,904장·새 클래스 60개)에서 동일 프로토콜로 baseline을
학습했다(17,687장 train, yolov8n/640/50ep/basic_aug, 117.6분, L4).

| 대상 | 클래스 | 불균형 | Pearson r | Spearman ρ |
|---|---|---|---|---|
| 주 실험 43클래스 | 43 | 10.5× | **−0.375 (p=0.013)** | **−0.408 (p=0.007)** |
| 103클래스 전체 | 103 | **140.3×** | **+0.044 (p=0.661)** | +0.038 (p=0.708) |
| 103클래스 중 **동일 43클래스만** | 43 | — | **−0.389 (p=0.010)** | **−0.506 (p=0.001)** |

**결론 세 가지.**

1. **역전은 재현된다 — 단, 그 클래스 부분집합 안에서.** 완전히 다른 모델(103클래스로
   학습, 라벨 granularity도 다름)로 재측정해도 같은 43클래스에서 r=−0.389가 나온다.
   주 실험의 −0.375가 학습 우연이나 설정 artifact가 아님을 독립 확인한 셈이다.
2. **103클래스 전체에서는 상관이 사라진다(r≈0).** 노이즈 탓이 아니다 — 인스턴스
   하한을 50/100/200으로 올려 재계산해도 +0.050 / −0.058 / −0.108로 모두 비유의.
   즉 60개 클래스가 추가되면서 관계가 희석된다.
3. **어느 조건에서도 빈도가 난이도를 양(+)으로 예측하지 않는다.** 통념(희소=어려움)은
   10.5배에서는 반대로, 140배에서는 무관계로 나타난다. 이게 역전 단독보다 강한
   일반화 진술이다.

**설계 이전 여부: 불가.** 빈도 하위 K와 AP 하위 K의 교집합이 **12/31 = 38.7%**로,
주 실험의 0%와 달리 상당히 겹친다. 통제 비교의 핵심인 집합 분리가 성립하지 않으므로
103클래스판에서 Stage 1(본실험)을 돌려도 대비가 흐려진다. **진행하지 않는다.**

논문 반영: §VII의 조건부 서술을 이 측정으로 대체하고, ②를 일반화 진술로 승격.
103클래스판은 독립 데이터셋이 아니므로(99.4% 부분집합) 외부 검증으로 제시하지 말 것.

원자료: `gs://military-od-stage0/`, `outputs_mad103/analysis/stage0_{per_class,summary}.csv`

## 8. 재현 정보

- 코드: <https://github.com/DaehyunY00/Aircraft_OD> (`53e26c4`)
- 산출물: `gs://military-od-d522190f/outputs_full/`
- 학습 환경: GCP `g2-standard-8` + NVIDIA L4 1기, 6 run 총 9시간 46분
- 합성 생성: Colab T4/A100, SD-inpainting(runwayml), 20 steps, 3000장
