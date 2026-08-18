# 확인 실험 결과 (2026-08-06 완료)

IJASS 투고용 확인(confirmatory) 실험. 본 실험(RESULTS.md)의 한계였던 **단일 seed 증강
arm**을 해소하기 위해 2×2 설계 전체를 seeds 42/43/44로 재학습했다. 모든 수치는
`outputs_confirmatory/`(GCS `gs://military-od-confirmatory/` 원본, checksum 175건 검증
완료)에서 계산했다. 주 지표는 test split mAP50-95.

## 1. 설계

- **2×2**: {빈도 tail 13, 측정 weak 13} class set × {uniform, weighted} 배분.
  네 arm 모두 예산 1000장, K=13, 동일 generator/prompt/QC.
  | set \ weighting | uniform | weighted |
  |---|---|---|
  | tail | `aug_uniform_inpaint` | `aug_selective_inpaint` |
  | weak | `aug_weakuniform_inpaint` (신규) | `aug_weakness_inpaint` |
- weak 집합은 본 실험의 basic_aug 3-seed val AP로 **사전 고정**
  (`[9,14,16,17,18,19,20,21,22,24,27,30,34]`, tail과 교집합 0). 13/14위 margin이
  0.0008이라 재측정 의존을 배제했다 (`selective_generation.weakness_class_ids`).
- allocator를 결정적 capped largest-remainder로 수정(구현 버그: 구 코드는 uniform
  1000/13에서 잔여 12장을 첫 클래스에 몰아 88/76×12 생성). uniform quota는 77×12+76.
- 분석 계획(주 contrast 5종)은 test metric 생성 전 freeze:
  `analysis/confirmatory_plan_freeze.json` (sha256 72ef9f31e08d…, 2026-08-04T15:00Z).
- baseline (real_only/basic_aug ×3 seeds)은 본 실험 run 재사용(데이터·설정 동일,
  구버전 run이라 fingerprint 필드 부재 → 내용 동일성은 base 데이터셋 재현성으로 보장).
  aug_rfs는 경로 변경으로 fingerprint 불일치 → seed 42 포함 3 seed 전량 신규 학습.
- 합성 pool 재사용: 파일명·시드가 (source, prompt 순서, seed 공식)에 결정적이고
  diffusion/verification 설정이 full.yaml과 동일 + resume 시 픽셀 재검증. 실제 신규
  생성은 uniform 1, selective 4, weakness 10장뿐. weakness_uniform은 1000장 전량 신규.

## 2. 주 결과 — arm × scope (test mAP50-95, 3 seeds, mean ± SD)

| arm | all (43) | tail (13) | weak (13) |
|---|---|---|---|
| basic_aug | 0.5752 ± 0.0070 | 0.6198 ± 0.0090 | 0.4561 ± 0.0304 |
| aug_uniform_inpaint | 0.6058 ± 0.0103 | 0.6797 ± 0.0038 | 0.4867 ± 0.0255 |
| aug_selective_inpaint | 0.5983 ± 0.0030 | 0.6818 ± 0.0154 | 0.4628 ± 0.0127 |
| aug_weakuniform_inpaint | 0.5912 ± 0.0172 | 0.6342 ± 0.0222 | 0.5015 ± 0.0161 |
| aug_weakness_inpaint | 0.5955 ± 0.0057 | 0.6494 ± 0.0059 | 0.4897 ± 0.0118 |
| **aug_rfs** | **0.6607 ± 0.0056** | **0.7182 ± 0.0236** | **0.5725 ± 0.0027** |

## 3. 사전 정의 contrast (seed-blocked paired t, n=3, Holm 보정)

| contrast | 추정치 | 95% CI | p | p(Holm) |
|---|---|---|---|---|
| **interaction (set × scope)** | **+0.0598** | [+0.0449, +0.0747] | 0.0033 | **0.0167** ✓ |
| tail arms − baseline @ tail | +0.0609 | [+0.0260, +0.0959] | 0.0173 | 0.0693 |
| weak arms − baseline @ weak | +0.0395 | [−0.0206, +0.0997] | 0.1055 | 0.3166 |
| weighted − uniform @ tail | +0.0021 | [−0.0311, +0.0354] | 0.8084 | 0.8084 |
| weighted − uniform @ weak | −0.0118 | [−0.0362, +0.0126] | 0.1722 | 0.3445 |

**핵심 확인**: interaction contrast — (weak arms − tail arms)의 격차가 weak scope와
tail scope 사이에서 +0.0598 벌어진다 — 가 Holm 보정 후에도 유의(p=0.017).
**배분 신호(class set)가 이득이 떨어지는 위치를 결정한다는 본 실험의 주 주장이
3 seed에서 확인되었다.** 예산·K·생성기·QC가 고정이므로 교란 변수는 없다.

## 4. 보조: 클래스 단위 Wilcoxon (seed 평균, vs basic_aug)

| arm | tail scope | weak scope |
|---|---|---|
| aug_uniform_inpaint | **+0.0599 (p=0.0012)** | **+0.0306 (p=0.0024)** |
| aug_selective_inpaint | **+0.0620 (p=0.0002)** | +0.0067 (p=0.216) |
| aug_weakuniform_inpaint | +0.0144 (p=0.127) | **+0.0455 (p=0.0017)** |
| aug_weakness_inpaint | **+0.0296 (p=0.0081)** | **+0.0336 (p=0.0081)** |
| aug_rfs | **+0.0984 (p=0.0002)** | **+0.1165 (p=0.0002)** |

- 가장 깨끗한 해리 쌍은 **selective(tail만 유의) ↔ weakuniform(weak만 유의)**.
- 같은 집합 내 재가중(weighted vs uniform)은 tail/weak 어느 쪽에서도 비유의 —
  본 실험의 "재가중 무효" 발견이 양쪽 집합에서 재확인됨. weak scope에서는 가중이
  오히려 수치상 손해(−0.0118).
- **RFS의 전 scope 우세가 3 seed로 재현**(all +0.0855, tail +0.0984, weak +0.1165;
  클래스 단위 p=1e-11~2e-4). 본 실험 한계 ②(단일 seed 취약성) 해소.

## 5. 해석 요약 (논문 반영 지침)

1. **주 주장(이중 해리→interaction) 생존**: 사전 freeze된 interaction contrast가
   Holm 후 유의. 논문의 핵심 진술을 "각 arm은 자기 집합에서만 유의"(패턴 기술)에서
   "**set × scope 상호작용 +0.060, 95% CI [0.045, 0.075], Holm p=0.017**"(사전 정의
   검정)로 승격할 것.
2. weak-set arm의 baseline 대비 절대 이득은 seed-blocked 검정에서는 비유의
   (+0.0395, CI가 0을 포함) — weak scope의 baseline seed 분산(SD 0.030)이 크기
   때문. 클래스 단위 Wilcoxon으로는 유의하나, 보수적으로 "위치 결정은 확인,
   절대 크기는 CI 폭 안"으로 기술할 것.
3. 재가중 축은 null — "무엇을 늘릴지(집합)가 어떻게 나눌지(가중)보다 중요"가
   2×2 완결 형태로 성립.
4. RFS 우세는 3 seed로 견고 — 합성 증강의 실용적 한계 논의 유지.

## 6. 재현

```bash
# 로컬 검증 (설정·plan·통계 로직)
python -m pytest tests/            # 82 passed
# 전체 파이프라인 (GCP VM, us-central1 L4)
python src/run_pipeline.py --config configs/confirmatory.yaml --download --stop-after-inpaint
python src/run_pipeline.py --config configs/confirmatory.yaml --skip-inpaint
python scripts/confirmatory_check.py --config configs/confirmatory.yaml
# 무인 실행 인프라
bash scripts/run_confirmatory_gcp.sh {prepare|create|watch|status|download}
```

## 7. 실측 자원·비용

- 신규 학습 15 run: inpaint 12 × ~1.44h + RFS 3 × 2.54h ≈ **25.0 GPU-h**
  (중단-재개 세그먼트 포함). 생성(weakness_uniform 1000장 + 부족분 15장) ≈ 3 GPU-h.
- VM 가동 합계 ≈ 29h (phase 11.33h + 11.33h + 4.94h + 초기 디버깅/재시도 ≈ 1.4h).
- **비용 ≈ $26~27** (g2-standard-8+L4 $0.854/h + 디스크 2×100GB + GCS ~5GB).
  hard cap $45의 60% 수준, 사전 견적 $28과 일치.

## 8. 실패·재시도 기록 (전부 해결)

| 사건 | 원인 | 조치 |
|---|---|---|
| 부팅 즉사 ×2 (8/4) | startup-script 최소 PATH에 gcloud 부재 | PATH 보강 |
| 마커·코드 접근 403 | 새 버킷에 compute SA IAM 없음 | storage.admin 부여 |
| pip 경로 실패 | 이미지에 /opt/conda 없음 (시스템 python) | 인터프리터 동적 탐지 |
| **diffusion 중 guest 종료 ×4** | tqdm \r 진행바가 metadata runner의 줄 스캐너(64KB) 초과 → `bufio.Scanner: token too long` → runner 사망 → 후속 shutdown | 태스크 출력 파일 리다이렉트 + `TQDM_DISABLE=1` |
| zone-a 재시작 불가 | L4 STOCKOUT (정지 VM은 zone 고정) | zone-b에 신규 VM(`military-od-conf-b`), zone-a는 정지 보존 |
| 재시작 5h 지연 (8/5) | 로컬 watch가 Mac 수면 중 정지 | GCE instance schedule(매시 start)로 Mac 독립화 |
| 완료 검증 실패 ×5 (8/6) | `tabulate` 부재로 통계 md 렌더링만 실패 | requirements 추가 + 매 부팅 보장 설치 |

## 9. 산출물

- durable 원본: `gs://military-od-confirmatory/outputs_confirmatory/` —
  runs(15 신규 + baseline 9 재사용), metrics(per_class_ap: 전 variant × 3 seed ×
  val/test), analysis(plan 4종, freeze JSON, confirmatory_{seed_macro,summary,
  contrasts}.csv, confirmatory_stats.md, statistical_tests.*), CHECKSUMS.sha256
- 로컬 사본: `outputs_confirmatory/` (2026-08-06 다운로드, **checksum 175건 전부 일치**)
- 합성물: `gs://military-od-confirmatory/synthetic_confirmatory/` (4 pool × 1000장 + 로그)
- 완료 검증: `scripts/confirmatory_check.py` 전 항목 통과 (ALL_DONE 2026-08-06 13:02 KST)

## 10. 인프라 정리 상태

- VM 2대 모두 TERMINATED (`military-od-conf`@a, `military-od-conf-b`@b) — 디스크만
  과금(각 ~$10/월). 결과 검증이 끝났으므로 **둘 다 삭제 가능** (zone-a는 유물 없음,
  zone-b 산출물은 GCS+로컬 이중 확보됨). instance schedule은 VM에서 분리됨 —
  resource policy `military-od-conf-sched`(us-central1)도 삭제 대상.
- `gs://military-od-confirmatory`는 결과 원본이므로 보존.
