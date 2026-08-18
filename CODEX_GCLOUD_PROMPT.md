# Codex CLI prompt: IJASS confirmatory experiments on Google Cloud

아래 내용을 Codex CLI에 그대로 전달하라. 이 작업은 코드 수정, 검증, Google Cloud VM 생성과
확인 실험 실행을 포함한다. 온디맨드 비용 상한은 **USD 45**이며, 이 범위 안의 VM 생성과
실험 실행을 승인한다. 상한을 넘길 가능성이 생기면 VM을 정지하고 사용자에게 보고하라.

```text
이 저장소의 IJASS 확인 실험을 준비하고 Google Cloud에서 끝까지 실행해줘.

목표
- 잘못된 synthetic budget allocator를 수정한다.
- class set(tail/weak) × within-set weighting(uniform/weighted)의 2×2 실험을 구성한다.
- 네 diffusion arm을 동일한 seeds 42/43/44로 학습한다.
- RFS는 기존 seed 42가 현재 설정과 정확히 일치할 때만 재사용하고 seeds 43/44를 추가한다.
- 결과를 seed-matched 방식으로 분석하고, policy × evaluation-scope interaction, 95% CI,
  Holm 보정 결과를 생성한다.
- 모든 산출물을 기존 outputs와 분리된 durable output에 보존한다.

안전·비용 제약
- Google Cloud project는 project-d522190f-d377-47af-bf2, 우선 region은 us-central1이다.
- 기본 VM은 on-demand g2-standard-8 + NVIDIA L4 1개 + 100 GB pd-balanced이다.
- 현재 프로젝트의 GPUS_ALL_REGIONS quota가 1이므로 GPU VM을 동시에 둘 이상 만들지 않는다.
- 예상 총비용 USD 45를 hard cap으로 사용한다. 예상 누적 비용이 USD 40에 도달하면 남은
  작업과 예상 비용을 다시 계산하고, USD 45를 넘기 전에 VM을 정지한다.
- L4 재고가 없다고 V100으로 자동 fallback하지 않는다. T4/다른 region/더 큰 disk가
  필요하면 새 비용을 계산해 사용자에게 승인을 요청한다.
- Spot은 기본 사용하지 않는다. checkpoint/resume와 interruption recovery를 실제로
  검증한 뒤 사용자가 요청할 때만 사용한다.
- 기존 인스턴스가 있으면 새 인스턴스를 만들지 않는다.
- 작업 종료나 실패 시 VM을 반드시 정지한다. 결과 동기화와 checksum 검증 전에는 VM이나
  disk를 삭제하지 않는다.
- 기존 outputs_full, outputs_gate, outputs_mad103와 사용자 변경 파일을 수정·삭제하지 않는다.
  새 결과는 outputs_confirmatory와 synthetic_confirmatory에 저장한다.
- commit/push는 요청하지 않았으므로 하지 않는다.

먼저 수행할 로컬 작업
1. git status와 현재 설정, 테스트, run fingerprint, synthetic manifest를 조사한다.
2. src/data/analyze_long_tail.py의 allocation을 deterministic capped largest-remainder로 고친다.
   uniform B=1000, K=13, min=5, max=200에서 12개 77장과 1개 76장이 되는 테스트를 추가한다.
   모든 plan에 대해 합계, bounds, deterministic tie-breaking을 테스트한다.
3. 다음 네 arm을 구현한다.
   - aug_tail_uniform_inpaint
   - aug_tail_weighted_inpaint
   - aug_weak_uniform_inpaint (신규)
   - aug_weak_weighted_inpaint
   모든 arm은 B=1000, K=13, 동일 generator/prompt/QC를 사용한다.
4. configs/confirmatory.yaml을 만들되 planning은 val만 사용하고 최종 보고는 test로 고정한다.
   detector seeds는 [42,43,44]로 한다. test 결과로 plan이나 설정을 바꾸지 못하게 한다.
5. 기존 생성물 재사용 여부는 model id, diffusion parameters, prompt, source image, label,
   mask/QC 설정의 manifest/hash가 모두 일치할 때만 허용한다. 일치하면 잘못 배분된 초과분을
   제외하고 부족분과 신규 weak-uniform만 생성한다. 증명할 수 없으면 네 pool을 전량 재생성한다.
6. pytest를 실행하고, 실제 GPU 없이 plan 생성 dry-run을 수행한다. 다음 invariant를 기계적으로
   확인한다: arm당 1000장, 13개 class, uniform quota 76/77, tail/weak set disjoint,
   val-only planning, train-only synthetic insertion.
7. 통계 분석에 seed blocking을 추가하고 다음을 출력한다.
   - seed별 all/tail/weak mAP50-95
   - arm별 mean, SD, 95% CI
   - class-set policy × scope의 사전 정의 interaction contrast와 p-value
   - primary contrasts의 Holm-adjusted p-values
   - uniform vs weighted 차이와 CI
   단일 seed arm과 3-seed baseline 평균을 직접 비교하지 않는다.

Google Cloud read-only preflight
1. gcloud auth/config의 account와 project를 확인한다.
2. instances, disks, L4 quota, GPUS_ALL_REGIONS quota와 후보 zone의 accelerator availability를 확인한다.
3. 실행할 run 수, 예상 GPU-hour, machine/hour rate, disk 비용, 최대 예상비용을 표로 출력한다.
4. 예상비용이 USD 45 이하이고 로컬 테스트가 모두 통과할 때만 VM을 생성한다.

실행
- 기존 gcp_create_vm.sh를 검토해 이번 실행에서는 L4-only, duplicate-instance 방지,
  12시간 max-run-duration, failure trap, 자동 stop이 보장되도록 한다.
- 12시간 안에 끝나는 phase로 나눈다: prepare/generate, core training batch 1,
  core training batch 2 + RFS, final evaluation/statistics.
- 각 run 종료 즉시 weights, config, training_meta, val/test per-class metrics, stdout/stderr,
  plan 및 manifest를 durable storage에 동기화하고 checksum을 기록한다.
- 중단되면 완료된 fingerprint를 확인해 마지막 미완료 run부터 resume한다.
- baseline basic_aug 42/43/44와 RFS 42는 data/config/fingerprint가 완전히 일치할 때만 재사용한다.
- test metric을 보기 전에 plan과 primary contrasts를 JSON/Markdown으로 freeze하고 timestamp/hash를 남긴다.

완료 조건
- 네 diffusion arm × 3 seeds의 정상 종료와 test per-class AP가 모두 존재한다.
- RFS seeds 42/43/44가 동일 설정으로 존재한다.
- allocation 및 manifest invariant가 모두 통과한다.
- interaction/CI/Holm 결과가 outputs_confirmatory/analysis에 생성된다.
- 재현 명령, 실제 GPU-hour, 추정 비용, 실패/재시도 기록을 RESULTS_CONFIRMATORY.md에 정리한다.
- durable output checksum 검증 후 VM을 stop한다.
- 마지막 응답에는 완료/미완료 run, 실제 elapsed time, 추정 비용, 결과 경로와 논문 핵심 주장에
  미치는 영향을 요약한다.
```

Codex CLI 예시:

```bash
codex --ask-for-approval on-request exec --sandbox workspace-write \
  - < CODEX_GCLOUD_PROMPT.md
```
