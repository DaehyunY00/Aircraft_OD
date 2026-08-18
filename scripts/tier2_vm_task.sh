#!/bin/bash
# Tier 2 확장 실험 on-VM 태스크 (spot 인스턴스, 매 부팅 실행).
#
# Cell 순서: C2 mad_rtdetr → C3 mar20_yolo → C4 mar20_rtdetr.
# 설계:
#  - 모든 단계 멱등: 완료 마커 파일(GEN_DONE_*, CELL_DONE_*)로 스스로 건너뜀.
#  - spot 선점/12h max-run 어느 쪽이든 (sync → marker → shutdown) 순서 보장.
#  - 부팅당 11h20m 데드라인 self-stop (confirmatory에서 검증된 패턴).
#  - C2 시작 전 RT-DETR 1-run 타이밍 파일럿: basic_aug seed 42를 먼저 학습해
#    실측 시간을 마커로 보고 → 로컬에서 비용 재견적 가능. 이 run은 파이프라인이
#    fingerprint 일치로 재사용하므로 낭비가 아니다.
#  - 합성 pool 재사용: C2의 uniform/selective pool은 confirmatory 버킷에서 시딩
#    (파일명·내용이 (source, plan, class, idx, seed42)에 결정적 + 채택 전 픽셀 재검증).
set -uo pipefail
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/lib/google-cloud-sdk/bin:/opt/google-cloud-sdk/bin:/snap/bin:$PATH"

BUCKET=gs://military-od-tier2
CONF_BUCKET=gs://military-od-confirmatory
ROOT=/content/drive/MyDrive/Military_OD
CODE=/opt/military_od
BOOT_TS=$(date -u +%Y%m%dT%H%M%SZ)
START=$(date +%s)
MAX_SECONDS=$((11 * 3600 + 20 * 60))
LOG=/var/log/tier2_${BOOT_TS}.log
CELLS=(mad_rtdetr mar20_yolo mar20_rtdetr)

exec > >(tee -a "$LOG") 2>&1
echo "[BOOT] $BOOT_TS 시작 (tier2)"

# 과거 부팅 로그를 durable로 밀어 진단 가능하게 유지
for old_log in /var/log/tier2_*.log; do
  [ "$old_log" = "$LOG" ] && continue
  gcloud storage cp "$old_log" "$BUCKET/logs/" >/dev/null 2>&1 || true
done

PY=""
for cand in /opt/conda/bin/python /usr/bin/python3 /usr/local/bin/python3; do
  [ -x "$cand" ] || continue
  if "$cand" -c "import torch" >/dev/null 2>&1; then PY="$cand"; break; fi
done
[ -z "$PY" ] && for cand in /opt/conda/bin/python /usr/bin/python3; do [ -x "$cand" ] && PY="$cand" && break; done
echo "[SETUP] python=$PY"

mark() { echo "$2" | gcloud storage cp - "$BUCKET/markers/$1" >/dev/null 2>&1 || true; }

sync_out() {  # sync_out [skip_checksums]
  # skip_checksums=1 은 heartbeat 전용. 체크섬은 run마다 130MB짜리 .pt를 전부
  # 재해싱하므로(C2는 최대 12 run × 2 파일) 30분마다 돌리면 수 GB 디스크 I/O가
  # 학습 데이터로더와 경합한다. 주기 동기화는 rsync만 하고, 체크섬은 phase 종료·
  # finish() 시점에만 갱신한다 — 최종 검증에 쓰이는 값은 그때 것이면 충분하다.
  local skip_checksums=${1:-0}
  echo "[SYNC] durable 동기화 시작 ($(date -u)) skip_checksums=${skip_checksums}"
  local rc=0
  for cell in "${CELLS[@]}"; do
    local out="$ROOT/outputs_${cell}"
    [ -d "$out" ] || continue
    if [ "$skip_checksums" != "1" ]; then
      (
        cd "$ROOT" &&
        find "outputs_${cell}" -type f \
          \( -name '*.csv' -o -name '*.yaml' -o -name '*.json' -o -name '*.md' -o -name 'best.pt' -o -name 'last.pt' \) \
          -exec sha256sum {} + | sort -k2 > "outputs_${cell}/CHECKSUMS.sha256"
      ) || true
    fi
    gcloud storage rsync -r "$out" "$BUCKET/outputs_${cell}" || rc=1
  done
  for pool in synthetic_mad_rtdetr synthetic_mar20; do
    [ -d "$ROOT/$pool" ] || continue
    gcloud storage rsync -r -x '.*DRY_RUN_MARKER.*' "$ROOT/$pool" "$BUCKET/$pool" || rc=1
  done
  gcloud storage cp "$LOG" "$BUCKET/logs/" || true
  echo "[SYNC] 완료 rc=$rc ($(date -u))"
  return $rc
}

# 누적 가동 시간 카운터. phase 마커의 elapsed 합만으로는 선점당한 부팅이 전부
# 누락된다(2026-08-08: 8시간 동안 5회 선점 → phase 마커 0건 → 비용 상한 무력화).
# heartbeat가 주기적으로 갱신해 선점되더라도 직전 주기까지는 계상된다.
UPTIME_OBJ="$BUCKET/markers/uptime_seconds"
read_uptime() {
  local cur
  cur=$(gcloud storage cat "$UPTIME_OBJ" 2>/dev/null | tr -dc '0-9')
  echo "${cur:-0}"
}
bump_uptime() {  # bump_uptime <seconds>
  local add=$1 cur
  cur=$(read_uptime)
  echo $((cur + add)) | gcloud storage cp - "$UPTIME_OBJ" >/dev/null 2>&1 || true
}

# 선점은 예고 없이 VM을 STOP시키므로 finish()에 도달하지 못한다. 학습 산출물이
# 부팅 디스크에만 남아 durable 사본이 수 시간 뒤처지는 것을 막기 위해 주기적으로
# 동기화한다. 어떤 실패도 파이프라인을 죽이지 않도록 전부 || true.
HEARTBEAT_INTERVAL=1800
heartbeat_loop() {
  while true; do
    sleep "$HEARTBEAT_INTERVAL"
    bump_uptime "$HEARTBEAT_INTERVAL" || true
    sync_out 1 >/dev/null 2>&1 || true
    echo "[HEARTBEAT] $(date -u) 누적 가동 $(read_uptime)s"
  done
}

finish() {  # finish <ALL_DONE|PHASE_END|FAILED> <detail>
  local status=$1 detail=${2:-}
  local elapsed=$(( $(date +%s) - START ))
  echo "[FINISH] status=$status detail=$detail elapsed=${elapsed}s"
  [ -n "${HEARTBEAT_PID:-}" ] && kill "$HEARTBEAT_PID" 2>/dev/null
  # 이 부팅에서 heartbeat가 아직 계상하지 않은 잔여 구간만 더한다
  bump_uptime $(( elapsed % HEARTBEAT_INTERVAL )) || true
  sync_out || mark "SYNC_FAILED_${BOOT_TS}" "rsync 실패 — 디스크를 삭제하지 말 것"
  mark "phase_${BOOT_TS}_${status}" "elapsed=${elapsed}s detail=${detail}"
  [ "$status" = "ALL_DONE" ] && mark ALL_DONE "$(date -u) boot=$BOOT_TS"
  [ "$status" = "FAILED" ] && mark "FAILED_${BOOT_TS}" "detail=${detail}"
  sync
  shutdown -h now
  exit 0
}

remaining() { echo $(( MAX_SECONDS - ( $(date +%s) - START ) )); }
require_time() { if [ "$(remaining)" -lt 600 ]; then finish PHASE_END "잔여 시간 부족 — 다음 부팅에서 계속"; fi; }

run_step() {  # run_step <detail> <cmd...>
  local detail=$1; shift
  require_time
  timeout "$(remaining)s" "$@"
  local rc=$?
  if [ $rc -eq 124 ]; then finish PHASE_END "$detail 시간 초과 — 다음 부팅에서 resume"; fi
  if [ $rc -ne 0 ]; then finish FAILED "$detail exit=$rc"; fi
}

# ---------- 0. 재시작 금지 조건 (PC 독립 안전장치) ----------
# GCE instance schedule이 매시 VM을 되살리므로, 로컬 watch가 없어도 VM이 스스로
# 멈춰야 한다. watch가 담당했던 세 가지 정지 조건을 여기서 직접 검사한다.
if gcloud storage ls "$BUCKET/markers/ALL_DONE" >/dev/null 2>&1; then
  echo "[INFO] ALL_DONE 존재 — shutdown"; shutdown -h now; exit 0
fi
# FAILED는 사람이 원인을 보고 마커를 지울 때까지 재시작하지 않는다. 무인
# 재시작 루프가 같은 실패를 11시간씩 반복하며 과금하는 것을 막는다.
# (PHASE_END는 정상적인 시간 초과이므로 여기서 걸리지 않는다.)
if gcloud storage ls "$BUCKET/markers/" 2>/dev/null | grep -q '/FAILED_'; then
  echo "[INFO] FAILED 마커 존재 — 사람 개입 전까지 재시작 금지, shutdown"
  gcloud storage cp "$LOG" "$BUCKET/logs/" >/dev/null 2>&1 || true
  shutdown -h now; exit 0
fi
# 누적 가동 시간 상한. heartbeat 카운터(uptime_seconds)를 1차 근거로 쓰고,
# phase 마커 elapsed 합과 비교해 큰 값을 택한다 — 카운터 도입 이전 부팅과
# 카운터 쓰기 실패를 모두 감당하기 위함.
BUDGET_HOURS=150     # spot ~$0.36/h 기준 ≈ $54 + 디스크 ≈ 상한 $60 이내. 추정 소요 129h(실측 기반).
phase_marker_seconds() {
  local total=0 obj secs
  for obj in $(gcloud storage ls "$BUCKET/markers/" 2>/dev/null | grep '/phase_'); do
    secs=$(gcloud storage cat "$obj" 2>/dev/null | grep -o 'elapsed=[0-9]*' | head -1 | cut -d= -f2)
    [ -n "$secs" ] && total=$((total + secs))
  done
  echo "$total"
}
ACC_SECONDS=$(read_uptime)
PHASE_SECONDS=$(phase_marker_seconds)
[ "${PHASE_SECONDS:-0}" -gt "${ACC_SECONDS:-0}" ] && ACC_SECONDS=$PHASE_SECONDS
echo "[BUDGET] 누적 가동 ${ACC_SECONDS}s (약 $((ACC_SECONDS / 3600))h) / 상한 ${BUDGET_HOURS}h"
if [ "${ACC_SECONDS:-0}" -gt $((BUDGET_HOURS * 3600)) ]; then
  echo "$(date -u) 누적 ${ACC_SECONDS}s > ${BUDGET_HOURS}h" \
    | gcloud storage cp - "$BUCKET/markers/BUDGET_EXCEEDED" >/dev/null 2>&1 || true
  echo "[STOP] 누적 가동 시간 상한 초과 — shutdown (재개하려면 마커 확인 후 BUDGET_HOURS 상향)"
  shutdown -h now; exit 0
fi
if gcloud storage ls "$BUCKET/markers/BUDGET_EXCEEDED" >/dev/null 2>&1; then
  echo "[STOP] BUDGET_EXCEEDED 마커 존재 — shutdown"; shutdown -h now; exit 0
fi

# ---------- 1. 환경 ----------
nvidia-smi >/dev/null 2>&1 || finish FAILED "nvidia-smi 실패 (GPU/드라이버 없음)"
export DEBIAN_FRONTEND=noninteractive
apt-get install -y -q libgl1 libglib2.0-0 >/dev/null 2>&1 || true
if [ ! -f /var/tmp/tier2_deps_ok ]; then
  echo "[SETUP] python 의존성 설치"
  "$PY" -m pip install -q -r "$CODE/requirements.txt" || finish FAILED "pip install 실패"
  "$PY" -m pip uninstall -y -q torchaudio >/dev/null 2>&1 || true
  "$PY" -c "import torch; assert torch.cuda.is_available()" || finish FAILED "torch.cuda 사용 불가"
  touch /var/tmp/tier2_deps_ok
fi
"$PY" -m pip install -q tabulate || true   # 통계 md 렌더링 필수 (2026-08-06 사고 재발 방지)
mkdir -p /root/.kaggle
cp "$CODE/kaggle.json" /root/.kaggle/kaggle.json && chmod 600 /root/.kaggle/kaggle.json

# ---------- 2. 데이터/산출물 복원 (멱등) ----------
mkdir -p "$ROOT" /content/data
for cell in "${CELLS[@]}"; do
  if [ ! -d "$ROOT/outputs_${cell}/runs" ] && gcloud storage ls "$BUCKET/outputs_${cell}/" >/dev/null 2>&1; then
    echo "[RESTORE] outputs_${cell} 버킷 복원"
    mkdir -p "$ROOT/outputs_${cell}"
    gcloud storage rsync -r "$BUCKET/outputs_${cell}" "$ROOT/outputs_${cell}" || finish FAILED "outputs_${cell} 복원 실패"
  fi
done
for pool in synthetic_mad_rtdetr synthetic_mar20; do
  if [ ! -d "$ROOT/$pool" ] && gcloud storage ls "$BUCKET/$pool/" >/dev/null 2>&1; then
    echo "[RESTORE] $pool 버킷 복원"
    mkdir -p "$ROOT/$pool"
    gcloud storage rsync -r "$BUCKET/$pool" "$ROOT/$pool" || finish FAILED "$pool 복원 실패"
  fi
done
# C2 pool 시딩: confirmatory의 uniform/selective pool + rejected 마커 재사용
for plan in uniform selective; do
  if [ ! -d "$ROOT/synthetic_mad_rtdetr/$plan/images/train" ]; then
    echo "[RESTORE] C2 pool 시딩: $plan (confirmatory 재사용)"
    mkdir -p "$ROOT/synthetic_mad_rtdetr/$plan"
    gcloud storage rsync -r "$CONF_BUCKET/synthetic_confirmatory/$plan" "$ROOT/synthetic_mad_rtdetr/$plan" \
      || finish FAILED "C2 pool 시딩 실패: $plan"
  fi
done
# MAR20 원본 (수동 업로드 필수 — prepare 단계에서 업로드)
if [ ! -d /content/data/mar20_raw ]; then
  echo "[RESTORE] MAR20 원본 복원"
  if ! gcloud storage ls "$BUCKET/data/mar20_raw.tar.gz" >/dev/null 2>&1; then
    finish FAILED "MAR20 아카이브 없음: $BUCKET/data/mar20_raw.tar.gz — 로컬에서 'run_tier2_gcp.sh prepare'로 업로드 필요"
  fi
  mkdir -p /content/data/mar20_raw
  gcloud storage cp "$BUCKET/data/mar20_raw.tar.gz" /tmp/mar20_raw.tar.gz || finish FAILED "MAR20 다운로드 실패"
  tar xzf /tmp/mar20_raw.tar.gz -C /content/data/mar20_raw || finish FAILED "MAR20 압축 해제 실패"
fi

# ---------- 3. 파이프라인 ----------
cd "$CODE"
export PYTHONUNBUFFERED=1
export TQDM_DISABLE=1     # \r 진행바 금지 — metadata runner 사망 사고(2026-08-04) 재발 방지
mark "boot_${BOOT_TS}_started" "$(date -u)"
heartbeat_loop &
HEARTBEAT_PID=$!
echo "[HEARTBEAT] 시작 pid=$HEARTBEAT_PID (${HEARTBEAT_INTERVAL}s 주기 동기화·가동시간 계상)"

# 3-0. RT-DETR 타이밍 파일럿 (C2의 basic_aug seed 42 — 이후 파이프라인이 재사용)
if [ ! -f "$ROOT/outputs_mad_rtdetr/PILOT_DONE" ]; then
  echo "[PHASE] RT-DETR 타이밍 파일럿"
  run_step "파일럿 분석" "$PY" src/run_pipeline.py --config configs/mad_rtdetr.yaml --download --only-analysis
  run_step "파일럿 학습" "$PY" src/train/train_yolo.py \
    --data /content/data/processed/base/data.yaml --config configs/mad_rtdetr.yaml \
    --name basic_aug --seed 42 --basic-aug
  secs=$("$PY" - <<'PYEOF'
from pathlib import Path
import yaml
runs = sorted(Path("/content/drive/MyDrive/Military_OD/outputs_mad_rtdetr/runs").glob("basic_aug_rtdetr-l_seed42_*"))
meta = yaml.safe_load((runs[-1] / "training_meta.yaml").read_text(encoding="utf-8")) if runs else {}
print(int(meta.get("training_seconds") or 0))
PYEOF
)
  mark "PILOT_rtdetr_mad" "training_seconds=${secs} (~$(( secs / 60 ))min/run) — 로컬에서 비용 재견적 후 계속"
  touch "$ROOT/outputs_mad_rtdetr/PILOT_DONE"
  sync_out || true
fi

# 3-1..3-3. cell 순차 실행
for cell in "${CELLS[@]}"; do
  cfg="configs/${cell}.yaml"
  if [ -f "$ROOT/outputs_${cell}/CELL_DONE" ]; then
    echo "[SKIP] ${cell} 완료됨"
    continue
  fi
  if [ ! -f "$ROOT/outputs_${cell}/GEN_DONE" ]; then
    echo "[PHASE] ${cell}: 분석 + baseline + plan + 생성"
    run_step "${cell} 생성" "$PY" src/run_pipeline.py --config "$cfg" --download --stop-after-inpaint
    touch "$ROOT/outputs_${cell}/GEN_DONE"
    sync_out || true
  fi
  echo "[PHASE] ${cell}: 데이터셋 빌드 + 학습 + 평가 + 통계"
  run_step "${cell} 학습" "$PY" src/run_pipeline.py --config "$cfg" --skip-inpaint
  echo "[PHASE] ${cell}: 완료 조건 검증"
  "$PY" scripts/tier2_check.py --config "$cfg" || finish FAILED "${cell} 완료 조건 검증 실패"
  touch "$ROOT/outputs_${cell}/CELL_DONE"
  mark "CELL_DONE_${cell}" "$(date -u)"
  sync_out || true
done

echo "[PHASE] 전체 cell 최종 검증"
"$PY" scripts/tier2_check.py \
  --config configs/mad_rtdetr.yaml --config configs/mar20_yolo.yaml --config configs/mar20_rtdetr.yaml \
  || finish FAILED "최종 검증 실패"

finish ALL_DONE "3 cell 전부 run/통계/invariant 통과"
