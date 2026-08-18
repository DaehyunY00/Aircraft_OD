#!/bin/bash
# 확인 실험 on-VM 태스크. startup-script가 매 부팅마다 실행한다.
#
# 설계:
#  - 디스크가 보존되므로 모든 단계는 멱등: 이미 끝난 단계는 스스로 건너뛴다.
#  - 부팅당 11시간 20분 안에 스스로 동기화 후 shutdown (12h max-run-duration 이전).
#  - 어떤 실패 경로에서도 (sync → marker → shutdown) 순서를 보장한다.
#  - 진행 상태는 gs://military-od-confirmatory/markers/ 로만 보고한다 (SSH 불필요).
set -uo pipefail
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/lib/google-cloud-sdk/bin:/opt/google-cloud-sdk/bin:/snap/bin:$PATH"

BUCKET=gs://military-od-confirmatory
GATE_BUCKET=gs://military-od-gate
ROOT=/content/drive/MyDrive/Military_OD
CODE=/opt/military_od
BOOT_TS=$(date -u +%Y%m%dT%H%M%SZ)
START=$(date +%s)
MAX_SECONDS=$((11 * 3600 + 20 * 60))
LOG=/var/log/confirmatory_${BOOT_TS}.log

exec > >(tee -a "$LOG") 2>&1
echo "[BOOT] $BOOT_TS 시작"

# 이전 부팅이 finish()를 못 거치고 죽으면(마커·sync 없는 guestTerminate) 로그가
# 디스크에만 남는다. 매 부팅 초입에 과거 로그를 전부 durable로 밀어 진단 가능하게 한다.
for old_log in /var/log/confirmatory_*.log; do
  [ "$old_log" = "$LOG" ] && continue
  gcloud storage cp "$old_log" "gs://military-od-confirmatory/logs/" >/dev/null 2>&1 || true
done
# 직전 부팅의 커널/시스템 로그 꼬리도 올린다 — diffusion 부하 중 guestTerminate가
# 반복되고 있어(2026-08-04 boot4/5) 커널 panic/Xid/OOM 흔적을 확인해야 한다.
for sys_log in /var/log/syslog.1 /var/log/kern.log.1 /var/log/syslog /var/log/kern.log; do
  [ -f "$sys_log" ] || continue
  name=$(basename "$sys_log" | tr '.' '_')
  tail -c 300000 "$sys_log" | gcloud storage cp - "gs://military-od-confirmatory/logs/prev_${name}_${BOOT_TS}.txt" >/dev/null 2>&1 || true
done

# 이미지마다 python 위치가 다르다(구 DLVM은 /opt/conda, 신형은 시스템 python).
# torch가 이미 들어 있는 인터프리터를 우선 선택한다.
PY=""
for cand in /opt/conda/bin/python /usr/bin/python3 /usr/local/bin/python3; do
  [ -x "$cand" ] || continue
  if "$cand" -c "import torch" >/dev/null 2>&1; then PY="$cand"; break; fi
done
if [ -z "$PY" ]; then
  for cand in /opt/conda/bin/python /usr/bin/python3; do
    [ -x "$cand" ] && PY="$cand" && break
  done
fi
echo "[SETUP] python=$PY | /opt=$(ls /opt 2>/dev/null | tr '\n' ' ')"
"$PY" -c "import torch; print('[SETUP] torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>&1 || true

mark() {  # mark <name> <message>
  echo "$2" | gcloud storage cp - "$BUCKET/markers/$1" >/dev/null 2>&1 || true
}

sync_out() {
  echo "[SYNC] durable storage 동기화 시작 ($(date -u))"
  if [ -d "$ROOT/outputs_confirmatory" ]; then
    (
      cd "$ROOT" &&
      find outputs_confirmatory -type f \
        \( -name '*.csv' -o -name '*.yaml' -o -name '*.json' -o -name '*.md' -o -name 'best.pt' -o -name 'last.pt' \) \
        -exec sha256sum {} + | sort -k2 > outputs_confirmatory/CHECKSUMS.sha256
    ) || true
    gcloud storage rsync -r "$ROOT/outputs_confirmatory" "$BUCKET/outputs_confirmatory" || return 1
  fi
  if [ -d "$ROOT/synthetic_confirmatory" ]; then
    gcloud storage rsync -r -x '.*DRY_RUN_MARKER.*' \
      "$ROOT/synthetic_confirmatory" "$BUCKET/synthetic_confirmatory" || return 1
  fi
  gcloud storage cp "$LOG" "$BUCKET/logs/" || true
  echo "[SYNC] 완료 ($(date -u))"
  return 0
}

finish() {  # finish <ALL_DONE|PHASE_END|FAILED> <detail>
  local status=$1 detail=${2:-}
  local elapsed=$(( $(date +%s) - START ))
  echo "[FINISH] status=$status detail=$detail elapsed=${elapsed}s"
  sync_out || mark "SYNC_FAILED_${BOOT_TS}" "rsync 실패 — 디스크를 삭제하지 말 것"
  mark "phase_${BOOT_TS}_${status}" "elapsed=${elapsed}s detail=${detail}"
  if [ "$status" = "ALL_DONE" ]; then mark ALL_DONE "$(date -u) boot=$BOOT_TS"; fi
  if [ "$status" = "FAILED" ]; then mark "FAILED_${BOOT_TS}" "detail=${detail}"; fi
  sync
  shutdown -h now
  exit 0
}

remaining() { echo $(( MAX_SECONDS - ( $(date +%s) - START ) )); }

require_time() {  # 최소 10분이 안 남으면 새 단계를 시작하지 않는다
  if [ "$(remaining)" -lt 600 ]; then finish PHASE_END "잔여 시간 부족 — 다음 부팅에서 계속"; fi
}

# ---------- 0. 이미 끝났으면 바로 종료 ----------
if gcloud storage ls "$BUCKET/markers/ALL_DONE" >/dev/null 2>&1; then
  echo "[INFO] ALL_DONE 마커 존재 — 즉시 shutdown"
  shutdown -h now
  exit 0
fi

# ---------- 1. 환경 ----------
if ! nvidia-smi >/dev/null 2>&1; then
  finish FAILED "nvidia-smi 실패 (GPU/드라이버 없음)"
fi
export DEBIAN_FRONTEND=noninteractive
apt-get install -y -q libgl1 libglib2.0-0 >/dev/null 2>&1 || true
if [ ! -f /var/tmp/confirmatory_deps_ok ]; then
  echo "[SETUP] python 의존성 설치"
  "$PY" -m pip install -q -r "$CODE/requirements.txt" || finish FAILED "pip install 실패"
  "$PY" -m pip uninstall -y -q torchaudio >/dev/null 2>&1 || true  # 이미지 잔재 ABI 깨짐 (검증된 조치)
  "$PY" -c "import torch; assert torch.cuda.is_available()" || finish FAILED "torch.cuda 사용 불가"
  touch /var/tmp/confirmatory_deps_ok
fi
mkdir -p /root/.kaggle
cp "$CODE/kaggle.json" /root/.kaggle/kaggle.json && chmod 600 /root/.kaggle/kaggle.json
# deps 마커가 이미 있어도 소량 추가 의존성은 매 부팅 보장한다
# (tabulate 부재로 통계 md 렌더링이 실패해 완료 검증이 막혔던 2026-08-06 사고 재발 방지)
"$PY" -m pip install -q tabulate || true

# ---------- 2. 데이터 복원 (멱등) ----------
mkdir -p "$ROOT" /content/data
# 이전 부팅 산출물이 디스크에 없으면(디스크 재생성 등) durable 사본에서 복원
if [ ! -d "$ROOT/outputs_confirmatory/runs" ] && gcloud storage ls "$BUCKET/outputs_confirmatory/" >/dev/null 2>&1; then
  echo "[RESTORE] outputs_confirmatory 버킷 복원"
  mkdir -p "$ROOT/outputs_confirmatory"
  gcloud storage rsync -r "$BUCKET/outputs_confirmatory" "$ROOT/outputs_confirmatory" || finish FAILED "outputs 복원 실패"
fi
# baseline run 재사용 (real_only/basic_aug 42/43/44 — 데이터 내용 동일 근거는 RESULTS_CONFIRMATORY.md 참조)
mkdir -p "$ROOT/outputs_confirmatory/runs"
if ! ls "$ROOT/outputs_confirmatory/runs" | grep -q '^basic_aug_'; then
  echo "[RESTORE] baseline runs 복원"
  gcloud storage cp -r "$BUCKET/baseline_runs/*" "$ROOT/outputs_confirmatory/runs/" || finish FAILED "baseline runs 복원 실패"
fi
# 합성 pool 재사용: gate 버킷의 검증된 pool + rejected 마커
for plan in uniform selective weakness; do
  if [ ! -d "$ROOT/synthetic_confirmatory/$plan/images/train" ]; then
    echo "[RESTORE] synthetic pool 복원: $plan"
    mkdir -p "$ROOT/synthetic_confirmatory/$plan"
    gcloud storage rsync -r "$GATE_BUCKET/synthetic_full/$plan" "$ROOT/synthetic_confirmatory/$plan" \
      || finish FAILED "pool 복원 실패: $plan"
  fi
  if [ ! -d "$ROOT/synthetic_confirmatory/$plan/rejected" ] && gcloud storage ls "$BUCKET/rejected_pools/$plan/" >/dev/null 2>&1; then
    mkdir -p "$ROOT/synthetic_confirmatory/$plan/rejected"
    gcloud storage rsync -r "$BUCKET/rejected_pools/$plan" "$ROOT/synthetic_confirmatory/$plan/rejected" || true
  fi
done

# ---------- 3. 파이프라인 (11h20m 데드라인, resume 안전) ----------
cd "$CODE"
export PYTHONUNBUFFERED=1
# \r 기반 진행바를 전부 끈다 — metadata runner의 줄 스캐너를 죽였던 원흉이고,
# 로그 파일 크기도 수백 MB로 불린다. [시간] 콜백 로그(개행 기반)는 그대로 남는다.
export TQDM_DISABLE=1
mark "boot_${BOOT_TS}_started" "$(date -u)"

if [ ! -f "$ROOT/outputs_confirmatory/GEN_DONE" ]; then
  echo "[PHASE] 분석 + plan + 생성 (--stop-after-inpaint)"
  require_time
  timeout "$(remaining)s" "$PY" src/run_pipeline.py --config configs/confirmatory.yaml --download --stop-after-inpaint
  rc=$?
  if [ $rc -eq 124 ]; then finish PHASE_END "생성 단계 시간 초과 — 다음 부팅에서 resume"; fi
  if [ $rc -ne 0 ]; then finish FAILED "생성 단계 exit=$rc"; fi
  touch "$ROOT/outputs_confirmatory/GEN_DONE"
  sync_out || true
fi

echo "[PHASE] 데이터셋 빌드 + 학습 + 평가 + 통계 (--skip-inpaint)"
require_time
timeout "$(remaining)s" "$PY" src/run_pipeline.py --config configs/confirmatory.yaml --skip-inpaint
rc=$?
if [ $rc -eq 124 ]; then finish PHASE_END "학습 단계 시간 초과 — 다음 부팅에서 resume"; fi
if [ $rc -ne 0 ]; then finish FAILED "학습/평가 단계 exit=$rc"; fi

echo "[PHASE] 완료 조건 기계 검증"
"$PY" scripts/confirmatory_check.py --config configs/confirmatory.yaml
rc=$?
if [ $rc -ne 0 ]; then finish FAILED "완료 조건 검증 실패 exit=$rc"; fi

finish ALL_DONE "모든 run/통계/invariant 통과"
