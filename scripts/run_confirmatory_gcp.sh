#!/bin/bash
# 확인 실험 로컬 오케스트레이터.
#
#   bash scripts/run_confirmatory_gcp.sh prepare   # 버킷 생성 + 코드/베이스라인/rejected 업로드
#   bash scripts/run_confirmatory_gcp.sh create    # L4 VM 생성 (재고 없으면 1회 실패)
#   bash scripts/run_confirmatory_gcp.sh watch     # 마커 폴링 + 자동 재시작 + 비용 상한 (무인 실행용)
#   bash scripts/run_confirmatory_gcp.sh status    # 상태 1회 출력
#   bash scripts/run_confirmatory_gcp.sh download  # ALL_DONE 후 결과 다운로드 + checksum 검증
#
# 안전 규칙:
#  - GPU VM은 동시에 1대만 (GPUS_ALL_REGIONS quota 1)
#  - L4 전용. 다른 GPU로 자동 fallback하지 않는다.
#  - 예상 누적 비용 USD 40 도달 시 경고, 45 도달 전 VM 정지 (hard cap)
set -uo pipefail

PROJECT=project-d522190f-d377-47af-bf2
# 2026-08-04 zone-a VM(military-od-conf)은 L4 STOCKOUT으로 재시작 불가 + diffusion 중
# guest 종료 3회로 정지 보존 중. 새 인스턴스는 재고가 확인된 b/c에 만든다.
INSTANCE=military-od-conf-b
BUCKET=gs://military-od-confirmatory
ZONES=(us-central1-b us-central1-c)
MACHINE=g2-standard-8
GPU=nvidia-l4
IMAGE_FAMILY=pytorch-2-9-cu129-ubuntu-2204-nvidia-580
IMAGE_PROJECT=deeplearning-platform-release
DISK_GB=100
MAX_RUN=12h
RATE_PER_HOUR=0.854          # g2-standard-8 + L4 us-central1 on-demand (USD)
COST_WARN=40
COST_CAP=45
MAX_BOOTS=12
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STATE_DIR="$HOME/.military_od_confirmatory"
COST_FILE="$STATE_DIR/running_seconds"
BOOTS_FILE="$STATE_DIR/boots"
WATCH_LOG="$STATE_DIR/watch.log"
mkdir -p "$STATE_DIR"

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

vm_status() {
  gcloud compute instances list --project="$PROJECT" --filter="name=$INSTANCE" \
    --format="value(status,zone)" 2>/dev/null
}

est_cost() {
  local secs=0
  [ -f "$COST_FILE" ] && secs=$(cat "$COST_FILE")
  # 디스크: 100GB pd-balanced ≈ $10/월 ≈ $0.014/h (관측 시간 전체에 부과된다고 보수적으로 가정해 +$1)
  python3 -c "print(f'{ $secs/3600 * $RATE_PER_HOUR + 1.0 :.2f}')"
}

add_running_time() {
  local add=$1 secs=0
  [ -f "$COST_FILE" ] && secs=$(cat "$COST_FILE")
  echo $((secs + add)) > "$COST_FILE"
}

marker_exists() { gcloud storage ls "$BUCKET/markers/$1" >/dev/null 2>&1; }

cmd_prepare() {
  log "버킷 확인/생성"
  if ! gcloud storage ls "$BUCKET/" >/dev/null 2>&1; then
    gcloud storage buckets create "$BUCKET" --project="$PROJECT" --location=us-central1 \
      --uniform-bucket-level-access || return 1
  fi
  log "코드 tarball 업로드"
  local tarball
  tarball=$(mktemp /tmp/code.XXXX.tar.gz)
  ( cd "$REPO_DIR" && tar czf "$tarball" src configs scripts tests requirements.txt kaggle.json ) || return 1
  gcloud storage cp "$tarball" "$BUCKET/code/code.tar.gz" || return 1
  rm -f "$tarball"
  log "baseline runs 업로드 (real_only/basic_aug × 42/43/44)"
  for run in real_only_yolov8n_seed4{2,3,4}_* basic_aug_yolov8n_seed4{2,3,4}_*; do
    for d in "$REPO_DIR"/outputs_full/runs/$run; do
      [ -d "$d" ] || continue
      gcloud storage rsync -r "$d" "$BUCKET/baseline_runs/$(basename "$d")" || return 1
    done
  done
  log "rejected pool 마커 업로드 (재기각 fast-path용)"
  for plan in uniform selective weakness; do
    if [ -d "$REPO_DIR/synthetic_full/$plan/rejected" ]; then
      gcloud storage rsync -r "$REPO_DIR/synthetic_full/$plan/rejected" "$BUCKET/rejected_pools/$plan" || return 1
    fi
  done
  log "prepare 완료"
}

cmd_create() {
  local existing
  existing=$(vm_status)
  if [ -n "$existing" ]; then
    log "인스턴스가 이미 존재합니다: $existing — 새로 만들지 않음"
    return 0
  fi
  for zone in "${ZONES[@]}"; do
    log "VM 생성 시도: $GPU @ $zone"
    if gcloud compute instances create "$INSTANCE" \
        --project="$PROJECT" \
        --zone="$zone" \
        --machine-type="$MACHINE" \
        --accelerator="type=$GPU,count=1" \
        --image-family="$IMAGE_FAMILY" \
        --image-project="$IMAGE_PROJECT" \
        --boot-disk-size="${DISK_GB}GB" \
        --boot-disk-type=pd-balanced \
        --maintenance-policy=TERMINATE \
        --max-run-duration="$MAX_RUN" \
        --instance-termination-action=STOP \
        --metadata-from-file=startup-script="$REPO_DIR/scripts/confirmatory_startup.sh" \
        --scopes=cloud-platform 2>/tmp/create_err.log; then
      log "VM 생성 성공: $zone (과금 시작)"
      echo $(( $( [ -f "$BOOTS_FILE" ] && cat "$BOOTS_FILE" || echo 0 ) + 1 )) > "$BOOTS_FILE"
      return 0
    fi
    if grep -q "QUOTA" /tmp/create_err.log; then
      log "할당량 부족 — 중단 (재시도 무의미)"; cat /tmp/create_err.log; return 2
    fi
    log "재고 부족: $zone"
  done
  return 1
}

cmd_status() {
  log "VM: $(vm_status || echo '없음')"
  log "누적 추정 비용: \$$(est_cost) (running $(( $( [ -f "$COST_FILE" ] && cat "$COST_FILE" || echo 0 ) / 3600 ))h, boots $( [ -f "$BOOTS_FILE" ] && cat "$BOOTS_FILE" || echo 0 ))"
  log "마커:"
  gcloud storage ls -l "$BUCKET/markers/" 2>/dev/null | tail -12
}

stop_vm() {
  local status_zone zone
  status_zone=$(vm_status)
  zone=$(echo "$status_zone" | awk '{print $2}')
  if [ -n "$zone" ]; then
    log "VM 정지: $zone"
    gcloud compute instances stop "$INSTANCE" --project="$PROJECT" --zone="$zone" --quiet || true
  fi
}

cmd_watch() {
  log "watch 시작 (5분 간격) — 로그: $WATCH_LOG"
  while true; do
    if marker_exists ALL_DONE; then
      log "ALL_DONE 마커 확인 — 실험 완료. VM 정지 확인 후 종료"
      stop_vm
      log "다음 단계: bash scripts/run_confirmatory_gcp.sh download"
      return 0
    fi
    local failed
    failed=$(gcloud storage ls "$BUCKET/markers/" 2>/dev/null | grep -c "FAILED_") || true
    if [ "${failed:-0}" -gt 0 ] && ! marker_exists ALL_DONE; then
      log "FAILED 마커 ${failed}건 존재 — VM 정지 후 수동 개입 필요"
      gcloud storage ls "$BUCKET/markers/" | grep "FAILED_" | tail -3
      stop_vm
      return 1
    fi
    local cost
    cost=$(est_cost)
    if python3 -c "exit(0 if $cost >= $COST_CAP - 2 else 1)"; then
      log "비용 상한 근접(\$$cost ≥ \$$((COST_CAP-2))) — hard cap 전 VM 정지"
      stop_vm
      return 2
    fi
    if python3 -c "exit(0 if $cost >= $COST_WARN else 1)"; then
      log "[경고] 누적 추정 비용 \$$cost ≥ \$$COST_WARN — 남은 작업 재견적 필요"
    fi
    local status_zone status boots
    status_zone=$(vm_status)
    status=$(echo "$status_zone" | awk '{print $1}')
    boots=$( [ -f "$BOOTS_FILE" ] && cat "$BOOTS_FILE" || echo 0 )
    case "$status" in
      RUNNING)
        add_running_time 300
        log "RUNNING (누적 \$$cost)"
        ;;
      TERMINATED|STOPPED)
        if [ "$boots" -ge "$MAX_BOOTS" ]; then
          log "최대 부팅 횟수($MAX_BOOTS) 도달 — 자동 재시작 중단, 수동 확인 필요"
          return 3
        fi
        local zone
        zone=$(echo "$status_zone" | awk '{print $2}')
        log "VM 정지 상태 & 미완료 — 재시작 (boot $((boots+1))/$MAX_BOOTS)"
        if gcloud compute instances start "$INSTANCE" --project="$PROJECT" --zone="$zone" --quiet; then
          echo $((boots + 1)) > "$BOOTS_FILE"
        else
          log "재시작 실패(재고 부족 가능) — 다음 주기에 재시도"
        fi
        ;;
      "")
        log "VM 없음 — 생성 시도"
        cmd_create || log "생성 실패 — 다음 주기에 재시도"
        ;;
      *)
        log "상태: $status (대기)"
        ;;
    esac
    sleep 300
  done
}

cmd_download() {
  if ! marker_exists ALL_DONE; then
    log "ALL_DONE 마커가 없습니다 — 다운로드 중단"
    return 1
  fi
  local dest="$REPO_DIR/outputs_confirmatory"
  mkdir -p "$dest"
  log "결과 다운로드: $BUCKET/outputs_confirmatory → $dest"
  gcloud storage rsync -r "$BUCKET/outputs_confirmatory" "$dest" || return 1
  log "checksum 검증"
  if ( cd "$REPO_DIR" && shasum -a 256 -c outputs_confirmatory/CHECKSUMS.sha256 > /tmp/confirmatory_checksum.log 2>&1 ); then
    log "checksum 전부 일치 ($(grep -c ': OK' /tmp/confirmatory_checksum.log)건)"
  else
    log "checksum 불일치 — 디스크/버킷 보존, 조사 필요:"
    grep -v ': OK' /tmp/confirmatory_checksum.log | head -10
    return 1
  fi
  log "생성 로그/plan 다운로드 완료. RESULTS_CONFIRMATORY.md 작성 가능"
}

case "${1:-}" in
  prepare) cmd_prepare ;;
  create) cmd_create ;;
  watch) cmd_watch 2>&1 | tee -a "$WATCH_LOG" ;;
  status) cmd_status ;;
  download) cmd_download ;;
  stop) stop_vm ;;
  *) echo "usage: $0 {prepare|create|watch|status|download|stop}"; exit 64 ;;
esac
