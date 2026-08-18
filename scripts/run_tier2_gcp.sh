#!/bin/bash
# Tier 2 확장 실험 로컬 오케스트레이터 (spot L4).
#
#   bash scripts/run_tier2_gcp.sh prepare [MAR20_DIR]  # 버킷/IAM + 코드 + MAR20 업로드
#   bash scripts/run_tier2_gcp.sh create               # spot L4 VM 생성
#   bash scripts/run_tier2_gcp.sh watch                # 마커 폴링 + 자동 재시작 + 비용 상한
#   bash scripts/run_tier2_gcp.sh status               # 상태 1회 출력
#   bash scripts/run_tier2_gcp.sh download             # ALL_DONE 후 결과 다운로드 + checksum
#   bash scripts/run_tier2_gcp.sh stop                 # VM 수동 정지
#
# 안전 규칙:
#  - GPU VM 동시 1대 (GPUS_ALL_REGIONS quota 1). L4 전용, fallback 없음.
#  - Spot 인스턴스: 선점 시 STOP → watch가 재시작 (선점 잦으면 boots 증가).
#  - 비용 추정 $50 경고, $57에서 정지 (hard cap $60, 사용자 승인 2026-08-07).
#  - 비용은 RUNNING 관측 시간 × 요율로 추정 — spot 요율은 변동하므로 보수값 사용.
set -uo pipefail

PROJECT=project-d522190f-d377-47af-bf2
INSTANCE=military-od-tier2
BUCKET=gs://military-od-tier2
CONF_BUCKET=gs://military-od-confirmatory
ZONES=(us-central1-a us-central1-b us-central1-c)
MACHINE=g2-standard-8
GPU=nvidia-l4
IMAGE_FAMILY=pytorch-2-9-cu129-ubuntu-2204-nvidia-580
IMAGE_PROJECT=deeplearning-platform-release
DISK_GB=150            # 3 cell 산출물 + MAD/MAR20 + pool 4종 (confirmatory 100GB에서 상향)
MAX_RUN=12h
RATE_PER_HOUR=0.36     # g2-standard-8+L4 spot 보수 추정 (변동, on-demand 0.854)
COST_WARN=50
COST_CAP=60
# 실측 선점률(2026-08-08: 10시간에 6회, 가동 구간 29~156분)이면 전체 실험에
# 200회 이상 재시작이 필요하다. 40이면 이틀 만에 소진돼 watch가 손을 놓는다.
MAX_BOOTS=400          # spot 선점 재시작 포함
# instance schedule은 GCP 제약상 시간당 1회가 최소라, 선점 직후 최대 59분을
# 놀게 된다(실측 가동률 62%). watch를 함께 돌리면 5분 내 재시작해 ~88%까지
# 올라간다. 둘은 충돌하지 않는다 — 실행 중인 VM에 대한 start는 no-op이다.
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STATE_DIR="$HOME/.military_od_tier2"
COST_FILE="$STATE_DIR/running_seconds"
BOOTS_FILE="$STATE_DIR/boots"
WATCH_LOG="$STATE_DIR/watch.log"
mkdir -p "$STATE_DIR"

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

vm_status() {
  gcloud compute instances list --project="$PROJECT" --filter="name=$INSTANCE" \
    --format="value(status,zone)" 2>/dev/null
}

# awk로 계산한다. python3에 의존하면 안 된다 — Windows(Git Bash)에서 python3는
# 실행되지 않는 스텁이고(exit 49) 그 결과 est_cost가 빈 문자열이 되어 아래 상한
# 판정이 조용히 전부 거짓이 됐다(2026-08-08 확인, 상한 감시가 무력화된 상태였음).
est_cost() {
  local secs=0
  [ -f "$COST_FILE" ] && secs=$(cat "$COST_FILE")
  # 디스크 150GB pd-balanced ≈ $15/월 — 관측 기간 보수 가산 +$2
  awk -v s="${secs:-0}" -v r="$RATE_PER_HOUR" 'BEGIN{printf "%.2f", s/3600*r + 2.0}'
}

# 부동소수 비교: 조건이 참이면 exit 0. awk는 Git Bash/Linux 모두에 항상 있다.
cost_at_least() { awk -v c="${1:-0}" -v t="$2" 'BEGIN{exit !(c+0 >= t+0)}'; }

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
  log "compute SA IAM 확인 (tier2 objectAdmin + confirmatory objectViewer)"
  local project_number sa
  project_number=$(gcloud projects describe "$PROJECT" --format="value(projectNumber)") || return 1
  sa="serviceAccount:${project_number}-compute@developer.gserviceaccount.com"
  gcloud storage buckets add-iam-policy-binding "$BUCKET" --member="$sa" --role=roles/storage.objectAdmin >/dev/null || return 1
  gcloud storage buckets add-iam-policy-binding "$CONF_BUCKET" --member="$sa" --role=roles/storage.objectViewer >/dev/null || return 1
  log "코드 tarball 업로드"
  local tarball
  tarball=$(mktemp /tmp/code.XXXX.tar.gz)
  # __pycache__ 제외: Google Drive FUSE 마운트에서 tar가 이 디렉터리를 읽다가
  # "Cannot savedir: File too large"로 죽는다(2026-08-08 확인). VM에서 쓰이지도
  # 않는 산출물이라(Windows .pyc는 Linux에서 무효) 빼는 것이 맞다.
  ( cd "$REPO_DIR" && tar czf "$tarball" \
      --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' \
      src configs scripts tests requirements.txt kaggle.json ) || return 1
  gcloud storage cp "$tarball" "$BUCKET/code/code.tar.gz" || return 1
  rm -f "$tarball"
  # MAR20 아카이브 업로드 (인자 경로 또는 기본 위치, 이미 버킷에 있으면 생략)
  if gcloud storage ls "$BUCKET/data/mar20_raw.tar.gz" >/dev/null 2>&1; then
    log "MAR20 아카이브가 이미 버킷에 있음 — 업로드 생략"
  else
    local mar20_dir="${1:-$REPO_DIR/data/mar20_raw}"
    if [ ! -d "$mar20_dir" ]; then
      log "[경고] MAR20 원본 디렉터리 없음: $mar20_dir"
      log "  NWPU 배포 링크에서 수동 다운로드 후: bash scripts/run_tier2_gcp.sh prepare <MAR20_DIR>"
      log "  (JPEGImages / Annotations/'Horizontal Bounding Boxes' / ImageSets/Main 포함 구조)"
      return 1
    fi
    log "MAR20 아카이브 생성/업로드: $mar20_dir"
    local mar20_tar
    mar20_tar=$(mktemp /tmp/mar20.XXXX.tar.gz)
    ( cd "$mar20_dir" && tar czf "$mar20_tar" . ) || return 1
    gcloud storage cp "$mar20_tar" "$BUCKET/data/mar20_raw.tar.gz" || return 1
    rm -f "$mar20_tar"
  fi
  log "prepare 완료"
}

cmd_create() {
  local existing
  existing=$(vm_status)
  if [ -n "$existing" ]; then
    log "인스턴스가 이미 존재: $existing — 새로 만들지 않음"
    return 0
  fi
  for zone in "${ZONES[@]}"; do
    log "spot VM 생성 시도: $GPU @ $zone"
    if gcloud compute instances create "$INSTANCE" \
        --project="$PROJECT" \
        --zone="$zone" \
        --machine-type="$MACHINE" \
        --accelerator="type=$GPU,count=1" \
        --image-family="$IMAGE_FAMILY" \
        --image-project="$IMAGE_PROJECT" \
        --boot-disk-size="${DISK_GB}GB" \
        --boot-disk-type=pd-balanced \
        --provisioning-model=SPOT \
        --instance-termination-action=STOP \
        --max-run-duration="$MAX_RUN" \
        --metadata-from-file=startup-script="$REPO_DIR/scripts/tier2_startup.sh" \
        --scopes=cloud-platform 2>/tmp/tier2_create_err.log; then
      log "VM 생성 성공: $zone (spot 과금 시작)"
      echo $(( $( [ -f "$BOOTS_FILE" ] && cat "$BOOTS_FILE" || echo 0 ) + 1 )) > "$BOOTS_FILE"
      return 0
    fi
    if grep -q "QUOTA" /tmp/tier2_create_err.log; then
      log "할당량 부족 — 중단"; cat /tmp/tier2_create_err.log; return 2
    fi
    log "재고 부족: $zone"
  done
  return 1
}

cmd_status() {
  log "VM: $(vm_status || echo '없음')"
  log "누적 추정 비용: \$$(est_cost) (running $(( $( [ -f "$COST_FILE" ] && cat "$COST_FILE" || echo 0 ) / 3600 ))h, boots $( [ -f "$BOOTS_FILE" ] && cat "$BOOTS_FILE" || echo 0 ))"
  log "마커:"
  gcloud storage ls -l "$BUCKET/markers/" 2>/dev/null | tail -15
  if gcloud storage ls "$BUCKET/markers/PILOT_rtdetr_mad" >/dev/null 2>&1; then
    log "RT-DETR 파일럿: $(gcloud storage cat "$BUCKET/markers/PILOT_rtdetr_mad" 2>/dev/null)"
  fi
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
  log "watch 시작 (5분 간격, spot) — 로그: $WATCH_LOG"
  local pilot_reported=0
  while true; do
    if marker_exists ALL_DONE; then
      log "ALL_DONE — 실험 완료. VM 정지 후 종료"
      stop_vm
      log "다음 단계: bash scripts/run_tier2_gcp.sh download"
      return 0
    fi
    if [ "$pilot_reported" -eq 0 ] && marker_exists PILOT_rtdetr_mad; then
      log "[파일럿] RT-DETR 실측: $(gcloud storage cat "$BUCKET/markers/PILOT_rtdetr_mad" 2>/dev/null)"
      pilot_reported=1
    fi
    local failed
    failed=$(gcloud storage ls "$BUCKET/markers/" 2>/dev/null | grep -c "FAILED_") || true
    if [ "${failed:-0}" -gt 0 ]; then
      log "FAILED 마커 ${failed}건 — VM 정지 후 수동 개입 필요"
      gcloud storage ls "$BUCKET/markers/" | grep "FAILED_" | tail -3
      stop_vm
      return 1
    fi
    local cost
    cost=$(est_cost)
    if cost_at_least "$cost" "$((COST_CAP-3))"; then
      log "비용 상한 근접(\$$cost ≥ \$$((COST_CAP-3))) — hard cap 전 VM 정지"
      stop_vm
      return 2
    fi
    if cost_at_least "$cost" "$COST_WARN"; then
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
          log "최대 부팅 횟수($MAX_BOOTS) 도달 — 자동 재시작 중단"
          return 3
        fi
        local zone
        zone=$(echo "$status_zone" | awk '{print $2}')
        log "VM 정지 상태(선점/12h/self-stop) & 미완료 — 재시작 (boot $((boots+1))/$MAX_BOOTS)"
        if gcloud compute instances start "$INSTANCE" --project="$PROJECT" --zone="$zone" --quiet; then
          echo $((boots + 1)) > "$BOOTS_FILE"
        else
          log "재시작 실패(spot 재고 부족 가능) — 다음 주기에 재시도"
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
  local rc=0
  for cell in mad_rtdetr mar20_yolo mar20_rtdetr; do
    local dest="$REPO_DIR/outputs_${cell}"
    mkdir -p "$dest"
    log "결과 다운로드: $BUCKET/outputs_${cell} → $dest"
    gcloud storage rsync -r "$BUCKET/outputs_${cell}" "$dest" || { rc=1; continue; }
    log "checksum 검증: outputs_${cell}"
    # VM이 sha256sum으로 만든 표준 형식이므로 같은 도구로 검증한다. python3에
    # 의존하지 않는다 — Windows(Git Bash)에서 python3는 실행되지 않는 스텁이라
    # 실험 종료 직후 검증이 통째로 실패했을 것이다.
    local sumcmd=""
    if command -v sha256sum >/dev/null 2>&1; then sumcmd="sha256sum -c"
    elif command -v shasum >/dev/null 2>&1; then sumcmd="shasum -a 256 -c"
    fi
    if [ -z "$sumcmd" ]; then
      log "sha256sum/shasum을 찾지 못해 검증을 건너뜁니다 — 수동 확인 필요"
      rc=1
    elif [ ! -f "$dest/CHECKSUMS.sha256" ]; then
      log "CHECKSUMS.sha256이 없습니다: outputs_${cell} — 검증 불가"
      rc=1
    elif ( cd "$REPO_DIR" && $sumcmd "outputs_${cell}/CHECKSUMS.sha256" > /tmp/tier2_sum_${cell}.log 2>&1 ); then
      log "checksum 일치: outputs_${cell} ($(grep -c ': OK' /tmp/tier2_sum_${cell}.log)건)"
    else
      log "checksum 불일치: outputs_${cell} — 버킷 보존, 조사 필요"
      grep -v ': OK' /tmp/tier2_sum_${cell}.log 2>/dev/null | head -10
      rc=1
    fi
  done
  [ $rc -eq 0 ] && log "3 cell 다운로드 완료. RESULTS_TIER2.md 작성 가능"
  return $rc
}

case "${1:-}" in
  prepare) shift || true; cmd_prepare "$@" ;;
  create) cmd_create ;;
  watch) cmd_watch 2>&1 | tee -a "$WATCH_LOG" ;;
  status) cmd_status ;;
  download) cmd_download ;;
  stop) stop_vm ;;
  *) echo "usage: $0 {prepare [MAR20_DIR]|create|watch|status|download|stop}"; exit 64 ;;
esac
