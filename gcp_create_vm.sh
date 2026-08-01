#!/bin/bash
# GPU VM 생성 재시도 스크립트.
#
# 2026-07-31 시점에 us-central1의 L4/V100/T4가 모든 존에서 재고 부족(STOCKOUT)이라
# 만들었다. 재고는 시간대를 타므로 같은 명령을 나중에 다시 돌리면 통하는 경우가 많다.
# 리전을 옮기는 선택지는 일부러 빼놓았다 — 다른 리전은 NVIDIA_*_GPUS 할당량이 0이라
# 신청·승인에 1~2일이 걸려 재시도보다 느리다.
#
# 사용법:
#   bash gcp_create_vm.sh            # 한 번만 시도
#   bash gcp_create_vm.sh --watch    # 성공할 때까지 20분 간격 재시도
#   bash gcp_create_vm.sh --watch 5  # 5분 간격
set -uo pipefail

INSTANCE=military-od
PROJECT=project-d522190f-d377-47af-bf2
IMAGE_FAMILY=pytorch-2-9-cu129-ubuntu-2204-nvidia-580
IMAGE_PROJECT=deeplearning-platform-release
DISK_GB=100
MAX_RUN=12h

# "GPU:머신타입:존들". 비용·성능 순서로 배열했다.
#  - L4  : 시간당 가장 저렴하고 학습도 충분히 빠름. g2 계열에만 붙는다.
#  - V100: L4보다 비싸지만 6 run을 12h 상한 안에 여유 있게 끝낸다.
#  - T4  : 가장 저렴하나 느려서 12h 상한에 아슬아슬하다. 최후 선택.
COMBOS=(
  "nvidia-l4:g2-standard-8:us-central1-a us-central1-b us-central1-c"
  "nvidia-tesla-v100:n1-standard-8:us-central1-a us-central1-b us-central1-c us-central1-f"
  "nvidia-tesla-t4:n1-standard-8:us-central1-a us-central1-b us-central1-c us-central1-f"
)

WATCH=0
INTERVAL=$((20 * 60))
if [ "${1:-}" = "--watch" ]; then
  WATCH=1
  [ -n "${2:-}" ] && INTERVAL=$(( $2 * 60 ))
fi

# 이미 떠 있는데 또 만들면 인스턴스가 둘이 되어 요금이 두 배로 나간다.
# 매 시도 전에 확인한다.
already_running() {
  local found
  found=$(gcloud compute instances list --project="$PROJECT" \
            --filter="name=$INSTANCE" --format="value(name,zone,status)" 2>/dev/null)
  if [ -n "$found" ]; then
    echo "[SKIP] 인스턴스가 이미 존재합니다: $found"
    return 0
  fi
  return 1
}

try_all() {
  local log=/tmp/gcp_vm_try.log
  for combo in "${COMBOS[@]}"; do
    local gpu="${combo%%:*}"
    local rest="${combo#*:}"
    local machine="${rest%%:*}"
    local zones="${rest#*:}"
    for z in $zones; do
      printf '  %-18s %-16s ... ' "$gpu" "$z"
      if gcloud compute instances create "$INSTANCE" \
          --project="$PROJECT" \
          --zone="$z" \
          --machine-type="$machine" \
          --accelerator="type=$gpu,count=1" \
          --image-family="$IMAGE_FAMILY" \
          --image-project="$IMAGE_PROJECT" \
          --boot-disk-size="${DISK_GB}GB" \
          --boot-disk-type=pd-balanced \
          --maintenance-policy=TERMINATE \
          --max-run-duration="$MAX_RUN" \
          --instance-termination-action=STOP \
          --scopes=cloud-platform >"$log" 2>&1; then
        echo "성공"
        cat <<EOF

=========================================================
 VM 생성됨: $gpu / $machine / $z
 이 시점부터 과금이 시작됩니다.
=========================================================

접속:
  gcloud compute ssh $INSTANCE --zone=$z

작업이 끝나면 반드시 정지 (안 하면 계속 과금):
  gcloud compute instances stop $INSTANCE --zone=$z

실험이 완전히 끝나면 삭제 (정지 상태에서도 디스크는 과금):
  gcloud compute instances delete $INSTANCE --zone=$z
EOF
        return 0
      fi
      # 재고 부족과 할당량 부족은 대응이 다르다. 전자는 재시도, 후자는 신청이 필요.
      if grep -q "QUOTA_EXCEEDED\|Quota .* exceeded" "$log"; then
        echo "할당량 부족 (재시도해도 소용없음)"
      else
        echo "재고 부족"
      fi
    done
  done
  return 1
}

already_running && exit 0

if [ "$WATCH" -eq 0 ]; then
  echo "[$(date '+%H:%M:%S')] GPU VM 생성 시도"
  try_all && exit 0
  echo
  echo "모든 조합이 재고 부족입니다. 나중에 다시 실행하거나 --watch 로 걸어두세요."
  exit 1
fi

cat <<EOF
[watch] $((INTERVAL / 60))분 간격으로 성공할 때까지 재시도합니다. Ctrl-c 로 중단.

주의: 자리를 비운 사이에 VM이 만들어지면 그 시점부터 과금됩니다.
      --max-run-duration=$MAX_RUN 이 걸려 있어 최악의 경우에도 $MAX_RUN 후 자동 정지되지만,
      돌아오시면 바로 접속해 작업을 시작하거나 정지시키세요.

EOF
while true; do
  echo "[$(date '+%H:%M:%S')] 시도"
  already_running && exit 0
  if try_all; then
    printf '\a'  # 터미널 벨로 알림
    exit 0
  fi
  echo "[$(date '+%H:%M:%S')] 전부 실패. $((INTERVAL / 60))분 후 재시도."
  sleep "$INTERVAL"
done
