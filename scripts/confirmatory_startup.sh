#!/bin/bash
# VM startup-script (매 부팅 실행): 최신 코드 tarball을 받아 태스크를 실행한다.
# 실패해도 VM이 켜진 채 과금되지 않도록 어떤 경로든 마지막에 shutdown.
#
# 주의: startup-script는 최소 PATH로 실행된다 — gcloud가 PATH에 없으면 다운로드도
# 마커 기록도 조용히 실패하고 shutdown만 남는다(2026-08-04 1차 부팅 사고).
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/lib/google-cloud-sdk/bin:/opt/google-cloud-sdk/bin:/snap/bin:$PATH"
echo "[startup] $(date -u) gcloud=$(command -v gcloud || echo MISSING)"
BUCKET=gs://military-od-confirmatory
mkdir -p /opt/military_od
if ! command -v gcloud >/dev/null 2>&1; then
  echo "[startup] gcloud를 찾지 못했습니다 — shutdown"
  shutdown -h now
  exit 0
fi
if ! gcloud storage cp "$BUCKET/code/code.tar.gz" /tmp/code.tar.gz; then
  echo "code.tar.gz 다운로드 실패" | gcloud storage cp - "$BUCKET/markers/FAILED_bootstrap_$(date -u +%Y%m%dT%H%M%SZ)" || true
  shutdown -h now
  exit 0
fi
rm -rf /opt/military_od/src /opt/military_od/configs /opt/military_od/scripts /opt/military_od/tests
tar xzf /tmp/code.tar.gz -C /opt/military_od
# 태스크 출력을 startup-script stdout으로 흘리지 않는다. tqdm의 \r 갱신은 개행 없는
# 초대형 한 줄이 되는데, metadata script runner의 줄 스캐너(64KB)가 "token too long"으로
# 죽으면서 게스트 종료로 이어졌다(2026-08-04 boot4~6, conf-b 1차의 실제 원인).
bash /opt/military_od/scripts/confirmatory_vm_task.sh > /var/log/confirmatory_task_runner.log 2>&1
rc=$?
# 여기 도달 = 태스크가 finish()(sync+마커+shutdown)를 거치지 않고 리턴했다는 뜻.
# 원인 구분을 위해 마커를 남긴다. 이 마커조차 없다면 외부 요인 poweroff다.
echo "task rc=$rc $(date -u)" | gcloud storage cp - "gs://military-od-confirmatory/markers/TASK_EXIT_rc${rc}_$(date -u +%Y%m%dT%H%M%SZ)" || true
gcloud storage cp /var/log/confirmatory_*.log gs://military-od-confirmatory/logs/ || true
shutdown -h now
