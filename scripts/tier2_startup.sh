#!/bin/bash
# Tier 2 VM startup-script (매 부팅 실행): 최신 코드 tarball을 받아 태스크 실행.
# 실패해도 VM이 켜진 채 과금되지 않도록 어떤 경로든 마지막에 shutdown.
# startup-script는 최소 PATH — gcloud 부재 시 조용한 실패 방지(2026-08-04 사고).
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/lib/google-cloud-sdk/bin:/opt/google-cloud-sdk/bin:/snap/bin:$PATH"
echo "[startup] $(date -u) gcloud=$(command -v gcloud || echo MISSING)"
BUCKET=gs://military-od-tier2
mkdir -p /opt/military_od
if ! command -v gcloud >/dev/null 2>&1; then
  echo "[startup] gcloud 없음 — shutdown"; shutdown -h now; exit 0
fi
if ! gcloud storage cp "$BUCKET/code/code.tar.gz" /tmp/code.tar.gz; then
  echo "code.tar.gz 다운로드 실패" | gcloud storage cp - "$BUCKET/markers/FAILED_bootstrap_$(date -u +%Y%m%dT%H%M%SZ)" || true
  shutdown -h now; exit 0
fi
rm -rf /opt/military_od/src /opt/military_od/configs /opt/military_od/scripts /opt/military_od/tests
tar xzf /tmp/code.tar.gz -C /opt/military_od
# 태스크 출력은 파일로 리다이렉트 — tqdm \r 한 줄이 metadata runner의 64KB 줄
# 스캐너를 죽였던 사고(2026-08-04) 재발 방지.
bash /opt/military_od/scripts/tier2_vm_task.sh > /var/log/tier2_task_runner.log 2>&1
rc=$?
echo "task rc=$rc $(date -u)" | gcloud storage cp - "$BUCKET/markers/TASK_EXIT_rc${rc}_$(date -u +%Y%m%dT%H%M%SZ)" || true
gcloud storage cp /var/log/tier2_*.log "$BUCKET/logs/" || true
shutdown -h now
