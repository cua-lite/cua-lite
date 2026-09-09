#!/usr/bin/env bash
set -Eeuo pipefail

storage_dir="${STORAGE:-/storage}"
disk_path="$storage_dir/${DISK_NAME:-data}.qcow2"
guest_ip="${WAA_GUEST_IP:-20.20.20.21}"

if [[ ! -c /dev/kvm || ! -r /dev/kvm || ! -w /dev/kvm ]]; then
  echo "WindowsAgentArena runner requires readable and writable /dev/kvm" >&2
  exit 69
fi
# Create the disposable overlay here so each VM needs only one container boot.
if [[ ! -e "$disk_path" ]]; then
  backing_disk=/images/base.qcow2
  [[ -f /images/ready.qcow2 ]] && backing_disk=/images/ready.qcow2
  qemu-img create -f qcow2 -F qcow2 -b "$backing_disk" "$disk_path"
fi
if [[ ! -s "$disk_path" ]]; then
  echo "WindowsAgentArena runner requires a non-empty disk at $disk_path" >&2
  exit 64
fi
qemu-img info "$disk_path" >/dev/null
touch "$storage_dir/windows.boot"

qemu_pid=""
bridge_pid=""
asset_server_pid=""
cleanup() {
  [[ -n "$asset_server_pid" ]] && kill "$asset_server_pid" 2>/dev/null || true
  [[ -n "$bridge_pid" ]] && kill "$bridge_pid" 2>/dev/null || true
  [[ -n "$qemu_pid" ]] && kill "$qemu_pid" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Keep snapshot restore from stalling in synchronous huge-page compaction on
# fragmented hosts. This setting follows only the QEMU launcher and its children.
python - <<'PY' &
import ctypes
import os

libc = ctypes.CDLL(None, use_errno=True)
libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
if libc.prctl(41, 1, 0, 0, 0) != 0:  # PR_SET_THP_DISABLE
    error = ctypes.get_errno()
    raise OSError(error, os.strerror(error))
os.execl("/run/entry.sh", "/run/entry.sh")
PY
qemu_pid=$!

if [[ -n "${WAA_INCOMING_STATE:-}" ]]; then
  # Snapshot restore: QEMU was launched with `-incoming defer`, so it starts paused
  # waiting for migration input. Feed it the saved RAM+device state through the HMP
  # monitor; the guest then resumes at the WAA-ready point (the probe loop below
  # waits for that) instead of cold-booting.
  incoming_deadline=$((SECONDS + 180))
  until (exec 3<>/dev/tcp/127.0.0.1/7100) 2>/dev/null; do
    exec 3>&- 3<&- 2>/dev/null || true
    if ! kill -0 "$qemu_pid" 2>/dev/null; then
      echo "QEMU exited before snapshot restore could start" >&2
      wait "$qemu_pid"
    fi
    (( SECONDS >= incoming_deadline )) && { echo "QEMU monitor never came up for restore" >&2; exit 71; }
    sleep 1
  done
  exec 3>&- 3<&- 2>/dev/null || true
  python -c "
import socket, sys, time
s = socket.create_connection(('127.0.0.1', 7100), timeout=10); s.settimeout(1.0); time.sleep(0.3)
try:
    while True: s.recv(65536)
except Exception: pass
s.sendall(('migrate_incoming \"exec:cat %s\"\n' % '${WAA_INCOMING_STATE}').encode()); time.sleep(1)
s.settimeout(60); buf = b''
try:
    while True:
        d = s.recv(65536)
        if not d: break
        buf += d
        if buf.rstrip().endswith(b'(qemu)'): break
except socket.timeout:
    pass
out = buf.decode('utf-8', 'replace')
sys.exit(1 if 'rror' in out else 0)
" || { echo "migrate_incoming failed to start" >&2; exit 71; }
fi

deadline=$((SECONDS + ${WAA_GUEST_READY_TIMEOUT_S:-900}))
until curl --fail --silent --max-time 2 "http://$guest_ip:5000/probe" >/dev/null; do
  if ! kill -0 "$qemu_pid" 2>/dev/null; then
    echo "QEMU exited before the WAA guest server became ready" >&2
    wait "$qemu_pid"
  fi
  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for WAA guest server at $guest_ip:5000" >&2
    exit 70
  fi
  sleep 5
done

python -m http.server 5051 \
  --bind 127.0.0.1 \
  --directory /opt/waa-assets \
  >/tmp/waa-asset-server.log 2>&1 &
asset_server_pid=$!
for _ in {1..20}; do
  curl --fail --silent --max-time 1 "http://127.0.0.1:5051/.complete.json" >/dev/null && break
  if ! kill -0 "$asset_server_pid" 2>/dev/null; then
    cat /tmp/waa-asset-server.log >&2 || true
    exit 70
  fi
  sleep 0.25
done
curl --fail --silent --max-time 2 "http://127.0.0.1:5051/.complete.json" >/dev/null

gunicorn \
  --bind "0.0.0.0:${WAA_BRIDGE_PORT:-5050}" \
  --workers 1 \
  --threads 1 \
  --timeout 1800 \
  --access-logfile - \
  --error-logfile - \
  bridge:app &
bridge_pid=$!

wait -n "$qemu_pid" "$bridge_pid" "$asset_server_pid"
