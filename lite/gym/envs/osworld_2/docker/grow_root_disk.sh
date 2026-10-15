#!/usr/bin/env bash
if [ -z "${OSWORLD_VOLUME_SIZE:-}" ]; then
    return 0
fi
if [[ ! "$OSWORLD_VOLUME_SIZE" =~ ^[0-9]+$ ]] ||
   ((OSWORLD_VOLUME_SIZE < 1 || OSWORLD_VOLUME_SIZE > 100)); then
    echo "Invalid OSWORLD_VOLUME_SIZE: $OSWORLD_VOLUME_SIZE" >&2
    exit 64
fi
if ((OSWORLD_VOLUME_SIZE > 50)); then
    qemu-img resize -f qcow2 /boot.qcow2 "${OSWORLD_VOLUME_SIZE}G"
fi
