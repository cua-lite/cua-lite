#!/usr/bin/env bash
set -euo pipefail

ADB=/root/Android/Sdk/platform-tools/adb
DEVICE=emulator-5554
AVD_NAME=${ANDROID_AVD_NAME:-lite_avd_androidworld}
APK_ZIP=/mnt/phoneworld_external_safe.zip
APK_DIR=/tmp/phoneworld-apks
EMU_LOG=/tmp/phoneworld-emulator.log

test -r "$APK_ZIP"
mkdir -p "$APK_DIR"
python - "$APK_ZIP" "$APK_DIR" <<'PY'
import sys, zipfile
from pathlib import Path, PurePosixPath

archive, target = Path(sys.argv[1]), Path(sys.argv[2])
with zipfile.ZipFile(archive) as zf:
    members = [info for info in zf.infolist() if info.filename.lower().endswith(".apk")]
    names = [PurePosixPath(info.filename).name for info in members]
    if len(members) != 34 or len(set(names)) != 34:
        raise SystemExit(f"expected 34 unique APKs, got {len(members)} entries/{len(set(names))} names")
    for info, name in zip(members, names, strict=True):
        if PurePosixPath(info.filename).is_absolute() or ".." in PurePosixPath(info.filename).parts:
            raise SystemExit(f"unsafe zip member: {info.filename}")
        (target / name).write_bytes(zf.read(info))
PY

emulator -avd "$AVD_NAME" -no-snapshot-load -no-window -no-metrics -no-audio \
  -gpu swiftshader_indirect -ports 5554,5555 -grpc 8554 >"$EMU_LOG" 2>&1 &
EMU_PID=$!
trap 'rc=$?; if [ "$rc" -ne 0 ]; then printf "%s\n" "$LINENO: $BASH_COMMAND" >/tmp/phoneworld-install-failure; fi; kill -9 "$EMU_PID" 2>/dev/null || true; exit "$rc"' EXIT

for _ in $(seq 1 150); do
  sleep 2
  if "$ADB" -s "$DEVICE" shell getprop sys.boot_completed 2>/dev/null | grep -q '^1'; then
    break
  fi
done
"$ADB" -s "$DEVICE" shell getprop sys.boot_completed | grep -q '^1'

expected_packages=$(find "$APK_DIR" -maxdepth 1 -type f -name '*.apk' -printf '%f\n' | sed 's/\.apk$//' | sort)
test "$(printf '%s\n' "$expected_packages" | sed '/^$/d' | wc -l)" -eq 34
"$ADB" -s "$DEVICE" install -r /usr/local/share/phoneworld/ADBKeyboard.apk | grep -q Success
while IFS= read -r apk; do
  package=$($ANDROID_SDK_ROOT/build-tools/33.0.0/aapt dump badging "$apk" | sed -n "s/^package: name='\([^']*\)'.*/\1/p")
  case "$package" in com.phoneuse.*) ;; *) echo "unexpected APK package: $package" >&2; exit 1 ;; esac
  "$ADB" -s "$DEVICE" install -r "$apk" | grep -q Success
done < <(find "$APK_DIR" -maxdepth 1 -type f -name '*.apk' -print | sort)

test "$("$ADB" -s "$DEVICE" shell pm list packages com.phoneuse | tr -d '\r' | sort -u | wc -l)" -eq 34
for _ in $(seq 1 30); do
  if "$ADB" -s "$DEVICE" shell ime list -a -s | tr -d '\r' | grep -qx 'com.android.adbkeyboard/.AdbIME'; then
    break
  fi
  sleep 1
done
"$ADB" -s "$DEVICE" shell ime list -a -s | tr -d '\r' | grep -qx 'com.android.adbkeyboard/.AdbIME'
"$ADB" -s "$DEVICE" shell ime enable com.android.adbkeyboard/.AdbIME
"$ADB" -s "$DEVICE" shell ime set com.android.adbkeyboard/.AdbIME
test "$("$ADB" -s "$DEVICE" shell settings get secure default_input_method | tr -d '\r')" = "com.android.adbkeyboard/.AdbIME"

# Real UTF-8 input smoke: focus Android Contacts' insert form, broadcast, then read UI text back.
"$ADB" -s "$DEVICE" shell am start -W -a android.intent.action.INSERT -t vnd.android.cursor.dir/contact >/dev/null
sleep 2
"$ADB" -s "$DEVICE" shell uiautomator dump /sdcard/phoneworld-ime.xml >/dev/null
"$ADB" -s "$DEVICE" pull /sdcard/phoneworld-ime.xml /tmp/phoneworld-ime.xml >/dev/null
coords=$(python - <<'PY'
import re, xml.etree.ElementTree as ET
for node in ET.parse('/tmp/phoneworld-ime.xml').iter('node'):
    if node.attrib.get('class') == 'android.widget.EditText':
        x1, y1, x2, y2 = map(int, re.findall(r'\d+', node.attrib['bounds']))
        print((x1 + x2) // 2, (y1 + y2) // 2)
        break
else:
    raise SystemExit('Contacts insert form has no EditText')
PY
)
read -r x y <<<"$coords"
"$ADB" -s "$DEVICE" shell input tap "$x" "$y"
payload=$(printf '中文测试' | base64 -w0)
"$ADB" -s "$DEVICE" shell am broadcast -a ADB_INPUT_B64 --es msg "$payload" | grep -q 'Broadcast completed'
sleep 1
"$ADB" -s "$DEVICE" shell uiautomator dump /sdcard/phoneworld-ime.xml >/dev/null
"$ADB" -s "$DEVICE" pull /sdcard/phoneworld-ime.xml /tmp/phoneworld-ime.xml >/dev/null
grep -q '中文测试' /tmp/phoneworld-ime.xml

"$ADB" -s "$DEVICE" shell am force-stop com.android.contacts || true
"$ADB" -s "$DEVICE" shell input keyevent 3
"$ADB" -s "$DEVICE" shell settings put system pointer_location 0
"$ADB" -s "$DEVICE" shell settings put system show_touches 0
test "$("$ADB" -s "$DEVICE" shell settings get secure default_input_method | tr -d '\r')" = "com.android.adbkeyboard/.AdbIME"
touch /tmp/phoneworld-smoke-ok
"$ADB" -s "$DEVICE" emu avd snapshot save default_boot || true
