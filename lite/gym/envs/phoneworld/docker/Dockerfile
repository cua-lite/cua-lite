FROM cua-lite/androidworld:base AS keyboard-builder

ARG ADB_KEYBOARD_COMMIT=98dc653e8c5e98b3bf9a69561b723545bc5fd5da
RUN mkdir -p /src/ADBKeyBoard /out/source \
 && cd /src/ADBKeyBoard \
 && git init -q \
 && git fetch -q --depth 1 https://github.com/senzhk/ADBKeyBoard.git "$ADB_KEYBOARD_COMMIT" \
 && git checkout -q "$ADB_KEYBOARD_COMMIT" \
 && test "$(git rev-parse HEAD)" = "$ADB_KEYBOARD_COMMIT" \
 && rm -f ADBKeyboard.apk \
 && ./gradlew --no-daemon :keyboardservice:assembleDebug \
 && cp keyboardservice/build/outputs/apk/debug/keyboardservice-debug.apk /out/ADBKeyboard.apk \
 && cp -a . /out/source/ \
 && rm -rf /out/source/.git /out/source/.gradle /out/source/keyboardservice/build \
 && find /out/source -name '*.apk' -delete

FROM cua-lite/androidworld:base

RUN apt-get update && apt-get install -y --no-install-recommends sqlite3 \
 && rm -rf /var/lib/apt/lists/*
COPY --from=keyboard-builder /out/ADBKeyboard.apk /usr/local/share/phoneworld/ADBKeyboard.apk
COPY --from=keyboard-builder /out/source /usr/share/src/ADBKeyBoard
COPY apps.sh /usr/local/bin/phoneworld-apps.sh
RUN chmod +x /usr/local/bin/phoneworld-apps.sh \
 && mkdir -p /usr/share/licenses/ADBKeyBoard \
 && cp /usr/share/src/ADBKeyBoard/LICENSE /usr/share/licenses/ADBKeyBoard/LICENSE \
 && printf '%s\n' \
      'ADBKeyBoard source: https://github.com/senzhk/ADBKeyBoard' \
      'Version: v2.4-dev / 98dc653e8c5e98b3bf9a69561b723545bc5fd5da' \
      'Built from source without using the repository prebuilt APK.' \
      > /usr/share/src/ADBKeyBoard/CUA-LITE-SOURCE-NOTICE
