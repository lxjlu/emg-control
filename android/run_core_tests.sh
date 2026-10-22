#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
SDK_ROOT=${ANDROID_SDK_ROOT:-${ANDROID_HOME:-$HOME/Library/Android/sdk}}
PLATFORM_JAR="$SDK_ROOT/platforms/android-34/android.jar"
OUT_DIR="$SCRIPT_DIR/build/core-self-test"

mkdir -p "$OUT_DIR"
javac -cp "$PLATFORM_JAR" -d "$OUT_DIR" \
  "$SCRIPT_DIR/app/src/main/java/cn/semg/armband/SemgFrame.java" \
  "$SCRIPT_DIR/app/src/main/java/cn/semg/armband/SemgPacketParser.java" \
  "$SCRIPT_DIR/app/src/main/java/cn/semg/armband/GestureModel.java" \
  "$SCRIPT_DIR/tools/CoreSelfTest.java"
java -cp "$OUT_DIR:$PLATFORM_JAR" cn.semg.armband.CoreSelfTest
