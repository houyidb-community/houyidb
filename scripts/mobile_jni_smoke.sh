#!/usr/bin/env bash
# 主机 JVM JNI 冒烟：需要先运行 scripts/build_mobile_android_aar.sh 生成 AAR。
# 在 macOS/Linux 上编译主机 JNI cdylib，解包 AAR 的 classes.jar，编译并运行
# JniSmoke；验证 JNI 名称/参数编组/u64/save-load，无需真机或模拟器。
# EN: Host-JVM JNI smoke test; run scripts/build_mobile_android_aar.sh first to
# produce the AAR. Builds the host JNI cdylib, extracts the AAR classes.jar,
# compiles and runs JniSmoke to verify JNI names/argument marshalling/u64/
# save-load without a device or emulator.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AAR="$ROOT/houyi-mobile/android/build/outputs/aar/houyi-mobile-android-release.aar"
if [[ ! -f "$AAR" ]]; then
  echo "AAR not found: $AAR (run scripts/build_mobile_android_aar.sh first)" >&2
  exit 1
fi

cd "$ROOT"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"
cargo build --release -p houyi-mobile-jni

case "$(uname -s)" in
  Darwin) LIB_EXT="dylib" ;;
  Linux)  LIB_EXT="so" ;;
  *) echo "unsupported host $(uname -s)" >&2; exit 1 ;;
 esac
HOST_LIB="$CARGO_TARGET_DIR/release/libhouyi_mobile_jni.$LIB_EXT"
if [[ ! -f "$HOST_LIB" ]]; then
  echo "host JNI library not found: $HOST_LIB" >&2
  exit 1
fi
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/lib" "$WORK/classes" "$WORK/aar"
cp "$HOST_LIB" "$WORK/lib/libhouyi_mobile.$LIB_EXT"
unzip -o -q "$AAR" classes.jar -d "$WORK/aar"

KOTLIN_STDLIB="${KOTLIN_STDLIB:-$(find "${HOME}/.gradle/caches/modules-2/files-2.1/org.jetbrains.kotlin/kotlin-stdlib/2.4.21" -name 'kotlin-stdlib-2.4.21.jar' 2>/dev/null | head -n 1 || true)}"
if [[ -z "$KOTLIN_STDLIB" || ! -f "$KOTLIN_STDLIB" ]]; then
  echo "kotlin-stdlib jar not found; set KOTLIN_STDLIB" >&2
  exit 1
fi
CP="$WORK/aar/classes.jar:$KOTLIN_STDLIB"
javac -cp "$CP" -d "$WORK/classes" "$ROOT/houyi-mobile/android/jni-smoke/JniSmoke.java"
java -Djava.library.path="$WORK/lib" -Xcheck:jni -cp "$WORK/classes:$CP" JniSmoke
