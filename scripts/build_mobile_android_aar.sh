#!/usr/bin/env bash
# 构建 Android AAR：先交叉编译 JNI cdylib（arm64-v8a），再交给 Gradle library
# 模块打包。产物 houyi-mobile/android/build/outputs/aar/*.aar 供 pamobile 直接
# 以本地 module 或 flatDir 引入。
#
# EN: Build the Android AAR: cross-compile the JNI cdylib for arm64-v8a, then
# package it with the Gradle library module. The AAR under
# houyi-mobile/android/build/outputs/aar/ can be consumed by pamobile directly.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ANDROID_DIR="$ROOT/houyi-mobile/android"
TARGET="aarch64-linux-android"
JNI_DIR="$ANDROID_DIR/src/main/jniLibs/arm64-v8a"

if [[ -z "${ANDROID_HOME:-}" ]]; then
  ANDROID_HOME="/opt/homebrew/share/android-commandlinetools"
fi
export ANDROID_HOME
export ANDROID_SDK_ROOT="$ANDROID_HOME"

if [[ -z "${ANDROID_NDK_HOME:-}" ]]; then
  # 便携选取 NDK：按字典序取最后一个目录（BSD/GNU 通用），目录缺失时保持空。
  # EN: Portable NDK pick: last directory in lexical order (works with BSD/GNU
  # sort); stays empty when the directory is missing.
  ndk_candidates=("$ANDROID_HOME"/ndk/*)
  if [[ -d "${ndk_candidates[0]}" ]]; then
    ANDROID_NDK_HOME="${ndk_candidates[${#ndk_candidates[@]}-1]}"
  fi
fi
if [[ -z "${ANDROID_NDK_HOME:-}" || ! -d "$ANDROID_NDK_HOME" ]]; then
  echo "Android NDK not found under $ANDROID_HOME/ndk (set ANDROID_NDK_HOME)" >&2
  exit 1
fi
case "$(uname -s)" in
  Darwin) NDK_HOST_TAG="darwin-x86_64" ;;
  Linux)  NDK_HOST_TAG="linux-x86_64" ;;
  *) echo "unsupported host $(uname -s)" >&2; exit 1 ;;
esac
# 链接 API 级别需与 houyi-mobile/android/build.gradle.kts 的 minSdk 一致。
# EN: The linker API level must match minSdk in build.gradle.kts.
API_LEVEL="${API_LEVEL:-26}"
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="${ANDROID_NDK_HOME}/toolchains/llvm/prebuilt/${NDK_HOST_TAG}/bin/aarch64-linux-android${API_LEVEL}-clang"

# Rust 构建（mobile profile：opt-level=s + LTO + strip）。锚定到仓库根，
# 并尊重 CARGO_TARGET_DIR。
# EN: Rust build with the size-optimized mobile profile, anchored at the repo
# root and honoring CARGO_TARGET_DIR.
cd "$ROOT"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"
cargo build --profile mobile -p houyi-mobile-jni --target "$TARGET"
JNI_LIB="$CARGO_TARGET_DIR/$TARGET/mobile/libhouyi_mobile_jni.so"
if [[ ! -f "$JNI_LIB" ]]; then
  echo "JNI library not found: $JNI_LIB" >&2
  exit 1
fi

mkdir -p "$JNI_DIR"
# 文件名必须是 Kotlin System.loadLibrary("houyi_mobile") 对应的 libhouyi_mobile.so。
# EN: The filename must be libhouyi_mobile.so for System.loadLibrary.
cp "$JNI_LIB" "$JNI_DIR/libhouyi_mobile.so"
# 显式 strip：AGP 在部分 NDK/工具链组合下无法识别 Rust 产物，AAR 会保留
# 未 strip 的符号表；用 NDK llvm-strip 保证体积。
# EN: Strip explicitly: AGP cannot always recognize Rust artifacts, leaving
# symbol tables in the AAR; use NDK llvm-strip for a predictable size.
STRIP="${ANDROID_NDK_HOME}/toolchains/llvm/prebuilt/${NDK_HOST_TAG}/bin/llvm-strip"
if [[ -x "$STRIP" ]]; then
  "$STRIP" --strip-unneeded "$JNI_DIR/libhouyi_mobile.so"
fi
ls -lh "$JNI_DIR/libhouyi_mobile.so"

cd "$ANDROID_DIR"
./gradlew :assembleRelease "$@"

AAR="$ANDROID_DIR/build/outputs/aar/houyi-mobile-android-release.aar"
if [[ ! -f "$AAR" ]]; then
  echo "AAR not produced: $AAR" >&2
  exit 1
fi
# 先捕获 listing 再 grep：避免 pipefail 下 grep 提前退出导致 unzip 收到 SIGPIPE。
# EN: Capture the listing before grepping so grep's early exit under pipefail
# cannot SIGPIPE unzip and report a false missing-library failure.
listing="$(unzip -l "$AAR")"
if ! grep -q "jni/arm64-v8a/libhouyi_mobile.so" <<<"$listing"; then
  echo "AAR is missing jni/arm64-v8a/libhouyi_mobile.so" >&2
  exit 1
fi
# 用文件字节数而不是 du：APFS/文件系统的块分配会让 du 偏大，导致文档与实测不一致。
# EN: Report file bytes rather than du; block allocation on APFS inflates du and
# makes the number drift from the documented artifact size.
aar_bytes=$(wc -c <"$AAR" | tr -d ' ')
echo "AAR: $AAR ($((aar_bytes / 1024)) KB, $aar_bytes bytes)"
