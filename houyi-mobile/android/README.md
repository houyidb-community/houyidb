# houyi-mobile Android AAR

Android library 工程，把 houyi-mobile 的 **Kotlin 门面 + JNI 桥 + arm64-v8a
native 库** 打成 AAR；下游（如 pamobile）以本地 module 或 flatDir 方式引入。

EN: Android library that packages the houyi-mobile Kotlin facade, JNI bridge,
and arm64-v8a native library into an AAR consumable as a local module or
flatDir dependency (e.g. by pamobile).

## 构建

```bash
./scripts/build_mobile_android_aar.sh
```

- 先交叉编译 `houyi-mobile-jni`（mobile profile，LTO+strip），复制为
  `src/main/jniLibs/arm64-v8a/libhouyi_mobile.so`，再由 Gradle `assembleRelease`。
- 依赖：Android SDK 34、NDK ≥26、Gradle 8.9（仓库自带 wrapper）、Java 17。
- 产物：`houyi-mobile/android/build/outputs/aar/houyi-mobile-android-release.aar`
  （约 404 KB；`jni/arm64-v8a/libhouyi_mobile.so` 未压缩约 763 KB）。
- Gradle/AGP 首次构建可能需要联网拉取 AGP 8.5.2 / Kotlin 2.4.21。

EN: The script cross-compiles `houyi-mobile-jni` with the size-optimized mobile
profile, copies it as `libhouyi_mobile.so`, then runs Gradle
`assembleRelease`. Requirements: SDK 34, NDK >= 26, Gradle 8.9 (wrapper
included), Java 17. Output is the AAR above (~404 KB; the native library is
~696 KB uncompressed). First Gradle builds may fetch AGP 8.5.2 / Kotlin 2.4.21.

## Kotlin API

```kotlin
import cc.houyidb.mobile.HouyiDB

val db = HouyiDB.open()
HouyiDB.createCollection(db, "faces", 4, "flat-clustered") // 历史默认 f16/余弦
HouyiDB.insertBatch(
    db, "faces",
    longArrayOf(9_007_199_254_740_993L, 42L),
    floatArrayOf(1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f),
    2, 4,
)
HouyiDB.build(db, "faces")
val hits = HouyiDB.search(db, "faces", floatArrayOf(1f, 0f, 0f, 0f), 10) // SearchHit(id: Long, score: Float)
HouyiDB.save(db, "/data/.../houyi.snap")
val restored = HouyiDB.load("/data/.../houyi.snap")
HouyiDB.close(restored)
HouyiDB.close(db)
```

- 显式 metric/dtype：
  `createCollection(db, "l1", 4, "flat", HouyiDB.Metric.L1, HouyiDB.Dtype.F32)`；
  内存紧张时 `flat-clustered` 可用 `HouyiDB.Dtype.I8`（每维 1 B + 每向量 4 B scale，
  召回略低于 f16）。
- 所有调用失败抛 `IllegalStateException`，附带 `HouyiDB.lastError()`。
- 新代码使用 `SearchHit.id: Long`（内部 u64），不要再依赖旧 f32 id ABI。
- 当前只打包 `arm64-v8a`（与 pamobile 一致）；其他 ABI 需在脚本中扩展。
- i8 适合 10 万级以上的内存受限场景；f16 仍是默认。

## 主机 JNI 冒烟（无需真机）

```bash
./scripts/mobile_jni_smoke.sh
```

解包 AAR 的 `classes.jar`、编译主机 JNI 库并在主机 JVM 上验证 native 名称、
参数编组、u64 id、save/load 与显式 metric/dtype。
