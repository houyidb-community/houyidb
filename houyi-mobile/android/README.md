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
  （约 405 KB；`jni/arm64-v8a/libhouyi_mobile.so` 未压缩约 764 KB）。
- **16 KB 页对齐**：脚本传入 `-Wl,-z,max-page-size=16384`，并用
  `llvm-readelf -l` 断言所有 LOAD 段 `Align = 0x4000`（Android 15 起的
  16 KB 页设备要求；4 KB 页设备仍兼容）。构建日志会打印 `LOAD align:`。
- Gradle/AGP 首次构建可能需要联网拉取 AGP 8.5.2 / Kotlin 2.4.21。

EN: The script cross-compiles `houyi-mobile-jni` with the size-optimized mobile
profile, copies it as `libhouyi_mobile.so`, then runs Gradle
`assembleRelease`. Requirements: SDK 34, NDK >= 26, Gradle 8.9 (wrapper
included), Java 17. Output is the AAR above (~405 KB; the native library is
~764 KB uncompressed). **16 KB pages**: the script passes
`-Wl,-z,max-page-size=16384` and asserts every LOAD segment reports
`Align = 0x4000` via `llvm-readelf -l` (required on Android 15+ 16 KB-page
devices; 4 KB devices remain compatible). First Gradle builds may fetch
AGP 8.5.2 / Kotlin 2.4.21.

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

// 内存快照：不落明文临时文件，交给 Vault 加密后再持久化。
// EN: In-memory snapshot: no plaintext temp file; let a Vault encrypt it.
val bytes: ByteArray = HouyiDB.saveBytes(db)
val fromBytes = HouyiDB.loadBytes(bytes)   // 读取后即可清零 bytes

HouyiDB.close(fromBytes)
HouyiDB.close(restored)
HouyiDB.close(db)
```

- `saveBytes` / `loadBytes` 与文件 API 共用同一格式与校验（magic/版本/
  长度/CRC32）；坏字节会抛 `IllegalStateException`。C 侧对应
  `houyi_mobile_db_save_buffer` / `houyi_mobile_db_load_buffer`。
- 召回/延迟预设：`setSearchPreset(db, name, SearchPreset.HIGH_RECALL)`；
  也可用 `setSearchOptions(perK, nprobe, efSearch)` 逐项调参（0 = 不变），
  `setBuildNprobe` 控制下次 build 的 nprobe。真机数据见
  `docs/HOUYIDB_MOBILE_BENCH.md`。

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
参数编组、u64 id、file save/load、`saveBytes/loadBytes`（含坏字节拒绝）
与显式 metric/dtype。
