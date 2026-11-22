package cc.houyidb.mobile

/**
 * HouyiDB Mobile 的 Android 门面：持有 native 句柄，所有调用失败抛
 * IllegalStateException 并附带 native lastError。
 * EN: Android facade for HouyiDB Mobile; native failures throw
 * IllegalStateException with the native last-error message.
 */
object HouyiDB {
    init {
        System.loadLibrary("houyi_mobile")
    }

    enum class Metric(val code: Int) {
        L2(0), COSINE(1), IP(2), L1(3)
    }

    enum class Dtype(val code: Int) {
        F32(0), F16(1), I8(2)
    }

    /** 召回/延迟预设：FAST 低延迟、BALANCED 默认、HIGH_RECALL 高召回。 */
    enum class SearchPreset(val code: Int) {
        FAST(0), BALANCED(1), HIGH_RECALL(2)
    }

    data class SearchHit(val id: Long, val score: Float)

    /** 集合诊断信息（由 native 填写，estimatedBytes 为堆内存粗估）。 */
    data class CollectionStats(
        val count: Int,
        val dims: Int,
        val indexType: String,
        val metric: Metric,
        val dtype: Dtype,
        val built: Boolean,
        val nprobe: Int,
        val numClusters: Int,
        val estimatedBytes: Long,
    )

    @JvmStatic external fun nativeOpen(): Long
    @JvmStatic external fun nativeClose(handle: Long)
    @JvmStatic external fun nativeCreateCollectionLegacy(
        handle: Long, name: String, dims: Int, indexType: String
    ): Int
    @JvmStatic external fun nativeCreateCollection(
        handle: Long, name: String, dims: Int, indexType: String,
        metric: Int, dtype: Int
    ): Int
    @JvmStatic external fun nativeInsert(
        handle: Long, name: String, id: Long, vector: FloatArray
    ): Int
    @JvmStatic external fun nativeInsertBatch(
        handle: Long, name: String, ids: LongArray, vectors: FloatArray,
        count: Int, dims: Int
    ): Int
    @JvmStatic external fun nativeSetSearchTarget(
        handle: Long, name: String, perK: Int
    ): Int
    @JvmStatic external fun nativeSetSearchOptions(
        handle: Long, name: String, perK: Int, nprobe: Int, efSearch: Int
    ): Int
    @JvmStatic external fun nativeSetSearchPreset(
        handle: Long, name: String, preset: Int
    ): Int
    @JvmStatic external fun nativeSetBuildNprobe(
        handle: Long, name: String, nprobe: Int
    ): Int
    @JvmStatic external fun nativeSetThreads(threads: Int): Int
    @JvmStatic external fun nativeCollectionStats(
        handle: Long, name: String, out: LongArray
    ): Int
    @JvmStatic external fun nativeMemoryUsage(handle: Long): Long
    @JvmStatic external fun nativeSearchBatch(
        handle: Long, name: String, queries: FloatArray, nq: Int, dims: Int, k: Int,
        outIds: LongArray, outScores: FloatArray, outCounts: IntArray
    ): Int
    @JvmStatic external fun nativeInsertScalarsWithId(
        handle: Long, name: String, id: Long, tagsJson: String
    ): Int
    @JvmStatic external fun nativeSearchHybridU64(
        handle: Long, name: String, query: FloatArray, k: Int, filtersJson: String,
        outIds: LongArray, outScores: FloatArray
    ): Int
    @JvmStatic external fun nativeBuild(handle: Long, name: String): Int
    @JvmStatic external fun nativeCount(handle: Long, name: String): Int
    @JvmStatic external fun nativeSearch(
        handle: Long, name: String, query: FloatArray, k: Int,
        outIds: LongArray, outScores: FloatArray
    ): Int
    @JvmStatic external fun nativeSave(handle: Long, path: String): Int
    @JvmStatic external fun nativeLoad(path: String): Long
    @JvmStatic external fun nativeLoadView(path: String): Long
    @JvmStatic external fun nativeSaveBytes(handle: Long): ByteArray?
    @JvmStatic external fun nativeLoadBytes(bytes: ByteArray): Long
    @JvmStatic external fun nativeLastError(): String

    fun open(): Long = nativeOpen()

    fun close(handle: Long) = nativeClose(handle)

    fun lastError(): String = nativeLastError()

    /** 历史默认配置：flat=L2/f32、flat-clustered=Cosine/f16、clustered-hnsw=Cosine/f32。 */
    fun createCollection(handle: Long, name: String, dims: Int, indexType: String) {
        require(name.isNotEmpty() && name.length <= 4096) { "collection name must be 1..4096 chars" }
        require(dims > 0) { "dims must be > 0" }
        check(nativeCreateCollectionLegacy(handle, name, dims, indexType))
    }

    fun createCollection(
        handle: Long,
        name: String,
        dims: Int,
        indexType: String,
        metric: Metric,
        dtype: Dtype,
    ) {
        require(name.isNotEmpty() && name.length <= 4096) { "collection name must be 1..4096 chars" }
        require(dims > 0) { "dims must be > 0" }
        check(nativeCreateCollection(handle, name, dims, indexType, metric.code, dtype.code))
    }

    fun insert(handle: Long, name: String, id: Long, vector: FloatArray) {
        check(nativeInsert(handle, name, id, vector))
    }

    /** 批量写入；count 行，vectors 长度为 count*dims。 */
    fun insertBatch(
        handle: Long,
        name: String,
        ids: LongArray,
        vectors: FloatArray,
        count: Int,
        dims: Int,
    ) {
        require(count >= 0) { "count must be >= 0" }
        require(ids.size >= count) { "ids shorter than count" }
        require(dims > 0 && vectors.size == count * dims) { "vectors must be count*dims" }
        check(nativeInsertBatch(handle, name, ids, vectors, count, dims))
    }

    /** 以显式向量 id 写入标量标签（tagsJson 同 C ABI）。 */
    fun insertScalarsWithId(handle: Long, name: String, id: Long, tagsJson: String) {
        require(name.isNotEmpty()) { "collection name must not be empty" }
        check(nativeInsertScalarsWithId(handle, name, id, tagsJson))
    }

    /**
     * 混合检索：向量 + 过滤；仅 flat-clustered 在候选扫描内下推，
     * flat/clustered-hnsw 为后过滤（返回值可能不足 k）。filtersJson 同 C ABI。
     * EN: Hybrid search; filter pushdown only for flat-clustered; flat/
     * clustered-hnsw post-filter (result count may be below k).
     */
    fun searchHybrid(
        handle: Long,
        name: String,
        query: FloatArray,
        k: Int,
        filtersJson: String,
    ): List<SearchHit> {
        require(k >= 0) { "k must be >= 0" }
        val ids = LongArray(k)
        val scores = FloatArray(k)
        val found = nativeSearchHybridU64(handle, name, query, k, filtersJson, ids, scores)
        check(found)
        return (0 until found).map { SearchHit(ids[it], scores[it]) }
    }

    /** 候选预算（每 k）；默认 2500，人脸场景可试 500–1000。 */
    fun setSearchBudget(handle: Long, name: String, perK: Int) {
        require(perK > 0) { "perK must be > 0" }
        check(nativeSetSearchTarget(handle, name, perK))
    }

    /**
     * 检索选项：perK 候选预算、nprobe 路由簇数、efSearch（仅 clustered-hnsw）。
     * 传 0 表示该项保持不变。
     * EN: Search options: perK budget, nprobe routed clusters, efSearch
     * (clustered-hnsw only). Pass 0 to leave a field unchanged.
     */
    fun setSearchOptions(
        handle: Long,
        name: String,
        perK: Int = 0,
        nprobe: Int = 0,
        efSearch: Int = 0,
    ) {
        require(perK >= 0 && nprobe >= 0 && efSearch >= 0) { "options must be >= 0" }
        check(nativeSetSearchOptions(handle, name, perK, nprobe, efSearch))
    }

    /** 应用 fast / balanced / high_recall 预设。 */
    fun setSearchPreset(handle: Long, name: String, preset: SearchPreset) {
        check(nativeSetSearchPreset(handle, name, preset.code))
    }

    /**
     * 下一次 build 使用的 nprobe（0 = 自适应）；已建图需重建才生效，
     * 运行期可用 setSearchOptions 的 nprobe 直接抬高路由簇数。
     * EN: nprobe for the next build (0 = adaptive); a built index needs a
     * rebuild, or raise routing immediately via setSearchOptions.
     */
    fun setBuildNprobe(handle: Long, name: String, nprobe: Int) {
        require(nprobe >= 0) { "nprobe must be >= 0" }
        check(nativeSetBuildNprobe(handle, name, nprobe))
    }

    /**
     * 集合诊断：数量/维度/索引类型/度量/dtype/是否建图/nprobe/簇数/内存估算。
     * EN: Collection diagnostics: count/dims/index type/metric/dtype/built/
     * nprobe/cluster count/memory estimate.
     */
    fun collectionStats(handle: Long, name: String): CollectionStats {
        val out = LongArray(9)
        check(nativeCollectionStats(handle, name, out))
        val indexType = when (out[2].toInt()) {
            0 -> "flat"
            1 -> "flat-clustered"
            else -> "clustered-hnsw"
        }
        return CollectionStats(
            count = out[0].toInt(),
            dims = out[1].toInt(),
            indexType = indexType,
            metric = Metric.entries.firstOrNull { it.code == out[3].toInt() } ?: Metric.COSINE,
            dtype = Dtype.entries.firstOrNull { it.code == out[4].toInt() } ?: Dtype.F32,
            built = out[5] != 0L,
            nprobe = out[6].toInt(),
            numClusters = out[7].toInt(),
            estimatedBytes = out[8],
        )
    }

    /**
     * 初始化检索线程数（须在首次检索/建图前调用；全局线程池只能设置一次）。
     * 真机 128d×100k：2 线程明显慢，4/6/8 查询差 10–15%，建图 8 最快。
     * EN: Initialize the search thread count (before the first search/build;
     * the global pool can only be set once). On-device 128d×100k: 2 threads are
     * clearly slower, 4/6/8 differ by 10–15% on queries, builds are fastest
     * with 8.
     */
    fun setThreads(threads: Int) {
        require(threads in 1..256) { "threads must be 1..256" }
        check(nativeSetThreads(threads))
    }

    /** 全库堆内存估算（字节）。 */
    fun memoryUsage(handle: Long): Long = nativeMemoryUsage(handle)

    /**
     * 批量检索：一次调用处理 nq 条查询（每条 dims 个 float），返回与查询
     * 对齐的结果列表。适合相册一次匹配多张脸。
     * EN: Batch search over nq queries (dims floats each), returning one list
     * per query; suited to matching several album faces at once.
     */
    fun searchBatch(
        handle: Long,
        name: String,
        queries: FloatArray,
        nq: Int,
        dims: Int,
        k: Int,
    ): List<List<SearchHit>> {
        require(nq >= 0 && dims > 0 && k >= 0) { "invalid batch arguments" }
        require(queries.size == nq * dims) { "queries must be nq*dims" }
        val ids = LongArray(nq * k)
        val scores = FloatArray(nq * k)
        val counts = IntArray(nq)
        check(nativeSearchBatch(handle, name, queries, nq, dims, k, ids, scores, counts))
        return (0 until nq).map { i ->
            (0 until counts[i]).map { j ->
                val idx = i * k + j
                SearchHit(ids[idx], scores[idx])
            }
        }
    }

    fun build(handle: Long, name: String): Int {
        val count = nativeBuild(handle, name)
        check(count)
        return count
    }

    fun count(handle: Long, name: String): Int {
        val count = nativeCount(handle, name)
        check(count)
        return count
    }

    /** k 结果；native 填充 id/score 两个数组，返回实际条数。 */
    fun search(
        handle: Long,
        name: String,
        query: FloatArray,
        k: Int,
    ): List<SearchHit> {
        require(k >= 0) { "k must be >= 0" }
        val ids = LongArray(k)
        val scores = FloatArray(k)
        val found = nativeSearch(handle, name, query, k, ids, scores)
        check(found)
        return (0 until found).map { SearchHit(ids[it], scores[it]) }
    }

    fun save(handle: Long, path: String) {
        check(nativeSave(handle, path))
    }

    /**
     * 从文件加载。内部使用只读 mmap：加载期间不得原地改写/截断该文件
     * （「临时文件 + rename」原子替换是安全的；宿主必须原地覆写时请改用 loadBytes）。
     * EN: Load from a file. Uses a read-only mmap internally: do not overwrite or
     * truncate the file in place during loading (atomic temp-file + rename is
     * safe; when an in-place host overwrite is required, use loadBytes).
     */
    fun load(path: String): Long {
        val handle = nativeLoad(path)
        if (handle == 0L) {
            throw IllegalStateException("HouyiDB load failed: ${lastError()}")
        }
        return handle
    }

    /**
     * 以 view 模式从文件加载：只读 mmap + 元数据解析，簇向量首次检索时按簇解码。
     *
     * 不校验整文件 CRC（结构校验完整）；加载后**不得原地截断或改写该文件**
     * （「临时文件 + rename」是安全的）。首次写入只解码触达的簇及其范数，其余簇
     * 保持惰性；未写入时保存直接复用原始构建态字节，写入后重新编码。
     * EN: Load from a file in view mode: read-only mmap plus metadata parsing;
     * cluster vectors decode per cluster on first search. The whole-file CRC is
     * skipped (structural validation remains complete); do not truncate or
     * overwrite the file in place after loading (temp file + rename is safe).
     * The first write materializes the clusters; afterwards behavior matches a
     * normal load.
     */
    fun loadView(path: String): Long {
        val handle = nativeLoadView(path)
        if (handle == 0L) {
            throw IllegalStateException("HouyiDB loadView failed: ${lastError()}")
        }
        return handle
    }

    /**
     * 把整个数据库编码成内存字节：不落临时文件，便于交给 Vault 加密后再持久化。
     * EN: Encode the whole database into memory bytes: no temp file, so the
     * caller can hand them to a Vault for encryption before persisting.
     */
    fun saveBytes(handle: Long): ByteArray {
        return nativeSaveBytes(handle)
            ?: throw IllegalStateException("HouyiDB saveBytes failed: ${lastError()}")
    }

    /**
     * 从内存字节恢复数据库；字节只在 native 调用期间被读取，之后可立即清零。
     * EN: Restore a database from memory bytes; the bytes are read only during
     * the native call and may be wiped immediately afterwards.
     */
    fun loadBytes(bytes: ByteArray): Long {
        require(bytes.isNotEmpty()) { "bytes must not be empty" }
        val handle = nativeLoadBytes(bytes)
        if (handle == 0L) {
            throw IllegalStateException("HouyiDB loadBytes failed: ${lastError()}")
        }
        return handle
    }

    private fun check(code: Int) {
        if (code < 0) {
            throw IllegalStateException("HouyiDB call failed ($code): ${lastError()}")
        }
    }
}
