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

    data class SearchHit(val id: Long, val score: Float)

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

    fun load(path: String): Long {
        val handle = nativeLoad(path)
        if (handle == 0L) {
            throw IllegalStateException("HouyiDB load failed: ${lastError()}")
        }
        return handle
    }

    private fun check(code: Int) {
        if (code < 0) {
            throw IllegalStateException("HouyiDB call failed ($code): ${lastError()}")
        }
    }
}
