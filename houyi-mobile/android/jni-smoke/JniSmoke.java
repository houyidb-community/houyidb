// 主机 JVM 的 JNI 冒烟测试：用 AAR 的 Kotlin classes + 主机编译的 JNI 库，
// 验证 native 名称、数组/字符串转换、u64、save/load 与显式 metric/dtype。
// EN: Host-JVM JNI smoke test using the AAR's Kotlin classes plus a host-built
// JNI library. Verifies native names, array/string marshalling, u64 ids,
// save/load, and explicit metric/dtype.
import cc.houyidb.mobile.HouyiDB;
import java.util.List;

public class JniSmoke {
    public static void main(String[] args) throws Exception {
        HouyiDB api = HouyiDB.INSTANCE;
        long db = api.open();
        if (db == 0) throw new AssertionError("open failed: " + api.lastError());
        api.createCollection(db, "faces", 4, "flat-clustered");
        long big = 9_007_199_254_740_993L;
        long[] ids = {big, 42L};
        float[] vecs = {1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f};
        api.insertBatch(db, "faces", ids, vecs, 2, 4);
        int built = api.build(db, "faces");
        if (built != 2) throw new AssertionError("build=" + built);
        api.setSearchBudget(db, "faces", 500);
        List<HouyiDB.SearchHit> hits = api.search(db, "faces", new float[]{1f, 0f, 0f, 0f}, 2);
        if (hits.size() != 2) throw new AssertionError("hits=" + hits.size());
        if (hits.get(0).getId() != big) throw new AssertionError("u64 id lost: " + hits.get(0).getId());
        if (!api.lastError().isEmpty()) throw new AssertionError("unexpected error: " + api.lastError());

        String path = "/tmp/houyi-mobile-jni-smoke-" + System.nanoTime() + ".bin";
        api.save(db, path);
        long loaded = api.load(path);
        if (loaded == 0) throw new AssertionError("load failed: " + api.lastError());
        if (api.count(loaded, "faces") != 2) throw new AssertionError("count after load");
        List<HouyiDB.SearchHit> again = api.search(loaded, "faces", new float[]{0f, 1f, 0f, 0f}, 1);
        if (again.size() != 1 || again.get(0).getId() != 42L) throw new AssertionError("loaded search");
        List<HouyiDB.SearchHit> bigAfter = api.search(loaded, "faces", new float[]{1f, 0f, 0f, 0f}, 1);
        if (bigAfter.size() != 1 || bigAfter.get(0).getId() != big) throw new AssertionError("u64 id lost after load");
        // 检索选项与预设：0 表示不变；high_recall 仍要返回正确结果。
        // EN: Search options and presets: 0 means unchanged; high_recall must
        // still return the correct hit.
        api.setSearchOptions(db, "faces", 0, 0, 0);
        api.setSearchPreset(db, "faces", HouyiDB.SearchPreset.HIGH_RECALL);
        List<HouyiDB.SearchHit> presetHits = api.search(db, "faces", new float[]{1f, 0f, 0f, 0f}, 1);
        if (presetHits.size() != 1 || presetHits.get(0).getId() != big) {
            throw new AssertionError("high_recall preset search");
        }
        api.setSearchPreset(db, "faces", HouyiDB.SearchPreset.BALANCED);
        api.setBuildNprobe(db, "faces", 0);

        // 内存快照：saveBytes/loadBytes 往返（Vault 加密路径），坏字节必须拒绝。
        // EN: In-memory snapshot: saveBytes/loadBytes roundtrip (the Vault
        // encryption path); corrupt bytes must be rejected.
        byte[] snapshot = api.saveBytes(db);
        if (snapshot.length == 0) throw new AssertionError("saveBytes returned empty");
        long fromBytes = api.loadBytes(snapshot);
        if (api.count(fromBytes, "faces") != 2) throw new AssertionError("count after loadBytes");
        List<HouyiDB.SearchHit> bytesHits = api.search(fromBytes, "faces", new float[]{1f, 0f, 0f, 0f}, 1);
        if (bytesHits.size() != 1 || bytesHits.get(0).getId() != big) throw new AssertionError("loadBytes search");
        api.close(fromBytes);
        byte[] corrupt = snapshot.clone();
        corrupt[0] ^= 0x7F;
        boolean rejected = false;
        try {
            api.loadBytes(corrupt);
        } catch (IllegalStateException expected) {
            rejected = true;
            if (api.lastError().isEmpty()) throw new AssertionError("corrupt snapshot error message missing");
        }
        if (!rejected) throw new AssertionError("corrupt bytes must be rejected");
        api.close(loaded);
        api.close(loaded); // 幂等：重复 close 不得崩溃
        api.close(db);
        api.close(db);

        long db2 = api.open();
        api.createCollection(db2, "l1", 4, "flat", HouyiDB.Metric.L1, HouyiDB.Dtype.F32);
        boolean conflict = false;
        try {
            api.createCollection(db2, "l1", 4, "flat", HouyiDB.Metric.L2, HouyiDB.Dtype.F32);
        } catch (IllegalStateException expected) {
            conflict = true;
            if (api.lastError().isEmpty()) throw new AssertionError("conflict error message missing");
        }
        if (!conflict) throw new AssertionError("same-name different config must be rejected");
        api.insert(db2, "l1", 7, new float[]{1f, 0f, 0f, 0f});
        List<HouyiDB.SearchHit> l1 = api.search(db2, "l1", new float[]{0f, 1f, 0f, 0f}, 1);
        if (l1.size() != 1 || l1.get(0).getId() != 7L) throw new AssertionError("L1 search");
        api.createCollection(db2, "i8", 4, "flat-clustered", HouyiDB.Metric.COSINE, HouyiDB.Dtype.I8);
        api.insert(db2, "i8", 10, new float[]{1f, 0f, 0f, 0f});
        api.insert(db2, "i8", 11, new float[]{0f, 1f, 0f, 0f});
        api.build(db2, "i8");
        List<HouyiDB.SearchHit> i8 = api.search(db2, "i8", new float[]{1f, 0f, 0f, 0f}, 2);
        if (i8.size() != 2 || i8.get(0).getId() != 10L || i8.get(1).getId() != 11L) {
            throw new AssertionError("i8 search");
        }
        api.createCollection(db2, "filtered", 4, "flat-clustered");
        api.insertBatch(db2, "filtered", new long[]{100L, 200L},
                new float[]{1f,0f,0f,0f, 0f,1f,0f,0f}, 2, 4);
        api.insertScalarsWithId(db2, "filtered", 100L, "[{\"col\":\"tag\",\"val\":\"text:a\"}]");
        api.insertScalarsWithId(db2, "filtered", 200L, "[{\"col\":\"tag\",\"val\":\"text:b\"}]");
        api.build(db2, "filtered");
        List<HouyiDB.SearchHit> filtered = api.searchHybrid(
                db2, "filtered", new float[]{1f,0f,0f,0f}, 10,
                "[{\"col\":\"tag\",\"op\":\"eq\",\"val\":\"text:a\"}]");
        if (filtered.size() != 1 || filtered.get(0).getId() != 100L) {
            throw new AssertionError("hybrid filter by id");
        }
        List<HouyiDB.SearchHit> none = api.searchHybrid(
                db2, "filtered", new float[]{1f,0f,0f,0f}, 10,
                "[{\"col\":\"tag\",\"op\":\"eq\",\"val\":\"text:zzz\"}]");
        if (none.size() != 0) throw new AssertionError("zero-hit filter must be empty");
        if (api.searchHybrid(db2, "filtered", new float[]{1f,0f,0f,0f}, 0,
                "[{\"col\":\"tag\",\"op\":\"eq\",\"val\":\"text:a\"}]").size() != 0) {
            throw new AssertionError("k=0 must be empty");
        }
        if (api.searchHybrid(db2, "filtered", new float[]{1f,0f,0f,0f}, 10, "").isEmpty()
                || api.searchHybrid(db2, "filtered", new float[]{1f,0f,0f,0f}, 10, "   ").isEmpty()) {
            throw new AssertionError("empty/whitespace filter must mean no filter");
        }
        boolean badFilter = false;
        try {
            api.searchHybrid(db2, "filtered", new float[]{1f,0f,0f,0f}, 10, "not-json");
        } catch (IllegalStateException expected) {
            badFilter = true;
            if (api.lastError().isEmpty()) throw new AssertionError("bad filter error message");
        }
        if (!badFilter) throw new AssertionError("invalid filters JSON must fail");
        boolean badTag = false;
        try {
            api.insertScalarsWithId(db2, "filtered", 100L, "[{\"col\":\"tag\",\"val\":\"int:abc\"}]");
        } catch (IllegalStateException expected) {
            badTag = true;
        }
        if (!badTag) throw new AssertionError("invalid scalar value must fail");
        api.close(db2);
        new java.io.File(path).delete();
        System.out.println("JNI smoke OK: u64 id=" + big + ", hits=" + hits.size()
                + ", AAR class loading and native calls verified");
    }
}
