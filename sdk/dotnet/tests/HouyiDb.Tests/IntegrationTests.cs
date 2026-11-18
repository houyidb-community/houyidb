// houyidb .NET SDK 与真实服务的集成测试。
// EN: Integration tests for the houyidb .NET SDK against a live server.
//
// 前置：houyi-server 监听 127.0.0.1:17000（与 Python/TS/Go/Rust 联调同一实例），
// 可用 HOUYIDB_URL 覆盖；服务不可达时各用例静默跳过（xUnit 无内建动态 Skip，
// 与其他 SDK 的跳过语义保持一致）。
// EN: Prerequisite: houyi-server listens on 127.0.0.1:17000 (shared with the
// Python/TS/Go/Rust SDKs); override with HOUYIDB_URL; when unreachable, each test
// silently skips (xUnit has no built-in dynamic Skip, keeping skip semantics
// consistent with the other SDKs).
// 运行：dotnet test
// EN: Run with: dotnet test
//
// 并行安全：xUnit 在同一测试类内串行执行，各用例仍独占 collection 名
// （WAL 持久化 + 同 id upsert 幂等，重跑安全）。检索类断言前一律 WaitFlushed。
// EN: Parallel safety: xUnit runs tests serially within one class, and each test
// still owns its collection name (WAL persistence plus idempotent upsert by id
// make reruns safe). Search assertions always call WaitFlushed first.
using System.Text.Json;
using HouyiDb;
using Konscious.Security.Cryptography;
using Xunit;

namespace HouyiDb.Tests;

public class IntegrationTests
{
    private const int TestDims = 16;

    private static string BaseUrl =>
        (Environment.GetEnvironmentVariable("HOUYIDB_URL") ?? "http://127.0.0.1:17000").TrimEnd('/');

    /// <summary>
    /// 确定性伪 embedding（与 Python/TS/Go/Rust 联调同款思路，自洽即可）。
    /// EN: Deterministic pseudo-embedding (same idea as the Python/TS/Go/Rust
    /// SDKs; it merely needs to be self-consistent).
    /// </summary>
    private static float[] FakeEmbed(string text)
    {
        var output = new float[TestDims];
        for (var counter = 0; counter < TestDims; counter++)
        {
            using var hasher = new HMACBlake2B(64);
            var digest = hasher.ComputeHash(
                System.Text.Encoding.UTF8.GetBytes($"{text}:{counter}"));
            var v = System.Buffers.Binary.BinaryPrimitives.ReadUInt64BigEndian(digest);
            output[counter] = (float)(v / 18446744073709551616.0 * 2.0 - 1.0);
        }
        return output;
    }

    /// <summary>
    /// 服务不可达时返回 null（用例跳过）。
    /// EN: Returns null when the service is unreachable (the test skips).
    /// </summary>
    private static async Task<HouyiClient.Collection?> ConnectAsync(string name)
    {
        var client = new HouyiClient(BaseUrl, TimeSpan.FromSeconds(3));
        if (!await client.HealthAsync())
        {
            return null; // houyi-server 不可达：--benchmark-http-addr 127.0.0.1:17000。 EN: houyi-server unreachable: --benchmark-http-addr 127.0.0.1:17000
        }
        return client.GetCollection(name);
    }

    /// <summary>
    /// 等待异步构图队列排空（双速写入：flush 后才对检索可见）。
    /// EN: Waits for the async graph-build queue to drain (dual-speed writes are
    /// visible to search only after flush).
    /// </summary>
    private static async Task WaitFlushedAsync(HouyiClient.Collection col)
    {
        var deadline = DateTime.UtcNow.AddSeconds(10);
        while (DateTime.UtcNow < deadline)
        {
            using var status = await col.StatusAsync();
            if (status.RootElement.TryGetProperty("pending_count", out var p) && p.GetInt64() == 0)
            {
                return;
            }
            await Task.Delay(20);
        }
        throw new TimeoutException($"collection {col.Name} flush timeout");
    }

    private static Vector Doc(string idText, string text, string category, int price) => new()
    {
        Id = StableId.Of(idText),
        Values = FakeEmbed(text),
        Payload = new Dictionary<string, object?>
        {
            ["text"] = text,
            ["category"] = category,
            ["price"] = price,
        },
    };

    [Fact]
    public async Task UpsertSearchRoundTrip()
    {
        var col = await ConnectAsync("dotnet_it");
        if (col is null) return;

        var inserted = await col.UpsertAsync(new[]
        {
            Doc("dotnet-doc-0", "apple is a fruit", "fruit", 5),
            Doc("dotnet-doc-1", "banana is yellow", "fruit", 3),
            Doc("dotnet-doc-2", "laptop is electronics", "electronics", 999),
        });
        Assert.Equal(3, inserted);
        await WaitFlushedAsync(col);

        var result = await col.SearchAsync(
            FakeEmbed("apple is a fruit"), 2, new SearchOptions { IncludePayload = true });
        Assert.Equal(2, result.Hits.Count);
        Assert.Equal("apple is a fruit", result.Hits[0].Payload!["text"].GetString());
        Assert.True(result.Hits[0].Score >= result.Hits[1].Score, "scores must be descending");
    }

    [Fact]
    public async Task HybridSearchFilters()
    {
        var col = await ConnectAsync("dotnet_it_hybrid");
        if (col is null) return;

        await col.UpsertAsync(new[]
        {
            Doc("dotnet-hybrid-0", "apple is a fruit", "fruit", 5),
            Doc("dotnet-hybrid-1", "banana is yellow", "fruit", 3),
            Doc("dotnet-hybrid-2", "laptop is electronics", "electronics", 999),
        });
        await WaitFlushedAsync(col);

        var options = new SearchOptions { IncludePayload = true };
        options.Filters.Add(new Filter("category", FilterOp.Eq, "fruit"));
        options.Filters.Add(new Filter("price", FilterOp.Lt, 4));
        var result = await col.HybridSearchAsync(FakeEmbed("apple is a fruit"), 3, options);
        var hit = Assert.Single(result.Hits);
        Assert.Equal("banana is yellow", hit.Payload!["text"].GetString());
    }

    /// <summary>
    /// 异键集 payload：PAX 新列回填的端到端守卫（曾在此场景整列错位）。
    /// EN: Disjoint payload key sets: end-to-end guard for PAX new-column backfill
    /// (a whole column once got misaligned in this scenario).
    /// </summary>
    [Fact]
    public async Task MixedPayloadKeySets()
    {
        var col = await ConnectAsync("dotnet_it_mixed_keys");
        if (col is null) return;

        await col.UpsertAsync(new[]
        {
            new Vector
            {
                Id = 1,
                Values = FakeEmbed("row-a"),
                Payload = new Dictionary<string, object?> { ["a"] = 1 },
            },
            new Vector
            {
                Id = 2,
                Values = FakeEmbed("row-b"),
                Payload = new Dictionary<string, object?> { ["b"] = 2 },
            },
        });
        // 过滤按标量页同步求值，但向量侧仍需构图完成才能命中。
        // EN: Filters are evaluated synchronously on scalar pages, but the vector
        // side still needs the graph build to finish before it can match.
        await WaitFlushedAsync(col);

        var options = new SearchOptions { IncludePayload = true };
        options.Filters.Add(new Filter("b", FilterOp.Eq, 2));
        var result = await col.HybridSearchAsync(FakeEmbed("row-b"), 5, options);
        var hit = Assert.Single(result.Hits);
        Assert.Equal(2UL, hit.Id);
        Assert.False(hit.Payload!.ContainsKey("a"), "payload must not leak column 'a'");
    }

    [Fact]
    public async Task LargeUInt64IdRoundTrip()
    {
        var col = await ConnectAsync("dotnet_it_bigid");
        if (col is null) return;

        // StableId 落点常态是 19 位十进制（>2^53）；.NET ulong 原生无损。
        // EN: StableId values are normally 19-digit decimals (>2^53); .NET ulong is
        // lossless natively.
        var id = StableId.Of("hello");
        await col.UpsertAsync(new[]
        {
            new Vector
            {
                Id = id,
                Values = FakeEmbed("uint64 probe"),
                Payload = new Dictionary<string, object?> { ["tag"] = "big" },
            },
        });
        await WaitFlushedAsync(col);

        var options = new SearchOptions { IncludePayload = true };
        options.Filters.Add(new Filter("tag", FilterOp.Eq, "big"));
        var result = await col.HybridSearchAsync(FakeEmbed("uint64 probe"), 1, options);
        Assert.NotEmpty(result.Hits);
        Assert.Equal(id, result.Hits[0].Id);

        Assert.True(await col.DeleteAsync(ids: new[] { id }) <= 1);
    }

    [Fact]
    public async Task DeleteByIdsAndFilters()
    {
        var col = await ConnectAsync("dotnet_it_delete");
        if (col is null) return;

        var probeId = StableId.Of("dotnet-delete-me");
        await col.UpsertAsync(new[]
        {
            new Vector
            {
                Id = probeId,
                Values = FakeEmbed("delete probe"),
                Payload = new Dictionary<string, object?> { ["tag"] = "del" },
            },
        });
        // delete-by-filter 不依赖构图排空（标量页同步落页）。
        // EN: delete-by-filter does not depend on the graph-build drain (scalar
        // pages land synchronously).
        Assert.Equal(1, await col.DeleteAsync(filters: new[] { new Filter("tag", FilterOp.Eq, "del") }));
        Assert.Equal(0, await col.DeleteAsync(ids: new[] { probeId })); // 幂等。 EN: idempotent
    }

    [Fact]
    public async Task ServerErrorSurfacesAsHouyiException()
    {
        var col = await ConnectAsync("dotnet_it_missing_collection");
        if (col is null) return;

        var ex = await Assert.ThrowsAsync<HouyiException>(() => col.StatusAsync());
        Assert.Equal(404, ex.StatusCode);
    }

    [Fact]
    public async Task DeleteRequiresIdsOrFilters()
    {
        var col = await ConnectAsync("dotnet_it");
        if (col is null) return;

        await Assert.ThrowsAsync<ArgumentException>(() => col.DeleteAsync());
    }

    /// <summary>
    /// collection 名含保留字符时路径转义（与 Python quote(safe="") 同口径）。
    /// EN: Path escaping when a collection name contains reserved characters
    /// (same rule as Python quote(safe="")).
    /// </summary>
    [Fact]
    public async Task CollectionNameWithReservedChars()
    {
        var probe = await ConnectAsync("dotnet_it");
        if (probe is null) return;

        var col = new HouyiClient(BaseUrl).GetCollection("dotnet 名/a?b#c");
        await col.UpsertAsync(new[]
        {
            new Vector
            {
                Id = 42,
                Values = FakeEmbed("quoted"),
                Payload = new Dictionary<string, object?> { ["k"] = "v" },
            },
        });
        Assert.Equal(1, await col.DeleteAsync(ids: new[] { 42UL }));
    }

    /// <summary>
    /// 删除整库后 404；同名重建得到空 collection。
    /// EN: Drops a whole collection (404 afterwards) and expects a same-name
    /// recreate to start empty.
    /// </summary>
    [Fact]
    public async Task DeleteCollectionAndRecreate()
    {
        var client = new HouyiClient(BaseUrl, TimeSpan.FromSeconds(3));
        if (!await client.HealthAsync())
        {
            return;
        }
        const string name = "dotnet_it_admin_drop";
        var col = client.GetCollection(name);
        await col.UpsertAsync(new[]
        {
            new Vector
            {
                Id = 7,
                Values = FakeEmbed("drop me"),
                Payload = new Dictionary<string, object?> { ["tag"] = "drop" },
            },
        });

        await client.DeleteCollectionAsync(name);

        var ex = await Assert.ThrowsAsync<HouyiException>(() => col.StatusAsync());
        Assert.Equal(404, ex.StatusCode);

        // 同名重建：首次写入隐式创建，旧向量不得复活。
        // EN: recreate by name: the first write implicitly creates it and the old
        // vectors must not reappear.
        await col.UpsertAsync(new[] { new Vector { Id = 8, Values = FakeEmbed("fresh") } });
        await WaitFlushedAsync(col);
        using var status = await col.StatusAsync();
        Assert.Equal(1, status.RootElement.GetProperty("indexed_count").GetInt64());
    }

    /// <summary>
    /// 显式建 f16 + int8 hnsw 集合：vector_dtype/quantization 透传到请求、
    /// 响应回显、status 可观测。
    /// EN: Explicitly creates an f16 + int8 hnsw collection: vector_dtype and
    /// quantization are forwarded in the request, echoed in the response, and
    /// observable via status.
    /// </summary>
    [Fact]
    public async Task CreateCollectionF16Int8RoundTrip()
    {
        var client = new HouyiClient(BaseUrl, TimeSpan.FromSeconds(3));
        if (!await client.HealthAsync())
        {
            return;
        }
        const string name = "dotnet_it_f16_dtype";
        // 幂等清理：上次异常退出可能残留同名集合。
        // EN: idempotent cleanup: a previous aborted run may have left the name behind.
        try
        {
            await client.DeleteCollectionAsync(name);
        }
        catch (HouyiException)
        {
            // 不存在（404）或启用了鉴权（401/403）：无需清理，直接继续。
            // EN: missing (404) or auth enabled (401/403): nothing to clean up.
        }

        var created = await client.CreateCollectionAsync(name, TestDims, new CreateCollectionOptions
        {
            Metric = "l2",
            IndexType = "hnsw",
            VectorDtype = "f16",
            Quantization = "int8",
            Hnsw = new HnswParams(16, 100, 32),
        });
        Assert.Equal("f16", created.VectorDtype);
        Assert.Equal("int8", created.Quantization);
        Assert.Equal("hnsw", created.IndexType);

        using var status = await client.GetCollection(name).StatusAsync();
        Assert.Equal("f16", status.RootElement.GetProperty("vector_dtype").GetString());
        Assert.Equal(
            "int8", status.RootElement.GetProperty("quantization").GetString());

        await client.DeleteCollectionAsync(name);
    }

    /// <summary>
    /// 显式建 rabitq（l2/cosine-only）hnsw 集合：请求透传、响应与 status 回显；
    /// ip 组合返回 400。
    /// EN: Explicitly creates a rabitq (l2/cosine-only) hnsw collection: the
    /// request is forwarded, the response and status echo it, and the ip
    /// combination returns 400.
    /// </summary>
    [Fact]
    public async Task CreateCollectionRabitqRoundTrip()
    {
        var client = new HouyiClient(BaseUrl, TimeSpan.FromSeconds(3));
        if (!await client.HealthAsync())
        {
            return;
        }
        const string name = "dotnet_it_rabitq";
        try
        {
            await client.DeleteCollectionAsync(name);
        }
        catch (HouyiException)
        {
            // 不存在（404）或启用了鉴权（401/403）：无需清理。
            // EN: missing (404) or auth enabled (401/403): nothing to clean up.
        }

        var created = await client.CreateCollectionAsync(name, TestDims, new CreateCollectionOptions
        {
            Metric = "cosine",
            IndexType = "hnsw",
            VectorDtype = "f32",
            Quantization = "rabitq",
            Hnsw = new HnswParams(16, 100, 32),
        });
        Assert.Equal("cosine", created.Metric);
        Assert.Equal("hnsw", created.IndexType);
        Assert.Equal("rabitq", created.Quantization);

        using var status = await client.GetCollection(name).StatusAsync();
        Assert.Equal(
            "rabitq", status.RootElement.GetProperty("quantization").GetString());

        var ex = await Assert.ThrowsAsync<HouyiException>(
            () => client.CreateCollectionAsync(name + "_ip", TestDims, new CreateCollectionOptions
            {
                Metric = "ip",
                IndexType = "hnsw",
                Quantization = "rabitq",
            }));
        Assert.Equal(400, ex.StatusCode);

        await client.DeleteCollectionAsync(name);
    }

    /// <summary>
    /// 删除不存在的 collection 返回上游 404。
    /// EN: Dropping an unknown collection surfaces the upstream 404.
    /// </summary>
    [Fact]
    public async Task DeleteMissingCollectionIs404()
    {
        var client = new HouyiClient(BaseUrl, TimeSpan.FromSeconds(3));
        if (!await client.HealthAsync())
        {
            return;
        }
        var ex = await Assert.ThrowsAsync<HouyiException>(
            () => client.DeleteCollectionAsync("dotnet_it_admin_never_created"));
        Assert.Equal(404, ex.StatusCode);
    }

    /// <summary>
    /// bm25 建集合 → 带稀疏 upsert → 融合检索（RRF/加权回显）→ sparse=[] 清空
    /// 后仅剩稠密分。
    /// EN: Creates a bm25 collection, upserts with sparse terms, runs a fusion
    /// search (RRF/weighted echoes), and verifies that clearing with sparse=[]
    /// leaves the document with a dense score only.
    /// </summary>
    [Fact]
    public async Task SparseCreateAndFusionSearch()
    {
        var client = new HouyiClient(BaseUrl, TimeSpan.FromSeconds(3));
        if (!await client.HealthAsync())
        {
            return;
        }
        const string name = "dotnet_it_fusion";
        try
        {
            await client.DeleteCollectionAsync(name);
        }
        catch (HouyiException)
        {
            // 不存在（404）或启用了鉴权（401/403）：无需清理。
            // EN: missing (404) or auth enabled (401/403): nothing to clean up.
        }

        var created = await client.CreateCollectionAsync(name, TestDims, new CreateCollectionOptions
        {
            IndexType = "flat",
            Sparse = "bm25",
        });
        Assert.Equal("bm25", created.Sparse);
        Assert.Equal("flat", created.IndexType);

        var col = client.GetCollection(name);
        await col.UpsertAsync(new[]
        {
            new Vector
            {
                Id = 1,
                Values = FakeEmbed("apple"),
                Payload = new Dictionary<string, object?> { ["text"] = "apple" },
                Sparse = [new SparseTerm(7, 1.0f), new SparseTerm(8, 0.5f)],
            },
            new Vector
            {
                Id = 2,
                Values = FakeEmbed("banana"),
                Payload = new Dictionary<string, object?> { ["text"] = "banana" },
                Sparse = [new SparseTerm(8, 1.0f)],
            },
            new Vector
            {
                Id = 3,
                Values = FakeEmbed("cherry"),
                Payload = new Dictionary<string, object?> { ["text"] = "cherry" },
                Sparse = [new SparseTerm(9, 1.0f)],
            },
        });
        await WaitFlushedAsync(col);

        var result = await col.FusionSearchAsync(
            FakeEmbed("apple"),
            [new SparseTerm(7, 1.0f), new SparseTerm(8, 0.5f)],
            2,
            new FusionOptions { IncludePayload = true });
        Assert.NotEmpty(result.Hits);
        Assert.Equal(1UL, result.Hits[0].Id);
        Assert.Equal("rrf", result.FusionMode);
        Assert.Equal(60u, result.RrfK);
        Assert.Null(result.Alpha);
        Assert.NotNull(result.Hits[0].DenseScore);
        Assert.NotNull(result.Hits[0].SparseScore);
        Assert.Equal("apple", result.Hits[0].Payload!["text"].GetString());
        Assert.Equal("cosine", result.Metric);

        var weighted = await col.FusionSearchAsync(
            FakeEmbed("apple"), [new SparseTerm(7, 1.0f)], 2,
            new FusionOptions { Fusion = "weighted", Alpha = 0.25f });
        Assert.Equal("weighted", weighted.FusionMode);
        Assert.Equal(0.25f, weighted.Alpha);
        Assert.Null(weighted.RrfK);

        // sparse=[] 清空该文档词项（null 才是「不改」）；清空后仅剩稠密分。
        // EN: sparse=[] clears this document's terms (null means "unchanged");
        // afterwards only the dense score remains.
        await col.UpsertAsync(new[]
        {
            new Vector
            {
                Id = 1,
                Values = FakeEmbed("apple"),
                Payload = new Dictionary<string, object?> { ["text"] = "apple" },
                Sparse = [],
            },
        });
        await WaitFlushedAsync(col);
        var cleared = await col.FusionSearchAsync(
            FakeEmbed("apple"), [new SparseTerm(8, 1.0f)], 3);
        var cleared1 = cleared.Hits.SingleOrDefault(h => h.Id == 1);
        Assert.NotNull(cleared1);
        Assert.NotNull(cleared1.DenseScore);
        Assert.Null(cleared1.SparseScore);

        await client.DeleteCollectionAsync(name);
    }

    /// <summary>
    /// bm25 仅 hnsw/flat（clustered 组合 400）；隐式创建的 none 集合上融合检索 400。
    /// EN: bm25 is hnsw/flat-only (a clustered combo is 400); fusion search on an
    /// implicitly created none collection is 400.
    /// </summary>
    [Fact]
    public async Task SparseCreateGuards()
    {
        var client = new HouyiClient(BaseUrl, TimeSpan.FromSeconds(3));
        if (!await client.HealthAsync())
        {
            return;
        }
        const string badName = "dotnet_it_sparse_clustered";
        const string implicitName = "dotnet_it_sparse_implicit";
        foreach (var name in new[] { badName, implicitName })
        {
            try
            {
                await client.DeleteCollectionAsync(name);
            }
            catch (HouyiException)
            {
                // 不存在（404）或启用了鉴权（401/403）：无需清理。
                // EN: missing (404) or auth enabled (401/403): nothing to clean up.
            }
        }

        var bad = await Assert.ThrowsAsync<HouyiException>(
            () => client.CreateCollectionAsync(badName, TestDims, new CreateCollectionOptions
            {
                IndexType = "clustered",
                Sparse = "bm25",
            }));
        Assert.Equal(400, bad.StatusCode);

        // 隐式创建恒为 none：融合检索被拒（400）。
        // EN: implicit creation is always none, so fusion search is rejected (400).
        await client.GetCollection(implicitName)
            .UpsertAsync(new[] { new Vector { Id = 1, Values = FakeEmbed("implicit none") } });
        var fusionErr = await Assert.ThrowsAsync<HouyiException>(
            () => client.GetCollection(implicitName).FusionSearchAsync(
                FakeEmbed("implicit none"), [new SparseTerm(1, 1.0f)], 1));
        Assert.Equal(400, fusionErr.StatusCode);

        await client.DeleteCollectionAsync(implicitName);
    }
}
