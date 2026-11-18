// P4 新增面的 wire 契约测试（自定义 HttpMessageHandler，无真实服务端）：
// 逐向量 sparse 键存在性语义、创建 sparse 选项与旧服务端缺席容忍、
// 融合检索路由/请求体/响应平行分数组与 fusion 参数块。
//
// EN: Wire contract tests for the P4 additions (custom HttpMessageHandler, no
// real server): per-vector sparse key-presence semantics, the create sparse
// option plus tolerance for older servers omitting it, and the fusion
// route/request body/response parallel score arrays plus the fusion params
// block.
using System.Net;
using System.Text;
using System.Text.Json;
using HouyiDb;
using Xunit;

namespace HouyiDb.Tests;

public class WireContractTests
{
    /// <summary>
    /// 回放固定响应并记录最后一次请求。
    /// EN: Replays a fixed response and records the last request.
    /// </summary>
    private sealed class CaptureHandler : HttpMessageHandler
    {
        private readonly string _response;

        public CaptureHandler(string response) => _response = response;

        public string? Method { get; private set; }

        public string? Path { get; private set; }

        public string? Body { get; private set; }

        protected override async Task<HttpResponseMessage> SendAsync(
            HttpRequestMessage request, CancellationToken cancellationToken)
        {
            Method = request.Method.Method;
            Path = request.RequestUri!.AbsolutePath;
            Body = request.Content is null
                ? null
                : await request.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(_response, Encoding.UTF8, "application/json"),
            };
        }
    }

    /// <summary>
    /// 携带 HttpClient 的测试夹具：由测试的 `using` 负责释放连接池，
    /// 避免每次构造都泄漏一个 HttpClient 实例。
    /// EN: Test fixture that owns the HttpClient so the test's `using`
    /// releases the connection pool instead of leaking one HttpClient
    /// instance per construction.
    /// </summary>
    private sealed class TestFixture : IDisposable
    {
        private readonly HttpClient _http;

        public TestFixture(string response)
        {
            Handler = new CaptureHandler(response);
            _http = new HttpClient(Handler);
            Client = new HouyiClient("http://127.0.0.1:1", _http);
        }

        public HouyiClient Client { get; }

        public CaptureHandler Handler { get; }

        public void Dispose() => _http.Dispose();
    }

    [Fact]
    public async Task VectorSparseWireShape()
    {
        using var fixture = new TestFixture("{\"inserted\":3}");
        await fixture.Client.GetCollection("docs").UpsertAsync(
        [
            new Vector { Id = 1, Values = [1.0f] },
            new Vector { Id = 2, Values = [1.0f], Sparse = [] },
            new Vector
            {
                Id = 3,
                Values = [1.0f],
                Sparse = [new SparseTerm(3, 1.5f), new SparseTerm(5, 0.25f)],
            },
        ]);

        using var body = JsonDocument.Parse(fixture.Handler.Body!);
        var vectors = body.RootElement.GetProperty("vectors");
        Assert.False(vectors[0].TryGetProperty("sparse", out _),
            "null sparse must omit the key");
        Assert.Equal(0, vectors[1].GetProperty("sparse").GetArrayLength());
        var sparse = vectors[2].GetProperty("sparse");
        Assert.Equal(2, sparse.GetArrayLength());
        Assert.Equal(3u, sparse[0][0].GetUInt32());
        Assert.Equal(1.5, sparse[0][1].GetDouble());
    }

    [Fact]
    public async Task CreateCollectionSendsSparse()
    {
        using var fixture = new TestFixture(
            "{\"created\":true,\"collection\":\"docs\",\"dims\":8," +
            "\"metric\":\"cosine\",\"index_type\":\"flat\",\"sparse\":\"bm25\"}");
        var result = await fixture.Client.CreateCollectionAsync(
            "docs", 8, new CreateCollectionOptions
            {
                IndexType = "flat",
                Sparse = "bm25",
            });

        Assert.Equal("PUT", fixture.Handler.Method);
        Assert.Equal("/v1/collections/docs", fixture.Handler.Path);
        using var body = JsonDocument.Parse(fixture.Handler.Body!);
        Assert.Equal("bm25", body.RootElement.GetProperty("sparse").GetString());
        Assert.Equal("flat", body.RootElement.GetProperty("index_type").GetString());
        Assert.Equal("bm25", result.Sparse);
    }

    [Fact]
    public async Task CreateCollectionOmitsSparseWhenUnset()
    {
        using var fixture = new TestFixture(
            "{\"created\":true,\"collection\":\"legacy\",\"dims\":4," +
            "\"metric\":\"l2\",\"index_type\":\"flat\"}");
        var result = await fixture.Client.CreateCollectionAsync("legacy", 4);

        using var body = JsonDocument.Parse(fixture.Handler.Body!);
        Assert.False(body.RootElement.TryGetProperty("sparse", out _),
            "unset sparse must not be sent");
        Assert.Null(result.Sparse);
    }

    [Fact]
    public async Task FusionSearchWireAndResponse()
    {
        using var fixture = new TestFixture(
            "{\"ids\":[9,4],\"scores\":[0.75,0.02],\"dense_scores\":[0.9,null]," +
            "\"sparse_scores\":[null,4.2],\"metric\":\"cosine\"," +
            "\"fusion\":{\"mode\":\"rrf\",\"rrf_k\":60,\"alpha\":null}," +
            "\"payloads\":{\"9\":{\"text\":\"nine\"}}}");
        var result = await fixture.Client.GetCollection("docs").FusionSearchAsync(
            [1.0f, 0.0f],
            [new SparseTerm(7, 0.5f), new SparseTerm(8, 0.25f)],
            2,
            new FusionOptions
            {
                IncludePayload = true,
                Fusion = "weighted",
                RrfK = 7,
                Alpha = 0.25f,
            });

        Assert.Equal("POST", fixture.Handler.Method);
        Assert.Equal("/v1/collections/docs/fusion/search", fixture.Handler.Path);
        using var body = JsonDocument.Parse(fixture.Handler.Body!);
        Assert.Equal(2, body.RootElement.GetProperty("k").GetInt32());
        Assert.True(body.RootElement.GetProperty("include_payload").GetBoolean());
        Assert.Equal("weighted", body.RootElement.GetProperty("fusion").GetString());
        Assert.Equal(7, body.RootElement.GetProperty("rrf_k").GetInt32());
        Assert.Equal(0.25, body.RootElement.GetProperty("alpha").GetDouble());
        Assert.Equal(2, body.RootElement.GetProperty("sparse").GetArrayLength());

        Assert.Equal("rrf", result.FusionMode);
        Assert.Equal(60u, result.RrfK);
        Assert.Null(result.Alpha);
        Assert.Equal("cosine", result.Metric);
        Assert.Equal(2, result.Hits.Count);
        Assert.Equal(0.9, result.Hits[0].DenseScore);
        Assert.Null(result.Hits[0].SparseScore);
        Assert.Equal("nine", result.Hits[0].Payload!["text"].GetString());
        Assert.Null(result.Hits[1].DenseScore);
        Assert.Equal(4.2, result.Hits[1].SparseScore);
    }

    [Fact]
    public async Task FusionResponseToleratesAbsentOptionalFields()
    {
        using var fixture = new TestFixture("{\"ids\":[1,2],\"scores\":[0.3,0.1]}");
        var result = await fixture.Client.GetCollection("docs").FusionSearchAsync(
            [1.0f], [new SparseTerm(1, 1.0f)], 2);

        Assert.Equal(2, result.Hits.Count);
        foreach (var hit in result.Hits)
        {
            Assert.Null(hit.DenseScore);
            Assert.Null(hit.SparseScore);
            Assert.Null(hit.Payload);
        }
        Assert.Null(result.Metric);
        Assert.Null(result.FusionMode);
        Assert.Null(result.RrfK);
        Assert.Null(result.Alpha);
    }
}
