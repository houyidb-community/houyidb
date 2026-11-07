// adminToken 的凭据隔离单测：管理请求带 Bearer，普通读写绝不携带，空 token 等价未配置。
// EN: Credential-isolation unit tests for adminToken: admin requests carry the
// Bearer header, normal reads/writes never do, and an empty token means unset.
using System.Net;
using System.Net.Http;
using System.Text;
using Xunit;

namespace HouyiDb.Tests;

public class AdminTokenTests
{
    /// <summary>
    /// 记录全部请求并统一回 200 的最小 handler；断言只看请求头/路径。
    /// EN: Minimal handler that records every request and always answers 200;
    /// assertions inspect only headers and path.
    /// </summary>
    private sealed class RecordingHandler : HttpMessageHandler
    {
        public List<HttpRequestMessage> Requests { get; } = new();

        protected override Task<HttpResponseMessage> SendAsync(
            HttpRequestMessage request,
            CancellationToken cancellationToken)
        {
            Requests.Add(request);
            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(
                    "{\"deleted\":true,\"collection\":\"docs\"}",
                    Encoding.UTF8,
                    "application/json"),
            });
        }
    }

    /// <summary>
    /// DELETE /v1/admin/* 携带 Bearer；同一客户端的普通状态读取不携带。
    /// EN: DELETE /v1/admin/* carries Bearer; a normal status read through the
    /// same client does not.
    /// </summary>
    [Fact]
    public async Task AdminTokenIsSentOnlyOnAdminRequests()
    {
        var handler = new RecordingHandler();
        using var http = new HttpClient(handler);
        var client = new HouyiClient("http://example.test", http, null, "s3cret");

        await client.DeleteCollectionAsync("docs");

        var delete = Assert.Single(handler.Requests);
        Assert.Equal(HttpMethod.Delete, delete.Method);
        Assert.Equal("/v1/admin/collections/docs", delete.RequestUri!.AbsolutePath);
        Assert.NotNull(delete.Headers.Authorization);
        Assert.Equal("Bearer", delete.Headers.Authorization!.Scheme);
        Assert.Equal("s3cret", delete.Headers.Authorization.Parameter);

        handler.Requests.Clear();
        using var status = await client.GetCollection("docs").StatusAsync();
        var read = Assert.Single(handler.Requests);
        Assert.Null(read.Headers.Authorization);
    }

    /// <summary>
    /// 空 token 等价未配置：管理请求照常发出且不带 Authorization。
    /// EN: an empty token means unset: the admin request still goes out without
    /// an Authorization header.
    /// </summary>
    [Fact]
    public async Task EmptyAdminTokenSendsNoAuthorization()
    {
        var handler = new RecordingHandler();
        using var http = new HttpClient(handler);
        var client = new HouyiClient("http://example.test", http, null, "");

        await client.DeleteCollectionAsync("docs");

        var delete = Assert.Single(handler.Requests);
        Assert.Null(delete.Headers.Authorization);
    }

    /// <summary>
    /// 外部 HttpClient 与 timeout 同时提供属于调用方混淆，构造器必须显式报错，
    /// 且参数名指向 timeout（而非静默丢弃其一）。
    /// EN: supplying both an injected HttpClient and a timeout is a caller
    /// mistake; the constructor must fail loudly with the offending parameter
    /// (timeout) instead of silently dropping one of them.
    /// </summary>
    [Fact]
    public void ClientAndTimeoutTogetherAreRejected()
    {
        using var http = new HttpClient();
        var ex = Assert.Throws<ArgumentException>(
            () => new HouyiClient("http://example.test", http, TimeSpan.FromSeconds(5), null));
        Assert.Equal("timeout", ex.ParamName);
    }

    /// <summary>
    /// collection 名必须按单个路径段转义（斜杠/空格不能改变端点路径）。
    /// EN: the collection name must be escaped as one path segment (slashes and
    /// spaces cannot alter the endpoint path).
    /// </summary>
    [Fact]
    public async Task CollectionNameIsEscapedAsOneSegment()
    {
        var handler = new RecordingHandler();
        using var http = new HttpClient(handler);
        var client = new HouyiClient("http://example.test", http, null, "tok");

        await client.DeleteCollectionAsync("a/b c");

        var delete = Assert.Single(handler.Requests);
        Assert.Equal("/v1/admin/collections/a%2Fb%20c", delete.RequestUri!.AbsolutePath);
    }
}
