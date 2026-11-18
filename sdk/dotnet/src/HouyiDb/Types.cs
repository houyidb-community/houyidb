using System.Text.Json;
using System.Text.Json.Serialization;

namespace HouyiDb;

/// <summary>
/// 稀疏词项 {term_id, weight}（term_id 取值 [1, 2^24-1]，weight 有限且为正；
/// 重复 term 由服务端求和）。wire 形状是 [term, weight] 二元数组。
/// EN: Sparse term {term_id, weight} (term_id in [1, 2^24-1], weight finite and
/// positive; duplicate terms are summed by the server). The wire shape is the
/// [term, weight] pair array.
/// </summary>
[JsonConverter(typeof(SparseTermConverter))]
public readonly record struct SparseTerm(uint Term, float Weight);

/// <summary>
/// <see cref="SparseTerm"/> 的 wire 编解码：[term, weight] 二元数组。
/// EN: Wire codec for <see cref="SparseTerm"/>: the [term, weight] pair array.
/// </summary>
public sealed class SparseTermConverter : JsonConverter<SparseTerm>
{
    public override SparseTerm Read(
        ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        if (reader.TokenType != JsonTokenType.StartArray)
        {
            throw new JsonException("sparse term must be a [term, weight] array");
        }
        reader.Read();
        var term = reader.GetUInt32();
        reader.Read();
        var weight = reader.GetSingle();
        reader.Read();
        if (reader.TokenType != JsonTokenType.EndArray)
        {
            throw new JsonException("sparse term must contain exactly two numbers");
        }
        return new SparseTerm(term, weight);
    }

    public override void Write(
        Utf8JsonWriter writer, SparseTerm value, JsonSerializerOptions options)
    {
        writer.WriteStartArray();
        writer.WriteNumberValue(value.Term);
        writer.WriteNumberValue(value.Weight);
        writer.WriteEndArray();
    }
}

/// <summary>
/// 一条待写入向量：id + 分量 + 可选 label/payload/sparse。
/// sparse 仅在 sparse=bm25 的集合上有效：null 表示不改该文档的稀疏词项，
/// 空列表（非 null）表示清空。
/// EN: A vector to write: id + components + optional label/payload/sparse.
/// sparse applies only to sparse=bm25 collections: null leaves the document's
/// sparse terms unchanged, while an empty (non-null) list clears them.
/// </summary>
public sealed class Vector
{
    [JsonPropertyName("id")]
    public ulong Id { get; set; }

    [JsonPropertyName("values")]
    public required float[] Values { get; set; }

    [JsonPropertyName("label"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Label { get; set; }

    [JsonPropertyName("payload"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public Dictionary<string, object?>? Payload { get; set; }

    /// <summary>
    /// 稀疏词项：null = 不改，空列表 = 清空。
    /// EN: Sparse terms: null = unchanged, empty list = clear.
    /// </summary>
    [JsonPropertyName("sparse"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public List<SparseTerm>? Sparse { get; set; }
}

/// <summary>
/// 过滤算子白名单（与服务端一致；多条 <see cref="Filter"/> 之间取交集）。
/// wire 值即小写字符串，直接序列化，无需枚举转换器。
/// EN: Filter operator whitelist (matches the server; multiple <see cref="Filter"/>
/// entries are intersected). The wire value is the lowercase string, serialized
/// directly with no enum converter.
/// </summary>
public static class FilterOp
{
    public const string Eq = "eq";
    public const string Gt = "gt";
    public const string Gte = "gte";
    public const string Lt = "lt";
    public const string Lte = "lte";
    public const string In = "in";
}

/// <summary>
/// 一条结构化过滤条件。
/// EN: One structured filter condition.
/// </summary>
public sealed class Filter
{
    public Filter(string column, string op, object value)
    {
        Column = column;
        Op = op;
        Value = value;
    }

    [JsonPropertyName("column")]
    public string Column { get; }

    [JsonPropertyName("op")]
    public string Op { get; }

    [JsonPropertyName("value")]
    public object Value { get; }
}

/// <summary>
/// 单条命中：id + 相似度分数 + 可选 payload（include_payload 时）。
/// EN: One hit: id + similarity score + optional payload (when include_payload is set).
/// </summary>
public sealed class SearchHit
{
    public required ulong Id { get; init; }
    public required double Score { get; init; }
    public IReadOnlyDictionary<string, JsonElement>? Payload { get; init; }
}

/// <summary>
/// 检索响应：hits 按相似度从高到低排列。
/// EN: Search response: hits ordered from highest to lowest similarity.
/// </summary>
public sealed class SearchResult
{
    public required IReadOnlyList<SearchHit> Hits { get; init; }

    public ulong[] Ids() => Hits.Select(h => h.Id).ToArray();

    public double[] Scores() => Hits.Select(h => h.Score).ToArray();

    /// <summary>
    /// 检索响应的 wire 形状：ids/scores 平行数组 + payloads 以 id 十进制字符串为键。
    /// EN: Wire shape of the search response: parallel ids/scores arrays plus
    /// payloads keyed by the decimal id string.
    /// </summary>
    internal sealed class Wire
    {
        [JsonPropertyName("ids")]
        public ulong[] Ids { get; set; } = [];

        [JsonPropertyName("scores")]
        public double[] Scores { get; set; } = [];

        [JsonPropertyName("payloads")]
        public Dictionary<string, Dictionary<string, JsonElement>> Payloads { get; set; } = new();
    }

    internal static SearchResult FromWire(Wire wire) => new()
    {
        Hits = wire.Ids.Select((id, i) => new SearchHit
        {
            Id = id,
            Score = i < wire.Scores.Length ? wire.Scores[i] : 0.0,
            Payload = wire.Payloads.TryGetValue(id.ToString(), out var p) ? p : null,
        }).ToList(),
    };
}

/// <summary>
/// 检索/混合检索可选参数（默认构造即全默认）。
/// EN: Optional search/hybrid-search parameters (default construction means all defaults).
/// </summary>
public sealed class SearchOptions
{
    public int? EfSearch { get; set; }
    public bool IncludePayload { get; set; }
    public List<Filter> Filters { get; } = [];
}

/// <summary>
/// 融合检索可选参数（默认构造即服务端默认：rrf + rrf_k=60）。
/// fusion 取 rrf | weighted；rrfK 仅 rrf 生效，alpha 仅 weighted 生效
/// （alpha ∈ [0,1]）。
/// EN: Optional fusion-search parameters (default construction means server
/// defaults: rrf with rrf_k=60). fusion is rrf | weighted; rrfK applies only to
/// rrf and alpha only to weighted (alpha in [0,1]).
/// </summary>
public sealed class FusionOptions
{
    public int? EfSearch { get; set; }
    public bool IncludePayload { get; set; }
    public List<Filter> Filters { get; } = [];
    public string? Fusion { get; set; }
    public int? RrfK { get; set; }
    public float? Alpha { get; set; }
}

/// <summary>
/// 单条融合命中：id + 融合分 + 两路原始分 + 可选 payload。
///
/// 融合分（Score）只表达排序（RRF 或加权），不属于任何度量口径；DenseScore
/// 按集合度量换算、仅稠密一路命中时非 null；SparseScore 为 BM25 原始分、
/// 仅稀疏一路命中时非 null。
/// EN: One fusion hit: id + fused score + both raw modality scores + optional
/// payload. The fused score expresses ranking only (RRF or weighted) and
/// belongs to no metric; DenseScore follows the collection metric and is
/// non-null only when the dense path hit; SparseScore is the raw BM25 score and
/// is non-null only when the sparse path hit.
/// </summary>
public sealed class FusionHit
{
    public required ulong Id { get; init; }
    public required double Score { get; init; }
    public double? DenseScore { get; init; }
    public double? SparseScore { get; init; }
    public IReadOnlyDictionary<string, JsonElement>? Payload { get; init; }
}

/// <summary>
/// 融合检索响应：hits 按融合分从高到低排列，附生效参数回显。
/// Metric 仅用于解释 DenseScore；FusionMode 为生效模式；RrfK 仅 rrf 模式
/// 回显、Alpha 仅 weighted 模式回显（缺席为 null）。
/// EN: Fusion response: hits ordered by fused score, highest first, with the
/// effective parameters echoed. Metric explains DenseScore only; FusionMode is
/// the effective mode; RrfK is echoed only under rrf and Alpha only under
/// weighted (null when absent).
/// </summary>
public sealed class FusionResult
{
    public required IReadOnlyList<FusionHit> Hits { get; init; }
    public string? Metric { get; init; }
    public string? FusionMode { get; init; }
    public uint? RrfK { get; init; }
    public float? Alpha { get; init; }

    public ulong[] Ids() => Hits.Select(h => h.Id).ToArray();

    public double[] Scores() => Hits.Select(h => h.Score).ToArray();

    /// <summary>
    /// 融合响应的 wire 形状：ids/scores 平行数组 + dense_scores/sparse_scores
    /// 平行分数组（缺席模态为 JSON null）+ metric + fusion 参数块 + 可选 payloads；
    /// 可选字段缺席时按 null 容忍。
    /// EN: Wire shape of the fusion response: parallel ids/scores arrays plus the
    /// parallel dense_scores/sparse_scores arrays (JSON null for an absent
    /// modality), metric, the fusion params block, and optional payloads;
    /// absent optional fields fall back to null.
    /// </summary>
    internal sealed class Wire
    {
        [JsonPropertyName("ids")]
        public ulong[] Ids { get; set; } = [];

        [JsonPropertyName("scores")]
        public double[] Scores { get; set; } = [];

        [JsonPropertyName("dense_scores")]
        public double?[] DenseScores { get; set; } = [];

        [JsonPropertyName("sparse_scores")]
        public double?[] SparseScores { get; set; } = [];

        [JsonPropertyName("metric")]
        public string? Metric { get; set; }

        [JsonPropertyName("fusion")]
        public FusionWire? Fusion { get; set; }

        [JsonPropertyName("payloads")]
        public Dictionary<string, Dictionary<string, JsonElement>> Payloads { get; set; } = new();
    }

    internal sealed class FusionWire
    {
        [JsonPropertyName("mode")]
        public string? Mode { get; set; }

        [JsonPropertyName("rrf_k")]
        public uint? RrfK { get; set; }

        [JsonPropertyName("alpha")]
        public float? Alpha { get; set; }
    }

    internal static FusionResult FromWire(Wire wire) => new()
    {
        Hits = wire.Ids.Select((id, i) => new FusionHit
        {
            Id = id,
            Score = i < wire.Scores.Length ? wire.Scores[i] : 0.0,
            DenseScore = i < wire.DenseScores.Length ? wire.DenseScores[i] : null,
            SparseScore = i < wire.SparseScores.Length ? wire.SparseScores[i] : null,
            Payload = wire.Payloads.TryGetValue(id.ToString(), out var p) ? p : null,
        }).ToList(),
        Metric = wire.Metric,
        FusionMode = wire.Fusion?.Mode,
        RrfK = wire.Fusion?.RrfK,
        Alpha = wire.Fusion?.Alpha,
    };
}

/// <summary>
/// 建集合时的 HNSW 参数三元组（必须三个全给，否则服务端 400）。
/// EN: HNSW parameter triple for explicit creation (all three are required or
/// the server returns 400).
/// </summary>
public sealed record HnswParams(int M, int EfConstruction, int EfSearch);

/// <summary>
/// 显式建集合的选项（省略即服务端默认：余弦 + 默认后端 + 默认 HNSW）。
/// metric 与后端创建后不可变，改参数只能删除重建（旧图按旧度量构建）。
/// EN: Options for explicit creation (omitted means server defaults: cosine,
/// default backend, default HNSW). Metric and backend are immutable afterwards;
/// changing them requires drop-and-recreate (old graphs were built under the
/// old metric).
/// </summary>
public sealed class CreateCollectionOptions
{
    /// <summary>
    /// 度量：l2 / cosine / ip / l1（大小写不敏感，服务端接受常用别名）。
    /// EN: Metric: l2 / cosine / ip / l1 (case-insensitive; common aliases accepted).
    /// </summary>
    public string? Metric { get; set; }

    /// <summary>
    /// 后端类型：hnsw / pq-flat / flat / clustered / flat-clustered / hii-l0。
    /// EN: Backend kind: hnsw / pq-flat / flat / clustered / flat-clustered / hii-l0.
    /// </summary>
    public string? IndexType { get; set; }

    /// <summary>
    /// 向量存储编码：f32（缺省）或 f16（仅 hnsw，省一半向量内存）。
    /// 创建后不可变；非 f32 与非 hnsw 组合由服务端拒绝（400）。
    /// EN: Vector storage encoding: f32 (default) or f16 (hnsw only, halves
    /// vector memory). Immutable after creation; non-f32 with a non-hnsw
    /// backend is rejected by the server (400).
    /// </summary>
    public string? VectorDtype { get; set; }

    /// <summary>
    /// 图侧量化：none（缺省）、int8 或 rabitq（仅 hnsw，压缩图内距离工作集；
    /// 主表保留原始向量并在检索时精确重排）。rabitq 为 1 bit/dim，仅支持
    /// metric l2|cosine（ip/l1 返回 400）。创建后不可变；非 none 与非 hnsw
    /// 组合由服务端拒绝（400）。
    /// EN: Graph-side quantization: none (default), int8, or rabitq (hnsw only,
    /// shrinks the in-graph distance working set; the main store keeps raw
    /// vectors and reranks exactly at search time). rabitq packs 1 bit/dim and
    /// supports metric l2|cosine only (ip/l1 return 400). Immutable after
    /// creation; non-none with a non-hnsw backend is rejected by the server
    /// (400).
    /// </summary>
    public string? Quantization { get; set; }

    /// <summary>
    /// 稀疏索引模式：none（缺省）或 bm25（仅 hnsw/flat 后端）。开启后集合获得
    /// 倒排稀疏索引并支持 FusionSearchAsync；创建后不可变，隐式创建恒为 none。
    /// EN: Sparse index mode: none (default) or bm25 (hnsw/flat backends only).
    /// When enabled the collection gains an inverted sparse index and supports
    /// FusionSearchAsync; it is immutable after creation, and implicit creation
    /// is always none.
    /// </summary>
    public string? Sparse { get; set; }

    public HnswParams? Hnsw { get; set; }
}

/// <summary>
/// 建集合响应：服务端回显实际生效的参数（hnsw 仅 hnsw 后端出现）。
/// EN: Creation response echoing the effective parameters (hnsw only for the
/// hnsw backend).
/// </summary>
public sealed class CreateCollectionResult
{
    [JsonPropertyName("created")]
    public bool Created { get; init; }

    [JsonPropertyName("collection")]
    public required string Collection { get; init; }

    [JsonPropertyName("dims")]
    public int Dims { get; init; }

    [JsonPropertyName("metric")]
    public required string Metric { get; init; }

    [JsonPropertyName("index_type")]
    public required string IndexType { get; init; }

    /// <summary>
    /// 实际生效的存储编码（f32 | f16）；旧服务端不返回时为 null。
    /// EN: Effective storage encoding (f32 | f16); null from older servers.
    /// </summary>
    [JsonPropertyName("vector_dtype")]
    public string? VectorDtype { get; init; }

    /// <summary>
    /// 实际生效的图侧量化（none | int8 | rabitq）；旧服务端不返回时为 null。
    /// EN: Effective graph-side quantization (none | int8 | rabitq); null from
    /// older servers.
    /// </summary>
    [JsonPropertyName("quantization")]
    public string? Quantization { get; init; }

    /// <summary>
    /// 实际生效的稀疏模式（none | bm25）；旧服务端不返回时为 null。
    /// EN: Effective sparse mode (none | bm25); null from older servers.
    /// </summary>
    [JsonPropertyName("sparse")]
    public string? Sparse { get; init; }

    [JsonPropertyName("hnsw")]
    public HnswParamsWire? Hnsw { get; init; }

    /// <summary>
    /// 服务端响应中的 hnsw 对象（字段名与请求侧不同）。
    /// EN: the hnsw object in the server response (field names differ from the request side).
    /// </summary>
    public sealed class HnswParamsWire
    {
        [JsonPropertyName("m")]
        public int M { get; init; }

        [JsonPropertyName("ef_construction")]
        public int EfConstruction { get; init; }

        [JsonPropertyName("ef_search")]
        public int EfSearch { get; init; }
    }
}
