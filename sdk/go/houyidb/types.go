package houyidb

import (
	"encoding/json"
	"fmt"
)

// 与服务端的契约要点：
// - collection 在首次 upsert 时隐式创建，维度由首个向量定型；
// - payload 键 "label" 为保留列，占用会 400（用顶层 Label 字段）；
// - 检索响应的 Score 是相似度（1 - cosine_distance，越大越相似）；
// - Go 的 uint64 原生覆盖 id 键空间，无 JS 的 2^53 精度问题。
//
// EN: Contract points with the server:
// EN: - A collection is created implicitly on the first upsert; dimensions are
// fixed by the first vector;
// EN: - The payload key "label" is reserved; using it returns 400 (use the
// top-level Label field instead);
// EN: - Score in search responses is a similarity (1 - cosine_distance,
// larger is more similar);
// EN: - Go's uint64 natively covers the id key space, with no JS-style 2^53
// precision issue.

// SparseTerm 稀疏词项：{term_id, weight}（term_id 取值 [1, 2^24-1]，weight
// 有限且为正；重复 term 由服务端求和）。
// EN: SparseTerm is a sparse term: {term_id, weight} (term_id in [1, 2^24-1]
// with a finite positive weight; duplicate terms are summed by the server).
type SparseTerm [2]float64

// Vector 一条待写入向量：ID + 分量 + 可选 Label/Payload/Sparse。
//
// Sparse 仅在 sparse=bm25 的集合上有效：nil 表示不改该文档的稀疏词项，
// 空切片（非 nil）表示清空。为了区分这两种语义，Vector 使用自定义
// MarshalJSON——encoding/json 的 omitempty 会把空切片也省略掉。
// 线格式映射只发生在 vectorWire 影子结构上，因此字段本身不挂 json 标签：
// 挂 `json:"-"` 只会让 Unmarshal 与嵌入场景静默得到零值，却不改变编码结果
//（自定义 MarshalJSON 优先于标签）。
// EN: Vector is a vector to write: ID + components + optional
// Label/Payload/Sparse. Sparse only applies to sparse=bm25 collections: nil
// leaves the document's sparse terms unchanged, while an empty (non-nil)
// slice clears them. Vector implements MarshalJSON to preserve that
// distinction—encoding/json's omitempty would omit an empty slice too. The
// wire mapping lives solely in the vectorWire shadow, so the fields carry no
// json tags: a `json:"-"` tag would only make Unmarshal and embedding
// scenarios silently yield zero values without changing the encoded shape
// (a custom MarshalJSON takes precedence over tags anyway).
type Vector struct {
	ID      uint64
	Values  []float64
	Label   string
	Payload map[string]interface{}
	Sparse  []SparseTerm
}

// vectorWire 是 Vector 的线格式影子结构；Sparse 用指针保留「缺省 vs 空」。
// EN: vectorWire is Vector's wire shadow; a Sparse pointer preserves the
// "absent vs empty" distinction.
type vectorWire struct {
	ID      uint64                 `json:"id"`
	Values  []float64              `json:"values"`
	Label   string                 `json:"label,omitempty"`
	Payload map[string]interface{} `json:"payload,omitempty"`
	Sparse  *[]SparseTerm          `json:"sparse,omitempty"`
}

// MarshalJSON 见 Vector 的字段说明。
// EN: MarshalJSON is described on Vector's fields.
func (v Vector) MarshalJSON() ([]byte, error) {
	var sparse *[]SparseTerm
	if v.Sparse != nil {
		// 取副本地址，避免调用方改动原切片时产生难以追踪的别名。
		// EN: take the address of a copy so caller-side mutations cannot alias.
		copied := v.Sparse
		sparse = &copied
	}
	return json.Marshal(vectorWire{
		ID:      v.ID,
		Values:  v.Values,
		Label:   v.Label,
		Payload: v.Payload,
		Sparse:  sparse,
	})
}

// FilterOp 过滤算子白名单（与服务端一致；多条 Filter 之间取交集）。
// EN: FilterOp is the filter operator whitelist (matches the server;
// multiple Filters are intersected).
type FilterOp string

const (
	OpEq  FilterOp = "eq"
	OpGt  FilterOp = "gt"
	OpGte FilterOp = "gte"
	OpLt  FilterOp = "lt"
	OpLte FilterOp = "lte"
	OpIn  FilterOp = "in"
)

// Filter 一条结构化过滤条件。
// EN: Filter is a structured filter condition.
type Filter struct {
	Column string      `json:"column"`
	Op     FilterOp    `json:"op"`
	Value  interface{} `json:"value"`
}

// SearchHit 单条命中：ID + 相似度分数 + 可选 Payload（IncludePayload 时）。
// EN: SearchHit is a single hit: ID + similarity score + optional Payload
// (when IncludePayload is set).
type SearchHit struct {
	ID      uint64
	Score   float64
	Payload map[string]interface{}
}

// SearchResult 检索响应：Hits 按相似度从高到低排列。
// EN: SearchResult is a search response: Hits are ordered by similarity from
// high to low.
type SearchResult struct {
	Hits []SearchHit
}

// IDs 返回命中的 id 序列（便捷访问器）。
// EN: IDs returns the hit id sequence (convenience accessor).
func (r *SearchResult) IDs() []uint64 {
	ids := make([]uint64, len(r.Hits))
	for i, h := range r.Hits {
		ids[i] = h.ID
	}
	return ids
}

// Scores 返回命中的分数序列（便捷访问器）。
// EN: Scores returns the hit score sequence (convenience accessor).
func (r *SearchResult) Scores() []float64 {
	scores := make([]float64, len(r.Hits))
	for i, h := range r.Hits {
		scores[i] = h.Score
	}
	return scores
}

// FusionHit 单条融合命中：ID + 融合分 + 两路原始分 + 可选 Payload。
//
// 融合分只表达排序（RRF 或加权），不属于任何度量口径；DenseScore 按集合
// 度量换算、仅稠密一路命中时非 nil；SparseScore 为 BM25 原始分、仅稀疏
// 一路命中时非 nil。
// EN: FusionHit is one fusion hit: ID + fused score + both raw modality
// scores + optional Payload. The fused score only expresses ranking (RRF or
// weighted) and belongs to no metric; DenseScore is converted under the
// collection metric and is non-nil only when the dense path hit; SparseScore
// is the raw BM25 score and is non-nil only when the sparse path hit.
type FusionHit struct {
	ID          uint64
	Score       float64
	DenseScore  *float64
	SparseScore *float64
	Payload     map[string]interface{}
}

// FusionResult 融合检索响应：Hits 按融合分从高到低排列，附生效参数回显。
//
// Metric 仅用于解释 DenseScore；FusionMode 为生效模式（rrf | weighted）；
// RRFK 仅 rrf 模式回显、Alpha 仅 weighted 模式回显（缺席为 nil）。
// EN: FusionResult is a fusion response: Hits are ordered by fused score,
// highest first, with the effective parameters echoed. Metric explains
// DenseScore only; FusionMode is the effective mode (rrf | weighted); RRFK is
// echoed only under rrf and Alpha only under weighted (nil when absent).
type FusionResult struct {
	Hits       []FusionHit
	Metric     string
	FusionMode string
	RRFK       *uint32
	Alpha      *float32
}

// IDs 返回命中的 id 序列（便捷访问器）。
// EN: IDs returns the hit id sequence (convenience accessor).
func (r *FusionResult) IDs() []uint64 {
	ids := make([]uint64, len(r.Hits))
	for i, h := range r.Hits {
		ids[i] = h.ID
	}
	return ids
}

// Scores 返回命中的融合分序列（便捷访问器）。
// EN: Scores returns the hit fused-score sequence (convenience accessor).
func (r *FusionResult) Scores() []float64 {
	scores := make([]float64, len(r.Hits))
	for i, h := range r.Hits {
		scores[i] = h.Score
	}
	return scores
}

// fusionResponse 融合响应的 wire 形状（与 searchResponse 同理，另加两路
// 平行分数组与 fusion 参数块；可选字段缺席时零值容忍）。
// EN: fusionResponse is the wire shape of a fusion response (same idea as
// searchResponse plus the two parallel modality score arrays and the fusion
// params block; absent optional fields fall back to zero values).
type fusionResponse struct {
	IDs          []uint64   `json:"ids"`
	Scores       []float64  `json:"scores"`
	DenseScores  []*float64 `json:"dense_scores"`
	SparseScores []*float64 `json:"sparse_scores"`
	Metric       string     `json:"metric"`
	Fusion       struct {
		Mode  string   `json:"mode"`
		RRFK  *uint32  `json:"rrf_k"`
		Alpha *float32 `json:"alpha"`
	} `json:"fusion"`
	Payloads map[string]map[string]interface{} `json:"payloads"`
}

func (r *fusionResponse) toResult() *FusionResult {
	hits := make([]FusionHit, 0, len(r.IDs))
	for i, id := range r.IDs {
		hit := FusionHit{ID: id}
		if i < len(r.Scores) {
			hit.Score = r.Scores[i]
		}
		if i < len(r.DenseScores) {
			hit.DenseScore = r.DenseScores[i]
		}
		if i < len(r.SparseScores) {
			hit.SparseScore = r.SparseScores[i]
		}
		if r.Payloads != nil {
			hit.Payload = r.Payloads[fmt.Sprintf("%d", id)]
		}
		hits = append(hits, hit)
	}
	return &FusionResult{
		Hits:       hits,
		Metric:     r.Metric,
		FusionMode: r.Fusion.Mode,
		RRFK:       r.Fusion.RRFK,
		Alpha:      r.Fusion.Alpha,
	}
}

// searchResponse 检索响应的 wire 形状：ids/scores 平行数组 +
// payloads 以 id 十进制字符串为键（服务端契约）。
// EN: searchResponse is the wire shape of a search response: parallel
// ids/scores arrays plus payloads keyed by the decimal string form of ids
// (server contract).
type searchResponse struct {
	IDs      []uint64                          `json:"ids"`
	Scores   []float64                         `json:"scores"`
	Payloads map[string]map[string]interface{} `json:"payloads"`
}

func (r *searchResponse) toResult() *SearchResult {
	hits := make([]SearchHit, 0, len(r.IDs))
	for i, id := range r.IDs {
		hit := SearchHit{ID: id}
		if i < len(r.Scores) {
			hit.Score = r.Scores[i]
		}
		if r.Payloads != nil {
			hit.Payload = r.Payloads[fmt.Sprintf("%d", id)]
		}
		hits = append(hits, hit)
	}
	return &SearchResult{Hits: hits}
}

// HouyiError 服务端返回的非 2xx 响应（StatusCode + 服务端诊断文本）。
// EN: HouyiError is a non-2xx response returned by the server (StatusCode +
// server diagnostic text).
type HouyiError struct {
	StatusCode int
	Message    string
}

func (e *HouyiError) Error() string {
	return fmt.Sprintf("houyidb HTTP %d: %s", e.StatusCode, e.Message)
}
