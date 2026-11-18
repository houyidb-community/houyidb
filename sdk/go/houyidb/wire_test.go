package houyidb

// 本文件是 P4 新增面的 wire 契约测试（httptest，无真实服务端）：
// 逐向量 sparse 键存在性语义、创建 sparse 选项与旧服务端缺席容忍、
// 融合检索路由/请求体/响应平行分数组与 fusion 参数块。
//
// EN: Wire contract tests for the P4 additions (httptest, no real server):
// per-vector sparse key-presence semantics, the create sparse option plus
// tolerance for older servers omitting it, and the fusion route/request
// body/response parallel score arrays plus the fusion params block.

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

// captureServer 回放固定响应并记录最后一次请求。
// EN: captureServer replays a fixed response and records the last request.
type captureServer struct {
	method string
	path   string
	body   map[string]interface{}
}

func serveCapture(t *testing.T, resp string, cap *captureServer) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		cap.method = r.Method
		cap.path = r.URL.Path
		cap.body = nil
		if len(raw) > 0 {
			if err := json.Unmarshal(raw, &cap.body); err != nil {
				t.Errorf("request body is not valid JSON: %v (%s)", err, raw)
			}
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(resp))
	}))
	t.Cleanup(srv.Close)
	return srv
}

// TestVectorSparseWireShape 缺省 nil 省略键（不改），空切片输出 []（清空），
// 词项输出 [term, weight] 二元数组。
// EN: TestVectorSparseWireShape: nil omits the key (unchanged), an empty
// slice emits [] (clear), and terms emit [term, weight] pairs.
func TestVectorSparseWireShape(t *testing.T) {
	unset, err := json.Marshal(Vector{ID: 1, Values: []float64{1.0}})
	if err != nil {
		t.Fatalf("marshal unset vector: %v", err)
	}
	var unsetWire map[string]interface{}
	if err := json.Unmarshal(unset, &unsetWire); err != nil {
		t.Fatalf("unmarshal unset vector: %v", err)
	}
	if _, ok := unsetWire["sparse"]; ok {
		t.Fatalf("nil sparse must omit the key: %s", unset)
	}

	cleared, err := json.Marshal(Vector{ID: 2, Values: []float64{1.0}, Sparse: []SparseTerm{}})
	if err != nil {
		t.Fatalf("marshal cleared vector: %v", err)
	}
	var clearedWire map[string]interface{}
	if err := json.Unmarshal(cleared, &clearedWire); err != nil {
		t.Fatalf("unmarshal cleared vector: %v", err)
	}
	if sparse, ok := clearedWire["sparse"].([]interface{}); !ok || len(sparse) != 0 {
		t.Fatalf("empty sparse must serialize as []: %s", cleared)
	}

	terms, err := json.Marshal(Vector{
		ID:     3,
		Values: []float64{1.0},
		Sparse: []SparseTerm{{3, 1.5}, {5, 0.25}},
	})
	if err != nil {
		t.Fatalf("marshal sparse vector: %v", err)
	}
	var termsWire map[string]interface{}
	if err := json.Unmarshal(terms, &termsWire); err != nil {
		t.Fatalf("unmarshal sparse vector: %v", err)
	}
	sparse, ok := termsWire["sparse"].([]interface{})
	if !ok || len(sparse) != 2 {
		t.Fatalf("sparse terms must serialize as two pairs: %s", terms)
	}
	first, ok := sparse[0].([]interface{})
	if !ok || len(first) != 2 || first[0] != float64(3) || first[1] != 1.5 {
		t.Fatalf("first sparse pair mismatch: %s", terms)
	}
}

// TestCreateCollectionSendsSparse 创建请求携带 sparse 且响应回显；
// 旧服务端响应缺 sparse 时按空串容忍。
// EN: TestCreateCollectionSendsSparse: the create request carries sparse and
// the response echoes it; a missing sparse from an older server is tolerated
// as an empty string.
func TestCreateCollectionSendsSparse(t *testing.T) {
	var got captureServer
	srv := serveCapture(t,
		`{"created":true,"collection":"docs","dims":8,"metric":"cosine","index_type":"flat","sparse":"bm25"}`,
		&got)
	client := NewClient(srv.URL)
	result, err := client.CreateCollection(context.Background(), "docs", 8,
		&CreateCollectionOptions{IndexType: "flat", Sparse: "bm25"})
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}
	if got.method != http.MethodPut || got.path != "/v1/collections/docs" {
		t.Fatalf("unexpected route: %s %s", got.method, got.path)
	}
	if got.body["sparse"] != "bm25" || got.body["index_type"] != "flat" {
		t.Fatalf("request body mismatch: %v", got.body)
	}
	if result.Sparse != "bm25" {
		t.Fatalf("sparse echo mismatch: %q", result.Sparse)
	}
}

// TestCreateCollectionOmitsSparseWhenUnset 未设置 sparse 时请求体不带该键。
// EN: TestCreateCollectionOmitsSparseWhenUnset: the request omits the key
// when sparse is unset.
func TestCreateCollectionOmitsSparseWhenUnset(t *testing.T) {
	var got captureServer
	srv := serveCapture(t,
		`{"created":true,"collection":"legacy","dims":4,"metric":"l2","index_type":"flat"}`,
		&got)
	client := NewClient(srv.URL)
	result, err := client.CreateCollection(context.Background(), "legacy", 4, nil)
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}
	if _, ok := got.body["sparse"]; ok {
		t.Fatalf("unset sparse must not be sent: %v", got.body)
	}
	if result.Sparse != "" {
		t.Fatalf("older-server response must leave sparse empty, got %q", result.Sparse)
	}
}

// TestFusionSearchWireAndResponse 融合路由/请求体/响应逐项对齐。
// EN: TestFusionSearchWireAndResponse: fusion route/request body/response
// line up item by item.
func TestFusionSearchWireAndResponse(t *testing.T) {
	var got captureServer
	srv := serveCapture(t,
		`{"ids":[9,4],"scores":[0.75,0.02],"dense_scores":[0.9,null],`+
			`"sparse_scores":[null,4.2],"metric":"cosine",`+
			`"fusion":{"mode":"rrf","rrf_k":60,"alpha":null},`+
			`"payloads":{"9":{"text":"nine"}}}`,
		&got)
	client := NewClient(srv.URL)
	col := client.Collection("docs")
	alpha := float32(0.25)
	result, err := col.FusionSearch(context.Background(),
		[]float64{1.0, 0.0}, []SparseTerm{{7, 0.5}, {8, 0.25}}, 2,
		&FusionSearchOptions{
			IncludePayload: true,
			Fusion:         "weighted",
			RRFK:           7,
			Alpha:          &alpha,
		})
	if err != nil {
		t.Fatalf("fusion search: %v", err)
	}

	if got.method != http.MethodPost || got.path != "/v1/collections/docs/fusion/search" {
		t.Fatalf("unexpected route: %s %s", got.method, got.path)
	}
	if got.body["k"] != float64(2) || got.body["include_payload"] != true {
		t.Fatalf("request body mismatch: %v", got.body)
	}
	if got.body["fusion"] != "weighted" || got.body["rrf_k"] != float64(7) {
		t.Fatalf("fusion knobs mismatch: %v", got.body)
	}
	// JSON 数字经 interface{} 解码为 float64；alpha 必须是显式 0.25。
	// EN: JSON numbers decode into interface{} as float64; alpha must be the
	// explicitly given 0.25.
	if got.body["alpha"] != 0.25 {
		t.Fatalf("alpha mismatch: %v", got.body["alpha"])
	}
	sparse := got.body["sparse"].([]interface{})
	if len(sparse) != 2 {
		t.Fatalf("sparse pairs mismatch: %v", got.body["sparse"])
	}

	if result.FusionMode != "rrf" || result.RRFK == nil || *result.RRFK != 60 {
		t.Fatalf("fusion echo mismatch: %+v", result)
	}
	if result.Alpha != nil {
		t.Fatalf("rrf mode must not echo alpha: %v", *result.Alpha)
	}
	if result.Metric != "cosine" || len(result.Hits) != 2 {
		t.Fatalf("result shape mismatch: %+v", result)
	}
	if result.Hits[0].DenseScore == nil || *result.Hits[0].DenseScore != 0.9 {
		t.Fatalf("dense score mismatch: %+v", result.Hits[0])
	}
	if result.Hits[0].SparseScore != nil {
		t.Fatalf("dense-only hit must have nil sparse score: %+v", result.Hits[0])
	}
	if result.Hits[1].DenseScore != nil {
		t.Fatalf("sparse-only hit must have nil dense score: %+v", result.Hits[1])
	}
	if result.Hits[1].SparseScore == nil || *result.Hits[1].SparseScore != 4.2 {
		t.Fatalf("sparse score mismatch: %+v", result.Hits[1])
	}
	if result.Hits[0].Payload["text"] != "nine" {
		t.Fatalf("payload mismatch: %+v", result.Hits[0].Payload)
	}
}

// TestFusionResponseToleratesAbsentOptionalFields 响应缺可选字段时不报错。
// EN: TestFusionResponseToleratesAbsentOptionalFields: a response missing
// optional fields does not error.
func TestFusionResponseToleratesAbsentOptionalFields(t *testing.T) {
	var got captureServer
	srv := serveCapture(t, `{"ids":[1,2],"scores":[0.3,0.1]}`, &got)
	client := NewClient(srv.URL)
	result, err := client.Collection("docs").FusionSearch(context.Background(),
		[]float64{1.0}, []SparseTerm{{1, 1.0}}, 2, nil)
	if err != nil {
		t.Fatalf("fusion search: %v", err)
	}
	if len(result.Hits) != 2 {
		t.Fatalf("hits mismatch: %+v", result.Hits)
	}
	for _, hit := range result.Hits {
		if hit.DenseScore != nil || hit.SparseScore != nil || hit.Payload != nil {
			t.Fatalf("absent optional fields must stay nil: %+v", hit)
		}
	}
	if result.Metric != "" || result.FusionMode != "" || result.RRFK != nil || result.Alpha != nil {
		t.Fatalf("absent fusion block must stay zero-valued: %+v", result)
	}
}
