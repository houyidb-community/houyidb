// houyidb Go SDK × 真实服务的集成测试。
//
// 前置：houyi-server 监听 127.0.0.1:17000（与 Python/TS 联调同一实例），
// 可用 HOUYIDB_URL 覆盖；服务不可达时整体 Skip。运行：go test ./...
//
// EN: Integration tests for the houyidb Go SDK against a real server.
//
// EN: Prerequisite: houyi-server listens on 127.0.0.1:17000 (the same
// instance used for Python/TS integration), overridable via HOUYIDB_URL;
// everything is skipped when the service is unreachable. Run: go test ./...
package houyidb

import (
	"context"
	"encoding/binary"
	"errors"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"golang.org/x/crypto/blake2b"
)

const (
	testCollection = "go_it"
	testDims       = 16
)

func testBaseURL() string {
	if v := os.Getenv("HOUYIDB_URL"); v != "" {
		return strings.TrimRight(v, "/")
	}
	return "http://127.0.0.1:17000"
}

// fakeEmbed 确定性伪 embedding（与 Python/TS 联调同款思路，自洽即可）。
// EN: fakeEmbed builds a deterministic pseudo-embedding (same idea as the
// Python/TS integration; self-consistency is enough).
func fakeEmbed(text string) []float64 {
	out := make([]float64, 0, testDims)
	for counter := 0; len(out) < testDims; counter++ {
		h, _ := blake2b.New(8, nil)
		h.Write([]byte(text + ":" + strconv.Itoa(counter)))
		v := binary.BigEndian.Uint64(h.Sum(nil))
		out = append(out, float64(v)/18446744073709551616.0*2.0-1.0)
	}
	return out[:testDims]
}

// waitFlushed 等待异步构图队列排空（双速写入：flush 后才对检索可见）。
// EN: waitFlushed waits for the async graph-building queue to drain
// (dual-speed writes: visible to search only after flush).
func waitFlushed(t *testing.T, ctx context.Context, col *Collection) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		status, err := col.Status(ctx)
		if err == nil {
			if pending, ok := status["pending_count"].(float64); ok && pending == 0 {
				return
			}
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("collection %s flush timeout", col.Name())
}

func newTestClient(t *testing.T) *Client {
	t.Helper()
	client := NewClient(testBaseURL())
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	if !client.Health(ctx) {
		t.Skipf("houyi-server 不可达（%s）：先启动 "+
			"./target/release/houyi-server --benchmark-http-addr 127.0.0.1:17000",
			testBaseURL())
	}
	return client
}

func TestUpsertSearchRoundTrip(t *testing.T) {
	client := newTestClient(t)
	ctx := context.Background()
	col := client.Collection(testCollection)

	texts := []string{"apple is a fruit", "banana is yellow", "laptop is electronics"}
	categories := []string{"fruit", "fruit", "electronics"}
	prices := []int{5, 3, 999}
	vectors := make([]Vector, len(texts))
	for i, text := range texts {
		vectors[i] = Vector{
			ID:     StableID("go-doc-" + strconv.Itoa(i)),
			Values: fakeEmbed(text),
			Payload: map[string]interface{}{
				"text": text, "category": categories[i], "price": prices[i],
			},
		}
	}
	inserted, err := col.Upsert(ctx, vectors)
	if err != nil {
		t.Fatalf("upsert: %v", err)
	}
	if inserted != 3 {
		t.Fatalf("inserted = %d, want 3", inserted)
	}
	waitFlushed(t, ctx, col)

	result, err := col.Search(ctx, fakeEmbed("apple is a fruit"), 2,
		&SearchOptions{IncludePayload: true})
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(result.Hits) != 2 {
		t.Fatalf("hits = %d, want 2", len(result.Hits))
	}
	if result.Hits[0].Payload["text"] != "apple is a fruit" {
		t.Errorf("top hit text = %v", result.Hits[0].Payload["text"])
	}
	if result.Hits[0].Score < result.Hits[1].Score {
		t.Error("scores must be descending")
	}
}

func TestHybridSearchFilters(t *testing.T) {
	client := newTestClient(t)
	ctx := context.Background()
	col := client.Collection(testCollection)

	result, err := col.HybridSearch(ctx, fakeEmbed("apple is a fruit"), 3,
		&HybridSearchOptions{
			Filters: []Filter{
				{Column: "category", Op: OpEq, Value: "fruit"},
				{Column: "price", Op: OpLt, Value: 4},
			},
			IncludePayload: true,
		})
	if err != nil {
		t.Fatalf("hybrid search: %v", err)
	}
	if len(result.Hits) != 1 || result.Hits[0].Payload["text"] != "banana is yellow" {
		t.Fatalf("expected only banana, got %+v", result.Hits)
	}
}

// 异键集 payload：PAX 新列回填的端到端守卫（曾在此场景整列错位）。
// EN: Mixed payload key sets: an end-to-end guard for PAX backfilling of new
// columns (a whole column was once misaligned in this scenario).
func TestMixedPayloadKeySets(t *testing.T) {
	client := newTestClient(t)
	ctx := context.Background()
	col := client.Collection("go_it_mixed_keys")

	if _, err := col.Upsert(ctx, []Vector{
		{ID: 1, Values: fakeEmbed("row-a"), Payload: map[string]interface{}{"a": 1}},
		{ID: 2, Values: fakeEmbed("row-b"), Payload: map[string]interface{}{"b": 2}},
	}); err != nil {
		t.Fatalf("upsert: %v", err)
	}
	// delete-by-filter 不依赖构图排空（标量页同步落页），hybrid 过滤同样按页求值。
	// EN: delete-by-filter does not depend on graph-building drain (scalar pages
	// are written synchronously); hybrid filtering is likewise evaluated on pages.
	result, err := col.HybridSearch(ctx, fakeEmbed("row-b"), 5,
		&HybridSearchOptions{
			Filters:        []Filter{{Column: "b", Op: OpEq, Value: 2}},
			IncludePayload: true,
		})
	if err != nil {
		t.Fatalf("hybrid: %v", err)
	}
	if len(result.Hits) != 1 || result.Hits[0].ID != 2 {
		t.Fatalf("filter b==2 must hit id 2 only, got %+v", result.Hits)
	}
	if _, leaked := result.Hits[0].Payload["a"]; leaked {
		t.Errorf("payload must not leak column 'a' from the other row: %+v",
			result.Hits[0].Payload)
	}
}

func TestLargeUint64IDRoundTrip(t *testing.T) {
	client := newTestClient(t)
	ctx := context.Background()
	col := client.Collection(testCollection)

	// StableID 落点常态是 19 位十进制（>2^53）；Go uint64 原生无损。
	// EN: StableID output is typically 19 decimal digits (>2^53); Go's uint64
	// is natively lossless.
	id := StableID("hello")
	if _, err := col.Upsert(ctx, []Vector{
		{ID: id, Values: fakeEmbed("uint64 probe"), Payload: map[string]interface{}{"tag": "big"}},
	}); err != nil {
		t.Fatalf("upsert: %v", err)
	}
	waitFlushed(t, ctx, col)
	result, err := col.HybridSearch(ctx, fakeEmbed("uint64 probe"), 1,
		&HybridSearchOptions{
			Filters:        []Filter{{Column: "tag", Op: OpEq, Value: "big"}},
			IncludePayload: true,
		})
	if err != nil {
		t.Fatalf("hybrid: %v", err)
	}
	if len(result.Hits) == 0 || result.Hits[0].ID != id {
		t.Fatalf("id round trip broken: want %d, got %+v", id, result.Hits)
	}
	if _, err := col.Delete(ctx, &DeleteOptions{IDs: []uint64{id}}); err != nil {
		t.Fatalf("cleanup delete: %v", err)
	}
}

func TestDeleteByIDsAndFilters(t *testing.T) {
	client := newTestClient(t)
	ctx := context.Background()
	col := client.Collection(testCollection)

	probeID := StableID("go-delete-me")
	if _, err := col.Upsert(ctx, []Vector{
		{ID: probeID, Values: fakeEmbed("delete probe"), Payload: map[string]interface{}{"tag": "del"}},
	}); err != nil {
		t.Fatalf("upsert: %v", err)
	}
	// delete-by-filter 不依赖构图排空（标量页同步落页）。
	// EN: delete-by-filter does not depend on graph-building drain (scalar
	// pages are written synchronously).
	deleted, err := col.Delete(ctx, &DeleteOptions{
		Filters: []Filter{{Column: "tag", Op: OpEq, Value: "del"}},
	})
	if err != nil {
		t.Fatalf("delete by filter: %v", err)
	}
	if deleted != 1 {
		t.Errorf("deleted = %d, want 1", deleted)
	}
	again, err := col.Delete(ctx, &DeleteOptions{IDs: []uint64{probeID}})
	if err != nil {
		t.Fatalf("delete by id: %v", err)
	}
	if again != 0 {
		t.Errorf("idempotent delete = %d, want 0", again)
	}
}

func TestServerErrorSurfacesAsHouyiError(t *testing.T) {
	client := newTestClient(t)
	ctx := context.Background()
	col := client.Collection("go_it_missing_collection")

	var houyiErr *HouyiError
	if _, err := col.Status(ctx); !errors.As(err, &houyiErr) {
		t.Fatalf("expected HouyiError, got %v", err)
	} else if houyiErr.StatusCode != 404 {
		t.Errorf("status = %d, want 404", houyiErr.StatusCode)
	}
}

func TestDeleteRequiresIDsOrFilters(t *testing.T) {
	client := newTestClient(t)
	if _, err := client.Collection(testCollection).Delete(
		context.Background(), &DeleteOptions{},
	); err == nil {
		t.Fatal("empty delete must fail")
	}
}

// TestCreateCollectionMetricAndHnsw 显式建集合回显度量与 HNSW 三元组；同名再建 409。
// EN: TestCreateCollectionMetricAndHnsw verifies the explicit-create echo of the
// metric and HNSW triple, and that a second create of the same name yields 409.
func TestCreateCollectionMetricAndHnsw(t *testing.T) {
	client := newTestClient(t)
	ctx := context.Background()
	const name = "go_it_create_metric"
	// 清理只容忍 404（从未创建）；其他错误直接失败。
	// EN: Cleanup tolerates only 404 (never created); anything else fails.
	var cleanupErr *HouyiError
	if err := client.DeleteCollection(ctx, name); err != nil &&
		(!errors.As(err, &cleanupErr) || cleanupErr.StatusCode != 404) {
		t.Fatalf("cleanup delete: %v", err)
	}

	created, err := client.CreateCollection(ctx, name, testDims, &CreateCollectionOptions{
		Metric:    "ip",
		IndexType: "hnsw",
		Hnsw:      &HnswConfig{M: 16, EFConstruction: 100, EFSearch: 32},
	})
	if err != nil {
		t.Fatalf("create collection: %v", err)
	}
	if !created.Created || created.Collection != name || created.Dims != testDims {
		t.Errorf("create echo = %+v, want created/%s/dims=%d", created, name, testDims)
	}
	if created.Metric != "ip" || created.IndexType != "hnsw" {
		t.Errorf("metric/index_type = %q/%q, want ip/hnsw", created.Metric, created.IndexType)
	}
	if created.Hnsw == nil ||
		created.Hnsw.M != 16 ||
		created.Hnsw.EFConstruction != 100 ||
		created.Hnsw.EFSearch != 32 {
		t.Errorf("hnsw echo = %+v, want m=16 ef_construction=100 ef_search=32", created.Hnsw)
	}

	var conflictErr *HouyiError
	if _, err := client.CreateCollection(ctx, name, testDims, nil); !errors.As(err, &conflictErr) {
		t.Errorf("second create must fail with HouyiError, got %v", err)
	} else if conflictErr.StatusCode != 409 {
		t.Errorf("status = %d, want 409", conflictErr.StatusCode)
	}
}

// TestDeleteCollectionAndRecreate 删除整库后 404；同名重建得到空 collection。
// EN: TestDeleteCollectionAndRecreate drops a whole collection (404 afterwards)
// and expects a same-name recreate to start empty.
func TestDeleteCollectionAndRecreate(t *testing.T) {
	client := newTestClient(t)
	ctx := context.Background()
	const name = "go_it_admin_drop"
	col := client.Collection(name)
	if _, err := col.Upsert(ctx, []Vector{
		{ID: 7, Values: fakeEmbed("drop me"), Payload: map[string]interface{}{"tag": "drop"}},
	}); err != nil {
		t.Fatalf("upsert: %v", err)
	}

	if err := client.DeleteCollection(ctx, name); err != nil {
		t.Fatalf("delete collection: %v", err)
	}

	var houyiErr *HouyiError
	if _, err := col.Status(ctx); !errors.As(err, &houyiErr) {
		t.Fatalf("expected HouyiError after drop, got %v", err)
	} else if houyiErr.StatusCode != 404 {
		t.Errorf("status = %d, want 404", houyiErr.StatusCode)
	}

	// 同名重建：首次写入隐式创建，旧向量不得复活。
	// EN: recreate by name: the first write implicitly creates it and the old
	// vectors must not reappear.
	if _, err := col.Upsert(ctx, []Vector{{ID: 8, Values: fakeEmbed("fresh")}}); err != nil {
		t.Fatalf("recreate: %v", err)
	}
	waitFlushed(t, ctx, col)
	status, err := col.Status(ctx)
	if err != nil {
		t.Fatalf("status after recreate: %v", err)
	}
	if indexed, _ := status["indexed_count"].(float64); indexed != 1 {
		t.Errorf("indexed_count = %v, want 1", status["indexed_count"])
	}
}

// TestDeleteMissingCollection404 删除不存在的 collection 返回上游 404。
// EN: TestDeleteMissingCollection404 expects the upstream 404 for an unknown
// collection.
func TestDeleteMissingCollection404(t *testing.T) {
	client := newTestClient(t)
	var houyiErr *HouyiError
	err := client.DeleteCollection(context.Background(), "go_it_admin_never_created")
	if !errors.As(err, &houyiErr) {
		t.Fatalf("expected HouyiError, got %v", err)
	} else if houyiErr.StatusCode != 404 {
		t.Errorf("status = %d, want 404", houyiErr.StatusCode)
	}
}
