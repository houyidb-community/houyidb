// Package houyidb 是 HouyiDB 向量数据库的官方 Go 客户端（collections HTTP API）。
// EN: Package houyidb is the official Go client for the HouyiDB vector
// database (collections HTTP API).
package houyidb

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// Client HouyiDB 客户端入口。
// EN: Client is the HouyiDB client entry point.
type Client struct {
	baseURL    string
	httpClient *http.Client
	// adminToken 可选管理 token：只附在 /v1/admin/* 请求上。
	// EN: adminToken is an optional admin token attached only to /v1/admin/* requests.
	adminToken string
}

// ClientOption 客户端可选配置。
// EN: ClientOption configures optional client settings.
type ClientOption func(*Client)

// WithTimeout 单次请求超时（连接+读取，默认 30s）。
// EN: WithTimeout sets the per-request timeout (connect + read, default 30s).
func WithTimeout(d time.Duration) ClientOption {
	return func(c *Client) { c.httpClient.Timeout = d }
}

// WithHTTPClient 注入自定义 http.Client（代理/传输层定制）。
// EN: WithHTTPClient injects a custom http.Client (proxy/transport customization).
func WithHTTPClient(hc *http.Client) ClientOption {
	return func(c *Client) { c.httpClient = hc }
}

// WithAdminToken 附加管理 token（空串等价于不配置）。
//
// token 只用于管理端点（如删除整个 collection）；未配置时管理请求照常发出，
// 由服务端 401/403 决定，不在客户端 fail-fast（服务端可能未启用鉴权）。
// EN: WithAdminToken attaches an admin token (empty means unset). It is used
// only by admin endpoints (e.g. dropping a collection); when unset the request
// still goes out and the server decides 401/403 — no client-side fail-fast,
// since the server may run without auth.
func WithAdminToken(token string) ClientOption {
	return func(c *Client) { c.adminToken = token }
}

// NewClient 构造客户端；baseURL 缺省 "http://127.0.0.1:7000"。
// EN: NewClient constructs a client; baseURL defaults to "http://127.0.0.1:7000".
func NewClient(baseURL string, opts ...ClientOption) *Client {
	if baseURL == "" {
		baseURL = "http://127.0.0.1:7000"
	}
	c := &Client{
		baseURL:    strings.TrimRight(baseURL, "/"),
		httpClient: &http.Client{Timeout: 30 * time.Second},
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// Collection 返回一个 named collection 的操作句柄。
// EN: Collection returns an operation handle for a named collection.
func (c *Client) Collection(name string) *Collection {
	return &Collection{
		client: c,
		name:   name,
		// URL 路径段必须转义：未编码的 / ? # 会改写路由结构，
		// 表现为莫名其妙的 404 且 collection 永远建不出来。
		// EN: URL path segments must be escaped: unencoded / ? # rewrite the
		// routing structure, producing a puzzling 404 where the collection can
		// never be created.
		quoted: url.PathEscape(name),
	}
}

// Health 服务健康检查（不可达/非 ok 一律 false，不返回错误）。
// EN: Health performs a service health check (unreachable or non-ok always
// yields false, without returning an error).
func (c *Client) Health(ctx context.Context) bool {
	var body struct {
		Status string `json:"status"`
	}
	if err := c.get(ctx, "/v1/health", &body); err != nil {
		return false
	}
	return body.Status == "ok"
}

// ListCollections 列出全部 collection。
// EN: ListCollections lists all collections.
func (c *Client) ListCollections(ctx context.Context) ([]map[string]interface{}, error) {
	var out []map[string]interface{}
	err := c.get(ctx, "/v1/collections", &out)
	return out, err
}

// IndexStatus 主索引双速写入指标（hot/pending/flush_count 等，可观测性用）。
// EN: IndexStatus reports primary-index dual-speed write metrics
// (hot/pending/flush_count, etc., for observability).
func (c *Client) IndexStatus(ctx context.Context) (map[string]interface{}, error) {
	var out map[string]interface{}
	err := c.get(ctx, "/v1/index/status", &out)
	return out, err
}

// HnswConfig 建集合时的 HNSW 参数三元组（必须三个全给，否则服务端 400）。
// EN: HnswConfig is the HNSW parameter triple for explicit creation (all three
// are required or the server returns 400).
type HnswConfig struct {
	M              int `json:"m"`
	EFConstruction int `json:"ef_construction"`
	EFSearch       int `json:"ef_search"`
}

// CreateCollectionOptions 显式建集合的选项（零值等价服务端默认：余弦 +
// 默认后端 + 默认 HNSW）。metric 与后端创建后不可变，改参数只能删除重建。
// EN: CreateCollectionOptions configures explicit creation (the zero value
// equals server defaults: cosine, default backend, default HNSW). Metric and
// backend are immutable after creation; changing them requires
// drop-and-recreate.
type CreateCollectionOptions struct {
	// Metric l2 / cosine / ip / l1（大小写不敏感，服务端接受常用别名）。
	// EN: l2/cosine/ip/l1 (case-insensitive; common aliases accepted).
	Metric string
	// IndexType 后端类型：hnsw / pq-flat / flat / clustered / flat-clustered / hii-l0。
	// EN: IndexType is the backend kind: hnsw / pq-flat / flat / clustered /
	// flat-clustered / hii-l0.
	IndexType string
	Hnsw      *HnswConfig
}

// CreateCollectionResult 建集合响应：服务端回显实际生效的参数。
// EN: CreateCollectionResult is the creation response echoing the effective
// parameters.
type CreateCollectionResult struct {
	Created    bool   `json:"created"`
	Collection string `json:"collection"`
	Dims       int    `json:"dims"`
	Metric     string `json:"metric"`
	IndexType  string `json:"index_type"`
	// Hnsw 仅 IndexType == "hnsw" 时存在。
	// EN: Hnsw is present only when IndexType == "hnsw".
	Hnsw *HnswConfig `json:"hnsw,omitempty"`
}

// CreateCollection 显式创建 collection（可指定度量/后端/HNSW 参数；需要写权限）。
//
// 与隐式创建（首次 upsert）不同，dims、metric、后端与 HNSW 参数在这里一次
// 定型并由服务端回显。普通写路径不携带管理 token；同名已存在 409、参数非法
// 400、follower/只读 503。
// EN: CreateCollection explicitly creates a collection (metric/backend/HNSW
// parameters are settable; requires write permission). Unlike implicit
// creation (first upsert), dims, metric, backend and HNSW parameters are fixed
// here and echoed back. This is the normal write path and does not carry the
// admin token; an existing name yields 409, invalid parameters 400, and
// followers/read-only nodes 503.
func (c *Client) CreateCollection(
	ctx context.Context, name string, dims int, opts *CreateCollectionOptions,
) (*CreateCollectionResult, error) {
	payload := map[string]interface{}{"dims": dims}
	if opts != nil {
		if opts.Metric != "" {
			payload["metric"] = opts.Metric
		}
		if opts.IndexType != "" {
			payload["index_type"] = opts.IndexType
		}
		if opts.Hnsw != nil {
			payload["hnsw_m"] = opts.Hnsw.M
			payload["hnsw_ef_construction"] = opts.Hnsw.EFConstruction
			payload["hnsw_ef_search"] = opts.Hnsw.EFSearch
		}
	}
	var out CreateCollectionResult
	err := c.put(ctx, "/v1/collections/"+url.PathEscape(name), payload, &out)
	if err != nil {
		return nil, err
	}
	return &out, nil
}

// DeleteCollection 删除整个 collection（不可逆；需要 admin 权限）。
//
// 服务端 WAL-first：记录先落盘，崩溃重放与 follower 复制都收敛到「已删除」；
// 不存在或已删除的 collection 返回 404。
// EN: DeleteCollection drops an entire collection (irreversible; admin required).
// The server is WAL-first, so crash replay and follower replication converge to
// "gone"; an unknown/already-dropped collection yields 404.
func (c *Client) DeleteCollection(ctx context.Context, name string) error {
	path := "/v1/admin/collections/" + url.PathEscape(name)
	req, err := http.NewRequestWithContext(ctx, http.MethodDelete, c.baseURL+path, nil)
	if err != nil {
		return err
	}
	// 管理请求才携带 Authorization；普通读写请求绝不上报 admin token。
	// EN: only admin requests carry Authorization; normal calls never do.
	if c.adminToken != "" {
		req.Header.Set("Authorization", "Bearer "+c.adminToken)
	}
	return c.do(req, &struct{}{})
}

// ── 内部传输 ──────────────────────────────────────────
// EN: Internal transport.

func (c *Client) get(ctx context.Context, path string, out interface{}) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+path, nil)
	if err != nil {
		return err
	}
	return c.do(req, out)
}

func (c *Client) post(ctx context.Context, path string, payload, out interface{}) error {
	return c.send(ctx, http.MethodPost, path, payload, out)
}

func (c *Client) put(ctx context.Context, path string, payload, out interface{}) error {
	return c.send(ctx, http.MethodPut, path, payload, out)
}

func (c *Client) send(
	ctx context.Context, method, path string, payload, out interface{},
) error {
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(
		ctx, method, c.baseURL+path, bytes.NewReader(body),
	)
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	return c.do(req, out)
}

func (c *Client) do(req *http.Request, out interface{}) error {
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		// 服务端错误体是纯文本诊断（axum 元组错误），原样透出。
		// EN: The server error body is a plain-text diagnostic (axum tuple
		// rejection); surface it verbatim.
		text, _ := io.ReadAll(resp.Body)
		msg := strings.TrimSpace(string(text))
		if msg == "" {
			msg = resp.Status
		}
		return &HouyiError{StatusCode: resp.StatusCode, Message: msg}
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

// Collection 一个 named collection 的操作句柄（经 Client.Collection 获取）。
// EN: Collection is an operation handle for a named collection (obtained via
// Client.Collection).
type Collection struct {
	client *Client
	name   string
	quoted string
}

// Name 返回 collection 名。
// EN: Name returns the collection name.
func (c *Collection) Name() string { return c.name }

// Upsert 写入一批向量（collection 首次写入时隐式创建）。返回插入条数。
// EN: Upsert writes a batch of vectors (the collection is created implicitly
// on the first write). Returns the number of inserted rows.
func (c *Collection) Upsert(ctx context.Context, vectors []Vector) (int, error) {
	var body struct {
		Inserted int `json:"inserted"`
	}
	err := c.client.post(ctx,
		fmt.Sprintf("/v1/collections/%s/upsert", c.quoted),
		map[string]interface{}{"vectors": vectors}, &body)
	return body.Inserted, err
}

// SearchOptions 检索可选参数（nil 即全默认）。
// EN: SearchOptions holds optional search parameters (nil means all defaults).
type SearchOptions struct {
	EFSearch       int
	IncludePayload bool
}

func (o *SearchOptions) apply(req map[string]interface{}) {
	if o == nil {
		return
	}
	if o.EFSearch > 0 {
		req["ef_search"] = o.EFSearch
	}
	if o.IncludePayload {
		req["include_payload"] = true
	}
}

// Search 纯向量 top-k 检索。
// EN: Search performs a pure vector top-k search.
func (c *Collection) Search(
	ctx context.Context, vector []float64, k int, opts *SearchOptions,
) (*SearchResult, error) {
	req := map[string]interface{}{"vector": vector, "k": k}
	opts.apply(req)
	var body searchResponse
	err := c.client.post(ctx,
		fmt.Sprintf("/v1/collections/%s/search", c.quoted), req, &body)
	if err != nil {
		return nil, err
	}
	return body.toResult(), nil
}

// HybridSearchOptions 混合检索可选参数。
// EN: HybridSearchOptions holds optional hybrid search parameters.
type HybridSearchOptions struct {
	Filters        []Filter
	EFSearch       int
	IncludePayload bool
}

// HybridSearch 向量 × 标量过滤的混合检索（Filters 取交集，仅 hnsw/flat 后端支持）。
// EN: HybridSearch performs a hybrid vector × scalar-filter search (Filters
// are intersected; only the hnsw/flat backends are supported).
func (c *Collection) HybridSearch(
	ctx context.Context, vector []float64, k int, opts *HybridSearchOptions,
) (*SearchResult, error) {
	req := map[string]interface{}{"vector": vector, "k": k}
	if opts != nil {
		if len(opts.Filters) > 0 {
			req["filters"] = opts.Filters
		}
		if opts.EFSearch > 0 {
			req["ef_search"] = opts.EFSearch
		}
		if opts.IncludePayload {
			req["include_payload"] = true
		}
	}
	var body searchResponse
	err := c.client.post(ctx,
		fmt.Sprintf("/v1/collections/%s/hybrid/search", c.quoted), req, &body)
	if err != nil {
		return nil, err
	}
	return body.toResult(), nil
}

// DeleteOptions 删除参数：IDs 与 Filters 至少给其一（两者取并集）。
// EN: DeleteOptions holds delete parameters: at least one of IDs or Filters
// (the two are unioned).
type DeleteOptions struct {
	IDs     []uint64
	Filters []Filter
}

// Delete 按 id 和/或过滤条件删除（幂等）。返回真实删除数。
//
// 按条件删除由服务端在标量页上求值：标量行随 upsert 同步落页，因此
// 刚写入、尚在异步构图队列中的向量也能被命中删除——这是「先检索拿 id
// 再删」无法保证的（检索在队列排空前看不到新写入）。
//
// EN: Delete deletes by ids and/or filter conditions (idempotent). Returns
// the actual number deleted.
//
// EN: Filter-based deletion is evaluated by the server on the scalar pages:
// scalar rows land on pages synchronously with upsert, so vectors just
// written and still in the async graph-building queue can also be matched
// and deleted—something "search first, then delete by id" cannot guarantee
// (search cannot see new writes before the queue drains).
func (c *Collection) Delete(ctx context.Context, opts *DeleteOptions) (int, error) {
	if opts == nil || (len(opts.IDs) == 0 && len(opts.Filters) == 0) {
		return 0, fmt.Errorf("delete requires at least one of ids or filters")
	}
	ids := opts.IDs
	if ids == nil {
		ids = []uint64{}
	}
	filters := opts.Filters
	if filters == nil {
		filters = []Filter{}
	}
	var body struct {
		Deleted int `json:"deleted"`
	}
	err := c.client.post(ctx,
		fmt.Sprintf("/v1/collections/%s/delete", c.quoted),
		map[string]interface{}{"ids": ids, "filters": filters}, &body)
	return body.Deleted, err
}

// Status collection 状态（indexed/pending/dims/index_type/scalar_row_count）。
// EN: Status returns the collection status
// (indexed/pending/dims/index_type/scalar_row_count).
func (c *Collection) Status(ctx context.Context) (map[string]interface{}, error) {
	var out map[string]interface{}
	err := c.client.get(ctx,
		fmt.Sprintf("/v1/collections/%s/status", c.quoted), &out)
	return out, err
}
