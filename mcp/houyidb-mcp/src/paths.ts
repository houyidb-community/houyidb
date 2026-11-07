/**
 * 构造 collections HTTP 路径。
 * EN: Builds collection HTTP paths.
 */

/**
 * 对路径段做 RFC 3986 转义。未转义的 `/`、`?`、`#` 会改写路由，
 * 表现为难以排查的 404；encodeURIComponent 留下了 `!'()*`，需补齐。
 * EN: Percent-encodes a path segment per RFC 3986. Unescaped `/`, `?`, or `#`
 * would rewrite the route and surface as a confusing 404; encodeURIComponent
 * leaves `!'()*` untouched, so those are encoded explicitly.
 */
export function encodePathSegment(segment: string): string {
  return encodeURIComponent(segment).replace(
    /[!'()*]/g,
    (char) => `%${char.charCodeAt(0).toString(16).toUpperCase()}`,
  );
}

/**
 * `collection` 路由后缀白名单，避免调用方拼出意外端点。
 * EN: Allowlisted collection route suffixes so callers cannot assemble
 * unintended endpoints.
 */
export type CollectionRoute =
  | "status"
  | "search"
  | "hybrid/search"
  | "upsert"
  | "delete";

/**
 * 返回 `/v1/collections/{name}/{route}`。
 * EN: Returns `/v1/collections/{name}/{route}`.
 */
export function collectionPath(
  collection: string,
  route: CollectionRoute,
): string {
  return `/v1/collections/${encodePathSegment(collection)}/${route}`;
}

/**
 * 返回 `/v1/collections/{name}`（PUT 显式建集合）。
 * EN: Returns `/v1/collections/{name}` (PUT explicit creation).
 */
export function collectionRootPath(collection: string): string {
  return `/v1/collections/${encodePathSegment(collection)}`;
}

/**
 * 返回管理端点 `/v1/admin/collections/{name}`（DELETE 删除整个 collection）。
 * EN: Returns the admin endpoint `/v1/admin/collections/{name}` (DELETE drops
 * the whole collection).
 */
export function adminCollectionPath(collection: string): string {
  return `/v1/admin/collections/${encodePathSegment(collection)}`;
}
