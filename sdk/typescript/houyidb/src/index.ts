export { HouyiClient, Collection } from "./client.js";
export type {
  HouyiClientOptions,
  CollectionStatus,
  SearchOptions,
  HybridSearchOptions,
  DeleteOptions,
  // 建集合的公共类型必须从唯一入口导出，否则使用方只能依赖方法返回值的
  // 推断类型，无法声明变量或实现接口（OCR #29）。
  // EN: Creation types must be exported from the sole entry point; otherwise
  // consumers only get the method's inferred return type and cannot declare
  // variables or implement interfaces (OCR #29).
  HnswParams,
  CreateCollectionOptions,
  CreateCollectionResult,
} from "./client.js";
export {
  SearchResult,
  HouyiError,
  toVectorId,
} from "./types.js";
export type {
  Vector,
  VectorIdInput,
  Filter,
  FilterOp,
  SearchHit,
} from "./types.js";
export { stableId } from "./stable-id.js";
