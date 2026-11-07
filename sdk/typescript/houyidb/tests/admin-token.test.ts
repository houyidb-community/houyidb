import { describe, expect, it } from "vitest";
import { HouyiClient } from "../src/index.js";

/**
 * adminToken 凭据隔离：仅 /v1/admin/* 携带 Bearer，普通读写绝不携带，
 * 空 token 等价未配置。注入 fetch 桩记录每个请求。
 * EN: adminToken credential isolation: only /v1/admin/* carries the Bearer
 * header, normal calls never do, and an empty token means unset. A fetch stub
 * records every request.
 */
interface Captured {
  url: string;
  method: string;
  headers: Headers;
}

function recordingFetch(records: Captured[]): typeof fetch {
  const impl = async (input: RequestInfo | URL, init?: RequestInit) => {
    const url =
      typeof input === "string"
        ? input
        : input instanceof URL
          ? input.toString()
          : input.url;
    records.push({
      url,
      method: init?.method ?? "GET",
      headers: new Headers(init?.headers),
    });
    return new Response(
      JSON.stringify({ status: "ok", deleted: true, collection: "docs" }),
      { status: 200, headers: { "content-type": "application/json" } },
    );
  };
  return impl as unknown as typeof fetch;
}

describe("adminToken isolation", () => {
  it("sends Bearer only on admin DELETE, never on normal reads", async () => {
    const records: Captured[] = [];
    const client = new HouyiClient("http://houyi.test", {
      adminToken: "admin-token",
      fetch: recordingFetch(records),
    });

    await client.deleteCollection("a/b c");

    expect(records).toHaveLength(1);
    expect(records[0]?.method).toBe("DELETE");
    expect(records[0]?.url).toBe(
      "http://houyi.test/v1/admin/collections/a%2Fb%20c",
    );
    expect(records[0]?.headers.get("authorization")).toBe("Bearer admin-token");

    records.length = 0;
    await client.health();
    expect(records).toHaveLength(1);
    expect(records[0]?.headers.get("authorization")).toBeNull();
  });

  it("treats an empty token as unset", async () => {
    const records: Captured[] = [];
    const client = new HouyiClient("http://houyi.test", {
      adminToken: "",
      fetch: recordingFetch(records),
    });

    await client.deleteCollection("docs");

    expect(records).toHaveLength(1);
    expect(records[0]?.headers.get("authorization")).toBeNull();
  });
});
