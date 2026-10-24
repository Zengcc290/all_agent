"""LLM 客户端：OpenAI 兼容 /chat/completions（支持流式）+ 硅基流动 /v1/embeddings。"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator, Callable

import httpx

from app import config

_retry = httpx.Limits(max_connections=20, max_keepalive_connections=10)


def _timeout(extra: float = 0) -> httpx.Timeout:
    return httpx.Timeout(config.llm.timeout + extra, connect=10.0)


class LLMError(RuntimeError):
    pass


async def chat_stream(messages: list[dict], on_delta: Callable[[str], Any] | None = None,
                      temperature: float | None = None, model: str | None = None,
                      max_tokens: int | None = None) -> str:
    """调用 OpenAI 兼容 /chat/completions，流式读取并拼接完整文本。
    on_delta 每收到一个增量片段回调一次；返回完整内容。"""
    payload = {
        "model": model or config.llm.model,
        "messages": messages,
        "temperature": config.llm.temperature if temperature is None else temperature,
        "max_tokens": max_tokens or config.llm.max_tokens,
        "stream": True,
    }
    headers = {"Content-Type": "application/json"}
    if config.llm.api_key:
        headers["Authorization"] = f"Bearer {config.llm.api_key}"

    chunks: list[str] = []
    try:
        async with httpx.AsyncClient(timeout=_timeout(60), limits=_retry) as client:
            async with client.stream("POST", config.llm.endpoint, json=payload, headers=headers) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode("utf-8", "ignore")[:800]
                    raise LLMError(f"LLM 请求失败 HTTP {resp.status_code}: {body}")
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        ev = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    for choice in ev.get("choices") or []:
                        delta = (choice.get("delta") or {}).get("content") or ""
                        if not delta:
                            continue
                        chunks.append(delta)
                        if on_delta:
                            r = on_delta(delta)
                            if hasattr(r, "__await__"):
                                await r
    except httpx.HTTPError as e:
        raise LLMError(f"LLM 网络错误: {type(e).__name__}: {e}") from e
    return "".join(chunks)


async def chat(messages: list[dict], temperature: float | None = None,
               model: str | None = None, max_tokens: int | None = None) -> str:
    """非流式便捷接口（内部仍走流式以复用同一套解析）。"""
    return await chat_stream(messages, None, temperature, model, max_tokens)


async def chat_json(messages: list[dict], **kw: Any) -> tuple[dict, str]:
    """直接要求 LLM 返回 JSON，解析失败时尝试一次修复。"""
    text = await chat(messages, **kw)
    from app.core.parser import parse_llm_output
    return parse_llm_output(text)[0], text


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """硅基流动标准 embeddings 请求：POST {base_url}/embeddings"""
    if not texts:
        return []
    headers = {"Content-Type": "application/json"}
    if config.embedding.api_key:
        headers["Authorization"] = f"Bearer {config.embedding.api_key}"
    out: list[list[float]] = []
    bs = max(1, config.embedding.batch_size)
    async with httpx.AsyncClient(timeout=_timeout(30), limits=_retry) as client:
        for i in range(0, len(texts), bs):
            batch = texts[i:i + bs]
            payload = {
                "model": config.embedding.model,
                "input": batch,
                "encoding_format": "float",
            }
            try:
                resp = await client.post(config.embedding.endpoint, json=payload, headers=headers)
            except httpx.HTTPError as e:
                raise LLMError(f"Embedding 网络错误: {e}") from e
            if resp.status_code >= 400:
                raise LLMError(f"Embedding 失败 HTTP {resp.status_code}: {resp.text[:800]}")
            data = resp.json()
            items = data.get("data") or []
            items.sort(key=lambda d: d.get("index", 0))
            for it in items:
                vec = it.get("embedding")
                if not vec:
                    raise LLMError(f"Embedding 响应缺少 embedding 字段: {str(data)[:400]}")
                out.append([float(x) for x in vec])
    if out and config.embedding.dim and len(out[0]) != config.embedding.dim:
        # 不致命：只是提示配置可能不匹配
        print(f"[embeddings] 注意：实际维度 {len(out[0])} 与配置 EMBEDDING_DIM="
              f"{config.embedding.dim} 不一致，请检查配置")
    return out


async def embed(text: str) -> list[float]:
    vs = await embed_texts([text])
    if not vs:
        raise LLMError("embedding 返回为空")
    return vs[0]


async def test_connection() -> dict:
    """健康探测：同时探测 LLM 与 Embedding。"""
    result: dict[str, Any] = {"llm": {"ok": False}, "embedding": {"ok": False}}
    try:
        text = await chat([{"role": "user", "content": "ping"}], max_tokens=8, temperature=0)
        result["llm"] = {"ok": True, "model": config.llm.model, "reply": text[:40]}
    except Exception as e:  # noqa: BLE001
        result["llm"] = {"ok": False, "error": str(e)}
    try:
        v = await embed("ping")
        result["embedding"] = {"ok": True, "model": config.embedding.model, "dim": len(v)}
    except Exception as e:  # noqa: BLE001
        result["embedding"] = {"ok": False, "error": str(e)}
    return result
