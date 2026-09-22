"""本地假 LLM / Embedding 服务：实现 OpenAI 流式 /chat/completions 与硅基流动 /v1/embeddings 协议。

用来在没有真实 API key 的情况下，端到端验证 app/llm/client.py 的：
  · SSE 流式解析（data: {...} / data: [DONE]）
  · /chat/completions 请求体构造
  · /v1/embeddings 的硅基流动标准请求与响应解析
  · 错误分支（可访问 /mode?fail=xxx 模拟）

运行：python _fake_llm_server.py    默认 http://127.0.0.1:8890
"""
from __future__ import annotations

import json
import sys
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(title="fake-llm")

MODE = {"fail": None, "delay": 0.02}

CHAT_REPLY = """```json
{"entities": [
  {"name": "张医生", "type": "人物", "key": "张医生", "time": ""},
  {"name": "北京协和医院", "type": "组织", "key": "北京协和医院", "time": "1921年"},
  {"name": "糖尿病", "type": "疾病", "key": "糖尿病", "time": ""}
],
"relations": [
  {"source": "张医生", "target": "北京协和医院", "predicate": "就职于", "directed": true, "time": "2023年"},
  {"source": "张医生", "target": "糖尿病", "predicate": "诊治", "directed": true, "time": ""}
]}```"""

SPLIT_REPLY = '{"sub_queries": ["胰岛素适用于哪些疾病", "胰岛素的用法用量", "糖尿病患者使用注意"]}'


def _is_split(payload: dict) -> bool:
    sysmsg = (payload.get("messages") or [{}])[0].get("content", "")
    return "检索查询改写" in sysmsg


@app.post("/v1/chat/completions")
async def chat(payload: dict, request: Request):
    if MODE["fail"] == "chat":
        return JSONResponse(status_code=401, content={"code": 30014, "message": "Token is invalid."})
    text = SPLIT_REPLY if _is_split(payload) else CHAT_REPLY
    # 按 token 切碎，真正走 SSE 分块下发
    pieces = [text[i:i + 8] for i in range(0, len(text), 8)]

    if not payload.get("stream"):
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "model": payload.get("model", "fake"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": len(pieces), "total_tokens": 10 + len(pieces)},
        }

    async def gen():
        cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        for i, piece in enumerate(pieces):
            if MODE["fail"] == "chat_mid":
                return
            ev = {
                "id": cid, "object": "chat.completion.chunk", "model": payload.get("model", "fake"),
                "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            if MODE["delay"]:
                import asyncio
                await asyncio.sleep(MODE["delay"])
        yield f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'model': 'fake', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


@app.post("/v1/embeddings")
async def emb(payload: dict, request: Request):
    if MODE["fail"] == "emb":
        return JSONResponse(status_code=500, content={"code": 500, "message": "embedding boom"})
    model = payload.get("model", "")
    inp = payload.get("input", [])
    if isinstance(inp, str):
        inp = [inp]
    dim = 8
    if "bge-large" in model:
        dim = 1024
    elif "bge-small" in model:
        dim = 512
    data = []
    for i, txt in enumerate(inp):
        v = [0.0] * dim
        for ch in str(txt):
            v[hash(ch) % dim] += 1.0 / max(1, len(str(txt)))
        n = sum(x * x for x in v) ** 0.5 or 1.0
        data.append({"object": "embedding", "index": i,
                     "embedding": [round(x / n, 6) for x in v]})
    return {
        "object": "list",
        "model": model,
        "data": data,
        "usage": {"prompt_tokens": sum(len(str(t)) for t in inp), "total_tokens": sum(len(str(t)) for t in inp)},
    }


@app.get("/mode")
async def mode(fail: str = ""):
    MODE["fail"] = fail or None
    return {"mode": MODE}


if __name__ == "__main__":
    import uvicorn
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8890
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
