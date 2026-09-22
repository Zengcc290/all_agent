import json, sys, urllib.request, urllib.error

def post(url, obj, timeout=90):
    data = json.dumps(obj).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8"), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "ignore"), None
    except Exception as e:
        return 0, "%s: %s" % (type(e).__name__, e), None

print("[1] 硅基流动格式 embeddings")
st, body, hd = post("http://127.0.0.1:8890/v1/embeddings",
                    {"model": "BAAI/bge-m3", "input": ["糖尿病", "胰岛素治疗糖尿病"], "encoding_format": "float"})
d = json.loads(body)
print("    status=%s data=%s dim=%d" % (st, len(d["data"]), len(d["data"][0]["embedding"])))
assert st == 200 and len(d["data"]) == 2

print("[2] 流式 chat/completions（SSE）")
req = urllib.request.Request(
    "http://127.0.0.1:8890/v1/chat/completions",
    data=json.dumps({"model": "fake", "messages": [{"role": "user", "content": "hi"}],
                     "stream": True, "temperature": 0.1, "max_tokens": 2048}).encode(),
    headers={"Content-Type": "application/json"}, method="POST")
with urllib.request.urlopen(req, timeout=60) as r:
    chunks = [l.decode() for l in r.readlines()]
events = [c for c in chunks if c.startswith("data:")]
print("    SSE 行数=%d, 含[DONE]=%s" % (len(events), any("[DONE]" in c for c in events)))
assert any("[DONE]" in c for c in events)
deltas = ""
for c in events:
    if "[DONE]" in c: continue
    ev = json.loads(c[5:].strip())
    deltas += ev["choices"][0]["delta"].get("content", "")
print("    拼回内容长度=%d, 开头=%s" % (len(deltas), deltas[:24]))
assert len(deltas) > 50

print("[3] 非流式 chat/completions")
st, body, _ = post("http://127.0.0.1:8890/v1/chat/completions",
                   {"model": "fake", "messages": [{"role": "user", "content": "hi"}]})
d = json.loads(body)
print("    status=%s content=%s..." % (st, d["choices"][0]["message"]["content"][:22]))

print("[4] 错误分支 chat 401")
st, body, _ = post("http://127.0.0.1:8890/v1/chat/completions",
                   {"model": "fake", "messages": [], "stream": True})
try:
    urllib.request.urlopen("http://127.0.0.1:8890/mode?fail=chat", timeout=10)
    st2, body2, _ = post("http://127.0.0.1:8890/v1/chat/completions",
                         {"model": "fake", "messages": [], "stream": True})
    print("    401 返回: status=%s body=%s" % (st2, body2[:80]))
    assert st2 == 401
finally:
    urllib.request.urlopen("http://127.0.0.1:8890/mode", timeout=10)
    print("    mode reset")

print("\nfake LLM 服务协议自检通过")
