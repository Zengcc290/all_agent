"""验证 qwen3-embedding-0.6b 真实接入与中文检索效果。

检查项：
  1. 配置读取（DASHSCOPE_API_KEY / .env / 模型名 / 端点）
  2. 单条与批量真实调用（维度、数值有效性、耗时）
  3. 中文子词检索（旧 TF-IDF 无法命中的场景，验证是否被根治）
  4. 检索精度抽样（不相关记忆不应被误召回）

用法:
    .venv\\Scripts\\python.exe check_embedding.py            # 完整验证（需要 API key）
    .venv\\Scripts\\python.exe check_embedding.py --offline  # 只检查配置，不发网络请求
"""

from __future__ import annotations

import argparse
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from memory import APIEmbedding, MemoryConfig, MemoryManager, MemoryType, load_dotenv_once

PASS = "[PASS]"
FAIL = "[FAIL]"
INFO = "[INFO]"

# 旧 TF-IDF 实现下查询 "中文" 与 "用户偏好用中文回答技术问题" 相似度为 0，
# 因此这组用例专门用来确认语义嵌入是否真正解决了中文子词检索。
MEMORIES = [
    ("用户偏好用中文回答技术问题", "偏好"),
    ("项目使用 Qdrant 作为向量数据库", "向量数据库"),
    ("记忆系统包含工作记忆和语义记忆", "工作记忆"),
    ("部署环境是 Windows 11 上的 Python 3.12", "Python"),
]
UNRELATED_QUERY = "法式甜点烘焙配方"


def banner(title: str, char: str = "=", width: int = 72) -> None:
    print(f"\n{char * width}\n{title}\n{char * width}")


def mask(value: str | None) -> str:
    if not value:
        return "(未设置)"
    return f"{value[:4]}...{value[-4:]}" if len(value) > 10 else "***"


def main() -> int:
    parser = argparse.ArgumentParser(description="qwen3-embedding-0.6b 接入验证")
    parser.add_argument("--offline", action="store_true", help="只检查配置，不发网络请求")
    args = parser.parse_args()

    failures: list[str] = []

    banner("[1/4] 配置检查")
    load_dotenv_once()
    api_key = os.getenv("DASHSCOPE_API_KEY")
    model = os.getenv("HELLOAGENTS_MEMORY_EMBEDDING_MODEL", "qwen3-embedding-0.6b")
    base_url = os.getenv(
        "HELLOAGENTS_MEMORY_EMBEDDING_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    print(f"  DASHSCOPE_API_KEY : {mask(api_key)}")
    print(f"  model             : {model}")
    print(f"  base_url          : {base_url}")
    print(f"  .env 文件         : {'存在' if os.path.isfile('.env') else '不存在'}")

    if not api_key:
        print(f"\n{FAIL} 未找到 API key。请任选一种方式配置：")
        print("     1) 在项目根目录 .env 写入：DASHSCOPE_API_KEY=sk-xxxx")
        print("     2) PowerShell 临时设置：$env:DASHSCOPE_API_KEY = 'sk-xxxx'")
        return 1
    print(f"{PASS} 已读取到 API key")

    if args.offline:
        print(f"\n{INFO} --offline 模式：跳过真实网络调用")
        return 0

    # ---------------------------------------------------------------- 真实调用
    banner("[2/4] 真实 API 调用（单条 + 批量）")
    try:
        embedding = APIEmbedding(api_key=api_key, model=model, base_url=base_url, dimension=1024)
    except Exception as exc:
        print(f"{FAIL} 构造 APIEmbedding 失败：{exc}")
        return 1

    try:
        start = time.perf_counter()
        vector = embedding.embed("记忆系统需要支持中文语义检索")
        elapsed = (time.perf_counter() - start) * 1000
        norm = sum(value * value for value in vector) ** 0.5
        print(f"  单条维度 = {len(vector)}（期望 1024）")
        print(f"  向量范数 = {norm:.4f}（期望接近 1）")
        print(f"  前 3 维  = {[round(v, 5) for v in vector[:3]]}")
        print(f"  单条耗时 = {elapsed:.0f} ms")
        if len(vector) != 1024:
            failures.append(f"维度不符：{len(vector)} != 1024")
            print(f"{FAIL} 维度不是 1024，请同步调整 MemoryConfig.embedding_dimension")
        else:
            print(f"{PASS} 维度与数值校验通过")
    except Exception as exc:
        print(f"{FAIL} 单条调用失败：{type(exc).__name__}: {exc}")
        return 1

    try:
        start = time.perf_counter()
        vectors = embedding.embed_batch(["第一条", "第二条", "第三条"])
        elapsed = (time.perf_counter() - start) * 1000
        ok = len(vectors) == 3 and all(len(v) == len(vector) for v in vectors)
        print(f"  批量 3 条 -> 返回 {len(vectors)} 条，耗时 {elapsed:.0f} ms")
        print(f"{PASS if ok else FAIL} 批量调用{'正常' if ok else '异常'}")
        if not ok:
            failures.append("批量调用返回数量或维度不一致")
    except Exception as exc:
        print(f"{FAIL} 批量调用失败：{type(exc).__name__}: {exc}")
        failures.append(f"批量调用失败：{exc}")

    # ------------------------------------------------------------ 中文语义检索
    banner("[3/4] 中文子词检索（MemoryManager 默认装配）")
    manager = MemoryManager(MemoryConfig(sqlite_path=":memory:"))
    print(f"  管理器默认 embedding = {manager.embedding!r}")
    ids: dict[str, str] = {}
    for content, _keyword in MEMORIES:
        item = manager.add(content, memory_type=MemoryType.SEMANTIC)
        ids[content] = item.id

    for content, keyword in MEMORIES:
        hits = manager.search(keyword, memory_type="semantic", limit=3)
        top = hits[0] if hits else None
        matched = top is not None and top.item.id == ids[content]
        print(
            f"  query='{keyword}' -> {len(hits)} 条, "
            f"top={'命中正确' if matched else (top.item.content[:20] if top else '无')} "
            f"score={top.score:.4f}" if top else f"  query='{keyword}' -> 0 条"
        )
        if not matched:
            failures.append(f"中文检索未命中：query='{keyword}' 期望 '{content}'")

    unrelated = manager.search(UNRELATED_QUERY, memory_type="semantic", limit=3)
    if unrelated and unrelated[0].score > 0.5:
        print(f"  {FAIL} 不相关查询 '{UNRELATED_QUERY}' 得分过高：{unrelated[0].score:.4f}")
        failures.append("不相关查询被高分误召回")
    else:
        top_score = f"{unrelated[0].score:.4f}" if unrelated else "无结果"
        print(f"  {PASS} 不相关查询 '{UNRELATED_QUERY}' 未误召回（top={top_score}）")
    manager.close()

    # -------------------------------------------------------------------- 汇总
    banner("[4/4] 结果汇总")
    if failures:
        print(f"{FAIL} 共 {len(failures)} 项未通过：")
        for item in failures:
            print(f"   - {item}")
        return 1
    print(f"{PASS} 全部检查通过：qwen3-embedding-0.6b 接入正常，中文检索有效。")
    return 0


if __name__ == "__main__":
    sys.exit(main())