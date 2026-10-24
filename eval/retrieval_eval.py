"""检索效果评测：Recall@k / Precision@k / MRR / 相似度分布，支持分词器 A/B 对比。

评测集构造方式：每条查询配一个唯一标记串（marker），
ground truth = 语料中包含该 marker 的 chunk。
查询为人工撰写的转述式问句（不直接复制原文句子），因此不是字面匹配。
语料按 chunk_size=300、overlap=0 切分，保证一条答案只落在一个 chunk 内，避免跨块误判。

用法：
    python eval/retrieval_eval.py                 # 基线分词器 + CJK bigram 补丁，双语料对比
    python eval/retrieval_eval.py --both-baseline # 只跑基线
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from memory import MemoryConfig, MemoryManager          # noqa: E402
from memory import embedding as embedding_module       # noqa: E402
from memory.rag.pipeline import RAGPipeline            # noqa: E402

CHUNK_SIZE = 300
OVERLAP = 0
TOP_K = 5

CJK_RANGE = r"\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
_CJK_BLOCK = re.compile(rf"[{CJK_RANGE}]+", re.UNICODE)
_CJK_OR_WORD = re.compile(rf"[{CJK_RANGE}]+|[\w]+", re.UNICODE)


def cjk_bigram_tokenize(text: str) -> list[str]:
    """CJK 按二元组切分，拉丁/数字按词切分。

    原实现 re.findall(r"[\w]+", ...) 会把一整段连续中文当作单个 token，
    中文语料因此几乎无法检索。CJK bigram 是 Lucene CJKAnalyzer 采用的经典方案，
    不需要外部分词词典，且维度稳定。
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    tokens: list[str] = []
    for match in _CJK_OR_WORD.finditer(text.casefold()):
        tok = match.group(0)
        if _CJK_BLOCK.fullmatch(tok):
            if len(tok) == 1:
                tokens.append(tok)
            else:
                tokens.extend(tok[i:i + 2] for i in range(len(tok) - 1))
        else:
            tokens.append(tok)
    return tokens


legacy_baseline_tokenize = embedding_module.TFIDFEmbedding.tokenize
_ORIGINAL_TOKENIZE = embedding_module.TFIDFEmbedding.tokenize


def apply_tokenizer(mode: str) -> None:
    """Switch the embedding tokenizer for A/B measurement.

    The baseline is pinned to a *reference copy* of the pre-fix implementation
    rather than to the current source file. Once the fix lands in
    memory/embedding.py, "baseline" would otherwise silently become the patched
    tokenizer and the comparison would read as a flat 0 delta.
    """
    if mode == "cjk-bigram":
        embedding_module.TFIDFEmbedding.tokenize = staticmethod(cjk_bigram_tokenize)
    elif mode == "baseline":
        embedding_module.TFIDFEmbedding.tokenize = staticmethod(legacy_tokenize)
    else:
        embedding_module.TFIDFEmbedding.tokenize = _ORIGINAL_TOKENIZE


def legacy_tokenize(text: str) -> list[str]:
    """Pre-fix implementation, kept as the A/B baseline reference."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return re.findall(r"[\w]+", text.casefold(), flags=re.UNICODE)


EN_QUERIES = [
    ("How are new tool plugins discovered without editing a central list?", "no central registration list"),
    ("Which flag must be enabled inside a module for the agent to register it as a tool?", "TOOL_ENABLED = True"),
    ("What prevents skill bodies from fragmenting the prompt prefix cache?", "never fragment the prompt prefix cache"),
    ("What happens to the prompt cache key when a skill file is edited?", "changes the prompt cache key"),
    ("Do skill directories execute any code?", "Skill directories never execute code"),
    ("Which information about a tool is printed when it registers successfully?", "complete current registry"),
    ("What is omitted from runtime logs to keep the output readable?", "Raw tool arguments and return values are omitted"),
    ("Which Pydantic setting prevents unknown fields from being silently discarded?", 'extra="forbid"'),
    ("Is recommended_before_tools an execution dependency or only a hint?", "never an execution dependency or runtime gate"),
    ("What does the runtime verify before executing a tool call?", "checks the registration generation and schema version/hash"),
    ("Can independent tool calls run at the same time?", "Independent calls can run concurrently"),
    ("How are tools marked unsafe for parallelism scheduled?", "isolated into single-call layers"),
    ("Which class schedules one batch of validated tool calls?", "ToolExecutionManager"),
    ("Which tool exposes the catalog for summary search and versioned schema loading?", "system.tool_catalog"),
    ("How does catalog-first schema exposure shrink the first request payload?", "the first request contains every registered tool name and only the complete schema"),
    ("Where are project changes recorded and which tool does it?", "recorded compactly in SQLite"),
    ("What must be supplied after every real modification of the project?", "generation-bound side-effect confirmation"),
    ("How are malformed or stale native tool calls handled?", "returned to the model as structured tool errors"),
    ("How do I send a request without reusing conversation history?", "use_history=False"),
    ("Which parameter keeps the prompt cache alive longer?", 'prompt_cache_retention="24h"'),
    ("Where should per-profile API keys be configured?", "config/provider.toml"),
    ("Is provider_name still supported or is it deprecated?", "deprecated alias during migration"),
    ("Which adapter do all provider profiles use today?", "openai_compatible"),
    ("Which embedding providers can be swapped inside the memory package?", "DashScope/local-transformer/TF-IDF embedding services"),
    ("How are development dependencies installed and tests executed?", "python -m pip install -e '.[dev]'"),
]

ZH_QUERIES = [
    ("单文件工具的文件名有什么限制？", "base.py"),
    ("TOOL_ENABLED 必须是真正的布尔值吗？", "TOOL_ENABLED"),
    ("工具名为什么要带命名空间前缀？", "web.search"),
    ("输入输出模型需要什么严格配置？", "strict=True"),
    ("Qdrant 在什么时候自动创建 collection？", "collection_name"),
    ("没有提供 Neo4j uri 时存储会怎样降级？", "进程内回退"),
    ("DashScope 嵌入默认按多大批量分批？", "batch_size=10"),
    ("要解析 PDF 文档需要额外安装什么？", "pypdf"),
    ("memory 系列工具默认把数据写到哪个文件？", "MEMORY_DB_PATH"),
    ("变更日志数据库的路径怎么覆盖？", "UPDATE_LOG_DB_PATH"),
    ("工作记忆如何控制容量与过期？", "TTL"),
]

CORPORA = {
    "en": {"label": "README.md (English)", "docs": ["README.md"], "queries": EN_QUERIES},
    "zh": {
        "label": "tool/README.md + memory/README.md + skills/README.md (中文)",
        "docs": ["tool/README.md", "memory/README.md", "skills/README.md"],
        "queries": ZH_QUERIES,
    },
}


def norm(text: str) -> str:
    return " ".join(text.split()).lower()


def evaluate(corpus_key: str, tokenizer_mode: str) -> dict:
    spec = CORPORA[corpus_key]
    apply_tokenizer(tokenizer_mode)

    db = os.path.join(
        tempfile.gettempdir(), f"eval_{corpus_key}_{tokenizer_mode.replace('-', '_')}.sqlite3"
    )
    if os.path.exists(db):
        os.remove(db)

    with MemoryManager(MemoryConfig(sqlite_path=db)) as manager:
        pipe = RAGPipeline(manager)
        ids, texts = [], []
        for doc in spec["docs"]:
            path = os.path.join(ROOT, doc)
            if not os.path.exists(path):
                print(f"  [warn] missing doc: {doc}")
                continue
            for item in pipe.ingest_source(path, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
                ids.append(str(item.id))
                texts.append(str(item.content))
        index_of = {mid: i for i, mid in enumerate(ids)}
        normed = [norm(t) for t in texts]

        rows, unmapped = [], []
        for query, marker in spec["queries"]:
            gt = [i for i, t in enumerate(normed) if norm(marker) in t]
            if not gt:
                unmapped.append({"query": query, "marker": marker})
                continue
            results = pipe.retrieve(query, limit=TOP_K)
            got = [index_of.get(c.memory_id) for c in results]
            got = [g for g in got if g is not None]
            rank = next((i + 1 for i, g in enumerate(got) if g in gt), None)
            rows.append({
                "query": query,
                "ground_truth": gt,
                "top_k": got,
                "rank": rank,
                "top1_score": round(float(results[0].score), 4) if results else 0.0,
                "hit@1": rank == 1,
                "hit@3": rank is not None and rank <= 3,
                "hit@5": rank is not None,
                "precision@5": round(sum(1 for g in got if g in gt) / TOP_K, 4),
                "rr": round(1.0 / rank, 4) if rank else 0.0,
            })

    n = len(rows) or 1
    hit_rows = [r for r in rows if r["rank"]]
    return {
        "summary": {
            "corpus": spec["label"],
            "corpus_key": corpus_key,
            "tokenizer": tokenizer_mode,
            "documents": spec["docs"],
            "chunks": len(texts),
            "queries_total": len(spec["queries"]),
            "queries_used": len(rows),
            "queries_unmapped": len(unmapped),
            "recall@1": round(sum(r["hit@1"] for r in rows) / n, 4),
            "recall@3": round(sum(r["hit@3"] for r in rows) / n, 4),
            "recall@5": round(sum(r["hit@5"] for r in rows) / n, 4),
            "precision@5": round(sum(r["precision@5"] for r in rows) / n, 4),
            "mrr": round(sum(r["rr"] for r in rows) / n, 4),
            "top1_score_mean": round(statistics.mean(r["top1_score"] for r in rows), 4) if rows else 0,
            "top1_score_median": round(statistics.median(r["top1_score"] for r in rows), 4) if rows else 0,
            "top1_score_when_correct": round(statistics.mean(r["top1_score"] for r in hit_rows), 4) if hit_rows else None,
        },
        "rows": rows,
        "unmapped": unmapped,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--both-baseline", action="store_true")
    args = ap.parse_args()

    modes = ["baseline"] if args.both_baseline else ["baseline", "cjk-bigram"]
    out: dict = {}
    for mode in modes:
        for key in CORPORA:
            out[f"{key}::{mode}"] = evaluate(key, mode)
        apply_tokenizer("source")

    print()
    header = f"{'语料':<10}{'分词器':<14}{'Recall@1':>10}{'Recall@3':>10}{'Recall@5':>10}{'MRR':>8}{'top1均分':>10}"
    print(header)
    print("-" * len(header))
    for mode in modes:
        for key in CORPORA:
            s = out[f"{key}::{mode}"]["summary"]
            print(
                f"{key.upper():<10}{mode:<14}"
                f"{s['recall@1']:>10.4f}{s['recall@3']:>10.4f}{s['recall@5']:>10.4f}"
                f"{s['mrr']:>8.4f}{s['top1_score_mean']:>10.4f}"
            )

    if len(modes) > 1:
        print()
        print("提升幅度（cjk-bigram - baseline）：")
        for key in CORPORA:
            b = out[f"{key}::baseline"]["summary"]
            c = out[f"{key}::cjk-bigram"]["summary"]
            print(
                f"  {key.upper()}: Recall@1 {b['recall@1']:.4f} -> {c['recall@1']:.4f} "
                f"(+{(c['recall@1'] - b['recall@1']):.4f}) | "
                f"Recall@5 {b['recall@5']:.4f} -> {c['recall@5']:.4f} "
                f"(+{(c['recall@5'] - b['recall@5']):.4f}) | "
                f"MRR {b['mrr']:.4f} -> {c['mrr']:.4f}"
            )

    os.makedirs(os.path.join(ROOT, "eval"), exist_ok=True)
    with open(os.path.join(ROOT, "eval", "results.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("\nsaved eval/results.json")


if __name__ == "__main__":
    main()
