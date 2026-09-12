"""把 SimKGC / KG-BERT 三元组数据集导入知识星云记忆库。

SimKGC 数据格式（https://github.com/intfloat/SimKGC，WN18RR / FB15k237 / wiki5m）：
  data/<TASK>/train.txt | valid.txt | test.txt   每行: head<TAB>relation<TAB>tail
  data/<TASK>/entities.dict                      idx<TAB>entity_id
  data/<TASK>/relations.dict                     idx<TAB>relation
  data/<TASK>/wordnet-mlj12-definitions.txt      synset_id<TAB>lemma<TAB>definition（仅 WN18RR）

导入方式：逐行调用 ``MemoryManager.semantic.add_fact(subject, predicate, object)``，
走项目原生的"向量化 + 图结构"入库通道——每个三元组生成两个实体节点 + 一条引力桥边。

用法示例：
  python scripts/import_simkgc.py --dir data/WN18RR --limit 300 --domain WordNet
  python scripts/import_simkgc.py --dir data/FB15k237 --limit 500
  python scripts/import_simkgc.py --dir data/WN18RR --limit 100 --definitions wordnet-mlj12-definitions.txt

⚠️ 规模红线：前端 Canvas2D 渲染上限约 500 节点。WN18RR train 有 4 万余三元组，
FB15k237 有 27 万+；全量导入必然卡死前端。请务必用 --limit 限量导入（建议 200~500 条）。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 使 `from web.support import get_manager` 可用（脚本位于 scripts/ 子目录）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def load_definitions(path: str | Path) -> dict[str, str]:
    """解析 wordnet-mlj12-definitions.txt：synset_id -> 可读词（去掉 __lemma_POS_N 前缀）。"""
    mapping: dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        return mapping
    for line in p.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        synset_id = parts[0].strip()
        lemma = parts[1].strip()
        # __german_shepherd_dog_NN_1 -> german shepherd dog（去掉 POS 与序号两段尾巴）
        tokens = lemma.lstrip("_").split("_")
        if len(tokens) >= 2:
            tokens = tokens[:-2]
        readable = " ".join(t for t in tokens if t)
        if readable:
            mapping[synset_id] = readable
    return mapping


def parse_triples(txt_path: str | Path) -> list[tuple[str, str, str]]:
    """解析 train/valid/test.txt：每行 head<TAB>relation<TAB>tail。"""
    triples: list[tuple[str, str, str]] = []
    for line in Path(txt_path).read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        head, rel, tail = parts[0].strip(), parts[1].strip(), parts[2].strip()
        if head and rel and tail:
            triples.append((head, rel, tail))
    return triples


def main() -> int:
    parser = argparse.ArgumentParser(description="SimKGC 三元组数据集 → 知识星云记忆库")
    parser.add_argument("--dir", required=True, help="数据集目录（含 train.txt 等）")
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"],
                        help="导入哪个文件（默认 train）")
    parser.add_argument("--limit", type=int, default=300,
                        help="限量导入条数，防止节点数超过前端渲染红线（默认 300）")
    parser.add_argument("--offset", type=int, default=0,
                        help="从第 N 条开始（用于分批导入不同子集）")
    parser.add_argument("--definitions", default="", help="wordnet-mlj12-definitions.txt 路径（把数字 ID 映射成可读词）")
    parser.add_argument("--domain", default="SimKGC", help="节点领域名（星云的恒星系）")
    parser.add_argument("--source", default="", help="来源标注，默认取文件名")
    parser.add_argument("--db", default="", help="记忆库路径（默认与 Web 服务共用 memory.sqlite3）")
    args = parser.parse_args()

    data_dir = Path(args.dir)
    txt_path = data_dir / f"{args.split}.txt"
    if not txt_path.exists():
        print(f"[错误] 找不到 {txt_path}")
        return 1

    triples = parse_triples(txt_path)
    total = len(triples)
    selected = triples[args.offset : args.offset + args.limit]
    print(f"文件 {txt_path}: 共 {total} 条三元组，本次导入 {len(selected)} 条"
          f"（offset={args.offset}, limit={args.limit}）")

    # 可选：数字 ID -> 可读词
    defs = load_definitions(data_dir / args.definitions) if args.definitions else {}
    if args.definitions:
        print(f"定义映射加载: {len(defs)} 个词条")

    def label(eid: str) -> str:
        if defs and eid in defs:
            return f"{defs[eid]} [{eid}]"
        return eid

    # 与 Web 服务共用同一套记忆库配置与 embedding 降级逻辑（无 key 时本地哈希）
    if args.db:
        os.environ["MEMORY_DB_PATH"] = str(Path(args.db).resolve())
    from web.support import get_manager

    manager = get_manager()
    try:
        added = 0
        skipped = 0
        source = args.source or f"{args.split}.txt"
        for head, rel, tail in selected:
            subj, pred, obj = label(head), rel, label(tail)
            try:
                manager.semantic.add_fact(
                    subj, pred, obj,
                    metadata={"source": source, "source_document": str(txt_path.name),
                              "domain": args.domain, "evidence": "SimKGC 数据集导入"},
                )
                added += 1
            except ValueError:
                skipped += 1
        print(f"完成：新增 {added} 条事实（跳过 {skipped} 条非法行）")
        try:
            print(f"记忆库统计: {manager.stats()}")
        except Exception as exc:  # pragma: no cover - 统计仅作展示
            print(f"（统计不可用: {exc}）")
        return 0
    finally:
        manager.close()


if __name__ == "__main__":
    raise SystemExit(main())