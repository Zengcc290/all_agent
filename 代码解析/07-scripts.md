# 07 · scripts/ —— 质量门禁与运维脚本

`scripts/` 提供四个可独立执行的入口脚本（都不在 `web`/`agent` 运行时链路上，属于开发/运维工具）。它们都先 `sys.path.insert(0, str(ROOT))` 以便在任意位置通过 `python scripts/xxx.py` 运行。

---

## 7.1 scripts/check.py —— 一键质量门禁

**用途**：`python scripts/check.py`，先跑 `ruff check .` 再跑完整 `pytest -q`，两个门都必须过；第一个失败立即停止并返回它的退出码（可作 pre-commit hook / CI 步骤）。

| 名称 | 含义 |
| --- | --- |
| `ROOT` | 脚本所在目录的上级（仓库根） |
| `GATES` | `((“ruff”, (“-m”, “ruff”, “check”, “.”)), (“pytest”, (“-m”, “pytest”, “-q”)))`——按顺序的两道门 |
| `main()` | 逐道门用**当前解释器**（`sys.executable`，不是硬编码路径）子进程执行，`cwd=ROOT`；返回 0=全过，否则第一个失败门的退出码。全部通过打印 `==> all checks passed` |

只在任意激活的虚拟环境（Windows 或 POSIX）里可用，无第三方依赖。

---

## 7.2 scripts/migrate_storage.py —— memories 回填 documents/chunks + 向量真值归位

**用途**：把历史版本存在 `memories` 表里的分块回填到 `documents`/`chunks` 两张真值源表，并让向量真值归位。

```text
.venv\Scripts\python.exe scripts\migrate_storage.py [--db memory.sqlite3] [--dry-run] [--no-backup]
```

设计要点（方案 P5 / D1 / D3）：

- 迁移前默认用 SQLite **在线备份**写出 `{db}.bak.pre_migration`（普通文件复制可能抓到半写的页）；
- 幂等：`documents` 按 `document_id`、`chunks` 按 `chunk_id` 存在即跳过；
- `memories` 表本身不动：事实条目与 working/episodic/perceptual 全部保留，只新增两张表；
- **`memories.embedding` 只在重嵌入成功之后才清空**——先清空再失败会既丢向量又没有新向量（Qdrant 是唯一向量真值，D3），所以嵌入网关不可用时整段跳过并保留原值；
- Neo4j 未开启时不重放图，只报告 0。

| 函数 | 行为 |
| --- | --- |
| `backup_database(db)` | `sqlite3.Connection.backup()` 在线一致性备份 → `{db.name}.bak.pre_migration` |
| `chunk_documents(items)` | 把 semantic 层的 chunk 行按 `document_id` 分组，组内按 `chunk_index` 排序 |
| `document_record(document_id, chunks)` | 由 chunk 行重建一文档的真值行：`raw_text` = chunk 文本按 `\n` 连接，`char_start/char_end` 就在这份文本上计算（偏移只相对切片它的文本有意义，方案 2.3）；`status="parsed"`（向量是否补齐由重嵌入阶段决定） |
| `is_fact(item)` | 是否 `(subject, predicate, object)` 事实 |
| `migrate(manager, dry_run, repository)` | 逐文档 upsert 真值行；事实条目在配置了 `neo4j_uri` 时用 `semantic.add_fact` **复用真实写入路径**重放图边（保证边属性与在线抽取完全一致）；返回 `{documents, chunks, facts_kept, graph_replayed}` |
| `reindex(manager, dry_run)` | 云端嵌入未配置时跳过（`{skipped: "no_cloud_embedding"}`）；一次批量请求 `embed_batch` 重嵌入全部条目 → 逐条写 `vector_store.upsert/upsert_chunk`（有 chunk 行走窄 payload）→ 文档状态 `vectorized` → **最后**才 `item.embedding = None` 并 upsert 回库（真正清掉 memories.embedding） |
| `main(argv)` | 解析参数 → 校验库存在 → 备份 → 建 `MemoryManager` → migrate → （未加 `--no-reindex`）reindex → 打印 JSON 报告 |

---

## 7.3 scripts/migrate_to_cloud.py —— chunks 用当前配置重向量化并入库云端 Qdrant

**场景**：嵌入模型切换后（如 qwen 网关 → SiliconFlow BAAI/bge-m3），SQLite 里已落库的 chunks 必须用新模型重算向量并写进云端 Qdrant；`documents` 行回填（此前可能缺失），`chunks.vector_status` 置 `indexed`。

```text
.venv\Scripts\python.exe scripts\migrate_to_cloud.py [--db memory.sqlite3] [--no-backup] [--recreate-collection]
```

- 云端连接参数全部来自 `config/services.toml`（`[embedding]`/`[qdrant]`）；
- 幂等：按 `chunk_id` upsert，重复执行不产生重复点；
- 退出码 0 = 全部成功；任何一步失败会抛错并保留现场（向量状态仍为 pending）；
- 输出：分阶段耗时（嵌入 / 上传 / 状态 / 核对）与入库统计。

`main(argv)` 流程（分四阶段）：

1. **嵌入**：`embedding.embed_batch(chunk.text)`，校验向量数与 chunk 数一致、维度一致；构造 `EmbeddingIdentity(model, dim)` 与 SQLite 锁比对，不一致且未加 `--recreate-collection` → 打印 `EmbeddingLockMismatch` 并退出码 2；
2. **（可选）重建集合**：`--recreate-collection` 时按新维度 `QdrantVectorStore.recreate_collection`（向量是 SQLite 真值的投影，删集合只丢投影；内存回退则无需重建）；
3. **上传**：逐 chunk `vector_store.upsert_chunk(...)` 幂等落云端；
4. **回填 + 状态**：按文档重组 `documents` 行（`raw_text` 以 `\n` 连接）、`set_status("vectorized")`、逐 chunk `set_chunk_vector_status("indexed")`；随后核对云端落库点数（`list_ids(limit=10000)`），最后 `set_embedding_lock(model, dimension)` 落新锁。

模块还自带 `backup_database(db)`（与前一个脚本同款在线备份）。

---

## 7.4 scripts/reindex_embeddings.py —— 全部记忆条目重向量化

**用途**：把记忆库中全部条目用当前配置的云端嵌入服务重新向量化。每家提供方/模型产出互不兼容的向量空间，切换模型后旧向量必须重算。

```text
.venv\Scripts\python.exe scripts\reindex_embeddings.py [--confirm-rebuild]
```

- 没有云端配置时**拒绝执行**（回落到了离线 `HashEmbedding`，离线向量与云端空间不兼容，硬灌只会污染现有向量），退出码 1；
- 向量写入「当前配置的向量存储」：配了 `[qdrant]` 就是云端 Qdrant 集合，否则内存回退——与 Web/Agent 运行时同一份投影，不会写丢；
- 维度变更（如 1024 → 4096）请先 `python scripts/migrate_to_cloud.py --recreate-collection` 重建集合并重灌 chunks，再运行本脚本重灌记忆条目向量。

`main()` 流程：

1. `MemoryConfig.from_config()` + `make_default_embedding(config)`；是 `HashEmbedding` 就拒绝；
2. `db_path = Path(default_sqlite_path())`；
3. 批量重索引时放宽嵌入超时到至少 180s（原本就是 180：一次请求慢不该判失败）；
4. `MemoryManager(config, embedding=embedding)` + `DocumentRepository(db_path)`；
5. `apply_embedding_lock(manager, repo, confirm_rebuild=args.confirm_rebuild)`：不一致且未确认 → 打印并退出码 2，提醒加 `--confirm-rebuild`；
6. 逐条 `embedding.embed(item.content)` → `document_store.upsert` + `vector_store.upsert`（**单条失败不中断整批**，记 `类型: 消息`，退出码 2）；
7. 打印 `向量存储：<类名>`、`待重索引：N 条`、`完成：X 条成功，Y 条失败`。

---

## 7.5 三个脚本的选用场景

| 场景 | 用哪个 |
| --- | --- |
| 老库升级：memories 里的分块回填两张真值源表 + 向量归位 | `migrate_storage.py` |
| 换嵌入模型：chunks 用新模型重向量化并入库云端 Qdrant | `migrate_to_cloud.py --recreate-collection`（维度变了时） |
| 换嵌入模型：记忆条目向量（非 chunk）也重算 | `reindex_embeddings.py --confirm-rebuild` |
| 交作业/CI 前自查 | `check.py` |

三者共同的安全约束：**向量都是 SQLite 真值的投影**——任何重建都以「先确认、后重建、失败保留现场」为原则，绝不静默混用两套向量空间。
