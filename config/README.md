# config 目录 —— 外部 API 调用配置索引

本项目的所有外部调用（LLM、嵌入、联网搜索、云存储）都在这个目录下能找到。
**先看这张表，再决定改哪个文件。** 本项目不再使用 `.env`：密钥直接明文写进
gitignored 的 `services.toml` / `provider.toml`（或用 `*_env` 指向系统环境变量）。

| 调用 | 端点/凭据配置在哪 |
|---|---|
| LLM 聊天 / 知识抽取（DeepSeek 等） | `provider.toml` → `[profiles.<name>]`（密钥 `api_key` 明文或 `api_key_env` 指向） |
| 记忆嵌入（云端 OpenAI 兼容：SiliconFlow / DashScope / …） | `services.toml` → `[embedding]`（`base_url`/`api_key`/`model`；默认值在 `constants.py`「嵌入服务」节） |
| 联网搜索（web.search / AnySearch） | `services.toml` → `[search]` |
| Qdrant Cloud | `services.toml` → `[qdrant]` |
| Neo4j Aura | `services.toml` → `[neo4j]` |
| 本地转发代理（Clash 等） | 云端 Qdrant / Neo4j 默认 `http://127.0.0.1:7890`；`services.toml` → `[proxy].url` 可覆盖；本机回环不走代理 |

## 规则

- **`services.toml` 是"所有外部 API 调用"的唯一集中配置**：端点、模型、凭据都写在这里，
  模板见 `services.example.toml`。密钥两种写法：`api_key = "明文"`（推荐，文件已
  gitignore）或 `api_key_env = "环境变量名"`（密钥留在系统环境变量里）。
- **优先级**：显式构造参数 > `services.toml`（仅此两层；历史上的 `.env` /
  `HELLOAGENTS_MEMORY_*` 环境变量优先层已删除）。
- **`provider.toml` 只管 LLM 聊天 profile**，不混入其他服务（其校验器只接受 http/https，
  塞不进 Neo4j 的 `bolt+s://`，schema 也对不上）。
- **`constants.py` 是入库默认值与运行开关的事实来源**；`services.toml` 只存本地部署值。
- 改了 `[embedding]` 的模型 = 换了向量空间（维度唯一开关就是 `model`）：先
  `scripts/migrate_to_cloud.py --recreate-collection` 重建集合，再跑
  `scripts/reindex_embeddings.py` 重灌记忆向量。

## 常用操作

```bash
# 首次接入云服务：复制模板并填写
copy config/services.example.toml config/services.toml

# 只放密钥（推荐）：在 .env 加一行，services.toml 里写 api_key_env 指向它
DEEPSEEK_API_KEY=sk-xxx              # 已在用
HELLOAGENTS_MEMORY_QDRANT_API_KEY=xxx
HELLOAGENTS_MEMORY_NEO4J_PASSWORD=xxx
```