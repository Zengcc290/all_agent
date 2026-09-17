# config 目录 —— 外部 API 调用配置索引

本项目的所有外部调用（LLM、嵌入、联网搜索、云存储）都在这个目录下能找到。
**先看这张表，再决定改哪个文件。**

| 调用 | 端点/凭据配置在哪 | 密钥本体在哪 |
|---|---|---|
| LLM 聊天 / 知识抽取（DeepSeek 等） | `provider.toml` → `[profiles.<name>]`（`api_key_env` 指向 .env） | `.env` → `DEEPSEEK_API_KEY` |
| 记忆嵌入（DashScope / Gemini / SiliconFlow / 网关） | `services.toml` → `[embedding]`；或 `.env` 的 `HELLOAGENTS_MEMORY_EMBEDDING_*`（默认值在 `constants.py`「嵌入服务」节） | `.env` → `DASHSCOPE_API_KEY` / `GEMINI_API_KEY` / `SILICONFLOW_API_KEY` |
| 联网搜索（web.search / AnySearch） | `services.toml` → `[search]`；或 `.env` 的 `SEARCH_BASE_URL` / `SEARCH_API_KEY` | `.env` → `SEARCH_API_KEY` |
| Qdrant Cloud | `services.toml` → `[qdrant]`；或 `.env` 的 `HELLOAGENTS_MEMORY_QDRANT_URL` / `..._QDRANT_API_KEY` | `.env` → `HELLOAGENTS_MEMORY_QDRANT_API_KEY` |
| Neo4j Aura | `services.toml` → `[neo4j]`；或 `.env` 的 `HELLOAGENTS_MEMORY_NEO4J_URI` / `_USERNAME` / `_PASSWORD` | `.env` → `HELLOAGENTS_MEMORY_NEO4J_PASSWORD` |
| 嵌入网关 SSH 隧道（部署信息，非 API） | `.env` → `EMBEDDING_TUNNEL_KEY` / `EMBEDDING_TUNNEL_HINT` / `EMBEDDING_BASE_URL` | `.env`（不进 services.toml） |

## 规则

- **`services.toml` 是"所有外部 API 调用"的集中配置**：端点、模型、凭据指向都写在这里，
  模板见 `services.example.toml`。密钥两种写法：`api_key = "明文"`（文件已 gitignore）
  或 `api_key_env = "环境变量名"`（推荐，密钥仍放 `.env`）。
- **优先级**：显式构造参数 > 环境变量 > `services.toml`。`.env` 里已有的值照旧生效，
  services.toml 只在两处都没设置时才补上——所以你可以在两处各放一半，不会打架。
- **`provider.toml` 只管 LLM 聊天 profile**，不混入其他服务（其校验器只接受 http/https，
  塞不进 Neo4j 的 `bolt+s://`，schema 也对不上）。
- **`constants.py` 是入库默认值的事实来源**；`services.toml` / `.env` 只存本地部署值。
- 改了 `[embedding]` 的提供方/端点/模型 = 换了向量空间，要重跑
  `scripts/reindex_embeddings.py`。

## 常用操作

```bash
# 首次接入云服务：复制模板并填写
copy config/services.example.toml config/services.toml

# 只放密钥（推荐）：在 .env 加一行，services.toml 里写 api_key_env 指向它
DEEPSEEK_API_KEY=sk-xxx              # 已在用
HELLOAGENTS_MEMORY_QDRANT_API_KEY=xxx
HELLOAGENTS_MEMORY_NEO4J_PASSWORD=xxx
```