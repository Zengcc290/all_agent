# config 目录 —— 外部 API 调用配置索引

本项目的所有外部调用（LLM、嵌入、联网搜索、云存储）都在这个目录下能找到。
**先看这张表，再决定改哪个文件。** 密钥直接明文写进 gitignored 的
`services.toml` / `provider.toml`（或用 `*_env` 指向系统环境变量）。

## 两套配置：本地版 vs 云端版

| 模式 | 模板（可提交） | 生效文件（gitignored） | 典型形态 |
|---|---|---|---|
| 本地版 | `services.local.toml` + `provider.local.toml` | `services.toml` + `provider.toml` | 嵌入强制 `hash` 离线；聊天走本机网关；Qdrant/Neo4j 留空 = 内存回退 |
| 云端版 | `services.cloud.toml` + `provider.cloud.toml` | `services.toml` + `provider.toml` | SiliconFlow 嵌入、Qdrant Cloud、Neo4j Aura、DeepSeek 聊天，经本地代理出网 |

**切换方式**（只复制文件、不碰代码，加载器仍只认 `services.toml` / `provider.toml`）：

```bash
# 切到本地版（生效文件已存在时提示，用 --force 覆盖）
python scripts/switch_config.py local

# 切到云端版
python scripts/switch_config.py cloud

# 查看当前生效模式
python scripts/switch_config.py status
```

切完后打开生效文件填入真实密钥（或用 `api_key_env` 指向 `.env`），重启服务生效。
模板里 `replace-with-*` 是占位符，直接使用会让对应服务保持“未配置”状态。

| 调用 | 端点/凭据配置在哪 |
|---|---|
| LLM 聊天 / 知识抽取（DeepSeek 等） | `provider.toml` → `[profiles.<name>]`（密钥 `api_key` 明文或 `api_key_env` 指向） |
| 记忆嵌入（云端 OpenAI 兼容：SiliconFlow / DashScope / …） | `services.toml` → `[embedding]`（`base_url`/`api_key`/`model`；默认值在 `constants.py`「嵌入服务」节） |
| 联网搜索（web.search / AnySearch） | `services.toml` → `[search]` |
| Qdrant Cloud / 本地 Qdrant | `services.toml` → `[qdrant]` |
| Neo4j Aura / 本地 Neo4j | `services.toml` → `[neo4j]` |
| 本地转发代理（Clash 等） | 云端 Qdrant / Neo4j 默认 `http://127.0.0.1:7890`；`services.toml` → `[proxy].url` 可覆盖；本机回环不走代理 |

## 规则

- **`services.toml` 是"所有外部 API 调用"的唯一集中配置**：端点、模型、凭据都写在这里，
  模板见 `services.local.toml` / `services.cloud.toml`（旧模板 `services.example.toml` 保留兼容）。
  密钥两种写法：`api_key = "明文"`（推荐，文件已 gitignore）或
  `api_key_env = "环境变量名"`（密钥留在系统环境变量 / `.env` 里）。
- **优先级**：显式构造参数 > `services.toml`（仅此两层；历史 `.env` /
  `HELLOAGENTS_MEMORY_*` 环境变量优先层已删除）。
- **`provider.toml` 只管 LLM 聊天 profile**，不混入其他服务（其校验器只接受 http/https，
  塞不进 Neo4j 的 `bolt+s://`，schema 也对不上）。
- **`constants.py` 是入库默认值与运行开关的事实来源**；`services.toml` 只存部署值。
- 改了 `[embedding]` 的模型 = 换了向量空间（维度唯一开关就是 `model`）：先
  `scripts/migrate_to_cloud.py --recreate-collection` 重建集合，再跑
  `scripts/reindex_embeddings.py` 重灌记忆向量；本地版模板默认
  `provider = "hash"` 离线嵌入，不涉及向量空间迁移。

## 常用操作

```bash
# 首次接入：选一套模板作为起点
python scripts/switch_config.py cloud   # 或 local，然后用 --force 覆盖

# 只放密钥（推荐）：在 .env 加一行，生效文件里写 api_key_env 指向它
DEEPSEEK_API_KEY=sk-xxx
