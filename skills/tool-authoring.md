---
description: 当 AI 需要为项目新增、修改或审查 Agent 可调用工具（tool/ 下的单文件插件）时使用;覆盖单文件工具协议、TOOL_ENABLED 与 create_tool 工厂、Pydantic Input/Output 契约、ToolSpec 元数据决策、execute 实现规范、测试与发现验证,以及完成后按交付流程留痕。
version: 1.0.0
triggers: 写工具, 新增工具, 工具开发, 工具协议, ToolSpec, create_tool, TOOL_ENABLED, 工具模板, tool_template, 修改工具
enabled: true
---

# 工具编写（tool-authoring）

把"为 all_agent 编写新工具"固化为标准行为。新增、修改或审查 `tool/` 下的工具时遵循本技能；完成后必须走 `code-change-delivery` 技能交付。

## 何时使用

- 用户要求新增一个 Agent 能力（读写文件、执行命令、调 API、查库等）。
- 需要修改现有工具的行为、Schema 或修复缺陷。
- 需要审查一个工具是否符合单文件协议。

## 一、开发前置

1. 用 `fs.read_text` 读 `tool/tool_template.py`（完整模板与规范）和 `tool/README.md`（协议细节）。
2. 用 `fs.read_dir` 查看 `tool/` 现有文件，避免重名与职责重叠。
3. 明确工具的单一职责、输入来源、稳定输出、外部依赖、是否写副作用、幂等性与并发特性。缺少会影响安全性的关键信息时先向用户确认；不得把写操作标成 read、不得虚构 API、字段或权限。
4. 如需参考真实实现：`tool/search.py`（外部 API + 环境变量配置）、`tool/current_time.py`（无参数）、`tool/shell_run.py`（execute 副作用）、`tool/git_commit_push.py`（复杂执行链路）、`tool/_shared.py`（共享助手，非发现目标）。

## 二、单文件协议（硬性约束）

新工具 = `tool/` 下一个新的 `.py` 文件，必须满足：

1. 文件名不能以 `_` 开头，不能叫 `base.py`，不要放进子目录。
2. 模块定义布尔常量 `TOOL_ENABLED`，只有严格等于 `True` 时发现器才会调用工厂。
3. 模块定义零参数 `create_tool() -> BaseTool`，返回工具实例（不是类/字典/None）。
4. 模块导入期只能做导入、声明类型、定义常量；禁止网络、写文件、启动线程、操作数据库。依赖与配置在 `create_tool()`/构造器中创建，业务在 `execute()` 中执行。
5. 一个文件只提供一个发现工厂；复杂能力拆成多个命名清晰的小工具。

## 三、Input/Output 模型（Pydantic 是唯一真相）

- 顶层 Input 与 Output 必须继承 `BaseModel` 并配置 `ConfigDict(extra="forbid", strict=True)`；嵌套模型建议同样配置。
- 每个字段写准确 `description`（语义、单位、格式、限制）并设置合理长度/范围；可空字段显式写成 `str | None`。
- 不要使用 `dict[str, ...]` 任意键结构（strict schema 不允许 additionalProperties）；需要自由对象时参考 `tool/search.py` 的 `WithJsonSchema` 写法。
- Output 必须稳定、精简、可序列化；不返回 SDK 对象、HTTP Response、未限制的大文本或密钥。
- `execute()` 直接返回 Output 实例（或可被 Output 严格校验的 dict）；不要自行拼装 `ToolResult`。

## 四、ToolSpec 元数据决策

| 字段 | 决策 |
| --- | --- |
| name | `namespace.tool_name`，ASCII 字母/数字/下划线，至少一个点；映射为 OpenAI 侧 `namespace__tool_name` 后 ≤64 字符；不写版本 |
| description | 面向 LLM：何时使用、完成什么、不能做什么，≤2000 字符；不放密钥/动态状态 |
| version | ≤32 字符；Schema、约束、权限、副作用或结果语义变化时必须升级 |
| input_model / output_model | 指向本文件定义的 Pydantic 模型类 |
| side_effect | 精确 `"read"` 视为无写副作用；`"write"`/`"execute"`/`"external_write"` 等都会要求确认。诚实标注（如 shell.run 用 execute） |
| permissions | 公开工具通常 `()`（兼容/审计元数据，运行时不做授权过滤） |
| timeout_seconds | 有限正数，是单次执行截止时间；网络/数据库客户端内部超时必须 ≤ 此值 |
| idempotent | 相同参数重复执行是否安全；决定错误是否标 retryable |
| parallel_safe | 是否可并发执行；共享可变状态、保序需求时为 False（如 update_log、shell.run） |
| max_concurrency | 该工具的最大并发（None 表示用全局限制） |
| tags | 稳定能力关键词（如 ("fs","file","read")），供目录检索 |
| recommended_before_tools | 可选；建议的命名空间工具名，仅供模型提示，不构成运行时依赖 |

## 五、execute() 实现规范

- 只接收已验证的 Input；不要重新解析 LLM 文本或 JSON。
- 外部 API/数据库原始响应必须清洗并归一化为 Output 模型。
- 失败时抛出明确异常（运行时转成结构化错误回传模型自愈）；异常消息不得包含密钥、隐私或未清洗的输入。
- 需要上下文时声明可选 `context: ExecutionContext | None = None` 参数（运行时注入）。
- 同步方法由运行时放入工作线程；需要原生异步可定义 `async def execute(...)`。
- 工具实现本身必须设置 I/O 超时并避免无限阻塞（同步线程超时后 Python 无法强制终止）。

## 六、测试与验证（强制）

在 `tests/test_<tool>.py` 中至少覆盖：

1. 发现协议：`TOOL_ENABLED` 是 bool、`create_tool()` 返回实例。
2. 合法输入 → 合法输出（断言 Output 字段）。
3. 非法输入被拒：缺字段/错类型/多余字段 → Pydantic ValidationError（strict）。
4. 写/执行工具在无确认时经 `ToolExecutionManager` 报 `CONFIRMATION_REQUIRED`；持有确认 key 后放行（参考 `tests/test_update_log.py`、`tests/test_git_status.py`）。
5. 真实依赖用注入替身、临时目录、本地回环服务隔离，禁止真实网络（参考 `tests/test_search.py`、`tests/test_git_status.py`）。

运行验证：

```text
python -m pytest tests/test_<tool>.py -q
python -m pytest -q          # 全量回归
python -m ruff check tool/<file>.py
```

发现与注册验证（临时脚本或 `python -c`）：

```text
from core import ToolRegistry, discover_tools
registry = ToolRegistry()
report = discover_tools(registry, package="tool", strict=True)
print(report.for_tool("<namespace>.<tool_name>").as_dict())  # status == "registered"
```

或构造 Agent 后：`agent.is_tool_registered("<namespace>.<tool_name>")` 为 True；`agent.tool_registration_status(...)` 显示 registered/version/schema_hash/generation。

## 七、交付收尾（强制）

新工具是"实际修改"，完成后必须走 `code-change-delivery` 技能：

1. `system.update_log` 留痕（files 含新工具文件与测试文件，action=added）。
2. 更新 `update_log_readme_first.md` 索引。
3. `git.commit_push` 提交（消息 `update-log-<id>: ...`，含 `tool/<file>.py`、`tests/test_<file>.py`、`update_log.sqlite3` 与索引文件）。
4. 推送远端。

## 验收清单

- [ ] `TOOL_ENABLED=True`，`create_tool()` 零参返回实例，发现报告 status=registered 且无 error
- [ ] Input/Output 均 `extra="forbid" + strict=True`，字段有 description 与边界
- [ ] side_effect 如实标注；写/执行工具已在确认路径下验证
- [ ] 新增测试通过，全量 pytest 与 ruff 通过
- [ ] 已按 `code-change-delivery` 完成留痕、提交、推送