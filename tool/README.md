# 单文件工具范式与调用链

把 [`tool_template.py`](tool_template.py) 复制成 `tool/` 下的新 `.py` 文件，修改输入模型、输出模型、`ToolSpec` 和 `execute()`，最后将 `TOOL_ENABLED` 设为 `True`。`Agent` 默认在构造时扫描 `tool` 包，因此不需要修改中央工具清单。

## 单文件协议

每个可发现模块必须同时提供：

```python
TOOL_ENABLED = True


class XxxInput(BaseModel): ...


class XxxOutput(BaseModel): ...


class XxxTool(BaseTool):
    spec = ToolSpec(
        name="namespace.tool_name",
        input_model=XxxInput,
        output_model=XxxOutput,
        # version、权限、副作用、超时、并发策略等
    )

    def execute(self, arguments: XxxInput) -> XxxOutput: ...


def create_tool() -> BaseTool:
    return XxxTool()
```

具体约束如下：

- 文件名不能以 `_` 开头，也不能叫 `base.py`；子目录不会作为单文件工具加载。
- `TOOL_ENABLED` 必须是真正的 `bool`。为 `False` 时发现器会记录 `disabled`，且不会调用工厂。
- `create_tool()` 必须能以零参数调用，并返回 `BaseTool` 实例。
- 工具名必须带命名空间，例如 `web.search`；运行时会为 OpenAI 兼容接口映射成 `web__search`。
- 输入和输出模型都必须配置 `ConfigDict(extra="forbid", strict=True)`。Pydantic 模型是 Schema、模型定义和运行时校验的唯一真相。
- `execute()` 只接收已经验证过的输入。外部 API 的原始响应也必须归一化为输出模型。
- 模块被导入时不能访问网络、写文件、启动线程或执行其他业务动作。依赖和配置应在 `create_tool()`/构造器中创建，实际工作放在 `execute()` 中。

## 项目更新日志工具

`update_log.py` 是项目内置的唯一更新日志写入工具，自动注册为
`system.update_log`。它将每次修改写成 SQLite `update_logs` 表中的一行，
使用 `AUTOINCREMENT` 生成从 1 开始的连续 `update_id`，并只返回新 ID、下一
ID、写入时间和系统名，不把历史正文带回模型上下文。

所有 AI 修改项目后都必须调用该工具一次。它是有副作用的本地写入工具；当前
`database.write` 只保留为权限元数据、不参与访问控制，运行环境仍需提供当前注册
代次的确认 key；详见项目根目录的
[`update_log_readme_first.md`](../update_log_readme_first.md)。数据库路径可用
`UPDATE_LOG_DB_PATH` 覆盖，默认是项目工作目录下的 `update_log.sqlite3`。

## 开关与自动发现

固定开关可以直接写：

```python
TOOL_ENABLED = True
```

也可以像 `search.py` 一样由环境变量控制：

```python
TOOL_ENABLED = os.getenv("MY_TOOL_ENABLED", "true").strip().casefold() not in {
    "0",
    "false",
    "no",
    "off",
}
```

环境变量应在构造 `Agent` 前设置。开关是“发现/注册开关”，不是正在运行进程中的强制卸载器；修改开关后应重建 Agent，或者以 `reload_modules=True` 重新扫描。已经注册的工具不会因为后续扫描发现模块关闭而被隐式删除。

`Agent` 默认采用容错扫描，错误模块会进入报告，但不会阻止其他合法工具注册：

```python
agent = MyAgent("demo")
report = agent.tool_discovery_report
print(report.as_dict())
```

需要让配置错误直接失败时：

```python
agent = MyAgent("demo", discovery_strict=True)

# 或在开发时重新加载所有模块；不同实现允许替换当前注册项。
report = agent.discover_tools(
    strict=True,
    reload_modules=True,
    replace=True,
)
```

也可以不使用 Agent，直接扫描：

```python
from core import ToolRegistry, discover_tools

registry = ToolRegistry()
report = discover_tools(registry, package="tool", strict=True)
```

发现报告的状态含义：

| `status` | 含义 |
| --- | --- |
| `registered` | 工厂创建的实例已注册 |
| `already_registered` | 相同类型和相同 `ToolSpec` 已经注册，未重复增加代次 |
| `disabled` | 开关关闭，工厂未执行 |
| `ignored` | 保留模块、下划线模块或子包被跳过 |
| `error` | 导入、协议、工厂或重复名称检查失败 |

严格扫描会在完成所有模块检查后抛出 `ToolDiscoveryError`，已成功的模块仍保持注册；可通过异常的 `report` 属性查看全部结果。

## 如何确认工具已经注册

```python
assert agent.is_tool_registered("web.search")
assert agent.is_tool_registered("web.search", version="1.0")

status = agent.tool_registration_status("web.search")
print(status)
# {
#   "name": "web.search",
#   "registered": True,
#   "version": "1.0",
#   "schema_hash": "...",
#   "generation": 1,
#   "implementation": "tool.search:SearchTool",
# }

record = agent.tool_discovery_report.for_tool("web.search")
```

`registered=True` 表示当前 Registry 中确实存在可执行实例。Repository 仅保存可检索的元数据，数据库中有记录不等于工具当前可执行。

## LLM 调用工具的函数级链路

模型不会直接调用 Python 函数。应用先把工具 JSON Schema 发给模型，模型返回结构化调用意图，运行时验证并执行，再把结果作为 `role=tool` 消息交给模型：

```text
调用方
└─ Agent.run_with_tools(messages, context)
   ├─ ToolRegistry.snapshot()
   ├─ Agent._definitions_for_registrations()
   │  └─ ToolSpec.input_schema -> OpenAI strict function schema
   ├─ LLM.complete(conversation, tools=definitions)
   │  └─ client.chat.completions.create(...)
   ├─ 模型返回 assistant.tool_calls[]
   ├─ parse_openai_tool_calls()
   │  ├─ provider 别名 -> 命名空间工具名
   │  └─ 绑定 schema version/hash + registry generation
   ├─ Agent.execute_tool_calls()
   │  └─ ToolExecutionManager.execute_batch()
   │     ├─ call_id/依赖图/注册快照校验
   │     ├─ Schema 版本和哈希校验
   │     ├─ 副作用确认校验（权限元数据当前不强制）
   │     ├─ input_model.model_validate(arguments, strict=True)
   │     ├─ 依赖分层、全局及单工具并发控制
   │     ├─ BaseTool.execute(validated_arguments)
   │     └─ output_model.model_validate(result, strict=True)
   ├─ ToolResult -> JSON -> role=tool + tool_call_id
   └─ 再次 LLM.complete(...)
      ├─ 若又返回 tool_calls：重复上述循环
      └─ 若返回普通 assistant.content：保存 history 并结束
```

一次调用的关键步骤是：

1. Agent 根据当前注册快照和 `tool_names` 选择可见工具；权限元数据不参与过滤。
2. 输入模型生成严格 JSON Schema，随对话发送给 LLM。
3. LLM 自己决定直接回答，或返回 `tool_calls`（名称、`call_id`、JSON 参数）。
4. Parser 只负责把提供商格式转为统一 `ToolCall`，并绑定模型所看到的注册代次；它不执行工具。
5. Runtime 在执行前检查重复 ID、依赖、工具存在性、注册代次、Schema、副作用确认和严格输入类型；权限元数据当前不参与拦截。
6. Runtime 按依赖层调度；同步实现进入工作线程，异步实现直接等待，并应用总并发与单工具并发限制。
7. 工具执行结果再次由输出模型校验。异常、超时或校验失败都转成统一 `ToolResult.error`。
8. Agent 按原 `call_id` 追加工具结果并再次请求模型，直到模型生成最终文本或超过 `max_rounds`。

如果启用 `defer_tool_loading=True`，第一轮只暴露 `system.tool_catalog`。模型先调用目录工具搜索/解析候选，Agent 的 `_load_catalog_result()` 再把选中的完整工具 Schema 加到下一轮；这减少大量工具同时占用上下文，但目录工具本身仍走同一套校验和执行链。

## 自动发现的内部链路

```text
Agent.__init__()
└─ Agent.discover_tools()
   └─ core.discover_tools()
      ├─ import tool 包
      ├─ pkgutil.iter_modules(tool.__path__)
      ├─ import 每个候选模块
      ├─ 检查 TOOL_ENABLED
      ├─ 检查并调用 create_tool()
      ├─ 校验 BaseTool + ToolSpec
      ├─ ToolRegistry.register()
      └─ ToolSpecRepository.save()（若注入 Repository）
```

自动发现只应扫描可信代码目录，因为 Python 导入本身会执行模块顶层代码。不要把用户上传的任意 `.py` 文件直接放入这个包。
