# 03 · agents/ —— Agent 运行时

`agents/` 把 `core/` 的工具协议封装成「能和模型多轮对话」的智能体：`Agent` 是基座（provider 管理 + 原生 function calling 循环），`ReActAgent` 是文本协议变体，`LLM` 是 OpenAI 兼容客户端，`ProviderRegistry` 管多厂商 profile，`message_utils` 提供形状归一化工具函数。

包入口 `agents/__init__.py` 再导出 `Agent / LLM / ProviderProfile / ProviderRegistry / ReActAgent / parse_react_response`。

---

## 3.1 agents/message_utils.py —— SDK 响应归一化

背景：不同 provider SDK 返回 Pydantic 模型、dataclass、dict 混杂形态；`_field` 等转换原来在每个模块各抄一份，一个 provider 怪癖在一处修好、另一处仍坏。本模块统一收拢，且不 import 运行时（只翻译形状）。

| 函数 | 作用 |
| --- | --- |
| `field(value, key, default)` | mapping 或对象属性通用取值 |
| `safe_tool_name(value)` | 日志/错误占位用的工具名：非字符串回落 `unknown.tool`，截断 200 字符 |
| `safe_tool_call_error(error)` | 错误文本截断 1000 字符；空消息回落异常类名 |
| `tool_call_dict(item)` | SDK tool_call 对象 → `{id,type,function{name,arguments}}`，None 字段剔除 |
| `message_dict(message)` | SDK assistant 消息 → 纯 dict：`model_dump(exclude_none=True)` 优先；否则 dict；否则取 role/content/tool_calls；补默认 `assistant`；tool_calls 逐个转字典 |
| `result_json(result)` | ToolResult → JSON 字符串：`model_dump(mode="json")` 失败回落普通 dump，`default=str` 兜底 |

模块常量：`DEFAULT_TOOL_NAME="unknown.tool"`、`MAX_TOOL_NAME_CHARS=200`、`MAX_TOOL_ERROR_CHARS=1000`。

---

## 3.2 agents/llm.py —— OpenAI 兼容客户端

`_FINAL_ANSWER_MARKER_RE`：行首最终答案标记（英文/Final Answer、中文/最终答案/最终回答/最终回复，可加粗）——流式回显靠它决定何时开始打印。

`_FinalAnswerEchoer`：终端回显控制器。
- `mode="content"`（原生工具轮）：收到即打印；
- `mode="react_final"`：先静默缓冲，直到出现最终答案标记才把标记之后的内容流式打印；以工具调用结束的轮次什么都不打印；
- `feed / flush / _emit`：写入失败只记 debug，绝不影响主流程；`echoed` 标记供调用方检测「答案没到过终端」。

`_assemble_streaming_response(stream, on_first_chunk, echo_mode, echo_write)`：把流式响应重组为**与非流式完全同形**的 dict（`choices[0].message` + `finish_reason`）。要点：
- 按 `index` 归并流式 `tool_calls` 碎片（id/name/arguments 都是分段到达的，需拼接）；
- `finally` 里关闭流，防提前退出时泄漏 HTTP 连接；
- 附带 `stream_echoed` 字段标记答案是否已回显。

`LLM` 类（构造参数 `api_key / base_url / model / max_retries`）：
- 构造时三项配置缺一即抛「Configuration Error」；内部建 `OpenAI` 客户端，HTTP 层显式挂 `httpx.Client(proxy=services.toml[proxy] 或 None, trust_env=False)`（代理只认配置，不偷偷吃环境变量）；
- `complete(messages, model, temperature, timeout, stream, prompt_cache_key, prompt_cache_retention, **kwargs)`：严格校验 timeout/temperature/cache 字段；保留字（messages/model/stream/…）禁止覆盖；未设置的 cache 字段不下发（老网关兼容）；
- `complete_streaming(...)`：同上但强制 `stream=True`；保留字检查；回调首字计时；
- `think(...)`：兼容旧接口，`stream_response_bool` 控制流式，最终返回纯文本；
- `_message_content(response)`：从 dict 或对象两种形态取 `choices[0].message.content`；
- `stream_response(response)`：生成器逐段 yield 文本并打印；`finally` 关流。

---

## 3.3 agents/providers.py —— 多 Provider 配置

`load_project_dotenv(path=None)`：把仓库根 `.env` 读进 `os.environ`，**不覆盖已有环境变量**；只加载一次（`_DOTENV_LOADED` 哨兵）；支持 `export ` 前缀与成对引号剥离。背景：provider.toml 常用 `api_key_env=DEEPSEEK_API_KEY`，当初抽取链路不加载 .env，导致静默走空抽取器、零实体。

`ProviderProfile`（frozen dataclass）：`name / base_url / api_key / default_model / models(tuple) / api_key_env / adapter("openai_compatible") / tool_mode("native_strict")`。
- `api_url` 属性：归一化 URL 的公开别名；
- `public_info()`：可外泄的元数据，api_key 打码成 `***`。

`ProviderRegistry`：
- `default_config_path()`：优先 `provider.toml` → `providers.toml` → `provider.example.toml`（新检出无配置时仍可用注入假客户端）；
- `reload()`：读 TOML；文件缺失抛 FileNotFoundError；语法错抛 ValueError；
- `get(name)`：取 profile，带「可用列表」提示；
- `resolve_api_key(profile_name)`：明文 key 优先，否则解析 `api_key_env` 指向的环境变量（先 `load_project_dotenv()`）；都没有则抛错；
- `register_ephemeral(...)`：内存注入 profile（测试/遗留集成的逃生舱）；
- `profiles` 属性返回只读 MappingProxy。

`_parse_document`：要求 `[defaults]` 表 + 至少一个 `[profiles.<name>]` 表；校验 active_profile 存在。
`_parse_profile`：逐字段校验——adapter 只接受 `openai_compatible`；base_url 必须绝对 HTTP(S) URL（拒绝 bolt+s 等，避免与服务配置混用）；api_key 支持旧的 `api_key_env` 字段迁移（是合法变量名→当环境变量名，否则当误内联的密钥）；models 非空无重复且 default_model 在其中；tool_mode ∈ `{native_strict, native_loose, text_react, none}`。

---

## 3.4 agents/agent.py —— Agent 基座

### 类属性与构造

- `TOOL_MODE_PROTOCOLS`：tool_mode → 协议映射（native_strict/native_loose→native，text_react→react，none→None）；
- `default_tool_protocol()`：按 active_profile 的 tool_mode 解析；无 profile 时默认 native；
- `__init__(name, llm, provider_config, provider_registry, repository, auto_discover_tools=True, tool_package="tool", discovery_strict=False)`：
  - 建 `ProviderRegistry`（未注入时按默认路径加载）；
  - `self.tools = ToolRegistry()` 并**先注册 `system.tool_catalog`**（目录工具永远在册）；
  - `auto_discover_tools=True` 时立即发现 `tool` 包全部工具；
  - `ToolExecutionManager(self.tools)` 接管执行；
  - 提示缓存热区状态：`_frozen_manifest`（首个请求固化的、prompt 可见的工具指纹表）、`_hot_tools`（后注册工具的热区缓冲）、`cache_epoch`（只有冻结前缀结构变化才 +1）；
  - 兼容字段：`role / prompt / history / max_retries / providers`。

### 工具管理

| 方法 | 作用 |
| --- | --- |
| `register_tool(tool, replace)` | 注册并同步持久化 spec |
| `register_hot_tool(tool, replace)` | 注册即可执行，但 schema 只走请求尾部的热区块，不改写缓存前缀 |
| `unregister_tool(name)` | 注销；删冻结区工具会 `cache_epoch+1`（缓存前缀失效），删热区工具免费 |
| `discover_tools(...)` | 包扫描 + 同步，报告存 `tool_discovery_report` |
| `is_tool_registered / tool_registration_status` | 查询注册状态 |
| `execute_tool_calls(calls, context)` | 委托执行管理器 |

### Prompt 冻结与缓存键

- `_prompt_fingerprint(spec)`：`schema_hash#description`——只改描述也要算变化（描述会原文渲染进 schema 块）；
- `_sync_frozen_manifest()`：**首个请求前**一次性固化工具清单（registry + repository），之后的注册走热区，不碰稳定前缀；
- `_with_registered_tool_names(conversation)`：每次请求在最前插一条 system：「All registered tool names: …」（含 repository-only 的懒加载工具名）；
- `_default_prompt_cache_key(...)`：对「版本+profile+模型+模式+配置提示词+工具名集合+cache_epoch」做 SHA-256 取 48 位；**刻意排除用户文本、工具结果、懒加载 schema**（它们在稳定前缀之后，不碎片化缓存）；react 模式额外把两段协议指令并入摘要；
- `_validate_prompt_cache_key / _validate_prompt_cache_retention`：OpenAI 限制 key ≤64 字符、retention ∈ {in_memory, 24h}。

### 核心循环 `run_with_tools(...)`

参数：`messages, context, max_rounds, model, temperature, timeout, tool_names, profile_name, provider_name(迁移期别名), use_history, defer_tool_loading, prompt_cache_key, prompt_cache_retention, enable_prompt_cache, stream_echo`。

流程：
1. 解析执行目标 `_completion_target` → (LLM 客户端, 模型, history_key)；
2. 取历史前缀（**按 profile 隔离**）或配置提示词，拼上本次 messages；
3. 固化 frozen manifest；确定 `loaded_order`：指定 tool_names 则白名单；`defer_tool_loading=True` 首轮只带 `system.tool_catalog`（catalog-first）；否则全部；
4. `ToolLoop` 逐轮：从当前快照取注册表 → `_definitions_for_registrations` 生成 OpenAI 工具定义（**名字典序排序、目录工具排最前**，保证跨进程字节稳定以命中前缀缓存；`web.search`→`web__search`；检测别名冲突；超 64 字符拒绝）→ 请求模型（走 `_dispatch_model_call`）→ 无 tool_calls 即存历史并返回最终答案 → 有则逐个 `parse_openai_tool_calls`（失败不中断整批：`TOOL_NOT_EXPOSED`/`INVALID_TOOL_CALL` 作为工具结果回给模型自纠）→ `execute_tool_calls` 批量执行 → 每条结果以 `role=tool` 消息回填 → `defer_tool_loading` 时把 catalog 返回的 specs 加入 loaded_order。

### 其它成员

- `_dispatch_model_call`（静态）：优先 `complete_streaming`（真 LLM 流式，避免长生成撞读超时被 SDK 静默重试），其次 `complete`，再退化 `think`（自动剔除不支持的参数）；三种路径都打点耗时；
- `run_auto`：按 tool_mode 分流——native → `run_with_tools`；react → `run_with_react`（仅 ReActAgent 有）；none → 拒绝；
- `_load_catalog_result`：catalog 结果里命中的工具追加进 loaded_order；
- `_configured_prompt_messages`：把 `self.prompt`（system/user/assistant/tool）转成消息列表；
- `_completion_target`：注入 llm 优先（history_key=`__injected__`）；否则校验模型在 profile.models 内，按 (profile, model) 缓存 LLM 客户端；
- `set_active_profile / reload_provider_profiles / profile_info / list_profiles / profiles`：profile 管理（reload 会丢弃客户端缓存使新 URL/key 生效，历史仍按 profile 隔离）；
- `set_system_prompt / _set_prompt`。

### 模块级函数

- `_strict_function_schema(schema)`：Pydantic schema → OpenAI strict function 形态：递归删 `default`、对象节点强制 `additionalProperties=False` 且 `required=列全部字段`、拒绝任意键对象；
- `compress_saved_history(conversation)`：把超过 12000 字符的 Observation/tool 消息替换成「压缩桩」（保留工具名、原始大小、400 字符预览与重新查询提示），避免大结果被钉死在后续每轮请求里；
- `trim_saved_history(conversation, max_messages=60)`：超限时从头部裁旧轮，保留开头 system 块；被裁出孤儿的 Observation 也删掉（没有 Action 的 Observation 会污染上下文）；
- `_observation_stub(payload)`：生成压缩桩文本。

---

## 3.5 agents/react.py —— 文本协议 ReAct 智能体

### 正则与标记（模块级）

- `_REACT_MARKER_LOOKAHEAD`：所有合法行首标记（英文 thought/action/action input/observation/final answer + 中文 思考/行动/行动输入/观察/最终答案/最终回答/最终回复/工具）；
- `_FINAL_RE / _ACTION_RE / _ACTION_INPUT_RE / _XML_ANSWER_RE / _EXPLICIT_FINAL_RE`：大小写不敏感、容忍 `**` 加粗与全角冒号；Final/Action Input 用「直到下一个行首标记」的环视截断；
- `CATALOG_FIRST_REACT_INSTRUCTIONS`：catalog-first 模式追加的协议说明。

`ParsedReActResponse`（frozen dataclass）：`raw / thought / final_answer / action / arguments / error`；`is_final`、`has_action` 两个便捷属性。**解析绝不执行工具**，错误转为 Observation 让模型自我纠正。

### 解析函数

| 函数 | 行为 |
| --- | --- |
| `parse_react_response(text)` | 优先级：Final Answer 标记 → `<answer>...</answer>` 整包裹 → 占位符（`<answer>`/`[answer]` 单独一行→error）→ 无 Action 标记时**纯文本即最终答案**（provider 兜底）→ 有 Action 则必须有 Action Input 且能解成 JSON 对象 |
| `_extract_thought(text)` | 提取 Thought/思考段 |
| `_decode_action_input(payload)` | 委托 `loads_model_json(object_only=True)`；非对象返回可操作的纠正文案 |
| `_strip_code_fence(value)` | 剥掉 ``` 围栏 |
| `_coerce_string_scalars(arguments, schema)` | **文本协议专用**：按 schema 把 `"5"`→5、`"true"`→True；转不了的原样留下走正常校验错误。解决「文本模型把数字序列化成字符串→INVALID_ARGUMENTS→原payload再失败→循环烧轮」 |
| `_declared_schema_types(subschema)` | 取字段声明类型（type/anyOf） |
| `_run_sync(coro)` | 同步入口：`ThreadPoolExecutor(1)` 里 `asyncio.run`（不能在有事件循环的线程里 asyncio.run） |

### ReActAgent

构造：`lazy_tools=False` 默认（即发现即注册）；`lazy_tools=True` 且给了 repository 时走「只存元数据、按需加载实现」；`catalog_tool.repository_only = lazy_tools`（保证目录永远查得到全量元数据）。

重写的方法：
- `discover_tools`：lazy 时用 `scratch_registry + metadata_only=True` 扫一遍只写 repository，随后丢弃临时注册表；
- `register_tool`：lazy 时只持久化 spec（含 `implementation_ref`），不保留实例；
- `is_tool_registered / tool_registration_status`：lazy 时查 repository；
- `tool_confirmation_key(name)`：未加载工具给 `name@version#hash:1`（首次加载代次为 1），已加载走注册表真实代次；
- `run(query, **kwargs)`：同步便捷入口；
- `run_with_react(...)`：主循环（参数与 run_with_tools 对齐，`defer_tool_loading` 默认 True）：
  1. 组装 conversation（历史按 profile 隔离）；
  2. 每轮：`_with_tool_instructions` 生成带工具清单/可用范围/schema 的请求消息；`_hot_tools` 非空时把热区块追加到**消息尾部**（永远在缓存前缀之后）；
  3. 模型返回：**有原生 tool_calls** 走 `_execute_native_calls`（与原生协议同一条解析+执行路径）；否则 `parse_react_response` 解析；
  4. 最终答案：未标记答案且请求含「实时/查询/搜索/search…」等意图词时，最多纠正 3 次（`REACT_UNMARKED_ANSWER_RETRY_LIMIT`），仍不改才接受；畸形答案同样最多纠 3 次，超了直接 RuntimeError 停机（不烧完轮数预算）；
  5. 有 Action：`round_loop.record_call` 防重复 → `_execute_action` → 结果以 `Observation: {json}` 作为 user 消息回填 → catalog/lazy 结果驱动 loaded_order 扩展；
- `_should_require_tool_action(...)`：判断「裸答案」是否该被纠正（只有含实时/检索意图词的请求才强制走工具，闲聊放行）；
- `_initial_conversation`：历史/配置提示词 + 请求消息；
- `_with_tool_instructions(...)`：渲染 ReAct 指令 + 全部工具名 + 「Available tools」可调用子集 + 各工具 schema；lazy 时只渲染目录工具；
- `_hot_zone_lines`：热区块的文本行；
- `_ensure_tool_loaded(name)`：按 `implementation_ref`（`模块:类名`）import → 调 `create_tool()` → 校验 name/version/schema_hash 与仓储一致 → 注册；
- `_load_catalog_result / _load_catalog_observation`：catalog 命中 → 记录 schema 并追加 loaded_order（后者从 Observation JSON 里还原 ToolResult）；
- `_strict_react_schema`：复用 Agent 的 strict schema；
- `_unavailable_tool_error(...)`：区分「本次请求白名单摘除」（TOOL_NOT_ENABLED，告诉模型别重试）与「schema 未加载」（可经 catalog 解决）；
- `_response_message(response)`：取 choices[0].message；
- `_execute_action(...)`：Action 名规范化（支持 `a__b`→`a.b`）→ 懒加载按需装载 → 未知工具 UNKNOWN_TOOL → 不可用 → `_coerce_string_scalars` → 构造 ToolCall → `execute_tool_calls`；
- `_execute_native_calls(...)`：协议内兼容原生 function calling（懒工具直接调用也会先 ensure 加载），结果统一成 `Observation:` 消息；
- `_canonical_action_name(action, snapshot)`：把 provider 别名还原成命名空间名。
