# agents/agent.py

## 一、这个文件是干什么的

这个文件是整个 Agent 运行时的「外壳（shell）」层，负责把模型客户端、工具注册表、工具执行器、提示词缓存策略和对话历史串成一条可运行的链路。文件里定义了抽象基类 `Agent`（继承 `ABC`），它本身不实现具体的 `run()` 协议，而是提供一整套通用能力：provider 配置解析与客户端缓存、工具注册/热注册/注销、工具发现、把工具 schema 转成 OpenAI function-calling 格式、以及最核心的 `run_with_tools()` 原生工具调用循环。除了类之外，文件底部还有几个模块级函数：`_strict_function_schema()` 负责把 Pydantic 风格的 JSON Schema 规范化成 OpenAI strict 模式，`compress_saved_history()` / `trim_saved_history()` / `_observation_stub()` 负责在持久化历史前压缩超大的工具返回载荷并按上限裁剪历史。真实运行时里，子类（例如 ReActAgent）会继承 `Agent` 并复用这些方法；`run_auto()` 会根据当前 provider profile 声明的 `tool_mode` 自动把请求路由到原生工具协议或 ReAct 协议。文件还承担「提示词前缀字节稳定」这一职责：通过 `_frozen_manifest`、`_hot_tools`、`cache_epoch` 三个状态，把工具清单和工具契约固定在可缓存的 system 前缀里，而把后来注册的工具放到请求尾部，以便复用 OpenAI 的 prefix/KV 缓存。

## 二、函数与类逐条详解

### `class Agent(ABC)` （第 61 行）
- **作用**：这是本文件唯一的类，也是整个 agents 包的抽象基类。它把「一次对话该怎样驱动模型与工具」这件事拆成可复用的公共部分，同时把「具体用什么协议跑」留给子类。它内部持有 provider 注册表与激活 profile、每个 profile 的 LLM 客户端缓存、每个 profile 的历史缓存、工具注册表 `ToolRegistry`、工具目录工具 `ToolCatalogTool`、工具执行管理器 `ToolExecutionManager`，以及提示词缓存相关的冻结清单与热区清单。类里同时定义了工具模式到协议的映射常量 `TOOL_MODE_PROTOCOLS`，把 profile 上声明的 `native_strict` / `native_loose` / `text_react` / `none` 翻译成 `native` / `react` / `None` 三种运行时协议。它被使用的方式通常是：子类继承它、在 `__init__` 里传入 llm 或 provider 配置、注册/发现工具，然后调用 `run_with_tools()`（或经 `run_auto()` 间接调用）完成一轮完整的工具调用对话。
- **参数**：类本身没有参数；它继承自 `abc.ABC`，因此带有抽象方法 `run`，不能被直接实例化。
- **返回**：不适用（类定义）。
- **内部流程**：类体中先声明类属性 `TOOL_MODE_PROTOCOLS`（第 66 行，类型为 `Mapping[str, str | None]`，映射关系为 `native_strict -> "native"`、`native_loose -> "native"`、`text_react -> "react"`、`none -> None`），随后按顺序定义方法：`default_tool_protocol`、`_save_history`、`_dispatch_model_call`、`run_auto`、`__init__`、抽象方法 `run`、`register_tool`、`register_hot_tool`、`unregister_tool`、`_prompt_fingerprint`、`_sync_frozen_manifest`、`discover_tools`、`is_tool_registered`、`tool_registration_status`、`execute_tool_calls`、`tool_definitions`、`_definitions_for_registrations`、`_openai_tool_name`、`run_with_tools`、`_load_catalog_result`、`_configured_prompt_messages`、`_with_registered_tool_names`、`_validate_prompt_cache_key`、`_validate_prompt_cache_retention`、`_default_prompt_cache_key`、`_completion_target`、`set_system_prompt`、`_set_prompt`。
- **异常/边界**：作为抽象基类，直接 `Agent(...)` 实例化会在创建时报 `TypeError`（因为 `run` 是 `@abstractmethod`）；类属性 `TOOL_MODE_PROTOCOLS` 中的 `None` 值语义是「使用 profile 声明的 tool_mode 对应协议」，只有显式声明 `tool_mode = "none"` 的 profile 才会真正禁用工具。
- **同文件关系**：类内部的方法互相调用形成完整闭环：`run_auto` 调用 `default_tool_protocol` 与 `run_with_tools`；`run_with_tools` 调用 `_validate_prompt_cache_key`、`_validate_prompt_cache_retention`、`_completion_target`、`_configured_prompt_messages`、`_sync_frozen_manifest`、`_default_prompt_cache_key`、`_definitions_for_registrations`、`_dispatch_model_call`、`_with_registered_tool_names`、`_save_history`、`execute_tool_calls`、`_openai_tool_name`、`_load_catalog_result`；`_save_history` 调用模块级 `compress_saved_history` 与 `trim_saved_history`；`register_hot_tool` 调用 `register_tool` 与 `_prompt_fingerprint`。

### `default_tool_protocol(self) -> str | None` （第 73 行）
- **作用**：解析当前激活的 provider profile 到底声明了哪种工具协议，是「协议路由」的单一事实来源。它从 `self.provider_registry.profiles` 里取出 `self.active_profile` 对应的 profile，再用类常量 `TOOL_MODE_PROTOCOLS` 把 profile 的 `tool_mode` 字符串翻译成运行时协议名。这样调用方（例如 `run_auto`）不需要了解 profile 的原始配置写法，只要拿到 `"native"`、`"react"` 或 `None` 就能决定分支。当注册表里找不到对应 profile 时（例如手工构造了注册表但没有 profile，或者 profile 名失效），它保守地回退到 `"native"`，保证旧调用方式仍然可用。
- **参数**：无显式参数；隐式使用 `self.provider_registry`（`ProviderRegistry`，提供 `.profiles` 字典）和 `self.active_profile`（`str`，当前激活的 profile 名）。
- **返回**：返回 `str | None`。正常返回 `"native"`、`"react"` 之一；当 profile 的 `tool_mode` 映射到 `None`（即 `tool_mode = "none"`）时返回 `None`；当 profile 不存在时返回 `"native"`；当 profile 存在但 `tool_mode` 的取值不在映射表中时，`Mapping.get` 的默认值 `"native"` 生效，也返回 `"native"`。
- **内部流程**：第一步调用 `self.provider_registry.profiles.get(self.active_profile)` 取 profile；第二步判断 `profile is None`，是则直接返回 `"native"`；第三步执行 `self.TOOL_MODE_PROTOCOLS.get(profile.tool_mode, "native")` 得到协议并返回。整个过程没有任何 I/O 或副作用，也不修改任何状态。
- **异常/边界**：自身不主动抛异常。边界情况包括：profile 缺失（回退 `"native"`）、`tool_mode` 取值未知（回退 `"native"`）、`tool_mode` 为 `"none"`（返回 `None`，由调用方决定如何处理）。
- **同文件关系**：只被 `run_auto` 调用；不调用本文件中的其它函数。它依赖类属性 `TOOL_MODE_PROTOCOLS`。

### `_save_history(self, history_key: str, conversation: list[dict[str, Any]]) -> None` （第 87 行）
- **作用**：把本轮结束时的完整对话写进「按 profile 分区」的历史缓存，并且在写入前做两件必要的加工：压缩超大的工具载荷、按消息条数上限裁剪。之所以需要它，是因为存下来的历史会在下一轮被当作 prompt 前缀重新发给模型，如果不压缩，一次读取类工具返回的几十 KB 数据会被永久钉在之后每一次请求里，最终拖慢或撑爆上下文窗口；如果不裁剪，长期存活的 agent 每轮都要重发整段对话，token 成本无上限增长。它还同步更新 `self.history`（把压缩后的内容再浅拷贝一份），让外部观察者看到的历史与内部缓存保持一致。它是 `run_with_tools` 在模型不再发出工具调用、即将返回最终答案前调用的收尾动作。
- **参数**：`history_key`（`str`）是历史分区的键，通常由 `_completion_target` 返回，可能是 `"__injected__"`（注入了 llm 的场景）或某个 provider profile 名；`conversation`（`list[dict[str, Any]]`）是本轮完整消息列表，包含前缀、用户消息、assistant 消息以及 tool 结果消息。
- **返回**：无返回值（`None`）。副作用是写入 `self._profile_histories[history_key]` 并整体替换 `self.history`。
- **内部流程**：先调用模块级 `compress_saved_history(conversation)` 得到压缩副本，再调用 `trim_saved_history(...)` 对压缩结果做条数裁剪，然后把结果赋给 `self._profile_histories[history_key]`，最后用列表推导 `[dict(item) for item in compressed]` 生成一份新的字典浅拷贝列表赋给 `self.history`。注意赋值给 `_profile_histories` 的是裁剪后的列表对象本身，而 `self.history` 是它的逐元素拷贝。
- **异常/边界**：自身不抛异常；`trim_saved_history` 保证裁剪后至少保留开头的 `system` 块，并且会丢弃被切断后孤立在队首的 `Observation:` 消息。若 `conversation` 为空列表，压缩与裁剪都会原样返回空列表，`self.history` 被置为空列表。若消息里 `content` 不是字符串（例如多模态结构），压缩逻辑会走 `dict(message)` 原样保留分支。
- **同文件关系**：调用模块级函数 `compress_saved_history` 与 `trim_saved_history`；被 `run_with_tools` 在模型返回无工具调用的最终答案时调用。

### `_dispatch_model_call(completion_llm: Any, messages: list[dict[str, Any]], options: dict[str, Any], *, round_number: int, echo_mode: EchoMode | None = None) -> Any` （第 100 行，`@staticmethod`）
- **作用**：执行「一次模型请求」，并且优先选择流式传输通道。它存在的原因非常具体：非流式请求会卡死工具循环——网关在完整补全生成完毕前不发送任何字节，长时间生成会触发读超时，而 OpenAI SDK 在超时后会静默重试同一个请求，造成重复计费和重复工具调用。因此对于真实的 `LLM` 客户端，它调用 `complete_streaming` 并在内部拼装成完整响应；对于注入的测试替身和旧式 `think` 风格包装器，则保留它们原有的 `complete` / `think` 通道。两条路径返回的响应形状被刻意保持一致，调用方无需区分。它同时负责采集「首包时间」和「完成时间」两个模型性能埋点。
- **参数**：`completion_llm`（`Any`）是模型客户端对象，可以是真实 `LLM`、测试替身或任意提供 `complete`/`think` 的对象；`messages`（`list[dict[str, Any]]`）是本次要发送的消息列表（已包含注入的工具清单 system 消息）；`options`（`dict[str, Any]`）是请求选项，典型键为 `model`、`temperature`、`timeout`、`stream`、`tools`、`prompt_cache_key`、`prompt_cache_retention`；`round_number`（`int`，仅关键字）是当前工具循环轮次，用于日志；`echo_mode`（`EchoMode | None`，仅关键字，默认 `None`）是流式回显模式，`run_with_tools` 会传 `"content"` 或 `None`。
- **返回**：返回模型客户端的原始响应对象（`Any`），形状与 OpenAI ChatCompletion 兼容（可被 `_field` 按 `choices[0].message` 取值）。具体分支：若存在可调用的 `complete_streaming` 且 `options["stream"]` 为 `False`，返回流式方法拼装后的响应；否则若有可调用的 `complete`，返回 `complete(...)` 的响应；否则若 `think` 可调用，返回 `think(...)` 的响应；三者都不满足时抛 `TypeError`。
- **内部流程**：第一步记录 `model_started_at = time.perf_counter()`；第二步用 `getattr(completion_llm, "complete_streaming", None)` 探测流式方法，若可调用且 `options.get("stream", False) is False`，则先把 `options` 中除 `"stream"` 之外的键过滤成 `streaming_options`，再创建一个空列表 `first_chunk_at` 与嵌套回调 `_log_first_chunk`（它把当前 `perf_counter()` 追加进列表），然后以 `messages`、`on_first_chunk=_log_first_chunk`、`echo_mode=echo_mode` 以及 `**streaming_options` 调用 `stream_method`；若 `first_chunk_at` 非空，则调用 `log_model_first_chunk(round_number, first_chunk_at[0] - model_started_at)` 上报首包延迟；随后调用 `log_model_completed(round_number, time.perf_counter() - model_started_at)` 上报总耗时并返回响应。第三步，若流式分支未命中，则探测 `complete`，可调用时直接 `complete(messages, **options)`，记录完成日志后返回。第四步，若前两者都没有，探测 `think`；不可调用则抛 `TypeError("llm must provide a complete() or complete_streaming() method")`。第五步，对 `think` 做参数适配：把 `options` 过滤成只含 `temperature`、`timeout`、`prompt_cache_key`、`prompt_cache_retention` 的 `think_options`；用 `inspect.signature(think).parameters` 取形参表（若抛 `TypeError`/`ValueError` 则退化为空字典 `{}`）；用 `any(... kind is inspect.Parameter.VAR_KEYWORD ...)` 判断是否接受 `**kwargs`；若形参中有 `stream_response_bool` 或接受 `**kwargs`，则补上 `think_options["stream_response_bool"] = False`；若不接受 `**kwargs`，则把形参中不存在的 `prompt_cache_key`、`prompt_cache_retention` 从 `think_options` 中剔除；最后调用 `think(messages, **think_options)`，记录完成日志并返回。
- **异常/边界**：客户端三者都不可调用时抛 `TypeError`；`inspect.signature` 失败时被捕获并退化为「无已知形参」的保守适配；`options` 中缺少 `"stream"` 键时 `options.get("stream", False)` 返回 `False`，即默认走流式；若显式传 `stream=True`，则跳过流式分支直接走 `complete`（若 `complete` 不存在则落到 `think` 适配逻辑）。该方法是同步函数，由 `run_with_tools` 通过 `asyncio.to_thread` 放到线程池执行，避免阻塞事件循环。
- **同文件关系**：调用嵌套函数 `_log_first_chunk`；被 `run_with_tools` 通过 `asyncio.to_thread` 调用。它使用外部库函数 `core.activity_log.log_model_first_chunk` 与 `log_model_completed` 做埋点。

### `_log_first_chunk() -> None` （第 129 行，嵌套在 `_dispatch_model_call` 内）
- **作用**：这是一个流式首包回调闭包，专门用来在「模型返回第一个数据块」的瞬间打一个时间戳。它被传给 `complete_streaming` 的 `on_first_chunk` 参数，从而让上层能够计算出「首包延迟」这一对交互体验最敏感的指标。之所以用列表而不是直接赋值给外层变量，是因为闭包只能读取而不能重新绑定外层变量，用 `append` 可以绕过这个限制并同时保留「只记录第一次」的语义（后续调用会继续追加，但调用方只读索引 0）。
- **参数**：无参数。
- **返回**：无返回值（`None`）。副作用是把 `time.perf_counter()` 追加到外层闭包变量 `first_chunk_at` 列表中。
- **内部流程**：仅一步——执行 `first_chunk_at.append(time.perf_counter())`。
- **异常/边界**：无特殊处理；若被多次调用，列表会累积多个时间戳，但 `_dispatch_model_call` 只取 `first_chunk_at[0]`，因此结果仍然是真正的首包时刻。
- **同文件关系**：被 `_dispatch_model_call` 定义并作为回调传给模型客户端的 `complete_streaming`；不调用本文件中的其它函数。

### `async run_auto(self, messages: str | list[dict[str, Any]], context: ExecutionContext | None = None, **kwargs: Any) -> str` （第 175 行）
- **作用**：这是「provider 无关」的统一入口，让调用方不必知道自己面对的是原生 function-calling 还是文本 ReAct 协议。它读取当前激活 profile 声明的协议：`native_strict` / `native_loose` 走 `run_with_tools`，`text_react` 走子类的 `run_with_react`，`none` 则直接拒绝请求。它还承担了一个易用性职责：如果调用方传的是裸字符串查询（`messages` 是 `str`），它会自动包装成 `[{"role": "user", "content": messages}]`，这样在原生协议下调用方不需要自己拼消息结构。它通常被 Web 层或上层编排代码当作「跑一轮对话」的默认方法使用。
- **参数**：`messages`（`str | list[dict[str, Any]]`）要么是一句用户输入字符串，要么是完整的消息列表；`context`（`ExecutionContext | None`，默认 `None`）是工具执行上下文，为 `None` 时由下层 `run_with_tools` 创建默认上下文；`**kwargs`（`Any`）是透传参数，会被原样转发给 `run_with_react` 或 `run_with_tools`，例如 `max_rounds`、`model`、`temperature`、`timeout`、`tool_names`、`profile_name`、`use_history`、`defer_tool_loading`、`prompt_cache_key`、`stream_echo` 等。
- **返回**：返回 `str`，即模型最终的文本回答。`react` 协议下返回的是 `run_with_react` 的结果；`native` 协议下返回 `run_with_tools` 的结果。
- **内部流程**：第一步调用 `self.default_tool_protocol()` 取得 `protocol`；第二步若 `protocol is None` 则抛 `ValueError`，提示当前 profile 通过 `tool_mode = 'none'` 禁用了工具；第三步若 `protocol == "react"`，用 `getattr(self, "run_with_react", None)` 取方法并检查 `callable`，不可调用则抛 `TypeError("tool_mode 'text_react' requires a ReActAgent instance")`，可调用则 `await react_runner(messages, context, **kwargs)` 并直接返回；第四步若 `messages` 是 `str`，包装成单条 user 消息；第五步 `await self.run_with_tools(messages, context, **kwargs)` 并返回。
- **异常/边界**：profile 声明 `tool_mode = "none"` 时抛 `ValueError`；声明 `text_react` 但当前实例没有 `run_with_react`（例如直接用 `Agent` 的子类而非 ReActAgent）时抛 `TypeError`；字符串消息在 `react` 分支下不会被包装（因为包装发生在 react 分支之后），由 `run_with_react` 自行处理；其余参数校验异常由下游 `run_with_tools` 抛出。
- **同文件关系**：调用 `default_tool_protocol` 与 `run_with_tools`；可能调用子类提供的 `run_with_react`（本文件未定义）；不调用其它模块级函数。

### `__init__(self, name: str, *, llm: LLM | None = None, provider_config: str | None = None, provider_registry: ProviderRegistry | None = None, repository: ToolSpecRepository | None = None, auto_discover_tools: bool = True, tool_package: str | ModuleType = "tool", discovery_strict: bool = False) -> None` （第 203 行）
- **作用**：构造一个 Agent 实例，并把运行所需的全部协作者装配起来。它做四类事情：一是对入参做严格的类型/非空校验，尽早暴露配置错误；二是建立 provider 体系（复用传入的 `ProviderRegistry`，或按 `provider_config` 新建一个），并把 `active_profile` 记录下来；三是搭建工具体系——创建 `ToolRegistry`、创建并注册 `ToolCatalogTool`（工具目录工具，用于延迟加载场景下让模型先看到目录）、可选地把目录工具的 spec 存进仓库、可选地执行一次工具自动发现、最后创建 `ToolExecutionManager`；四是初始化提示词、历史、重试次数以及提示词缓存热更新所需的三个状态（`_frozen_manifest`、`_hot_tools`、`cache_epoch`）。它是所有子类初始化的必经路径。
- **参数**：`name`（`str`，必填）agent 名称，必须是非空字符串（仅空白也视为非法）；`llm`（`LLM | None`，仅关键字，默认 `None`）注入的模型客户端，一旦提供，`_completion_target` 会优先使用它并跳过 provider 解析；`provider_config`（`str | None`，仅关键字，默认 `None`）传给 `ProviderRegistry` 的配置名/路径，仅在未提供 `provider_registry` 时生效；`provider_registry`（`ProviderRegistry | None`，仅关键字，默认 `None`）外部注入的 provider 注册表，必须严格是 `ProviderRegistry` 实例；`repository`（`ToolSpecRepository | None`，仅关键字，默认 `None`）工具 spec 仓库，用于持久化与延迟加载；`auto_discover_tools`（`bool`，仅关键字，默认 `True`）是否在构造时立即扫描工具包；`tool_package`（`str | ModuleType`，仅关键字，默认 `"tool"`）工具包名或模块对象；`discovery_strict`（`bool`，仅关键字，默认 `False`）传给 `discover_tools` 的严格模式开关。
- **返回**：无返回值（`None`）。构造完成后实例具备：`name`、`llm`、`provider_registry`、`active_profile`、`_profile_clients`、`_profile_histories`、`repository`、`tool_package`、`tools`、`catalog_tool`、`tool_discovery_report`、`execution_manager`、`role`、`prompt`、`history`、`max_retries`、`_frozen_manifest`、`_hot_tools`、`cache_epoch`。
- **内部流程**：先做五组校验：`name` 必须是 `str` 且 `name.strip()` 非空，否则 `ValueError`；`repository` 必须是 `ToolSpecRepository` 或 `None`，否则 `TypeError`；`auto_discover_tools` 必须是 `bool`，否则 `TypeError`；`discovery_strict` 必须是 `bool`，否则 `TypeError`；`tool_package` 必须是 `ModuleType`，或者是非空字符串，否则 `TypeError`。随后赋值 `self.name`、`self.llm`，校验 `provider_registry` 类型（必须是 `ProviderRegistry` 或 `None`），然后 `self.provider_registry = provider_registry if provider_registry is not None else ProviderRegistry(provider_config)`，并设 `self.active_profile = self.provider_registry.active_profile`。接着初始化 `self._profile_clients: dict[tuple[str, str], LLM] = {}`（键为 `(profile, model)`）与 `self._profile_histories: dict[str, list[dict[str, Any]]] = {}`。然后建工具体系：`self.repository = repository`、`self.tool_package = tool_package`、`self.tools = ToolRegistry()`、`self.catalog_tool = ToolCatalogTool(self.tools, repository)`、`self.tools.register(self.catalog_tool)`；若 `self.repository is not None` 则 `self.repository.save(self.catalog_tool.spec, replace=True)`；`self.tool_discovery_report = None`；若 `auto_discover_tools` 为真则调用 `self.discover_tools(strict=discovery_strict)`；最后 `self.execution_manager = ToolExecutionManager(self.tools)`。收尾阶段设置兼容字段与方法表 `self.role = ["user", "assistant", "system", "tool"]`、`self.prompt: dict[str, str] = {}`、`self.history: list[dict[str, Any]] = []`、`self.max_retries = DEFAULT_MAX_RETRIES`，以及缓存状态 `self._frozen_manifest = None`、`self._hot_tools = {}`、`self.cache_epoch = 0`。
- **异常/边界**：`name` 为空或全空白抛 `ValueError`；`repository`、`provider_registry`、`auto_discover_tools`、`discovery_strict`、`tool_package` 类型不符分别抛 `TypeError`；`ProviderRegistry(provider_config)` 内部可能因配置非法抛异常（本文件不处理）；`discover_tools` 在 `strict=True` 时可能因发现失败抛异常。值得注意的边界：第 254-256 行的注释声明 `providers` 是给旧原型调用方保留的只读兼容视图，但本文件构造函数里并没有任何代码给 `self.providers` 赋值，因此该属性在本文件中实际不存在，只有注释层面的说明。
- **同文件关系**：调用 `discover_tools`；间接依赖 `_prompt_fingerprint`、`_sync_frozen_manifest`（通过后续调用）；创建的对象 `self.catalog_tool`、`self.execution_manager`、`self.tools` 被本文件几乎所有其它方法使用；`self.max_retries` 被 `_completion_target` 使用。

### `run(self, query: str) -> str` （第 273 行，`@abstractmethod`）
- **作用**：这是抽象基类留给子类的唯一强制契约，代表「用某个具体协议跑一次查询并返回答案」。本文件不提供实现，只声明签名，子类（如 ReActAgent）必须覆盖它，否则实例化会失败。它的存在保证了所有 Agent 子类在「同步单次查询」这个最基础的用法上接口一致，上层代码可以在不知道具体协议的情况下调用 `agent.run(query)`。
- **参数**：`query`（`str`）用户查询文本。
- **返回**：声明为 `str`，具体由子类实现决定。
- **内部流程**：方法体只有 `raise NotImplementedError`，作为抽象方法的占位。
- **异常/边界**：未被覆盖时，实例化阶段 `ABCMeta` 就会阻止创建对象（`TypeError`）；若子类显式调用 `super().run(...)`，会抛 `NotImplementedError`。
- **同文件关系**：本文件内无调用方，也无对本文件其它函数的调用；与 `run_auto`、`run_with_tools` 是并列的入口概念，但 `run_auto` 并不调用 `run`。

### `register_tool(self, tool: BaseTool, *, replace: bool = False) -> None` （第 277 行）
- **作用**：把一个工具正式注册进运行时工具表，并且在配置了仓库时把它的 spec 一并落库。这是最基础的注册入口，`register_hot_tool` 也是先通过它完成真正的注册动作。它让工具立刻变成可执行、可被 `snapshot()` 看到、可被 `tool_definitions()` 导出的状态。
- **参数**：`tool`（`BaseTool`）工具实例，必须提供 `.spec`；`replace`（`bool`，仅关键字，默认 `False`）是否允许覆盖同名已注册工具。
- **返回**：无返回值（`None`）。副作用是写入 `self.tools` 以及可能的 `self.repository`。
- **内部流程**：第一步 `self.tools.register(tool, replace=replace)`；第二步若 `self.repository is not None`，则 `self.repository.save(tool.spec, replace=True)`——注意这里对仓库始终传 `replace=True`，与入参 `replace` 无关。
- **异常/边界**：同名工具已存在且 `replace=False` 时，异常由 `ToolRegistry.register` 抛出（本文件不捕获）；`tool` 缺少 `spec` 属性时会在第二步或第一步抛出属性错误；仓库保存失败时异常向上传播。注意本方法不会更新 `_frozen_manifest` / `_hot_tools` / `cache_epoch`，因此直接调用它注册的工具在已有冻结前缀的情况下不会出现在尾部热区（需要 `register_hot_tool` 才会登记）。
- **同文件关系**：调用 `self.tools.register` 与 `self.repository.save`；被 `register_hot_tool` 调用；被 `__init__` 间接用于注册 `catalog_tool`（`__init__` 中直接调的是 `self.tools.register`）。

### `register_hot_tool(self, tool: BaseTool, *, replace: bool = False) -> None` （第 282 行）
- **作用**：注册一个「热更新工具」，让它在不改写已缓存的提示词前缀的前提下被模型看见。工具本身立刻完全可执行，但它的 schema 只在请求尾部的热区块里被宣传，而不是插进稳定的 system 前缀——这样才能保住 OpenAI 的 prefix/KV 缓存命中率。如果还没有冻结前缀（即还没有发出过第一次请求），它就直接成为首次冻结前缀的一部分，不做额外处理。它是运行期动态加工具（例如插件热插拔）的标准入口。
- **参数**：`tool`（`BaseTool`）要注册的工具实例；`replace`（`bool`，仅关键字，默认 `False`）是否覆盖同名工具。
- **返回**：无返回值（`None`）。副作用是注册工具，并可能写入或清除 `self._hot_tools[name]`。
- **内部流程**：第一步调用 `self.register_tool(tool, replace=replace)` 完成实际注册与落库；第二步取 `name = tool.spec.name`，并计算 `fingerprint = self._prompt_fingerprint(tool.spec)`；第三步若 `self._frozen_manifest is None`（尚无冻结前缀）则直接 `return`，让该工具自然进入将来的冻结前缀；第四步若 `self._frozen_manifest.get(name) == fingerprint`（即该工具已经存在于冻结前缀且渲染内容完全一致），则 `self._hot_tools.pop(name, None)` 把它从热区移除并返回——避免重复宣传；第五步否则把 `self._hot_tools[name] = fingerprint` 记入热区。
- **异常/边界**：注册阶段的异常同 `register_tool`（同名冲突、缺少 `spec`）；工具 spec 缺少 `schema_hash` 或 `description` 时 `_prompt_fingerprint` 会抛 `AttributeError`；本方法不会修改 `cache_epoch`，这是刻意设计——热区变化不应改变缓存路由键。
- **同文件关系**：调用 `register_tool` 与 `_prompt_fingerprint`；读取 `self._frozen_manifest` 与 `self._hot_tools`；与 `unregister_tool`、`_sync_frozen_manifest` 共同维护同一套缓存状态。

### `unregister_tool(self, name: str) -> None` （第 300 行）
- **作用**：从运行时工具表和提示词分区中彻底移除一个工具，并正确处理缓存失效：如果被移除的工具属于冻结前缀，就必须让缓存的提示词前缀失效，因此递增 `cache_epoch`；如果它只在热区，移除它是零成本的，不需要动缓存命名空间。当 agent 还没有发出过任何请求（`_frozen_manifest is None`）时，只需要注销并顺手清掉可能存在的热区记录。它保证了「工具下线」这一操作在缓存语义上是正确的。
- **参数**：`name`（`str`）要移除的工具名，必须是非空字符串。
- **返回**：无返回值（`None`）。副作用是修改 `self.tools`、`self._frozen_manifest`、`self._hot_tools`，并可能递增 `self.cache_epoch`。
- **内部流程**：第一步校验 `name` 是非空字符串，否则抛 `ValueError("tool name must be a non-empty string")`；第二步 `self.tools.unregister(name)`；第三步若 `self._frozen_manifest is None`，则 `self._hot_tools.pop(name, None)` 后返回；第四步若 `name in self._frozen_manifest`，则 `self._frozen_manifest.pop(name)` 并 `self.cache_epoch += 1`，然后返回；第五步否则（只在热区）`self._hot_tools.pop(name, None)`。
- **异常/边界**：`name` 为空或非字符串抛 `ValueError`；工具不存在时 `self.tools.unregister` 的行为由 `ToolRegistry` 决定（本文件不捕获其异常）；注意它只改冻结清单而不触碰 `self.repository`，即仓库里的记录不会被删除。
- **同文件关系**：调用 `self.tools.unregister`；读取/修改 `_frozen_manifest`、`_hot_tools`、`cache_epoch`；与 `register_hot_tool`、`_sync_frozen_manifest`、`_default_prompt_cache_key` 共享同一套缓存状态。

### `_prompt_fingerprint(spec: Any) -> str` （第 320 行，`@staticmethod`）
- **作用**：为一个工具的 spec 计算「提示词指纹」，用来判断重新注册同一个工具是否真的改变了模型可见的提示词文本。它刻意把 `schema_hash` 和 `description` 拼在一起：只用 `schema_hash` 会漏掉「只改了描述文字」的编辑，而描述是被逐字渲染进 schema 块的，因此必须一起纳入。只有当拼接结果完全相同时，`register_hot_tool` 才会认为这次重注册是无操作，从而避免无意义地污染热区。
- **参数**：`spec`（`Any`）工具规格对象，需具备 `schema_hash` 与 `description` 两个属性（通常是 `ToolSpec`）。
- **返回**：返回 `str`，格式为 `f"{spec.schema_hash}#{spec.description}"`。注意这里直接用了 `#` 作为分隔符，不做转义，因此理论上若 `schema_hash` 内含 `#` 存在极小概率的歧义，但 `schema_hash` 是哈希串，实践中不会出现。
- **内部流程**：单步 f-string 拼接并返回，无任何副作用。
- **异常/边界**：`spec` 缺少 `schema_hash` 或 `description` 属性时抛 `AttributeError`；属性为 `None` 时会被格式化成字符串 `"None"` 而不会报错。
- **同文件关系**：被 `register_hot_tool` 与 `_sync_frozen_manifest` 调用；不调用本文件中的其它函数。

### `_sync_frozen_manifest(self) -> None` （第 332 行）
- **作用**：在第一次请求发出之前，一次性捕获「冻结清单」。第一次请求定义了字节稳定的提示词前缀：那一刻已注册的每个工具都会从冻结的 system 消息里被宣传；之后注册的工具只能进热区，不能碰这个前缀。它刻意不去重新扫描工具包（扫描时机由调用方通过 `discover_tools` 决定），因此它是一个纯粹的「快照」动作，而不是「同步/发现」动作。它通过 `self._frozen_manifest is not None` 实现幂等，保证只生效一次。
- **参数**：无参数。隐式使用 `self.tools`、`self.repository`、`self._frozen_manifest`。
- **返回**：无返回值（`None`）。副作用是把 `self._frozen_manifest` 从 `None` 变成 `dict[str, str]`（工具名到指纹的映射）。
- **内部流程**：第一步若 `self._frozen_manifest is not None` 则直接 `return`（幂等保护）；第二步 `names = set(self.tools.snapshot())` 取运行时注册的工具名集合，若 `self.repository is not None` 再用 `self.repository.active_tool_names()` 并入仓库中的活跃工具名；第三步初始化空 `manifest: dict[str, str] = {}`，遍历 `names`：先用 `self.tools.maybe_get(name)` 取运行时工具，若取到则 `manifest[name] = self._prompt_fingerprint(tool.spec)`；否则若存在仓库，用 `self.repository.get(name)` 取存储记录，若记录非空且 `stored.get("schema_hash")` 是字符串，则 `manifest[name] = f"{stored['schema_hash']}#{stored.get('description', '')}"`；第四步 `self._frozen_manifest = manifest`。
- **异常/边界**：`maybe_get` 返回 `None` 且无仓库时该工具被静默跳过；仓库记录缺少 `schema_hash` 或它不是字符串时该工具被跳过；`_prompt_fingerprint` 可能抛 `AttributeError`；`self.tools.snapshot()` 的返回值被当作可迭代的工具名集合使用。
- **同文件关系**：调用 `_prompt_fingerprint`；读取 `self.tools`、`self.repository`；被 `run_with_tools` 在构造缓存路由键之前调用；其产物 `_frozen_manifest` 被 `register_hot_tool`、`unregister_tool`、`_default_prompt_cache_key` 读取。

### `discover_tools(self, *, package: str | ModuleType | None = None, replace: bool = False, strict: bool = False, reload_modules: bool = False) -> ToolDiscoveryReport` （第 360 行）
- **作用**：扫描一个受信任的 Python 包，把其中发现的工具同步进运行时注册表（以及仓库），并返回一份发现报告。它是 `__init__` 中 `auto_discover_tools` 的实现体，也可以在运行期被显式调用来重新发现/重载工具。报告被保存在 `self.tool_discovery_report` 上，供上层检查发现了什么、跳过了什么、是否有失败。
- **参数**：`package`（`str | ModuleType | None`，仅关键字，默认 `None`）要扫描的包，为 `None` 时回退到构造时保存的 `self.tool_package`；`replace`（`bool`，仅关键字，默认 `False`）是否覆盖已注册的同名工具；`strict`（`bool`，仅关键字，默认 `False`）严格模式，发现失败时是否直接报错而不是记录到报告；`reload_modules`（`bool`，仅关键字，默认 `False`）是否重新加载已导入模块（用于热更新工具代码）。
- **返回**：返回 `ToolDiscoveryReport`，即底层 `discover_tool_modules` 的返回值，同时被赋给 `self.tool_discovery_report`。
- **内部流程**：第一步 `selected_package = self.tool_package if package is None else package`；第二步调用 `discover_tool_modules(self.tools, package=selected_package, repository=self.repository, replace=replace, strict=strict, reload_modules=reload_modules)`（模块级导入时被别名为 `discover_tool_modules`，来自 `core.discover_tools`）；第三步 `self.tool_discovery_report = report`；第四步返回 `report`。
- **异常/边界**：`strict=True` 时底层发现过程可能抛异常并向上传播；`package` 既不是字符串也不是模块时异常由底层函数抛出（本文件不做类型校验）；空包或没有任何工具时返回的报告通常是空发现结果而非异常。
- **同文件关系**：被 `__init__`（当 `auto_discover_tools=True`）调用；调用外部 `core.discover_tools`（别名 `discover_tool_modules`）；写入 `self.tool_discovery_report`。

### `is_tool_registered(self, name: str, *, version: str | None = None, schema_hash: str | None = None) -> bool` （第 381 行）
- **作用**：查询某个工具是否已注册，并可选地校验版本与 schema 哈希是否匹配。它是对 `ToolRegistry.is_registered` 的薄封装，为上层（例如插件管理器判断「这个工具是否已经加载过」或「加载的是不是同一个版本」）提供统一入口，避免上层直接触碰 `self.tools`。
- **参数**：`name`（`str`）工具名；`version`（`str | None`，仅关键字，默认 `None`）可选期望版本，为 `None` 时不校验版本；`schema_hash`（`str | None`，仅关键字，默认 `None`）可选期望 schema 哈希，为 `None` 时不校验哈希。
- **返回**：返回 `bool`。注册存在且（若给了 `version`/`schema_hash`）对应字段匹配时为 `True`，否则 `False`。
- **内部流程**：单步 `return self.tools.is_registered(name, version=version, schema_hash=schema_hash)`。
- **异常/边界**：无特殊处理；不存在的工具名返回 `False` 而不是抛异常（由底层实现保证）；`name` 非字符串时的行为取决于 `ToolRegistry`。
- **同文件关系**：调用 `self.tools.is_registered`；本文件内无其它调用方。

### `tool_registration_status(self, name: str) -> dict[str, Any]` （第 394 行）
- **作用**：返回某个工具注册状态的详细字典，用于诊断与审计（例如 Web 界面展示「这个工具当前注册的是哪个版本、schema 哈希是多少、来自哪个来源」）。它同样是对 `ToolRegistry.registration_status` 的透传封装，保证诊断信息的唯一来源是注册表本身。
- **参数**：`name`（`str`）工具名。
- **返回**：返回 `dict[str, Any]`，具体键值由 `ToolRegistry.registration_status` 决定（本文件不做加工）。
- **内部流程**：单步 `return self.tools.registration_status(name)`。
- **异常/边界**：无特殊处理；工具不存在时的行为（返回空字典还是抛异常）取决于 `ToolRegistry` 实现。
- **同文件关系**：调用 `self.tools.registration_status`；本文件内无其它调用方。

### `async execute_tool_calls(self, calls: list[ToolCall], context: ExecutionContext | None = None)` （第 397 行）
- **作用**：批量执行一组工具调用，并把执行上下文传下去。它是 Agent 对外暴露的「执行工具」入口，也是 `run_with_tools` 内部实际执行模型请求的工具时的通道。把执行集中到 `ToolExecutionManager` 的好处是：并发调度、超时、错误包装、权限与审计逻辑都统一在一个地方，Agent 只负责转发。
- **参数**：`calls`（`list[ToolCall]`）要执行的工具调用列表；`context`（`ExecutionContext | None`，默认 `None`）执行上下文，为 `None` 时由执行管理器自行处理。
- **返回**：返回 `await self.execution_manager.execute_batch(calls, context)` 的结果（一个批量执行结果对象，包含 `.results` 等字段，具体类型由 `ToolExecutionManager` 定义）。注意本方法没有写返回类型标注。
- **内部流程**：单步 await 调用 `self.execution_manager.execute_batch(calls, context)` 并返回其结果。
- **异常/边界**：无特殊处理；单个工具的失败通常被包装进 `ToolResult.error`，而调度层面的异常会向上传播给 `run_with_tools`。
- **同文件关系**：被 `run_with_tools` 调用（传入 `calls` 与 `context`）；调用 `self.execution_manager.execute_batch`。

### `tool_definitions(self, names: list[str] | None = None) -> list[dict[str, Any]]` （第 404 行）
- **作用**：根据已注册工具的 schema，构建 OpenAI 兼容的 function 定义列表（即请求体里的 `tools` 字段内容）。它支持两种模式：不传 `names` 时导出全部已注册工具；传 `names` 时只导出指定工具，并且会先校验这些名字是否都已注册，避免把不存在的工具名发给 provider 造成难以诊断的报错。它是外部调用方（例如需要手工构造请求、或想检查工具导出结果）的公开接口，`run_with_tools` 内部走的是更细粒度的 `_definitions_for_registrations`。
- **参数**：`names`（`list[str] | None`，默认 `None`）要导出的工具名列表；为 `None` 时导出全部。传入时必须是非空字符串组成的 list 或 tuple。
- **返回**：返回 `list[dict[str, Any]]`，每个元素形如 `{"type": "function", "function": {"name": ..., "description": ..., "parameters": ..., "strict": True}}`。顺序上：`names=None` 时按 `self.tools.specs()` 的顺序取全部名字后再经 `_definitions_for_registrations` 做规范排序；`names` 给定时按调用方传入的顺序（`list(names)`）。
- **内部流程**：第一步若 `names is not None`，校验它是 `list`/`tuple` 且所有元素都是非空字符串，否则抛 `TypeError("names must be a list of non-empty strings")`；第二步 `selected = set(names) if names is not None else None`；第三步若 `selected is not None`，用 `registered = {spec.name for spec in self.tools.specs()}` 求差集 `unknown = selected - registered`，非空则抛 `ValueError("unknown tool name(s): " + ", ".join(sorted(unknown)))`；第四步计算 `ordered_names`：`names is None` 时取 `[spec.name for spec in self.tools.specs()]`，否则 `list(names)`；第五步 `registrations = self.tools.snapshot(ordered_names)` 拿到 `{名字: (工具, 序号)}` 映射；第六步 `definitions, _ = self._definitions_for_registrations(registrations)` 丢弃别名映射，只返回定义列表。
- **异常/边界**：`names` 类型或元素非法抛 `TypeError`；包含未注册工具名抛 `ValueError`；`names` 为空列表时 `selected` 为空集，`unknown` 为空，`ordered_names` 为空列表，`snapshot([])` 通常返回空映射，最终返回空定义列表；`_definitions_for_registrations` 还可能因为名字过长或别名冲突抛 `ValueError`。
- **同文件关系**：调用 `self.tools.specs`、`self.tools.snapshot` 与 `_definitions_for_registrations`；本文件内无其它调用方（`run_with_tools` 直接用 `_definitions_for_registrations`）。

### `_definitions_for_registrations(self, registrations: Mapping[str, tuple[BaseTool, int]]) -> tuple[list[dict[str, Any]], dict[str, str]]` （第 424 行）
- **作用**：把一个「工具名 -> (工具, 注册序号)」的映射批量转换成 OpenAI function 定义，同时产出「provider 别名 -> 规范工具名」的反查表。它承担三个关键职责：一是规范化排序，让 `tools` 请求字段在不同运行、不同文件系统扫描顺序下都字节稳定（这是 OpenAI 前缀/KV 缓存复用的前提，因为工具定义属于被缓存的前缀内容）；二是把工具名映射成 provider 安全的名字（点号换成双下划线）并做长度与冲突校验；三是把输入 schema 转成 strict 形式。它是 `tool_definitions` 与 `run_with_tools` 共用的核心转换器。
- **参数**：`registrations`（`Mapping[str, tuple[BaseTool, int]]`）工具名到 `(工具实例, 注册序号)` 的映射，通常来自 `ToolRegistry.snapshot()`。
- **返回**：返回二元组 `(definitions, aliases)`。`definitions` 是 `list[dict[str, Any]]`，每个元素是 `{"type": "function", "function": {"name": alias, "description": spec.model_description, "parameters": _strict_function_schema(spec.input_schema), "strict": True}}`；`aliases` 是 `dict[str, str]`，键为 provider 别名、值为规范工具名。若 `registrations` 为空，返回 `([], {})`。
- **内部流程**：初始化 `definitions = []` 与 `aliases: dict[str, str] = {}`；然后按 `sorted(registrations, key=lambda item: (item != self.catalog_tool.spec.name, item))` 排序遍历——这个 key 的含义是「目录工具排在最前，其余按名字字典序」，从而保证顺序确定且目录工具位置固定；对每个 `name`：取出 `tool, _ = registrations[name]` 与 `spec = tool.spec`；计算 `alias = self._openai_tool_name(spec.name)`；若 `len(alias) > 64` 抛 `ValueError`（provider 的 64 字符限制）；查 `previous = aliases.get(alias)`，若存在且不等于当前 `spec.name`，说明两个不同工具映射到了同一个别名，抛 `ValueError`；否则 `aliases[alias] = spec.name`；最后把定义字典 append 进 `definitions`。循环结束后返回 `(definitions, aliases)`。
- **异常/边界**：别名长度超过 64 抛 `ValueError`；两个不同工具名映射到同一别名抛 `ValueError`；同一个工具名重复出现时 `previous == spec.name`，不会报错；`_strict_function_schema` 对非 object 根 schema 抛 `ValueError`，对含任意键对象的 schema 抛 `ValueError`；`spec` 缺少 `model_description` 或 `input_schema` 会抛 `AttributeError`。
- **同文件关系**：调用 `_openai_tool_name`、`_strict_function_schema`（模块级）；被 `tool_definitions` 与 `run_with_tools` 调用；读取 `self.catalog_tool.spec.name` 决定排序。

### `_openai_tool_name(name: str) -> str` （第 465 行，`@staticmethod`）
- **作用**：把内部可能带命名空间的工具名（例如 `file.read`）转换成 provider 能接受的函数名（`file__read`）。之所以需要它，是因为 OpenAI 的 function name 只允许字母数字和下划线，点号会被拒绝；同时内部仍希望保留带点号的可读 ID，所以用「别名映射」的方式在两边桥接：请求时用别名，回程时通过 `_definitions_for_registrations` 产出的别名表还原成规范名。它被设计成纯函数，方便在多个地方（定义构建、响应解析）复用同一套映射规则。
- **参数**：`name`（`str`）内部工具名，通常可能包含 `.`。
- **返回**：返回 `str`，即 `name.replace(".", "__")` 的结果。不含点号的名字原样返回。
- **内部流程**：单步字符串替换并返回，无副作用。
- **异常/边界**：无特殊处理；`name` 非字符串时抛 `AttributeError`；它不做长度校验，长度检查由 `_definitions_for_registrations` 负责。
- **同文件关系**：被 `_definitions_for_registrations` 与 `run_with_tools`（构建 `all_aliases` 反查表时）调用。

### `async run_with_tools(self, messages: list[dict[str, Any]], context: ExecutionContext | None = None, *, max_rounds: int | None = None, model: str | None = None, temperature: float = DEFAULT_TEMPERATURE, timeout: float = DEFAULT_TIMEOUT, tool_names: list[str] | None = None, profile_name: str | None = None, provider_name: str | None = None, use_history: bool = True, defer_tool_loading: bool = False, prompt_cache_key: str | None = None, prompt_cache_retention: str | None = None, enable_prompt_cache: bool = True, stream_echo: bool = False) -> str` （第 470 行）
- **作用**：这是整个文件最核心的方法——原生 function-calling 协议的完整工具循环。它把「历史前缀 + 本次消息 + 工具清单 system 消息」拼成一次请求，交给模型；若模型返回工具调用，就解析、执行、把结果作为 `tool` 消息追加进对话，再进入下一轮；若模型不再返回工具调用，就把本轮对话压缩裁剪后存为历史并返回最终文本。它还负责：把注册表中的工具转成 OpenAI 定义、按需只加载目录工具实现延迟加载（`defer_tool_loading`）、为稳定前缀计算并复用 prompt cache 路由键、对网关缺失 call id 的容错、对「工具未暴露给本轮」与「工具调用非法」两类错误的区分包装、以及轮次上限保护。
- **参数**：`messages`（`list[dict[str, Any]]`）本轮新增消息，必须是 mapping 组成的 list/tuple；`context`（`ExecutionContext | None`，默认 `None`）工具执行上下文，为 `None` 时自动创建 `ExecutionContext()`；`max_rounds`（`int | None`，仅关键字，默认 `None`）最大工具轮次，必须为 `None` 或正整数（`bool` 被显式拒绝）；`model`（`str | None`，仅关键字，默认 `None`）覆盖模型名，`None` 时用 profile 默认模型；`temperature`（`float`，仅关键字，默认 `DEFAULT_TEMPERATURE`）采样温度；`timeout`（`float`，仅关键字，默认 `DEFAULT_TIMEOUT`）请求超时；`tool_names`（`list[str] | None`，仅关键字，默认 `None`）显式指定本轮暴露给模型的工具子集；`profile_name`（`str | None`，仅关键字，默认 `None`）指定 provider profile；`provider_name`（`str | None`，仅关键字，默认 `None`）`profile_name` 的别名参数，两者同时给出且不一致时抛错；`use_history`（`bool`，仅关键字，默认 `True`）是否把已保存的 profile 历史作为前缀；`defer_tool_loading`（`bool`，仅关键字，默认 `False`）是否只先暴露目录工具、由模型按需加载其余工具；`prompt_cache_key`（`str | None`，仅关键字，默认 `None`）显式指定的缓存路由键；`prompt_cache_retention`（`str | None`，仅关键字，默认 `None`）缓存保留策略，只能为 `"in_memory"`、`"24h"` 或 `None`；`enable_prompt_cache`（`bool`，仅关键字，默认 `True`）是否启用缓存键；`stream_echo`（`bool`，仅关键字，默认 `False`）是否开启流式内容回显（为真时 `echo_mode = "content"`）。
- **返回**：返回 `str`，即模型最终的文本内容（`_field(message, "content") or ""`，内容为 `None` 时返回空字符串）。若工具轮次用尽仍未得到最终答案，则抛出 `RuntimeError("maximum tool-call rounds exceeded")` 而不是返回。
- **内部流程**：**（1）参数校验**：`max_rounds` 非 `None` 且（是 `bool`、不是 `int`、或小于 1）则抛 `ValueError`；`messages` 不是 list/tuple 或含非 Mapping 元素则抛 `TypeError`；`context` 为 `None` 则新建 `ExecutionContext()`，否则必须严格是 `ExecutionContext` 实例否则抛 `TypeError`；`use_history` 与 `defer_tool_loading` 必须是 `bool`；`enable_prompt_cache`、`stream_echo` 必须是 `bool`；`prompt_cache_key` 经 `_validate_prompt_cache_key` 校验、`prompt_cache_retention` 经 `_validate_prompt_cache_retention` 校验；`echo_mode = "content" if stream_echo else None`。**（2）选目标**：`completion_llm, selected_model, history_key = self._completion_target(profile_name, model, provider_name=provider_name)`。**（3）拼前缀**：若 `use_history` 为真，从 `self._profile_histories.get(history_key, [])` 逐条浅拷贝成 `prefix`，否则 `prefix = []`；若 `prefix` 为空，则回退到 `self._configured_prompt_messages()`；然后 `conversation = prefix + [dict(message) for message in messages]`。**（4）冻结前缀**：`initial_snapshot = self.tools.snapshot()`；随后立刻调用 `self._sync_frozen_manifest()`——注释说明必须在派生路由键之前冻结，因为第一次请求同时冻结了「被宣传的清单」和 `cache_epoch`，这样后续热注册永远不会改变原生路由键。`visible_order = list(initial_snapshot)`。**（5）决定本轮加载哪些工具**：若 `tool_names` 非 `None`，校验其为非空字符串列表，去重成 `requested_order`（`dict.fromkeys` 保序），求 `unknown = set(requested_order) - set(initial_snapshot)`，非空抛 `ValueError`，然后 `requested_names = set(requested_order)`、`loaded_order = requested_order`；否则若 `defer_tool_loading` 为真，则 `requested_names = None`、`loaded_order = [self.catalog_tool.spec.name]`（只暴露目录工具）；否则 `requested_names = None`、`loaded_order = visible_order`。**（6）缓存键**：若 `enable_prompt_cache`，`cache_key = prompt_cache_key or self._default_prompt_cache_key(history_key, selected_model or getattr(completion_llm, "model", None), mode="native")`。**（7）循环**：创建 `round_loop = ToolLoop(max_rounds, safety_limit=ToolLoop.DEFAULT_SAFETY_LIMIT)`，对 `round_loop.rounds()` 产生的每个 `round_number`：a. `current_snapshot = self.tools.snapshot()`；b. 从 `loaded_order` 里筛出仍在 `current_snapshot` 中的项组成 `registrations`；c. `tool_definitions, name_map = self._definitions_for_registrations(registrations)`；d. 组装 `completion_options`：`model`、`temperature`、`timeout`、`stream=False`，若 `cache_key` 非空加 `prompt_cache_key`，若 `prompt_cache_retention` 非空加 `prompt_cache_retention`，若 `tool_definitions` 非空加 `tools`；e. `response = await asyncio.to_thread(self._dispatch_model_call, completion_llm, self._with_registered_tool_names(conversation, registrations), completion_options, round_number=round_number, echo_mode=echo_mode)`；f. 取 `choices = _field(response, "choices")`，为空抛 `RuntimeError("LLM response contained no choices")`；g. 取 `message = _field(choices[0], "message")`，为 `None` 抛 `RuntimeError("LLM response contained no message")`；h. `native_calls = _field(message, "tool_calls") or []`，`assistant_message = _message_dict(message)`，把 assistant 消息追加进 `conversation`；i. 若 `native_calls` 为空，则 `self._save_history(history_key, conversation)` 并返回 `_field(message, "content") or ""`。**（8）解析工具调用**：初始化 `calls`、`positions`、`results_by_position`，并构建 `all_aliases = {self._openai_tool_name(tool.spec.name): name for name, (tool, _) in current_snapshot.items()}`；对每个 `(position, native_call)`：取 `function = _field(native_call, "function")`、`provider_tool_name = _field(function, "name", "unknown.tool")`，用 `all_aliases.get(...)` 把 provider 别名还原成 `canonical_name`（非字符串则原样保留）；取 `call_id = _field(native_call, "id")`，若它不是非空字符串，则用稳定的兜底 `f"native-call-{position + 1}"`（因为部分网关不返回 id）；随后 `try` 调用 `parse_openai_tool_calls([native_call], self.tools, name_map, registrations)[0]` 得到规范 `ToolCall`，成功则 `calls.append(call)`、`positions.append(position)`；失败（`TypeError`/`ValueError`）则构造失败结果：错误码为 `"TOOL_NOT_EXPOSED"`（当 `canonical_name` 是字符串、在 `current_snapshot` 中存在、但不在本轮 `registrations` 中，即工具存在但没暴露给这轮）否则为 `"INVALID_TOOL_CALL"`，并写入 `results_by_position[position] = ToolResult(call_id=call_id, tool_name=_safe_tool_name(canonical_name), ok=False, error=ToolError(code=code, message=_safe_tool_call_error(exc)))`，然后 `continue`。**（9）执行**：若 `calls` 非空，`batch = await self.execute_tool_calls(calls, context)`，再用 `zip(positions, batch.results, strict=True)` 把结果按原位置写回 `results_by_position`。**（10）回写 tool 消息**：再次按 `enumerate(native_calls)` 遍历，取 `result = results_by_position[position]`，追加 `{"role": "tool", "tool_call_id": _field(native_call, "id"), "content": _result_json(result)}` 到 `conversation`；若 `defer_tool_loading` 且 `requested_names is None`，则调用 `self._load_catalog_result(result, loaded_order, context)` 把目录工具返回的 spec 对应的工具加入下一轮加载顺序。**（11）循环结束后**抛 `RuntimeError("maximum tool-call rounds exceeded")`。
- **异常/边界**：`max_rounds` 非法抛 `ValueError`；`messages`/`context`/布尔开关类型错误抛 `TypeError`；`tool_names` 含未注册工具抛 `ValueError`；`prompt_cache_key` 非法抛 `ValueError`；`prompt_cache_retention` 非法抛 `ValueError`；响应缺 `choices` 或缺 `message` 抛 `RuntimeError`；轮次耗尽抛 `RuntimeError`；单个工具调用解析失败不抛出而是转成 `ok=False` 的 `ToolResult` 回给模型，让模型有机会自我纠正；网关缺失 `call id` 时自动生成 `native-call-N` 兜底并原样写进 tool 消息的 `tool_call_id`（注意回写时用的是原始 `_field(native_call, "id")` 而不是兜底值）；`results_by_position` 用 `position` 索引，保证即使部分调用解析失败也能在回写阶段取到每一条结果。
- **同文件关系**：调用 `_validate_prompt_cache_key`、`_validate_prompt_cache_retention`、`_completion_target`、`_configured_prompt_messages`、`_sync_frozen_manifest`、`_default_prompt_cache_key`、`_definitions_for_registrations`、`_dispatch_model_call`、`_with_registered_tool_names`、`_save_history`、`execute_tool_calls`、`_openai_tool_name`、`_load_catalog_result`；被 `run_auto` 调用；使用 `_field`、`_message_dict`、`_result_json`、`_safe_tool_name`、`_safe_tool_call_error` 等消息工具别名。

### `_load_catalog_result(self, result: ToolResult, loaded_order: list[str], context: ExecutionContext) -> None` （第 677 行）
- **作用**：这是延迟工具加载（`defer_tool_loading`）的「加载触发器」。当模型先调用目录工具（`ToolCatalogTool`）查询可用工具后，目录工具的返回数据里会带上若干工具的 spec；本方法解析这些 spec，把它们对应的已注册工具追加进 `loaded_order`，从而让下一轮的 `tools` 定义里出现这些工具的完整 schema。它只处理成功且确实来自目录工具的结果，其它结果一律忽略，避免误加载。
- **参数**：`result`（`ToolResult`）刚刚执行完的某个工具结果；`loaded_order`（`list[str]`）本轮用于构建工具定义的顺序列表，本方法会就地修改它（原地 append）；`context`（`ExecutionContext`）执行上下文，签名中保留用于与执行体系保持一致（方法体内并未使用它）。
- **返回**：无返回值（`None`）。副作用是可能向 `loaded_order` 追加工具名。
- **内部流程**：第一步若 `not result.ok` 或 `result.tool_name != self.catalog_tool.spec.name`，直接 `return`；第二步 `data = result.data if isinstance(result.data, Mapping) else {}`；第三步取 `raw_specs = data.get("specs")`，若不是 `list` 则替换为空列表；第四步取 `raw_spec = data.get("spec")`，若它是 `Mapping` 且不在 `raw_specs` 中，则 `raw_specs.insert(0, raw_spec)`（把单个 spec 提到最前面优先处理）；第五步遍历 `raw_specs`：跳过非 `Mapping` 的元素；取 `tool_name = candidate.get("tool_name")`，不是非空字符串则跳过；`registration = self.tools.maybe_resolve(tool_name)`，为 `None`（无法解析到已注册工具）则跳过；若 `tool_name not in loaded_order` 则 `loaded_order.append(tool_name)`（去重追加）。
- **异常/边界**：结果失败或不是目录工具的结果时静默返回；`result.data` 非 Mapping 时当作空字典处理；`specs` 不是列表时当作空列表；单个候选不是 Mapping 或缺 `tool_name` 时跳过；工具未注册（`maybe_resolve` 返回 `None`）时跳过；`loaded_order` 的重复项通过 `in` 判断避免。`context` 参数在本方法体内未被使用，这是签名与实现之间的一个已知不一致。
- **同文件关系**：被 `run_with_tools` 在 `defer_tool_loading` 且 `requested_names is None` 时对每条工具结果调用；调用 `self.tools.maybe_resolve` 与 `self.catalog_tool.spec.name`。

### `_configured_prompt_messages(self) -> list[dict[str, Any]]` （第 704 行）
- **作用**：把用户在 `self.prompt` 字典里配置的提示词转换成消息列表，作为「没有历史时」的对话前缀。它按固定顺序 `system`、`user`、`assistant` 输出已有角色，并把特殊的 `tool` 键单独转成一条额外的 system 消息，前缀上 `Tool-use instructions: ` 文字。这样设计的目的是让工具使用说明以 system 身份进入前缀，而不是伪装成某种对话角色，从而既明确又稳定。它同时被 `run_with_tools` 与 `_default_prompt_cache_key` 使用，因此它输出的内容本身就是缓存键材料的一部分。
- **参数**：无参数。读取 `self.prompt`（`dict[str, str]`），键为角色名，值为提示词文本。
- **返回**：返回 `list[dict[str, Any]]`，元素形如 `{"role": role, "content": self.prompt[role]}`。顺序严格为 `system`、`user`、`assistant` 中存在的键；若 `self.prompt` 中存在 `"tool"` 键，则在末尾追加 `{"role": "system", "content": f"Tool-use instructions: {self.prompt['tool']}"}`。`self.prompt` 为空字典时返回空列表。
- **内部流程**：第一步用列表推导按 `("system", "user", "assistant")` 的顺序筛出存在的键并构造消息；第二步单独判断 `"tool" in self.prompt`，为真则 append 一条 system 消息；第三步返回 `messages`。
- **异常/边界**：无特殊处理；`self.prompt` 的值若为空字符串不会被过滤（只有键存在与否决定是否输出），因此调用方应通过 `_set_prompt` / `set_system_prompt` 写入非空内容。
- **同文件关系**：被 `run_with_tools`（无历史前缀时的回退）与 `_default_prompt_cache_key`（作为 `configured_prompt` 材料）调用；其内容来源是 `_set_prompt` 与 `set_system_prompt` 写入的 `self.prompt`。

### `_with_registered_tool_names(self, conversation: list[dict[str, Any]], registrations: Mapping[str, tuple[BaseTool, int]] | None = None) -> list[dict[str, Any]]` （第 719 行）
- **作用**：在真正发给模型的消息前面插一条 system 消息，内容包括两部分：一是「所有已注册工具名」的清单（故意包含只存在于仓库、尚未提供完整 schema 的工具，因为延迟加载需要模型先知道有它们），二是本轮真正可调用工具的人类可读完整契约（用途、使用规则、每个输入变量的类型/是否必填/默认值/约束及其描述、输出字段、是否需要人工确认）。这么做的原因很实际：provider 的 `tools` 字段虽然带了机器可读 schema，但模型有时会忽略或误读 function-calling 载荷，把同一份契约再用自然语言写进提示词，能显著提高工具被正确调用的概率。它是 `run_with_tools` 每轮都要调用的消息加工器。
- **参数**：`conversation`（`list[dict[str, Any]]`）已有的完整消息列表；`registrations`（`Mapping[str, tuple[BaseTool, int]] | None`，默认 `None`）本轮可调用的工具映射，为 `None` 或空时不渲染契约块，只渲染清单。
- **返回**：返回 `list[dict[str, Any]]`，即 `[新的 system 消息, *conversation]`。注意返回的是新列表（前缀是新建的列表，`conversation` 元素是同一批对象引用，未被复制）。
- **内部流程**：第一步 `names = set(self.tools.snapshot())`，若 `self.repository is not None` 再并入 `self.repository.active_tool_names()`；第二步 `inventory = ", ".join(sorted(names)) or "(none)"`（空集合时用 `"(none)"` 占位）；第三步 `content = ["All registered tool names: " + inventory]`；第四步若 `registrations` 为真，则定义表头 `header = "Tool contracts (full variable-level specification of the tools callable in this request):"`，并用 `content.extend(("", header, *render_tool_catalog(registrations)))` 追加空行、表头和 `render_tool_catalog` 渲染出的契约行；第五步构造 `prefix` 为单元素列表 `[{"role": "system", "content": "\n".join(content)}]`；第六步 `return [*prefix, *conversation]`。
- **异常/边界**：无特殊处理；`registrations` 为空映射（`{}`）时因 falsy 而跳过契约块；`render_tool_catalog` 的异常（若有）会向上传播；当仓库为空且运行时无工具时清单为 `"(none)"`。
- **同文件关系**：被 `run_with_tools` 在每轮请求前调用；调用 `self.tools.snapshot`、`self.repository.active_tool_names` 以及外部 `render_tool_catalog`（来自 `core.tool_docs`）。

### `_validate_prompt_cache_key(value: str | None) -> str | None` （第 759 行，`@staticmethod`）
- **作用**：校验 OpenAI prompt-cache 的路由键。OpenAI 目前把这个键限制在 64 字符以内，本方法把这条约束也放在 Agent 层，使得「注入的假客户端」与「真实 SDK 客户端」观察到完全一致的契约——这对测试和可替换性都很重要。它还顺带做了 `.strip()` 归一化，避免首尾空格造成同一语义的键被拆成不同缓存命名空间。
- **参数**：`value`（`str | None`）待校验的键。
- **返回**：返回 `str | None`。`value is None` 时返回 `None`；合法时返回 `value.strip()`（去除首尾空白后的字符串）；非法时抛异常。
- **内部流程**：第一步若 `value is None` 返回 `None`；第二步若 `value` 不是 `str`、或 `value.strip()` 为空、或 `len(value) > 64`，抛 `ValueError("prompt_cache_key must be a non-empty string of at most 64 characters")`；第三步返回 `value.strip()`。
- **异常/边界**：非字符串、全空白、超长（注意长度判断用的是未 strip 的 `value`）都抛 `ValueError`；恰好 64 字符（含首尾空白时可能 strip 后更短）通过。
- **同文件关系**：被 `run_with_tools` 在组装请求前调用；不调用本文件中的其它函数。

### `_validate_prompt_cache_retention(value: str | None) -> str | None` （第 776 行，`@staticmethod`）
- **作用**：校验 prompt cache 的保留策略取值，只允许 OpenAI 目前支持的两种字面量 `"in_memory"` 与 `"24h"`（或 `None` 表示不指定）。它把非法取值挡在请求发出之前，避免 provider 返回难以定位的 400 错误。
- **参数**：`value`（`str | None`）待校验的保留策略。
- **返回**：返回 `str | None`，原样返回通过校验的 `value`（不做 strip、不做大小写归一化）。
- **内部流程**：单步判断——若 `value is not None` 且 `value not in {"in_memory", "24h"}`，抛 `ValueError("prompt_cache_retention must be 'in_memory', '24h', or None")`；否则 `return value`。
- **异常/边界**：任何不在白名单中的非 `None` 取值（包括大小写变体如 `"24H"`、带空格的 `" 24h"`、空字符串）都抛 `ValueError`；`None` 直接通过。
- **同文件关系**：被 `run_with_tools` 调用；不调用本文件中的其它函数。

### `_default_prompt_cache_key(self, provider_key: str, model: str | None, *, mode: str) -> str` （第 784 行）
- **作用**：为「稳定的提示词前缀」构造一个确定性的缓存路由键。它的设计要点是**排除**那些属于前缀之后的内容：用户文本、工具执行结果、以及延迟加载进来的目录 schema，都不能参与键的构造，否则会不断把请求打散到不同缓存路由上，彻底破坏前缀缓存命中率。热重载的工具名通过热区名单参与，因为它们的渲染块位于请求尾部；但热重载本身绝不能改变路由键——只有 `cache_epoch`（冻结区的结构性变化）才会开启一个新的缓存命名空间。它在 `mode == "react"` 时还会把 ReAct 的类常量指令模板纳入摘要，这样一旦提示词模板部署发生变化，缓存命名空间会自然更新。
- **参数**：`provider_key`（`str`）provider 或 profile 的标识，参与摘要；`model`（`str | None`）模型名，为 `None` 时用空字符串占位；`mode`（`str`，仅关键字）协议模式，`run_with_tools` 传 `"native"`，ReAct 路径会传 `"react"`。
- **返回**：返回 `str`，格式为 `f"{PROMPT_CACHE_KEY_VERSION}-{digest}"`，其中 `digest` 是 48 个字符的十六进制 SHA-256 前缀。
- **内部流程**：第一步 `names = set(self.tools.snapshot())`，若 `self.repository is not None` 并入 `self.repository.active_tool_names()`；第二步若 `self._frozen_manifest is not None`，则用 `names = set(self._frozen_manifest)` 覆盖——注释说明热重载工具不能改变路由键，而冻结清单只会随 `cache_epoch` 一起变化；第三步构造 `material` 字典，键为 `version`（`PROMPT_CACHE_KEY_VERSION`）、`provider`（`provider_key`）、`model`（`model or ""`）、`mode`、`configured_prompt`（`self._configured_prompt_messages()`）、`tool_names`（`sorted(names)`）、`cache_epoch`（`self.cache_epoch`）；第四步若 `mode == "react"`，追加 `material["react_instructions"] = [getattr(self, "REACT_INSTRUCTIONS", ""), getattr(self, "CATALOG_FIRST_REACT_INSTRUCTIONS", "")]`（用 `getattr` 默认空串，使得非 ReAct 类也能安全取到）；第五步 `encoded = json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")`；第六步 `digest = hashlib.sha256(encoded).hexdigest()[:48]`；第七步返回 `f"{PROMPT_CACHE_KEY_VERSION}-{digest}"`。
- **异常/边界**：`material` 中所有值都是可 JSON 序列化的（字符串、列表、整数），因此正常情况下 `json.dumps` 不会失败；`model` 为 `None` 时被替换为空串；`_configured_prompt_messages()` 的结果是字典列表，`sort_keys=True` 保证键序稳定，`separators` 去掉多余空格保证字节稳定；`self._frozen_manifest` 为 `None` 时用实时工具名集合，这意味着「首次请求之前」构造键会包含所有已注册工具名，而首次请求之后则锁定为冻结清单。
- **同文件关系**：调用 `_configured_prompt_messages`；读取 `self.tools`、`self.repository`、`self._frozen_manifest`、`self.cache_epoch`；被 `run_with_tools` 调用（`mode="native"`）；ReAct 路径（本文件外）会以 `mode="react"` 调用它。

### `_completion_target(self, profile_name: str | None, model: str | None, *, provider_name: str | None = None) -> tuple[Any, str | None, str]` （第 836 行）
- **作用**：解析「这一次请求到底该用哪个 LLM 客户端、哪个模型、哪个历史分区」。它支持三种来源：显式注入的 `self.llm`（优先级最高，直接返回并标记历史键为 `"__injected__"`）、显式指定的 profile（`profile_name` 或等价的 `provider_name`）、以及回退到 `self.active_profile`。它还实现了按 `(profile, model)` 缓存的客户端复用——同一个组合只会构造一次 `LLM`，避免每轮请求都重新建连；同时会在构造客户端前通过注册表解析 API key。模型合法性也在这一层校验：模型必须出现在该 profile 声明的 `models` 列表里，否则直接报错而不是把无效模型发给 provider。
- **参数**：`profile_name`（`str | None`）目标 profile 名，为 `None` 时使用 `self.active_profile`（或注入的 `self.llm`）；`model`（`str | None`）目标模型名，为 `None` 时使用 profile 的 `default_model`；`provider_name`（`str | None`，仅关键字，默认 `None`）`profile_name` 的别名，用于兼容旧调用；两者同时给出时必须一致。
- **返回**：返回三元组 `(client, selected_model, history_key)`：`client` 是 `LLM` 实例或注入的 `self.llm`（类型标注为 `Any`）；`selected_model` 是 `str | None`（注入 `self.llm` 且未显式指定 `model` 时为 `None`，否则为解析出的模型名）；`history_key` 是 `str`，注入场景下为 `"__injected__"`，否则为 profile 名。
- **内部流程**：第一步校验 `profile_name`：非 `None` 且（不是 `str` 或 strip 后为空）则抛 `ValueError("profile_name must be a non-empty string or None")`。第二步若 `provider_name is not None`：若 `profile_name` 也非 `None` 且两者不相等，抛 `ValueError("profile_name and provider_name must match when both are set")`；然后把 `profile_name = provider_name`。第三步校验 `model`：非 `None` 且（不是 `str` 或 strip 后为空）抛 `ValueError("model must be a non-empty string or None")`。第四步 `selected_provider = profile_name`、`selected_model = model`。第五步若 `selected_provider is None and self.llm is not None`，直接 `return self.llm, selected_model, "__injected__"`。第六步 `selected_provider = selected_provider or self.active_profile`，`profile = self.provider_registry.get(selected_provider)`。第七步 `selected_model = selected_model or profile.default_model`；若 `selected_model not in profile.models`，抛 `ValueError`（提示该 profile 不支持该模型）。第八步 `cache_key = (selected_provider, selected_model)`，从 `self._profile_clients` 取客户端；若为 `None`，则 `api_key = self.provider_registry.resolve_api_key(selected_provider)`，构造 `LLM(api_key=api_key, base_url=profile.base_url, model=selected_model, max_retries=self.max_retries)` 并写入缓存。第九步返回 `(client, selected_model, selected_provider)`。
- **异常/边界**：`profile_name`/`provider_name` 非法或互相矛盾抛 `ValueError`；`model` 非法抛 `ValueError`；profile 不存在时 `self.provider_registry.get` 的行为由其实现决定（通常抛异常）；模型不在 `profile.models` 中抛 `ValueError`；`resolve_api_key` 可能因缺少密钥抛异常；注入 `self.llm` 时不校验模型、不解析 key、也不走 profile 的 `models` 白名单，历史分区固定为 `"__injected__"`，因此不同注入 llm 之间会共用同一历史键（这是需要注意的边界）。
- **同文件关系**：被 `run_with_tools` 调用；读取 `self.llm`、`self.active_profile`、`self.provider_registry`、`self._profile_clients`、`self.max_retries`；其返回的 `history_key` 被 `run_with_tools` 用于读写 `_profile_histories` 并参与缓存键构造。

### `set_system_prompt(self, prompt: str) -> None` （第 880 行）
- **作用**：公开的便捷方法，用来设置 system 提示词。它把 `"system"` 角色和文本转发给内部通用设置方法，保证与其它角色的设置走同一套非空校验。上层（例如 Web 应用读取配置后注入人设/规则）通过它来定制 agent 的系统提示。
- **参数**：`prompt`（`str`）system 提示词文本，必须是非空字符串（仅空白视为非法）。
- **返回**：无返回值（`None`）。副作用是写入 `self.prompt["system"]`。
- **内部流程**：单步调用 `self._set_prompt("system", prompt)`。
- **异常/边界**：`prompt` 非字符串或全空白时，由 `_set_prompt` 抛 `ValueError`。
- **同文件关系**：调用 `_set_prompt`；写入的 `self.prompt["system"]` 会被 `_configured_prompt_messages` 读取并进入缓存键材料。

### `_set_prompt(self, role: str, prompt: str) -> None` （第 883 行）
- **作用**：所有提示词写入的统一底层入口，负责非空校验后写入 `self.prompt` 字典。之所以把校验集中在这里，是为了保证无论通过哪个角色的便捷方法（目前是 `set_system_prompt`）写入，都不会把空字符串塞进提示词前缀，从而避免生成一条内容为空的 system 消息。
- **参数**：`role`（`str`）角色名，实际写入时作为 `self.prompt` 的键（本文件内的调用只传 `"system"`，但方法本身不限制取值）；`prompt`（`str`）提示词文本。
- **返回**：无返回值（`None`）。副作用是 `self.prompt[role] = prompt`。
- **内部流程**：第一步若 `prompt` 不是 `str` 或 `prompt.strip()` 为空，抛 `ValueError("prompt must be a non-empty string")`；第二步 `self.prompt[role] = prompt`（保存的是原始未 strip 的文本）。
- **异常/边界**：`prompt` 非字符串或全空白抛 `ValueError`；`role` 不做类型或取值校验，传入非字符串键会导致 `self.prompt` 中出现非字符串键，进而可能影响 `_configured_prompt_messages` 的遍历结果。
- **同文件关系**：被 `set_system_prompt` 调用；写入结果被 `_configured_prompt_messages` 与 `_default_prompt_cache_key` 使用。

### `_strict_function_schema(schema: dict[str, Any]) -> dict[str, Any]` （第 889 行，模块级函数）
- **作用**：把一个（通常来自 Pydantic 的）JSON Schema 递归转换成 OpenAI strict function-calling 所要求的形式。strict 模式有三条硬约束：每个对象都必须显式声明 `additionalProperties: False`、每个对象的 `required` 必须列出全部属性、以及不允许出现 `default` 关键字。本函数通过一次深度拷贝加递归遍历，把这三条一次性落实，并在遇到「无法满足 strict 要求」的 schema（根不是 object，或某个对象允许任意键）时立刻报错，从而把问题暴露在发请求之前。它是 `_definitions_for_registrations` 生成 `parameters` 字段时调用的关键工具函数。
- **参数**：`schema`（`dict[str, Any]`）输入 schema 字典，根必须是 `{"type": "object", "properties": {...}}` 形式。
- **返回**：返回 `dict[str, Any]`，是规范化后的**深拷贝**（原 schema 不被修改）。规范化内容包括：递归删除所有 `default` 键；对每个 `type == "object"` 的节点强制 `additionalProperties = False`（若原本存在且不是 `False` 则报错）并把 `required` 设为该节点 `properties` 的全部键。
- **内部流程**：第一步 `normalized = copy.deepcopy(schema)`；第二步校验 `normalized.get("type") != "object"` 或 `normalized.get("properties")` 不是 `dict` 时抛 `ValueError("function input schemas must have an object root")`；第三步定义嵌套函数 `normalize(value)` 并调用 `normalize(normalized)`；第四步返回 `normalized`。
- **异常/边界**：根不是 object 或缺 `properties` 抛 `ValueError`；任意对象节点的 `additionalProperties` 存在且不是 `False` 时抛 `ValueError("strict function schemas cannot contain arbitrary object keys")`；`properties` 不是 dict 的 object 节点会被补上 `additionalProperties = False` 但不设置 `required`；`normalize` 对列表递归处理每个元素，对非 dict 非 list 的值（字符串、数字、`None`、布尔）直接返回；注意 `normalize` 会遍历 `value.values()`，因此在补完 `additionalProperties`/`required` 之后，这些新增值也会被递归访问（它们是 `False` 与字符串列表，属于安全类型）。
- **同文件关系**：被 `_definitions_for_registrations` 调用；内部定义并调用嵌套函数 `normalize`；不调用本文件中的其它模块级函数。

### `normalize(value: Any) -> None` （第 897 行，嵌套在 `_strict_function_schema` 内）
- **作用**：这是 `_strict_function_schema` 的递归工作单元，对 schema 树的任意节点就地做 strict 规范化。它同时处理三种情况：列表要逐元素递归；非字典值（标量）直接忽略；字典节点则先删 `default`，再在 `type == "object"` 时校验并强制 `additionalProperties = False`、把 `required` 设为全部属性名，最后继续递归它自己的所有子值。它是「一次遍历完成整棵树改写」的实现核心。
- **参数**：`value`（`Any`）当前待处理的节点，可能是 list、dict 或任意标量。
- **返回**：无返回值（`None`）；通过就地修改传入的字典/列表产生效果。
- **内部流程**：第一步若 `isinstance(value, list)`，对每个元素调用 `normalize(item)` 后 `return`；第二步若 `not isinstance(value, dict)`，直接 `return`；第三步 `value.pop("default", None)` 删除默认值键（strict 模式不允许）；第四步 `properties = value.get("properties")`；第五步若 `value.get("type") == "object"`：取 `additional = value.get("additionalProperties")`，若它非 `None` 且不是 `False` 则抛 `ValueError("strict function schemas cannot contain arbitrary object keys")`；否则 `value["additionalProperties"] = False`；若 `isinstance(properties, dict)` 则 `value["required"] = list(properties)`；第六步 `for child in value.values(): normalize(child)` 递归所有子值。
- **异常/边界**：对象节点显式声明了非 `False` 的 `additionalProperties`（例如 `True` 或一个 schema 字典）时抛 `ValueError`；`additionalProperties` 为 `None`（未声明）会被补成 `False`；`properties` 不是 dict 时只补 `additionalProperties` 而不动 `required`；递归深度受 Python 递归栈限制，极深的 schema 理论上可能触发 `RecursionError`；由于遍历 `value.values()`，同一个子对象若被多个键引用会被重复处理（幂等，无副作用）。
- **同文件关系**：被 `_strict_function_schema` 定义并调用（含自递归）；不调用本文件中的其它函数。

### `compress_saved_history(conversation: list[dict[str, Any]]) -> list[dict[str, Any]]` （第 922 行，模块级函数）
- **作用**：在把对话写入历史之前，把超大的工具返回载荷替换成「存根」，从而防止一次大读取把几十 KB 原文永久钉进之后每一次请求，最终导致 provider 卡顿或超出上下文窗口。存根里保留了工具名、原始大小、一小段预览以及「请重新调用同一工具查询」的提示，这样模型既知道这里曾经有数据、又知道怎么重新拿到它。它处理两类消息：ReAct 风格的 `user` 消息（内容以 `Observation: ` 开头）和原生协议的 `tool` 角色消息。它是 `_save_history` 的第一步加工。
- **参数**：`conversation`（`list[dict[str, Any]]`）完整对话消息列表。
- **返回**：返回 `list[dict[str, Any]]`，是一个**新列表**：被压缩的消息是原消息的浅拷贝（`{**message, "content": ...}`），未被压缩的消息是 `dict(message)` 浅拷贝。原列表与原字典均不被修改。
- **内部流程**：初始化 `compressed: list[dict[str, Any]] = []`；逐条遍历 `conversation`，取 `content = message.get("content")`；第一个分支：若 `message.get("role") == "user"` 且 `content` 是 `str` 且以 `"Observation: "` 开头 且 `len(content) > OBSERVATION_COMPRESS_THRESHOLD` 且不以 `f"Observation: {OBSERVATION_STUB_PREFIX}"` 开头（避免二次压缩存根本身），则 `payload = content[len("Observation: "):]`，追加 `{**message, "content": "Observation: " + _observation_stub(payload)}`；第二个分支：否则若 `message.get("role") == "tool"` 且 `content` 是 `str` 且 `len(content) > OBSERVATION_COMPRESS_THRESHOLD`，追加 `{**message, "content": _observation_stub(content)}`；否则追加 `dict(message)`。遍历结束返回 `compressed`。
- **异常/边界**：`content` 缺失或不是字符串时走原样保留分支；长度恰好等于阈值时不压缩（判断用的是严格大于）；已经是存根（以 `Observation: {OBSERVATION_STUB_PREFIX}` 开头）的 user 消息不会被重复压缩，但 `tool` 角色消息没有这层「已是存根」的判断，因此若一条 tool 消息本身已是存根且长度仍超阈值，会被再压一次；非 `user`/`tool` 角色的超大消息（例如超长的 assistant 输出）不会被压缩。
- **同文件关系**：调用 `_observation_stub`；被 `_save_history` 调用，且必须在 `trim_saved_history` 之前执行。

### `trim_saved_history(conversation: list[dict[str, Any]], *, max_messages: int = HISTORY_MAX_MESSAGES) -> list[dict[str, Any]]` （第 956 行，模块级函数）
- **作用**：给持久化的历史设定条数上限，同时保证开头的 `system` 块不会被裁掉。没有这个上限，长期存活的 agent 每一轮都要重发整段对话，token 成本无限制增长；有了它，较旧的轮次会从前面被丢弃。它还额外做了一件正确性保护：如果裁剪后队首正好是一条 `Observation:` 消息，就把它也丢掉——因为那条观察结果对应的动作已经被裁掉，单独留下一条没有上下文的观察会误导模型。它是 `_save_history` 的第二步加工。
- **参数**：`conversation`（`list[dict[str, Any]]`）已经压缩过的对话消息列表；`max_messages`（`int`，仅关键字，默认 `HISTORY_MAX_MESSAGES`）允许保留的「非 leading system」消息条数上限。
- **返回**：返回 `list[dict[str, Any]]`。注意它的返回策略是「尽量少改动」：若 `max_messages < 1` 或 `len(conversation) <= max_messages`，直接返回**原列表对象**；若分离出的 `rest` 长度不超过 `max_messages`，也返回**原列表对象**；只有在确实需要裁剪时，才返回新构造的 `leading + kept` 列表（其中的元素都是逐条 `dict(...)` 浅拷贝）。
- **内部流程**：第一步若 `max_messages < 1` 或 `len(conversation) <= max_messages`，`return conversation`。第二步遍历 `conversation`，把「`rest` 仍为空且角色是 `system`」的消息逐条拷贝进 `leading`（即只收集开头连续的 system 块），其余全部拷贝进 `rest`。第三步若 `len(rest) <= max_messages`，`return conversation`（说明超出的其实只有 system 块，不必裁）。第四步 `kept = rest[-max_messages:]` 取最后 `max_messages` 条。第五步 `while kept and str(kept[0].get("content", "")).startswith("Observation: "): kept = kept[1:]`——循环丢弃队首的孤立观察消息。第六步 `return leading + kept`。
- **异常/边界**：`max_messages < 1` 时直接返回原列表（相当于不裁剪）；`conversation` 为空时长度 0 不大于上限（上限 ≥1 时），返回原列表；注意开头的 `system` 消息不计入 `max_messages` 配额，因此最终列表长度可能超过 `max_messages`（等于 `len(leading) + max_messages`）；丢弃队首观察消息后实际保留数可能少于 `max_messages`；`content` 缺失时 `str(...)` 得到 `"None"`，不会误判为 `Observation:`。
- **同文件关系**：被 `_save_history` 调用（在 `compress_saved_history` 之后）；不调用本文件中的其它函数。

### `_observation_stub(payload: str) -> str` （第 988 行，模块级函数）
- **作用**：生成被压缩工具载荷的「存根文本」。它把原始载荷截取前 `OBSERVATION_PREVIEW_CHARS` 个字符作为预览，并附带原始字符数（带千分位逗号）和一句中文提示，告诉模型如需完整数据要重新调用同一工具。它是 `compress_saved_history` 的文本生成器，决定了压缩后模型能看到多少线索——既要足够提示数据存在与内容主题，又要足够短以免重新造成体积问题。
- **参数**：`payload`（`str`）原始的工具返回内容文本。
- **返回**：返回 `str`，格式为 `f"{OBSERVATION_STUB_PREFIX} | 原始大小: {len(payload):,} 字符 | 预览: {preview}... | 如需完整数据请让 AI 重新调用同一工具查询。]"`，其中 `preview = payload[:OBSERVATION_PREVIEW_CHARS]`。注意该字符串以 `]` 结尾，但本函数并不生成开头的 `[`（`OBSERVATION_STUB_PREFIX` 常量本身是否含 `[` 由常量定义决定）。
- **内部流程**：第一步 `preview = payload[:OBSERVATION_PREVIEW_CHARS]` 截取预览；第二步用 f-string 拼接常量前缀、`len(payload):,` 格式化后的原始长度、预览和固定中文提示，并返回。
- **异常/边界**：无特殊处理；`payload` 为空字符串时预览为空、长度为 0，仍能正常生成存根；`payload` 不是字符串时切片与 `len` 可能抛 `TypeError`（但调用方 `compress_saved_history` 已保证是字符串）。
- **同文件关系**：被 `compress_saved_history` 调用；不调用本文件中的其它函数。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `Agent` | 抽象基类，装配 provider、工具注册表、执行管理器与提示词缓存状态，并提供原生工具调用循环等通用能力。 |
| `default_tool_protocol` | 把当前 profile 的 `tool_mode` 翻译成 `native` / `react` / `None` 三种运行时协议。 |
| `_save_history` | 把本轮对话压缩、裁剪后写入按 profile 分区的历史缓存并同步 `self.history`。 |
| `_dispatch_model_call` | 执行一次模型请求，优先走流式通道并采集首包与完成耗时。 |
| `_log_first_chunk` | 流式回调闭包，在收到第一个数据块时记录时间戳。 |
| `run_auto` | provider 无关的统一入口，按协议分发到 `run_with_tools` 或 `run_with_react`。 |
| `__init__` | 校验入参并装配 provider 体系、工具体系、提示词、历史与缓存状态。 |
| `run` | 抽象方法，声明子类必须实现的「单次查询」契约。 |
| `register_tool` | 把工具注册进运行时工具表并可选择落库保存 spec。 |
| `register_hot_tool` | 以热区方式注册工具，避免改写已缓存的提示词前缀。 |
| `unregister_tool` | 移除工具并正确处理冻结区/热区的缓存失效与 `cache_epoch` 递增。 |
| `_prompt_fingerprint` | 用 `schema_hash` 加 `description` 生成工具在提示词中的渲染指纹。 |
| `_sync_frozen_manifest` | 在首次请求前一次性冻结工具清单，作为字节稳定的提示词前缀。 |
| `discover_tools` | 扫描工具包并同步发现的工具，返回并保存发现报告。 |
| `is_tool_registered` | 查询工具是否已注册，可选校验版本与 schema 哈希。 |
| `tool_registration_status` | 返回某个工具的注册状态详情字典，用于诊断与审计。 |
| `execute_tool_calls` | 把一批工具调用交给执行管理器批量执行。 |
| `tool_definitions` | 把已注册工具（或指定子集）导出为 OpenAI function 定义列表。 |
| `_definitions_for_registrations` | 将工具映射稳定排序并转换成 function 定义与别名反查表。 |
| `_openai_tool_name` | 把带点号的内部工具名替换成 provider 安全的双下划线别名。 |
| `run_with_tools` | 原生 function-calling 主循环：请求模型、执行工具、回写结果直到得到最终答案。 |
| `_load_catalog_result` | 解析目录工具返回的 spec，把对应工具加入下一轮加载顺序（延迟加载）。 |
| `_configured_prompt_messages` | 把 `self.prompt` 里的角色提示词转成消息前缀，工具说明转为额外 system 消息。 |
| `_with_registered_tool_names` | 在消息前插入包含全部工具名清单与本轮工具完整契约的 system 消息。 |
| `_validate_prompt_cache_key` | 校验并归一化 prompt cache 路由键（非空、不超过 64 字符）。 |
| `_validate_prompt_cache_retention` | 校验 prompt cache 保留策略只能是 `in_memory`、`24h` 或 `None`。 |
| `_default_prompt_cache_key` | 用版本、provider、模型、模式、配置提示词、工具名与 `cache_epoch` 摘要出确定性缓存键。 |
| `_completion_target` | 解析本次请求使用的 LLM 客户端、模型与历史分区，并按 `(profile, model)` 缓存客户端。 |
| `set_system_prompt` | 设置 system 提示词的公开便捷方法。 |
| `_set_prompt` | 所有提示词写入的统一入口，做非空校验后写入 `self.prompt`。 |
| `_strict_function_schema` | 把 Pydantic 风格 schema 深拷贝并规范化成 OpenAI strict function 形式。 |
| `normalize` | 递归工作单元，删除 `default`、强制 `additionalProperties = False` 并补全 `required`。 |
| `compress_saved_history` | 把超大的 `Observation:` 与 `tool` 载荷替换成带预览和重查提示的存根。 |
| `trim_saved_history` | 按条数上限裁剪历史，保留开头 system 块并丢弃孤立的队首观察消息。 |
| `_observation_stub` | 生成压缩存根文本，含原始大小、预览和重新查询提示。 |
