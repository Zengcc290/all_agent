# web/support.py

## 一、这个文件是干什么的

这个文件是 Web 层的「支撑设施」模块，本身不定义任何 FastAPI 路由，也不实现任何具体业务算法，它的职责是把 Web API、Agent 工具、RAG 管道、问答抽取这四条链路需要用到的**共享单例与路径约定**集中到一处，让所有调用方拿到的是同一份资源实例。

具体来说，它做了四件事：

1. **路径与配置常量**：在模块导入时就把 `PROJECT_ROOT`、`WEB_DIR`、`STATIC_DIR`、`DB_PATH` 固定下来。其中 `DB_PATH` 直接取 `memory.base.default_sqlite_path()` 的返回值，这是「记忆共享」的数据面保证——Web API 与 Agent 工具读写同一个 SQLite 记忆库，`MEMORY_DB_PATH` 是唯一保留的路径覆盖入口。
2. **进程级单例**：`get_manager()` 提供唯一的 `MemoryManager`，`get_pipeline()` 提供唯一的 `RAGPipeline`，`get_agent()` 提供唯一的 ReActAgent（知识管家）。三者都用「先判空、再加锁、再判空」的双重检查锁模式实现懒加载，避免并发首次调用时创建出多份实例。
3. **图缓存失效信号**：`GRAPH_REVISION` 是一个整数版本号，配合 `bump_graph_revision()` / `graph_revision()` 两个带锁读写函数，让知识星图的缓存在有新的图事实写入时失效，而不需要给后台抽取线程加锁。
4. **聊天链路策略**：`SYSTEM_PROMPT`、`CHAT_CONFIRMED_TOOLS`、`CHAT_TOOL_ALLOWLIST`、`SEARCH_TOOL_NAME`、`chat_confirmed_side_effects()`、`search_available()`、`chat_tool_names()`、`chat_ready()` 一起决定了「前端知识管家能看见哪些工具、哪些写操作可以自动放行、模型没配好时给用户什么提示」。
5. **问答落库与图抽取**：`record_qa()` 把一次问答写成 episodic 记忆，`extract_graph_patches()` 系列函数把问答转成知识图谱补丁，并且刻意把抽取放在响应返回之后异步执行，不拖慢聊天延迟。

它被 `web/app.py`（或同层路由模块）、Agent 工具构造流程、种子脚本、后台抽取线程等地方导入使用；文件顶部的模块 docstring 明确写了「Web 与 Agent 工具同一份」这一设计约束。

## 二、函数与类逐条详解

> 说明：本文件**没有定义任何 class**，全部是模块级函数与模块级变量。下面按代码出现顺序逐个讲解 16 个函数。

### `build_knowledge_extractor()` （第 46 行）

- **作用**：构造并返回一个「知识抽取器」实例，供 RAG 管道使用。它要解决的问题是：抽取器依赖真实的大模型（LLM）配置，但很多运行环境（测试、离线部署、没填 key 的开发机）根本没有可用的模型，此时程序不应该崩，而应该退化成一个什么都不做的空实现。因此这个函数先尝试按 `ProviderRegistry` 读配置、解析 API Key、创建 `LLM` 客户端，一切顺利才返回真正会调用大模型的 `LLMKnowledgeExtractor`；任何一步不满足条件就返回 `NullKnowledgeExtractor`。它每次被调用都会重新读配置并新建客户端，因此不是单例，`get_pipeline()` 和 `extract_graph_patches()` 各自调用它一次。
- **参数**：无参数。
- **返回**：返回一个抽取器对象。成功时是 `memory.rag.LLMKnowledgeExtractor`（构造参数为 `client.complete` 这个可调用对象、`model` 为 provider 的默认模型名、`vision_model` 为 `config/services.toml` 中 `[vision].model` 或回退到默认模型）；失败或未配置时是 `memory.rag.NullKnowledgeExtractor`（一个不做任何抽取的安全兜底对象）。没有返回 `None` 的分支。
- **内部流程**：
  1. 函数内部延迟导入 `agents.llm.LLM` 与 `agents.providers.ProviderRegistry`，避免模块导入期的循环依赖与不必要的开销。
  2. 调用 `ProviderRegistry.default_config_path()` 拿到配置路径；判断 `path.name != "provider.toml"` 或 `not path.is_file()`，即配置文件名不是 `provider.toml`（说明只回退到了 `provider.example.toml` 占位文件）或文件根本不存在，直接返回 `NullKnowledgeExtractor()`。
  3. 在 `try` 块中构造 `ProviderRegistry(path)`，用 `registry.active_profile` 取出当前激活的 profile，再调 `registry.resolve_api_key(profile.name)` 解析真实 key。
  4. 如果 key 为空，或者 key 以字符串 `"replace-with"` 开头（说明还是示例里的占位符），记录一条 `LOGGER.warning("knowledge extractor disabled: provider api_key missing")` 并返回 `NullKnowledgeExtractor()`。
  5. 用 `api_key`、`profile.base_url`、`profile.default_model` 三个参数构造 `LLM` 客户端。
  6. 调用 `load_services_config().vision.model` 读取视觉模型名，用 `or ""` 归一化为字符串；注释说明模型选择统一放在 `config/services.toml` 的 `[vision]` 段，`.env` 不再承载模型名。
  7. 返回 `LLMKnowledgeExtractor(client.complete, model=profile.default_model, vision_model=vision_model or profile.default_model)`——注意把 `client.complete` 这个**绑定方法**作为回调传进去，而不是传整个 client，这样抽取器只需要一个「补全函数」。
  8. `try` 块内任意步骤抛异常（配置解析失败、网络无关的构造错误等）都会被 `except Exception` 捕获，用 `LOGGER.exception(...)` 记录完整堆栈，然后返回 `NullKnowledgeExtractor()`。
- **异常/边界**：函数对外**从不抛异常**，所有异常都在内部被吞掉并降级为空抽取器。边界情况包括：配置文件缺失、文件名是示例文件、key 为空字符串、key 是 `replace-with` 开头的占位符、`load_services_config()` 读取失败、`LLM` 构造失败。vision 模型名为空时会回退到 `profile.default_model`。
- **同文件关系**：被 `get_pipeline()` 调用（构造 RAG 管道时传入 `extractor=`），也被 `extract_graph_patches()` 调用（每次抽取时新建管道）。它自身不调用本文件的其它函数。

### `bump_graph_revision() -> None` （第 88 行）

- **作用**：把模块级全局计数器 `GRAPH_REVISION` 加一，作为「图事实发生了变化」的失效信号。知识星图的渲染结果会被缓存，只要有新的实体或关系写进图里，缓存就必须作废重建；与其给后台抽取线程加锁去直接操作缓存，不如只递增一个版本号，让缓存在读取时自己比对版本号决定是否重建。这样既避免了锁竞争，也避免了「抽取失败时白白重建一次星图」——只有真正写入了实体或关系才会调用本函数。
- **参数**：无参数。
- **返回**：返回 `None`，没有返回值。它的效果体现在全局变量 `GRAPH_REVISION` 被递增这一副作用上。
- **内部流程**：
  1. 用 `global GRAPH_REVISION` 声明要修改模块级变量。
  2. 进入 `with _graph_revision_lock:` 临界区，保证多线程并发递增不会丢更新。
  3. 执行 `GRAPH_REVISION += 1`。
- **异常/边界**：无特殊处理。整数递增不会溢出（Python 整数无上限），锁也不会抛业务异常。理论上如果进程重启，计数会回到 0，但由于缓存同样在进程内存里，重启后缓存本就不存在，因此不构成问题。
- **同文件关系**：被 `extract_graph_patches()` 调用（当抽取报告里 `entities` 或 `relations` 非空时）。与 `graph_revision()` 共用同一把 `_graph_revision_lock`，是它的写侧配对函数。

### `graph_revision() -> int` （第 94 行）

- **作用**：读取当前图版本号，供缓存层判断「我缓存的星图是否已经过期」。它和 `bump_graph_revision()` 是读写配对：抽取线程写、渲染请求读。加锁读取是为了在 CPython 之外的解释器实现或未来代码变动下仍然保证读到的是一个完整一致的值，同时保持与写侧使用同一把锁的对称性。
- **参数**：无参数。
- **返回**：返回当前全局变量 `GRAPH_REVISION` 的整数值。首次导入后、还没有任何图写入时返回 0；每成功写入一次图事实（含实体或关系）就增大一次。
- **内部流程**：
  1. 进入 `with _graph_revision_lock:` 临界区。
  2. 直接 `return GRAPH_REVISION`，把模块级变量的当前值返回给调用方。
- **异常/边界**：无特殊处理。不存在空值或非法值的情况，因为初值就是整数 0 且只做自增。
- **同文件关系**：与 `bump_graph_revision()` 配对（共用 `_graph_revision_lock`）。在本文件内部没有被其它函数调用，它的调用方在 Web 渲染层（星图缓存）中。

### `get_manager() -> MemoryManager` （第 99 行）

- **作用**：返回进程级的 `MemoryManager` 单例，这是整个系统「记忆共享」的核心入口。Web API 的读写、RAG 管道、Agent 的记忆工具、种子脚本全都通过它拿到同一个管理器，从而操作同一个 SQLite 记忆库和同一套向量/图后端。之所以要单例而不是各模块各自 `MemoryManager(...)`，是因为每份实例都会持有自己的连接与嵌入配置，多份实例会导致「Agent 工具写进去的记忆，Web API 查不到」这类漂移问题。
- **参数**：无参数。
- **返回**：返回 `memory.MemoryManager` 实例。第一次调用时创建并缓存到模块级 `_manager`；后续调用直接返回同一个对象。永远不会返回 `None`。
- **内部流程**：
  1. 用 `global _manager` 声明要写模块级变量。
  2. 第一次判空：`if _manager is None:`，这是无锁快路径，避免每次调用都抢锁。
  3. 进入 `with _manager_lock:` 后**再判一次空**（双重检查锁），防止两个线程同时通过第一次判断、各自创建一份实例。
  4. 在锁内调用 `MemoryConfig.from_config()` 读取 `config/services.toml` 的全套配置（Qdrant / Neo4j 的开关在这里生效；未配置时行为与旧版完全一致：内存向量 + 内存图）。
  5. 把 `config.sqlite_path` 显式覆写为模块级常量 `DB_PATH`，确保无论配置文件里写了什么，都落到那个由 `default_sqlite_path()` 决定的统一路径上。
  6. 用该 config 构造 `MemoryManager(config)` 并赋给 `_manager`。
  7. 返回 `_manager`。
- **异常/边界**：如果 `MemoryConfig.from_config()` 或 `MemoryManager(...)` 抛异常（配置非法、目录不可写等），异常会**直接向上抛出**，不会被吞掉；此时 `_manager` 仍是 `None`，下次调用会重新尝试创建。本函数没有对 `DB_PATH` 做存在性校验，路径创建由被调用的库负责。
- **同文件关系**：被 `get_pipeline()`（取 manager 构造管道）、`get_agent()`（把 manager 注入四个记忆工具）、`extract_graph_patches()`（`manager is None` 时的默认回退）调用。它调用本文件的常量 `DB_PATH`，但不调用本文件其它函数。与 `close_manager()` 是配对的生命周期函数。

### `close_manager() -> None` （第 113 行）

- **作用**：关闭并释放进程级 `MemoryManager` 单例，通常由 Web 应用的关闭钩子（shutdown 事件）调用，用来优雅释放 SQLite 连接、向量库客户端、图库连接等资源。把 `_manager` 置回 `None` 也是为了让「关闭后如果又被调用」能够重新创建实例，而不是持有一个已经关掉的僵尸对象。
- **参数**：无参数。
- **返回**：返回 `None`。
- **内部流程**：
  1. 用 `global _manager` 声明要写模块级变量。
  2. 判断 `if _manager is not None:`，只有存在实例时才处理。
  3. 调用 `_manager.close()` 释放底层资源。
  4. 把 `_manager = None` 复位。
- **异常/边界**：如果 `close()` 内部抛异常，异常会向上传播，且 `_manager = None` 这一行**不会执行**（因为它在 `close()` 之后），于是全局变量仍指向那个可能已半关闭的实例——这是一个需要注意的边界行为。若 `_manager` 本来就是 `None`，函数直接什么都不做，属于幂等调用。
- **同文件关系**：与 `get_manager()` 配对。本文件内部没有调用它；调用方是 Web 应用的关闭流程。需要注意 `extract_graph_patches()` 的 docstring 特别警告：后台线程绝不能再调 `get_manager()`，因为关闭时全局单例先被关闭，后台线程再去抢同一把锁就会永久挂住。

### `get_pipeline() -> RAGPipeline` （第 129 行）

- **作用**：返回进程级的 `RAGPipeline` 单例，即检索增强生成管道。Web API 的检索接口、知识管家的 `memory.rag_search` / `memory.rag` 工具、问答抽取都共用这一份管道，保证它们背后是同一个 `MemoryManager`（同一份记忆库）和同一套抽取器配置。管道构造有一定成本（要建抽取器、拿管理器），所以只建一次。
- **参数**：无参数。
- **返回**：返回 `memory.rag.RAGPipeline` 实例。首次调用时创建并缓存到模块级 `_pipeline`；之后返回同一对象。不会返回 `None`。
- **内部流程**：
  1. 用 `global _pipeline` 声明。
  2. 第一次判空 `if _pipeline is None:` 作为无锁快路径。
  3. 在判空之后、进入锁之前先调用 `manager = get_manager()` 拿到管理器实例（这里复用 `get_manager()` 的双重检查锁）。
  4. 进入 `with _manager_lock:`——注意这里复用的是 **manager 的锁**而不是为 pipeline 单独建一把锁；再判一次 `if _pipeline is None:`。
  5. 用 `RAGPipeline(manager, extractor=build_knowledge_extractor())` 构造管道并赋给 `_pipeline`。
  6. 返回 `_pipeline`。
- **异常/边界**：`get_manager()` 或 `RAGPipeline(...)` 抛出的异常会向上传播，此时 `_pipeline` 保持 `None`，下次调用重试。`build_knowledge_extractor()` 自身不会抛异常（内部已兜底），所以抽取器配置缺失不会导致这里失败。没有超时或空值处理逻辑。
- **同文件关系**：调用 `get_manager()` 和 `build_knowledge_extractor()`。被 `get_agent()` 调用两次（分别注入 `RAGSearchTool` 与 `RAGTool`）。注意 `extract_graph_patches()` **没有**复用它，而是自己新建管道，这是有意的——后台线程不能碰全局单例。

### `get_agent()` （第 143 行）

- **作用**：懒加载并返回「知识管家」这个 `ReActAgent` 单例，也就是前端聊天页面背后真正干活的 Agent。它在构造时做了三件关键配置：关闭自动工具发现、注入本文件组合出来的 `SYSTEM_PROMPT`、把记忆类工具全部替换成显式注入了 Web 单例后端的版本。docstring 特别说明：即使真实聊天模型没有配置，构造本身也能成功（因为构造过程不联网、不校验 key），真正的可用性由 `chat_ready()` 在调用前把关。
- **参数**：无参数。
- **返回**：返回一个 `agents.ReActAgent` 实例。首次调用时创建并缓存到模块级 `_agent`；之后返回同一对象。不会返回 `None`。
- **内部流程**：
  1. 用 `global _agent` 声明。
  2. 第一次判空 `if _agent is None:` 作为无锁快路径。
  3. 进入 `with _agent_lock:` 后再判一次空（双重检查锁，用的是 agent 专属的锁 `_agent_lock`，与 manager/pipeline 的锁分开）。
  4. 在锁内延迟导入 `ReActAgent` 以及六个工具类：`MemoryAddTool`、`MemoryQueryTool`、`MemoryManageTool`、`RAGSearchTool`、`RAGTool`、`SearchTool`。
  5. `ReActAgent("knowledge-butler", auto_discover_tools=False)` —— 名字固定为 `knowledge-butler`，并且**关闭自动发现**，注释说明前端知识管家只挂检索/写入记忆的工具，不自动发现 `fs` / `update_log` / `current_time` 这些项目脚手架工具。
  6. `agent.set_system_prompt(SYSTEM_PROMPT)` 注入模块级组合好的系统提示词。
  7. 依次 `register_tool(..., replace=True)` 注册六个工具，`replace=True` 表示如果同名工具已存在就替换掉：
     - `MemoryQueryTool(manager=get_manager())`
     - `MemoryAddTool(manager=get_manager())`
     - `MemoryManageTool(manager=get_manager())`
     - `RAGSearchTool(pipeline=get_pipeline())`
     - `RAGTool(pipeline=get_pipeline())`
     - `SearchTool()`
     注释指出五个记忆工具统一注入 Web 单例后端，是为了避免发现机制各自创建的默认连接与嵌入配置和 Web API 漂移。
  8. 赋值 `_agent = agent`，然后在锁外 `return _agent`。
- **异常/边界**：导入失败、工具构造失败、`register_tool` 失败都会向上抛异常，此时 `_agent` 仍为 `None`，下次调用会重试整个构造过程。没有对模型未配置做特殊处理（那是 `chat_ready()` 的职责）。由于是懒加载 + 单例，构造只发生一次，但意味着**如果第一次构造抛异常，缓存不会被污染**。
- **同文件关系**：调用 `get_manager()`（三次，构造三个记忆工具）、`get_pipeline()`（两次，构造两个 RAG 工具），并使用模块级常量 `SYSTEM_PROMPT`。本文件内部没有其它函数调用它；调用方是 Web 聊天路由层。

### `chat_confirmed_side_effects(agent, *, destructive_call: dict[str, Any] | None = None) -> frozenset[str]` （第 185 行）

- **作用**：为一次聊天回合计算「已确认的副作用钥匙集合」，返回的 `frozenset` 会被传给 Agent 的确认机制，表示这些操作已经被授权执行、不必再弹人工确认。它包含两类：一类是固定放行的**增量写入**（`CHAT_CONFIRMED_TOOLS` 里的 `memory.add`，对应「记住这件事」这一唯一允许自动执行的写操作）；另一类是可选的、针对**某一次精确调用**的破坏性操作确认（由 `destructive_call` 描述）。第二类的 key 是从**校验并归一化后的参数**推导出来的，因此模型无法通过替换另一个 id 来蒙混过关，也无法把 `delete` 升级成 `clear`。模块注释强调：delete/clear/ingest 仍需人工确认，提示词注入最多让模型多记一条，不能删库或改库。
- **参数**：
  - `agent`：位置参数，Agent 实例。函数会在它身上依次尝试两个查找位置：`agent.tool_confirmation_key` 属性，以及 `agent.tools.confirmation_key`（即 `getattr(getattr(agent, "tools", None), "confirmation_key", None)`）。只要其中一个可调用就使用它。要求该对象支持 `agent.tools.resolve(name)` 与 `agent.tools.call_confirmation_key(name, normalized)` 两个方法（在 `destructive_call` 分支中会用到）。
  - `destructive_call`：仅关键字参数，类型 `dict[str, Any] | None`，默认 `None`。当不为 `None` 时应当是一个形如 `{"tool_name": ..., "arguments": {...}}` 的字典，用来描述这次要额外放行的破坏性调用。约束：`tool_name` 必须恰好是 `"memory.manage"`（其它名字会被判非法），`arguments` 必须是 `dict`。
- **返回**：返回 `frozenset[str]`，即一组确认钥匙字符串。集合内容取决于执行过程：正常情况至少包含 `memory.add` 的确认 key（如果 lookup 成功）；如果某个工具的 lookup 抛出 `KeyError`/`ValueError`/`TypeError`，那个 key 会被跳过并记一条 warning；`destructive_call` 合法时再追加一个精确调用的 key；`destructive_call` 非法时被忽略，只记 warning。极端情况下可能返回空集合（所有 lookup 都失败且没有有效的 destructive_call）。
- **内部流程**：
  1. 初始化 `keys: set[str] = set()`。
  2. 外层 `for name in CHAT_CONFIRMED_TOOLS:` 遍历固定放行清单（当前只有 `memory.add`）。
  3. 内层 `for lookup in (...)` 依次尝试两个候选的可调用对象：`agent.tool_confirmation_key` 和 `agent.tools.confirmation_key`。
  4. 若 `not callable(lookup)` 则 `continue` 换下一个候选。
  5. 用 `try` 调用 `lookup(name)`，成功就把结果 `keys.add(...)`，然后 `else: break` —— 注意这里的 `break` 挂在 `try/except/else` 的 `else` 上，语义是「成功取到 key 就跳出候选循环，不再尝试第二个候选」。
  6. 若抛出 `(KeyError, ValueError, TypeError)` 之一，`continue` 去尝试下一个候选。
  7. 如果两个候选都不可用或都失败，`for...else` 的 `else` 分支执行：`LOGGER.warning("chat confirmation key unavailable for tool %s", name)`。
  8. 处理 `destructive_call`：`if destructive_call is not None:` 进入 `try`：
     - `name = str(destructive_call["tool_name"])`（取不到 key 会抛 `KeyError`）；
     - `arguments = destructive_call["arguments"]`；
     - `if name != "memory.manage" or not isinstance(arguments, dict): raise ValueError("only memory.manage can be confirmed here")`——双重校验工具名和参数类型；
     - `tool, _ = agent.tools.resolve(name)` 解析出工具对象（注意解包成二元组，取第一个元素）；
     - `normalized = tool.spec.input_model.model_validate(arguments, strict=True).model_dump(mode="json")`——用工具的 Pydantic 输入模型做**严格模式**校验，再转成纯 JSON 可序列化字典，这就是「归一化参数」；
     - `keys.add(agent.tools.call_confirmation_key(name, normalized))` 用归一化后的参数算 key。
  9. `except (KeyError, TypeError, ValueError):` 捕获非法输入（缺字段、名字不对、参数不是字典、校验失败等），`LOGGER.warning("invalid destructive chat confirmation ignored")`，不追加任何 key。
  10. 最后 `return frozenset(keys)` 把可变集合转成不可变集合返回，防止调用方篡改。
- **异常/边界**：函数对外**不抛异常**。关键边界：`agent` 上两个 lookup 位置都不存在或不可调用 → 记 warning、跳过；lookup 抛 `KeyError`/`ValueError`/`TypeError` → 跳过并尝试下一候选；`destructive_call` 缺 `tool_name`/`arguments` 键、工具名不是 `memory.manage`、参数不是 dict、Pydantic 严格校验失败 → 整体忽略并记 warning。注意 `model_validate(..., strict=True)` 意味着类型不严格匹配（例如把字符串塞进 int 字段）也会失败，这正是防止模型绕过确认的设计意图。返回的 `frozenset` 在没有任何成功项时为空集，调用方需要能接受空集。
- **同文件关系**：使用本文件模块级常量 `CHAT_CONFIRMED_TOOLS` 和模块级 `LOGGER`。它不调用本文件其它函数；本文件内部也没有函数调用它，调用方是 Web 聊天路由层。

### `search_available() -> bool` （第 241 行）

- **作用**：判断联网搜索能力（AnySearch）是否已经配置好，只有 `base_url` 与 `api_key` **同时**存在才视为可用。前端聊天有一个「联网/非联网」开关，如果用户开了联网但后端根本没配搜索服务，就不应该把 `web.search` 工具暴露给模型，否则模型会去调一个必然失败的工具。docstring 说明唯一来源是 `config/services.toml` 的 `[search]` 段——这是外部 API 调用的集中配置，历史上基于 `SEARCH_*` / `ANYSEARCH_*` 环境变量的入口已经被删除。
- **参数**：无参数。
- **返回**：返回 `bool`。`load_services_config().search` 的 `base_url` 和 `api_key` 都非空（在布尔语境下为真）时返回 `True`，否则返回 `False`。
- **内部流程**：
  1. 调用 `load_services_config()` 加载服务配置对象。
  2. 取出 `.search` 段。
  3. `return bool(search.base_url) and bool(search.api_key)` —— 用 `bool()` 显式归一化，保证返回的是真正的布尔值而不是字符串或 `None`，并且要求两个字段同时为真（短路求值：`base_url` 为空时不会再看 `api_key`）。
- **异常/边界**：如果 `load_services_config()` 本身抛异常（配置文件缺失或格式错误），异常会向上传播，本函数没有做兜底。对空字符串、`None`、缺失字段的处理由 `bool()` 承担：统一视为不可用。
- **同文件关系**：被 `chat_tool_names()` 调用（决定是否追加 `web.search`）。它自身不调用本文件其它函数。

### `chat_tool_names(agent, *, online: bool) -> list[str]` （第 251 行）

- **作用**：按当前聊天模式计算本次要暴露给 LLM 的工具名清单，结果会作为 `tool_names=` 参数传给 `agent.run(...)`。它实现了一条白名单策略：知识管家永远只能看见记忆/图/向量检索相关的工具，项目脚手架工具不进这一层；并且 docstring 特别强调 `None` 不再表示「发现到的全部工具」，因此必须显式给出清单，避免默认放开。联网模式且 AnySearch 确实已配置时，才额外把 `web.search` 加进去。
- **参数**：
  - `agent`：位置参数，Agent 实例。要求它有一个 `agent.tools` 对象，且该对象提供 `snapshot()` 方法（返回当前已注册工具名的可迭代集合）。函数会调用 `snapshot()` **两次**（一次在列表推导里，一次在末尾的 `in` 判断里）。
  - `online`：仅关键字参数，类型 `bool`，无默认值，必须显式传入。表示前端聊天是否处于「联网」模式。`True` 时才可能追加搜索工具。
- **返回**：返回 `list[str]`，元素顺序为：先按 `agent.tools.snapshot()` 的迭代顺序输出所有既在 `CHAT_TOOL_ALLOWLIST` 里、又不是 `SEARCH_TOOL_NAME` 的工具名；如果联网且搜索可用且 `web.search` 确实已注册，则把 `SEARCH_TOOL_NAME` **追加到列表末尾**。可能返回空列表（如果没有任何白名单工具被注册）。
- **内部流程**：
  1. 用列表推导构建 `names`：遍历 `agent.tools.snapshot()`，保留满足 `name in CHAT_TOOL_ALLOWLIST and name != SEARCH_TOOL_NAME` 的名字。这里显式排除搜索工具，是为了让搜索工具只受下面的联网开关控制，而不是因为它恰好在白名单里就被无条件放出来。
  2. 判断 `if online and search_available() and SEARCH_TOOL_NAME in agent.tools.snapshot():` —— 三个条件用 `and` 短路连接：必须联网、必须搜索已配置、必须该工具真的被注册过。
  3. 条件成立时 `names.append(SEARCH_TOOL_NAME)`。
  4. `return names`。
- **异常/边界**：`agent` 没有 `tools` 属性、`snapshot()` 不存在或抛异常时，异常向上传播，本函数不兜底。对空注册表返回空列表。`online=False` 时永远不会包含搜索工具。注意本函数不校验 `SEARCH_TOOL_NAME` 是否真的实现了可用逻辑，只看它是否已注册——真正的可用性由 `search_available()` 的配置判断把关。
- **同文件关系**：调用 `search_available()`，并使用本文件模块级常量 `CHAT_TOOL_ALLOWLIST`、`SEARCH_TOOL_NAME`。本文件内部没有函数调用它；调用方是 Web 聊天路由层。

### `record_qa(manager: MemoryManager, question: str, answer: str, *, mode: str) -> MemoryItem` （第 267 行）

- **作用**：把一次问答作为 episodic（情景）记忆写进记忆库，并返回写入的记忆条目。这样做的价值是让「我这两天问过什么」「上次我问的那个问题」这类**基于时间的回顾型提问**能够被 `memory.query` / `memory.search` 检索到——如果问答只留在前端会话里、不落库，Agent 就无法回忆。写入内容以「问：…／答：…」的文本形式组织，同时 metadata 里保留结构化字段（问题、答案、模式、时间），星图时间线上则以「问：…」作为事件标题出现。
- **参数**：
  - `manager`：位置参数，`MemoryManager` 实例。函数通过它访问 `manager.episodic.record(...)`，因此要求该管理器有 `episodic` 子模块且支持 `record` 方法。调用方应传入共享单例（通常来自 `get_manager()`），以保证写入的是同一份记忆库。
  - `question`：位置参数，字符串（或可被 `str()` 转换的对象）。会先做空值兜底和长度截断，最多保留 `constants.WEB_QA_QUESTION_MAX_CHARS` 个字符。
  - `answer`：位置参数，字符串（或可被 `str()` 转换的对象）。同样做空值兜底和截断，最多保留 `constants.WEB_QA_ANSWER_MAX_CHARS` 个字符。
  - `mode`：仅关键字参数，字符串，无默认值。表示这次问答的运行模式（例如同步/流式、联网/非联网等），会被原样存进 metadata 的 `mode` 字段，便于事后按模式筛选。
- **返回**：返回 `memory.MemoryItem`，即 `manager.episodic.record(...)` 的返回值，代表刚落库的那条情景记忆条目（含 id、时间戳、内容等字段）。
- **内部流程**：
  1. `now = utc_now()` 取当前 UTC 时间，后续时间戳和 metadata 都复用它，保证同一条记忆内部时间一致。
  2. `question = str(question or "")[:WEB_QA_QUESTION_MAX_CHARS]` —— `question or ""` 把 `None` 和空字符串统一成 `""`，`str()` 兜住非字符串输入，切片做长度上限。
  3. `answer = str(answer or "")[:WEB_QA_ANSWER_MAX_CHARS]` 对答案做同样处理。
  4. 调用 `manager.episodic.record(...)`，参数为：
     - 位置参数正文：`f"问：{question}\n答：{answer}"`，用换行分隔问答，便于将来按「问/答」文本检索。
     - `metadata` 字典：`kind="qa"`、`type="qa"`（两个字段都设，兼容不同检索口径）、`title=f"问：{question[:NEBULA_EVENT_TITLE_CHARS]}"`（标题用问题前若干字符，长度受 `NEBULA_EVENT_TITLE_CHARS` 限制，供星图时间线展示）、`question`（完整截断后的问题）、`answer`（完整截断后的答案）、`mode`（原样传入的模式）、`asked_at=now.isoformat()`（ISO 8601 字符串形式的时间）。
     - `timestamp=now`：把同一个时间对象作为记忆时间戳传入。
  5. 直接把 `record(...)` 的返回值 `return` 出去，不做二次包装。
- **异常/边界**：如果 `manager.episodic.record(...)` 抛异常（数据库不可写、磁盘满等），异常向上传播，本函数不捕获。边界处理集中在输入侧：`None`、空字符串、超长字符串都被归一化或截断；`question` 为空时仍会写入一条「问：\n答：…」的记忆，函数**不会**因为空问题而跳过写入。截断可能把一个多字节字符序列切断在字符边界上（Python 切片按字符，不会切坏编码）。`asked_at` 使用 ISO 格式字符串，而 `timestamp` 使用 datetime 对象，两者语义相同但类型不同。
- **同文件关系**：使用本文件导入的常量 `WEB_QA_QUESTION_MAX_CHARS`、`WEB_QA_ANSWER_MAX_CHARS`、`NEBULA_EVENT_TITLE_CHARS` 和 `memory.utc_now`。它不调用本文件其它函数；本文件内部也没有函数调用它，调用方是 Web 问答路由层（通常与 `schedule_qa_extraction()` 配合，先落库再触发抽取）。

### `knowledge_extract_enabled() -> bool` （第 297 行）

- **作用**：返回「问答是否要触发图抽取」这个开关的当前值。之所以把它单独包成一个函数而不是让调用方直接读常量，是为了给测试留一个稳定的 monkeypatch 点——docstring 明确写了测试会 monkeypatch `constants.WEB_QA_EXTRACT`。当没有真实聊天模型时，这个开关实际上应该处于关闭状态（没有抽取器可用，抽取只会白跑一次），判断逻辑通过读取该常量体现。
- **参数**：无参数。
- **返回**：返回 `bool`，即模块级导入的常量 `constants.WEB_QA_EXTRACT` 的当前值。注意它在函数体内**按名字读取模块全局**，所以即使常量值被替换，再次调用也能读到新值（这正是 monkeypatch 生效的机制）。
- **内部流程**：只有一步——`return WEB_QA_EXTRACT`，直接返回导入进来的常量。没有任何判断、循环或副作用。
- **异常/边界**：无特殊处理。常量是布尔量，不存在空值或非法值的情况；函数不读配置、不访问文件，因此不会抛异常。
- **同文件关系**：被 `schedule_qa_extraction()` 作为第一道闸门调用。它自身不调用本文件其它函数。

### `extract_graph_patches(question: str, answer: str, *, manager: MemoryManager | None = None) -> dict[str, Any] | None` （第 305 行）

- **作用**：把一次问答交给 LLM 抽取成知识图谱补丁并落库，最后返回本次抽取的报告字典。它只负责「图侧增量」——问答原文的 episodic 留痕由调用方自己负责。抽取被**刻意排除在聊天延迟之外**：调用方应当在 HTTP 响应返回之后再调用它。抽取范围也做了约束：只从「用户陈述」和「助手依据知识库给出的事实」里抽取，避免把模型自己的推测固化成图里的边。函数会新建一个 `RAGPipeline`（而不是复用全局单例），并通过 `manager` 参数接收调用方自己的管理器——docstring 特别警告：后台线程绝不能再调 `get_manager()`，因为 Web 应用关闭时全局单例会先被关闭，后台线程再抢同一把锁就会永久挂住。
- **参数**：
  - `question`：位置参数。期望是 `str`，会被 `strip()` 后拼进抽取文本。若不是字符串类型，函数直接返回 `None`（不做强制转换）。
  - `answer`：位置参数。期望是 `str`，同样会被 `strip()` 后拼进抽取文本。非字符串类型直接返回 `None`。
  - `manager`：仅关键字参数，类型 `MemoryManager | None`，默认 `None`。传入时用它构造管道；为 `None` 时回退到全局单例 `get_manager()`（适合前台同步调用的场景）。后台线程必须显式传入。
- **返回**：返回 `dict[str, Any] | None`。
  - `question` 或 `answer` 不是字符串 → `None`；
  - 任一去空白后为空字符串 → `None`；
  - 抽取过程抛异常 → 记 `LOGGER.exception` 后返回 `None`；
  - 成功 → 返回 `pipeline.last_ingest_report` 的**浅拷贝字典**（`dict(...)`），里面通常含 `entities`、`relations` 等统计字段。
- **内部流程**：
  1. 类型校验：`if not isinstance(question, str) or not isinstance(answer, str): return None`。
  2. 空值校验：`if not question.strip() or not answer.strip(): return None`。
  3. 管理器回退：`if manager is None: manager = get_manager()`。
  4. `pipeline = RAGPipeline(manager, extractor=build_knowledge_extractor())` —— 每次调用都新建管道和抽取器，不复用 `_pipeline` 单例。
  5. 延迟导入 `from memory.rag import Document`（放在函数体内，避免模块级循环依赖）。
  6. 拼装抽取文本 `text`：先是字面量 `"【用户陈述】\n"`，接 `question.strip()`；再是提示性字面量 `"【助手回答（只抽取其中依据知识库给出的事实，推测性表述不要抽取）】\n"`，接 `answer.strip()`。这段提示直接写进被抽取的文档正文里，用来约束 LLM 的抽取范围。
  7. 在 `try` 块中调用 `pipeline.ingest(Document(text, metadata={"source": "问答抽取", "filename": "问答抽取", "kind": "qa"}), chunk_size=QA_EXTRACT_CHUNK_SIZE, overlap=0)` —— 用 `Document` 包住文本，metadata 标注来源、文件名、种类，分块大小取常量 `QA_EXTRACT_CHUNK_SIZE`，`overlap=0` 表示分块之间不重叠。
  8. `except Exception:` 捕获**所有**抽取异常，`LOGGER.exception("QA graph extraction failed")` 记录堆栈，`return None`。注释说明「抽取失败不能影响问答本身」。
  9. `report = dict(pipeline.last_ingest_report)` —— 把管道上最近一次 ingest 的报告复制成普通字典（防御调用方修改管道内部状态）。
  10. `if report.get("entities") or report.get("relations"): bump_graph_revision()` —— 只有真的抽出了实体或关系才递增图版本号，从而让星图缓存失效；抽取空手而归时不动版本号，避免无谓重建。
  11. `return report`。
- **异常/边界**：类型不对或内容为空 → 静默返回 `None`（不记日志）。`pipeline.ingest` 的任何异常都被吞掉并记 `LOGGER.exception`，对外只表现为 `None`。注意**没有超时控制**：LLM 调用可能很慢，这也是为什么配套的 `extract_graph_patches_async()` 要用 `asyncio.to_thread` 把它挪出事件循环。`manager` 为 `None` 且 `get_manager()` 抛异常时，异常会在 try 块**之外**抛出，不会被吞掉——这是有意的，属于调用方的编程错误。`pipeline.last_ingest_report` 若为 `None`，`dict(None)` 会抛 `TypeError`（同样在 try 块之外），这依赖库保证该属性总是一个字典。
- **同文件关系**：调用 `get_manager()`（manager 为 `None` 时的回退）和 `build_knowledge_extractor()`（构造抽取器），并在有图事实时调用 `bump_graph_revision()`。被 `extract_graph_patches_async()` 与 `schedule_qa_extraction()` 调用。

### `extract_graph_patches_async(question: str, answer: str, *, manager: MemoryManager | None = None) -> None` （第 358 行）

- **作用**：`extract_graph_patches()` 的异步包装器。抽取过程包含阻塞式的 LLM 网络调用和 SQLite 写入，如果直接在事件循环里执行，会把整个 Web 服务的响应能力卡住。因此这个协程用 `asyncio.to_thread` 把同步函数丢到线程池里跑，事件循环立刻可以继续处理其它请求。它是「不阻塞聊天」这一设计目标的执行者。
- **参数**：
  - `question`：位置参数，原样透传给 `extract_graph_patches`。
  - `answer`：位置参数，原样透传。
  - `manager`：仅关键字参数，类型 `MemoryManager | None`，默认 `None`。以关键字形式透传给同步函数（`manager=manager`），因此同步函数收到的仍是关键字参数而非位置参数，这点与它的签名匹配。
- **返回**：返回 `None`。它**不返回**同步函数的抽取报告——报告在后台线程里被丢弃，只通过 `bump_graph_revision()` 的副作用让缓存失效。协程本身在 `await asyncio.to_thread(...)` 完成（线程里的同步函数执行完毕）之后才结束。
- **内部流程**：只有一步——`await asyncio.to_thread(extract_graph_patches, question, answer, manager=manager)`。`asyncio.to_thread` 把可调用对象和参数打包提交给默认线程池执行器，返回一个可等待对象；`await` 它即等待线程执行完成并把结果（这里是 `None` 或报告字典）丢弃。
- **异常/边界**：如果 `extract_graph_patches` 在 `to_thread` 中抛异常，该异常会被 `to_thread` 的 future 捕获并在 `await` 处**重新抛出**，从而传播到调用这个协程的地方。由于同步函数内部已经把 ingest 异常兜住了，这里能漏出来的主要是参数校验之外的意外错误（例如 `dict(None)` 的 `TypeError`）。如果协程被取消，`await` 会抛 `asyncio.CancelledError`；注意 `asyncio.to_thread` 无法真正中断已经在线程里运行的阻塞代码，取消只是让等待方提前返回。
- **同文件关系**：调用 `extract_graph_patches()`。被 `schedule_qa_extraction()` 通过 `asyncio.create_task(...)` 调度执行。它自身不使用本文件其它函数。

### `schedule_qa_extraction(question: str, answer: str, *, manager: MemoryManager | None = None) -> None` （第 370 行）

- **作用**：以「发射后不管」（fire-and-forget）的方式触发一次问答图抽取。它是 Web 问答路由层真正调用的入口：路由写完 episodic 记忆、把 HTTP 响应发出去之后，调用这个函数让抽取在后台发生，用户不会为 LLM 抽取等待。函数内部处理了三种运行环境的差异：抽取功能未开启、没有真实聊天模型、有/没有正在运行的事件循环。docstring 逐条列出了这三条规则。
- **参数**：
  - `question`：位置参数，原样透传给下游抽取函数。
  - `answer`：位置参数，原样透传。
  - `manager`：仅关键字参数，类型 `MemoryManager | None`，默认 `None`。一路以关键字形式透传；当它为 `None` 且最终走到内联/异步抽取时，同步函数会回退到 `get_manager()`。后台线程场景应由调用方显式传入自己的管理器，以避免 `get_manager()` 在关闭时的死锁风险。
- **返回**：返回 `None`。函数本身不返回抽取结果，也不返回任何句柄；异步模式下创建的 task 也没有被保存，因此调用方无法取消或等待它（这正是 fire-and-forget 的含义）。
- **内部流程**：
  1. `if not knowledge_extract_enabled(): return` —— 第一道闸门，抽取功能关闭时直接返回。注释说明没有真实聊天模型时也应跳过，因为那时没有抽取器，抽取只会白跑一次。
  2. `ready, _ = chat_ready()` —— 第二道闸门，检测聊天模型是否真的配好。用元组解包丢弃第二个提示字符串（`_`）。`if not ready: return`，模型不可用时直接跳过。
  3. `if WEB_QA_EXTRACT_SYNC:` —— 第三道分支，当常量 `constants.WEB_QA_EXTRACT_SYNC` 为真时改为**内联同步执行** `extract_graph_patches(question, answer, manager=manager)` 然后 `return`。docstring 说明测试与脚本用它拿到确定的执行顺序（通过 monkeypatch 该常量）。
  4. `try: asyncio.get_running_loop()` —— 探测当前线程是否有正在运行的事件循环。
  5. `except RuntimeError:` —— 没有事件循环时（例如纯同步脚本里调用），也改为内联执行 `extract_graph_patches(question, answer, manager=manager)` 并 `return`。注释说明这是为了「避免创建永远不跑的协程」——因为 `asyncio.create_task` 在没有运行中的循环时会直接报错。
  6. 走到这里说明有运行中的循环：`asyncio.create_task(extract_graph_patches_async(question, answer, manager=manager))` 把协程包装成任务交给循环调度，然后函数返回，不 `await`、不保存 task 引用。
- **异常/边界**：前两道闸门只是静默返回，不记日志。内联分支下 `extract_graph_patches` 的异常会**向上传播**给调用方（因为它被直接调用）；异步分支下异常发生在后台任务里，除非任务被回收时打印「Task exception was never retrieved」，否则不会打扰调用方。因为 task 引用没有被保存，存在被垃圾回收提前取消的理论风险（CPython 中事件循环持有弱引用）。`chat_ready()` 自身不抛异常，所以第二道闸门是安全的。
- **同文件关系**：调用 `knowledge_extract_enabled()`、`chat_ready()`、`extract_graph_patches()`、`extract_graph_patches_async()`，并使用常量 `WEB_QA_EXTRACT_SYNC`。本文件内部没有函数调用它；调用方是 Web 问答路由层。

### `chat_ready() -> tuple[bool, str]` （第 402 行）

- **作用**：检测「真实聊天模型是否已经配置好」，并给出可直接展示给用户的中文错误提示。它存在的根本原因是 `ProviderRegistry` 在 `config/provider.toml` 缺失时会**静默回退**到 `config/provider.example.toml`（里面的 key 是占位符），如果只检查「配置能加载」就会误判为可用，于是用户会看到一个能打开但一说话就报错的聊天界面。因此这里必须显式区分「文件真的是 provider.toml」和「key 不是 placeholder」。返回的二元组让调用方既能做布尔判断（`ready`），又能把提示原样回显到前端。
- **参数**：无参数。
- **返回**：返回 `tuple[bool, str]`，即 `(是否就绪, 提示信息)`。
  - `(False, "...")` 的几种情况：配置文件不是 `provider.toml` 或文件不存在，提示「未配置聊天模型：请复制 config/provider.example.toml 为 config/provider.toml，填入 api_key（或用 api_key_env 指向环境变量），然后重启服务。」；key 为空或以 `replace-with` 开头，提示「provider.toml 已存在但 api_key 为空/占位符，请填入真实 key。」；配置解析抛异常，提示 `f"provider.toml 解析失败：{type(exc).__name__}: {exc}"`（把异常类名和消息都带上，便于用户自查）。
  - `(True, "")`：一切正常时第二个元素是**空字符串**（不是 `None`），调用方可以直接用它做判空。
- **内部流程**：
  1. 延迟导入 `from agents.providers import ProviderRegistry, load_project_dotenv`。
  2. `load_project_dotenv()` —— 先把项目 `.env` 加载进环境变量，这样后面用 `api_key_env` 指向环境变量时才能读到值。
  3. `path = ProviderRegistry.default_config_path()` 拿到配置路径。
  4. `if path.name != "provider.toml" or not path.is_file():` —— 文件名不是 `provider.toml`（说明回退到了示例文件）或文件不存在，立即返回 `(False, 那段复制示例文件的中文提示)`。
  5. 进入 `try`：`registry = ProviderRegistry()` 用默认路径构造注册表；`profile = registry.get(registry.active_profile)` 取当前激活的 profile。
  6. `key = profile.api_key`；`if not key and profile.api_key_env: key = os.getenv(profile.api_key_env, "")` —— 直接 key 为空时，才尝试从 profile 指定的环境变量名读取（`os.getenv` 第二个参数 `""` 保证读不到时是空字符串而不是 `None`）。
  7. `if not key or str(key).startswith("replace-with"):` 返回 `(False, "provider.toml 已存在但 api_key 为空/占位符，请填入真实 key。")`。用 `str(key)` 是为了防止 key 是某些非字符串类型时 `startswith` 报错。
  8. `except Exception as exc:` —— 捕获配置解析过程中的**所有**异常，注释说明即使配置解析失败也要给用户可读的信息；返回 `(False, f"provider.toml 解析失败：{type(exc).__name__}: {exc}")`，用异常类名 + 异常消息拼装。
  9. 全部通过后 `return True, ""`。
- **异常/边界**：函数对外**从不抛异常**——所有解析异常都被捕获并转成可读提示。边界覆盖：示例配置回退、文件缺失、key 为空、key 为 `None`、key 是 `replace-with` 开头的占位符、`api_key_env` 指向的变量未设置、`ProviderRegistry` 构造或 profile 解析失败。注意 `load_project_dotenv()` 和 `default_config_path()` 在 `try` 块**之外**调用，如果它们抛异常则不会被兜住（实际实现中它们通常不抛）。另外它只校验 key 的存在与形态，**不做网络连通性测试**，所以 `(True, "")` 只代表「配置看起来是真的」。
- **同文件关系**：被 `schedule_qa_extraction()` 调用作为第二道闸门。它自身不调用本文件其它函数。与 `build_knowledge_extractor()` 共享同一套「区分 provider.toml 与 provider.example.toml、识别 replace-with 占位符」的判定逻辑，但两者是各自独立实现的。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `build_knowledge_extractor()` | 按 provider 与 services 配置构造 LLM 知识抽取器，任何缺失或异常都降级为不做事的 `NullKnowledgeExtractor`。 |
| `bump_graph_revision()` | 在锁内把全局图版本号加一，通知星图缓存失效。 |
| `graph_revision()` | 在锁内读取当前图版本号，供缓存判断是否需要重建。 |
| `get_manager()` | 双重检查锁懒加载进程级 `MemoryManager` 单例，并把 SQLite 路径固定为统一的 `DB_PATH`。 |
| `close_manager()` | 关闭全局 `MemoryManager` 并复位为 `None`，用于应用退出时释放资源。 |
| `get_pipeline()` | 双重检查锁懒加载进程级 `RAGPipeline` 单例，复用同一个管理器与抽取器。 |
| `get_agent()` | 懒加载「知识管家」`ReActAgent` 单例，关闭自动发现并注入系统提示词与六个显式绑定共享后端的工具。 |
| `chat_confirmed_side_effects(agent, *, destructive_call=None)` | 计算本回合已确认的副作用钥匙集合：固定放行 `memory.add`，并可精确放行一次经严格校验归一化的 `memory.manage` 调用。 |
| `search_available()` | 判断 `config/services.toml` 的 `[search]` 段是否同时配好了 base_url 与 api_key。 |
| `chat_tool_names(agent, *, online)` | 按白名单和联网开关返回本次聊天要暴露给 LLM 的工具名列表。 |
| `record_qa(manager, question, answer, *, mode)` | 把一次问答按长度截断后写入 episodic 记忆，附带结构化 metadata 与时间戳。 |
| `knowledge_extract_enabled()` | 返回问答是否触发图抽取的开关常量值，作为测试可 monkeypatch 的判定点。 |
| `extract_graph_patches(question, answer, *, manager=None)` | 把问答拼成受约束的文档交给 RAG 管道抽取图补丁，成功且有实体/关系时递增图版本号，失败只记日志返回 `None`。 |
| `extract_graph_patches_async(question, answer, *, manager=None)` | 用 `asyncio.to_thread` 把阻塞的图抽取挪出事件循环的异步包装协程。 |
| `schedule_qa_extraction(question, answer, *, manager=None)` | 经过开关与模型就绪两道闸门后，按同步/无事件循环/异步三种情况发射后不管地触发图抽取。 |
| `chat_ready()` | 检测真实聊天模型是否已配置，返回布尔结果与可直接展示的中文提示，严格排除示例配置与占位 key。 |
