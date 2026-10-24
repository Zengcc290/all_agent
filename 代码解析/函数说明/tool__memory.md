# tool/_memory.py

## 一、这个文件是干什么的

这个文件是 memory 相关 agent 工具的「公共底座」，专门存放被多个工具模块共享的 schema 片段与默认后端构造逻辑。它本身刻意做成不可被发现的状态：`core.discovery` 会忽略所有以下划线开头的模块名，而本文件既没有定义 `TOOL_ENABLED`，也没有定义 `create_tool()` 工厂函数，所以它不会作为工具被注册进运行时。它的核心内容是：一个用于描述记忆元数据的 Pydantic 模型 `MemoryMetadata`、一个类型别名 `MemoryScope`、两个元数据格式转换小工具（`normalize_metadata_payload` 与 `metadata_dict`），以及两个默认后端构造函数（`build_default_manager` 负责打开磁盘上的共享记忆库，`build_default_pipeline` 负责搭建 RAG 流水线并在 LLM 不可用时降级）。导入这个模块不会打开任何数据库、也不会发起任何网络请求，所有后端都是「首次使用时」才在各自的工具里真正构建。文件末尾用 `__all__` 明确导出了对外公开的符号，供其它 tool 模块按需引用。由于它承担的是「共享 + 默认值」的职责，所以这里的函数大多是纯函数或轻量工厂，逻辑简单但对上层工具的一致性很关键。

## 二、函数与类逐条详解

### `MemoryMetadata` （类，第 25 行）

- **作用**：定义记忆元数据在工具入参中的标准结构。记忆条目除了正文之外往往还需要携带键值对形式的附加信息（例如来源、标签、时间戳标记等），这个模型把这种附加信息固定成「一个键 + 一个值」的条目对象，从而让上层工具在接收到模型（LLM）生成的参数时能做严格的类型校验，而不是任由任意字典结构流入记忆管理器。它被当作列表元素使用，多个条目组成一份元数据集合，再通过 `metadata_dict` 压平成管理器需要的映射形式。之所以单独抽出来放在这个共享文件里，是因为多个记忆工具（写入类、检索类等）都需要同一套元数据表示，集中定义可以避免各处结构漂移。它继承自 `pydantic.BaseModel`，因此构造时就会自动完成字段类型检查与缺失字段检查，工具层不需要再手写校验代码。
- **参数**：类本身没有自定义的 `__init__` 参数，字段即构造参数。`key`：字符串类型，通过 `Field(min_length=1)` 约束，必须是长度至少为 1 的字符串，空字符串会被 Pydantic 拒绝；没有默认值，因此构造时必须提供。`value`：字符串类型，没有最小长度约束，允许空字符串，同样没有默认值，构造时必须提供。类级别的 `model_config` 是一个 `ConfigDict(extra="forbid", strict=True)` 配置对象：`extra="forbid"` 表示传入未声明的字段会直接报校验错误，`strict=True` 表示关闭类型强制转换（例如传整数不会被自动转成字符串）。
- **返回**：类不返回东西；实例化后得到一个 `MemoryMetadata` 对象，带有 `.key` 与 `.value` 两个字符串属性，可被 `metadata_dict` 读取。
- **内部流程**：类体里只做两件事：第一行把 `model_config` 绑定为 `ConfigDict(extra="forbid", strict=True)`，改变 Pydantic 的校验行为；随后声明 `key` 与 `value` 两个字段，其中 `key` 用 `Field(min_length=1)` 附加了长度约束。实际的字段解析、校验、`ValidationError` 抛出、以及 `model_dump()` 之类的序列化能力全部由 Pydantic 的 `BaseModel` 元类机制在类创建时生成。本文件没有为这个类定义任何自定义方法（没有自定义 `__init__`、没有 `model_post_init`、也没有属性方法），构造与校验完全依赖 Pydantic 的默认实现。
- **异常/边界**：构造时若缺少 `key` 或 `value`、`key` 为空字符串、字段类型不是字符串（在 `strict=True` 下），或传入了额外字段，Pydantic 会抛出 `pydantic.ValidationError`。本文件没有捕获这些异常，由调用方（各工具模块）自行处理。
- **同文件关系**：被 `metadata_dict` 用作入参元素的类型标注并在函数体内访问 `.key` / `.value`；被模块末尾的 `__all__` 导出。它自身不调用本文件里的任何函数。

### `normalize_metadata_payload(value: Any) -> Any` （第 32 行）

- **作用**：把模型（LLM）经常产出的「友好写法」元数据转换成工具 schema 要求的列表写法。语言模型在生成工具参数时，往往更倾向于输出 `{"metadata": {"k": "v"}}` 这种直观的字典映射，而工具层定义的 schema 是 `[{"key": ..., "value": ...}, ...]` 的列表结构，两者不兼容会直接导致校验失败。这个函数就是那道兼容层：它在真正校验之前，把字典形式的 `metadata` 就地展开成条目列表，使模型的自然输出也能顺利通过校验。它只做形状转换，不改变语义，也不丢弃信息。由于它是纯函数、无副作用，可以被安全地放在参数解析链路的最前端调用。
- **参数**：`value`（`Any`）：待规范化的载荷，通常是工具参数的整个字典（也可能已经是列表或别的类型）。没有默认值。约束是：只有当 `value` 本身是 `dict`、并且它的 `"metadata"` 键对应的值也是 `dict` 时才会触发转换；其它任何形态（不是字典、没有 `metadata` 键、`metadata` 不是字典、`metadata` 已经是列表）都会原样返回。
- **返回**：返回 `Any`。触发转换时返回一个新的字典（原字典的浅拷贝，`metadata` 键被替换成条目列表）；未触发转换时返回传入的原始对象本身（同一个引用，未做拷贝）。
- **内部流程**：第一步用 `isinstance(value, dict)` 判断是否为字典，再用 `isinstance(value.get("metadata"), dict)` 判断其 `metadata` 字段是否为字典，两个条件同时成立才继续。第二步用 `dict(value)` 做一次浅拷贝，避免修改调用方传入的原始字典。第三步用列表推导遍历 `value["metadata"].items()`，把每个键值对变成 `{"key": key, "value": str(item)}`，注意值被强制 `str()` 字符串化，以适配 `MemoryMetadata.value: str` 的严格字符串约束；键没有被字符串化（依赖原始键本身已是字符串）。最后返回拷贝后的字典。
- **异常/边界**：空字典 `{}` 会走原样返回分支（没有 `metadata` 键）；`{"metadata": {}}` 会走转换分支，得到一个空列表 `[]`；`{"metadata": None}`、`{"metadata": [...]}`、非字典入参都原样返回。若 `value["metadata"]` 的某个值对象的 `__str__` 抛异常，异常会向上传播；除此之外没有主动抛出异常，也不做日志记录。
- **同文件关系**：函数体只用到内置类型与 `str()`，不调用本文件里的任何函数；被模块末尾的 `__all__` 导出，供各记忆工具在参数校验前调用（本文件内部没有调用它）。它转换出的结构在形状上与 `MemoryMetadata` 列表一致，因此与 `MemoryMetadata` 是配套关系。

### `metadata_dict(entries: list[MemoryMetadata] | None) -> dict[str, str]` （第 42 行）

- **作用**：把工具层使用的「元数据条目列表」压平成记忆管理器所需的「键值映射」。`MemoryManager` 在写入记忆时接受的是普通字典，而工具 schema 为了能被严格校验、被模型稳定生成，用的是条目列表，这个函数就是两种表示之间的单向转换器。它让上层工具在拿到校验通过的参数后，只需一行调用就能得到可直接传给管理器的字典。它是纯函数，不修改入参、不产生副作用。因为空元数据是常见情况（大多数写入调用不带任何附加信息），所以它同时承担了 `None` 与空列表的兜底职责。
- **参数**：`entries`（`list[MemoryMetadata] | None`）：元数据条目列表，允许传入 `None`，没有默认值（调用方需显式传参，但可以传 `None`）。列表中的每个元素预期是 `MemoryMetadata` 实例，函数只访问其 `.key` 与 `.value` 属性。
- **返回**：返回 `dict[str, str]`。当 `entries` 为 `None` 或空列表时返回空字典 `{}`；否则返回由每个条目的 `key` 到 `value` 构成的字典。
- **内部流程**：使用 `entries or []` 做空值兜底——`None` 和空列表都会退化为空列表；然后在一个字典推导里遍历该列表，逐项取 `entry.key` 作为键、`entry.value` 作为值，构造并返回新字典。
- **异常/边界**：`None` 与 `[]` 都安全地返回 `{}`。若列表中存在 `key` 重复的条目，后出现的条目会覆盖先出现的（字典推导的天然语义），函数不会报错也不会警告。若列表元素不是 `MemoryMetadata`（例如是普通字典），访问 `.key` 会抛 `AttributeError`；若元素的 `key` 不可哈希则字典构造会抛 `TypeError`。本文件不捕获这些异常。
- **同文件关系**：它把 `MemoryMetadata` 实例作为输入消费；不调用本文件里的其它函数，也不被本文件里的其它函数调用（本文件内部没有使用点），由模块末尾的 `__all__` 导出供各工具使用。

### `build_default_manager() -> MemoryManager` （第 48 行）

- **作用**：在没有外部注入管理器的情况下，构造并返回指向磁盘上那份共享记忆库的 `MemoryManager`。记忆类工具在运行时既可能拿到 Web 层传入的管理器，也可能被单独调用（例如命令行、测试或独立 agent 场景），这时就需要一个默认实现来打开「同一份」数据库，避免出现多份互不可见的记忆。函数通过统一读取配置来保证这一点：外部服务（Qdrant / Neo4j）的开关来自 `services.toml`，而 SQLite 文件路径只由 `default_sqlite_path()` 单点决定，从而与 Web 层的 `get_manager` 保持完全一致的口径。它是轻量工厂，每次调用都会新建一个 `MemoryManager` 实例，调用方需自行决定是否复用。
- **参数**：无参数。所有输入都来自配置系统：`MemoryConfig.from_config()` 读取项目配置（含 `services.toml` 中的外部存储设置），`default_sqlite_path()` 给出 SQLite 落盘路径（其唯一的环境变量覆盖入口是 `MEMORY_DB_PATH`）。
- **返回**：返回一个配置好的 `MemoryManager` 实例，其 `config.sqlite_path` 已被显式设置为 `default_sqlite_path()` 的返回值。
- **内部流程**：第一步调用 `MemoryConfig.from_config()` 得到基础配置对象；第二步把 `config.sqlite_path` 覆盖为 `default_sqlite_path()` 的结果（这一步是为了绕开配置文件里可能存在的旧路径或相对路径，确保落盘位置唯一确定）；第三步用该配置实例化 `MemoryManager(config)` 并直接 `return`。函数体内没有连接数据库的显式语句，真正的连接/懒加载由 `MemoryManager` 自身在首次操作时完成。
- **异常/边界**：若 `MemoryConfig.from_config()` 读取配置失败（文件缺失、格式错误）或 `default_sqlite_path()` 解析路径失败，异常会直接向上抛出，本函数不做捕获也不做降级；若磁盘目录不可写，异常会在 `MemoryManager` 实际打开数据库时才暴露。无参数，因此没有参数空值问题。
- **同文件关系**：它调用 `build_default_manager` 自身之外的 `MemoryConfig`、`default_sqlite_path`、`MemoryManager`（均来自 `memory` 包，非本文件）；本文件中 `build_default_pipeline` 会调用它来给 `RAGPipeline` 提供管理器。它也被模块末尾的 `__all__` 导出。

### `build_default_pipeline() -> RAGPipeline` （第 58 行）

- **作用**：构造默认的 RAG（检索增强生成）流水线，并在无法使用大模型时优雅降级。知识抽取器决定「写入记忆时能否顺便让 LLM 抽取结构化知识」，如果配置了可用的模型 provider，就用 `LLMKnowledgeExtractor` 做真正的知识抽取；如果没有 API Key、Key 还是占位符、或者配置读取失败，则退回 `NullKnowledgeExtractor`，此时记忆写入仍然可用，只是退化为纯向量检索，并把降级原因记进日志。这种设计保证 RAG 功能不会因为模型配置问题而整体不可用。函数把「管理器」与「抽取器」组装成 `RAGPipeline` 一次性返回，供工具在需要检索或写入知识时使用。
- **参数**：无参数。所需信息全部通过内部调用获取：`ProviderRegistry()` 的当前激活 profile、`registry.resolve_api_key(profile.name)` 解析出的密钥、`profile.base_url` 与 `profile.default_model` 给出的服务地址与模型名，以及 `load_services_config().vision.model` 提供的视觉模型名（对应 `config/services.toml` 的 `[vision]` 段）。
- **返回**：返回一个 `RAGPipeline` 实例，其底层管理器来自 `build_default_manager()`，抽取器为 `LLMKnowledgeExtractor` 或 `NullKnowledgeExtractor`。任何异常情况下都会返回一个使用空抽取器的可用流水线，而不会返回 `None`。
- **内部流程**：第一步先无条件创建 `NullKnowledgeExtractor()` 作为兜底，保证后续任何一步失败都仍有可用的抽取器。第二步进入 `try` 块，依次做局部导入（`agents.llm.LLM`、`agents.providers.ProviderRegistry`、`core.services_config.load_services_config`）——放在函数内部是为了避免模块导入期就产生对上层 agent 模块的依赖。第三步构造 `ProviderRegistry()`，取当前激活的 `profile`（`registry.active_profile` 作为键传给 `registry.get`），再用 `registry.resolve_api_key(profile.name)` 解析密钥。第四步做双重可用性判断：`key` 必须非空，且不能以 `"replace-with"` 开头（这是配置模板里的占位符前缀），两者同时满足才认为模型可用。第五步构造 `LLM` 客户端（传入 `api_key`、`base_url`、`model`），再读取 `load_services_config().vision.model` 作为视觉模型名，若为空字符串则回退为 `profile.default_model`。第六步用 `client.complete` 作为回调、加上模型名与视觉模型名构造 `LLMKnowledgeExtractor`，覆盖掉兜底的 `NullKnowledgeExtractor`。最后跳出 `try/except`，用 `RAGPipeline(build_default_manager(), extractor=extractor)` 组装并返回。
- **异常/边界**：`try` 块内捕获的是裸 `Exception`，覆盖面很广——provider 注册表为空、没有激活 profile、密钥缺失、配置不可读、`LLM` 构造失败等都会被捕获；捕获后用 `LOGGER.warning(..., exc_info=True)` 记录降级原因与完整堆栈，并继续使用 `NullKnowledgeExtractor`。需要注意的是 `except` 块只包住抽取器的构建，最后的 `build_default_manager()` 与 `RAGPipeline(...)` 在 `try` 之外，如果它们抛异常（例如配置读取失败或数据库路径异常），异常会向上传播而不会降级。密钥为空字符串、或密钥以 `"replace-with"` 开头时不会抛异常，只是静默保持空抽取器（没有任何日志说明是这两种情况之一，只有真正抛异常时才有 warning）。视觉模型名为空时回退为 `profile.default_model`。
- **同文件关系**：调用本文件中的 `build_default_manager()` 获取管理器，并使用模块级 `LOGGER` 记录降级日志；不被本文件里的其它函数调用。它由模块末尾的 `__all__` 导出，供各记忆工具在需要默认 RAG 流水线时调用。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `MemoryMetadata` | 用 Pydantic 严格定义「一个键 + 一个值」的记忆元数据条目结构（`extra="forbid"`、`strict=True`、`key` 至少 1 个字符）。 |
| `normalize_metadata_payload` | 把模型常输出的 `{"metadata": {"k": "v"}}` 字典映射形式转换成工具 schema 要求的条目列表形式，其余形态原样返回。 |
| `metadata_dict` | 把 `MemoryMetadata` 条目列表（或 `None`）压平成 `MemoryManager` 需要的 `{key: value}` 字典。 |
| `build_default_manager` | 按统一配置口径（`MemoryConfig` + `default_sqlite_path()`）构造指向共享磁盘记忆库的默认 `MemoryManager`。 |
| `build_default_pipeline` | 构造默认 `RAGPipeline`，在 provider/密钥不可用时降级为 `NullKnowledgeExtractor` 并记录警告日志。 |
