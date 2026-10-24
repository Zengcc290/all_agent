# tool/import_knowledge.py

## 一、这个文件是干什么的

这个文件是知识库的「写工具」实现，职责是把导出的知识载荷（`knowledge-nebula-export/v1` 格式的 JSON，或者一个裸的 items 数组）幂等地写回记忆库。它是整个项目里**唯一**的导入逻辑实现：`web/app.py` 里原先内联的导入循环和 `_reject_json_constant` 已经被删除，`POST /api/import` 现在只负责上传文件、调用这里、再把异常映射成 HTTP 错误码。文件开头用文档字符串明确立下三条不可动摇的规则：第一是**严格 JSON**，拒绝 `NaN`/`Infinity`，因为导入是唯一的外来数据入口，一旦放行非有限数值，它就会顺着记忆库扩散到嵌入、相似度和图权重里；第二是**幂等**，已存在的 id 直接跳过，事实类条目按 `(主语, 谓语, 宾语)` 三元组去重，避免重复导入把知识图长成多重边；第三是**单条失败不拖垮整批**，逐条 try，失败只跳过该条并把原因写进 `errors`，且 `errors` 有上限以保证响应有界。文件里主要包含两类东西：一组纯函数（`reject_json_constant`、`parse_import_payload`、`import_items` 及其内嵌的 `note_error`），以及一组 Pydantic 模型与工具类（`ImportKnowledgeInput`、`ImportKnowledgeOutput`、`ImportKnowledgeTool`、`create_tool`）。它通过 `BaseTool`/`ToolSpec` 接入项目的工具注册体系，被 Agent 以工具名 `knowledge.import` 调用，也被 Web 层直接调用；模块末尾用 `__all__` 显式声明了对外暴露的名字。

## 二、函数与类逐条详解

### `reject_json_constant(value: str) -> None` （第 34 行）
- **作用**：这是给 `json.loads` 的 `parse_constant` 钩子准备的回调函数。Python 标准库的 `json` 模块默认允许把 `NaN`、`Infinity`、`-Infinity` 这三个非标准字面量解析成对应的浮点值，而本项目的模型与 API 全链路都假设数值严格有限，所以必须在唯一的外来数据入口处把这三个字面量堵死。它本身不做任何解析，唯一的行为就是抛异常，从而让 `json.loads` 在遇到这些字面量时立即失败。它被 `parse_import_payload` 作为回调传入，是整个文件「严格 JSON」这条规则的具体落点。
- **参数**：`value: str`，`json` 模块在词法扫描到 `NaN`、`Infinity` 或 `-Infinity` 字面量时原样传进来的字符串，取值就是这三个字面量之一。
- **返回**：没有返回值；它永远以抛异常结束，正常的返回路径不存在。
- **内部流程**：只有一步——直接 `raise ValueError(f"invalid JSON constant: {value}")`，把原始字面量拼进错误消息里，方便调用方看出到底命中了哪个非法常量。
- **异常/边界**：必然抛出 `ValueError`。这个异常会被 `json.loads` 向上传播，最终由 `parse_import_payload` 的 `except (UnicodeDecodeError, ValueError)` 捕获并包装成面向用户的中文报错。没有其它边界情况需要处理。
- **同文件关系**：被 `parse_import_payload` 以 `parse_constant=reject_json_constant` 的形式调用；它自己不调用本文件里的任何东西。

### `parse_import_payload(raw: bytes) -> list[Any]` （第 40 行）
- **作用**：把一段原始的字节载荷解码成「待导入条目的原始列表」，也就是导入流程的第一道关卡。它同时负责三件事：按 UTF-8 解码、用严格模式做 JSON 解析、从解析结果里取出 items 数组。之所以要把这段逻辑单独抽成函数，是因为载荷可能来自 HTTP 上传的文件，也可能来自工具调用里的一段字符串，两边的入口都想复用同一套校验与同一套报错文案。它抛出的 `ValueError` 消息是直接给用户看的，HTTP 层据此回 400，所以这里的错误信息必须写得像人话。
- **参数**：`raw: bytes`，原始的载荷字节，必须是合法的 UTF-8 编码。它不接收 str，调用方需要自己先编码，例如 `arguments.payload.encode("utf-8")`。
- **返回**：`list[Any]`，即待处理的条目列表；列表里的元素类型**不做任何假设**，可能是 dict，也可能是字符串、数字、列表或 null，具体的逐条类型校验留给 `import_items` 处理。如果解析出来的顶层是 dict，就返回它的 `"items"` 字段；如果顶层本身是 list，就原样返回这个 list。
- **内部流程**：第一步用 `raw.decode("utf-8")` 解码；第二步调用 `json.loads(..., parse_constant=reject_json_constant)`，把 `NaN`/`Infinity` 这类非标准常量交给 `reject_json_constant` 去拒绝；第三步判断顶层结构——若 `isinstance(data, dict)` 则取 `data.get("items")`，否则把 `data` 本身当作 entries；第四步校验 `isinstance(entries, list)`，不是 list 就抛 `ValueError("JSON 中找不到 items 数组")`；最后把 entries 返回。
- **异常/边界**：解码失败（`UnicodeDecodeError`）或 JSON 语法非法、含非法常量（`ValueError`，`reject_json_constant` 抛的也是这一类）都会被捕获，统一重新抛出为 `ValueError(f"不是合法的 JSON：{exc}")`，并用 `from exc` 保留原始异常链。顶层不是 dict 也不是 list（例如是个数字、字符串或 null）时，`entries` 不是 list，同样抛 `ValueError("JSON 中找不到 items 数组")`。dict 里没有 `items` 键时 `get` 返回 None，也走同一条报错分支。代码里特意留了注释说明：载荷结构错误属于**数据**问题而非编程错误，所以刻意用 `ValueError` 而不是静态检查工具建议的 `TypeError`，并在该行加了 `# noqa: TRY004` 抑制告警。
- **同文件关系**：调用了本文件的 `reject_json_constant`；被本文件的 `ImportKnowledgeTool.execute` 调用。

### `import_items(manager: MemoryManager, entries: list[Any], *, max_errors: int = WEB_IMPORT_ERRORS_MAX) -> dict[str, Any]` （第 58 行）
- **作用**：这是整个文件的核心函数，负责把 `parse_import_payload` 解析出的条目列表真正写进记忆库，并返回一份导入结果统计。它贯彻文件开头的三条规则：写之前先查重以实现幂等、逐条独立 try 让单条失败不影响整批、错误原因收集有上限。为了做到幂等，它在循环开始前先把库里已有的全部语义事实三元组拉出来做成一个集合，这样每条待导入事实的判重都是 O(1) 的集合查询，而不是反复查库；同时在导入过程中把新写入的三元组也加进这个集合，从而让同一批载荷内部的重复三元组也能被正确去重。它是 Web 层 `POST /api/import` 与工具 `knowledge.import` 共用的实现。
- **参数**：
  - `manager: MemoryManager`，记忆库管理器实例，提供 `list`、`get`、`add` 以及 `semantic.add_fact` 等能力，是实际写入的落点。
  - `entries: list[Any]`，待导入条目列表，元素应为 dict，但函数本身对非 dict 元素做了容错。
  - `max_errors: int = WEB_IMPORT_ERRORS_MAX`（关键字参数，默认值来自 `constants` 模块的常量），最多收集多少条失败原因；必须是**正整数**，且显式排除了 bool 类型。
- **返回**：`dict[str, Any]`，固定包含三个键：`"imported"`（成功写入的条目数，int）、`"skipped"`（跳过数，int，包含 id 已存在、缺字段、metadata 非法、memory_type 未知、事实三元组重复以及写入抛异常这几类）、`"errors"`（失败原因列表 `list[str]`，长度不超过 `max_errors`）。
- **内部流程**：
  1. 先校验 `max_errors`：`isinstance(max_errors, bool) or not isinstance(max_errors, int) or max_errors < 1` 三者任一成立就抛 `ValueError("max_errors must be a positive integer")`。把 bool 单独拎出来是因为 Python 里 `True`/`False` 也是 int 的子类。
  2. 预取已有事实：遍历 `manager.list(memory_type=MemoryType.SEMANTIC)`，对每个 item 取 `metadata` 里的 `subject`、`predicate`、`object`，三者**都非空**时才把三元组放进 `existing_facts` 集合；任一缺失的条目被过滤掉，不会污染集合。
  3. 计算 `known_types = {type_.value for type_ in MemoryType}`，得到所有合法 memory_type 字符串的集合。
  4. 初始化计数器 `imported = skipped = 0` 和错误列表 `errors: list[str] = []`。
  5. 进入 `for position, raw_item in enumerate(entries, start=1)` 主循环，`position` 从 1 开始，用于给用户报出「第几项」出错。
  6. 逐条校验：非 dict → `skipped += 1` 并记录「第 N 项：不是 JSON 对象」后 `continue`；取 `item_id = raw_item.get("id")`、`content = raw_item.get("content") or ""`，任一为空 → 记录「第 N 项（id=...）：缺少 id 或 content」，其中 id 缺失时显示为「缺失」。
  7. 幂等判重：`manager.get(item_id) is not None` 说明该 id 已在库中，`skipped += 1` 后**静默跳过**（不写 errors，因为这是正常的幂等命中而不是失败）。
  8. 校验 metadata：`md = raw_item.get("metadata") or {}`，若取到的不是 dict → 记录「{item_id}: metadata 必须是 JSON 对象」并跳过。
  9. 校验 memory_type：`memory_type = raw_item.get("memory_type") or "semantic"`，若不是字符串或不在 `known_types` 里 → 记录「未知 memory_type」并跳过。
  10. 取 `importance = raw_item.get("importance", 0.5)`，默认 0.5；再取出 `subject`、`predicate`、`obj` 三个 metadata 字段。
  11. 写入分支（整段包在 `try` 里）：若 `subject and predicate and obj` 三者齐全，视为「事实」——先查 `(subject, predicate, obj)` 是否已在 `existing_facts` 中，是则 `skipped += 1` 并 `continue`；否则把该三元组加入集合（防止同批内重复），再调用 `manager.semantic.add_fact(subject, predicate, obj, metadata=md, confidence=float(importance))`。否则走通用分支，调用 `manager.add(content, memory_type=memory_type, metadata=md, item_id=item_id, importance=float(importance))`。两个分支成功后都 `imported += 1`。
  12. 写入异常被 `except Exception` 捕获（带 `# noqa: BLE001`），`skipped += 1` 并记录 `f"{item_id}: {type(exc).__name__}: {exc}"`，异常类型名一并带上，方便定位。
  13. 循环结束后返回 `{"imported": imported, "skipped": skipped, "errors": errors}`。
- **异常/边界**：`max_errors` 非法时抛 `ValueError`。单条写入的任何异常都被吞掉并转成 errors 里的一条文本，不会中断整批。空 `entries` 列表会直接返回三个零值结果（errors 为空列表）。条目缺 id 或 content、metadata 不是对象、memory_type 未知都会被当作 skipped 并附原因。事实三元组只在三者都非空时走 `add_fact`，否则退化为普通条目走 `manager.add`。`float(importance)` 转换若因 importance 是字符串等非法值失败，也会被同一条 except 捕获并计入 skipped。`manager.list` 或 `manager.get` 自身抛出的异常不在 try 保护范围内（只有写入那一段被 try 包住），会向上传播。
- **同文件关系**：调用了本文件内嵌定义的 `note_error`；被本文件的 `ImportKnowledgeTool.execute` 调用。`note_error` 是它内部的嵌套函数，只能在这里使用。

### `note_error(message: str) -> None` （第 83 行，`import_items` 内部的嵌套函数）
- **作用**：这是 `import_items` 内部定义的一个小助手，用来把一条失败原因追加到 `errors` 列表，但只在列表长度还没达到 `max_errors` 时追加。它存在的意义是保证 HTTP 响应体有界：如果一个上万条的载荷全部失败，不可能把上万条原因全塞回响应里，所以只回报前 `max_errors` 条，其余失败仍然计入 `skipped` 计数。它被主循环里所有失败分支调用，是整个「错误上报有上限」策略的唯一实现点。因为它定义在 `import_items` 体内，所以天然闭包捕获了外层的 `errors` 列表和 `max_errors` 参数。
- **参数**：`message: str`，一条面向人类的失败原因文本，调用方在拼装时已经带上了位置或 id 前缀（例如「第 3 项：不是 JSON 对象」或「abc-123: ValueError: ...」）。
- **返回**：没有返回值（`None`）；它的副作用是可能向闭包中的 `errors` 列表追加一项。
- **内部流程**：只有一个判断——`if len(errors) < max_errors:` 成立时执行 `errors.append(message)`，不成立时什么也不做（静默丢弃这条原因）。
- **异常/边界**：无特殊处理。它不抛异常，也不做消息去重或截断；超出上限的原因被直接丢弃，但对应的条目仍会被外层计入 `skipped`。注意它不修改 `skipped`，计数由调用方自己维护。
- **同文件关系**：被外层函数 `import_items` 在多个失败分支中调用；它自己不调用本文件里的任何函数。

### `class ImportKnowledgeInput(BaseModel)` （第 149 行）
- **作用**：这是 `knowledge.import` 工具的输入模型，用 Pydantic 定义了工具被调用时必须满足的参数结构与约束。它存在的价值是把「载荷是不是合法、max_errors 是不是在合理范围」这类校验前移到框架层，让 `execute` 里拿到的参数一定是干净可用的。类上配置了 `extra="forbid"` 与 `strict=True`：前者禁止调用方传入未声明的多余字段（避免拼写错误被静默忽略），后者关闭 Pydantic 的宽松类型强转（例如拒绝用字符串 `"5"` 冒充整数）。它被 `ToolSpec` 的 `input_model` 引用，用于工具的参数校验与 schema 生成。
- **参数**：作为模型类，它的「参数」就是两个字段：
  - `payload: str`，必填，`min_length=2`，含义是「导出的 JSON 文本：knowledge-nebula-export/v1 载荷，或裸 items 数组」。最小长度 2 是为了让空串和单字符这类明显无效的输入在模型层就被拒掉。
  - `max_errors: int`，可选，默认值为 `WEB_IMPORT_ERRORS_MAX`，约束 `ge=1`、`le=100`，含义是「最多回报多少条失败原因（其余只计入 skipped）」。上限 100 与 `import_items` 的运行时校验形成双重保险。
- **返回**：类本身不返回值；实例化得到的是一个不可变语义的输入对象，`execute` 通过 `arguments.payload` 与 `arguments.max_errors` 读取字段。
- **内部流程**：Pydantic 在实例化时按字段声明顺序收集输入、套用 `strict=True` 的类型检查、执行长度与数值区间约束，并把 `extra="forbid"` 之外的键直接报错。本文件没有为它定义自定义校验器或 `__init__`，全部行为由声明式配置驱动。
- **异常/边界**：字段缺失、类型不符、`payload` 长度不足 2、`max_errors` 小于 1 或大于 100、出现未声明字段时，Pydantic 会抛 `ValidationError`，由工具框架统一捕获并转成工具调用错误；本文件内部不额外处理。
- **同文件关系**：被本文件的 `ImportKnowledgeTool.spec` 通过 `input_model=ImportKnowledgeInput` 引用，并被 `ImportKnowledgeTool.execute` 的形参类型标注引用；在模块末尾的 `__all__` 中导出。

### `class ImportKnowledgeOutput(BaseModel)` （第 164 行）
- **作用**：这是 `knowledge.import` 工具的输出模型，规定了工具返回值的形状。它与 `import_items` 返回的 dict 一一对应，`execute` 直接用 `ImportKnowledgeOutput(**result)` 把字典展开构造成它。把输出也建模成 Pydantic 对象的好处是：返回值可以被框架自动序列化、被 schema 描述、被上层稳定消费，而不会因为字典里多一个少一个键就悄悄破坏调用方。类上同样配置了 `extra="forbid"` 与 `strict=True`，保证返回结构与声明严格一致。
- **参数**：作为模型类，它的「参数」是三个字段：
  - `imported: int`，必填，含义是「真正写入的条目数」。
  - `skipped: int`，必填，含义是「跳过数：已存在、缺字段、类型未知或写入失败」。
  - `errors: list[str]`，可选，用 `default_factory=list` 生成默认空列表，含义是「失败原因（有上限）」。
- **返回**：类本身不返回值；实例化后即为工具调用的最终输出对象。
- **内部流程**：由 Pydantic 在 `ImportKnowledgeOutput(**result)` 调用时完成字段绑定与类型校验；`errors` 未提供时走 `default_factory` 新建一个空列表（用工厂而不是可变默认值，避免多个实例共享同一个列表）。本文件没有为它定义自定义方法。
- **异常/边界**：若 `result` 缺少 `imported` 或 `skipped`，或 `errors` 元素不是字符串，Pydantic 会抛 `ValidationError`。实际调用路径中 `result` 恒由 `import_items` 产出，三个键一定齐全，因此正常运行时不会触发。
- **同文件关系**：被本文件的 `ImportKnowledgeTool.spec` 通过 `output_model=ImportKnowledgeOutput` 引用，并被 `ImportKnowledgeTool.execute` 作为返回类型构造与返回；在模块末尾的 `__all__` 中导出。

### `class ImportKnowledgeTool(BaseTool)` （第 172 行）
- **作用**：这是知识导入工具的类本体，把前面那些纯函数包装成一个符合项目工具协议的、可被 Agent 调用的对象。它继承 `core.BaseTool`，并在类属性 `spec` 里用一个 `ToolSpec` 声明了工具的全部元信息：工具名 `knowledge.import`、英文描述、版本 `1.0.0`、输入输出模型、副作用类型 `write`、空权限元组、600 秒超时、幂等标记 `idempotent=True`、`parallel_safe=False`（因为它要写库，并发执行不安全）、标签 `("knowledge", "import", "restore", "write")`，以及一段中文 `guidance` 使用指引。这段 guidance 明确告诉模型：把 `knowledge.export` 产出的载荷灌回库时用它、它是幂等的所以可以安全重试、载荷里的 NaN/Infinity 会被拒绝、不要因为个别条目失败就整批重复导入。类本身还持有对 `MemoryManager` 的懒加载引用。
- **参数**：类无构造参数意义上的「参数」；其类属性 `spec` 的字段含义如上（`name`、`description`、`version`、`input_model`、`output_model`、`side_effect`、`permissions`、`timeout_seconds`、`idempotent`、`parallel_safe`、`tags`、`guidance`）。
- **返回**：类本身不返回；实例化得到工具对象，供框架按 `spec` 调度。
- **内部流程**：类体先定义 `spec` 类属性（在类定义时即完成 `ToolSpec` 构造），随后定义 `__init__`、`manager` 属性与 `execute` 方法。框架侧读取 `spec` 做注册、参数校验与超时控制，再调用 `execute`。
- **异常/边界**：类定义阶段若 `ToolSpec` 参数非法会在导入模块时立刻报错（本文件的写法是静态常量，正常情况下不会）。工具执行期的异常由 `execute` 及其下游决定。
- **同文件关系**：引用本文件的 `ImportKnowledgeInput`、`ImportKnowledgeOutput`；其方法 `execute` 调用 `parse_import_payload` 与 `import_items`；被本文件的 `create_tool` 实例化；在模块末尾的 `__all__` 中导出。

### `ImportKnowledgeTool.__init__(self, manager: MemoryManager | None = None) -> None` （第 196 行）
- **作用**：这是工具的构造函数，唯一职责是接收一个可选的 `MemoryManager` 并把它存到实例的私有属性 `self._manager` 上。之所以把它设计成可选，是为了让这个工具既能被框架在不知道具体管理器的情况下创建（此时后续按需懒加载默认管理器），也能在 Web 层或测试里显式注入一个已经建好的管理器（例如与 `POST /api/import` 共用同一个实例），从而保证写入的是同一份记忆库。它刻意不做任何初始化重活，把建库成本推迟到第一次真正要用的时候，避免模块导入或工具注册阶段就产生副作用。
- **参数**：`manager: MemoryManager | None = None`，可选的记忆库管理器；传 `None` 表示「先不绑定，用到时再建默认的」。
- **返回**：`None`。
- **内部流程**：只有一行——`self._manager = manager`，原样保存引用，不做类型检查、不做拷贝、不触发任何 IO。
- **异常/边界**：无特殊处理；传入非 `MemoryManager` 的对象也不会在这里报错，问题会在真正调用其方法时才暴露。
- **同文件关系**：它设置的 `self._manager` 被本类的 `manager` 属性读取与回填；被本文件的 `create_tool` 间接调用（`ImportKnowledgeTool()`）。

### `ImportKnowledgeTool.manager` （第 199 行，`@property`）
- **作用**：这是一个只读属性，对外暴露一个「一定可用」的 `MemoryManager`。它实现了懒加载：如果 `self._manager` 还是 `None`，就从同包的 `_memory` 模块导入 `build_default_manager` 并调用它建一个默认管理器，回填到 `self._manager`，然后返回。把它做成属性而不是普通方法的用意是让 `execute` 里可以像读普通字段一样写 `self.manager`，同时又保留了「首次访问才建库」的延迟语义。这样无论调用方是否注入过管理器，工具都能正常工作。
- **参数**：无（`self` 除外）。
- **返回**：`MemoryManager` 实例；首次访问时是新建的默认管理器，之后每次返回同一个缓存实例。
- **内部流程**：判断 `if self._manager is None:`，成立则执行函数内导入 `from ._memory import build_default_manager`（延迟到需要时才导入，避免模块级循环依赖与启动开销），调用 `build_default_manager()` 并把结果赋给 `self._manager`；最后 `return self._manager`。
- **异常/边界**：无特殊处理。若 `build_default_manager` 内部失败（例如配置缺失），异常会从属性访问处直接抛出，且不会留下半初始化的状态——因为赋值发生在调用成功之后。
- **同文件关系**：读取并回填 `__init__` 设置的 `self._manager`；被本类的 `execute` 通过 `self.manager` 调用。

### `ImportKnowledgeTool.execute(self, arguments: ImportKnowledgeInput) -> ImportKnowledgeOutput` （第 207 行）
- **作用**：这是工具真正干活的方法，也是「解析载荷 → 写入记忆库 → 包装结果」这条链路的粘合点。它把输入模型里的 `payload` 字符串按 UTF-8 编码后交给 `parse_import_payload` 解析，再把解析出的条目列表、当前管理器、以及调用方指定的 `max_errors` 一起交给 `import_items` 执行导入，最后把返回的字典展开构造成 `ImportKnowledgeOutput` 返回。因为下游两个函数都自带完整校验与容错，这个方法本身保持得很薄，不重复做任何检查，也不自己捕获异常——解析失败（载荷非法）会以 `ValueError` 向上抛出，由工具框架或 HTTP 层映射成相应的错误码。
- **参数**：`arguments: ImportKnowledgeInput`，已经通过 Pydantic 校验的输入对象；它提供 `payload`（JSON 文本，长度至少 2）和 `max_errors`（1 到 100 之间的整数）。
- **返回**：`ImportKnowledgeOutput` 实例，字段为 `imported`、`skipped`、`errors`，分别对应成功写入数、跳过数与有上限的失败原因列表。
- **内部流程**：第一步 `arguments.payload.encode("utf-8")` 把字符串变成 `parse_import_payload` 需要的字节；第二步调用 `parse_import_payload` 得到 `entries`；第三步调用 `import_items(self.manager, entries, max_errors=arguments.max_errors)`——注意这里 `self.manager` 会触发懒加载；第四步 `return ImportKnowledgeOutput(**result)` 把结果字典展开构造成输出模型。
- **异常/边界**：载荷不是合法 JSON、含 `NaN`/`Infinity`、或顶层找不到 items 数组时，`parse_import_payload` 抛出的 `ValueError` 会原样向上传播（工具框架负责把它转成工具调用错误，HTTP 层转成 400）。`max_errors` 的非法值在输入模型层就被拦下，因此不会走到 `import_items` 的运行时校验。单条导入失败不会抛异常，只体现在返回值的 `skipped` 与 `errors` 里。
- **同文件关系**：调用本文件的 `parse_import_payload` 与 `import_items`，读取本类的 `manager` 属性，构造并返回本文件的 `ImportKnowledgeOutput`。

### `create_tool() -> BaseTool` （第 213 行）
- **作用**：这是工具的工厂函数，供项目里的工具注册/发现机制按统一约定调用。它不接收任何参数，直接 `return ImportKnowledgeTool()`，也就是创建一个未绑定管理器的工具实例——管理器会在第一次执行 `execute`、访问 `manager` 属性时才懒加载出来。把创建逻辑收敛到一个无参工厂，是为了让注册表能用一个统一的调用签名批量实例化各个工具，而不需要知道每个工具各自的构造细节。
- **参数**：无。
- **返回**：`BaseTool`，实际类型是 `ImportKnowledgeTool` 实例；返回类型标注为基类是为了让注册方按统一协议处理。
- **内部流程**：只有一行——构造并返回 `ImportKnowledgeTool()`，不传 `manager`，不做任何条件判断或缓存。
- **异常/边界**：无特殊处理。若 `ImportKnowledgeTool` 的类体或 `spec` 构造有问题，异常会在模块导入时就已经抛出，不会等到这里。
- **同文件关系**：调用本文件的 `ImportKnowledgeTool` 构造函数；在模块末尾的 `__all__` 中导出。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `reject_json_constant` | 作为 `json.loads` 的 `parse_constant` 钩子，遇到 `NaN`/`Infinity` 就抛 `ValueError`，保证导入的 JSON 数值严格有限。 |
| `parse_import_payload` | 把原始字节按 UTF-8 解码并严格解析成条目列表，顶层是 dict 就取 `items` 数组，否则原样当数组，非法时抛带中文原因的 `ValueError`。 |
| `import_items` | 导入的核心实现：预取已有事实三元组与合法类型集合，逐条校验并幂等写入记忆库，返回 `imported`/`skipped`/`errors` 统计。 |
| `note_error` | `import_items` 的内嵌助手，只在错误数未达 `max_errors` 时追加一条失败原因，保证响应有界。 |
| `ImportKnowledgeInput` | 工具输入模型，声明 `payload`（至少 2 字符的 JSON 文本）与 `max_errors`（1–100，默认取常量），并禁止多余字段与宽松类型转换。 |
| `ImportKnowledgeOutput` | 工具输出模型，声明 `imported`、`skipped` 与默认空列表的 `errors` 三个字段。 |
| `ImportKnowledgeTool` | 继承 `BaseTool` 的知识导入工具本体，用 `ToolSpec` 声明名称、描述、输入输出模型、写副作用、600 秒超时与幂等标记等元信息。 |
| `ImportKnowledgeTool.__init__` | 构造函数，仅把可选注入的 `MemoryManager` 存到 `self._manager`，不做其它初始化。 |
| `ImportKnowledgeTool.manager` | 只读属性，首次访问时通过 `build_default_manager` 懒加载并缓存默认记忆库管理器，保证 `self.manager` 总是可用。 |
| `ImportKnowledgeTool.execute` | 把 `payload` 编码后解析成条目、调用 `import_items` 写入记忆库，再把结果字典包装成 `ImportKnowledgeOutput` 返回。 |
| `create_tool` | 无参工厂函数，返回一个未绑定管理器的 `ImportKnowledgeTool` 实例，供工具注册机制统一创建。 |
