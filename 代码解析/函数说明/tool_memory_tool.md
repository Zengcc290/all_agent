# tool/memory_tool.py

## 一、这个文件是干什么的

这个文件实现了一个内建的 Agent 工具 `memory.manage`，它是记忆系统里 `memory.query`（查询）和 `memory.add`（写入）的**管理型对偶工具**，专门负责「删除单条记忆」和「清空整层记忆」这两类破坏性操作。

文件开头明确写着：这两个动作都是破坏性的（destructive），所以工具声明的副作用等级是写入型/破坏型，运行时要求调用方给出显式确认键；而 Web 聊天链路里没有任何地方会发放这种确认键，因此**模型无法自己擦除记忆**，必须由人确认后才可能真正执行。

整个文件的结构非常扁平：两个 Pydantic 输入/输出模型（`MemoryManageInput`、`MemoryManageOutput`）、一个继承 `BaseTool` 的工具类（`MemoryManageTool`），以及一个零参数工厂函数 `create_tool`。它通过 `ToolSpec` 把工具名、描述、版本、输入输出模型、副作用、所需权限、超时、幂等性、并发安全性、标签和给模型的中文使用指引一次性声明清楚，属于项目里「声明式工具定义」的典型写法。

文件还导入了 `MemoryScope` 与 `build_default_manager`（来自同包的 `._memory`），前者是记忆层枚举、用于约束模型只能填写合法的层名，后者负责惰性构造默认的 `MemoryManager`。默认管理器指向 `MEMORY_DB_PATH` 环境变量指定的数据库（或项目旁的 `memory.sqlite3`），需要换后端的应用可以在构造工具时注入自己的 `MemoryManager`。

## 二、函数与类逐条详解

### `class MemoryManageInput(BaseModel)` （第 27 行）
- **作用**：这是 `memory.manage` 工具的输入契约模型，用来在模型真正执行删除/清空之前，把模型生成的参数做一次严格校验与结构化。它存在的意义是防止模型传入拼错的字段名、传错类型、或者传一个不存在的记忆层名，从而在进入真正破坏性的数据库操作之前就失败。它的 `model_config` 设成 `extra="forbid", strict=True`，意味着多传任何未知字段都会被拒绝，且类型不会被宽松地强制转换（例如不能把字符串 `"1"` 当作整数用）。这个模型会被 `MemoryManageTool.spec` 里的 `input_model` 引用，运行时在调用 `execute` 之前用它解析参数。它不包含任何业务逻辑，是纯粹的 DTO（数据传输对象）。
- **参数**：本类是数据模型，没有构造参数列表；它的字段即「参数」，共三个：
  - `action`：类型为 `Literal["delete", "clear"]`，必填、无默认值。取值只能是字符串 `"delete"`（删除单条）或 `"clear"`（清空整层），其它值会在校验阶段直接报错。
  - `memory_type`：类型为 `MemoryScope | None`，默认 `None`。字段描述说明它表示要操作的目标记忆层，缺省时按 `'working'`（工作记忆）处理，这样 `clear` 永远不会意外清空整个存储。合法取值范围由 `MemoryScope` 枚举决定。
  - `item_id`：类型为 `str | None`，默认 `None`。字段描述为「用于 delete 的条目 id」。当 `action == "delete"` 时它实际上必须提供，但这个约束不是由 Pydantic 在这里强制的，而是在 `execute` 里显式检查。
- **返回**：本类不是函数，没有返回值；实例化后返回一个经过校验的 `MemoryManageInput` 对象，供 `execute` 读取字段。
- **内部流程**：没有显式方法体。实例化时 Pydantic 会依据字段注解依次完成：拒绝未知字段（`extra="forbid"`）、对 `action` 做字面量枚举校验、对 `memory_type` 做 `MemoryScope` 枚举校验（允许 `None`）、对 `item_id` 做字符串类型校验（允许 `None`）、对缺失的 `memory_type`/`item_id` 填入默认值 `None`，最后产出不可随意改写的模型实例。
- **异常/边界**：当 `action` 不是 `"delete"` 或 `"clear"` 时抛 `pydantic.ValidationError`；当 `memory_type` 不是合法 `MemoryScope` 成员时同样抛 `ValidationError`；当 `item_id` 不是字符串时抛 `ValidationError`；传入未声明字段时抛 `ValidationError`。缺失的 `action` 会因无默认值而报校验错误；缺失的 `memory_type` 与 `item_id` 合法地取 `None`，不报错。`item_id` 是否必填这一业务约束不在这里处理。
- **同文件关系**：被 `MemoryManageTool.spec` 的 `input_model` 字段引用；其字段由 `MemoryManageTool.execute` 读取；被模块末尾的 `__all__` 导出。不调用本文件里的任何函数。

### `class MemoryManageOutput(BaseModel)` （第 41 行）
- **作用**：这是 `memory.manage` 工具的输出契约模型，用来把执行结果规范化成固定形状再回传给 Agent 运行时。它声明了三个字段，让上层能统一地拿到「做了什么动作、影响了几条、具体条目是什么」。因为工具的成功返回需要可序列化、可校验，所以用 Pydantic 模型而不是裸 dict 承载。`model_config` 同样是 `extra="forbid", strict=True`，保证输出不会夹带未声明字段、类型不会被隐式转换。当前 `execute` 里 `items` 始终被填成空列表，说明这个字段是为将来的「返回被删条目的快照」预留的扩展位。它被 `MemoryManageTool.spec` 的 `output_model` 引用，是工具返回值类型标注的一部分。
- **参数**：本类是数据模型，没有构造参数列表；字段即参数，共三个：
  - `action`：类型 `str`，必填、无默认值。用来回显实际执行的动作（`"delete"` 或 `"clear"`）。
  - `count`：类型 `int`，默认 `0`。表示本次操作影响的条目数量；删除时来自底层 `delete` 的返回值，清空时来自 `clear` 的返回值。
  - `items`：类型 `list[dict[str, Any]]`，通过 `Field(default_factory=list)` 提供默认值。使用工厂函数而非可变默认值，避免多个实例共享同一个列表对象；当前执行路径下始终为空列表。
- **返回**：本类不是函数，没有返回值；实例化后返回一个经过校验的 `MemoryManageOutput` 对象，作为 `execute` 的返回结果。
- **内部流程**：没有显式方法体。实例化时 Pydantic 依次校验：`action` 必须是字符串且必须提供；`count` 必须是整数，缺省填 `0`；`items` 必须是字典列表，缺省时调用 `list` 工厂生成新空列表；任何未声明字段被拒绝。
- **异常/边界**：`action` 缺失或类型错误、`count` 非整数、`items` 不是列表或元素不是字典时抛 `pydantic.ValidationError`；传入额外字段抛 `ValidationError`。注意 `strict=True` 下布尔值等非整数类型不会被当成 `int` 接受。空列表是合法值，不视为异常。
- **同文件关系**：被 `MemoryManageTool.spec` 的 `output_model` 引用；由 `MemoryManageTool.execute` 构造并返回；被模块末尾的 `__all__` 导出。不调用本文件里的任何函数。

### `class MemoryManageTool(BaseTool)` （第 49 行）
- **作用**：这是本文件的核心工具类，把「删除/清空记忆」这个能力包装成项目统一的 `BaseTool` 子类，供 Agent 的工具注册与调度体系发现和调用。类体里先声明了一个类级 `spec = ToolSpec(...)`，集中描述工具的元信息：名称 `memory.manage`、说明文字（删除单条或清空整层、破坏性、需要用户显式确认）、版本 `3.0.0`、输入模型 `MemoryManageInput`、输出模型 `MemoryManageOutput`、副作用等级 `"destructive"`、所需权限 `("memory.write",)`、超时 `10.0` 秒、`idempotent=False`（重复执行不保证等价）、`parallel_safe=False`（不可与其他调用并发安全地执行）、标签 `("memory", "storage", "admin")`，以及一段中文 `guidance`：删除或清空前必须已获得用户确认，清空整层前必须让用户明确说出层名，不确定时先用 `memory.query` 列候选，禁止在无人确认时批量删除。这个类本身不实现删除算法，真正的数据库操作全部委托给 `MemoryManager`，因此它是一层「权限与契约外壳 + 委派」。
- **参数**：作为类没有函数参数；构造行为见 `__init__`。类级属性 `spec` 是工具元信息声明，属于类属性而非实例属性。
- **返回**：类本身不可调用；实例化返回 `MemoryManageTool` 对象，该对象必须实现 `execute` 方法以满足基类契约。
- **内部流程**：定义时构造 `ToolSpec` 实例并绑定到类属性 `spec`（工具发现机制通常读取它来生成模型可见的工具清单）；继承 `BaseTool` 以获得基类提供的通用行为（例如依据 `input_model` 解析原始参数、依据 `timeout_seconds` 施加超时、依据 `permissions` 做鉴权、依据 `side_effect`/`idempotent`/`parallel_safe` 决定调度策略）；随后定义 `__init__`、`manager` 属性和 `execute` 三个成员来完成实例化、惰性依赖获取与实际执行。
- **异常/边界**：类定义阶段本身不抛异常。运行期的异常处理落在 `execute` 与 `manager` 上：删除时缺 `item_id` 抛 `ValueError`；底层 `MemoryManager` 抛出的异常（例如数据库不可用、条目不存在等）会沿调用链向上传播，本类不做捕获与吞掉。权限不满足、超时超限等由基类/运行时按 `spec` 声明处理。
- **同文件关系**：引用了同文件的 `MemoryManageInput`、`MemoryManageOutput`（作为 `spec` 的模型）；`manager` 属性内部调用同文件导入的 `build_default_manager`；`execute` 调用 `manager` 属性；被同文件的 `create_tool` 实例化；被模块末尾的 `__all__` 导出。此外 `__init__` 的类型标注引用了外部 `MemoryManager`。

### `MemoryManageTool.__init__(self, manager: MemoryManager | None = None) -> None` （第 70 行）
- **作用**：构造工具实例，并把可选的外部 `MemoryManager` 依赖保存到实例字段 `self._manager` 上。这里刻意只做赋值、不做任何初始化动作：注释写明「Created lazily so importing/discovering the tool never opens SQLite.」，即惰性创建是为了让「导入本模块」和「工具发现/列举工具」这两个高频动作永远不会打开 SQLite 连接。这样即使运行环境里数据库路径不可写、文件被占用或根本不存在，Agent 依然能正常启动并把工具列进清单，只有在真正执行删除/清空时才会去碰数据库。当调用方（例如测试、或使用非默认后端的应用）传入自定义管理器时，工具就会使用那个后端而不是默认后端。
- **参数**：
  - `self`：实例自身，隐式传入。
  - `manager`：类型 `MemoryManager | None`，默认 `None`。传入具体实例时表示使用该外部管理器；传 `None`（或不传）表示稍后由 `manager` 属性惰性构造默认管理器。约束是该对象需要提供 `delete(item_id, memory_type=...)` 与 `clear(memory_type=...)` 这两个方法。
- **返回**：`None`。构造函数不返回管理器，只完成实例状态初始化。
- **内部流程**：唯一一步是把形参 `manager` 直接赋给 `self._manager`。此处不做判空、不做类型校验、不触发数据库连接、不读取环境变量。默认管理器相关的 `MEMORY_DB_PATH` 解析、`memory.sqlite3` 回退等逻辑都不发生在这个方法里，而是推迟到 `manager` 属性首次被访问时。
- **异常/边界**：本方法自身不抛异常，也不做参数校验；传入非 `MemoryManager` 的鸭子类型对象不会在这里报错，问题会推迟到 `execute` 调用其方法时暴露。传 `None` 是合法且有意义的输入，表示「使用默认后端」。
- **同文件关系**：只写入实例字段 `self._manager`，供同文件的 `manager` 属性读取；不调用本文件里的任何函数。被同文件的 `create_tool` 间接调用（`create_tool` 里 `MemoryManageTool()` 无参实例化）。

### `MemoryManageTool.manager` （property，第 74 行）`-> MemoryManager`
- **作用**：这是一个只读属性，作为获取记忆管理器的统一入口，实现「按需初始化 + 缓存」的惰性单例语义。第一次访问时，如果 `self._manager` 仍是 `None`，就用 `build_default_manager()` 构造默认管理器并写回 `self._manager`；之后每次访问都直接返回已缓存的对象。这样既保证了工具在构造阶段不打开数据库，又保证后续多次 `execute` 共用同一个管理器实例，不会反复建立连接或反复解析数据库路径。它让「注入外部管理器」与「使用默认管理器」这两条路径在 `execute` 里表现得完全一致，`execute` 无需关心依赖从哪来。属性的类型标注是 `MemoryManager`，对调用方承诺永远拿到非 `None` 的管理器。
- **参数**：无显式参数（属性访问形式为 `tool.manager`），隐式 `self`。
- **返回**：返回 `MemoryManager` 实例。若构造时注入了管理器，则返回该注入对象；若注入的是 `None`，则返回由 `build_default_manager()` 创建并缓存下来的默认管理器。任何情况下都不会返回 `None`。
- **内部流程**：第一步判断 `self._manager is None`；条件为真时调用本文件导入的 `build_default_manager()`，把结果赋给 `self._manager`；随后（无论走哪个分支）执行 `return self._manager`。这是一次典型的惰性初始化（lazy init），没有加锁，因为工具声明 `parallel_safe=False`，不预期被并发调用。
- **异常/边界**：当 `build_default_manager()` 内部出错（例如默认数据库路径不可用、驱动缺失）时，异常会从属性访问处向上抛出，本属性不做捕获。若 `self._manager` 已被成功赋值则不会再次构造，因此不会重复触发该异常。返回值不可能为 `None`；若外部注入的对象本身行为异常，本属性也不做校验。
- **同文件关系**：调用了同文件导入的 `build_default_manager`；读取 `__init__` 写入的 `self._manager`；被同文件的 `execute` 调用（`self.manager.delete(...)` 与 `self.manager.clear(...)`）。

### `MemoryManageTool.execute(self, arguments: MemoryManageInput) -> MemoryManageOutput` （第 80 行）
- **作用**：这是工具的真正的执行入口，运行时在参数通过 `MemoryManageInput` 校验之后调用它，由它把抽象动作翻译成对 `MemoryManager` 的具体调用。它按 `action` 分派两条路径：`"delete"` 走单条删除，其它值（即 `"clear"`）走整层清空。之所以在 `delete` 分支里再次检查 `item_id`，是因为「删除必须给 id」是跨字段的业务约束，Pydantic 的逐字段校验表达不了，只能在这里显式兜底，避免把 `None` 传进底层导致难以理解的错误。它还负责把 `memory_type` 从可能为 `None` 的枚举值规范成具体记忆层：缺省时统一落到 `MemoryType("working")`，呼应输入模型里「默认只动工作记忆，`clear` 不会误清全库」的设计意图。最后它把结果装进 `MemoryManageOutput` 返回，并固定填 `items=[]`。
- **参数**：
  - `self`：实例自身，隐式传入。
  - `arguments`：类型 `MemoryManageInput`，必填、无默认值。是已经过校验的输入模型实例，其中 `action` 必为 `"delete"` 或 `"clear"`，`memory_type` 可能是 `None` 或某个 `MemoryScope` 成员，`item_id` 可能是 `None` 或字符串。
- **返回**：返回 `MemoryManageOutput` 实例。`action` 字段回显传入的动作字符串；`count` 字段在删除路径上是 `int(self.manager.delete(...))` 的结果，在清空路径上是 `self.manager.clear(...)` 的结果；`items` 固定为空列表。删除操作若底层返回 0，也会正常返回 `count=0`，不视为异常。
- **内部流程**：按顺序执行以下步骤：① `action = arguments.action`，取出动作字符串；② `memory_type = MemoryType(arguments.memory_type or "working")`，把 `memory_type` 为空（`None` 或假值）时替换为字符串 `"working"`，再通过 `MemoryType(...)` 转换成记忆类型枚举；③ 初始化 `count = 0`；④ 判断 `action == "delete"`：若成立，先检查 `arguments.item_id is None`，为真则 `raise ValueError("item_id is required for delete")`，否则调用 `self.manager.delete(arguments.item_id, memory_type=memory_type)`，并把结果强制转成 `int` 赋给 `count`；⑤ 否则（`action` 为 `"clear"`）调用 `self.manager.clear(memory_type=memory_type)`，把结果赋给 `count`；⑥ 构造并返回 `MemoryManageOutput(action=action, count=count, items=[])`。其中第④步的 `self.manager` 触发同文件 `manager` 属性的惰性初始化。
- **异常/边界**：当 `action == "delete"` 且 `item_id` 为 `None` 时抛 `ValueError`，消息为 `"item_id is required for delete"`。当 `arguments.memory_type` 是一个 `MemoryType(...)` 无法识别的值时，构造枚举会抛 `ValueError`；注意这里用的是 `or "working"`，因此空字符串等假值也会被当成缺省值而落到 `working`。底层 `manager.delete`/`manager.clear` 抛出的异常（条目不存在、数据库锁定、磁盘错误、超时等）不被捕获，直接向上传播，由运行时按 `spec` 的 `timeout_seconds=10.0`、`side_effect="destructive"` 等声明处理。`count` 若底层返回非整数，`delete` 路径会因 `int(...)` 转换而抛 `TypeError`/`ValueError`，`clear` 路径则会把原值交给 `MemoryManageOutput` 的严格校验，可能抛 `pydantic.ValidationError`。返回的 `items` 始终为空，不携带被删数据快照。
- **同文件关系**：读取同文件的 `MemoryManageInput` 字段、构造并返回同文件的 `MemoryManageOutput`、通过 `self.manager` 调用同文件的 `manager` 属性（进而间接调用同文件导入的 `build_default_manager`）；不调用 `create_tool`。被运行时/基类在参数解析后调用，是本文件对外的实际执行点。

### `create_tool() -> BaseTool` （第 93 行）
- **作用**：这是模块级的零参数工厂函数，是工具包约定的「创建工具实例」的标准出口。运行时或工具注册表按名字加载本模块后，通过调用它来获得一个可用的 `MemoryManageTool` 实例，而不需要自己知道类的构造细节，也不需要自己准备管理器（无参构造即表示走默认后端、惰性连接）。它同时起到解耦作用：调用方只依赖 `BaseTool` 这个抽象类型和 `create_tool` 这个名字，将来若替换具体实现类，只要保持工厂签名不变，调用方无需改动。函数体只有一行 `return MemoryManageTool()`，因此没有任何副作用，也不会在创建时打开 SQLite（因为 `__init__` 是惰性的）。
- **参数**：无参数。它不接收管理器等任何可配置项，因此通过它创建的工具必然使用默认管理器路径（`MEMORY_DB_PATH` 或项目旁的 `memory.sqlite3`）。需要自定义后端的调用方应直接实例化 `MemoryManageTool(manager=...)`。
- **返回**：返回一个 `MemoryManageTool` 实例，声明类型为基类 `BaseTool`（返回类型标注宽于实际类型，便于调用方按抽象类型使用）。
- **内部流程**：唯一一步是调用 `MemoryManageTool()` 无参构造并把实例返回。无参意味着 `__init__` 里 `manager` 取默认 `None`，`self._manager` 被置为 `None`，默认管理器的构造被推迟到 `manager` 属性首次访问（即第一次 `execute`）时。
- **异常/边界**：正常路径不抛异常。由于构造过程不触达数据库，即使数据库路径不可用，本函数也能成功返回实例，问题会推迟到 `execute` 阶段暴露。没有任何空值/非法值处理，因为它不接受参数。
- **同文件关系**：调用同文件的 `MemoryManageTool.__init__`（经由类实例化）间接创建工具；被模块末尾的 `__all__` 导出；不被本文件内其它成员调用，是本文件对外的创建入口。

### 模块级语句与常量（第 13–24 行、第 97–102 行）
- **作用**：这些不是函数或类，但构成文件的运行环境与对外契约，一并说明。
  - `from __future__ import annotations`（第 13 行）：让所有注解以字符串形式延迟求值，使 `MemoryManager | None`、`MemoryScope | None` 这类现代联合类型写法在较低 Python 版本上也能安全解析。
  - `from typing import Any, Literal`（第 15 行）：`Any` 用于输出模型的 `list[dict[str, Any]]`，`Literal` 用于把 `action` 限定为两个字面量。
  - `from pydantic import BaseModel, ConfigDict, Field`（第 17 行）：`BaseModel` 作为两个输入输出模型的基类，`ConfigDict` 配置严格模式与禁止额外字段，`Field` 为 `memory_type`、`item_id`、`items` 提供默认值与描述。
  - `from core import BaseTool, ToolSpec`（第 19 行）：工具基类与工具元信息声明结构，是 `MemoryManageTool` 的继承与声明来源。
  - `from memory import MemoryManager, MemoryType`（第 20 行）：管理器类型用于类型标注，`MemoryType` 用于把输入层名转换成运行期枚举。
  - `from ._memory import MemoryScope, build_default_manager`（第 22 行）：同包私有模块提供的记忆层枚举（约束输入合法取值）与默认管理器工厂（惰性构造用）。
  - `TOOL_ENABLED = True`（第 24 行）：模块级开关常量，声明本工具在工具发现阶段处于启用状态。
  - `__all__`（第 97–102 行）：显式声明模块对外导出 `MemoryManageInput`、`MemoryManageOutput`、`MemoryManageTool`、`create_tool` 四个名字，未列入的（如 `TOOL_ENABLED`）不参与 `from module import *`。
- **参数**：无（均为模块级语句）。
- **返回**：无返回值；执行后建立模块的导入绑定、常量和导出清单。
- **内部流程**：导入阶段按书写顺序解析各依赖，随后绑定 `TOOL_ENABLED = True`，在文件末尾构造 `__all__` 列表。导入过程不实例化工具、不打开数据库。
- **异常/边界**：若 `core`、`memory` 或 `._memory` 任一模块不可导入，本模块导入即失败并抛 `ImportError`；`TOOL_ENABLED` 无任何边界逻辑；`__all__` 中的名字必须真实存在，否则 `from ... import *` 会报 `AttributeError`。
- **同文件关系**：这些绑定被本文件所有类与函数引用（`BaseTool`/`ToolSpec` 被 `MemoryManageTool` 使用，`MemoryScope` 被 `MemoryManageInput` 使用，`MemoryType` 与 `MemoryManager` 被 `execute`/`__init__` 使用，`build_default_manager` 被 `manager` 属性使用）；`__all__` 汇总导出本文件的公开成员。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `MemoryManageInput` | `memory.manage` 的严格输入模型，用 `action`/`memory_type`/`item_id` 三个字段描述「删哪条、清哪层」。 |
| `MemoryManageOutput` | `memory.manage` 的输出模型，回传 `action`、影响条数 `count` 与预留的 `items` 列表。 |
| `MemoryManageTool` | 继承 `BaseTool` 的破坏性记忆管理工具类，用 `ToolSpec` 声明元信息并把操作委派给 `MemoryManager`。 |
| `MemoryManageTool.__init__` | 只把可选管理器存入 `self._manager`，刻意不触发任何数据库连接（惰性创建）。 |
| `MemoryManageTool.manager` | 只读属性，首次访问时用 `build_default_manager()` 惰性构造并缓存默认管理器。 |
| `MemoryManageTool.execute` | 按 `action` 分派删除或清空，校验 `delete` 必须有 `item_id`，返回带 `count` 的输出模型。 |
| `create_tool` | 零参数工厂函数，返回一个走默认后端的 `MemoryManageTool` 实例。 |
| 模块级常量与导入（`TOOL_ENABLED`、`__all__` 等） | 建立依赖绑定、声明工具启用状态并规定模块对外导出的四个名字。 |
