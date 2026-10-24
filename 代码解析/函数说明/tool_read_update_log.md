# tool/read_update_log.py

## 一、这个文件是干什么的

这个文件实现了一个「单条更新日志查看器」工具，是项目更新日志（update log）体系里的读侧入口之一。它对外暴露的工具名是 `system.read_update_log`，作用是：给定一个正整数形式的 `update_id`，从持久化的更新日志仓库里取出**恰好一条**完整记录，解码成字段稳定的结构化对象返回给调用方（也就是 Agent 运行时里的模型/工具调用者）。

文件开头的模块文档字符串明确写了它的设计意图：这个查看器**故意只做单条查询**，绝不把整份 append-only（只追加）的历史日志列举出来或整体加载进调用方的上下文。这是为了在审计、排障时只取需要的那一条，避免上下文被大量历史记录撑爆。

文件里主要包含四类东西：两个 Pydantic 输入/输出模型（`ReadUpdateLogInput`、`ReadUpdateLogOutput`），一个继承自框架基类 `BaseTool` 的工具类（`ReadUpdateLogTool`），一个给自动发现机制用的零参工厂函数（`create_tool`），以及一个模块级开关常量 `TOOL_ENABLED` 和导出清单 `__all__`。

它怎么被用到：运行时会扫描 `tool/` 目录下的模块，看到 `TOOL_ENABLED = True` 就把这个模块当作已启用的工具；再调用 `create_tool()` 拿到零参构造的工具实例，把实例的 `spec`（工具名、描述、输入输出模型、副作用、权限、超时、幂等性、并发上限、标签、中文引导语）注册进工具注册表。之后模型在需要查看某一条更新记录时会调用 `system.read_update_log`，框架校验参数后调用 `ReadUpdateLogTool.execute()`，由它去访问 `UpdateLogRepository` 取数据并组装返回值。

依赖关系上，它从 `core` 拿 `BaseTool` 和 `ToolSpec` 这两个框架基建，从 `core.update_log` 拿数据访问层 `UpdateLogRepository`，从同目录的 `tool.update_log` 复用文件变更记录模型 `UpdateLogFileChange`——也就是说写侧（`tool/update_log.py`）定义的字段结构在这里被原样复用于读侧，保证读写两侧字段名一致。

## 二、函数与类逐条详解

### `TOOL_ENABLED`（模块级常量，第 15 行）
- **作用**：这是工具自动发现机制的开关标记。运行时在遍历 `tool/` 包时，会检查每个模块是否带有这个常量并且值为真；只有为真时才会把该模块当成一个可加载的工具模块，进而调用它的 `create_tool()` 去实例化工具。它存在的意义是让「写好了但暂时不想让模型看到」的工具可以通过改成 `False` 快速下线，而不必删除文件或改动注册代码。本文件把它设为 `True`，表示 `system.read_update_log` 在项目运行中是默认可用的。
- **参数**：无（它是常量，不是函数）。
- **返回**：无。它的值 `True` 被自动发现逻辑读取。
- **内部流程**：无流程，纯粹是一个模块导入时即被赋值的布尔字面量。它位于所有 import 之后、第一个类定义之前，因此模块被导入完成的瞬间它就已经存在于模块命名空间中。
- **异常/边界**：无特殊处理。若被改成 `False` 或被删除，工具不会被注册，但模块本身仍可被显式导入（此时 `ReadUpdateLogTool` 等符号依然可用）。
- **同文件关系**：被本文件的 `__all__` 导出；不被本文件内任何函数调用，只被外部（工具发现机制）读取。

### `class ReadUpdateLogInput(BaseModel)` （第 18 行）
- **作用**：这是工具调用参数的输入模型，用来描述并校验「调用这个工具时必须且只能提供什么」。它只接受一个标识符 `update_id`，也就是要查看的那一条日志的主键。之所以单独定义成 Pydantic 模型，是为了让框架在做参数校验时能统一地把模型生成的 JSON Schema 交给模型，同时用 `ge=1` 这样的约束在进入业务逻辑之前就把非法的 0、负数拦掉，避免下游仓储层拿到无意义的 ID 去查库。它是 `ReadUpdateLogTool.spec` 里的 `input_model`，每次模型发起调用、框架反序列化参数时都会构造它。
- **参数**：类的构造参数即它的字段：
  - `update_id: int`，必填，无默认值；约束为 `ge=1`，即必须是大于等于 1 的整数；描述文本为 “Positive update ID of the single log entry to retrieve”，说明它是单条日志记录的正数主键。
  - 另外通过 `model_config = ConfigDict(extra="forbid", strict=True)` 施加两条模型级约束：`extra="forbid"` 表示不允许出现模型未声明的多余字段（多传字段会直接报校验错误）；`strict=True` 表示严格类型模式，不做宽松的类型强制转换（例如不会把字符串 `"3"` 悄悄当成整数 3）。
- **返回**：作为类本身它不返回业务值；实例化后得到一个携带 `update_id` 属性的不可变语义的校验对象（Pydantic v2 的 `BaseModel`）。
- **内部流程**：类体先声明 `model_config`，再声明唯一字段 `update_id`，用 `Field(ge=1, description=...)` 附加约束与说明。Pydantic 在类创建时（元类阶段）收集这些声明并生成校验器；实例化时按 `strict` + `ge` 规则校验传入数据，通过则构造实例，不通过则抛 `pydantic.ValidationError`。本类没有任何自定义方法，全部行为由 Pydantic 提供。
- **异常/边界**：传入缺失 `update_id`、`update_id` 为 0 或负数、类型不是严格整数、或携带了额外字段时，Pydantic 会抛 `pydantic.ValidationError`，不会进入工具的业务代码。没有超时概念。
- **同文件关系**：被 `ReadUpdateLogTool.spec` 引用为 `input_model`，并在 `ReadUpdateLogTool.execute()` 中通过 `isinstance` 做类型检查；本文件内不调用任何其它函数。

### `class ReadUpdateLogOutput(BaseModel)` （第 29 行）
- **作用**：这是工具的成功返回模型，定义了「一条完整、已解码的更新日志行」对外呈现的稳定字段集合。之所以需要它，是因为底层仓储返回的是一条字典形式的原始行，直接暴露给模型既不稳定也不安全；用一个显式模型把所有字段名、长度上限、必填性固定下来，可以让输出契约稳定（不会因为底层列增减而漂移），同时用 `min_length`/`max_length` 给每个文本字段加上边界，防止异常大的内容把调用方上下文撑爆。它也是 `ReadUpdateLogTool.spec` 的 `output_model`，框架据此生成输出 schema，`execute()` 的最后一步就是用构造它的方式完成「解码 + 校验」。
- **参数**：类的构造参数就是下列字段（全部必填，无默认值）：
  - `update_id: int`，约束 `ge=1`，存储的更新 ID。
  - `timestamp: str`，长度 1–64，UTC 写入时间戳。
  - `system_name: str`，长度 1–200，记录该次更新时所在的操作系统名称。
  - `executor: str`，长度 1–200，做出改动的 AI/模型或人。
  - `update_type: str`，长度 1–64，变更分类。
  - `title: str`，长度 1–300，简短更新标题。
  - `task_background: str`，长度 1–4000，变更的原因与目标。
  - `update_details: str`，长度 1–12000，具体实现细节与决策。
  - `added_features: str`，长度 1–6000，新增能力（没有则写“无”之类的占位文本）。
  - `files: list[UpdateLogFileChange]`，长度 1–100，本次更新涉及的文件列表；元素类型复用写侧模型 `UpdateLogFileChange`。
  - `behavior_impact: str`，长度 1–6000，兼容性与用户影响说明。
  - `validation: str`，长度 1–6000，实际执行过的检查及其结果。
  - `risks: str`，长度 1–4000，已知风险与回滚说明。
  - `follow_up: str`，长度 1–4000，剩余待办（没有则写占位文本）。
  - `latest_update_id: int`，约束 `ge=0`，当前已存储的最大更新 ID；字段描述里特别说明：要审计完整历史时，可以从 1 到这个值逐条读取。
  - 模型级约束同样为 `model_config = ConfigDict(extra="forbid", strict=True)`，即不允许多余字段、严格类型。
- **返回**：类本身不返回业务值；实例化后得到一个经过校验的输出对象，框架再把它序列化交给调用方。
- **内部流程**：类体逐条声明 15 个字段及各自的 `Field` 约束，全部没有默认值，因此任一字段缺失都会校验失败。`execute()` 中通过 `ReadUpdateLogOutput(**record)` 把仓储返回的字典按关键字展开传入，Pydantic 逐字段校验后构造实例。
- **异常/边界**：如果仓储返回的字典缺少任何一个字段、字段类型不严格匹配（例如 `update_id` 是字符串）、文本为空字符串、超出长度上限、`files` 为空列表或超过 100 条、或含未声明字段，都会抛 `pydantic.ValidationError`。注意所有文本字段的 `min_length=1`，意味着底层必须存有非空内容，否则会在这里暴露为校验错误而不是返回空串。没有超时概念。
- **同文件关系**：被 `ReadUpdateLogTool.spec` 引用为 `output_model`；在 `ReadUpdateLogTool.execute()` 的返回语句中被构造。它自身依赖同目录 `tool.update_log` 的 `UpdateLogFileChange` 作为 `files` 元素类型。

### `class ReadUpdateLogTool(BaseTool)` （第 71 行）
- **作用**：这是本文件的核心工具类，语义是「按 ID 精确读取恰好一条不可变的更新日志记录」。它继承框架基类 `BaseTool`，把工具元数据（`spec`）和真正的执行逻辑（`execute`）绑在一起，并通过可注入的仓储对象与数据层解耦。运行时在自动发现阶段会用 `create_tool()` 构造它的实例并注册；模型调用 `system.read_update_log` 时，框架最终调到它的 `execute()`。因为它的元数据声明了 `side_effect="read"`、`idempotent=True`、`parallel_safe=True`，运行时可以放心地把多次调用并发执行、也可以安全重试，不会产生写副作用。
- **参数**：类本身无构造参数语义上的业务参数，只有自定义的 `__init__` 接受一个可选仓储（见下条）。类体级属性 `spec` 是一个 `ToolSpec`，其中各字段取值为：`name="system.read_update_log"`；`description` 说明它按正数 ID 取回一条完整更新记录，用于定向审计或排障，只读、绝不批量列出或返回历史；并说明每次结果都会附带当前最大 ID，便于从 1 顺序读完；`version="1.0"`；`input_model=ReadUpdateLogInput`；`output_model=ReadUpdateLogOutput`；`side_effect="read"`；`permissions=()`（空元组，不申请任何额外权限）；`timeout_seconds=5.0`；`idempotent=True`；`parallel_safe=True`；`max_concurrency=8`；`tags=("update-log", "audit", "read", "project")`；`guidance` 是一句中文引导语，提示「需要看某一条更新记录的完整内容时用它；要连续读一段用 system.read_update_logs。返回值里带当前最大 ID，可据此从 1 顺序读完。」
- **返回**：作为类不返回业务值；实例是一个可被注册和调用的工具对象。
- **内部流程**：类体先以类属性方式构造并绑定 `ToolSpec`（这一步在模块导入、类定义时完成，是静态元数据），随后定义 `__init__` 与 `execute` 两个方法。没有任何类方法、静态方法或属性装饰器，也没有嵌套函数。
- **异常/边界**：类定义阶段本身不抛异常；`ToolSpec` 的字段若填错类型会在导入时暴露。业务异常都发生在 `execute()` 里（见下）。
- **同文件关系**：被同文件 `create_tool()` 实例化并返回；引用同文件 `ReadUpdateLogInput`、`ReadUpdateLogOutput` 作为 spec 的输入输出模型；其 `execute()` 内部使用 `self.repository`（`UpdateLogRepository`）以及输出模型。

### `ReadUpdateLogTool.__init__(self, repository: UpdateLogRepository | None = None) -> None` （第 97 行）
- **作用**：构造工具实例，并解决「仓储从哪来」的问题。它接受一个可选的仓储对象：测试或需要复用连接时可以从外部注入一个 `UpdateLogRepository`；正常运行时没人传参，它就自己新建一个默认仓储。这种「可选依赖注入 + 默认回退」的写法让工具既能被自动发现机制零参构造（`create_tool()` 就是零参调用），又能在单元测试里换成假仓储而无需改代码。`UpdateLogRepository` 被保存在实例属性 `self.repository` 上，供后续每次 `execute()` 复用，避免每次读日志都重新建立数据访问对象。
- **参数**：
  - `self`：实例自身。
  - `repository: UpdateLogRepository | None`，默认值 `None`。传 `None` 表示「不注入，自己创建默认仓储」；传入具体仓储实例则原样使用。类型注解使用 `|` 联合语法，配合文件顶部的 `from __future__ import annotations` 保证在旧版本 Python 上也能安全地延迟求值注解。
- **返回**：返回 `None`（构造函数约定），效果是设置好 `self.repository`。
- **内部流程**：单行三元表达式 `self.repository = repository if repository is not None else UpdateLogRepository()`。判断依据是 `is not None`，因此传 `None` 与不传等价，都会新建默认仓储；注意这里没有做「仓储可用性探测」，创建动作是否真的连上存储由 `UpdateLogRepository` 自己的构造逻辑决定。
- **异常/边界**：如果 `UpdateLogRepository()` 在构造时（例如读取配置、打开文件、连接数据库）失败，异常会从这个 `__init__` 直接向上抛出，工具实例就创建不出来，自动发现阶段会表现为该工具注册失败。对 `None` 的处理就是回退到默认仓储；对非法类型（例如传了个字符串）不做检查，直到后续 `execute()` 调用其方法时才会报错。
- **同文件关系**：被同文件 `create_tool()` 间接调用（`ReadUpdateLogTool()` 零参构造即走这里）；它自身不调用本文件任何其它函数，只实例化外部类 `UpdateLogRepository`。

### `ReadUpdateLogTool.execute(self, arguments: ReadUpdateLogInput) -> ReadUpdateLogOutput` （第 100 行）
- **作用**：这是工具真正干活的方法，也是模型调用 `system.read_update_log` 时被执行的那段逻辑。它按传入的 `update_id` 去仓储里取一条记录：取到了就把「当前最大 ID」补进这条记录，再用输出模型构造并返回；取不到就明确抛 `LookupError` 告诉调用方这条记录不存在。之所以要在返回里附上 `latest_update_id`，是因为工具刻意不提供「列出全部」的能力，调用方只能靠这个最大 ID 自己从 1 逐条推进，从而在保持单条查询语义的同时仍然能够完成整段历史的审计。整个方法没有任何写入动作，符合 spec 里 `side_effect="read"`、`idempotent=True` 的声明，可以安全并发和重试。
- **参数**：
  - `self`：工具实例，提供 `self.repository`。
  - `arguments: ReadUpdateLogInput`，必填。必须是 `ReadUpdateLogInput` 的实例（方法内部会显式校验），其中携带 `arguments.update_id`（严格正整数，`ge=1`）。框架在正常调用路径上已经用输入模型校验过原始 JSON 参数，所以这里拿到的是已校验对象；显式 `isinstance` 检查是为了防止有人绕过框架直接以字典或其它类型调用。
- **返回**：返回 `ReadUpdateLogOutput` 实例，即一条完整解码、字段稳定的更新日志记录，并且其中的 `latest_update_id` 已经被填成仓储当前的最大 ID（取值 `ge=0`）。只有在记录存在且所有字段都通过输出模型校验时才会返回；记录不存在时不会返回 `None`，而是抛异常。
- **内部流程**：
  1. 首先做类型守卫：`if not isinstance(arguments, ReadUpdateLogInput): raise TypeError("arguments must be a ReadUpdateLogInput instance")`，把「传错类型」和「记录不存在」两种情况区分成不同异常，便于上层给出准确提示。
  2. 调用 `self.repository.get(arguments.update_id)` 执行单条查询，结果记为 `record`。这一步是唯一的 I/O，具体读文件还是读数据库由仓储实现决定。
  3. 判空：`if record is None: raise LookupError(f"update log entry {arguments.update_id} was not found")`，错误信息里带上具体 ID 便于排障。
  4. 补字段：`record["latest_update_id"] = self.repository.latest_id()`，通过第二次仓储调用拿到当前最大 ID 并写回这条记录字典。注意这里是**就地修改**了仓储返回的字典对象。
  5. 构造输出：`return ReadUpdateLogOutput(**record)`，把字典按关键字展开交给 Pydantic 做字段级校验并生成最终返回对象。
- **异常/边界**：
  - `TypeError`：`arguments` 不是 `ReadUpdateLogInput` 实例时抛出。
  - `LookupError`：仓储返回 `None`（该 ID 不存在）时抛出，消息形如 `update log entry <ID> was not found`。
  - `pydantic.ValidationError`：`record` 缺少输出模型要求的字段、类型不严格匹配、文本为空或超出长度上限、`files` 数量越界、或含多余字段时，在 `ReadUpdateLogOutput(**record)` 这一步抛出。
  - 仓储自身抛出的异常（读文件失败、连接断开等）不在这里捕获，会原样向上冒泡。
  - 超时方面，方法内没有显式超时控制，依赖 spec 里的 `timeout_seconds=5.0` 由框架层施加。
  - 边界细节：即使 `update_id` 合法但大于当前最大 ID，也会走 `record is None` 分支抛 `LookupError`；`latest_update_id` 允许为 0，表示仓库为空（此时任何 ID 都查不到）。
  - 就地修改 `record` 字典是一个隐含副作用：如果仓储实现返回的是内部缓存的引用而非副本，这次赋值会污染缓存中的对象；在正常的「每次查库返回新字典」实现下无影响。
- **同文件关系**：调用本文件内的 `ReadUpdateLogInput`（`isinstance` 校验）和 `ReadUpdateLogOutput`（构造返回值）；被外部框架调用，本文件内没有其它函数调用它。

### `create_tool() -> BaseTool` （第 110 行）
- **作用**：这是给自动发现机制准备的标准工厂函数。运行时扫描到 `TOOL_ENABLED = True` 之后，需要一个统一、零参的方式来拿到工具实例，`create_tool()` 就承担这个角色：它不接收任何参数，直接返回一个新构造的 `ReadUpdateLogTool`，因此每个调用点都会得到互相独立、各自持有默认仓储的实例。它不自己构造仓储，是为了让 `ReadUpdateLogTool.__init__` 的默认逻辑成为唯一入口，避免两处重复决策。函数体只有一行，属于薄封装。
- **参数**：无参数。
- **返回**：返回 `BaseTool` 类型的对象；实际运行时返回的具体类型是 `ReadUpdateLogTool`，标注为基类是为了让发现逻辑面向抽象编程。永远不会返回 `None`，除非构造过程中抛异常。
- **内部流程**：执行 `return ReadUpdateLogTool()`。这一行会走 `ReadUpdateLogTool.__init__`，因未传 `repository`，`__init__` 内部回退为 `UpdateLogRepository()`，于是每次调用都会创建一个新的默认仓储绑定到这个实例上。
- **异常/边界**：没有参数，因此没有参数校验问题；若 `UpdateLogRepository()` 构造失败（配置缺失、存储不可达等），异常会直接传播给调用方，发现流程会因此报错。无超时处理。
- **同文件关系**：调用同文件的 `ReadUpdateLogTool`（构造）；本文件内没有函数调用它，它只服务于外部的自动发现机制。

### `__all__`（模块级导出清单，第 116 行）
- **作用**：显式声明本模块对外公开的符号列表，包含 `"TOOL_ENABLED"`、`"ReadUpdateLogInput"`、`"ReadUpdateLogOutput"`、`"ReadUpdateLogTool"`、`"create_tool"` 五项。它让 `from tool.read_update_log import *` 只导出这五个名字，同时也向读者和静态检查工具表明：这五项是本模块的公共接口，其余（如导入进来的 `BaseTool`、`ToolSpec`、`UpdateLogRepository`、`UpdateLogFileChange`、`BaseModel`、`ConfigDict`、`Field`）都是实现细节，不属于本模块的对外契约。
- **参数**：无（它是模块级列表字面量）。
- **返回**：无。
- **内部流程**：无流程，模块导入时即创建该列表。注意它只影响星号导入与文档语义，不阻止显式 `from tool.read_update_log import BaseTool` 这类直接引用。
- **异常/边界**：无特殊处理。
- **同文件关系**：列出本文件定义的 `TOOL_ENABLED`、两个输入输出模型、工具类与 `create_tool`；不被本文件任何函数调用。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `TOOL_ENABLED` | 模块级布尔开关，为 `True` 时让自动发现机制加载并注册本工具。 |
| `class ReadUpdateLogInput(BaseModel)` | 工具输入模型，只接受一个严格正整数 `update_id`，禁止多余字段。 |
| `class ReadUpdateLogOutput(BaseModel)` | 工具输出模型，用 15 个带长度与取值约束的字段固定单条更新日志的返回契约。 |
| `class ReadUpdateLogTool(BaseTool)` | 工具主体类，声明 `system.read_update_log` 的元数据（只读、幂等、可并发、5 秒超时、并发上限 8）并提供执行逻辑。 |
| `ReadUpdateLogTool.__init__(self, repository=None)` | 构造工具实例，可选注入仓储，未注入时新建默认 `UpdateLogRepository`。 |
| `ReadUpdateLogTool.execute(self, arguments)` | 校验参数类型后按 ID 取一条记录，补上当前最大 ID 并返回，查不到则抛 `LookupError`。 |
| `create_tool()` | 零参工厂，返回一个新的 `ReadUpdateLogTool()` 实例供自动发现使用。 |
| `__all__` | 模块导出清单，公开 `TOOL_ENABLED`、两个模型、工具类与 `create_tool` 五项。 |
