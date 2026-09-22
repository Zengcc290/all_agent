# memory/types/working.py

## 一、这个文件是干什么的

这个文件实现了四层记忆系统里的「工作记忆」（Working Memory）这一层，也就是最贴近当前对话、容量最小、最易失的那一层记忆。整个文件只有 34 行，核心是一个继承自 `BaseMemory` 的子类 `WorkingMemory`，它在父类提供的通用存储/检索能力之上，额外叠加了两条策略：**容量上限（capacity）** 与 **基于重要性的淘汰（eviction）**。文件开头的模块文档字符串 `"""Working memory with TTL and capacity-based eviction."""` 已经点明了它的职责关键词：TTL（存活时间，由父类体系负责）与按容量淘汰（由本文件负责）。

类内部通过类属性 `memory_type = MemoryType.WORKING` 向基类体系声明自己的身份，这样在工厂创建、按类型路由、序列化持久化、以及上层检索时，都能凭这个枚举值把工作记忆和其它层（如短期、长期、情景记忆）区分开。构造函数 `__init__` 除了把 `memory_type` 透传给父类外，还解析出 `capacity` 并做严格的类型与取值校验，保证容量永远是「正整数」。覆写后的 `add` 方法在真正写入之后立刻触发一次淘汰检查，从而让容量约束成为「写入即生效」的硬约束，而不是事后惰性清理。私有方法 `_evict_if_needed` 是淘汰逻辑的唯一实现处：它拉取当前全部条目，超限时按 `(importance, updated_at)` 升序排序，优先删掉重要性最低、且最久没有被更新过的那些条目，直到条目数回落到容量以内。

在运行时，它被上层 Agent 循环在每一轮对话中频繁调用：用户输入、工具返回、模型思考片段都会被塞进工作记忆作为「当前上下文」，而 `list()`/`search()` 之类的读取由父类提供。文件末尾的 `__all__ = ["WorkingMemory"]` 显式声明了本模块对外只导出这一个名字，避免调用方误引用内部细节。

## 二、函数与类逐条详解

### `class WorkingMemory(BaseMemory)` （第 10 行）
- **作用**：这是本文件唯一对外暴露的类，代表「工作记忆」这一层记忆的完整实现。它本身只声明了 `memory_type = MemoryType.WORKING` 这一个类属性，用来向基类与上层框架表明自己的记忆类型身份；真正的存储、读取、序列化等通用能力全部继承自 `BaseMemory`，本类只负责叠加工作记忆特有的两条策略——容量上限与超限淘汰。之所以需要单独建一个类而不是直接用 `BaseMemory`，是因为工作记忆的语义是「当前活跃上下文的小窗口」，必须有别于可无限增长的长期记忆，因此需要独立的容量约束与淘汰规则。它会被上层在构建四层记忆系统时实例化一次（通常在记忆工厂或应用启动阶段），随后在每一轮对话中被反复写入与读取。类属性 `memory_type` 还会被构造函数透传给父类，使父类内部的元数据、持久化记录、按类型检索都能正确归类。
- **参数**：类定义本身不接收参数；实例化时接收的关键字参数见下面的 `__init__`。
- **返回**：类不是函数，实例化后返回 `WorkingMemory` 实例，该实例同时是 `BaseMemory` 的实例（满足 `isinstance(obj, BaseMemory)`）。
- **内部流程**：类体只有一行有效代码 `memory_type = MemoryType.WORKING`，这是类级别（而非实例级别）的常量赋值，在类定义被导入执行时即完成绑定。继承关系决定了属性查找顺序：实例上找不到的属性会先查 `WorkingMemory` 类，再沿 MRO 查到 `BaseMemory`，因此本类不需要重复实现 `list`、`delete`、`search`、`get` 等父类方法。当子类方法（如 `add`）内部调用 `super().add(...)` 时，实际执行的是父类实现，从而在复用父类逻辑的同时插入自己的淘汰钩子。
- **异常/边界**：类定义阶段不抛异常；`MemoryType.WORKING` 若在枚举中不存在会在导入期就抛 `AttributeError`（当前代码假定它存在）。实例化阶段的校验异常由 `__init__` 负责。
- **同文件关系**：它定义了本文件中的 `__init__`、`add`、`_evict_if_needed` 三个方法；`add` 调用 `_evict_if_needed`，`_evict_if_needed` 调用父类的 `list` 与 `delete`，`__init__` 调用父类的 `__init__`。被本文件末尾的 `__all__` 导出。

### `__init__(self, *, capacity: int | None = None, **kwargs: Any) -> None` （第 13 行）
- **作用**：这是 `WorkingMemory` 的构造函数，负责把实例初始化到可用状态，并确定这一层工作记忆最多能容纳多少条记录。它做了两件事：一是把类属性 `memory_type` 连同其余关键字参数一起交给父类 `BaseMemory.__init__`，让父类完成存储后端、TTL 配置、命名空间等通用初始化；二是解析容量值——如果调用方显式传了 `capacity` 就用调用方的，否则退回读取配置对象上的 `working_memory_capacity` 默认值。紧接着它对容量做严格校验，确保它既不是布尔值、也不是非整数、更不能小于 1，否则直接抛 `ValueError`。这样设计的原因是容量是本类淘汰逻辑的核心参数，一旦是 0、负数或浮点数，`_evict_if_needed` 中的切片运算和比较就会产生难以追踪的错误行为，因此在构造阶段就「快速失败」。这个方法在应用启动构建记忆系统时被调用一次，通常不会在运行期反复调用。
- **参数**：
  - `self`：实例本身，由 Python 自动传入。
  - `capacity: int | None = None`：关键字限定参数（由 `*` 强制只能以关键字形式传入，不能按位置传）。为 `None` 时表示「未显式指定」，此时从 `self.config.working_memory_capacity` 读取默认容量；为整数时表示显式覆盖。约束是最终解析出的值必须是**正整数**（`int` 且 `>= 1`），并且不能是 `bool`（因为 Python 中 `True`/`False` 是 `int` 的子类，会被 `isinstance(x, int)` 误判为合法）。
  - `**kwargs: Any`：任意额外的关键字参数，不做任何检查与消费，原样透传给父类 `BaseMemory.__init__`（可能包含存储后端、TTL、命名空间、配置对象等），因此其具体合法键由父类决定。
- **返回**：无返回值（返回 `None`），构造函数只产生副作用——初始化实例状态并绑定 `self.capacity`。
- **内部流程**：第一步调用 `super().__init__(memory_type=self.memory_type, **kwargs)`，把类属性 `MemoryType.WORKING` 作为 `memory_type` 显式传入，同时把 `kwargs` 展开传给父类；父类执行完后，实例上才会有 `self.config`（本方法下一行就要用它）。第二步计算容量：`self.capacity = self.config.working_memory_capacity if capacity is None else capacity`，这是一个三元表达式，注意判断条件用的是 `capacity is None` 而不是假值判断，因此显式传 `0` 不会被误当成「未指定」而回退到默认值，而是会走到下一步被校验拦下并抛错。第三步做三重校验：`isinstance(self.capacity, bool)` 拦截布尔值，`not isinstance(self.capacity, int)` 拦截浮点数/字符串/`None` 等一切非整数，`self.capacity < 1` 拦截 0 与负数；三者用 `or` 连接，只要任一成立就 `raise ValueError("capacity must be a positive integer")`。校验通过后 `self.capacity` 保持为已赋值的整数。
- **异常/边界**：当 `capacity` 解析结果为布尔值、非整数或小于 1 时抛 `ValueError("capacity must be a positive integer")`；若 `capacity` 为 `None` 且 `self.config` 没有 `working_memory_capacity` 属性，会由属性访问抛出 `AttributeError`（本文件未做兜底）；父类 `__init__` 自身也可能因 `kwargs` 非法而抛异常，本文件不做捕获与转换。边界情况：`capacity=1` 合法，表示只保留一条；`capacity=True` 会被拒绝（因为显式排除了 `bool`）。
- **同文件关系**：调用父类 `BaseMemory.__init__`（跨文件）；被本文件的 `_evict_if_needed` 间接依赖（后者读取 `self.capacity`）；不调用本文件中的其它方法。它由外部实例化 `WorkingMemory(...)` 时自动调用。

### `add(self, content: str, **kwargs: Any) -> MemoryItem` （第 19 行）
- **作用**：这是对父类 `add` 的覆写，作用是在「写入一条新记忆」这个动作上附加工作记忆专属的容量控制。它先完整复用父类的写入流程——由父类负责生成 `MemoryItem`（分配 id、记录时间戳、计算重要性与 TTL、落盘或写入后端）——拿到返回的 `item` 之后，立刻调用 `self._evict_if_needed()` 检查是否超容，超了就删掉最不重要的若干条。之所以把淘汰放在 `add` 之后而不是放在读取时惰性执行，是为了保证任何时刻工作记忆的条目数都不会超出容量上限，读取方（上层 Agent 拼装上下文时）拿到的永远是已经裁剪过的窗口。这个方法在运行时被调用得最频繁：每一轮对话中新产生的消息、工具调用结果、中间推理片段都会通过它进入工作记忆。
- **参数**：
  - `self`：实例本身。
  - `content: str`：要写入的记忆正文，通常是文本内容（用户输入、模型输出、工具结果等）。本方法不对它做类型或非空校验，直接交给父类处理，因此空字符串是否被接受取决于父类。
  - `**kwargs: Any`：写入的可选控制项，原样透传给父类，例如重要性（importance）、TTL、标签、元数据、来源等；具体合法键由父类定义，本方法不做检查、不做消费。
- **返回**：返回父类 `add` 产生的 `MemoryItem` 对象，即刚写入的那条记忆。即使这条记忆随后在淘汰阶段被删除（理论上当容量为 1 且新条目重要性最低时可能发生），返回的仍是该 `MemoryItem` 本身，本方法不做「是否被淘汰」的二次判断或 `None` 返回。
- **内部流程**：第一步 `item = super().add(content, **kwargs)`，把内容与全部关键字参数交给父类完成真正的写入并接收返回的记忆对象。第二步 `self._evict_if_needed()` 做容量裁剪——注意调用时没有使用返回值，因为该方法本身返回 `None`，其效果体现在存储状态的变化上。第三步 `return item` 把父类返回的对象原样交还给调用方。
- **异常/边界**：本方法自身不抛新异常；父类 `add` 抛出的异常（如后端不可用、参数非法）会原样向上传播，且因为异常发生在 `_evict_if_needed` 之前，此时不会触发淘汰。若 `_evict_if_needed` 内部的 `list()` 或 `delete()` 抛异常，也会向上传播，此时条目已写入但可能未完成裁剪。边界情况：`content` 为 `None` 或非字符串时本方法不做拦截；写入导致条目数恰好等于容量时不触发删除。
- **同文件关系**：调用父类 `BaseMemory.add`（跨文件）以及本文件内的私有方法 `_evict_if_needed`；它自身不被本文件中的其它方法调用，由外部上层代码调用。

### `_evict_if_needed(self) -> None` （第 24 行）
- **作用**：这是工作记忆容量淘汰策略的唯一实现处，也是本文件的核心业务逻辑。它在每次写入之后被调用，检查当前存储的条目总数是否超过 `self.capacity`，若超过则挑选出「最不值得保留」的若干条并删除，直到数量回落到容量以内。挑选规则是「重要性优先，其次时间」：先比 `importance`，重要性数值越小越先被淘汰；当重要性相同时，再比 `updated_at`，更新（修改）时间越早的越先被淘汰，也就是让陈旧且不重要的内容先出局，从而让工作记忆窗口里留下的是既重要又新鲜的信息。之所以要这样设计，是因为工作记忆面向的是「当前上下文」，容量有限，必须有确定性的、可解释的淘汰顺序，而不能随机丢弃。它只被 `add` 调用，属于纯内部实现细节，因此以单下划线开头命名。
- **参数**：
  - `self`：实例本身，通过它读取 `self.capacity` 并调用 `self.list()`、`self.delete()`。
- **返回**：返回 `None`。它不返回被删除的条目列表，也不返回删除数量；调用方只能通过存储状态的变化感知结果（例如之后调用 `list()` 观察条目数）。
- **内部流程**：第一步 `items = self.list()` 从父类提供的接口取回当前全部记忆条目（是一个可迭代的集合/列表），这一步同时确定了「当前条目数」。第二步 `if len(items) <= self.capacity: return` 做早退判断——未超容就直接返回，不做任何排序与删除，因此正常写入场景下这个方法的开销只是一次 `list()` 加一次长度比较。第三步在超容时构造淘汰名单：`victims = sorted(items, key=lambda item: (item.importance, item.updated_at))[: len(items) - self.capacity]`。这里的 `key` 是一个 lambda，返回二元组 `(重要性, 更新时间)`，Python 的元组比较是字典序，于是先按重要性升序、再按更新时间升序排列；排序后切片取前 `len(items) - self.capacity` 个，也就是恰好超出容量的那几条，作为待删除名单。第四步 `for item in victims: self.delete(item.id)` 逐个按 id 调用父类的删除方法，把受害者真正移除。循环结束后条目数即等于 `capacity`。
- **异常/边界**：本方法自身不抛异常，也不做任何异常捕获；`self.list()` 或 `self.delete()` 抛出的异常（如后端故障）会直接向上传播给 `add` 的调用方。边界情况：条目数等于容量时早退，不做任何操作；`items` 为空列表时 `len(items) <= capacity` 成立（容量至少为 1），同样早退，因此不会出现对空列表排序的问题；如果 `item.importance` 或 `item.updated_at` 缺失或类型不可比较，`sorted` 会在比较时抛 `TypeError`，本方法不处理，这一点依赖父类 `MemoryItem` 始终提供这两个可比较字段；若 `list()` 返回的集合中包含重复对象，切片数量按元素个数计算，删除时按 id 去重效果由 `delete` 决定。
- **同文件关系**：调用父类的 `list()` 与 `delete()`（跨文件）；被本文件中的 `add` 方法调用；不调用本文件中的其它方法。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `WorkingMemory` | 继承 `BaseMemory` 的工作记忆类，声明 `memory_type = MemoryType.WORKING`，在父类通用存储能力之上叠加容量上限与超限淘汰策略。 |
| `__init__` | 透传 `memory_type` 与 `kwargs` 给父类，并解析 `capacity`（默认取配置 `working_memory_capacity`），校验其必须是正整数，否则抛 `ValueError`。 |
| `add` | 覆写父类写入方法：先由父类写入并拿到 `MemoryItem`，再触发一次容量淘汰，最后把该条目返回给调用方。 |
| `_evict_if_needed` | 取回全部条目，未超容则早退；超容时按 `(importance, updated_at)` 升序选出超出容量的条目并逐个按 id 删除。 |
