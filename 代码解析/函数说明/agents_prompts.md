# agents/prompts.py

## 一、这个文件是干什么的

这个文件是整个 Agent 运行时的「系统提示词仓库」，它把过去写成一大坨字符串的行为规范拆成**按功能分节的具名文本块**，再用一个组合函数拼成最终交给大模型的 system prompt。它属于纯数据 + 一个组装函数的模块：不连接数据库、不调用任何工具、不依赖项目内其它模块（只 `from __future__ import annotations`），因此可以被任何层安全导入，也不会产生循环依赖。

模块里主要包含这些内容：身份总则 `IDENTITY`；检索纪律 `RETRIEVAL_DISCIPLINE`（先检索后回答、四层记忆各司其职、降级必须说明）；时间纪律 `TIME_DISCIPLINE`（相对时间先取真实时间）；写入纪律 `WRITE_DISCIPLINE`（写什么用哪个工具、人工确认边界、不重复写）；一致性治理纪律 `CONSISTENCY_DISCIPLINE`（对账、自愈与删除的人工边界）；多模态纪律 `MULTIMODAL_DISCIPLINE`（图片入库与诚实降级）；诚实与引用 `HONESTY_RULES`；以及模式专门化 `MODE_ONLINE` / `MODE_OFFLINE`。此外还有一个有序的节列表常量 `PROMPT_SECTIONS`，以及一个组装函数 `build_system_prompt`，最后用 `__all__` 显式声明对外导出面。

它在运行中的用法是：Web 层（`web/app.py` 的聊天端点）在构造请求时调用 `build_system_prompt()` 拿到整段系统提示词，交给共享的 agent 单例使用。文件顶部注释特别说明了一个设计取舍：聊天端点是「共享 agent 单例 + 串行锁」，如果按每个请求去改 agent 的提示词状态会污染并发上下文，所以这里把联网与离线两套模式规则**同时写进提示词**，并让模型自己以「本次工具清单里有没有 web.search」为判据来选择行为——同一份提示词对两种模式都成立，且线程安全。因此 `build_system_prompt` 的 `include_modes` 参数通常保持默认 `True`，只有在明确不需要模式段（例如离线调试、单元测试断言某个子串）时才会传 `False`。

还有两条值得注意的约束：其一，分节顺序就是渲染顺序（总则 → 检索 → 时间 → 写入 → 治理 → 多模态 → 诚实 → 模式），改 `PROMPT_SECTIONS` 的元组顺序就等于改最终提示词的段落顺序；其二，提示词里出现的工具名（如 `knowledge.hybrid_recall`、`memory.rag_search`、`web.search`、`system.current_time`）有测试守护（`tests/test_prompts.py`），凡是提到的工具名都必须真的注册在案，改名时测试会失败，用来避免「提示词描述的工具」与「实际注册的工具」发生漂移。

本文件中**没有定义任何类**，也没有任何方法、嵌套函数或 `__init__` 之类的 dunder 方法；可执行的代码单元只有模块级常量赋值、`PROMPT_SECTIONS` 元组、一个函数 `build_system_prompt`，以及 `__all__` 列表。模块级常量共 9 个提示词文本块：`IDENTITY`、`RETRIEVAL_DISCIPLINE`、`TIME_DISCIPLINE`、`WRITE_DISCIPLINE`、`CONSISTENCY_DISCIPLINE`、`MULTIMODAL_DISCIPLINE`、`HONESTY_RULES`、`MODE_ONLINE`、`MODE_OFFLINE`；它们都是 `str` 字面量拼接的结果，全部用 `【】` 标注小节标题，行内用 `\n` + 序号排版，以便模型按条遵循。

## 二、函数与类逐条详解

### `build_system_prompt(*, include_modes: bool = True) -> str` （第 125 行）

- **作用**：把本模块里所有具名的提示词文本块按 `PROMPT_SECTIONS` 声明的顺序拼成一段完整的系统提示词并返回，是外部代码访问本模块内容的唯一函数入口。之所以需要它，是因为提示词被刻意拆成了 9 个独立小节，调用方不应该自己去引用 `IDENTITY`、`RETRIEVAL_DISCIPLINE` 等常量再手工拼接，否则一旦新增小节或调整顺序，所有调用点都要跟着改；有了这个函数，顺序与分隔规则只在一个地方维护。它的 `include_modes` 参数提供了一种「裁剪」能力：默认会把联网与离线两段模式规则一起带上（这是 Web 端点的常规用法，保证共享 agent 单例在不同请求间不用改状态），传 `False` 时可以产出一份不含模式段的提示词，适合离线调试、评测或只关心基础行为规范的场景。它不会访问网络、不会读文件、没有副作用，属于纯函数，因此可以安全地在并发请求中被反复调用。
- **参数**：
  - `include_modes`：`bool`，默认 `True`。这是**仅限关键字参数**（签名里的 `*` 使其不能按位置传参，必须写成 `build_system_prompt(include_modes=False)`）。取值只有两种：`True` 表示把 `PROMPT_SECTIONS` 里所有小节都渲染出来，包含 `MODE_ONLINE` 与 `MODE_OFFLINE`；`False` 表示过滤掉所有名字以 `mode_` 开头的节，即只保留身份、检索、时间、写入、治理、多模态、诚实这 7 段。传非布尔值（如 `None`、`0`、字符串）时 Python 会按真值语义处理，`0`/`None`/空字符串等价于 `False`，其余等价于 `True`，函数本身不做类型校验。
- **返回**：返回一个 `str`，即拼接好的完整系统提示词。当 `include_modes=True`（默认）时返回全部 9 段，用两个换行符 `"\n\n"` 作为段间分隔，因此每段之间会有一个空行；当 `include_modes=False` 时返回其中 7 段，同样以 `"\n\n"` 分隔。函数不会返回 `None`，也不会返回空字符串——因为 `PROMPT_SECTIONS` 是模块级常量且至少包含 `identity` 与 `honesty` 等非 `mode_` 开头的节，即便在 `include_modes=False` 的最坏情况下也一定至少包含这些基础段落。
- **内部流程**：第一步，用一个列表推导式遍历模块级常量 `PROMPT_SECTIONS`，该常量是 `tuple[tuple[str, str], ...]`，每一项形如 `("identity", IDENTITY)`，即「节名 + 节文本」二元组；循环时把元组解包为 `name` 与 `text`。第二步，对每一项应用过滤条件 `include_modes or not name.startswith("mode_")`：当 `include_modes` 为真时短路求值直接通过，所有节都保留；当 `include_modes` 为假时继续判断节名是否以 `"mode_"` 开头，只有不以 `mode_` 开头的节才被保留，于是 `("mode_online", MODE_ONLINE)` 与 `("mode_offline", MODE_OFFLINE)` 被剔除。第三步，收集到的 `text` 组成列表 `blocks`。第四步，用 `"\n\n".join(blocks)` 把各段文本用一个空行连接成单一字符串并返回。整个过程没有任何输入输出、没有状态修改、没有随机性。
- **异常/边界**：正常情况下不会抛异常。`include_modes` 传非布尔值不会报错，按 Python 真值语义处理；传 `None` 时等价于 `False`，会得到不含模式段的提示词。由于 `PROMPT_SECTIONS` 是模块级硬编码常量，不存在空集合导致的异常风险；唯一会抛出异常的情形是有人在本模块内把 `PROMPT_SECTIONS` 改成非可迭代对象或元素不是二元组（那会在遍历/解包时抛 `TypeError` 或 `ValueError`），但这属于修改源码而非调用者的边界情况。函数不做任何缓存，重复调用会重复拼接，代价极小。
- **同文件关系**：它读取本文件中的模块级常量 `PROMPT_SECTIONS`，并间接使用被该元组引用的 9 个提示词文本常量 `IDENTITY`、`RETRIEVAL_DISCIPLINE`、`TIME_DISCIPLINE`、`WRITE_DISCIPLINE`、`CONSISTENCY_DISCIPLINE`、`MULTIMODAL_DISCIPLINE`、`HONESTY_RULES`、`MODE_ONLINE`、`MODE_OFFLINE`（以及元组里的节名字符串字面量）。本文件内没有其它函数或方法调用它——它是被外部模块（如 `web/app.py` 组装请求时）调用的导出接口，同时被 `__all__` 声明为公开成员。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `build_system_prompt(*, include_modes: bool = True) -> str` | 按 `PROMPT_SECTIONS` 的顺序把所有提示词分节拼成完整系统提示词，可选择是否包含联网/离线两段模式规则。 |
