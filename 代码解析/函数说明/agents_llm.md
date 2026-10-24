# agents/llm.py

## 一、这个文件是干什么的

这个文件是项目里**唯一直接与 OpenAI 兼容网关对话的底层客户端封装**。它把「一个已经解析好的模型配置（api_key / base_url / model / 重试次数）」变成一个可复用的 `LLM` 对象，并对外暴露三种调用形态：非流式补全（`complete`）、流式补全并重建响应信封（`complete_streaming`）、以及只要纯文本结果的便捷入口（`think`）。之所以要单独抽出这一层，是因为 Agent 的工具调用循环必须使用流式请求：如果走非流式，网关在生成完整回答之前一个字节都不会发送，读超时会在网关还没写完时就被触发，OpenAI SDK 随后会在后台重试同一个请求，造成重复生成和超时雪崩；改成流式后，同一个超时值衡量的是「两个 token 之间的间隔」而不是「整段生成的总时长」。

文件里主要包含四块东西：一是模块级的小工具函数 `_default_echo_write`，负责把文本写到标准输出；二是类 `_FinalAnswerEchoer`，它是一个「终端回声控制器」，在流式过程中决定什么时候把内容打印到用户终端；三是函数 `_assemble_streaming_response`，它把 OpenAI 流式分块（chunk）重新拼装成一个与非流式响应结构完全一致的字典，从而让上层的消息提取代码不需要分支处理两种形态；四是核心类 `LLM`，它构造真实的 `openai.OpenAI` 客户端（并挂上显式配置了代理的 `httpx.Client`），并提供 `complete`、`complete_streaming`、`think`、`stream_response`、`_message_content` 等方法。

在项目运行中，它被 ReAct 风格的文本协议（会有 `final answer`、`最终答案` 这类行首标记）和原生 tool_calls 两种模式共同使用：`echo_mode="react_final"` 时回声器会静默缓冲，直到看见最终答案标记才把标记之后的内容打到终端，避免把中间思考过程喷到屏幕上；`echo_mode="content"` 时则原样实时转发每个增量。拼装函数还会在返回的字典里塞一个 `stream_echoed` 布尔字段，调用方靠它判断「这次回答一个字都没进终端」，从而决定要不要做一次性兜底打印。此外，`complete` 与 `complete_streaming` 都支持转发 OpenAI 的提示缓存路由字段（`prompt_cache_key`、`prompt_cache_retention`），让多轮对话能命中同一个缓存键，同时在字段缺省时依然兼容较老的第三方兼容网关。

## 二、函数与类逐条详解

### `_default_echo_write(text: str) -> None` （第 29 行）

- **作用**：这是整个文件里默认的终端输出实现。它把传入的字符串直接写到 `sys.stdout`，并立刻调用 `flush()` 强制刷新缓冲区。之所以需要它，是因为回声器 `_FinalAnswerEchoer` 允许调用方传入自定义的写函数（比如写进日志、写进 WebSocket、写进某个 UI 组件），而当调用方没有提供写函数时，就需要一个「写到当前进程标准输出」的兜底实现。它被用在 `_FinalAnswerEchoer.__init__` 里作为 `write` 参数为空时的替代品，因此任何使用流式回声但没有自定义输出通道的场景都会走到它。它不返回任何东西，只产生副作用。由于它在每次写入后都刷新，所以即使回答是逐 token 到达的，用户也能立刻看到，而不是等程序结束才一次性出现。
- **参数**：`text`（`str`）：要写到标准输出的文本片段。可以是单个字符、一个词、一整段，甚至包含换行符的字符串；没有默认值，调用方必须显式传入。该函数不做任何类型检查，如果传入非字符串对象，`sys.stdout.write` 会自行抛出 `TypeError`。
- **返回**：返回 `None`。它的价值完全体现在副作用（把文本打印到终端）上，调用方无法从返回值得到任何信息。
- **内部流程**：第一步调用 `sys.stdout.write(text)` 把文本写入标准输出流；第二步调用 `sys.stdout.flush()` 立即清空缓冲区，保证内容即时可见。整个函数只有这两行，没有分支、没有循环、没有异常捕获。
- **异常/边界**：无特殊处理。它自己不捕获任何异常，所以如果标准输出已被关闭、被重定向到已断开的管道、或者当前环境没有可用的 stdout，`write` 或 `flush` 抛出的 `ValueError` / `OSError` 会向上传播。不过在实际调用路径上，`_FinalAnswerEchoer._emit` 用 `try/except Exception` 包住了对写函数的调用并只记 debug 日志，因此这些异常不会中断流式消费。传入空字符串时会正常执行两次调用，不报错，但也不会输出任何可见内容。
- **同文件关系**：被 `_FinalAnswerEchoer.__init__` 引用（作为 `write` 参数为 `None` 时的默认值）。它不调用本文件里的任何其他函数。

### `class _FinalAnswerEchoer` （第 34 行）

- **作用**：这是一个「一次流式补全对应的终端回声控制器」。它的存在解决了一个具体的体验问题：在 ReAct 文本协议下，模型会在同一个流里先吐出一大段思考过程、工具调用意图，最后才给出真正给用户看的最终答案；如果把流里的每个增量都直接打到终端，用户会看到满屏的内部推理噪声。这个类因此提供两种模式：`content` 模式下它把每个增量立即转发出去（原生 tool_calls 轮次通常不携带 content，所以实际上只会流出答案文本）；`react_final` 模式下它先静默缓冲，直到缓冲区里出现行首的最终答案标记（`final answer`、`最终答案`、`最终回答`、`最终回复` 后面跟冒号），才切换到「实时」状态并只把标记之后的内容打出去；如果这一轮以工具调用结束、始终没有出现标记，那么它一个字都不会输出。类实例还维护 `echoed` 标志，供上层判断是否需要对「从未进入终端的回答」做兜底打印。
- **同文件关系**：被 `_assemble_streaming_response` 实例化并驱动（调用 `feed` 与 `flush`）；内部使用模块级常量 `_FINAL_ANSWER_MARKER_RE` 做匹配、使用 `_default_echo_write` 作为默认写函数、使用模块级类型别名 `EchoMode` 约束模式取值、使用模块级 `LOGGER` 记录写失败。

#### `__init__(self, mode: EchoMode, write: Callable[[str], None] | None, prefix: str = "\nAI：") -> None` （第 44 行）

- **作用**：构造回声控制器并确定它的工作模式与输出通道。它把调用方给的 `mode` 翻译成一个布尔开关 `_live`：只有当 `mode == "content"` 时初始就是「实时」状态，其他情况（包括 `react_final`）都必须先缓冲、等标记。同时它决定「写函数」是使用外部传入的自定义实现还是退回到 `_default_echo_write`，并把前缀字符串存起来，供第一次真正输出时打印一次「AI：」这样的提示。这个构造函数在每次流式补全开始时被调用一次（由 `_assemble_streaming_response` 在 `echo_mode` 非空时创建），因此每个流都有独立的缓冲区和独立的 `echoed` 状态，不会互相串扰。
- **参数**：`mode`（`EchoMode`，即 `Literal["react_final", "content"]`）：回声模式，没有默认值。传 `"content"` 表示立刻实时转发每个增量；传 `"react_final"` 表示先缓冲、等最终答案标记出现后再转发。参数没有运行时校验，如果传入其他字符串，行为等价于「非 content」，也就是会走缓冲分支，但由于 `_live` 只在等于 `"content"` 时为真，这种非法值不会崩溃，只是永远等不到标记而已。`write`（`Callable[[str], None] | None`）：自定义的文本写入回调，没有默认值（必须显式传，允许传 `None`）。传 `None` 时内部会替换为 `_default_echo_write`，也就是打印到标准输出。`prefix`（`str`，默认 `"\nAI："`）：在第一次真正输出内容之前先写出的提示前缀，默认值以一个换行开头，保证答案从新的一行开始并以「AI：」引导。传空字符串表示不要任何前缀。
- **返回**：构造函数返回 `None`（Python 构造函数的惯例）。它创建的实例被隐式返回给调用方。
- **内部流程**：第一步执行 `self._write = write or _default_echo_write`，把 `None` 或假值替换成默认写函数；第二步保存 `self._prefix = prefix`；第三步把内部缓冲区 `self._buffer` 初始化为空字符串；第四步根据 `mode == "content"` 计算并保存 `self._live`；第五步把公开标志 `self.echoed` 初始化为 `False`，表示「目前还没有任何内容被真正写到终端」。整个过程没有校验、没有循环、没有异常捕获。
- **异常/边界**：无特殊处理。构造函数不校验 `mode` 的取值是否合法，也不校验 `write` 是否真的可调用；如果传入了不可调用的对象（例如字符串），错误会在后续 `_emit` 调用 `self._write(...)` 时才出现，并且会被 `_emit` 内部的 `try/except Exception` 吞掉，只留一条 debug 日志。`prefix` 传入 `None` 也不会在这里报错，而是在 `_emit` 的 `if not self.echoed and self._prefix` 判断中被当作假值跳过。
- **同文件关系**：调用 `_default_echo_write`（作为 `write` 缺省值）。被 `_assemble_streaming_response` 调用。

#### `feed(self, piece: str) -> None` （第 56 行）

- **作用**：把流式响应里的一个内容增量喂给回声器，由回声器决定是立刻输出还是先攒着。这是整个回声机制的核心状态机：如果实例已经处于实时状态（`content` 模式，或 `react_final` 模式下此前已经命中过标记），它直接把这一片交给 `_emit` 输出；否则先把这一片追加到内部缓冲区，然后用正则在整个缓冲区里搜索行首的最终答案标记。一旦找到标记，它就把缓冲区清空、把状态翻转为实时，并且只把标记**之后**的那部分（`tail`）输出出去——标记本身不会被打印，从而用户看不到「最终答案：」这几个字，只看到答案正文。这个方法在 `_assemble_streaming_response` 的流循环里，每收到一个非空 `delta.content` 就被调用一次，调用顺序与 token 到达顺序严格一致。
- **参数**：`piece`（`str`）：本次收到的内容增量片段。在 OpenAI 流式协议里它通常是一个或几个 token 的文本，也可能是空字符串。函数开头用 `if not piece: return` 做了空值短路，因此传空字符串（或 `None`、`0` 这类假值）都会被直接忽略，不会污染缓冲区。
- **返回**：返回 `None`。所有效果都通过内部状态变更（`self._buffer`、`self._live`、`self.echoed`）和写函数调用的副作用体现。
- **内部流程**：第一步，判断 `piece` 是否为空，为空则直接返回；第二步，判断 `self._live` 是否为真，为真则调用 `self._emit(piece)` 并返回（这条路径下缓冲区永远不参与）；第三步，把 `piece` 追加到 `self._buffer`；第四步，用预编译正则 `_FINAL_ANSWER_MARKER_RE.search(self._buffer)` 在缓冲区里查找标记，并检查匹配结果是否为 `None`；第五步，如果匹配到了，取出 `tail = self._buffer[match.end():]`（标记结束位置到缓冲区末尾的部分），把 `self._buffer` 重置为空字符串，把 `self._live` 置为 `True`，最后在 `tail` 非空时调用 `self._emit(tail)` 输出答案正文。注意正则使用了 `(?im)` 标志，即大小写不敏感加多行模式，所以标记必须出现在某一行的行首（可以带 `**` 加粗包裹）才会被识别。
- **异常/边界**：无特殊处理，函数自身不捕获异常。空字符串会被静默忽略。如果标记恰好被拆分到两个增量之间（例如前一片以「最终答」结尾、后一片以「案：」开头），因为每次都是在**累积后的整个缓冲区**上做搜索，所以跨分片的标记依然能正确识别，只是要等到标记的后半部分到达时才会触发。如果整个流里始终没有出现标记，缓冲区会一直增长到流结束（内容不会被丢弃，但也不会输出），并在 `flush` 阶段被忽略——`flush` 只负责收尾换行，不会把未输出的缓冲区内容补打出来。
- **同文件关系**：调用 `_emit`（同类方法）；使用模块级 `_FINAL_ANSWER_MARKER_RE`。被 `_assemble_streaming_response` 调用。

#### `flush(self) -> None` （第 71 行）

- **作用**：在一条流式补全消费完毕时做收尾：如果此前确实往终端写过东西，就补一个换行符，让后续的输出（例如下一轮提示符或下一条日志）从新的一行开始，避免和答案正文黏在同一行。它的文档字符串明确写了「Close the echo line once anything was written to the terminal」，也就是它的职责边界仅限于「收尾换行」，不做任何内容补打或状态重置。它在 `_assemble_streaming_response` 的 `finally` 块里被调用，因此无论流是正常结束还是中途抛异常，都会执行到。
- **参数**：无参数（除隐式的 `self`）。
- **返回**：返回 `None`。没有可观察的返回值，效果只是可能写出一个 `"\n"`。
- **内部流程**：只有一步判断：检查 `self.echoed` 是否为真，为真则调用 `self._write("\n")` 写出一个换行。如果整条流从未输出过任何内容（`react_final` 模式下没有出现最终答案标记，或流里根本没有 content），`self.echoed` 仍为 `False`，此时函数什么都不做，不会凭空打印一个空行。
- **异常/边界**：无特殊处理，没有 `try/except`。与 `_emit` 不同，这里的 `self._write("\n")` 没有被异常保护，所以如果自定义写函数在此刻抛出异常，异常会向上传播到 `_assemble_streaming_response` 的 `finally` 块中——注意该 `finally` 块本身没有再包一层捕获，因此这个异常会从 `_assemble_streaming_response` 抛出，可能掩盖原本正在传播的流异常。这是本文件里少数几处「写函数异常不会被吞掉」的路径。重复调用 `flush` 是安全的：第一次调用后 `echoed` 仍为 `True`，所以第二次会再写一个换行（也就是说它不是幂等的，多次调用会多出空行）。
- **同文件关系**：调用 `self._write`（即 `_default_echo_write` 或外部传入的写函数）。被 `_assemble_streaming_response` 在 `finally` 块中调用。

#### `_emit(self, text: str) -> None` （第 77 行）

- **作用**：这是回声器唯一真正接触输出通道的私有方法，负责「写出前缀（只写一次）+ 写出正文」这两件事。它把前缀逻辑和正文逻辑集中在一处，使得 `feed` 的两条路径（实时转发与命中标记后转发）都只需要调用它一次即可。它还承担了「首次输出时才打印 `AI：` 前缀」的判断：只要 `self.echoed` 已经是 `True`，说明前面已经打过前缀了，就不再重复打，因此多轮增量拼接出来的答案只会有一个前缀。它同时把「写失败」的风险隔离掉，保证终端写异常不会破坏流式消费流程。
- **参数**：`text`（`str`）：本次要写出的文本正文，可能是标记之后的残余内容，也可能是实时模式下的一个增量。函数开头用 `if not text: return` 短路空值。没有默认值。
- **返回**：返回 `None`。所有效果都体现在对写函数的调用以及 `self.echoed` 的状态变化上。
- **内部流程**：第一步，判断 `text` 是否为空，为空直接返回；第二步，判断是否「尚未输出过内容且前缀非空」（`not self.echoed and self._prefix`），成立则在一个 `try` 块里调用 `self._write(self._prefix)` 写出前缀，若抛异常则用 `LOGGER.debug("stream echo write failed", exc_info=True)` 记一条调试日志并继续；第三步，无条件把 `self.echoed = True`（注意这一步在 `try` 之外，所以即使前缀写失败，也会被认为「已经输出过」）；第四步，再在一个 `try` 块里调用 `self._write(text)` 写出正文，同样捕获所有异常并记 debug 日志。
- **异常/边界**：不向调用方抛异常——两个 `try/except Exception` 把写通道的所有失败都吞掉并降级为 debug 日志，这是刻意的设计，目的是「终端写失败绝不能中断模型调用」。空字符串会被静默忽略且**不会**把 `echoed` 置为 `True`（因为它在第一行就返回了），所以如果前缀也没写成功，`echoed` 会保持 `False`。`self._prefix` 为 `None` 或空串时，前缀分支被跳过，但正文仍会照常输出。`LOGGER.debug(..., exc_info=True)` 只有在日志级别为 DEBUG 时才会真正格式化异常栈，默认级别下开销极小。
- **同文件关系**：被同类方法 `feed`（两处：实时分支与命中标记分支）调用。它自身调用 `self._write`（即 `_default_echo_write` 或外部写函数）并写 `LOGGER`。

### `_assemble_streaming_response(stream: Iterable[Any], *, on_first_chunk: Callable[[], None] | None = None, echo_mode: EchoMode | None = None, echo_write: Callable[[str], None] | None = None) -> dict[str, Any]` （第 92 行）

- **作用**：这是流式链路的「重组装器」。OpenAI 的流式接口把一次回答拆成很多个 chunk 依次送达：正文以 `delta.content` 的碎片形式出现，而工具调用则以 `delta.tool_calls` 的形式出现，每个片段只带 `index`、可选的 `id`、可选的 `function.name` 和 `function.arguments` 的一部分，需要按 `index` 归组、把字符串逐段拼接才能还原成完整的工具调用对象。这个函数遍历整个流，把这些碎片累积成 `content_parts` 与按索引归并的 `tool_calls`，最后构造出一个结构与**非流式响应完全一致**的字典（`choices[0].message.role/content/tool_calls` 加 `choices[0].finish_reason`），使得上层那些按非流式结构写的消息提取代码不需要为流式额外分支。它同时承担两个附带职责：在收到第一个有效 chunk 时触发一次 `on_first_chunk` 回调（调用方常用它来取消「等待中」的提示或记录首 token 延迟），以及在配置了 `echo_mode` 时驱动 `_FinalAnswerEchoer` 把答案回显到终端。
- **参数**：`stream`（`Iterable[Any]`，位置参数）：要消费的流式对象，通常是 OpenAI SDK 返回的 `Stream`，也可能是任何可迭代的 chunk 序列（包括字典形式的分块，因为内部统一用 `field()` 做兼容访问）。没有默认值。`on_first_chunk`（`Callable[[], None] | None`，仅关键字，默认 `None`）：在第一个「有 choices 的 chunk」被处理时调用一次的无参回调；传 `None` 表示不需要通知。`echo_mode`（`EchoMode | None`，仅关键字，默认 `None`）：回声模式，传 `None` 表示完全不做终端回显（此时也不会创建回声器）；传 `"content"` 或 `"react_final"` 则创建一个对应的 `_FinalAnswerEchoer`。`echo_write`（`Callable[[str], None] | None`，仅关键字，默认 `None`）：回声使用的写函数，仅在 `echo_mode` 非 `None` 时有意义；传 `None` 时回声器内部会退回标准输出。
- **返回**：返回一个 `dict[str, Any]`，结构为 `{"choices": [{"message": {...}, "finish_reason": ...}]}`。其中 `message` 固定包含 `"role": "assistant"` 与 `"content"`（把所有内容碎片拼接后得到；如果整条流没有任何内容碎片，则值为 `None` 而不是空字符串）；只有在确实收集到工具调用时才会额外包含 `"tool_calls"` 键，其值是按 `index` 升序排序后的工具调用字典列表。`finish_reason` 取自最后一个非 `None` 的 `choice.finish_reason`，如果整条流都没有给出该字段，则为 `None`。此外，当 `echo_mode` 非 `None` 时，返回字典还会带一个顶层键 `"stream_echoed"`，其布尔值等于回声器的 `echoed` 标志——调用方用它检测「这次回答一个字都没进终端」，从而决定是否需要一次性兜底打印。
- **内部流程**：第一步，初始化累加器：`content_parts`（内容碎片列表）、`tool_calls`（`int -> dict` 的索引映射）、`finish_reason`（`None`）、`first_chunk_seen`（`False`），并在 `echo_mode` 非空时构造 `_FinalAnswerEchoer`。第二步，进入 `try` 块并用 `for chunk in stream` 逐个消费分块；对每个分块先用 `field(chunk, "choices")` 取 choices，取不到就 `continue` 跳过。第三步，只处理 `choices[0]`；如果是第一次看到有效分块，就把 `first_chunk_seen` 置为 `True` 并在 `try/except` 中调用 `on_first_chunk`（回调抛异常只记 debug 日志，不影响主流程）。第四步，用海象运算符读取 `choice.finish_reason`，非 `None` 就更新 `finish_reason`（因此后面的值会覆盖前面的）。第五步，取 `choice.delta`，为 `None` 就跳过该分块。第六步，取 `delta.content`，非空则追加到 `content_parts`，并在存在回声器时调用 `echoer.feed(piece)`。第七步，取 `delta.tool_calls` 片段列表，为空就进入下一轮循环；否则对每个片段读取 `index`（缺失或为 0 时用 `or 0` 归一到 0），用 `setdefault` 保证该索引对应一个初始结构 `{"id": "", "type": "function", "function": {"name": "", "arguments": ""}}`。第八步，如果片段带 `id`，则「首次直接赋值、后续追加拼接」地累积到 `entry["id"]`；如果片段带 `function`，则把 `function.name` 与 `function.arguments` 分别累加到对应字段上。第九步，循环结束后进入 `finally` 块：先调用 `echoer.flush()` 收尾换行，再用 `getattr(stream, "close", None)` 拿到可选的 `close` 方法并在 `try/except` 中调用它关闭底层 HTTP 连接（失败只记 debug 日志）。第十步，在 `finally` 之后构造 `message` 字典（`content` 用 `"".join(content_parts) or None`），有工具调用时按排序后的索引列表挂上 `tool_calls`，再包成 `choices` 结构返回，并在有回声器时补上 `stream_echoed`。
- **异常/边界**：`on_first_chunk` 的异常、`stream.close()` 的异常都被吞掉并降级为 debug 日志；`echoer.flush()` 的异常**不会**被捕获（见 `flush` 条目）。迭代 `stream` 本身抛出的异常（例如网络中断、SDK 报错）不会被捕获，会向上传播——但 `finally` 保证回声器已收尾、流已尝试关闭，所以不会泄漏连接。空流（一个 chunk 都没有）会返回 `content: None`、`finish_reason: None`、无 `tool_calls` 键的合法结构。chunk 的 `choices` 为空列表时被跳过，不会触发 `on_first_chunk`（回调只在真正有 choices 的第一个分块上触发）。`delta` 为 `None`、`content` 为空串、`tool_calls` 为空列表都会走 `continue`/短路分支，不做无谓处理。`index` 缺失时统一归到 0，所以多个都没带 `index` 的片段会被合并进同一个工具调用，这是兼容降级行为。返回的 `content` 使用 `or None` 而不是空字符串，这是刻意的：非流式响应里「只有工具调用、没有正文」时 content 也是 `None`，保持两者形状一致。
- **同文件关系**：调用 `_FinalAnswerEchoer`（构造、`feed`、`flush`）与 `field`（来自同包的 `agents/message_utils`）。被 `LLM.complete_streaming` 调用。

### `class LLM` （第 191 行）

- **作用**：这是本项目对「一个已解析模型配置」的 OpenAI 兼容客户端封装。一个 `LLM` 实例固定绑定一组 api_key / base_url / model / max_retries，内部持有一个真实的 `openai.OpenAI` 客户端对象，因此上层拿到实例后就可以反复发起补全请求而不用重复读取配置。它刻意保持「薄」：不做提示词拼装、不做工具调用循环、不做记忆管理，这些都属于上层；它只负责把参数校验好、把流式参数设对、把提示缓存字段按需转发，并保证老旧的第三方兼容网关在缺省这些新字段时依然可用。类里同时提供流式与非流式两条路径，其中 `complete_streaming` 是工具循环的推荐入口，`complete` 是通用底层入口，`think` 是「我只要一段文本」的便捷包装。
- **同文件关系**：内部使用 `_assemble_streaming_response` 完成流式拼装，使用 `field` 读取分块，使用常量 `DEFAULT_MAX_RETRIES` / `DEFAULT_TEMPERATURE` / `DEFAULT_TIMEOUT`。被上层 Agent 运行时调用（在本文件之外）。

#### `__init__(self, *, api_key: str, base_url: str, model: str, max_retries: int = DEFAULT_MAX_RETRIES) -> None` （第 199 行）

- **作用**：构造一个 LLM 客户端实例。它先做一轮严格的类型与取值校验（重试次数必须是非负整数、三个字符串参数必须是字符串），再把它们 `strip()` 去掉首尾空白，然后检查去掉空白后是否有空值，只要有空就抛出带具体字段名的配置错误——这样问题在启动/构造阶段就暴露，而不是等到第一次发请求时才以一个难懂的 401 或 URL 错误出现。校验通过后，它延迟导入 `httpx`、`openai.OpenAI` 和项目的 `core.services_config`，从服务配置里读取代理地址，构造一个**显式设置了代理且关闭环境变量探测**（`trust_env=False`）的 `httpx.Client`，再把这个 client 交给 `OpenAI` 使用。这样做的好处是：请求的出口完全由项目自己的配置决定，不会被运行环境里残留的 `HTTP_PROXY` 之类的变量意外改写。
- **参数**：`api_key`（`str`，仅关键字，必填）：网关的鉴权密钥；必须是字符串，构造时会 `strip()`，去空白后不能为空。`base_url`（`str`，仅关键字，必填）：OpenAI 兼容网关的基础地址（例如 `https://.../v1`）；必须是字符串，`strip()` 后不能为空。`model`（`str`，仅关键字，必填）：默认模型名；必须是字符串，`strip()` 后不能为空。它会被保存为实例的默认模型，在后续 `complete` / `complete_streaming` 未显式传 `model` 时使用。`max_retries`（`int`，仅关键字，默认 `DEFAULT_MAX_RETRIES`）：SDK 层面的自动重试次数；必须是**非负整数**，且布尔值不被接受（`True`/`False` 会因为 `isinstance(max_retries, bool)` 判定而直接报错，避免把布尔误当整数 1/0）。
- **返回**：返回 `None`（构造函数的惯例），实例被隐式返回。
- **内部流程**：第一步，校验 `max_retries`：若是布尔值、或不是 `int`、或小于 0，抛出 `ValueError("max_retries must be a non-negative integer")`。第二步，依次校验 `api_key`、`base_url`、`model` 是否为 `str`，不是则抛 `TypeError`（三个字段各自的错误消息不同）。第三步，把三个字段分别 `strip()` 后存到 `self.api_key`、`self.base_url`、`self.model`，并把 `max_retries` 存到 `self.max_retries`。第四步，用一个列表推导筛出所有去空白后为空的字段名，若列表非空则抛出 `ValueError`，消息形如 `Configuration Error: api_key, model is not configured.`（字段名以英文逗号加空格连接）。第五步，进入 `try` 块，导入 `httpx`、`openai.OpenAI` 以及 `core.services_config.load_services_config`。第六步，调用 `load_services_config().proxy.url` 取得代理地址，并用 `httpx.Client(proxy=proxy or None, trust_env=False)` 构造 HTTP 客户端——代理为空时传 `None` 表示不使用代理，`trust_env=False` 表示忽略环境变量里的代理设置。第七步，用 `OpenAI(api_key=..., base_url=..., max_retries=..., http_client=...)` 构造真正的 SDK 客户端并存到 `self.client`。第八步，如果上述导入过程中抛出 `ImportError`，则用 `from exc` 链式抛出 `RuntimeError("The 'openai' package is required to use LLM.")`。
- **异常/边界**：`ValueError`：`max_retries` 为布尔/非整数/负数时；任一字符串字段去空白后为空时。`TypeError`：`api_key`、`base_url`、`model` 三者中有不是字符串的。`RuntimeError`：缺少 `openai` 包（只捕获 `ImportError`，因此缺少 `httpx` 或 `core.services_config` 时同样会被这个 `RuntimeError` 包装并抛出）。注意 `load_services_config()` 本身的读取失败、配置文件缺失、代理地址非法、`httpx.Client` 构造失败、`OpenAI` 构造失败（例如 base_url 格式非法）都不在捕获范围内，会原样向上抛出，不会被转成 `RuntimeError`。传入全空白的字符串会被判为「未配置」，而不是被当作有效值。
- **同文件关系**：无（不使用本文件里其他函数或类，只依赖模块级常量 `DEFAULT_MAX_RETRIES` 和外部库）。

#### `complete_streaming(self, messages: list[dict[str, Any]], *, model: str | None = None, temperature: float = DEFAULT_TEMPERATURE, timeout: float = DEFAULT_TIMEOUT, on_first_chunk: Callable[[], None] | None = None, echo_mode: EchoMode | None = None, echo_write: Callable[[str], None] | None = None, prompt_cache_key: str | None = None, prompt_cache_retention: str | None = None, **kwargs: Any) -> dict[str, Any]` （第 256 行）

- **作用**：这是工具调用循环应当使用的流式补全入口。它强制 `stream=True`，把参数组装好后调用 SDK 的 `chat.completions.create`，再把返回的流交给 `_assemble_streaming_response` 拼装成与非流式同形的字典返回。之所以必须有这个方法，文档字符串里说得非常明确：如果工具循环使用非流式请求，网关在生成完整回答之前不会发送任何字节，读超时会在网关还没写完时就触发，OpenAI SDK 随后会在后台重试同一个请求，造成重复生成；而流式请求从第一个 token 起就让连接保持活跃，同一个超时值因此变成衡量「token 之间间隔」的指标，而不是「整段生成总时长」的指标。它同时把首块回调、终端回声、提示缓存路由字段一路透传给底层。
- **参数**：`messages`（`list[dict[str, Any]]`，位置参数，必填）：完整的对话消息列表，直接作为 `messages` 传给网关，本方法不做任何加工。`model`（`str | None`，仅关键字，默认 `None`）：本次请求使用的模型名；传 `None` 时使用构造时保存的 `self.model`。`temperature`（`float`，仅关键字，默认 `DEFAULT_TEMPERATURE`）：采样温度，直接透传；注意本方法**不做**像 `complete` 那样的有限数校验。`timeout`（`float`，仅关键字，默认 `DEFAULT_TIMEOUT`）：读超时秒数，直接透传给 SDK；在流式模式下它衡量的是相邻 token 之间的最大等待时间。`on_first_chunk`（`Callable[[], None] | None`，仅关键字，默认 `None`）：第一个有效分块到达时调用一次的回调，交给 `_assemble_streaming_response` 触发；用于通知「已经开始有输出了」。`echo_mode`（`EchoMode | None`，仅关键字，默认 `None`）：终端回声模式，`None` 表示不回显，`"content"` 表示实时转发内容增量，`"react_final"` 表示等最终答案标记出现后再转发。`echo_write`（`Callable[[str], None] | None`，仅关键字，默认 `None`）：回声使用的自定义写函数，仅在 `echo_mode` 非 `None` 时有意义，`None` 时退回标准输出。`prompt_cache_key`（`str | None`，仅关键字，默认 `None`）：提示缓存路由键；**非 `None`** 时才会放进请求参数，从而在多轮之间保持稳定缓存键；为 `None` 时该字段整体省略，以兼容不支持它的老网关。注意本方法不校验其长度（校验在 `complete` 里做）。`prompt_cache_retention`（`str | None`，仅关键字，默认 `None`）：提示缓存保留策略；非 `None` 时放进请求参数，`None` 时省略。本方法同样不校验其取值。`**kwargs`（`Any`）：其余任意参数原样透传给 `chat.completions.create`，例如 `tools`、`tool_choice`、`max_tokens` 等。
- **返回**：返回 `dict[str, Any]`，即 `_assemble_streaming_response` 拼装出的响应字典：`{"choices": [{"message": {"role": "assistant", "content": str | None, 可能带 "tool_calls": [...]}, "finish_reason": str | None}], 可能带 "stream_echoed": bool}`。`stream_echoed` 只在传了 `echo_mode` 时出现。
- **内部流程**：第一步，用集合交集检查 `kwargs` 里是否混入了保留参数（`messages`、`model`、`stream`），有则抛出 `TypeError`，消息列出被非法覆盖的参数名，防止调用方用 `kwargs` 悄悄替换掉关键字段。第二步，构造 `options` 字典：`model` 取 `model or self.model`，`messages` 用传入值，`temperature` 用传入值，`stream` 固定为 `True`，`timeout` 用传入值。第三步，按需把 `prompt_cache_key` 与 `prompt_cache_retention` 加进 `options`（只有非 `None` 才加）。第四步，调用 `self.client.chat.completions.create(**options, **kwargs)` 得到流对象并赋给 `stream`。第五步，把流、`on_first_chunk`、`echo_mode`、`echo_write` 一起交给 `_assemble_streaming_response`，直接返回它的结果。
- **异常/边界**：`TypeError`：`kwargs` 里出现 `messages`、`model`、`stream` 中的任意一个。除此之外没有参数校验——`temperature` 传 `NaN`、`timeout` 传负数、`prompt_cache_key` 传超长字符串、`prompt_cache_retention` 传非法值都不会在这里被拦下，而是由 SDK 或网关报错（这是与 `complete` 的显著差异）。SDK 调用本身抛出的异常（鉴权失败、连接失败、超时、限流）不被捕获，原样向上传播；流消费过程中抛出的异常会穿过 `_assemble_streaming_response` 的 `finally` 继续向上传播，同时保证流被关闭。
- **同文件关系**：调用 `_assemble_streaming_response`。被 `LLM.think` 在流式分支下调用。

#### `think(self, messages: list[dict[str, Any]], temperature: float = DEFAULT_TEMPERATURE, timeout: float = DEFAULT_TIMEOUT, stream_response_bool: bool = True, prompt_cache_key: str | None = None, prompt_cache_retention: str | None = None, **kwargs: Any) -> str` （第 307 行）

- **作用**：这是一个「只要一段纯文本」的便捷包装。上层很多地方只关心模型说了什么，不关心 choices、finish_reason、tool_calls 这些结构，`think` 就把这些细节收进内部：它先调用 `complete` 拿到响应，如果 `stream_response_bool` 为真，则用 `stream_response` 把响应里的内容增量逐段取出并拼接成一个字符串返回；否则直接用 `_message_content` 从响应对象里取出正文。它的默认行为是流式（`stream_response_bool=True`），因为流式在长回答场景下更稳；需要一次性拿到完整响应对象时可以把该参数设为 `False`。需要注意的是它返回的只是文本，工具调用信息会被丢弃，所以它适合「问一句、拿一段话」的场景，不适合需要处理工具调用的 Agent 主循环。
- **参数**：`messages`（`list[dict[str, Any]]`，位置参数，必填）：对话消息列表，原样转发给 `complete`。`temperature`（`float`，默认 `DEFAULT_TEMPERATURE`）：采样温度，转发给 `complete`，因此会经过 `complete` 的有限数校验。`timeout`（`float`，默认 `DEFAULT_TIMEOUT`）：超时秒数，转发给 `complete`，因此会经过 `complete` 的「有限正数」校验。`stream_response_bool`（`bool`，默认 `True`）：是否走流式；为 `True` 时请求带 `stream=True` 并用 `stream_response` 拼接文本，为 `False` 时请求非流式并用 `_message_content` 取正文。它同时也被当作 `complete` 的 `stream` 参数传入，所以必须能通过 `complete` 里 `isinstance(stream, bool)` 的检查。`prompt_cache_key`（`str | None`，默认 `None`）：提示缓存键，转发给 `complete`，因此会经过「非空且不超过 64 字符」的校验。`prompt_cache_retention`（`str | None`，默认 `None`）：提示缓存保留策略，转发给 `complete`，因此只能是 `"in_memory"`、`"24h"` 或 `None`。`**kwargs`（`Any`）：其余参数原样转发给 `complete`，再由 `complete` 透传给 SDK；如果其中包含 `complete` 的保留参数名，会由 `complete` 抛出 `TypeError`。
- **返回**：返回 `str`。流式分支下是所有内容增量按到达顺序拼接后的字符串（如果流里没有任何内容，返回空字符串 `""`）；非流式分支下是 `_message_content` 的结果，该函数同样在 content 为假值时返回空字符串。因此本方法永远不会返回 `None`，只会返回字符串。
- **内部流程**：第一步，调用 `self.complete(...)`，把 `messages` 作为位置参数，`temperature`、`timeout`、`stream=stream_response_bool`、`prompt_cache_key`、`prompt_cache_retention` 以及 `**kwargs` 作为关键字参数传入，把返回的响应对象存到局部变量 `response`。第二步，判断 `stream_response_bool` 是否为真；为真则执行 `"".join(self.stream_response(response))`——即用生成器逐块取出文本增量、在内存里拼接成完整字符串并返回。第三步，为假则返回 `self._message_content(response)`，也就是从非流式响应里安全地取出 `choices[0].message.content`（取不到或为空时返回空字符串）。
- **异常/边界**：本方法自身不做任何校验、不捕获任何异常。所有校验错误（`timeout` 非正、`temperature` 非有限数、`stream` 非布尔、缓存键非法、保留参数被覆盖）都从 `complete` 抛出。网络或网关错误从 `complete` 或流消费过程中向上抛出。当 `stream_response_bool=True` 时，响应会被 `stream_response` 边打印边消费，也就是说这个方法在流式模式下会**有终端输出副作用**（内容会被 `print` 到标准输出），调用方如果不希望打印需要自己改用 `complete_streaming`。非流式分支下 content 缺失返回空字符串而不报错（与 `_message_content` 在缺少 choices/message 时才报错的行为一致）。
- **同文件关系**：调用 `LLM.complete`、`LLM.stream_response`、`LLM._message_content`。不被本文件内其他函数调用（由外部上层调用）。

#### `complete(self, messages: list[dict[str, Any]], *, model: str | None = None, temperature: float = DEFAULT_TEMPERATURE, timeout: float = DEFAULT_TIMEOUT, stream: bool = False, prompt_cache_key: str | None = None, prompt_cache_retention: str | None = None, **kwargs: Any) -> Any` （第 330 行）

- **作用**：这是类里最底层、最通用的补全入口，`think` 就是基于它实现的。它把调用方给的参数做一轮严格校验（超时必须是有限正数、温度必须是有限数、`stream` 必须是布尔、缓存键必须是去空白后非空且不超过 64 字符、缓存保留策略只能是 `in_memory` 或 `24h`），拒绝任何试图通过 `kwargs` 覆盖保留参数的行为，然后组装 `options` 并调用 `self.client.chat.completions.create`，把原始响应对象（或原始流对象）原样返回。它刻意不做拼装：`stream=True` 时返回的就是 SDK 的流对象，需要由调用方自己用 `stream_response` 或 `complete_streaming` 消费。提示缓存字段采用「非 `None` 才带上」的策略，注释里说明了原因：这两个字段是 OpenAI Chat Completions 的一等参数，但不设置时省略它们可以让更老的 OpenAI 兼容网关继续正常工作。
- **参数**：`messages`（`list[dict[str, Any]]`，位置参数，必填）：对话消息列表，原样透传，不做校验。`model`（`str | None`，仅关键字，默认 `None`）：模型名；`None` 时用 `self.model`。`temperature`（`float`，仅关键字，默认 `DEFAULT_TEMPERATURE`）：采样温度；必须是 `int` 或 `float`（但**不接受布尔值**）且是有限数（`math.isfinite` 为真），也就是说 `NaN` 和 `inf` 都会被拒绝。`timeout`（`float`，仅关键字，默认 `DEFAULT_TIMEOUT`）：超时秒数；必须是 `int` 或 `float`（同样不接受布尔值）、有限、且**大于 0**。`stream`（`bool`，仅关键字，默认 `False`）：是否流式；必须是布尔类型，传 `1`/`0` 之类的整数会被拒绝。`prompt_cache_key`（`str | None`，仅关键字，默认 `None`）：提示缓存路由键；非 `None` 时必须是字符串、`strip()` 后非空、且长度不超过 64 个字符，通过校验后会被 `strip()` 后再放进请求。`prompt_cache_retention`（`str | None`，仅关键字，默认 `None`）：提示缓存保留策略；必须是 `"in_memory"`、`"24h"` 或 `None` 三者之一。`**kwargs`（`Any`）：其余参数原样透传给 SDK，例如 `tools`、`tool_choice`、`max_tokens`、`response_format` 等；其中不允许出现保留参数名。
- **返回**：返回 `Any`，即 `self.client.chat.completions.create` 的原始返回值：`stream=False` 时是一个 ChatCompletion 响应对象（支持属性访问，也可能被上层按 Mapping 访问），`stream=True` 时是一个可迭代的流对象（需要调用方自行消费并关闭）。本方法不对返回值做任何包装或转换。
- **内部流程**：第一步，校验 `timeout`：不是 `int`/`float`、或是 `bool`、或不是有限数、或 `<= 0`，任一成立则抛 `ValueError("timeout must be a finite positive number")`。第二步，校验 `temperature`：不是 `int`/`float`、或是 `bool`、或不是有限数，任一成立则抛 `ValueError("temperature must be a finite number")`。第三步，校验 `stream` 是 `bool`，否则抛 `TypeError("stream must be a boolean")`。第四步，若 `prompt_cache_key` 非 `None`，则检查它是非空字符串且 `len()` 不超过 64，否则抛 `ValueError`；通过后把它 `strip()` 并重新赋给局部变量（注意是重新赋值，因此后续用的是去空白版本）。第五步，若 `prompt_cache_retention` 非 `None`，则检查它是否在 `{"in_memory", "24h"}` 集合内，不在则抛 `ValueError`。第六步，用集合交集检查 `kwargs` 是否包含 `messages`、`model`、`temperature`、`stream`、`timeout`、`prompt_cache_key`、`prompt_cache_retention` 中任意一个，有则抛 `TypeError` 并列出被覆盖的参数名。第七步，构造 `options`：`model` 取 `model or self.model`，其余用校验后的 `messages`、`temperature`、`stream`、`timeout`。第八步，按需把去空白后的 `prompt_cache_key` 和非 `None` 的 `prompt_cache_retention` 加进 `options`（注释说明省略它们是为了兼容老网关）。第九步，调用并返回 `self.client.chat.completions.create(**options, **kwargs)`。
- **异常/边界**：`ValueError`：`timeout` 非法（含布尔、`NaN`、`inf`、0 和负数）；`temperature` 非法（含布尔、`NaN`、`inf`）；`prompt_cache_key` 不是字符串、去空白后为空、或超过 64 字符；`prompt_cache_retention` 不在允许集合内。`TypeError`：`stream` 不是布尔；`kwargs` 里出现任意保留参数名。本方法不捕获 SDK 抛出的任何异常，网络错误、鉴权错误、超时、限流都会原样向上传播。`messages` 为空列表或结构错误不会被本方法拦截，会直接送到网关。布尔值被刻意排除，因为 Python 里 `True` 是 `int` 的实例，不排除的话 `timeout=True` 会被当成 1 秒接受。
- **同文件关系**：被 `LLM.think` 调用（两个分支都先经过它）。它自身不调用本文件里的其他函数或方法。

#### `_message_content(response: Any) -> str` （第 405 行）

- **作用**：这是一个静态方法，用来从「可能是对象、也可能是字典」的响应里安全地取出助手正文。它之所以要写成双形态兼容，是因为流式拼装出来的结果是普通字典，而 SDK 返回的是带属性的对象，两者都会流到这一层；如果只按一种形态取值，另一条路径就会崩。它逐级做存在性检查：先取 `choices`，为空则报错；再取 `choices[0]` 的 `message`，为 `None` 则报错；最后取 `message.content`，取不到或用假值时返回空字符串。因为是静态方法，它不依赖实例状态，可以在没有构造 `LLM` 的情况下直接调用，也方便测试。
- **参数**：`response`（`Any`）：要解析的响应对象。既可以是 `Mapping`（例如 `_assemble_streaming_response` 返回的字典），也可以是任意带属性的对象（例如 `openai` SDK 的 ChatCompletion）。没有默认值。函数对它的类型不做断言，而是用 `isinstance(..., Mapping)` 逐层分流。
- **返回**：返回 `str`。正常路径下返回 `message.content` 的字符串内容；当 `content` 为 `None`、空字符串或其他假值（例如空列表）时，用 `or ""` 归一为空字符串返回。因此本方法不会返回 `None`，但会在结构缺失时抛异常而不是返回空串。
- **内部流程**：第一步，判断 `response` 是否为 `Mapping` 实例：是则调用 `response.get("choices")`，否则用 `getattr(response, "choices", None)`，把结果存到 `choices`。第二步，如果 `choices` 为假值（`None`、空列表等），抛 `ValueError("LLM response contained no choices")`。第三步，对 `choices[0]` 做同样的双形态分流取出 `message`（`Mapping` 用 `.get("message")`，否则用 `getattr(..., "message", None)`）。第四步，如果 `message is None`，抛 `ValueError("LLM response contained no message")`。第五步，对 `message` 再做一次双形态分流取出 `content`。第六步，返回 `content or ""`，把假值统一成空字符串。
- **异常/边界**：`ValueError("LLM response contained no choices")`：`choices` 缺失、为 `None` 或为空列表。`ValueError("LLM response contained no message")`：`choices[0]` 上没有 `message` 属性/键，或该值为 `None`。若 `response` 既不是 `Mapping` 也没有 `choices` 属性，会被第一步的 `getattr(..., None)` 变成 `None`，从而走到第一个 `ValueError`，不会抛 `AttributeError`。`content` 为空、`None` 或其他假值时统一返回 `""`，不报错。注意若 `choices` 是一个非空但不可下标的对象（理论上不太可能），`choices[0]` 会抛 `TypeError`，本方法不做处理。
- **同文件关系**：被 `LLM.think` 在非流式分支调用。它自身不调用本文件里的其他函数或方法。

#### `stream_response(self, response: Iterable[Any])` （第 428 行）

- **作用**：这是一个生成器方法，用来从「SDK 对象形态或字典形态」的流式分块里逐个抽出正文增量，一边打印到标准输出，一边 `yield` 给调用方。它和 `_assemble_streaming_response` 的区别在于：后者是把整条流拼装成一个完整响应字典（适合工具循环），而它只关心文本、边收边吐（适合只想要文字的场景，`think` 就用它）。它还有一个重要的资源管理职责：用 `try/finally` 保证即使调用方中途停止迭代（提前 `break`、或者生成器被关闭/垃圾回收导致 `GeneratorExit`），底层的流也会被关闭，从而不会泄漏一条打开的 HTTP 连接。
- **参数**：`response`（`Iterable[Any]`）：要消费的流式对象，通常是 SDK 的 `Stream`（既支持迭代也有 `close()`），也可以是任何可迭代的 chunk 序列。没有默认值。由于内部统一用 `field()` 读取，字典形态的分块也能处理。
- **返回**：返回一个生成器对象（函数体内含 `yield`，所以调用它不会立即执行任何代码）。迭代它时，每个「非空 content 增量」会按到达顺序产出一个 `str`；没有内容的 chunk 会被跳过而不产出任何值。生成器耗尽时正常结束。
- **内部流程**：第一步，进入 `try` 块并 `for chunk in response` 遍历分块。第二步，对每个分块用 `field(chunk, "choices")` 取 choices，为空则 `continue` 跳过（注意这里没有像 `_assemble_streaming_response` 那样触发首块回调，也没有处理 `finish_reason`）。第三步，取 `choices[0]`，再取它的 `delta`，再取 `delta` 的 `content`。第四步，如果 `content` 为真值，先用 `print(content, end="", flush=True)` 把它不换行地、立即刷新地打到标准输出，然后用 `yield content` 把它交给调用方。第五步，无论循环是正常结束还是中途抛出异常（包括生成器被提前关闭），都进入 `finally`：用 `getattr(response, "close", None)` 取可选的 `close` 方法，若可调用则在一个 `try/except Exception` 中调用它；关闭失败只记一条 debug 日志，注释里明确说明「关闭是尽力而为，绝不能掩盖更有用的流异常」。
- **异常/边界**：迭代 `response` 时抛出的异常会穿过 `finally` 继续向调用方传播（`finally` 只做关闭，不吞掉原异常）。`response.close()` 抛出的任何异常被吞掉并降级为 debug 日志。当调用方提前停止迭代时，生成器的 `finally` 会执行，流被关闭，不会泄漏连接。`choices` 为空、`delta` 为 `None`、`content` 为空串或 `None` 都会被静默跳过，不产出值。与 `_assemble_streaming_response` 不同，它不收集 `tool_calls`、不记录 `finish_reason`、不触发 `on_first_chunk`，因此**不能**用于需要工具调用的轮次。它有一个明确的副作用：每次产出文本前都会 `print` 到标准输出，调用方无法关闭这个打印行为。
- **同文件关系**：调用 `field`（来自同包 `agents/message_utils`）读取分块，并使用模块级 `LOGGER` 记录关闭失败。被 `LLM.think` 在流式分支调用（用 `"".join(...)` 消费它）。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `_default_echo_write` | 把文本写到 `sys.stdout` 并立即刷新，作为回声器的默认输出通道。 |
| `_FinalAnswerEchoer` | 单条流的终端回声控制器，按 `content` / `react_final` 两种模式决定何时把答案打到终端。 |
| `_FinalAnswerEchoer.__init__` | 初始化回声器的写函数、前缀、缓冲区和实时开关。 |
| `_FinalAnswerEchoer.feed` | 接收一个内容增量，实时转发或在缓冲区内等待最终答案标记出现后再转发标记之后的内容。 |
| `_FinalAnswerEchoer.flush` | 若此前输出过内容则补一个换行，收尾终端回声行。 |
| `_FinalAnswerEchoer._emit` | 首次输出时先写一次前缀，再写出正文，并吞掉写通道的异常。 |
| `_assemble_streaming_response` | 遍历流式分块，拼接 `content` 与按索引归并的 `tool_calls`，还原成与非流式同形的响应字典。 |
| `LLM` | 绑定一组模型配置的 OpenAI 兼容客户端封装，提供流式、非流式与纯文本三种调用方式。 |
| `LLM.__init__` | 校验并保存 api_key / base_url / model / max_retries，构造带显式代理的 httpx 客户端与 OpenAI 客户端。 |
| `LLM.complete_streaming` | 强制 `stream=True` 发起补全，并把流交给拼装函数返回完整响应字典，避免工具循环被读超时打断。 |
| `LLM.think` | 调 `complete` 后按是否流式分别拼接增量文本或直接取正文，只返回纯文本。 |
| `LLM.complete` | 严格校验超时、温度、`stream`、提示缓存字段后透传调用 `chat.completions.create`，原样返回响应或流。 |
| `LLM._message_content` | 从字典或对象形态的响应里逐级安全取出 `choices[0].message.content`，缺失时返回空串或报错。 |
| `LLM.stream_response` | 生成器：逐个抽取流式分块的正文，边打印边 `yield`，并在 `finally` 中关闭流。 |
