# memory/rag/document.py

## 一、这个文件是干什么的

这个文件是记忆系统里 RAG（检索增强生成）管线的「文档规范化与切块」底层模块。它定义了整条记忆管线共用的两个数据载体：`Document`（一份被规范化后的文本 + 稳定 id + 元数据）和 `ChunkSpan`（一个切块，外加它在规范化全文中的字符区间 `[char_start, char_end)`）。核心工作类 `DocumentProcessor` 负责两件事：一是把各种来源（`pathlib.Path`、`bytes`、文本流、纯字符串）读成 `Document`，并对 `.jsonl` / `.json` / `.csv` / `.html` / `.pdf` 等本地常见格式做解析；二是把文档按固定字符窗口加重叠切成可检索的块，或按中英文句读号切成句级块，并且两种切法都保留块在规范化全文中的偏移量。之所以要「偏移量」和「唯一规范化入口」，是因为数据库里存的 `documents.raw_text` 和向量库里的块必须来自同一份文本、同一套空白折叠规则，否则偏移会指向错误的字符，UI 高亮和按区间重切就会全部错位。文件末尾的 `resolve_within` 是安全边界工具：凡是接受调用方（可能由模型驱动）传入路径的入库入口，都用它把路径解析并限制在允许的根目录内，防止越界读取工作区之外的文件。整个文件不依赖任何其它业务模块，只从 `constants` 取两个切块参数，因此它可以被记忆层、RAG 层和 Web 层共同复用。

## 二、函数与类逐条详解

### `Document` 类（第 19 行）

- **作用**：全项目在 RAG 管线中传递「一份文档」的统一值对象。它把文本内容、文档 id 和元数据绑成一个不可变整体，让后续的切块、入库、检索都能拿到同样的标识与附加信息。它是 `@dataclass(frozen=True)`，因此一旦构造完成就不能再改字段，避免下游某处悄悄改掉 `content` 却让已经写库的 `raw_text` 与偏移量对不上。它被 `DocumentProcessor.parse`（解析单个来源时）、`chunks_with_spans`、`sentences_with_spans`（生成每个切块时）反复构造，是整个文件里出现频率最高的类型。
- **参数**：作为 dataclass，字段即构造参数。`content: str` 是必填的正文，必须是非空字符串（允许前后有空白，但 `strip()` 后不能为空）。`id: str = field(default_factory=lambda: str(uuid4()))` 是可选 id，默认用一个随机 UUID4 字符串，便于调用方不显式给 id 时也能得到全局唯一标识。`metadata: Mapping[str, Any] = field(default_factory=dict)` 是可选元数据映射，默认空字典，可以传任意只读映射（如 `MappingProxyType`），构造时会被复制成普通 `dict`。
- **返回**：无返回值（类构造返回实例本身）。
- **内部流程**：dataclass 自动生成的 `__init__` 先按顺序接收三个字段并赋值；随后自动调用 `__post_init__` 做校验与规整（见下一条）。由于类是 frozen 的，字段赋值走的是 dataclass 生成的冻结写入逻辑，普通属性赋值会抛 `FrozenInstanceError`。
- **异常/边界**：`content` 不是 `str`、或是空串/纯空白串时抛 `ValueError`；`id` 不是 `str` 或纯空白时抛 `ValueError`；`metadata` 传 `None` 时不会报错，`__post_init__` 里 `self.metadata or {}` 会把它变成空字典。实例化后修改任意字段会抛 `FrozenInstanceError`（由 frozen dataclass 机制抛出）。
- **同文件关系**：被 `DocumentProcessor.parse`、`DocumentProcessor.chunks_with_spans`、`DocumentProcessor.sentences_with_spans` 构造；`ChunkSpan` 的第一个字段类型就是 `Document`；`normalized_text` 读取它的 `content`；`chunks_with_spans` / `sentences_with_spans` 读取它的 `id` 与 `metadata`。它本身不调用本文件任何函数。

### `Document.__post_init__(self) -> None` （第 25 行）

- **作用**：dataclass 在 `__init__` 之后自动调用的校验钩子，用来守住 `Document` 的两个不变量：正文必须是有内容的字符串、id 必须是有内容的字符串。同时它把传进来的 `metadata` 统一复制成普通 `dict`，这样调用方后续改自己那份原始字典（或原始字典本身是只读映射）都不会影响已构造的文档对象，也保证元数据可以被下游用 `metadata.update(...)` 正常增补（切块时就需要这么做）。没有这一步，`Mapping` 类型的只读映射会在切块阶段直接报错。
- **参数**：`self`，即刚完成字段赋值的 `Document` 实例。无其它参数。
- **返回**：`None`。
- **内部流程**：第一步判断 `self.content` 是否为 `str`，再判断 `self.content.strip()` 是否为空，任一不满足即抛 `ValueError("document content must be a non-empty string")`。第二步对 `self.id` 做同样的类型与去空白判空检查，失败抛 `ValueError("document id must be a non-empty string")`。第三步调用 `object.__setattr__(self, "metadata", dict(self.metadata or {}))`：因为类是 frozen 的，普通 `self.metadata = ...` 会被拒绝，所以必须绕过 dataclass 的冻结写入走 `object.__setattr__`；`self.metadata or {}` 兼顾了传入 `None` 或传入空映射的情况，`dict(...)` 完成浅拷贝。
- **异常/边界**：`content` 类型不对或空白 → `ValueError`；`id` 类型不对或空白 → `ValueError`；`metadata` 为 `None`/空 → 静默变成 `{}`；`metadata` 不是映射类型（例如传列表）时 `dict(...)` 会抛 `TypeError` 或 `ValueError`。注意校验只做 `strip()` 判空，不做长度上限或编码检查。
- **同文件关系**：由 `Document` 类在构造时自动调用；它内部只用到 `object.__setattr__`，不调用本文件其它函数。

### `ChunkSpan` 类（第 33 行）

- **作用**：把「一个切块」和「这个切块在规范化全文里的字符区间」绑在一起返回。它存在的理由是 RAG 的两处下游需求：一是按区间重新切片/重新切块（例如改变窗口大小后从 `raw_text` 里重取），二是前端在原文上高亮命中片段。文档字符串明确写了 `char_start` / `char_end` 索引进 `DocumentProcessor.normalized_text` 的输出，而这个输出也正是数据库 `documents.raw_text` 存的内容——这个「同源」约定就是偏移量可用的前提。`DocumentProcessor.chunks_with_spans` 和 `sentences_with_spans` 都返回它的列表。
- **参数**：作为 `@dataclass(frozen=True)`，字段即参数。`chunk: Document` 是切出来的那个块本身（注意它的 `content` 是规范化文本的切片，`id` 形如 `原文id:序号`）。`char_start: int` 是块在规范化全文中的起始下标（含）。`char_end: int` 是结束下标（不含）。
- **返回**：无返回值（构造返回实例本身）。
- **内部流程**：dataclass 生成 `__init__` 依次赋值三个字段；没有 `__post_init__`，因此不做任何校验或规整；frozen 使得实例创建后不可修改。
- **异常/边界**：不做任何校验，因此传入负下标、`char_end < char_start`、非整数偏移都不会在这里报错，问题会推迟到使用偏移量的地方。字段缺失（少传一个）时由 dataclass 生成的标准 `TypeError` 报出。
- **同文件关系**：被 `DocumentProcessor.chunks_with_spans`、`DocumentProcessor.sentences_with_spans` 构造并返回；其 `chunk` 字段类型是 `Document`；自身不调用本文件任何函数。

### `DocumentProcessor` 类（第 48 行）

- **作用**：文件里的主力工作类，职责是「解析本地常见格式 + 把文本切成可检索的块」。它没有实例状态（没有自定义 `__init__`，也没有实例属性），所有能力都通过实例方法或静态方法暴露，因此实际使用时可以随便 `DocumentProcessor()` 出一个实例，或按需要复用同一个实例。它对外提供四条公开能力：`parse`（多来源归一成一个 `Document`）、`normalized_text`（唯一的空白规范化入口）、`chunks_with_spans`（定长重叠切块）、`sentences_with_spans`（句级切块），以及一条私有静态能力 `_parse_bytes`（按扩展名解析字节）。RAG 入库流程通常先 `parse` 得到 `Document`，再选一种切块方式拿到 `ChunkSpan` 列表，然后逐块写库/写向量。
- **参数**：无构造参数（使用 dataclass 之外的普通类定义，隐式继承 `object.__init__`）。
- **返回**：无返回值（构造返回实例本身）。
- **内部流程**：类体内按顺序定义 `parse`、`normalized_text`、`chunks_with_spans`、`sentences_with_spans`、`_parse_bytes`（带 `@staticmethod` 装饰）。除 `_parse_bytes` 外都是实例方法，第一个参数为 `self`（虽然它们并不读取实例状态）。
- **异常/边界**：构造本身不会抛异常。类内方法各自的异常见下面各条。
- **同文件关系**：`parse` 调用 `_parse_bytes`；`chunks_with_spans` 与 `sentences_with_spans` 都调用 `normalized_text` 并构造 `ChunkSpan` / `Document`；`_parse_bytes` 只在 `parse` 内部被调用。类的文档字符串说明了它的整体定位。

### `DocumentProcessor.parse(self, source: str | Path | io.TextIOBase | bytes, *, metadata: Mapping[str, Any] | None = None) -> Document` （第 51 行）

- **作用**：把「一个来源」统一归一成一个 `Document`，是入库流程的第一步。它刻意区分了「字面文本」和「文件路径」：`str` 永远按字面文本处理，绝不拿去探测文件系统。这个设计是有安全含义的——如果字符串被当成路径去尝试打开，那么一段恰好等于某个文件名的短文本、或者任何绝对路径，都会变成可读文件，模型通过 `memory.rag` 工具传入的 source 就能逃出工作区沙箱。因此文件输入必须显式：传 `Path` / `os.PathLike`（或者走会做路径转换与包含性校验的 `RAGPipeline.ingest_source`）。当来源是路径时，它还会自动补上 `source`、`filename`、`extension` 三个元数据字段，再让调用方传入的 `metadata` 覆盖它们。
- **参数**：`source`（必填，位置参数）可以是 `str`（字面文本）、`bytes`（原始字节，按 UTF-8 宽松解码）、`io.TextIOBase`（已打开的文本流，调用 `read()`）、或 `pathlib.Path` / 任何带 `__fspath__` 的 `os.PathLike`（按文件读取并解析）。`metadata`（仅关键字，默认 `None`）是一个映射，会作为文档元数据；传 `None` 时等价于空字典。
- **返回**：始终返回一个 `Document` 实例。路径来源时返回的是 `_parse_bytes` 解析后的正文，元数据里至少含 `source`（路径字符串）、`filename`（文件名）、`extension`（小写后缀，如 `.pdf`）；字节/流/字符串来源时正文分别是解码结果、`read()` 结果、原字符串，元数据就是传入的 `metadata or {}`。
- **内部流程**：按顺序做类型分支判断。第一支：`isinstance(source, Path) or hasattr(source, "__fspath__")` 为真时，用 `Path(source)` 包一层，`path.read_bytes()` 读原始字节，构造基础元数据字典 `{"source": str(path), "filename": path.name, "extension": path.suffix.lower()}`，再用 `base.update(metadata or {})` 让调用方的元数据覆盖默认值，最后 `Document(self._parse_bytes(raw, path.suffix.lower()), metadata=base)`。第二支：`bytes` 时用 `source.decode("utf-8", errors="replace")` 宽松解码，元数据用 `metadata or {}`。第三支：`io.TextIOBase` 时取 `source.read()`。第四支：`str` 时直接当正文。四支都不匹配则抛 `TypeError`。
- **异常/边界**：来源类型不在四类之内 → `TypeError("source must be text, bytes, a path, or a text stream")`。路径不存在、无权限、是目录 → 由 `path.read_bytes()` 抛 `FileNotFoundError` / `PermissionError` / `IsADirectoryError`。`.json` / `.jsonl` 内容非法 → `_parse_bytes` 里 `json.loads` 抛 `json.JSONDecodeError`。`.pdf` 且未安装 `pypdf` → `RuntimeError("PDF parsing requires pypdf")`。传入的 `metadata` 若不是映射，`base.update(...)` 或 `Document.__post_init__` 里的 `dict(...)` 会抛 `TypeError`/`ValueError`。注意本方法自身不做任何目录包含性校验——那是 `resolve_within` 与上层入口的职责。
- **同文件关系**：调用 `DocumentProcessor._parse_bytes` 完成字节解析，并构造 `Document`；不调用 `normalized_text`、`chunks_with_spans`、`sentences_with_spans`。它自身不被本文件其它函数调用，是外部（RAG 管线/入库入口）进入本模块的主要入口。

### `DocumentProcessor.normalized_text(self, document: Document) -> str` （第 77 行）

- **作用**：定义整个模块唯一的「规范化文本」形式，供切块和落库共同使用。它把所有连续空白（空格、制表符、换行、以及 Python 正则 `\s` 覆盖的其它空白字符）折叠成单个空格，再去掉首尾空白。为什么必须「只在这里规范化、别处不许再规范化」：如果数据库 `documents.raw_text` 和切块偏移量来自两套不同的空白处理，那么 `char_start` / `char_end` 就会指向错误的字符位置，高亮和重切全部失效。因此它是 `chunks_with_spans` 与 `sentences_with_spans` 计算偏移前必经的一步。
- **参数**：`document`（必填，位置参数）是一个 `Document` 实例；方法只读取它的 `content` 字段，不读 `id` 和 `metadata`。
- **返回**：返回规范化后的字符串。当 `document.content` 全是空白时，返回空字符串 `""`（`re.sub` 会先把它变成空格串，`strip()` 再清空）。
- **内部流程**：单表达式实现：`re.sub(r"\s+", " ", document.content).strip()`。先用 `\s+` 把所有空白串替换成一个普通空格，再用 `strip()` 去掉两端可能残留的空格。没有缓存、没有副作用，同样输入必然得到同样输出。
- **异常/边界**：`document.content` 已由 `Document.__post_init__` 保证是非空字符串，因此正常路径不会因空值报错；但如果绕过校验（例如用 `object.__new__` 造对象）传了非字符串，`re.sub` 会抛 `TypeError`。空白文档返回 `""`，调用方（两个切块方法）都会据此提前返回空列表，不做无意义的切块。
- **同文件关系**：被 `DocumentProcessor.chunks_with_spans` 和 `DocumentProcessor.sentences_with_spans` 调用；它读取 `Document.content`；自身不调用本文件其它函数。

### `DocumentProcessor.chunks_with_spans(self, document: Document, *, chunk_size: int = RAG_CHUNK_SIZE, overlap: int = RAG_CHUNK_OVERLAP) -> list[ChunkSpan]` （第 87 行）

- **作用**：把文档按「固定字符窗口 + 相邻窗口重叠」切成检索用的块，并同时给出每块在规范化全文中的字符区间。重叠窗口的意义是避免一句话或一个语义单元被硬生生切断后两边都检索不到。它是 RAG 入库最常用的切块方式（相对句级切块而言是默认粒度）。每个块都是一个独立的 `Document`，id 形如 `原文id:序号`，元数据里带 `document_id` 和 `chunk_index`，便于检索命中后回溯到源文档并排序。
- **参数**：`document`（必填，位置参数）待切块的 `Document`。`chunk_size`（仅关键字，默认取常量 `RAG_CHUNK_SIZE`）是每块的目标字符数，必须是正整数；显式传 `bool` 会被拒绝。`overlap`（仅关键字，默认取常量 `RAG_CHUNK_OVERLAP`）是相邻块的重叠字符数，必须是非负整数且严格小于 `chunk_size`；显式传 `bool` 同样会被拒绝。
- **返回**：返回 `list[ChunkSpan]`，按文本顺序排列，每个元素含 `chunk`（`Document`）、`char_start`、`char_end`。若规范化后的文本为空，返回空列表 `[]`。当文档长度不超过 `chunk_size` 时，返回只含一个块的列表，其区间为 `[0, len(text))`。
- **内部流程**：先做两道参数校验（`chunk_size` 必须是 `int`、非 `bool`、≥1；`overlap` 必须是 `int`、非 `bool`、≥0 且 `< chunk_size`），再调 `self.normalized_text(document)` 拿到统一文本；文本为空直接返回 `[]`。接着算步长 `step = chunk_size - overlap`（由校验保证 ≥1，不会死循环），用 `enumerate(range(0, len(text), step))` 逐个起点推进：每轮 `end = min(start + chunk_size, len(text))`，切出 `chunk = text[start:end]`；若切出来是空串就 `break`。然后 `metadata = dict(document.metadata)` 浅拷贝原元数据，`metadata.update({"document_id": document.id, "chunk_index": index})` 补上溯源信息，追加 `ChunkSpan(Document(chunk, id=f"{document.id}:{index}", metadata=metadata), start, end)`。最后判断 `if end >= len(text): break`，即已经切到文本末尾时立刻结束，避免再产出一个完全落在重叠区里的重复尾块。
- **异常/边界**：`chunk_size` 非法（`bool`、非整数、小于 1）→ `ValueError("chunk_size must be a positive integer")`；`overlap` 非法（`bool`、非整数、负数、大于等于 `chunk_size`）→ `ValueError("overlap must be non-negative and smaller than chunk_size")`。空文档/纯空白文档 → 返回 `[]`。`document.metadata` 若是只读映射也没问题，因为先 `dict(...)` 拷贝了。若 `document.metadata` 本身不是可被 `dict()` 接受的映射，会抛 `TypeError`（正常路径下 `Document.__post_init__` 已保证是 `dict`）。
- **同文件关系**：调用 `DocumentProcessor.normalized_text` 取规范化文本；构造 `Document`（每个块）与 `ChunkSpan`（返回值元素）；不调用 `parse`、`sentences_with_spans`、`_parse_bytes`。自身不被本文件其它函数调用。

### `DocumentProcessor.sentences_with_spans(self, document: Document) -> list[ChunkSpan]` （第 112 行）

- **作用**：句级切块，用于「逐句导入」这一场景（代码注释标注为 F4）。它把文档按中英文常用句读号切成一句一条记录，每句单独入库，好处是检索粒度极细、命中后返回的片段短而精确。句边界同样索引到 `normalized_text` 的输出，与 `chunks_with_spans` 同源，因此偏移量可直接用于重切与前端高亮。为了和定长字符块区分，句级块的元数据里额外带 `metadata["granularity"] = "sentences"`，`chunk_index` 按结果顺序从 0 连续编号。它还会专门处理「结尾没有句读号的残句」，否则最后一句会凭空丢失。
- **参数**：`document`（必填，位置参数）待切句的 `Document`；方法只读它的 `content`、`id`、`metadata`。没有其它参数，切句规则（句读号集合）在方法内部硬编码。
- **返回**：返回 `list[ChunkSpan]`，按文本顺序排列。每个 `ChunkSpan.chunk` 是一个句级 `Document`，`id` 形如 `原文id:序号`（序号与 `chunk_index` 一致），元数据含 `document_id`、`chunk_index`、`granularity="sentences"`。规范化文本为空时返回 `[]`；文本不含任何句读号时，残句兜底逻辑会把整段文本作为唯一一块返回。
- **内部流程**：先 `text = self.normalized_text(document)`，空则返回 `[]`。设游标 `start = 0`，逐字符 `enumerate(text)`：字符不在 `"。！？!?.;\n"` 集合里就跳过；是句读号时，取 `chunk = text[start : index + 1].strip()`（含句读号本身），若 `chunk` 非空就构造元数据（拷贝原元数据后写入 `document_id`、`chunk_index = len(result)`、`granularity`），计算 `end = start + len(chunk)`，追加 `ChunkSpan(Document(chunk, id=f"{document.id}:{len(result)}", metadata=metadata), start, end)`；无论是否产出块，都把 `start = index + 1` 推过这个句读号。循环结束后取尾部 `tail = text[start:].strip()`，若非空则同样构造元数据并追加一块：起始位置 `begin` 用 `start + (len(text[start:]) - len(text[start:].lstrip()))` 精确跳过尾部前导空白，区间为 `[begin, begin + len(tail))`。
- **异常/边界**：`document` 为空内容时（经 `Document` 校验其实不可能为空串，但纯空白可以）规范化结果为 `""`，直接返回 `[]`。注意一个已知的偏移近似：当句读号后紧跟空格时，`start` 指向那个空格，而 `chunk` 被 `strip()` 掉了它，于是该块的 `char_start` 会指向空格、`char_end = char_start + len(chunk)` 也比真实结束位置偏左一点（尾部兜底分支用 `lstrip` 计数修正了这一点，主循环没有）。元数据拷贝后 `update` 需要 `dict`，正常路径由 `Document.__post_init__` 保证。本方法不做任何参数校验，也没有句读号集合可配置的入口。
- **同文件关系**：调用 `DocumentProcessor.normalized_text`；构造 `Document`（每句）与 `ChunkSpan`（返回值元素）；不调用 `parse`、`chunks_with_spans`、`_parse_bytes`。自身不被本文件其它函数调用。

### `DocumentProcessor._parse_bytes(raw: bytes, extension: str) -> str` （第 168 行）

- **作用**：私有静态方法，按文件扩展名把原始字节解析成可供切块的纯文本，是 `parse` 在路径分支上的实际干活者。它把「格式差异」都收敛到这一处：结构化文本（JSONL/JSON）会被重新序列化成易读文本，CSV 会被拍成用 ` | ` 分隔的行，HTML 会被剥掉标签，PDF 会抽文字，其余一律按 UTF-8 宽松解码。这样上层 `parse` 只需关心「来源是什么」，不必关心「格式怎么读」。因为是 `@staticmethod`，调用时不需要 `self`，也不访问任何实例状态。
- **参数**：`raw: bytes` 是文件原始字节内容。`extension: str` 是小写扩展名（`parse` 传入 `path.suffix.lower()`），用于分派解析器；支持的分支有 `.jsonl`、`.json`、`.csv`、`.html`/`.htm`、`.pdf`，其它值走默认解码分支。
- **返回**：返回解析出的字符串文本。`.jsonl` 返回每个非空行经 `json.loads` 后再 `json.dumps(..., ensure_ascii=False)` 的结果用换行连接（即逐行规范化，中文不转义）；`.json` 返回 `json.dumps(value, ensure_ascii=False, indent=2)` 的缩进美化结果；`.csv` 返回各行用 `" | "` 连接后的文本；`.html`/`.htm` 返回把 `<...>` 标签替换成空格后的文本；`.pdf` 返回各页 `extract_text()` 结果用换行连接（某页无文字时用 `""` 占位）；其它扩展名返回 `raw.decode("utf-8", errors="replace")`。
- **内部流程**：一串 `if` 分派。`.jsonl`：`raw.decode("utf-8")` 后 `splitlines()`，过滤掉 `line.strip()` 为空的行走生成器，逐行 `json.dumps(json.loads(line), ensure_ascii=False)`，最后 `"\n".join(...)`。`.json`：整体 `json.loads(raw.decode("utf-8"))` 后用 `indent=2` 重新输出。`.csv`：`csv.reader(io.StringIO(raw.decode("utf-8", errors="replace")))` 逐行把字段用 `" | "` 拼接。`.html`/`.htm`：正则 `re.sub(r"<[^>]+>", " ", ...)` 粗暴剥标签（不做实体解码、不区分脚本内容）。`.pdf`：在函数内部延迟 `from pypdf import PdfReader`，用 `PdfReader(io.BytesIO(raw))` 读内存字节，`"\n".join(page.extract_text() or "" for page in reader.pages)`。默认分支直接宽松解码。
- **异常/边界**：未安装 `pypdf` 时 `ImportError` 被捕获并转成 `RuntimeError("PDF parsing requires pypdf")`（用 `from exc` 保留原始异常链）。`.jsonl`/`.json` 内容非法 → `json.JSONDecodeError`；`.jsonl` 解码用严格 UTF-8（`raw.decode("utf-8")` 无 `errors` 参数），遇到非法字节会抛 `UnicodeDecodeError`，而 `.csv`/`.html`/默认分支都用 `errors="replace"` 容错。加密或损坏的 PDF 可能由 `pypdf` 抛异常。扩展名大小写不匹配（例如 `.JSON`）不会进对应分支，因为调用方已 `lower()`。无参数校验。
- **同文件关系**：只被 `DocumentProcessor.parse` 调用；自身不调用本文件其它函数（`pypdf` 是延迟导入的外部依赖）。

### `resolve_within(base_dir: str | Path, path: str | Path) -> Path` （第 190 行）

- **作用**：模块级安全工具函数，把调用方给出的路径解析成绝对路径，并校验它落在允许的根目录 `base_dir` 之内，越界就拒绝。它服务的是「入库入口接受调用方传路径」这条链路：由于这些调用可能由模型驱动（工具调用），如果不加约束，模型就能用 `../..` 或绝对路径读到工作区之外的任意文件。比较时对两侧都做了解析（`resolve()`，会展开符号链接与 `..`）和大小写规范化（`os.normcase`），与项目里 `fs.*` 系列工具的包含性规则保持一致。
- **参数**：`base_dir: str | Path` 是允许的根目录，必须是 `str` 或 `pathlib.Path`（其它类型直接 `TypeError`），会经 `expanduser()` 与 `resolve()` 处理。`path: str | Path` 是待校验的候选路径；相对路径会被拼到 `base` 之下再解析，绝对路径保持绝对。
- **返回**：返回解析后的绝对 `Path` 对象（`candidate.resolve()` 的结果）。校验通过时它一定是 `base` 本身或 `base` 的子路径；调用方应当使用这个返回值而不是原始 `path`，以免后续操作再走一次不同的解析。
- **内部流程**：第一步类型检查 `isinstance(base_dir, (str, Path))`，不满足抛 `TypeError("base_dir must be a string or pathlib.Path")`。第二步 `base = Path(base_dir).expanduser().resolve()`。第三步 `candidate = Path(path).expanduser()`，若 `not candidate.is_absolute()` 则 `candidate = base / candidate`。第四步 `resolved = candidate.resolve()`。第五步取 `base_text = os.path.normcase(str(base))` 与 `resolved_text = os.path.normcase(str(resolved))`，判断「相等 或 以 `base_text + os.sep` 为前缀」；两者都不满足则抛 `ValueError(f"path '{path}' resolves outside the allowed directory '{base}'")`。通过则返回 `resolved`。
- **异常/边界**：`base_dir` 类型不对 → `TypeError`；`path` 类型不对 → `Path(path)` 抛 `TypeError`。越界（含 `..` 逃逸、绝对路径指向别处、以及指向 `base` 的兄弟目录时因前缀比较带 `os.sep` 而被正确拒绝）→ `ValueError`。路径不存在不会报错（`resolve()` 在严格模式下仍可解析不存在的路径，本函数未使用 `strict=True`）。符号链接指向外部时，因为用的是 `resolve()` 后的文本比较，会被判为越界。前缀比较依赖 `os.sep`，因此在 Windows 上比较基于反斜杠、经 `normcase` 统一大小写；它不做 UNC 或驱动器号特殊处理。
- **同文件关系**：不调用本文件任何函数，也不被本文件其它函数调用（文件内部没有引用它）；它是给上层入库入口（如 `RAGPipeline.ingest_source` 这类接受字符串路径的地方）配合 `DocumentProcessor.parse` 使用的独立工具，并通过 `__all__` 对外导出。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `Document` | 不可变的文档值对象，绑定非空正文、唯一 id 与元数据，是 RAG 管线传递文档的统一载体。 |
| `Document.__post_init__` | 校验正文与 id 非空，并把元数据复制成可变字典。 |
| `ChunkSpan` | 把一个切块与其在规范化全文中的字符区间 `[char_start, char_end)` 绑在一起。 |
| `DocumentProcessor` | 负责解析本地常见格式并把文本切成可检索块的处理器类。 |
| `DocumentProcessor.parse` | 把字符串文本、字节、文本流或文件路径归一成一个 `Document`，路径输入自动补 source/filename/extension。 |
| `DocumentProcessor.normalized_text` | 唯一的空白折叠与首尾去空白入口，供切块与落库共用，保证偏移量同源。 |
| `DocumentProcessor.chunks_with_spans` | 按固定字符窗口加重叠切块，并返回每块在规范化全文中的字符区间。 |
| `DocumentProcessor.sentences_with_spans` | 按中英文句读号切句（含结尾残句兜底），返回带区间的句级块。 |
| `DocumentProcessor._parse_bytes` | 私有静态方法，按扩展名解析 jsonl/json/csv/html/pdf 等字节内容为纯文本。 |
| `resolve_within` | 把调用方路径解析为绝对路径并拒绝越出允许根目录的路径，防止越界读文件。 |
