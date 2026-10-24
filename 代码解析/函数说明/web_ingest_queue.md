# web/ingest_queue.py

## 一、这个文件是干什么的

这个文件实现了「一句话后台入库队列」，是整个 Web 应用中把用户随手写下的一句话异步转成知识（向量化 + LLM 抽取）的调度中心。它的核心设计是：SQLite 持久化任务状态 + 单线程串行消费。用户提交后立即返回、前端可以马上清空输入框，真正的重活由后台线程按提交顺序完成。之所以必须串行而不能并行，是因为后一次入库可能要检索、依赖上一次已经写入的图与向量，并行会导致读取到不完整的状态。

每条任务的生命周期是 `pending → running → done/failed`，状态写在 memories 使用的同一个 SQLite 文件的 `ingest_jobs` 表里，因此历史随时可查、进程重启也不丢。进程异常退出后，`IngestJobQueue.start()` 会把遗留的 `running` 任务复位为 `pending` 并重新入队，配合 `MAX_ATTEMPTS` 上限防止坏任务造成死循环。

文件内容分为三块：模块级常量与展示口径（`MAX_ATTEMPTS`、`JOB_LABELS`）、一个把数据库记录转成 API 友好字典的纯函数 `job_to_dict`，以及主体类 `IngestJobQueue`（对外提供 `available/start/submit/job/retry/list/wait/shutdown`，内部用 `_enqueue/_run_job/_notify` 完成调度与执行）。它依赖 `memory.rag` 的 `Document`/`RAGPipeline` 做实际入库，依赖 `memory.storage.document_repo` 的 `DocumentRepository`/`IngestJobRecord` 做持久化。典型用法是：Web 层在启动时构造队列并 `start()`，请求进来时 `submit()` 拿到记录即刻回显，需要同步结果时用 `wait()`，进程退出时 `shutdown()`。

## 二、函数与类逐条详解

### `job_to_dict(job: IngestJobRecord, *, text_preview: int = 120) -> dict[str, Any]` （第 37 行）

- **作用**：把数据库里的一条入库任务记录（`IngestJobRecord` 对象）转换成可以直接被 JSON 序列化、直接返回给前端或 API 的普通字典。因为 API 层需要统一的字段口径和可读的中文状态文案，而数据库记录是内部结构、`result` 字段还是 JSON 字符串，所以需要一个集中转换点。它还顺带做了两件展示层的事情：把过长的原始文本截断成预览，以及把 `result` 字符串解析回字典。此外它计算出 `retryable` 字段，告诉前端「这条任务现在能不能点重试按钮」，避免前端自己判断状态字符串。只要 Web 层要返回任务信息，就会用到它。
- **参数**：
  - `job`：必填，类型 `IngestJobRecord`，来自 `memory.storage.document_repo`，是数据库中的一条任务记录，需要具备 `job_id`、`kind`、`text`、`status`、`attempts`、`error`、`result`、`created_at`、`updated_at` 这些属性。调用方需保证它不为 `None`（函数内没有做空值判断）。
  - `text_preview`：仅关键字参数，类型 `int`，默认 `120`。表示返回的 `text` 字段最多保留多少个字符。取值应为非负整数；传 `0` 会得到空字符串，传负数会按 Python 切片语义截取到倒数第几个字符之前。约束上它只是切片长度，不做合法性校验。
- **返回**：返回一个 `dict[str, Any]`，固定包含 11 个键：`job_id`（原样透传的任务 id）、`kind`（任务种类）、`text`（截断后的文本预览）、`status`（原始状态字符串）、`label`（由 `JOB_LABELS` 映射出的中文文案，未登记的状态直接回落为原状态字符串）、`attempts`（已尝试次数）、`error`（错误信息，正常时为 `None`）、`result`（解析后的字典，解析失败或原值为空时是空字典 `{}`）、`created_at`、`updated_at`（时间字符串）、`retryable`（布尔值，仅当 `status == "failed"` 时为 `True`）。任何情况下都返回字典，不会返回 `None`。
- **内部流程**：第一步把局部变量 `result` 初始化为空字典 `{}`。第二步判断 `job.result` 是否为真值（非空字符串等），若为真则用 `json.loads(job.result)` 尝试解析；解析成功且结果是 `dict` 实例时才赋值给 `result`，解析出的不是字典（例如列表、数字）则保持空字典。第三步用 `try/except ValueError` 包住解析过程，遇到非法 JSON 文本时把 `result` 重置为空字典，保证不向上抛异常。第四步构造并直接返回最终字典，其中 `text` 用 `job.text[:text_preview]` 切片截断，`label` 用 `JOB_LABELS.get(job.status, job.status)` 查询，`retryable` 用 `job.status == "failed"` 现算。
- **异常/边界**：只捕获 `ValueError`（`json.loads` 对非法 JSON 抛的就是它），因此 JSON 格式错误会被静默吞掉并降级为空 `result`。如果 `job` 为 `None`，访问 `job.result` 会抛 `AttributeError`，本函数不处理。如果 `job.text` 为 `None`，切片会抛 `TypeError`，也不处理。`json.loads` 若因其他原因（如类型问题）抛 `TypeError`，不会被这个 `except` 捕获，会向上传播。
- **同文件关系**：它调用了模块级常量 `JOB_LABELS`。本文件内部没有任何函数调用它；它是给外部 API/前端层使用的导出函数（在 `__all__` 中列出）。

### `class IngestJobQueue` （第 61 行）

- **作用**：这是本文件的主体类，封装了一个「一句话入库」的后台任务队列。它对外提供提交、查询、重试、列举、等待、启停这一整套任务管理接口，对内负责把任务投递给唯一的后台工作线程并执行真正的向量化与抽取流程。它把持久化（SQLite 中的 `ingest_jobs` 表）与执行（`RAGPipeline.ingest`）粘合在一起，使得「提交立即返回、后台慢慢跑、状态随时可查、进程重启可恢复」这几个目标同时成立。类文档字符串特别说明：当底层存储不可用（例如 `:memory:` 内存测试库）时，实例会处于不可用状态，此时提交会直接报错而不是静默丢弃任务。
- **参数**：类本身不接收参数，参数由 `__init__` 定义。
- **返回**：类，实例化后得到队列对象。
- **内部流程**：实例化时在 `__init__` 中判断存储路径是否可用，决定是否创建 `DocumentRepository`；`start()` 时创建单 worker 的线程池并恢复上次未完成任务；之后所有提交都通过 `_enqueue` 投递到该线程池，由 `_run_job` 串行执行；`shutdown()` 负责收尾并释放资源。
- **异常/边界**：构造阶段不抛异常（即使存储不可用也只是把 `_repo` 置为 `None`）；不可用状态下调用 `submit()` 和 `retry()` 会抛 `RuntimeError`，其余查询类方法返回空结果或 `None`。
- **同文件关系**：它是 `job_to_dict` 所描述的记录的上层管理者；内部依赖模块级常量 `MAX_ATTEMPTS` 和 `JOB_LABELS`（`JOB_LABELS` 实际由 `job_to_dict` 使用）。它定义了 `__init__`、`available`、`start`、`submit`、`job`、`retry`、`list`、`wait`、`shutdown`、`_enqueue`、`_run_job`、`_notify` 这些成员。

### `__init__(self, manager: Any, extractor: Any, *, on_progress: Callable[[str], None] | None = None) -> None` （第 64 行）

- **作用**：构造函数，负责把外部依赖（记忆管理器、抽取器、进度回调）保存下来，并在构造时就把「持久化是否可用」这个判断做完。它通过读取 `manager.document_store.path` 来决定能否建立 `DocumentRepository`：只有真正落到磁盘文件的库才可用，`:memory:` 这种内存库无法跨线程/跨进程共享，因此被显式排除。这一步判断之所以放在构造而不是放到 `start()`，是为了让 `available` 属性能在任何时刻准确回答「这个队列能不能用」，从而让 Web 层可以据此决定是否暴露入库功能。构造时故意不创建线程池，线程池留给 `start()` 建立，这样构造对象本身没有副作用、不会起后台线程。
- **参数**：
  - `manager`：必填，类型标注为 `Any`（实际是记忆管理器对象）。它需要具备 `document_store` 属性（其 `path` 用来判断持久化可用性），并且需要具备 `episodic` 属性（在任务成功时记录情节记忆）。本函数只读它的 `document_store.path`。
  - `extractor`：必填，类型标注为 `Any`。是 LLM 抽取器对象，会原样保存为 `self._extractor`，并在每个任务里传给 `RAGPipeline(..., extractor=...)` 使用。本函数不对它做任何校验。
  - `on_progress`：仅关键字参数，类型 `Callable[[str], None] | None`，默认 `None`。是一个接收单个字符串（`job_id`）参数、返回 `None` 的回调，用于在任务状态变化时通知外部（例如通过 SSE/WebSocket 推送进度）。为 `None` 表示不需要通知。
- **返回**：`None`（构造函数）。
- **内部流程**：先把 `manager`、`extractor`、`on_progress` 分别保存到 `self._manager`、`self._extractor`、`self._on_progress`。然后用 `getattr(manager.document_store, "path", None)` 安全取出存储路径（`document_store` 不存在会抛 `AttributeError`，但 `path` 属性缺失会安全返回 `None`）。接着计算 `usable = bool(path) and str(path) != ":memory:"`，即路径非空且不是内存库标记。若 `usable` 为真，则 `self._repo = DocumentRepository(path)`，否则 `self._repo = None`。注释说明文件库每次调用使用独立连接，因此线程安全。最后把 `self._executor` 初始化为 `None`，注释说明固定单 worker 的原因：后一次入库可能检索/依赖上一次写入的图与向量。
- **异常/边界**：如果 `manager` 没有 `document_store` 属性，`getattr` 的外层访问会抛 `AttributeError`；如果 `DocumentRepository(path)` 构造失败（如路径不可写），异常会直接向上抛。路径为空字符串、`None` 或等于 `":memory:"` 时不会抛错，只是让实例进入不可用状态。回调不做任何校验，错误延迟到 `_notify` 中处理。
- **同文件关系**：它设置的状态被 `available`、`start`、`submit`、`job`、`retry`、`list`、`wait`、`shutdown`、`_enqueue`、`_run_job`、`_notify` 全部读取使用。它本身不调用本文件的其它函数。

### `available` （属性，第 81 行）

- **作用**：这是一个只读属性（用 `@property` 装饰），用来回答「这个队列当前是否可用」。它的判断依据非常简单直接：只要内部的 `DocumentRepository` 存在就说明持久化存储可用，队列能接收任务。Web 层通常会先读这个属性，再决定是启用异步入库还是退化成同步入库或直接提示用户不支持。它之所以做成属性而不是方法，是为了让调用点读起来像一个状态字段。属性是实时的，`shutdown()` 把 `_repo` 置为 `None` 后它会立刻变成 `False`。
- **参数**：无（`self` 除外）。
- **返回**：返回 `bool`。`self._repo is not None` 时为 `True`，否则为 `False`。
- **内部流程**：单行表达式，直接对 `self._repo` 做 `is not None` 判断并返回结果，没有分支、循环或副作用。
- **异常/边界**：不会抛异常。注意如果实例被 `shutdown()` 关闭，`_repo` 已为 `None`，本属性返回 `False`，此时再调用 `submit()`/`retry()` 会抛 `RuntimeError`。
- **同文件关系**：读取 `__init__` 设置的 `self._repo`，与 `shutdown` 修改该字段的行为相互配合。不调用本文件其它函数。

### `start(self) -> None` （第 85 行）

- **作用**：启动后台工作池，并负责「续跑上次退出时未完成的任务」。它是队列从「只存不跑」进入「真正开始消费」的开关，通常在 Web 应用启动时被调用一次。除了创建线程池，它还做崩溃恢复：先让仓储把残留的 `running` 任务复位，再把所有 `pending` 任务重新投递。这样即使上次进程是被强杀的，任务也不会永久卡在 `running` 状态。整个方法对不可用队列是安全的空操作。
- **参数**：无（`self` 除外）。
- **返回**：`None`。
- **内部流程**：第一步判断 `self._repo is None`，若是则直接 `return`，不创建线程池。第二步用 `ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest-job")` 创建单线程池并保存到 `self._executor`，线程名前缀便于日志排查。第三步调用 `self._repo.restart_stale_ingest_jobs()` 得到被复位的任务数量 `recovered`。第四步若 `recovered` 为真值，用 `LOGGER.info` 打印一条中文日志说明恢复了几条。第五步遍历 `self._repo.list_ingest_jobs(status="pending")` 返回的所有任务，对每个任务调用 `self._enqueue(job.job_id)` 重新投递。
- **异常/边界**：`_repo` 为 `None` 时静默返回（不报错）。若仓储的 `restart_stale_ingest_jobs` 或 `list_ingest_jobs` 抛异常（例如数据库损坏），异常会向上传播，线程池已经创建但不会被回滚。重复调用 `start()` 会用新线程池覆盖旧的 `_executor`，旧线程池不会被显式关闭，属于调用方需要注意的边界。
- **同文件关系**：调用 `_enqueue` 投递恢复出的任务；读取 `__init__` 设置的 `_repo`；与 `shutdown` 是一对启停接口（`shutdown` 把 `_executor` 和 `_repo` 置空后，再次 `start()` 将因 `_repo is None` 直接返回）。

### `submit(self, text: str, *, event_at: str = "") -> IngestJobRecord` （第 99 行）

- **作用**：提交一条新的一句话入库任务。它先把任务以 `pending` 状态落库，拿到带 `job_id` 的记录后立即入队，然后把记录原样返回给调用方，让 API 可以马上回显「排队中」而不必等待真正的入库完成。这是整个「提交立即返回」体验的入口。因为记录在入队前就已经写进 SQLite，即使进程在任务执行前崩溃，`start()` 也能凭 `pending` 状态把它捞回来续跑。队列不可用时它不会假装成功，而是明确抛错，避免用户以为内容已经保存。
- **参数**：
  - `text`：必填，类型 `str`。用户输入的原始一句话文本，会原样交给仓储持久化，后续在 `_run_job` 中被包成 `Document` 做向量化。函数本身不做长度或内容校验。
  - `event_at`：仅关键字参数，类型 `str`，默认空字符串 `""`。表示这条知识对应的事件时间（而不是录入时间），会透传给 `create_ingest_job` 并在执行时写入文档元数据 `event_at`。空字符串表示未指定。
- **返回**：返回 `IngestJobRecord`，即刚落库的那条任务记录，包含新生成的 `job_id` 与 `pending` 状态，供 API 即刻回显。调用方可用它的 `job_id` 去 `job()` 或 `wait()` 查询后续状态。
- **内部流程**：第一步检查 `self._repo is None`，若是则抛 `RuntimeError("ingest queue unavailable (no persistent sqlite store)")`。第二步调用 `self._repo.create_ingest_job(text, event_at=event_at)` 把任务写入数据库并得到记录 `job`。第三步调用 `self._enqueue(job.job_id)` 把它投给后台线程池。第四步返回 `job`。
- **异常/边界**：队列不可用时抛 `RuntimeError`（消息为英文）。落库失败（数据库错误）时由仓储抛出的异常向上传播，此时任务不会入队。注意若 `start()` 尚未被调用，`_executor` 为 `None`，`_enqueue` 会静默什么都不做，任务只停在 `pending`，需要后续 `start()` 才能被消费。
- **同文件关系**：调用 `_enqueue`；依赖 `__init__` 建立的 `_repo`。它创建的记录会被 `job`、`wait`、`list` 读取，被 `_run_job` 执行。

### `job(self, job_id: str) -> IngestJobRecord | None` （第 108 行）

- **作用**：按 `job_id` 查询单条任务的最新状态。它是所有状态轮询的基础，`wait()` 内部就是靠反复调用它来观察状态变化，API 层也用它来回答「这条任务现在怎么样了」。它把「队列不可用」这种情况统一处理成返回 `None`，这样调用方不需要先判断 `available` 再查询，简化了上层代码。因为每次查询都是直接从 SQLite 读最新行，所以能立刻反映后台线程刚写入的状态变更。
- **参数**：
  - `job_id`：必填，类型 `str`。任务唯一标识，通常来自 `submit()` 的返回值。若该 id 不存在，函数返回 `None` 而不是抛异常。
- **返回**：返回 `IngestJobRecord | None`。队列可用且找到记录时返回记录对象；`self._repo is None`（未启动或已关闭）时返回 `None`；id 不存在时也返回 `None`（由仓储的 `get_ingest_job` 决定）。调用方无法仅凭返回值区分「不可用」和「不存在」。
- **内部流程**：单行三元表达式：`None if self._repo is None else self._repo.get_ingest_job(job_id)`。没有循环、没有副作用，也不做缓存。
- **异常/边界**：队列不可用时返回 `None`，不抛异常。数据库读取错误由仓储抛出并向上传播。`job_id` 为空字符串等非法值时行为取决于仓储实现，本函数不校验。
- **同文件关系**：被 `wait` 反复调用；`_run_job` 内部不通过它而是直接用 `repo.get_ingest_job` 取记录。它读取 `__init__` 建立的 `_repo`。

### `retry(self, job_id: str) -> IngestJobRecord` （第 111 行）

- **作用**：把一条已经失败的任务重新排队执行。与自动重试不同，这是用户手动触发的重试，所以它会把 `attempts` 计数重置，避免「重试次数超限」这个上限把用户的手动重试也一并卡死——这是文档字符串里明确写出的设计意图。它会先做两道校验：任务必须存在，且当前状态必须是 `failed`，否则拒绝重试，防止对正在跑或已经成功的任务重复执行导致重复写入知识。校验通过后调用仓储复位该任务并重新入队。
- **参数**：
  - `job_id`：必填，类型 `str`。要重试的任务标识。必须对应数据库中存在的一条记录，否则抛 `LookupError`。
- **返回**：返回 `IngestJobRecord`，即复位之后的任务记录（由仓储的 `reset_failed_ingest_job` 返回），通常是状态回到 `pending`、`attempts` 归零、`error` 清空的记录。调用方可用它立即回显。
- **内部流程**：第一步检查 `self._repo is None`，若是则抛 `RuntimeError("ingest queue unavailable (no persistent sqlite store)")`。第二步用 `self._repo.get_ingest_job(job_id)` 取记录，若为 `None` 则抛 `LookupError(job_id)`。第三步判断 `job.status != "failed"`，若成立则抛 `ValueError`，错误消息为中文「只有失败任务可重试，当前状态是 {状态}」。第四步调用 `self._repo.reset_failed_ingest_job(job_id)`，若返回 `None` 说明复位时记录已消失（可能被并发删除），再次抛 `LookupError(job_id)`。第五步调用 `self._enqueue(job_id)` 重新投递。第六步返回复位后的记录 `reset`。
- **异常/边界**：抛三类异常——队列不可用时 `RuntimeError`；任务不存在或复位失败时 `LookupError`；状态不是 `failed` 时 `ValueError`。注意异常类型与 `submit` 保持一致，便于上层统一处理。若 `_executor` 为 `None`（未 `start()`），`_enqueue` 静默不投递，任务会停留在复位后的 `pending` 状态等待下次 `start()`。
- **同文件关系**：调用 `_enqueue`；与 `job` 一样读取 `_repo`；被重试后的任务最终由 `_run_job` 执行，而 `_run_job` 中的 `MAX_ATTEMPTS` 检查正是它要绕开的限制。

### `list(self, *, status: str | None = None, limit: int = 50) -> list[IngestJobRecord]` （第 127 行）

- **作用**：按条件列出任务记录，供管理页面或 API 展示入库历史。它支持按状态过滤和限制条数，方便前端做「只看待处理的」「只看失败的」这类视图。与 `job()` 一样，队列不可用时它统一返回空列表，让上层无需额外判断。它只做透传，不排序、不做内存过滤，具体的排序与筛选语义由仓储层的 `list_ingest_jobs` 决定。
- **参数**：
  - `status`：仅关键字参数，类型 `str | None`，默认 `None`。用于过滤任务状态，取值通常是 `"pending"`、`"running"`、`"done"`、`"failed"` 之一（与 `JOB_LABELS` 的键对应）。传 `None` 表示不过滤、返回所有状态。
  - `limit`：仅关键字参数，类型 `int`，默认 `50`。返回的最大条数。应为正整数；传 `0` 或负数时的行为取决于仓储实现（可能返回空列表或报错），本函数不校验。
- **返回**：返回 `list[IngestJobRecord]`。队列可用时返回仓储查询结果；`self._repo is None` 时返回空列表 `[]`。永远不会返回 `None`。
- **内部流程**：单行三元表达式：`[] if self._repo is None else self._repo.list_ingest_jobs(status=status, limit=limit)`。两个参数都以关键字形式转发给仓储。没有循环或副作用。
- **异常/边界**：队列不可用时返回空列表，不抛异常。数据库错误由仓储抛出并向上传播。非法 `status` 字符串的行为由仓储决定，可能返回空列表。
- **同文件关系**：与 `start` 中的恢复逻辑相似（`start` 也调用 `self._repo.list_ingest_jobs(status="pending")`，但走的是仓储而非本方法，因为 `start` 还需要配合 `_enqueue` 逐个投递）。它读取 `__init__` 建立的 `_repo`。

### `wait(self, job_id: str, timeout: float = 180.0) -> IngestJobRecord | None` （第 130 行）

- **作用**：阻塞等待某条任务进入终态（`done` 或 `failed`），或者等到超时为止。它是给 `wait=true` 这类同步调用方准备的：调用方希望「提交后直接拿到结果」，不想自己写轮询循环。实现方式是每 50 毫秒查一次数据库，因此实现简单、不需要线程间的事件通知机制，代价是有轻微轮询开销和最多 50 毫秒的额外延迟。如果任务在超时前没有结束，它会把当前（可能仍是 `running`/`pending`）的记录返回，由调用方自己判断是否继续等。
- **参数**：
  - `job_id`：必填，类型 `str`。要等待的任务标识。若该 id 根本不存在，第一次查询就得到 `None`，循环条件不成立，函数立即返回 `None`。
  - `timeout`：类型 `float`，默认 `180.0`（3 分钟）。最长等待秒数，内部用 `time.monotonic()` 计算截止时刻，因此不受系统时间被调整的影响。传 `0` 或负数时，若任务尚未终态则立即返回当前记录；传很大的值会长时间占用调用线程。
- **返回**：返回 `IngestJobRecord | None`。任务在超时前到达 `done` 或 `failed` 时返回该终态记录；超时时返回最后一次查询到的记录（状态可能仍是 `pending`/`running`）；`job_id` 不存在或队列不可用时返回 `None`。
- **内部流程**：第一步在函数内部延迟 `import time`（避免模块级导入，属于局部导入而非嵌套函数）。第二步用 `deadline = time.monotonic() + timeout` 计算截止时间。第三步先查一次 `job = self.job(job_id)` 做快速路径。第四步进入 `while` 循环，循环条件同时要求 `job is not None`、`job.status not in ("done", "failed")`、`time.monotonic() < deadline` 三者都成立；循环体内先 `time.sleep(0.05)` 再重新查询 `job = self.job(job_id)`，实现 50 毫秒间隔的轮询。第五步返回最后一次得到的 `job`。
- **异常/边界**：不抛异常（查询异常由 `job`/仓储抛出并传播）。`job_id` 不存在时立即返回 `None`。超时后返回的是中间状态记录，调用方必须自行检查 `status`，不能假定返回值一定是终态。若 `timeout` 为 `None` 会在加法处抛 `TypeError`，本函数不处理。
- **同文件关系**：循环内反复调用本文件的 `job()` 方法；`job()` 又依赖 `__init__` 建立的 `_repo`。它不调用 `_notify`，也不依赖进度回调，纯靠数据库轮询。

### `shutdown(self, *, wait: bool = True) -> None` （第 142 行）

- **作用**：停止后台工作池并释放仓储资源，是应用退出时的清理入口。默认 `wait=True` 会等在跑的任务收尾，这是有意的设计：文档字符串说明 v1.65 之前异步关闭会残留仍在访问已关闭 manager/document store 的后台线程，造成进程退出竞争。所以默认行为优先保证安全收尾，只有在需要立刻退出时才传 `wait=False`；那种情况下未完成的任务会留在 SQLite 里保持 `running`，下次启动由 `start()` 复位续跑，不会丢。关闭完成后实例进入不可用状态。
- **参数**：
  - `wait`：仅关键字参数，类型 `bool`，默认 `True`。`True` 表示阻塞等待线程池中已提交的任务执行完毕；`False` 表示不等待、立即返回（正在执行的任务会被继续跑完但不再等它）。
- **返回**：`None`。
- **内部流程**：第一步判断 `self._executor is not None`，成立则调用 `self._executor.shutdown(wait=wait)` 关闭线程池，随后把 `self._executor = None` 置空，防止重复关闭。第二步判断 `self._repo is not None`，成立则调用 `self._repo.close()` 关闭仓储连接，然后把 `self._repo = None` 置空。这两步都用 `is not None` 保护，因此可以安全地重复调用。
- **异常/边界**：两个资源都不存在时是空操作，不会抛异常。`_repo.close()` 若抛异常，异常向上传播且此时 `_executor` 已被置空、`_repo` 尚未置空（因为置空语句在 `close()` 之后），后续调用 `available` 仍会返回 `True` 但仓储可能已损坏。`wait=False` 时后台线程可能仍在使用 `self._manager`，调用方需自行保证 manager 的生命周期长于这些线程。
- **同文件关系**：与 `start` 是配对的启停接口（`start` 建 `_executor` 与恢复任务，`shutdown` 拆 `_executor` 并关闭 `_repo`）；关闭后 `available` 变为 `False`，`submit`/`retry` 会抛 `RuntimeError`，`job`/`list` 会返回 `None`/空列表。它不调用本文件其它函数。

### `_enqueue(self, job_id: str) -> None` （第 159 行）

- **作用**：内部投递方法，把一个 `job_id` 交给后台线程池执行 `_run_job`。它被设计得非常宽容：如果线程池还不存在（`start()` 尚未调用或已经 `shutdown()`），它什么都不做而不是报错，这样 `submit()`、`retry()`、`start()` 三条路径都可以无条件调用它而不必各自判断状态。之所以单独抽出一个方法，是为了让「投递」这个动作在所有入口处保持完全一致的行为。
- **参数**：
  - `job_id`：必填，类型 `str`。要执行的任务标识。它不做存在性校验，任务是否存在由 `_run_job` 里再查一次数据库确认。
- **返回**：`None`。
- **内部流程**：单层 `if self._executor is not None:` 判断，成立时调用 `self._executor.submit(self._run_job, job_id)`，把方法本身作为可调用对象和参数一起提交给线程池。由于线程池 `max_workers=1`，所有任务天然按提交顺序串行执行。
- **异常/边界**：`_executor` 为 `None` 时静默丢弃投递（任务仍留在数据库的 `pending` 状态，可被后续 `start()` 恢复）。线程池已关闭时 `submit` 会抛 `RuntimeError`，但本文件在 `shutdown` 中会把 `_executor` 置为 `None`，因此正常流程不会命中这个分支。
- **同文件关系**：被 `start`（恢复遗留任务）、`submit`（新任务）、`retry`（手动重试）三处调用；它负责调用本文件的 `_run_job`。

### `_run_job(self, job_id: str) -> None` （第 163 行）

- **作用**：这是后台线程里真正执行入库的核心方法，承担了状态机推进、向量化与抽取调用、结果汇总、情节记忆写入、失败落库和进度通知的全部工作。它的执行顺序是：确认任务有效 → 检查尝试次数上限 → 标记 `running` → 构造独立 pipeline 并入库 → 根据报告里的 errors 判定成功或失败 → 成功时额外记录一条情节记忆 → 写终态 → 通知回调。整个方法用 `try/except Exception` 兜底，保证任何异常都会把任务标记为 `failed` 而不是让状态永远卡在 `running`。文档字符串之外的关键设计是「每任务独立 pipeline」，这样串行执行时 `last_ingest_report` 不会在任务之间串味。
- **参数**：
  - `job_id`：必填，类型 `str`。由 `_enqueue` 通过线程池传入的任务标识。方法内部会重新从数据库读取该记录，因此传入的 id 必须已经落库。
- **返回**：`None`。所有结果都通过数据库状态和进度回调体现，不返回值。
- **内部流程**：第一步把 `self._repo` 取到局部变量 `repo`（避免执行过程中被 `shutdown` 置空导致不一致），若为 `None` 直接 `return`。第二步用 `repo.get_ingest_job(job_id)` 取记录，若为 `None` 或 `job.status == "done"` 则直接返回（防止重复执行已完成任务）。第三步检查 `job.attempts >= MAX_ATTEMPTS`，成立则调用 `repo.set_ingest_job_status(job_id, "failed", error="重试次数超限")`，调用 `self._notify(job_id)` 后返回。第四步调用 `repo.set_ingest_job_status(job_id, "running")` 标记为执行中。第五步进入 `try`：先 `RAGPipeline(self._manager, extractor=self._extractor)` 构造每任务独立的 pipeline；再用 `job.text.strip().splitlines()[0][:40] if job.text.strip() else "一句话入库"` 计算 `preview`（取第一行前 40 字符，空文本时回落为固定文案）；接着调用 `pipeline.ingest(Document(job.text, metadata={...}))`，元数据里包含 `source`（固定为「一句话入库」）、`filename`（即 `preview`）、`note`（`job.text[:400]`）、`event_at`（任务的事件时间）、`reference_time`（`datetime.now(UTC).isoformat()` 生成的当前 UTC 时间）、`ingest_job_id`，返回值存到 `items`。第六步取 `report = pipeline.last_ingest_report or {}`，用 `json.dumps({"chunks": len(items), "report": report}, ensure_ascii=False)` 生成 `summary` 摘要字符串。第七步取 `errors = report.get("errors") or []`，若非空则把任务标记为 `failed`，错误信息取 `str(errors[0])`，同时把 `summary` 存进 `result`。第八步若没有错误，则调用 `self._manager.episodic.record(...)` 写入一条情节记忆，内容为 `f"添加了一条知识：{job.text[:80]}"`，元数据含 `title`、`source`、`ingest_job_id`，然后把任务标记为 `done` 并写入 `result=summary`。第九步 `except Exception as exc` 分支用 `LOGGER.exception("入库任务 %s 失败", job_id)` 记录完整堆栈，并把任务标记为 `failed`，错误文本格式为 `f"{type(exc).__name__}: {exc}"`（不含 result）。最后无论成功失败都执行一次 `self._notify(job_id)`。
- **异常/边界**：`repo` 为 `None` 时静默返回；任务不存在或已 `done` 时静默返回；尝试次数超限时标记 `failed` 并写「重试次数超限」。`try/except Exception` 捕获所有入库相关异常（包括 pipeline 构造、向量化、LLM 抽取、情节记忆写入失败），统一转为 `failed` 状态并把异常类名与消息写进 `error` 字段。注意 `repo.set_ingest_job_status` 本身若抛异常不在保护范围内（`except` 块内的这次写入也可能抛出并逃逸），`_notify` 在方法末尾调用，若它抛异常会被其内部自行吞掉。`job.text` 为空字符串时 `preview` 回落为「一句话入库」，不会因 `splitlines()[0]` 索引越界而崩溃。
- **同文件关系**：被 `_enqueue` 投递执行；调用 `self._notify`；使用模块级常量 `MAX_ATTEMPTS`；使用 `__init__` 保存的 `_manager` 与 `_extractor`。它是本文件唯一真正触发 `RAGPipeline.ingest` 的地方。

### `_notify(self, job_id: str) -> None` （第 212 行）

- **作用**：内部回调触发方法，在任务状态发生变化后通知外部（例如向前端推送进度）。它把「回调可能出错」这件事完全隔离在队列内部：外部传入的 `on_progress` 无论是抛异常还是本身为 `None`，都不会影响入库任务的状态机推进，最多留下一条警告日志。这样设计是因为进度通知属于尽力而为的增强功能，绝不能让推送失败反过来把已经成功的入库标记成失败。
- **参数**：
  - `job_id`：必填，类型 `str`。刚发生状态变化的任务标识，会原样传给回调，回调据此自行去数据库查询最新状态并推送。
- **返回**：`None`。
- **内部流程**：第一步判断 `self._on_progress is None`，若是则直接 `return`，不产生任何日志。第二步在 `try` 中调用 `self._on_progress(job_id)`。第三步用 `except Exception` 捕获回调抛出的任何异常，并调用 `LOGGER.warning("入库进度回调失败", exc_info=True)` 记录警告与堆栈，然后正常返回。
- **异常/边界**：不向外抛任何异常。回调为 `None` 时静默跳过；回调抛异常时降级为警告日志。`job_id` 本身不做校验，是否有效由回调自行判断。
- **同文件关系**：被 `_run_job` 在多个分支调用（尝试次数超限、标记 `failed`、标记 `done`，以及方法末尾的统一调用）；读取 `__init__` 保存的 `self._on_progress`。它不调用本文件其它函数。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `job_to_dict` | 把一条入库任务记录转成带中文状态文案、文本预览和可重试标记的 JSON 友好字典。 |
| `IngestJobQueue` | 基于 SQLite 持久化状态与单线程串行消费的后台一句话入库队列。 |
| `IngestJobQueue.__init__` | 保存 manager/extractor/进度回调，并按存储路径是否可用决定是否建立仓储。 |
| `IngestJobQueue.available` | 只读属性，判断队列是否有可用的持久化存储。 |
| `IngestJobQueue.start` | 创建单 worker 线程池，复位遗留 running 任务并重新投递所有 pending 任务。 |
| `IngestJobQueue.submit` | 落库一条 pending 任务并立即入队，返回记录供 API 即刻回显。 |
| `IngestJobQueue.job` | 按 job_id 查询单条任务的最新记录，不可用或不存在时返回 None。 |
| `IngestJobQueue.retry` | 校验任务存在且为 failed 后复位 attempts 并重新入队，支持用户手动重试。 |
| `IngestJobQueue.list` | 按状态过滤并限制条数地列出任务记录，不可用时返回空列表。 |
| `IngestJobQueue.wait` | 以 50 毫秒轮询阻塞等待任务到达 done/failed 或超时。 |
| `IngestJobQueue.shutdown` | 关闭线程池并释放仓储，默认等待在跑任务收尾以避免退出竞争。 |
| `IngestJobQueue._enqueue` | 把 job_id 投递给单线程池执行，线程池不存在时静默跳过。 |
| `IngestJobQueue._run_job` | 后台执行入库全流程：状态推进、RAGPipeline 入库、结果汇总、情节记忆与失败落库。 |
| `IngestJobQueue._notify` | 安全地触发进度回调，回调缺失或抛异常时只记录警告。 |
