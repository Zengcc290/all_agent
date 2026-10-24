# tool/_shared.py

## 一、这个文件是干什么的

这个文件是 `tool/` 目录下所有「面向文件系统」的工具（读文件、写文件、列目录、搜索等）共用的**路径安全基础设施**，只做一件事：把用户/模型给过来的路径字符串，安全地解析成一个绝对的 `pathlib.Path`，并判断它是否逃出了工作区（workspace）边界。它对外提供四个函数：`workspace_root()` 负责回答「当前工作区根目录在哪」，`allow_outside_workspace()` 负责回答「本次部署是否允许越界访问」，`resolve_path()` 是把两者组合起来的核心入口（做类型校验、相对路径拼接、`..` 归一化、越界拦截），`_is_within()` 则是纯粹的字符串前缀比较工具，供 `resolve_path()` 做包含性判断。模块顶部定义了两个环境变量名常量 `WORKSPACE_ROOT_ENV = "WORKSPACE_ROOT"` 与 `ALLOW_OUTSIDE_ENV = "WORKSPACE_ALLOW_OUTSIDE"`，分别用来指定工作区根目录和是否放开越界限制。文件头部的 docstring 明确说明了它的两个设计约束：其一，它**故意不可被发现**——`core.discovery` 会忽略以 `_` 开头的模块名，且本文件既不定义 `TOOL_ENABLED` 也不定义 `create_tool()` 工厂，所以它永远不会被当成一个可注册的工具；其二，它**只有标准库依赖**（`os`、`pathlib`），导入时不做网络请求、不写文件、不启动子进程，因此可以放心地被任何工具模块在导入阶段直接引用。在实际运行中，每个文件类工具在处理一次调用时，都会先调用 `resolve_path()` 拿到经过校验的绝对路径，再拿这个路径去真正读/写磁盘，从而保证 Agent 无法通过 `../../etc/passwd` 这类输入突破沙箱。它同时兼顾了跨平台（Windows 大小写不敏感）与「部署者决策优先于模型输入」的安全原则：是否允许越界由环境变量或调用方显式参数决定，模型无法通过输入字段自行解锁。

## 二、函数与类逐条详解

### `workspace_root() -> Path` （第 19 行）

- **作用**：回答「当前这次运行的工作区根目录到底是哪个目录」这个问题，是整个模块乃至所有文件类工具的路径基准点。它的取值策略是「环境变量优先、进程当前目录兜底」：如果部署方设置了 `WORKSPACE_ROOT`，就用它；否则退化为进程启动时的当前工作目录（CWD）。之所以需要这样一个函数而不是让各处直接写 `Path.cwd()`，是因为同一个进程在不同场景下（例如 Web 应用被从别的目录启动、或者被测试用例切换工作区）需要能够通过环境变量把「工作区」重新指向别处，而不必改动代码。它被调用的典型时机是：某个文件工具在创建实例或处理请求时，需要确定「默认允许访问的根目录」；也常常与 `allow_outside_workspace()` 配合，一起交给 `resolve_path()` 作为 `base_dir` 使用。由于它每次调用都重新读取环境变量并重新 `resolve()`，所以运行期修改环境变量是会即时生效的，不存在缓存导致的陈旧值问题。
- **参数**：无参数。
- **返回**：返回一个 `pathlib.Path` 对象，且保证是**绝对路径**（内部调用了 `.resolve()`）。当环境变量 `WORKSPACE_ROOT` 存在且去除首尾空白后非空时，返回该值的路径形式；否则返回 `Path.cwd().resolve()`，即进程当前工作目录的绝对、已归一化路径。返回值永远是同一个类型，不会返回 `None`。
- **内部流程**：第一步，`raw = os.getenv(WORKSPACE_ROOT_ENV)` 读取环境变量（键名来自模块级常量 `"WORKSPACE_ROOT"`），未设置时得到 `None`。第二步，判断 `raw is not None and raw.strip()`：既要求变量存在，也要求它不是纯空白字符串（例如 `"   "` 会被视为未设置），这是一个很实用的健壮性处理。第三步，条件成立时执行 `Path(raw.strip()).expanduser().resolve()`：先去首尾空白，再用 `expanduser()` 把 `~` 展开成用户主目录，最后 `resolve()` 把相对路径转成绝对路径并消解 `.`、`..` 和符号链接。第四步，条件不成立时执行 `Path.cwd().resolve()`，取进程当前目录并同样做绝对化与归一化。
- **异常/边界**：环境变量未设置、为空串、或只有空白字符时，都不会报错，而是安静地回退到 CWD。`Path.cwd()` 在极端情况下（进程当前目录已被删除或不可访问）可能抛出 `FileNotFoundError` 或 `OSError`；`resolve()` 在遇到循环符号链接时在较新的 Python 上会抛 `OSError`（旧版本为 `RuntimeError`），遇到权限不足的父目录也可能抛 `OSError`/`PermissionError`。这些异常本函数都不捕获，会直接向上传播给调用方处理。函数本身不做任何越界判断——它只回答「根在哪」，不负责「能不能越界」。
- **同文件关系**：它调用了本文件外的标准库 `os.getenv` 与 `pathlib.Path`，不调用本文件内的其它函数（它是依赖链的最底层之一）。它被本文件内的 `resolve_path()` 间接依赖（调用方通常把它的返回值当作 `resolve_path()` 的 `base_dir`），但文件内部没有直接的函数调用语句；`allow_outside_workspace()` 与它并列，共同构成 `resolve_path()` 的两个策略输入。

### `allow_outside_workspace() -> bool` （第 27 行）

- **作用**：回答「本次部署是否允许工具触碰工作区根目录之外的路径」这个问题，也就是沙箱是否放开的开关。它是**部署方（运维/开发者）的决策**，而不是模型的决策：模型无论怎么构造输入，都无法通过这个函数改变结果，因为它唯一的输入来源是进程环境变量。默认行为是**拒绝**——只要 `WORKSPACE_ALLOW_OUTSIDE` 没有设置，就返回 `False`，意味着所有越界路径都会被 `resolve_path()` 拒绝。只有在明确把该变量设成真值（如 `1`、`true`、`yes`、`on`）时才返回 `True`，此时越界路径会被放行。它被调用的典型时机是在 `resolve_path()` 里，当调用方没有显式传入 `allow_outside` 参数时作为兜底策略；也可能被文件工具在启动自检、生成工具描述或写日志时单独调用，以便对外声明当前沙箱强度。
- **参数**：无参数。
- **返回**：返回布尔值。当环境变量 `WORKSPACE_ALLOW_OUTSIDE` 根本不存在（`os.getenv` 返回 `None`）时返回 `False`；当它存在时，把值 `strip()` 去空白并 `casefold()` 转小写后，判断是否落在集合 `{"1", "true", "yes", "on"}` 中，在则返回 `True`，不在则返回 `False`。注意：空字符串、`"0"`、`"false"`、`"no"`、`"off"`、`"maybe"` 以及任何其它随机字符串都会返回 `False`，即「非真值一律视为禁止」，这是一个安全默认（fail-closed）的设计。
- **内部流程**：第一步，`raw = os.getenv(ALLOW_OUTSIDE_ENV)` 读取环境变量（键名来自模块级常量 `"WORKSPACE_ALLOW_OUTSIDE"`）。第二步，显式判断 `raw is None`，是则直接 `return False`——这一步与后面分支分开写，是为了让「变量未设置」与「变量设置了但不认识」在语义上清晰，虽然二者返回值相同。第三步，对非 `None` 的值执行 `raw.strip().casefold()`：`strip()` 容忍 `" true "` 这类带空白的写法，`casefold()` 做比 `lower()` 更彻底的 Unicode 大小写折叠，从而让 `"TRUE"`、`"True"`、`"tRuE"` 都能识别。第四步，用 `in {"1", "true", "yes", "on"}` 做成员测试并返回结果。
- **异常/边界**：函数内部不抛任何异常（`os.getenv` 对任意键都安全返回 `None` 或字符串）。边界情况包括：值为纯空白 `"   "` → `strip()` 后为空串 → 不在集合中 → 返回 `False`；值为 `"yes!"` 等带额外字符的 → 返回 `False`（只接受精确匹配的四种写法）；值非字符串（环境变量在 `os.environ` 中始终是字符串，故实际不会发生）也无特殊处理。整个函数是纯读取、无副作用。
- **同文件关系**：不调用本文件内的任何函数，只调用标准库 `os.getenv` 与字符串方法。它被本文件内的 `resolve_path()` 调用（当 `allow_outside` 参数为 `None` 时的兜底判定）；`workspace_root()` 与它没有直接调用关系，二者是并列的两个策略读取函数。

### `resolve_path(base_dir: str | Path, path: str, *, allow_outside: bool | None = None) -> Path` （第 35 行）

- **作用**：这是本文件的核心函数，也是所有文件系统工具真正会调用的入口。它把「基准目录」和「用户给出的路径字符串」合成一个经过校验的绝对路径：相对路径挂到 `base_dir` 下面，绝对路径也必须在 `base_dir` 之内（除非越界被显式放开），任何试图用 `..`、符号链接、绝对路径等方式逃出工作区的行为都会被拦截并抛出 `ValueError`。它存在的意义是给上层工具提供一个**单点、不可绕过的路径规范化与边界检查**：工具作者只要在真正读写磁盘前调用它一次，就不必自己重复实现拼接与越界判断，也避免了各处实现不一致导致的安全漏洞。docstring 特别强调 `allow_outside` 是「工具所有者做出的部署决策，绝不来自 LLM 的输入字段」，也就是说上层工具不允许把模型传入的布尔值直接转交给这个参数，从而杜绝模型自我提权。典型调用时机是每一次工具调用（每次读/写/列目录）开始时。
- **参数**：
  - `base_dir`（位置参数，类型 `str | Path`）：路径基准目录，也就是「允许访问的根」。可以是字符串或 `pathlib.Path`。内部会做 `expanduser()` 与 `resolve()`，所以传相对路径或带 `~` 的路径也能工作。若传入既不是 `str` 也不是 `Path` 的对象（例如 `None`、`int`、`list`），直接抛 `TypeError`。
  - `path`（位置参数，类型 `str`）：待解析的目标路径。必须是**非空字符串**；传入非字符串或去除空白后为空的字符串会抛 `ValueError`。它可以是相对路径（如 `"data/a.txt"`、`"../x"`）也可以是绝对路径（如 `"C:\\tmp\\x"`、`"/etc/hosts"`），绝对路径在 `allow_outside` 为假时会被限制在 `base_dir` 内。
  - `allow_outside`（仅关键字参数，类型 `bool | None`，默认 `None`）：是否允许结果落在 `base_dir` 之外。`None` 表示「未表态」，此时函数会调用 `allow_outside_workspace()` 读取环境变量决定；`True` 表示强制放行越界；`False` 表示强制禁止越界（即使环境变量打开了也禁止，因为后续判断是 `if not allow_outside and ...`，显式 `False` 会走拦截分支）。由于它是 keyword-only（参数列表里有 `*`），调用时必须写成 `allow_outside=True`，不能按位置传第三个参数。
- **返回**：返回一个 `pathlib.Path` 对象，保证是**绝对路径且已归一化**（所有 `.`、`..` 已消解，符号链接在 `resolve()` 默认行为下也被展开）。返回的路径在 `allow_outside` 为假时一定位于 `base_dir` 之内；为真时可能位于任何位置。任何情况下都不会返回 `None`，也从不返回原始未处理的字符串。
- **内部流程**：第一步，类型校验 `if not isinstance(base_dir, (str, Path))`，不合法立刻 `raise TypeError("base_dir must be a string or pathlib.Path")`。第二步，校验 `path`：`if not isinstance(path, str) or not path.strip()`，非字符串或空白串则 `raise ValueError("path must be a non-empty string")`。第三步，`base = Path(base_dir).expanduser().resolve()` 把基准目录规范化成绝对路径（注意这里仍使用原始的 `base_dir`，并没有用 `path`）。第四步，`candidate = Path(path).expanduser()` 先把目标路径构造成 `Path` 并展开 `~`，此时**先不做 resolve**。第五步，判断 `if not candidate.is_absolute()`：若是相对路径，则执行 `candidate = base / candidate`，即把相对路径锚定在基准目录下（用 `/` 运算符拼接，等价于 `os.path.join` 的语义但返回 `Path`）。第六步，`resolved = candidate.resolve()` 做最终归一化，这一步是关键——它会消解 `..` 与符号链接，因此 `base / "../secret"` 会被折叠成 `base` 的父目录，从而在下一步被检测出来。第七步，`if allow_outside is None: allow_outside = allow_outside_workspace()`，即参数未指定时回退到环境变量策略。第八步，`if not allow_outside and not _is_within(base, resolved):` 抛出 `ValueError`，错误消息形如 `path '<原始 path>' resolves outside the workspace root '<base>'`，注意消息里回显的是调用方传入的**原始字符串**和规范化后的**基准目录**，便于排查是哪次调用越界。第九步，通过检查后 `return resolved`。
- **异常/边界**：会主动抛出的异常有三类：`TypeError`（`base_dir` 类型错误）、`ValueError`（`path` 非字符串/空白、或路径越界）。此外，`resolve()` 与 `Path()` 在底层可能抛出 `OSError` 家族异常：路径过长（Windows 上超长路径）会抛 `OSError`；符号链接成环时新版本 Python 抛 `OSError`；父目录无权限时可能抛 `PermissionError`；路径中含 Windows 非法字符（如 `\0`）时抛 `ValueError`。这些都不被捕获，直接向上传播。边界语义要点：`allow_outside` 显式传 `False` 会覆盖环境变量、强制禁止越界；显式传 `True` 则完全跳过包含性检查（连 `_is_within` 都不会被调用）；`path` 恰好等于 `base_dir` 时 `_is_within` 返回真、被放行；`path` 为 `""` 或 `"   "` 会被拒绝而不是当作当前目录；`path` 为 `"."` 是合法非空字符串，会被解析为 `base` 本身并放行。函数自身不做文件存在性检查，也不创建任何目录，因此解析一个不存在的路径不会报错。
- **同文件关系**：它调用了本文件内的 `allow_outside_workspace()`（在 `allow_outside is None` 时获取默认策略）与 `_is_within()`（做最终的包含性判定）；它不调用 `workspace_root()`，但 `workspace_root()` 的返回值通常是上层工具传入的 `base_dir`，所以二者在运行时通过调用方间接协作。它是本文件唯一的公开主入口，被 `tool/` 下各个文件系统工具在每次操作前调用；`_is_within()` 则是它专用的私有辅助函数。

### `_is_within(base: Path, candidate: Path) -> bool` （第 67 行）

- **作用**：判断 `candidate` 是否等于 `base` 或位于 `base` 的目录树之下，是 `resolve_path()` 实现「工作区包含性检查」的底层判定函数。它不用 `Path.relative_to()` 或 `Path.is_relative_to()`，而是把两个路径都转成字符串后做规范化与前缀比较，这样做的好处是：一是不依赖较新 Python 才有的 `is_relative_to()` API，二是显式处理了「路径分隔符结尾」的问题（必须匹配 `base + os.sep`，否则 `/tmp/foobar` 会被误判为在 `/tmp/foo` 之内）。它被设计为私有（下划线开头），因为它是实现细节，外部工具不应该直接依赖这种字符串前缀语义。它在 `resolve_path()` 里只被调用一次，且仅在 `allow_outside` 为假时才会执行。
- **参数**：
  - `base`（位置参数，类型 `Path`）：基准目录，期望是已经 `resolve()` 过的绝对路径。函数不对它做绝对化处理，所以调用方必须自行保证（`resolve_path()` 中传入的是 `base = Path(base_dir).expanduser().resolve()`）。
  - `candidate`（位置参数，类型 `Path`）：待判定的候选路径，同样期望已被 `resolve()` 归一化（`resolve_path()` 中传入的是 `resolved`），否则 `..` 或符号链接可能造成误判。
- **返回**：返回布尔值。当两个路径在 `os.path.normcase` 规范化后完全相等时返回 `True`（即 candidate 就是 base 自身）；当 candidate 的规范化字符串以 `base` 的规范化字符串加上 `os.sep` 为前缀时返回 `True`（即 candidate 是 base 的子路径）；其余情况返回 `False`。在 Windows 上 `os.path.normcase` 会把反斜杠统一为正斜杠并转小写，所以 `C:\Work\A` 与 `c:\work\a\b.txt` 能正确匹配；在 POSIX 上 `normcase` 是恒等函数，大小写敏感，所以 `/tmp/Foo` 与 `/tmp/foo` 被视为不同路径。
- **内部流程**：第一步，`base_text = os.path.normcase(str(base))`，把 `Path` 转成字符串再按平台规范化大小写与分隔符。第二步，`candidate_text = os.path.normcase(str(candidate))` 对候选路径做同样处理。第三步，返回 `candidate_text == base_text or candidate_text.startswith(base_text + os.sep)`：先判断是否相等（覆盖「访问工作区根目录本身」这一合法情况，因为此时拼接后的前缀 `base + os.sep` 并不匹配 base 自己），再用 `startswith` 判断是否为子路径。注意前缀里显式加上 `os.sep`，这是防止 `/a/bc` 被当作 `/a/b` 的子路径的关键一步。
- **异常/边界**：本身不抛异常（`str()` 与 `os.path.normcase` 对任意字符串都安全）。边界语义：`base` 与 `candidate` 完全相同时返回 `True`；`candidate` 是 `base` 的父目录时返回 `False`；`candidate` 是与 `base` 同级的、名字以 base 名称为前缀的目录（如 base=`/a/b`、candidate=`/a/bc`）时返回 `False`，因为缺少分隔符；若 `base` 以分隔符结尾（例如手工传入 `Path("/tmp/foo/")`，其 `str()` 在多数平台会保留尾斜杠），拼接出的前缀会变成 `/tmp/foo//`，可能导致本应命中的子路径被判为不包含——但 `resolve_path()` 传入的 `base` 是 `resolve()` 的结果，通常不会带尾分隔符，所以实际使用中不会触发该问题；`base` 为根目录 `/` 时 `str()` 为 `/`，前缀为 `//`，在 POSIX 上会导致除了 candidate 恰为 `/` 之外全部返回 `False`，这是一个理论边界。函数无副作用、不做 IO。
- **同文件关系**：不调用本文件内的任何其它函数，只使用标准库 `os.path.normcase`、`os.sep` 与字符串方法。它被本文件内的 `resolve_path()` 调用（在 `not allow_outside` 分支里做包含性判定），是本文件调用链的最末端；没有任何本文件内的函数调用它之外的用途。

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `workspace_root()` | 返回配置的工作区根目录，未配置时回退到进程当前工作目录的绝对路径。 |
| `allow_outside_workspace()` | 读取 `WORKSPACE_ALLOW_OUTSIDE` 环境变量，判断本次部署是否允许访问工作区之外的路径，默认禁止。 |
| `resolve_path()` | 把目标路径相对基准目录解析成绝对路径并强制工作区包含性校验，越界即抛 `ValueError`。 |
| `_is_within()` | 用平台规范化后的字符串前缀比较，判断候选路径是否等于基准目录或位于其子树之下。 |
