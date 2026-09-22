# agents/providers.py

## 一、这个文件是干什么的

这个文件负责管理「命名化的 OpenAI 兼容 provider 配置档案（provider profile）」。它把「用哪个大模型服务、走哪个 API 地址、用哪个密钥、默认模型是什么、支持哪些模型、工具调用模式怎么走」这些信息，从 `config/provider.toml` 这类 TOML 文件里读出来，校验成结构化的、不可变的对象，然后对外提供一个注册表让运行时的其它部分按名字取用。

它的核心设计目标是「每个 profile 自包含」：每个 profile 的 TOML 条目里自带 API URL 和 key，所以切换 provider 时不需要去改进程的环境变量。真实使用的 `provider.toml` 被 Git 忽略，仓库里发布的是安全模板 `provider.example.toml`。

文件里主要包含四类东西：一是模块级的正则与常量（profile 名合法字符、环境变量名合法字符、允许的 tool_mode 集合、dotenv 是否已加载的标记）；二是 `load_project_dotenv()`，把仓库根目录 `.env` 里的键值对灌进 `os.environ`（不覆盖已存在的变量），这样 `provider.toml` 里写的 `api_key_env = "DEEPSEEK_API_KEY"` 才能真正取到值；三是两个数据类/类——冻结 dataclass `ProviderProfile`（单个档案的不可变快照，并提供脱敏的 `public_info()`）与 `ProviderRegistry`（负责定位配置文件、加载 TOML、校验、按名字查询、解析出真实密钥）；四是一组模块级私有辅助函数 `_parse_document`、`_parse_profile`、`_required_string`、`_optional_string`、`_validate_model`，承担真正的解析与校验工作。

它在项目里的典型用法是：启动阶段（例如 `python -m web.app` 初始化 Agent 运行时）构造一个 `ProviderRegistry`，拿到 `active_profile` 与 `profiles`；需要调用 LLM 时按名字 `get()` 出 `ProviderProfile` 以取得 `base_url`、`default_model`、`models`、`tool_mode`、`adapter`；需要发请求前调用 `resolve_api_key()` 取真实密钥；Web 层展示配置时用 `public_info()`，它把密钥替换成 `***`，避免把秘密泄露到接口响应或日志里。

需要特别注意的是本文件的错误策略：配置文件缺失抛 `FileNotFoundError`，TOML 语法错误被转换成 `ValueError`，字段类型/取值非法抛 `TypeError` 或 `ValueError`，而「profile 不存在」「密钥解析不出来」这类调用期问题统一抛 `ValueError`，并且错误信息里会带上可用 profile 列表或所缺的环境变量名，便于排查。

---

## 二、函数与类逐条详解

### `load_project_dotenv(path: Path | None = None) -> None` （第 27 行）

- **作用**：把仓库根目录下的 `.env` 文件内容加载进进程环境变量 `os.environ`，让后续 `os.getenv()` 能读到它们。之所以需要这个函数，是因为 `provider.toml` 通常把 `api_key_env` 指向 `DEEPSEEK_API_KEY` 这类变量名，而真正的值写在 `.env` 里；文件注释明确说明，之前的抽取流程会跳过这个文件，导致 ingest 静默地退化成 `NullKnowledgeExtractor` 并写入零条实体。它默认带「只加载一次」的缓存语义，因此可以在多个调用点放心重复调用而不会反复读盘。参数 `path` 非空时则视为显式指定文件，走不带缓存的路径，常用于测试或指定非标准位置。它返回 `None`，效果完全体现在对 `os.environ` 的副作用上。
- **参数**：
  - `path: Path | None = None`：`.env` 文件的路径。默认 `None`，表示使用仓库根目录（即本文件所在目录 `agents/` 的上一级）下的 `.env`，并且启用「只加载一次」的缓存；传入具体 `Path` 时按显式模式处理，每次都真正读取该文件，且不读取也不修改 `_DOTENV_LOADED` 标记。
- **返回**：始终返回 `None`。成功时表现为 `os.environ` 被补充了若干键值对；文件不存在或读不出来时静默返回，不产生任何变化。
- **内部流程**：
  1. 声明 `global _DOTENV_LOADED`，因为要在函数内修改模块级标记。
  2. 用 `explicit = path is not None` 判断是否为显式调用。
  3. 非显式模式下：若 `_DOTENV_LOADED` 已为真则直接 `return`（快速短路）；否则立刻把 `_DOTENV_LOADED` 置为 `True`（注意是「先置位再尝试读取」，所以即使首次读取失败，后续默认调用也不会再试），再用 `Path(__file__).resolve().parent.parent / ".env"` 定位仓库根的 `.env`。
  4. 若目标路径不是文件（`path.is_file()` 为假）则 `return`。
  5. 用 `path.read_text(encoding="utf-8")` 读取全文；捕获 `OSError`（权限、编码之外的 IO 问题等）后静默 `return`。
  6. 逐行遍历 `text.splitlines()`：对每行先 `strip()`；跳过空行、以 `#` 开头的注释行、以及不含 `=` 的行。
  7. 若行以 `export ` 开头，则去掉前 7 个字符再 `strip()`，兼容 shell 风格的 `.env`。
  8. 用 `line.partition("=")` 拆成 `key`、`_`（分隔符，丢弃）、`value`，并对 `key` 做 `strip()`。
  9. 若 `key` 为空，或 `key` 已存在于 `os.environ`（`key in os.environ`），则跳过——这就是「不覆盖运行中变量」的策略，保证真实环境变量优先于 `.env`。
  10. 对 `value` 做 `strip()`；若长度不小于 2 且首尾字符相同且该字符是单引号或双引号，则剥掉这对引号。
  11. 最后执行 `os.environ[key] = value` 写入。
- **异常/边界**：路径不是文件（不存在、是目录）→ 直接返回，不抛异常；读取时 `OSError` → 捕获后返回；空行、注释行、无 `=` 的行、空 key → 跳过；重复键不会覆盖已有环境变量；带引号的值会去引号；不支持多行值、变量插值、转义序列等高级 dotenv 语法。显式传入 `path` 时不会设置 `_DOTENV_LOADED`，也不会被缓存短路，因此每次都会真实读盘。函数本身不主动抛异常。
- **同文件关系**：只被 `ProviderRegistry.resolve_api_key()` 调用（在取密钥前先确保 `.env` 已加载）；它自身不调用本文件中的其它函数。

---

### `class ProviderProfile` （第 64 行）

- **作用**：这是一个用 `@dataclass(frozen=True)` 声明的不可变数据类，表示「一个已校验通过的、命名的 OpenAI 兼容 provider 配置」。它是本模块对外交付的核心数据类型：注册表里存的是它，`get()` 返回的是它，运行时拿它去读 `base_url`、`default_model`、`models`、`tool_mode`、`adapter`。之所以冻结（frozen），是为了让配置在加载后不会被意外改写，从而让「同一个 profile 的 base_url 与 models 在整个进程生命周期内保持一致」这一假设成立，也让 `MappingProxyType` 包装的注册表真正做到只读。类本身不包含加载或校验逻辑，那些都在 `_parse_profile()` 里完成，本类只负责承载结果并提供脱敏展示。
- **字段**：
  - `name: str`：profile 的名字，即 `[profiles.<name>]` 里的 `<name>`，同时是注册表字典的键。
  - `base_url: str`：API 基础地址，已经去掉结尾的 `/`；同时以 `api_url` 的名义对外暴露（见 `public_info()`）。
  - `api_key: str`：直接内联写在配置里的密钥；若走环境变量方式则为空字符串。
  - `default_model: str`：默认使用的模型名，必须出现在 `models` 中。
  - `models: tuple[str, ...]`：该 profile 声明支持的模型名元组，非空且无重复。
  - `api_key_env: str | None = None`：密钥所在的环境变量名；为 `None` 表示不使用环境变量方式。
  - `adapter: str = "openai_compatible"`：适配器类型，目前只支持 `openai_compatible`。
  - `tool_mode: str = "native_strict"`：工具调用模式，取值范围限定在 `native_strict`、`native_loose`、`text_react`、`none` 之一。
- **返回**：不适用（类定义）。
- **内部流程**：由 dataclass 机制自动生成 `__init__`、`__repr__`、`__eq__` 等方法；因为 `frozen=True`，实例创建后给字段赋值会抛 `FrozenInstanceError`，同时 dataclass 会据字段生成 `__hash__`，使实例可哈希。校验与默认值填充不在这里，而在 `_parse_profile()` 构造实例之前完成。
- **异常/边界**：作为数据类本身不抛业务异常；对实例字段赋值会因冻结而失败；`models` 期望是 `tuple`，若被直接构造时传入 `list`，类型注解不会在运行时强制，但后续 `public_info()` 里的 `list(self.models)` 仍能工作。
- **同文件关系**：被 `_parse_profile()` 构造并返回；被 `_parse_document()` 收集进字典；被 `ProviderRegistry.reload()` 装入 `MappingProxyType`；被 `ProviderRegistry.get()` 返回给调用方；其 `public_info()` 是本类的方法。

#### `public_info(self) -> dict[str, Any]` （第 77 行）

- **作用**：把 profile 的关键信息导成一个普通字典，用于「可被外部检视」的场景，比如 Web 接口返回配置列表、日志打印、前端展示 provider 下拉框。它与直接把 dataclass 暴露出去的最大区别在于安全性：密钥字段被替换成 `"***"`（无密钥时是空字符串），因此调用方即使把返回值序列化成 JSON 发给浏览器或写进日志，也不会泄露真实密钥。同时它把 `base_url` 额外以 `api_url` 的键名重复输出，是为了兼容仍然使用旧字段名的调用方，避免迁移期的键名不一致。它不解析环境变量、不触发 `.env` 加载，因此是一个纯粹的、无副作用的只读投影。
- **参数**：`self`：当前的 `ProviderProfile` 实例，隐含传入，不需要调用方提供。
- **返回**：返回 `dict[str, Any]`，固定包含九个键：`name`（str）、`adapter`（str）、`base_url`（str）、`api_url`（与 `base_url` 相同）、`api_key`（`"***"` 或 `""`）、`api_key_env`（str 或 None）、`default_model`（str）、`models`（`list[str]`，由元组转换而来，避免调用方拿到不可变序列）、`tool_mode`（str）。
- **内部流程**：直接返回一个字面量字典，逐键写入 `self` 的对应字段；其中 `api_key` 用条件表达式 `"***" if self.api_key else ""` 判定是否脱敏；`models` 用 `list(self.models)` 做一次浅拷贝转换；`base_url` 被写两次（`base_url` 与 `api_url`）。没有循环、没有分支以外的逻辑、没有外部调用。
- **异常/边界**：正常路径不抛异常。边界情况：若实例的 `api_key` 为空字符串（说明密钥走环境变量或配置缺失），返回空字符串而不是 `"***"`，这一点在判断「是否需要提示用户配置密钥」时可能被用到；若 `models` 被非常规地设成非可迭代对象，`list()` 会抛 `TypeError`，但正常构造路径不会出现这种情况。
- **同文件关系**：只读取自身字段，不调用本文件其它函数；本文件内没有其它函数调用它，它主要供外部（Web 层、调用方）使用。

---

### `class ProviderRegistry` （第 93 行）

- **作用**：这是本文件的门面类，负责「定位配置文件 → 读取 TOML → 校验 → 缓存 → 按名查询 → 解析真实密钥」这一整条链路。项目启动时通常只构造一次，之后所有需要 provider 信息的地方都通过这个实例访问，从而保证配置只解析一遍、行为一致。它内部用 `MappingProxyType` 包装 profile 字典，使外部只能读不能改；同时记录 `active_profile`（来自 `[defaults].active_profile`）告诉调用方「默认该用哪个 profile」。它把「文件在哪」和「文件里写了什么」解耦：`default_config_path()` 决定位置，`reload()` 决定内容，因此支持运行期重新加载配置（例如用户改了 TOML 之后调用 `reload()`）。密钥的真实解析被单独放在 `resolve_api_key()` 里，与查询 profile 元数据分开，避免在不必要的时候触碰环境变量。
- **属性**：
  - `config_path`：实际使用的配置文件路径，构造时确定。
  - `_profiles`：`Mapping[str, ProviderProfile]`，只读映射，键是 profile 名。
  - `active_profile`：字符串，当前激活的 profile 名，构造与 `reload()` 时更新。
- **返回**：不适用（类定义）。
- **内部流程**：见其 `__init__` 与各方法。整体上，构造即完成一次加载，后续 `reload()` 可在不重建对象的情况下刷新。
- **异常/边界**：类本身不抛异常，具体行为见各方法；构造时会因为 `reload()` 而立即抛出文件缺失、TOML 非法或校验失败等异常，属于「失败尽早」的设计。
- **同文件关系**：`__init__` 调用 `default_config_path()` 与 `reload()`；`reload()` 调用 `_parse_document()`；`get()` 读取 `_profiles`；`resolve_api_key()` 调用 `load_project_dotenv()` 与 `get()`。

#### `__init__(self, config_path: str | Path | None = None) -> None` （第 96 行）

- **作用**：构造注册表实例，确定要读哪个配置文件，初始化内部只读映射与激活名，并立即执行一次加载。这样设计的用意是让「拿到对象」就等于「配置可用」：调用方不需要额外记得调用 `reload()`，一旦构造成功，`profiles`、`active_profile`、`get()` 立刻可用。若配置文件缺失或内容非法，异常会在构造点抛出，问题暴露在启动阶段而不是第一次发请求时。传入自定义 `config_path` 的用法主要用于测试（指向临时 TOML）或多环境部署（不同路径的配置文件）。
- **参数**：
  - `self`：实例本身。
  - `config_path: str | Path | None = None`：配置文件路径。为 `None` 时调用 `default_config_path()` 按优先级探测；非 `None` 时用 `Path(config_path).expanduser().resolve()` 处理——`expanduser()` 展开 `~`，`resolve()` 转成绝对路径并消解符号链接与 `..`，因此 `self.config_path` 一定是绝对路径。
- **返回**：`None`（构造函数）。
- **内部流程**：
  1. 计算 `self.config_path`：非空则 `Path(config_path).expanduser().resolve()`，否则 `self.default_config_path()`（注意这里通过实例调用静态方法）。
  2. 初始化 `self._profiles = MappingProxyType({})`，即一个空的只读映射占位，避免加载失败时属性不存在。
  3. 初始化 `self.active_profile = ""`。
  4. 调用 `self.reload()` 完成真正的读取与校验，成功时覆盖上面两个属性。
- **异常/边界**：由 `reload()` 传播的异常：配置文件不存在 → `FileNotFoundError`；TOML 语法错误 → `ValueError`；结构或字段非法 → `TypeError` / `ValueError`。若 `config_path` 传入的是非法类型（如整数），`Path()` 会抛 `TypeError`。构造过程中不会留下半初始化状态对外可见（对象尚未返回）。
- **同文件关系**：调用 `default_config_path()`（静态方法）与 `reload()`；`reload()` 内部再调用 `_parse_document()`。

#### `default_config_path() -> Path` （第 106 行）

- **作用**：静态方法，用于在没有任何显式配置路径时决定「该去读哪个文件」。它按优先级依次探测三个文件名：`provider.toml`（私有运行时配置，真实使用）、`providers.toml`（另一种常见命名）、`provider.example.toml`（仓库里公开的安全模板）。注释说明了为什么保留模板兜底：新克隆的仓库在用户还没复制出私有文件之前，仍然可以被注入式/假的 LLM 使用，不至于因为缺配置而完全跑不起来。它返回一个 `Path`，无论文件是否存在都会返回某个路径（全部不存在时返回 `provider.toml` 的路径），因此「文件不存在」的判断留给了 `reload()`，而不是在这里报错。声明为 `@staticmethod`，所以既可以 `ProviderRegistry.default_config_path()` 调用，也可以像 `__init__` 里那样通过实例调用。
- **参数**：无参数（静态方法，不需要 `self`）。
- **返回**：`Path` 对象。若 `config/provider.toml` 存在则返回它；否则若 `config/providers.toml` 存在则返回它；否则若 `config/provider.example.toml` 存在则返回它；三者都不存在时返回 `config/provider.toml`（该文件并不存在，后续 `reload()` 会因此抛 `FileNotFoundError`）。
- **内部流程**：
  1. `config_dir = Path(__file__).resolve().parent.parent / "config"`：以本文件（`agents/providers.py`）为锚点，先解析成绝对路径，取父目录（`agents/`），再取上一级（项目根），拼上 `config`，从而得到与当前工作目录无关的配置目录。
  2. 用 `for filename in ("provider.toml", "providers.toml", "provider.example.toml")` 按顺序遍历候选文件名。
  3. 对每个候选构造 `config_dir / filename`，用 `candidate.is_file()` 判断；第一个存在的立即 `return`。
  4. 循环结束仍未命中，则 `return config_dir / "provider.toml"`。
- **异常/边界**：正常情况不抛异常。`is_file()` 对目录返回假，所以同名目录不会被误选；权限受限导致 `is_file()` 抛 `OSError` 的极端情况不会被捕获，会向上传播。返回的路径不保证存在。
- **同文件关系**：被 `ProviderRegistry.__init__()` 调用（当 `config_path` 为 `None` 时）；它不调用本文件其它函数。

#### `profiles` （property，第 117 行）

- **作用**：这是一个只读属性，对外暴露内部加载好的 profile 映射。之所以用 `@property` 而不是直接暴露 `_profiles` 属性名，是为了把「内部存储」和「对外接口」分开：内部字段名带下划线，未来即使换存储方式也不影响调用方；同时因为返回的是 `MappingProxyType`，调用方拿到的是一个真正的只读视图，无法 `pop`、无法赋值新键，从而保证注册表内容只能通过 `reload()` 变更。调用方通常用它来遍历所有可用 profile（例如做配置展示、校验 `active_profile` 是否存在、列举可切换的目标）。
- **参数**：`self`：实例本身。
- **返回**：`Mapping[str, ProviderProfile]`，即当前的 `self._profiles`。正常加载后是包含全部已校验 profile 的只读映射；若尚未成功加载过，则可能是构造时置入的空只读映射。
- **内部流程**：函数体只有一行 `return self._profiles`，没有分支、循环或额外计算。属性的读取代价恒定。
- **异常/边界**：不抛异常。因为返回的是 `MappingProxyType`，调用方尝试写入会抛 `TypeError`，这是预期行为而非缺陷。空配置不会走到这里（`reload()` 会先因 `raw_profiles` 为空而抛错）。
- **同文件关系**：只读取 `self._profiles`；不调用本文件其它函数，也不被本文件其它函数调用（供外部使用）。

#### `reload(self) -> None` （第 121 行）

- **作用**：重新读取并解析配置文件，用新结果替换内部状态。它是「配置热更新」的入口：用户编辑完 TOML 后，调用方可以调用它让运行中的进程用上新配置，而不必重启。同时它也是构造流程的实际执行者，`__init__` 里就是通过调用它来完成首次加载。它把「文件缺失」「TOML 语法错误」这两类与文件直接相关的问题就地转换成明确的异常，并把结构化校验全部委托给 `_parse_document()`。加载成功后，`_profiles` 被替换成新的只读映射、`active_profile` 被更新；加载失败时异常直接抛出，因此原有状态不会被部分覆盖（替换发生在解析成功之后）。
- **参数**：`self`：实例本身。方法不接收路径参数，因为路径已在构造时存入 `self.config_path`；要换文件需要新建实例或直接修改该属性后再调用。
- **返回**：`None`。效果是更新 `self._profiles` 与 `self.active_profile`。
- **内部流程**：
  1. `if not self.config_path.is_file()`：检查文件是否存在，不存在则抛 `FileNotFoundError`，消息中带上完整路径。
  2. 用 `with self.config_path.open("rb") as handle:` 以二进制只读方式打开，并调用 `tomllib.load(handle)` 解析成字典 `document`。用二进制模式是因为 `tomllib` 要求读取 bytes。
  3. 捕获 `tomllib.TOMLDecodeError`，重新抛出为 `ValueError`，消息为 `invalid provider profile TOML: <路径>`，并用 `from exc` 保留原始异常链，便于定位具体语法错误。
  4. `profiles, active_profile = _parse_document(document, self.config_path)`：把字典校验并转换成 `dict[str, ProviderProfile]` 与激活名。
  5. `self._profiles = MappingProxyType(profiles)`：包装成只读映射后替换。
  6. `self.active_profile = active_profile`：更新激活名。
- **异常/边界**：文件不存在 → `FileNotFoundError`；TOML 语法错误 → `ValueError`（原始 `TOMLDecodeError` 挂在 `__cause__` 上）；结构/字段非法 → 由 `_parse_document()` 抛出的 `TypeError` 或 `ValueError`；文件存在但无读取权限 → `open()` 抛 `OSError`，本函数不捕获，会直接向上传播。这些异常发生时，`_profiles` 与 `active_profile` 保持调用前的值，不会出现「解析到一半」的中间状态。
- **同文件关系**：调用 `_parse_document()`；被 `ProviderRegistry.__init__()` 调用；供外部在配置变更后手动触发。

#### `get(self, name: str) -> ProviderProfile` （第 139 行）

- **作用**：按名字取出一个 `ProviderProfile`。它是调用方访问配置元数据的主入口，`resolve_api_key()` 内部也依赖它。它做了两层防御：一是校验入参确实是「非空字符串」（`name.strip()` 为空也拒绝），避免因为传了 `None` 或空白字符串导致 `KeyError` 这种信息量低的错误；二是当名字不存在时，把 `KeyError` 转换成 `ValueError`，并在消息里列出当前所有可用 profile 名，让调用者一眼看出是拼写错误还是配置里根本没有这一项。与 `resolve_api_key()` 的分工是：`get()` 只返回 profile 对象本身，不解析环境变量、不读取 `.env`。
- **参数**：
  - `self`：实例本身。
  - `name: str`：目标 profile 名，必须是非空（且 `strip()` 后非空）的字符串，通常来自 `active_profile` 或用户显式指定。
- **返回**：`ProviderProfile` 实例，即 `self._profiles[name]` 对应的那个已校验档案。
- **内部流程**：
  1. `if not isinstance(name, str) or not name.strip():` 为真则抛 `ValueError("profile_name must be a non-empty string")`。
  2. 在 `try` 块里执行 `return self._profiles[name]`。
  3. 捕获 `KeyError`：用 `", ".join(self._profiles)` 拼出可用名字列表（若映射为空则用字面量 `"(none)"`），抛出 `ValueError`，消息格式为 `provider profile '<name>' was not found; available profiles: <列表>`，并用 `from exc` 保留原因。
  4. 正常命中时直接返回，不做任何拷贝（返回的是同一个不可变实例，可安全共享）。
- **异常/边界**：非字符串或空白字符串 → `ValueError`；名字不存在 → `ValueError`（附可用列表）；映射为空时列表显示为 `(none)`。注意入参只做「是否为空」校验，不做「字符是否合法」校验，因此带空格或奇怪字符的名字不会在这里被拒，只会走「未找到」分支。返回的 profile 已冻结，调用方无法修改。
- **同文件关系**：读取 `self._profiles`；被 `ProviderRegistry.resolve_api_key()` 调用；不调用本文件其它函数。

#### `resolve_api_key(self, profile_name: str) -> str` （第 150 行）

- **作用**：解析并返回指定 profile 真正可用的 API 密钥。密钥有两个来源：一是配置里直接内联的 `api_key`（新格式），二是配置里写的 `api_key_env` 环境变量名（此时真实值来自进程环境或 `.env` 文件）。本函数按「先内联、后环境变量」的顺序查找，并且在查找前先调用 `load_project_dotenv()`，确保仓库根的 `.env` 已经被灌进 `os.environ`——这正是文件注释里提到「之前跳过 `.env` 导致 ingest 静默产出零实体」的那个修复点。把密钥解析从 `get()` 里拆出来单独成一个方法，可以让不涉及发请求的代码（例如只展示 profile 列表）永远不触碰密钥，降低泄露面。解析失败时抛出信息明确的 `ValueError`，区分「环境变量没设置」和「profile 根本没有配置密钥」两种情况。
- **参数**：
  - `self`：实例本身。
  - `profile_name: str`：目标 profile 名，透传给 `get()`，因此同样要求非空字符串，否则会由 `get()` 抛 `ValueError`。
- **返回**：`str`，返回的密钥已经过 `strip()`（环境变量分支显式 strip，内联分支在解析阶段就已 strip），因此不会是首尾带空白的字符串。
- **内部流程**：
  1. `load_project_dotenv()`：先把 `.env` 载入环境变量（默认路径、只加载一次）。
  2. `profile = self.get(profile_name)`：取出 profile，名字非法或不存在时在此抛错。
  3. `if profile.api_key:` 为真 → 直接返回内联密钥，不再看环境变量。
  4. 否则若 `profile.api_key_env` 非空，用 `os.getenv(profile.api_key_env)` 取值；若取到的是字符串且 `strip()` 后非空，则返回 `api_key.strip()`。
  5. 若 `profile.api_key_env` 非空但环境变量取不到值，抛 `ValueError`，消息说明该 profile 需要哪个环境变量。
  6. 走到最后说明既无内联密钥也无环境变量名，抛 `ValueError`，消息说明该 profile 的 `api_key` 为空。
- **异常/边界**：`profile_name` 非法或不存在 → 由 `get()` 抛 `ValueError`；环境变量名已配置但变量缺失或为纯空白 → `ValueError`（消息含变量名）；两者都没有 → `ValueError`（消息说明 `api_key` 为空）。边界细节：内联密钥为纯空白字符串时在解析阶段就会被拒（`_parse_profile` 抛错），所以运行到这里的内联值要么非空要么就是空串；环境变量值会 strip 后再返回，纯空白等同于缺失。
- **同文件关系**：调用 `load_project_dotenv()`（模块级函数）与 `ProviderRegistry.get()`；不调用其它解析辅助函数；被外部（发请求前取密钥的代码）调用。

---

### `_parse_document(document: Mapping[str, Any], config_path: Path) -> tuple[dict[str, ProviderProfile], str]` （第 169 行）

- **作用**：这是整个加载流程的「结构层校验器」。它接收 `tomllib` 解析出的整份文档字典，检查顶层结构是否合法（必须有 `[defaults]` 表、必须有非空的 `[profiles]` 表、`active_profile` 必须是有效字符串），然后逐个把 `[profiles.<name>]` 子表交给 `_parse_profile()` 转成 `ProviderProfile`，最后校验 `active_profile` 指向的 profile 确实存在。把这些检查集中在一处，可以保证「注册表里的数据一定是自洽的」：只要 `_parse_document()` 返回成功，调用方就不必再担心引用了不存在的 profile。它把结构错误（缺表、类型不对、名字不合法、激活项不存在）与单条 profile 的字段错误分层处理，错误信息里带上 `config_path` 或 profile 名，方便定位。
- **参数**：
  - `document: Mapping[str, Any]`：由 `tomllib.load()` 得到的整份配置字典（顶层键通常是 `defaults` 与 `profiles`）。
  - `config_path: Path`：配置文件路径，仅用于拼装错误信息，帮助使用者知道是哪份文件出了问题。
- **返回**：`tuple[dict[str, ProviderProfile], str]`——第一个元素是以 profile 名为键、`ProviderProfile` 为值的普通字典（尚未包装成只读映射，由 `reload()` 负责包装）；第二个元素是 `active_profile` 字符串。
- **内部流程**：
  1. `if not isinstance(document, Mapping):` 为真则抛 `TypeError`，消息带 `config_path`（防御非字典输入）。
  2. 用 `document.get("defaults")` 取 `defaults`，用 `document.get("profiles")` 取 `raw_profiles`。
  3. `defaults` 不是 `Mapping` → 抛 `TypeError`，说明缺少 `[defaults]` 表。
  4. `raw_profiles` 不是 `Mapping` 或为空 → 抛 `ValueError`，说明至少需要一个 `[profiles.<name>]` 表。
  5. `active_profile = _required_string(defaults, "active_profile", "[defaults]")`：取必需的非空字符串，位置标签是 `[defaults]`，缺失或空白时由 `_required_string()` 抛 `ValueError`。
  6. 初始化空字典 `profiles: dict[str, ProviderProfile] = {}`。
  7. 遍历 `raw_profiles.items()`：对每个 `name` 先用 `isinstance(name, str)` 与 `_PROFILE_NAME_RE.fullmatch(name)` 校验（必须以字母或数字开头，后续可含字母、数字、下划线、连字符，总长 1 到 64 字符），不合法抛 `ValueError` 并说明命名规则；若 `raw_profile` 不是 `Mapping`，抛 `TypeError`，消息形如 `[profiles.<name>] must be a TOML table`；两项都通过则 `profiles[name] = _parse_profile(name, raw_profile)`。
  8. 循环结束后 `if active_profile not in profiles:` 则抛 `ValueError`，说明 `[defaults].active_profile '<值>'` 没有指向已配置的 profile。
  9. 返回 `(profiles, active_profile)` 元组。
- **异常/边界**：`document` 非映射 → `TypeError`；缺 `[defaults]` → `TypeError`；`[profiles]` 缺失或为空 → `ValueError`；profile 名不合规 → `ValueError`；某个 profile 不是表 → `TypeError`；`active_profile` 缺失/空白 → `ValueError`；`active_profile` 不存在 → `ValueError`；单条 profile 内部的字段问题由 `_parse_profile()` 抛出，异常会直接穿过本函数向上传播。注意 `[defaults]` 里除 `active_profile` 外的其它键会被静默忽略，未知的顶层键同样被忽略（`tomllib` 不会因此报错，本函数也不做白名单检查）。
- **同文件关系**：调用 `_required_string()` 与 `_parse_profile()`（后者又调用 `_optional_string()`、`_required_string()`、`_validate_model()`，并读取模块级常量 `_PROFILE_NAME_RE`、`_TOOL_MODES`）；被 `ProviderRegistry.reload()` 调用。

---

### `_parse_profile(name: str, raw_profile: Mapping[str, Any]) -> ProviderProfile` （第 198 行）

- **作用**：把单个 `[profiles.<name>]` 子表校验并转换成冻结的 `ProviderProfile` 实例。这是字段级校验的集中地：它确认适配器只支持 `openai_compatible`；确认 API 地址存在且是绝对的 HTTP(S) URL；处理 `api_key` 与 `api_key_env` 两代字段的兼容逻辑；确认 `default_model` 非空且确实出现在 `models` 里；确认 `models` 是非空、无重复的字符串数组；确认 `tool_mode` 落在允许集合内。它还负责把 `base_url` 结尾多余的 `/` 去掉，避免调用方拼接路径时出现双斜杠。整个函数以「发现第一个问题就抛异常」的方式工作，因此错误信息具体到字段，便于用户修配置。
- **参数**：
  - `name: str`：profile 名（已由 `_parse_document()` 校验过命名规则），用于拼装错误信息并写入返回对象的 `name` 字段。
  - `raw_profile: Mapping[str, Any]`：该 profile 的原始 TOML 子表，键可能是 `adapter`、`api_url` / `base_url`、`api_key`、`api_key_env`、`default_model`、`models`、`tool_mode`。
- **返回**：构造完成的 `ProviderProfile` 实例，字段取值：`name` 为入参名；`adapter` 为校验后的适配器字符串；`base_url` 为 strip 掉结尾 `/` 的地址；`api_key` 为内联密钥（若走环境变量方式则为空字符串）；`api_key_env` 为环境变量名或 `None`；`default_model`、`models`、`tool_mode` 均为校验后的值。
- **内部流程**：
  1. `adapter = _optional_string(raw_profile, "adapter", "openai_compatible")`：缺省即 `openai_compatible`；若不是该值则抛 `ValueError`，消息说明只实现了 `openai_compatible`。
  2. 取 API 地址：用 `"api_url" if "api_url" in raw_profile else "base_url"` 选择键名（优先新名 `api_url`，回退旧名 `base_url`），交给 `_required_string()` 取必需非空字符串。
  3. `parsed_url = urlsplit(base_url)`：校验 `parsed_url.scheme` 属于 `{"http", "https"}` 且 `parsed_url.netloc` 非空，否则抛 `ValueError`，说明必须是绝对的 HTTP(S) URL。
  4. 取 `raw_api_key = raw_profile.get("api_key")`，初始化 `api_key_env: str | None = None`。
  5. 若 `raw_api_key is None`（说明配置用的是旧字段或压根没写）：取 `legacy_key = raw_profile.get("api_key_env")`；若它是非空字符串，则用 `_ENV_NAME_RE.fullmatch(legacy_key.strip())` 判断它像不像环境变量名——像则视为环境变量名，`api_key_env` 记录它并把 `raw_api_key` 置为空字符串；不像（例如旧版本误把真实密钥写进了 `api_key_env`）则把整串当作内联密钥使用，保证旧的本地配置继续可用；若 `legacy_key` 不是非空字符串，则抛 `ValueError`，说明需要非空 `api_key`。
  6. 类型校验：`raw_api_key` 不是 `str` → 抛 `TypeError`。
  7. 取值校验：`raw_api_key.strip()` 为空且 `api_key_env is None` → 抛 `ValueError`（两个来源都没有密钥）。
  8. `default_model = _required_string(raw_profile, "default_model", f"profile '{name}'")`。
  9. `raw_models = raw_profile.get("models")`：不是 `list` 或为空 → 抛 `ValueError`，说明必须是非空的 TOML 数组。
  10. `models = tuple(_validate_model(name, model) for model in raw_models)`：逐个校验为非空字符串并 strip，转成元组。
  11. `if len(set(models)) != len(models):` → 抛 `ValueError`，说明不能有重复模型。
  12. `if default_model not in models:` → 抛 `ValueError`，说明默认模型必须出现在 `models` 中。
  13. `tool_mode = _optional_string(raw_profile, "tool_mode", "native_strict")`；若不在 `_TOOL_MODES` 集合中，抛 `ValueError`，并把允许项排序后用 `", ".join(...)` 拼进消息。
  14. 用关键字参数构造并返回 `ProviderProfile`，其中 `base_url=base_url.rstrip("/")`、`api_key=raw_api_key.strip()`。
- **异常/边界**：不支持的适配器 → `ValueError`；地址缺失/空白 → `ValueError`（来自 `_required_string`）；地址不是 http/https 或缺少主机 → `ValueError`；旧字段 `api_key_env` 缺失且没有 `api_key` → `ValueError`；`api_key` 类型不是字符串 → `TypeError`；密钥为空且没有环境变量名 → `ValueError`；`default_model` 缺失 → `ValueError`；`models` 非数组或为空 → `ValueError`；`models` 中含非字符串或空白项 → `ValueError`（来自 `_validate_model`）；`models` 有重复 → `ValueError`；`default_model` 不在 `models` 中 → `ValueError`；`tool_mode` 非法 → `ValueError`。边界处理上：`base_url` 结尾的单个或多个 `/` 都会被 `rstrip("/")` 去掉；`api_key` 与 `api_key_env` 同时存在时以内联 `api_key` 为准，且此时不再解释 `api_key_env`（不会记录到 `api_key_env` 字段）；未知的多余键被静默忽略。
- **同文件关系**：调用 `_optional_string()`、`_required_string()`、`_validate_model()`，并使用模块级常量 `_ENV_NAME_RE` 与 `_TOOL_MODES`，最终构造 `ProviderProfile`；被 `_parse_document()` 在遍历 profile 表时调用。

---

### `_required_string(source: Mapping[str, Any], key: str, location: str) -> str` （第 257 行）

- **作用**：一个小而通用的取「必需非空字符串」的工具函数，用来消除重复的取值加校验代码。它被 `_parse_document()`（取 `active_profile`）和 `_parse_profile()`（取 API 地址、`default_model`）共同使用。它把两种常见错误统一成一条 `ValueError`：键不存在、值不是字符串、值是纯空白字符串，这三者在配置场景里都属于「没配好」，因此合并处理。`location` 参数让错误信息能指明问题出在哪个表或哪个 profile，例如 `[defaults]` 或 `profile 'deepseek'`，使用者据此能立刻定位到 TOML 的哪一行区域。
- **参数**：
  - `source: Mapping[str, Any]`：被检查的映射（某个 TOML 表）。
  - `key: str`：要读取的键名。
  - `location: str`：用于错误信息的人类可读位置描述，例如 `"[defaults]"` 或 `"profile 'deepseek'"`。
- **返回**：`str`，返回 `value.strip()` 之后的结果，因此返回值一定非空且不含首尾空白。
- **内部流程**：
  1. `value = source.get(key)`：取键值，不存在得到 `None`。
  2. `if not isinstance(value, str) or not value.strip():` 为真则抛 `ValueError(f"{location} requires non-empty string '{key}'")`。
  3. 否则 `return value.strip()`。
- **异常/边界**：键缺失、值类型非字符串（如列表、整数、布尔）、值为纯空白 → 均抛 `ValueError`，消息包含 `location` 与 `key`。不做类型转换（不会把数字 5 变成 `"5"`），也不会截断或做正则校验。
- **同文件关系**：不调用本文件其它函数；被 `_parse_document()` 与 `_parse_profile()` 调用。

---

### `_optional_string(source: Mapping[str, Any], key: str, default: str) -> str` （第 264 行）

- **作用**：取「可选字符串」的工具函数，语义是「没配就用默认值，配了就必须是非空字符串」。它与 `_required_string()` 的差别在于允许缺省，但不允许显式配成空串或空白串——这样可以防止用户在 TOML 里写了 `tool_mode = ""` 却被静默当成默认值，从而掩盖配置错误。它被 `_parse_profile()` 用于 `adapter`（默认 `openai_compatible`）与 `tool_mode`（默认 `native_strict`）两个字段。注意它在 `source.get(key, default)` 里把默认值作为 `get` 的缺省参数，所以「键不存在」走默认值，「键存在但值非法」则抛异常。
- **参数**：
  - `source: Mapping[str, Any]`：被检查的映射（某个 TOML 表）。
  - `key: str`：要读取的键名。
  - `default: str`：键不存在时使用的默认值，调用方传入的必须是字符串（本函数不校验 `default` 自身的类型）。
- **返回**：`str`。键不存在时返回 `default`（原样返回，注意此时不做 `strip()`，因为它是调用方写死的字面量）；键存在且合法时返回 `value.strip()`。
- **内部流程**：
  1. `value = source.get(key, default)`。
  2. `if not isinstance(value, str) or not value.strip():` 为真则抛 `ValueError(f"'{key}' must be a non-empty string when configured")`。
  3. 否则 `return value.strip()`。
- **异常/边界**：键存在但值不是字符串，或值是空串/纯空白 → 抛 `ValueError`；键不存在 → 返回默认值，不抛异常。由于默认值分支不经过 `strip()`，如果调用方传入带空白的默认值会原样返回（本文件内的调用方传入的都是干净字面量）。
- **同文件关系**：不调用本文件其它函数；被 `_parse_profile()` 调用两次（`adapter` 与 `tool_mode`）。

---

### `_validate_model(profile_name: str, value: Any) -> str` （第 271 行）

- **作用**：校验 `models` 数组里的单个元素，确保它是「非空字符串」。它被 `_parse_profile()` 用在生成器表达式里逐项处理 `models`，是模型列表校验的最小单元。之所以单独抽出来，是为了在错误信息里带上所属 profile 名，让用户知道是哪个 profile 的模型列表写错了（例如误把模型名写成了嵌套表或数字）。它同时负责 strip，从而让 `models` 与 `default_model` 的比较不受首尾空白影响。
- **参数**：
  - `profile_name: str`：所属 profile 名，仅用于错误信息。
  - `value: Any`：待校验的模型元素，期望是 `str`；可能是 `int`、`bool`、`list`、`dict` 等（TOML 数组里允许混合类型）。
- **返回**：`str`，即 `value.strip()` 的结果，保证非空、无首尾空白。
- **内部流程**：
  1. `if not isinstance(value, str) or not value.strip():` 为真则抛 `ValueError(f"profile '{profile_name}' models must contain non-empty strings")`。
  2. 否则 `return value.strip()`。
- **异常/边界**：元素不是字符串，或是空串/纯空白 → 抛 `ValueError`，消息包含 profile 名。不做去重（去重在调用方 `_parse_profile()` 里用 `set` 完成），也不校验模型名格式。
- **同文件关系**：不调用本文件其它函数；被 `_parse_profile()` 在 `models` 的生成器表达式里逐个调用。

---

## 三、一句话总览表

| 函数/类 | 一句话作用 |
| --- | --- |
| `load_project_dotenv(path=None)` | 把仓库根 `.env`（或显式指定文件）的键值对灌入 `os.environ`，且不覆盖已有变量、默认只执行一次。 |
| `ProviderProfile` | 冻结 dataclass，承载单个已校验的 OpenAI 兼容 provider 配置（名字、地址、密钥、默认模型、模型列表、适配器、工具模式）。 |
| `ProviderProfile.public_info()` | 把 profile 导成字典并对外暴露，密钥用 `***` 脱敏，同时以 `api_url` 复刻 `base_url` 兼容旧键名。 |
| `ProviderRegistry` | 门面类，负责定位/读取/校验/缓存 TOML 配置，并按名提供 profile 查询与密钥解析。 |
| `ProviderRegistry.__init__(config_path=None)` | 确定配置文件路径、初始化空的只读映射与激活名，并立即调用 `reload()` 完成首次加载。 |
| `ProviderRegistry.default_config_path()` | 静态方法，在项目 `config/` 目录按 `provider.toml` → `providers.toml` → `provider.example.toml` 顺序探测并返回路径。 |
| `ProviderRegistry.profiles` | 只读属性，返回内部 `MappingProxyType` 包装的 profile 映射。 |
| `ProviderRegistry.reload()` | 重新读取并解析 TOML，把结果替换进 `_profiles` 与 `active_profile`，缺文件/语法错误时抛明确异常。 |
| `ProviderRegistry.get(name)` | 按名取出 `ProviderProfile`，入参为空抛 `ValueError`，名字不存在时在错误信息里列出全部可用 profile。 |
| `ProviderRegistry.resolve_api_key(profile_name)` | 先加载 `.env`，再按「内联 `api_key` 优先、其次 `api_key_env` 环境变量」的顺序解析出真实密钥。 |
| `_parse_document(document, config_path)` | 校验整份文档的顶层结构（`[defaults]`、非空 `[profiles]`、`active_profile` 有效）并逐条解析出 profile 字典与激活名。 |
| `_parse_profile(name, raw_profile)` | 对单个 profile 表做字段级校验（适配器、URL、密钥新旧字段兼容、模型列表、默认模型、工具模式）并构造 `ProviderProfile`。 |
| `_required_string(source, key, location)` | 取必需的非空字符串，缺失/非字符串/纯空白统一抛带位置信息的 `ValueError`，返回 strip 后的值。 |
| `_optional_string(source, key, default)` | 取可选字符串，键缺失时返回默认值，键存在但为空或非字符串时抛 `ValueError`。 |
| `_validate_model(profile_name, value)` | 校验 `models` 中的单个元素为非空字符串并 strip，否则抛出带 profile 名的 `ValueError`。 |
