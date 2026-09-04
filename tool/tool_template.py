"""单文件工具模板：复制本文件后即可实现一个可自动发现的工具。

本文件既是可运行示例，也是给代码生成模型的完整实现规范。生成新工具时，
必须先理解下面的约束，再根据目标工具的真实行为替换示例，而不是只修改类名。

给代码生成模型的输出指令
==========================

当你依据本模板生成工具时，先从需求中确定：工具的单一职责、调用参数及约束、
稳定输出结构、外部依赖和配置、是否产生副作用、所需权限、是否幂等、是否支持
并发。如果缺少会影响安全性的关键信息，应先向用户确认，不能擅自把写操作标成
read，也不能虚构 API、字段或权限。

最终应输出一个完整的 Python 模块，而不是设计说明或代码片段。生成结果必须：

* 不含 ``TODO``、``pass``、省略号、伪代码、未定义名称或待用户补写的方法；
* 包含所需全部 import、Input、Output、Tool、ToolSpec、execute 和 create_tool；
* 将示例名称、描述、字段和逻辑全部替换成目标工具的真实内容；
* 除非用户明确要求默认关闭，否则把新工具的 ``TOOL_ENABLED`` 设为 ``True``；
* 优先只使用标准库和项目已有依赖。新增第三方包必须是需求明确允许的依赖；
* 不修改 ``core`` 或 ``agents`` 来迁就单个工具，不绕过 Registry 和 Runtime；
* 生成后逐项执行本文件末尾的验收要求，不能只保证语法可以解析。

一、最终文件必须满足的模块协议
================================

1. 文件必须直接放在 ``tool/`` 下，文件名不能以 ``_`` 开头，不能叫
   ``base.py``，也不要放进子目录。
2. 模块必须定义布尔常量 ``TOOL_ENABLED``。只有它严格等于 ``True`` 时，
   自动发现器才会调用工厂；不能使用字符串 ``"true"`` 代替布尔值。
3. 模块必须定义一个可零参数调用的 ``create_tool()``。它必须返回一个
   ``BaseTool`` 实例，不能返回类、字典、函数或 ``None``。
4. 一个文件只提供一个发现工厂。复杂能力应拆成多个命名清晰的小工具。
5. 模块导入阶段只能做导入、声明类型、定义常量等轻量操作。不能访问网络、
   写文件、启动线程、修改数据库或调用工具。配置读取和依赖创建优先放入
   ``create_tool()`` 或工具构造器，实际业务操作必须放入 ``execute()``。

二、生成一个新工具时必须依次完成
================================

1. 理清工具的单一职责、输入来源、稳定输出、外部依赖、权限和副作用。
2. 重命名文件、Input/Output/Tool 类、工具名和环境变量名。
3. 用 Pydantic 定义所有输入和输出；为字段补充 description、范围和长度限制。
4. 完整填写 ``ToolSpec``。如有可选的上下文准备工具，填写
   ``recommended_before_tools``；它只用于提示模型，不是运行时依赖。Schema 或
   可观察语义改变时必须升级 version。
5. 在 ``execute()`` 中只使用已经验证的 Input，清洗外部数据并返回 Output。
6. 在 ``create_tool()`` 中读取配置、创建客户端或注入依赖，然后返回工具实例。
7. 将复制后文件中的 ``TOOL_ENABLED`` 设为 ``True``，再检查发现报告与注册状态。

三、Input/Output 模型规则
=========================

* 顶层 Input 和 Output 必须继承 ``BaseModel``，并配置
  ``ConfigDict(extra="forbid", strict=True)``。建议所有嵌套模型也这样配置。
* Pydantic 模型是唯一协议来源。不要再手写另一份 JSON Schema，也不要在
  ``execute()`` 中重新解析 LLM 的自然语言或 JSON 文本。
* ``strict=True`` 表示不要依赖 ``"5" -> 5`` 等隐式类型转换。
* 每个输入字段都要有准确的 ``description``，让 LLM 知道字段语义、单位、格式
  和限制；字符串、数字、列表应设置合理的长度或范围，防止无界输入。
* OpenAI strict function schema 会把 Input 的所有 properties 都声明为 required。
  因此不要依赖“LLM 会省略带默认值字段”。如字段允许空值，类型必须显式写成
  ``str | None`` 等形式，但调用 JSON 中仍应包含该字段。
* 不要在 Input 中使用允许任意键的 ``dict[str, ...]``。严格函数 Schema 不允许
  任意 ``additionalProperties``；请改成字段明确的嵌套模型或有界列表。
* Output 必须是稳定、精简、可序列化的内部结构。不要直接返回第三方 SDK 对象、
  HTTP Response、大段未限制文本、异常对象或包含密钥的内容。
* ``execute()`` 推荐直接返回 Output 实例。即使返回字典，运行时也会再次用
  Output 严格校验；不符合时会得到 ``INVALID_OUTPUT``，不会传给 LLM。

四、ToolSpec 每个字段如何选择
==============================

``spec`` 通常像示例一样定义成工具类的不可变类属性。不要在执行过程中修改它。
如果构造配置确实会改变 Runtime 合约（例如用户配置了不同的执行超时），应在
构造器中用 ``dataclasses.replace(type(self).spec, timeout_seconds=实际值)`` 创建
实例级 spec，并确保底层客户端超时与它一致；Schema、版本和权限不能悄悄漂移。

``name``
    永久、唯一的名字，格式为 ``namespace.tool_name``，只能包含 ASCII 字母、
    数字和下划线，且至少有一个点。点会发送给提供商时映射为双下划线；映射后
    总长度不能超过 64。不要把版本写进名字。
``description``
    面向 LLM 描述“何时使用、完成什么、不能做什么”，最长 2000 字符。不要放
    API Key、动态状态或可被外部用户修改的提示词。
``version``
    非空字符串，最长 32。输入/输出字段、约束、权限、副作用或结果语义变化时
    升级版本。只改内部实现且行为兼容时可以不升级。
``input_model`` / ``output_model``
    分别指向本文件定义的 Pydantic 模型类，不能传实例。
``side_effect``
    只有精确值 ``"read"`` 被运行时视为无写副作用；``"write"``、
    ``"external_write"``、``"execute"`` 等其他非空值都会要求调用者提供与
    当前 Schema 和注册代次绑定的确认。不要为了省确认而错误标成 read。
``permissions``
    可选的兼容性/审计元数据。当前公开工具部署不执行权限过滤，所有调用者均可
    使用已注册工具；新工具通常直接写 ``permissions=()``。如未来接入多租户或
    私有工具，再恢复基于该字段的授权策略。
``timeout_seconds``
    有限正数，是单次实际执行的 Runtime 截止时间。网络客户端、数据库客户端也
    应配置不大于此值的内部超时。同步线程超时后 Python 无法强制终止线程，所以
    工具实现本身仍必须设置 I/O 超时并避免无限阻塞。
``idempotent``
    相同参数重复执行是否安全。纯读取通常为 True；发消息、扣费、写文件等通常
    为 False。它决定执行错误/超时是否标记 retryable，不代表 Runtime 自动重试。
``parallel_safe``
    多次调用能否安全并发。依赖非线程安全客户端、共享可变状态或必须保持顺序时
    设为 False。
``max_concurrency``
    此工具允许的并发上限；``None`` 表示使用 Runtime 全局上限。外部服务有限流
    时填写正整数。``parallel_safe=False`` 时实际并发会被限制为 1。
``tags``
    供 Catalog 搜索的稳定关键词元组，不参与授权，也不要塞入大段说明。
``recommended_before_tools``
    可选的命名空间工具名元组，用于提示调用当前工具前可能有帮助的工具。该字段
    只会出现在模型描述和 Catalog 元数据中；运行时不会自动调用、检查顺序或强制
    任何前置工具。工具之间应保持可插拔，不要在 Agent/ReAct 代码中写死具体名称。

五、execute() 实现规则
=======================

* 默认使用本示例的同步签名：
  ``def execute(self, arguments: XxxInput) -> XxxOutput``。
  Runtime 会把同步实现放入工作线程；原生异步逻辑不要再用 ``asyncio.run()``
  包一层，应直接采用下一条的异步签名。
* 原生异步 I/O 可以直接改成：
  ``async def execute(self, arguments: XxxInput) -> XxxOutput``。
* 如果业务确实需要调用主体或权限信息，可导入 ``ExecutionContext``，并使用
  参数名必须为 ``context`` 的签名：
  ``def execute(self, arguments: XxxInput, *, context: ExecutionContext)``。
  Runtime 会自动注入它；绝不能允许 LLM 在 Input 中伪造权限或确认字段。
* 不要自行返回 ``ToolResult``。正常情况返回 Output；失败时抛出合适的 Python
  异常，由 Runtime 统一转为结构化 ``EXECUTION_ERROR``。异常消息和日志不得
  包含密钥、令牌、完整用户隐私数据或第三方原始敏感响应。
* 写操作应尽量使用幂等键、事务或原子文件替换，并在提交前完成所有可做校验。
* 不要在工具内部实现无限重试。若确需重试，必须有很小的上限、退避策略，且总
  时间受 ``timeout_seconds`` 约束；非幂等操作默认不重试。
* 避免类级共享可变状态。Runtime 可能并发调用同一个工具实例；共享缓存或客户端
  必须是线程/协程安全的，否则应设置 ``parallel_safe=False``。

六、create_tool() 与开关
========================

* 工厂必须保持零参数。自动发现器不知道业务构造参数。
* 在工厂/构造器中读取环境变量并验证配置；不要在源码中写密钥。
* 工厂只创建依赖和实例，不调用 ``execute()``。创建失败会进入发现报告 error。
* 静态开关使用 ``TOOL_ENABLED = True``。环境开关可以参考：

  ``TOOL_ENABLED = os.getenv("MY_TOOL_ENABLED", "false").casefold() in``
  ``{"1", "true", "yes", "on"}``

  环境变量必须在模块首次导入前设置。关闭开关只阻止本次发现调用工厂，不会从
  已存在的 Agent 中热卸载旧实例；关闭后应重建 Agent。

七、运行时链路和验收条件
=========================

``Agent`` 发现模块 -> 检查开关 -> 调用工厂 -> Registry 注册 -> Input 生成工具
Schema -> LLM 返回 tool_call -> Parser 绑定版本/Schema 哈希/注册代次 -> Runtime
检查副作用确认并验证 Input -> ``execute()`` -> Runtime 验证 Output -> ToolResult
返回 LLM。

完成后的最低验收：模块可导入；发现报告无 error；工具状态 registered；合法
Input 得到合法 Output；缺字段、错类型、多余字段被拒绝；未确认写操作不会执行；
权限元数据不限制访问；异常、超时和非法外部响应能安全失败；代码通过 Ruff 和 pytest。

下面的示例实现是完整可执行的。生成真实工具时保留协议结构，替换示例领域逻辑。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from core import BaseTool, ToolSpec

# 模板必须保持关闭，避免被当成业务工具。复制后选择以下一种方式：
# 1. 固定启用：TOOL_ENABLED = True
# 2. 环境开关：按模块文档中的写法读取一个工具专属环境变量。
TOOL_ENABLED = False


class ExampleInput(BaseModel):
    """LLM 能够提供的全部参数；不要加入权限、密钥或内部客户端对象。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    text: str = Field(
        min_length=1,
        max_length=10_000,
        description="需要统计的 UTF-8 文本；必须包含至少一个非空白字符。",
    )

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        """表达 JSON Schema 长度无法覆盖的领域规则。"""
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class ExampleOutput(BaseModel):
    """返回给 Runtime 和 LLM 的稳定、经过清洗的结果。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    character_count: int = Field(
        ge=0,
        description="Python len() 计算得到的 Unicode 字符数量。",
    )
    word_count: int = Field(
        ge=0,
        description="按连续空白字符分隔后得到的词语数量。",
    )


class ExampleTool(BaseTool):
    """无外部副作用的文本统计示例；真实工具应改成自己的单一职责。"""

    spec = ToolSpec(
        name="template.text_statistics",
        description=(
            "Count Unicode characters and whitespace-delimited words in provided "
            "text. Use it only when exact deterministic counts are needed."
        ),
        version="1.0.0",
        input_model=ExampleInput,
        output_model=ExampleOutput,
        side_effect="read",
        permissions=(),
        timeout_seconds=5.0,
        idempotent=True,
        parallel_safe=True,
        max_concurrency=8,
        tags=("text", "statistics", "count"),
    )

    def execute(self, arguments: ExampleInput) -> ExampleOutput:
        """只接收已验证参数，并始终返回符合 ExampleOutput 的对象。"""
        return ExampleOutput(
            character_count=len(arguments.text),
            word_count=len(arguments.text.split()),
        )


def create_tool() -> BaseTool:
    """自动发现器调用的唯一零参数工厂。"""
    # 有配置时在这里读取并验证，例如：
    # endpoint = os.environ["MY_TOOL_ENDPOINT"]
    # client = MyClient(endpoint=endpoint, timeout=ExampleTool.spec.timeout_seconds)
    # return MyTool(client=client)
    return ExampleTool()
