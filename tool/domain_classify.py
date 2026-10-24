"""本地主题领域分类工具：把文本/文档归类到恒星系（领域）。

为什么这是一个独立能力
======================

"这段内容属于哪个领域"是纯函数、零依赖、确定性（同文本永远同结果），却被三处复用：
星云图把知识块挂到领域恒星上、文档按多数领域归属、以及图检索往提示词里注入已知领域清单。
原先它藏在 ``web/domain_classifier.py``（Web 层的一个私有模块），Agent 无法调用；
现在它是 ``knowledge.classify_domain``，Web 层与记忆层都从同一份实现取用。

设计约束（保持原样，未做改动）
==============================

- 零依赖、纯规则：不联网、不消耗 LLM key，未配置任何云端 key 也能用；
- 确定性：跨进程稳定，可测试（词表与权重都在 ``constants.py`` 里）；
- 轻量：一次遍历关键词表，万级节点构建时可忽略不计。

本模块是这段逻辑的**唯一实现**：``web/domain_classifier.py`` 已删除，
``web/app.py`` 的图接口与 ``memory/rag/knowledge.build_graph_context`` 改为从这里取用。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from constants import DEFAULT_DOMAIN as DEFAULT
from constants import DOMAIN_TITLE_WEIGHT
from core import BaseTool, ToolSpec

TOOL_ENABLED = True

#: 领域 → 命中关键词表（中英混合，按主题覆盖度维护）。
#: 关键词按“主题区分度”人工挑选：太通用的词（如“数据”“系统”）容易串类，故不收录。
DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "编程开发": (
        "编程", "代码", "函数", "变量", "程序", "算法", "软件", "程序库", "库函数",
        "python", "java", "javascript", "c语言", "c++", "go语言", "rust",
        "前端", "后端", "数据库", "sql", "api", "接口", "git", "linux", "服务器",
        "数据结构", "链表", "栈", "队列", "排序", "递归", "面向对象", "类", "对象",
        "机器学习", "人工智能", "深度学习", "神经网络", "大模型", "提示词", "embedding",
        "bug", "调试", "编译", "运行", "框架", "依赖", "组件", "配置",
    ),
    "数学": (
        "数学", "方程", "几何", "代数", "微积分", "概率", "统计", "导数",
        "矩阵", "向量", "定理", "证明", "数论", "线性", "函数图像", "积分",
        "三角", "对数", "指数", "集合", "命题", "公理", "假设检验", "分布",
    ),
    "物理": (
        "物理", "力学", "电磁", "量子", "相对论", "能量", "原子", "光谱",
        "热力学", "电学", "磁场", "波动", "引力", "牛顿", "运动学", "加速度",
        "动量", "功与能", "光学", "电流", "电压", "电阻", "粒子", "质量",
    ),
    "化学": (
        "化学", "元素", "分子", "反应", "化合物", "酸碱", "有机", "无机",
        "氧化", "还原", "催化剂", "化学键", "原子序数", "周期表", "溶液",
        "摩尔", "离子", "沉淀", "方程式", "官能团",
    ),
    "生物医学": (
        "生物", "细胞", "基因", "dna", "rna", "蛋白质", "细菌", "病毒",
        "医学", "疾病", "治疗", "药物", "人体", "植物", "动物", "进化",
        "酶", "激素", "免疫", "遗传", "染色体", "代谢", "器官", "组织",
        "临床", "症状", "诊断", "疫苗", "神经", "心脏", "血液",
    ),
    "历史人文": (
        "历史", "朝代", "战争", "文明", "考古", "皇帝", "革命", "世纪",
        "文化", "哲学", "社会", "宗教", "民族", "王朝", "帝国", "史书",
        "古代", "近代", "遗址", "文献", "思想", "伦理", "逻辑", "美学",
    ),
    "经济管理": (
        "经济", "市场", "金融", "货币", "投资", "管理", "企业", "公司",
        "营销", "贸易", "供需", "资本", "预算", "成本", "利润", "股票",
        "银行", "财政", "税收", "消费", "生产", "分配", "商业模式", "增长",
    ),
    "文学艺术": (
        "文学", "小说", "诗歌", "散文", "艺术", "音乐", "绘画", "戏剧",
        "电影", "美学", "作家", "作品", "剧本", "旋律", "色彩", "雕塑",
        "书法", "舞蹈", "摄影", "意象", "修辞", "叙事",
    ),
}

#: 暴露领域清单，供 UI/测试/文档使用。
KNOWN_DOMAINS: tuple[str, ...] = tuple(DOMAIN_KEYWORDS.keys())

def classify_domain(text: str, *, title: str = "", default: str = DEFAULT) -> str:
    """把文本归类到最匹配的领域，返回领域名。

    打分规则：统计该领域关键词在 ``text`` 中出现的次数，标题命中额外加权；
    得分最高者胜出；全部为 0 → 返回 ``default``。
    """

    if not text and not title:
        return default
    text_lower = (text or "").lower()
    title_lower = (title or "").lower()
    best_domain, best_score = default, 0
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = 0
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower in title_lower:
                score += DOMAIN_TITLE_WEIGHT
            elif kw_lower in text_lower:
                score += 1
        if score > best_score:
            best_domain, best_score = domain, score
    return best_domain


def majority_domain(domains: list[str], *, default: str = DEFAULT) -> str:
    """取众数领域（文档实体按多数知识块的领域挂恒星系）；空输入回退 default。"""

    if not domains:
        return default
    counts: dict[str, int] = {}
    for d in domains:
        counts[d] = counts.get(d, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


class ClassifyDomainInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    text: str = Field(default="", description="要分类的正文（可与 title 一起给，也可只给一个）。")
    title: str = Field(default="", description="标题或文件名；命中关键词时加权（文件名常含主题词）。")


class ClassifyDomainOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    domain: str = Field(description="命中的领域名；无命中时为「未分类」。")
    matched: bool = Field(description="false 表示所有领域关键词都没命中，domain 是兜底值。")
    known_domains: list[str] = Field(default_factory=list, description="全部可选领域。")


class ClassifyDomainTool(BaseTool):
    spec = ToolSpec(
        name="knowledge.classify_domain",
        description=(
            "Classify text into one of the local topic domains (编程开发/数学/物理/...)"
            " with a deterministic, offline keyword rule. Returns 未分类 when nothing "
            "matches. Use it to decide which star system a note belongs to."
        ),
        version="1.0.0",
        input_model=ClassifyDomainInput,
        output_model=ClassifyDomainOutput,
        side_effect="read",
        permissions=(),
        timeout_seconds=30.0,
        idempotent=True,
        parallel_safe=True,
        tags=("knowledge", "domain", "classify", "read"),
        guidance=(
            "需要把一段文字归到本地领域体系时用它；纯规则、离线、结果可复现。它只分类不写库，分类结果要落库请走 memory.rag 或 knowledge.hybrid_index 的元数据。"
            "文字不属于任何已知领域时会返回未分类，这是正确结果，不要强行套一个领域。"
        ),
    )

    def execute(self, arguments: ClassifyDomainInput) -> ClassifyDomainOutput:
        domain = classify_domain(arguments.text, title=arguments.title)
        return ClassifyDomainOutput(
            domain=domain,
            matched=domain != DEFAULT,
            known_domains=list(KNOWN_DOMAINS),
        )


def create_tool() -> BaseTool:
    return ClassifyDomainTool()


__all__ = [
    "DOMAIN_KEYWORDS",
    "KNOWN_DOMAINS",
    "ClassifyDomainInput",
    "ClassifyDomainOutput",
    "ClassifyDomainTool",
    "classify_domain",
    "create_tool",
    "majority_domain",
]
