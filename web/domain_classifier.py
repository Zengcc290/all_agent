"""本地主题领域分类器：把知识块/文档内容自动归类到恒星系（领域）。

设计目标：
- 零依赖、纯规则，不联网、不消耗 LLM key（未配置 DASHSCOPE/DEEPSEEK key 也能用）；
- 确定性：同样的文本永远得到同样的领域（跨进程稳定，可测试）；
- 轻量：一次遍历关键词表 O(len(keywords))，万级节点构建时可忽略不计。

用法：:

    from web.domain_classifier import classify_domain
    domain = classify_domain("Python 的函数与变量", title="c语言笔记.txt")

返回 ``KNOWN_DOMAINS`` 中的领域名；命不中任何主题时返回 DEFAULT（"未分类"）。
"""

from __future__ import annotations

#: 兜底领域：任何主题关键词都没命中时的归宿。
DEFAULT = "未分类"

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

#: 标题命中的加权系数（文件名常含主题词，如“c语言笔记.txt”）。
_TITLE_WEIGHT = 3


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
                score += _TITLE_WEIGHT
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
