# 中文检索失效的根因与修复

## 问题

`memory/embedding.py` 中 `HashEmbedding.tokenize()`（合入 `python-course-design` 前名为
`TFIDFEmbedding`）使用：

```python
return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)
```

`\w` 在 Unicode 模式下匹配中文，且中文没有空格分隔，所以**一整段连续中文只会产出 1 个 token**。
查询与文档里的中文表述只要不完全一致，token 就完全不相交，中文检索实质失效。

## 修复方案：CJK 二元组切分

CJK 按二元组（bigram）切分，拉丁/数字仍按词切分。这是 Lucene `CJKAnalyzer` 的经典做法，
不依赖外部分词词典，向量维度保持稳定，对词表外的新词也鲁棒。

### 直接替换（memory/embedding.py）

在 import 区之后加入：

```python
CJK_RANGES = (
    f"{chr(0x3400)}-{chr(0x4dbf)}"
    f"{chr(0x4e00)}-{chr(0x9fff)}"
    f"{chr(0xf900)}-{chr(0xfaff)}"
)
_CJK_BLOCK = re.compile(f"[{CJK_RANGES}]+", re.UNICODE)
_CJK_OR_WORD = re.compile(f"[{CJK_RANGES}]+|[\\w]+", re.UNICODE)
```

把 `HashEmbedding.tokenize` 改为：

```python
    @staticmethod
    def tokenize(text: str) -> list[str]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        tokens: list[str] = []
        for match in _CJK_OR_WORD.finditer(text.casefold()):
            token = match.group(0)
            if _CJK_BLOCK.fullmatch(token):
                if len(token) == 1:
                    tokens.append(token)
                else:
                    tokens.extend(token[i:i + 2] for i in range(len(token) - 1))
            else:
                tokens.append(token)
        return tokens
```

> 注意：`_CJK_OR_WORD` 里 CJK 分支必须排在 `\w+` 之前，否则 `\w` 会把中文整段吃掉。

## 验证

```bash
python eval/retrieval_eval.py
```

| 语料 | 分词器 | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|---|
| EN | baseline | 0.5882 | 0.8235 | 0.9412 | 0.7196 |
| EN | CJK bigram | 0.5882 | 0.8235 | 0.9412 | 0.7196 |
| **ZH** | baseline | 0.0909 | 0.2727 | **0.5455** | 0.2409 |
| **ZH** | **CJK bigram** | **0.6364** | **0.7273** | **0.7273** | **0.6818** |

- 中文 Recall@1：0.0909 → 0.6364（**+54.5 个百分点**）
- 中文 MRR：0.2409 → 0.6818（**+183%**）
- 英文：四项指标完全无变化（无回退）

## 后续可做

1. 加单字（unigram）索引做混合，进一步覆盖单字查询
2. 切到 `APIEmbedding` 真实嵌入，用同一评测集对比词法检索 vs 语义检索
3. 评测集扩到 60~100 条，加入长文档与多跳事实问答
