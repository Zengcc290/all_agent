# 中文检索失效的根因与修复

## 问题

`memory/embedding.py` 的 `TFIDFEmbedding.tokenize()` 使用：

```python
return re.findall(r"[\w]+", text.casefold(), flags=re.UNICODE)
```

`\w` 在 Unicode 模式下匹配中文，且中文没有空格分隔，所以**一整段连续中文只会产出 1 个 token**。
查询与文档里的中文表述只要不完全一致，token 就完全不相交，中文检索实质失效。

## 修复方案：CJK 二元组切分

CJK 按二元组（bigram）切分，拉丁/数字仍按词切分。这是 Lucene `CJKAnalyzer` 的经典做法，
不依赖外部分词词典，向量维度保持稳定，对词表外的新词也鲁棒。

### 直接替换（memory/embedding.py）

在文件顶部 import 区加入：

```python
CJK_RANGE = r"\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
_CJK_BLOCK = re.compile(rf"[{CJK_RANGE}]+", re.UNICODE)
_CJK_OR_WORD = re.compile(rf"[{CJK_RANGE}]+|[\w]+", re.UNICODE)
```

把 `TFIDFEmbedding.tokenize` 改为：

```python
    @staticmethod
    def tokenize(text: str) -> list[str]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        tokens: list[str] = []
        for match in _CJK_OR_WORD.finditer(text.casefold()):
            tok = match.group(0)
            if _CJK_BLOCK.fullmatch(tok):
                if len(tok) == 1:
                    tokens.append(tok)
                else:
                    tokens.extend(tok[i:i + 2] for i in range(len(tok) - 1))
            else:
                tokens.append(tok)
        return tokens
```

> 注意：`_CJK_OR_WORD` 里 CJK 分支必须排在 `[\w]+` 之前，否则 `\w` 会把中文整段吃掉。

## 验证

```bash
python eval/retrieval_eval.py
```

结果（同一评测集、同一语料、同一 chunk 参数）：

| 语料 | 分词器 | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|---|
| EN | baseline | 0.6000 | 0.8500 | 0.9000 | 0.7183 |
| EN | CJK bigram | 0.6000 | 0.8500 | 0.9000 | 0.7183 |
| **ZH** | baseline | 0.1818 | 0.2727 | **0.2727** | 0.2121 |
| **ZH** | **CJK bigram** | **0.3636** | **0.6364** | **0.7273** | **0.4879** |

- 中文 Recall@5：0.2727 → 0.7273（**+45.5 个百分点**，相对提升约 2.7 倍）
- 中文 MRR：0.2121 → 0.4879（**+130%**）
- 英文：完全无变化（无回退）

## 后续可做

1. 加单字（unigram）索引做混合，进一步覆盖单字查询
2. 接 DashScope / 本地 Transformer 真实嵌入，用同一评测集对比语义检索 vs 词法检索
3. 评测集扩到 60~100 条，加入长文档与多跳事实问答
