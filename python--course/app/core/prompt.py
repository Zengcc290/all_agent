"""系统提示词：所有可用工具的作用与调用参数都从这里动态生成，绝不写死。"""
from __future__ import annotations

from app.core.registry import registry

SYSTEM_PROMPT = """你是一个「知识图谱抽取与入库助手」，负责从用户给出的文本中抽取实体与实体关系，并调用工具完成入库。

# 工作原则
1. 只输出一个 JSON 对象（或连续多个 JSON 对象），不要输出自然语言解释、不要加 markdown 代码围栏。
2. 参数必须满足各工具 schema 中标注 [必填] 的项；标 [可选] 的可以省略。
3. 互不依赖的多个工具可以并行调用。
4. 抽取实体关系要「越准确越详细越好」：
   - 已有的实体必须复用（用系统中已有的名字，不要另起别名）；
   - 不存在的实体可以新建；
   - 一个实体可以在多条关系里被重复利用；
   - 关系要写含义清晰的中文谓词，并标注 directed（true=有向，false=无向/双向）。
5. 时间抽取：检测一句话里是否有时间（年/月/日/时，具体到哪一级都可以）；
   有时间就原样抽取填到 time 字段；**没有时间就留空字符串，流程图会自动调用 get_current_time 兜底**，
   你不允许自己编造时间。

# 输出格式（严正要求）
{"tool": "<工具名>", "args": { ... }}
"""


def build_tool_prompt() -> str:
    """把 registry 里当前所有工具渲染成提示词片段。
    新增工具只需在 app/tools/ 下加文件，下一次调用此函数即自动出现在提示词里。"""
    return registry.describe()


def build_full_prompt() -> str:
    return f"{SYSTEM_PROMPT}\n\n{build_tool_prompt()}"


def build_extract_messages(text: str, existing_entities: list[dict],
                           existing_relations: list[dict] | None = None) -> list[dict]:
    """构造「一句话入库」的抽取提示词。"""
    ent_lines = []
    for e in existing_entities[:200]:
        aliases = e.get("aliases") or []
        alias_txt = f"（别名：{'、'.join(aliases)}）" if aliases else ""
        ent_lines.append(f"- {e.get('name')} | 类型:{e.get('type') or '未知'}{alias_txt}")
    ent_block = "\n".join(ent_lines) if ent_lines else "（当前图库中还没有任何实体）"

    rel_block = ""
    if existing_relations:
        rl = [f"{r.get('src_name') or r.get('src')} -[{r.get('predicate')}]-> {r.get('tgt_name') or r.get('tgt')}"
              for r in existing_relations[:100]]
        rel_block = "\n\n# 已存在的关系（供参考，避免冲突）\n" + "\n".join(rl)

    user = f"""请从下面这句话中抽取全部实体与实体关系，并尽量详尽准确。

要抽取的原文：
{text}

{rel_block}

# 系统中已经存在的实体（请优先复用这些名字，不要另造别名）
{ent_block}

# 输出要求
- 只输出一个 JSON 对象，格式严格如下，不要任何额外文字：
{{"entities": [
   {{"name": "实体名", "type": "类型", "key": "稳定标识(小写英文或拼音)", "aliases": ["别名"], "time": "时间或空字符串"}}
 ],
 "relations": [
   {{"source": "实体名", "target": "实体名", "predicate": "中文谓词",
     "directed": true, "time": "时间或空字符串", "evidence": "支持这句话的原文片段"}}
 ]}}
- 已有实体必须复用其名字；不存在的实体可以新建。
- 一个实体可以被多条关系重复引用。
- time：检测到时间就填（可到年/月/日/时），没有时间一律填空字符串 ""，由系统自动取当前时间。
- directed：单向语义（如「属于」「成立于」「治疗」）填 true，双向对称语义（如「同为」「合作」「相似」）填 false。
"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
