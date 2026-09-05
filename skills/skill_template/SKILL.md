---
description: 替换这一行:何时使用本 skill + 覆盖什么内容。这是模型匹配的唯一依据,必须写清使用场景。
version: 1.0.0
triggers: 示例触发词, example trigger
enabled: false
---

# <skill-name>

(把本段替换为给模型的操作指令。正文直接写步骤、规范、示例,不要写成面向用户的介绍文。)

## 使用步骤

1. 第一步……
2. 第二步……

## 示例

(一个具体的输入/输出示例,帮助模型稳定复现期望行为。)

## 参考文件(可选)

需要更长的参考资料时,放到本目录 `references/` 下,并在正文里注明
"详细信息调用 system.skill_catalog(action=read_reference) 读取 references/<文件名>"。
