# Skills 指令包编写指南

Skill 是**只读指令包**:模型按需阅读的 Markdown 玩法手册,不是可执行代码。
Agent 启动时扫描本目录下的一级子目录,每个包含 `SKILL.md` 的子目录注册为一个 skill。

## 运行机制(按需查看)

1. 每次模型请求都会在系统消息里看到技能目录:`名字 + 版本 + 描述 + 触发词`,**不含正文**。
2. 模型判断当前任务命中某个 skill 的描述或触发词后,调用
   `system.skill_catalog`(`action: view`, `skill_name: <名字>`)读取完整正文。
3. `view` 返回的正文出现在对话后段,不破坏提示词前缀缓存;
   skill 增删改会自动改变 prompt cache key,开启新的缓存命名空间。

## 目录结构

```
skills/
├── README.md            # 本文件
├── skill_template/      # 模板(复制后改名,enabled: false 保持默认禁用)
│   └── SKILL.md
└── <skill-name>/
    ├── SKILL.md         # 必需:frontmatter + 正文
    └── references/      # 可选:参考文件,模型可用 read_reference 读取
        └── example.md
```

## SKILL.md 格式

frontmatter 使用最简单的 `key: value` 格式,以两行 `---` 包裹:

```markdown
---
name 可省略;目录名就是 skill 名(小写字母/数字/连字符,最长 64 字符)
description: 一行说明。写清"何时使用 + 覆盖什么内容",这是模型匹配的唯一依据。
version: 1.0.0
triggers: 触发词1, trigger two, 触发词三
enabled: true
---

(正文:给模型的操作步骤、格式规范、示例……使用中文或英文均可)
```

### 字段说明

| 字段 | 必需 | 说明 |
|---|---|---|
| `description` | 是 | 非空,最长 2000 字符。**决定 skill 会不会被触发**,务必写场景 |
| `version` | 否 | 默认 `1.0.0`,最长 32 字符 |
| `triggers` | 否 | 逗号分隔字符串或列表,每条最长 200 字符 |
| `enabled` | 否 | `false` 时该 skill 被跳过(注册状态 `disabled`) |

未知字段会直接报错(对齐工具契约的 `extra="forbid"` 风格),防止拼写错误静默丢失。

## description 写法

❌ 差:`关于论文审稿的内容` —— 模型无法判断何时该用。

✅ 好:`当用户要求撰写、修改或回应论文审稿意见时使用;涵盖常见审稿维度、结构化写法与回复信模板。`

## 约束

- 正文是**给模型读的指令**,不是给用户的文档;直接写步骤和示例。
- skill 不会执行任何代码;需要可执行能力时请走 `tool/` 单文件工具协议。
- `references/` 下的文件只能被 `system.skill_catalog`(`action: read_reference`)读取,
  且路径不得越出 skill 目录(防路径穿越)。
