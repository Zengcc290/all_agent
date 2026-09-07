---
description: 当 AI 完成任何实际项目修改（代码、配置、依赖、接口、数据结构、脚本、资源、文档或行为变化）后，需要按项目协议留痕并交付时使用;覆盖 system.update_log 强制写入、update_log_readme_first.md 索引维护、git 提交与远端推送的完整流程、提交消息格式与失败处理。
version: 1.0.0
triggers: 更新日志, 提交, 推送, 交付, 修改代码, update_log, git commit, git push, 远端同步, commit message
enabled: true
---

# 代码变更交付（code-change-delivery）

把"修改 → 留痕 → 提交 → 推送"固化为强制行为。凡是产生了实际文件或外部状态变化的任务，只有在以下流程全部成功后才算完成；只读、分析、解释类请求不适用本技能。

## 何时使用

- 你刚完成（或即将完成）任何项目文件或行为的修改。
- 用户要求"提交""推送""交付""留痕""同步远端"本次改动。
- 需要核对某次改动的交付状态或排查交付失败。

## 前置检查（每次必做）

1. 读取 `update_log_readme_first.md`（用 `fs.read_text`），确认当前"已写入的最后更新 ID"与"下一次必须写入的更新 ID"。不要一次性把 SQLite 历史全部加载进上下文。
2. 用 `git.status` 查看工作区：branch、staged / unstaged / untracked / conflicted，确认本次涉及哪些文件。
3. 明确本次改动的性质：feature / fix / refactor / config / docs / demo 等。

## 交付流程（严格按序执行，缺一不可）

### 第 1 步：验证修改

用 `shell.run` 运行相关检查并记录真实结果：

- 定向单测：`python -m pytest tests/test_<模块>.py -q`
- 全量回归：`python -m pytest -q`
- 静态检查：`python -m ruff check <改动的文件>`（若项目配置了 ruff）

失败必须修复后重跑，不得用"应该能过"代替真实验证。将真实结果写入交付说明。

### 第 2 步：调用 system.update_log 留痕（强制，一次）

修改完成后必须主动调用 `system.update_log` 一次，如实填写全部字段：

| 字段 | 要求 |
| --- | --- |
| executor | 执行者，例如 `deepseek-v4-flash (dsh coding agent)` |
| update_type | feature / fix / refactor / config / docs / demo 等 |
| title | 简短标题，建议与后续 commit 标题一致 |
| task_background | 为什么改、目标是什么 |
| update_details | 具体实现细节与关键决策 |
| added_features | 新增能力；无则写 `none` |
| files | 至少 1 项，每项含 path（项目相对路径）、action（added/modified/deleted/renamed/generated）、description |
| behavior_impact | 兼容性/API/配置/数据/部署影响；无则 `none` |
| validation | 实际运行的测试与真实结果 |
| risks | 已知风险与回滚说明；无则 `none` |
| follow_up | 剩余工作；无则 `none` |

约束：

- 不要手写 SQL、直接改数据库或伪造 ID；ID 由工具在事务中生成。
- 工具只返回 `update_id` / `next_update_id` 等紧凑确认，不返回历史正文。
- 若调用失败（例如缺确认），取得该工具当前注册代次的确认后重试；仍失败必须向用户报告阻塞原因，不得假装完成。

### 第 3 步：更新索引文件

用工具返回的 `next_update_id` 更新 `update_log_readme_first.md` 的"当前索引"两行（用 `fs.edit_text` 精确替换）：

- "已写入的最后更新 ID" ← 返回的 `update_id`
- "下一次必须写入的更新 ID" ← 返回的 `next_update_id`

该索引更新属于刚记录的同一任务，**不需要**再次调用 `system.update_log`。只改索引两行，不要改动文件其他规则，也不要复制数据库历史正文。

### 第 4 步：git 提交

用 `git.commit_push`，message 固定使用刚返回的 `update_id`，格式：

```
update-log-<update_id>: <简短标题>
```

例如：`update-log-52: 新增 code-change-delivery 与 tool-authoring 技能`

- 提交内容必须包含 `update_log.sqlite3` 与 `update_log_readme_first.md`（索引更新）。
- 建议用 `paths` 显式列出本次涉及的文件，避免把无关的临时文件、密钥或 `.env` 带入提交；如确有其他未跟踪文件属于本任务，一并列出。
- 不得使用下一次编号或自行编造编号。

### 第 5 步：推送远端

`git.commit_push` 默认 push=true；若分支无上游会自动使用 `-u` 设置。推送失败时读取 `push_output` 诊断原因（无凭据/网络/无远端），处理后再推送；无法解决时如实向用户报告，在远端同步成功前不得声称任务已完成。

## 验收清单（全部满足才算完成）

- [ ] 修改已通过验证（pytest / ruff 真实通过）
- [ ] `system.update_log` 成功调用，拿到本次 `update_id`
- [ ] `update_log_readme_first.md` 索引已更新且与数据库一致
- [ ] commit 消息为 `update-log-<update_id>: ...`，提交包含 `update_log.sqlite3` 与索引文件
- [ ] 推送成功（或已向用户明确报告失败原因）
- [ ] 最终回复包含：改了什么、验证结果、update_id

## 失败处理速查

| 情况 | 处理 |
| --- | --- |
| `system.update_log` 报 CONFIRMATION_REQUIRED | 用 `agent.tools.confirmation_key("system.update_log")` 构造确认后重试 |
| pytest 失败 | 修复后重跑，禁止跳过验证直接交付 |
| commit 失败 | 读 stderr 诊断（分支、索引冲突等），解决后重试 |
| push 失败 | 读 push_output：无凭据→配置凭据助手；无远端→检查 remote；无上游→用 -u |
| 工作区有无关未跟踪文件 | 用 paths 显式选择，不提交无关文件 |

## 与工具集的关系

本技能依赖 P0 工具集落地：`git.status` 看工作区、`fs.read_text`/`fs.read_dir` 核对文件、`shell.run` 跑验证、`git.commit_push` 完成提交推送。不要绕过这些工具手工执行 git 或写库。
