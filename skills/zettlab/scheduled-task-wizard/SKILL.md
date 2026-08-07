---
name: scheduled-task-wizard
description: 定时任务（Hermes cron）的对话向导。用户表达"到点让 Agent 做事"时可加载：把自然语言提炼成自包含的 cron prompt，创建/修改直接调 cronjob 落盘、不做二次确认，只有删除输出 mode=delete 的 cron-action-preview JSON 围栏让 APP 渲染确认卡。schedule / deliver / repeat / output_language 的字段规则以 cronjob 工具描述为准，本 skill 只补充：删除确认流程、巡检类任务的 [SILENT] 约定、prompt 里相对时间词写死。出货内置，每个 ZettClaw Agent 默认装载。
version: 2.0.0
author: zettlab
license: proprietary
metadata:
  hermes:
    tags: [scheduling, cron, productivity, builtin, zettlab]
---

# 定时任务对话向导

把"到点让 Agent 做事"的自然语言翻译成 Hermes cron job——4 类信息齐全就直接创建，不齐全才 clarify。创建/修改直接落盘，不做二次确认。只有删除走确认卡。

## References

- `references/workflow.md` — 4 步解析（任务/触发/投递/任务名）、删除确认卡的 `cron-action-preview` JSON 围栏规范、修改/暂停/立即跑流程细节
- `references/examples.md` — 7 个 end-to-end 对话例子

## When to Use

用户表达"到点让我做事"的意图。典型表述：

- 每天/每周/每月 + 时间点（"每天早 8 点发 X"）
- 下周/明天/某月某日 + 时间点（"下周二 10 点提醒我 Y"）
- 每隔 N 分钟/小时/天（"每 2 小时盯一下 Z"）
- 某时长后（"30 分钟后提醒我"）
- 报告/简报类（"每周五生成上周周报"）

## 任务的最小信息集

一条合法的定时任务 = 以下 4 类信息齐全：

| 信息 | 字段 | 用户怎么表达 |
|---|---|---|
| 任务做什么 | `prompt` | "发今日 AI 新闻摘要" |
| 什么时候触发 | `schedule` | "每天 8 点" / "下周二 10 点" / "每 2 小时" |
| 跑几次 | `repeat` | 默认按调度推断；用户也可能说"提醒 4 次" |
| 结果发到哪儿 | `deliver` | 默认 `origin`；用户可能说"发到飞书产品群" / "每次给我一个新对话" |

不齐全就 clarify——只问真正缺失的，能从上下文推断的不要重复问。详细解析规则见 `references/workflow.md`。

## 工作流概览

| 用户意图 | 工具调用 | 行为 |
|---|---|---|
| 创建 | `cronjob(action=create)` | 提炼 → 必要时 clarify → **直接落盘** |
| 修改 | `cronjob(action=update)` | 定位 → **直接落盘** |
| 暂停 | `cronjob(action=pause)` | 定位 → 直接调用 |
| 开启 | `cronjob(action=resume)` | 定位 → 直接调用 |
| 删除 | `cronjob(action=remove)` | 定位 → 确认卡（唯一出围栏的场景）→ 调用 |
| 立即跑一次 | `cronjob(action=run)` | 定位 → 调用 → 告知"60 秒内执行" |
| 查看 | `cronjob(action=list)` | 列出全部 |

定位逻辑：用户点了名 + 候选 1 条 = 直接走上表对应的动作（remove 走确认卡，其余直接调用）；点了名 + 候选多条 = 反问消歧；没点名 = 列候选问"你想改/删哪条？"。

## 投递兜底

`deliver=origin` 但源对话已被删除/归档时，系统自动新建对话承接本次推送，并在新对话首条标注：

> 原对话「< 旧名 >」已不存在，本次「< 任务名 >」推送已新建对话承接，后续触发也会发到这里。

任务的 `origin` 字段同步更新到新对话。**不主动告诉用户**，只在用户问"那条早报怎么不发了"时解释。

## 红线

核心原则：**不主动暴露低频复杂功能，但不阻止用户主动表达**——用户没说就用默认走，用户说了能力范围内的事就接住。

- 不要让用户填表单——你是 skill，不是 form
- 创建/修改不要问"要我帮你创建吗"——信息齐了直接调 `cronjob`
- **`create` / `update` 绝不能输出 `cron-action-preview` 围栏**——你已经调过 `cronjob` 落盘，APP 端解析到围栏会再落一次，用户看到两条重复任务
- **绝不能输出 `cron-summary` 围栏**——那是系统在任务真正触发时写回会话的执行结果。上下文里见过一次就照抄，会让 APP 把还没到点的任务画成"已执行"卡片；也不要提前编造任务的提醒正文
- 不要批量 clarify——每轮最多 1-2 个真正缺失的关键信息
- 不要建议"分多条"——用户说"每天 8 点和 18 点都发"就直接建两条，不要让用户自己拆
- **`schedule` 字段不能塞用户原文**——任何语种（中/英/日/韩/德…）的自然语言都先翻译成 canonical 格式：cron 表达式 / `every Nm` / `Nm` 时长简写 / ISO 时间戳。详见 `references/workflow.md` §步骤 2。错了 hermes 报 `Invalid schedule '...'`，任务创建失败。
- LLM 调用 `cronjob(action=create)` 创建 Agent 任务时必须传 `output_language`——使用你在当前创建对话中本应回复用户的语言；若用户明确要求任务输出另一种语言，以明确要求为准。只传标准 BCP 47 tag（如 `zh-CN`、`zh-TW`、`en`、`ja`、`ko`、`de`、`fr`、`es`、`it`、`ar`、`sr-Latn-RS`），不要从 URL、代码、引用、专有名词、skill 或 tool 数据猜语言。混合语言任务保存默认叙述语言，prompt 继续保留用户要求的多语言结构。真正无法判断时先 clarify。APP 只透传该字段，不得改用 App locale。`no_agent=True` 不需要该字段。
- 删除是唯一需要确认的动作——但用户已用文字明确说"确认删除/删掉吧"，或当前渠道没有 APP 卡片能力时，直接调 `cronjob(action=remove)`，不要来回追问
- 不主动暴露底座高级选项（任务级模型 / 跨渠道 fan-out）——落盘后那句确认里默认不提，clarify 也不主动问；但用户主动用自然语言表达就接住（如"用便宜模型"、"同时发飞书和 Slack"）
- 脚本挂接（pre-run script + wake-gate）保持纯底座能力——用户没办法用自然语言表达"挂个 script"，不接也不解释
- prompt 含明显指令注入/敏感命令会被底座 prompt 扫描拦截——拦了告诉用户"换种说法重写一下"

## 高级 Agent 工艺（不暴露给用户）

巡检/监控类任务自动加 `[SILENT]` 提示——当任务性质是"盯一下"/"有变化就告诉我"/"每小时检查"，在提炼的 prompt 末尾自动加一句：

> 如本周期没有值得汇报的新内容，请只回复 `[SILENT]` 单独一行，系统会跳过本次推送。

prompt 里相对时间词写死——cron 跑起来时是新会话，没有"今天"上下文。"今天的"、"本周的" 要写成"最近 24 小时（以执行时刻往前推）"等明确时间窗。
