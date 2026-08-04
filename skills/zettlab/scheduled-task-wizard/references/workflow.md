# 工作流细节

主入口：`../SKILL.md`

---

## 1. 创建任务：4 步解析

### 步骤 1：识别"任务做什么"（`prompt`）

把用户原话**提炼成自包含的指令**——cron 跑起来时是新会话，没有当前上下文，指令必须明确。

例子：

- 用户："每天早上给我发 AI 新闻"
- 提炼：`"用 web_search 搜索最近 24 小时内的全球 AI 重要新闻，按重要性排前 5 条，每条一句话总结+链接。用中文输出。"`

规则：

- 不要原话照抄——"AI 新闻"太模糊，cron 跑起来 Agent 不知道用什么工具
- 要补足执行细节——用什么工具、输出格式、语言、长度
- 跨周期内容（"昨天的"/"上周的"）要写成明确时间窗

### 步骤 2：识别"什么时候触发"（`schedule`）+ clarify 重复次数

> ⚠️ **`schedule` 字段格式硬约束**：必须是下列 4 种 canonical 格式之一，**任何语言（中文/英文/日文/韩文/德语…）的自然语言原文都不被 hermes 接受**，直接 emit 会让 `cronjob(action=create/update)` 报 `Invalid schedule '...'` 500。
>
> | kind | 格式 | 例子 |
> |---|---|---|
> | once | ISO 8601（带不带 offset 都行） | `2026-04-22T10:00` / `2026-04-22T10:00:00+08:00` |
> | once | 时长简写（从现在起算） | `30m` / `2h` / `1d` |
> | recurring | `every Nm` / `every Nh` / `every Nd`（注意必须英文 `every`） | `every 30m` / `every 2h` |
> | recurring | cron 表达式（5 字段 `M H DOM MON DOW`） | `0 9 * * *` / `0 18 * * 5` / `0 18 * * 1-5` |
>
> **你的任务**：拿到任何语种的用户输入，**先在脑子里翻译成上面四类**，再 emit。不要原文塞 `schedule` 字段。

| 用户说 | `schedule` | `repeat` |
|---|---|---|
| "下周二 10 点..." / "5 月 1 日 10 点..." | ISO 时间戳，如 `2026-04-22T10:00` | `1` |
| "30 分钟后..." / "2 小时后..." / "1 天后" | 时长简写 `30m` / `2h` / `1d` | `1` |
| "每天 9 点..." / "每周五 18 点..." / "工作日 18 点" | cron 表达式 `0 9 * * *` / `0 18 * * 5` / `0 18 * * 1-5` | `null`（永远） |
| "每 30 分钟" / "每隔 30 分钟" / "每 30 分钟一次" / "每 30 分钟一回" | `every 30m`（**不是**"每 30 分钟"原文！） | `null` |
| "每 2 小时" / "每隔 2 小时" / "每 2 小时一次" / "每 2 时" | `every 2h` | `null` |
| "每天每 30 分钟"（"每天"是冗余） | `every 30m` | `null` |
| "Every 30 minutes" / "30 分ごと" / "30분마다" | `every 30m`（同语义跨语言归一） | `null` |

**反例（千万别这么写）**：

| ❌ 错误 emit | 错在哪 | 正确 emit |
|---|---|---|
| `"schedule": "每 30 分钟一次"` | 原文中文 | `"schedule": "every 30m"` |
| `"schedule": "明天下午三点"` | 中文日期 | `"schedule": "<按当前日期计算后的 ISO 时间戳>"` |
| `"schedule": "30 分钟后"` | 中文时长 | `"schedule": "30m"` |
| `"schedule": "每 30m"` | 中英混 | `"schedule": "every 30m"` |
| `"schedule": "每天上午 10:30"` | 中文 + 时刻 | `"schedule": "30 10 * * *"` |

**必须主动 clarify 的两类情况**：

| 用户说法 | 必须问 |
|---|---|
| "今天每隔 2 小时提醒我喝水" | "今天一天大约 5-6 次？还是想一直循环到我手动停？" |
| "本周每天 18 点检查需求池" | "只本周这一周，下周一就停？还是想长期跑下去？" |
| "下周提醒我..." | "下周哪一天？" |
| "上午发给我" | "几点合适？" |
| "周末..." | "周六还是周日？还是两天都要？" |

已经清楚的不要重复问——"每天早 8 点" 直接 `0 8 * * *`，不要再问"具体几点"。

### 步骤 3：识别"发到哪儿"（`deliver`）

三档投递模式——大多数情况下默认即可，**只有用户主动表达不一样的需求时才走非默认**：

| 模式 | `deliver` 值 | 用户表达 | 卡片展示 |
|---|---|---|---|
| 回当前对话（默认）| `origin` | 没说 / "发回这里" / "就在这" | 📍 当前对话：< 对话名 > |
| 每次新建对话 | `new_session` | "每次给我开个新对话" / "每天一个独立对话" | 🆕 每次执行新建对话承接 |
| 指定其它对话 | `<platform>:<chat_id>` | "发到飞书产品群" / "发到 APP 的 XX 对话" | 🎯 < 平台 > / < 对话名 > |

指定其它对话时定位顺序：

1. 调 `channel_directory_lookup` 按用户给的关键词搜
2. 命中 1 条 → 卡片直接填
3. 命中多条 → 反问"是 A 还是 B？"
4. 命中 0 条 → "我看到你有这些可发送目标：[列表]，要哪个？"

**不主动建议"发到别处"**——大多数预期就是"在哪儿建发回哪儿"，多问一次反而打扰。

**用户主动表达多目的地**（"同时发飞书和 Slack" / "也给我发一份到产品群"）→ 接住，落到 `deliver` 逗号分隔多目的地（如 `origin,feishu:产品群`），卡片用多行展示投递目标：

```
发送到：
  📍 当前对话
  🎯 飞书 / 产品群
```

但不要主动建议 fan-out。

### 步骤 4：起一个任务名（`name`）

从 `prompt` 提一句简短描述，10 字以内。不让用户自己起。

- 搜全球 AI 新闻 → `AI 新闻早报`
- 总结上周工作周报 → `上周工作周报`
- 提醒领保险金 → `保险金领取提醒`

---

### 创建时派生字段：执行结果语言（`output_language`）

Agent cron 会在触发时启动全新 session，创建对话不会自动带过去。调用 `cronjob(action=create)` 时必须单独保存本次任务的默认输出语言：

- 使用你在当前创建对话中本应回复用户的语言，而不是 App 系统语言。
- 用户明确要求任务用另一种语言输出时，以明确要求为准。
- 只传标准 BCP 47 tag，例如 `zh-CN`、`zh-TW`、`en`、`ja`、`ko`、`de`、`fr`、`es`、`it`、`ar`、`sr-Latn-RS`。
- URL、代码、引用、专有名词、skill 内容和 tool 返回数据都不是语言依据。
- 对混合语言任务，保存默认叙述语言；prompt 中继续写清哪些部分需要使用其它语言。
- 真正无法判断时 clarify，不要传 `und`、`mul`、`zxx` 或私有标签。
- `no_agent=True` 的脚本任务不需要该字段。

例：中文对话里创建一个 prompt 只有 URL 的摘要任务，也要传 `"output_language": "zh-CN"`。中文对话中用户明确说“结果请用英文”，则传 `"output_language": "en"`。

## 2. 创建预览卡片（APP 交互式路径优先）

当用户正在 APP / zet_agent 里用自然语言创建任务，且还没有明确确认时，优先把 4 类信息提炼成结构化卡片让用户过目。卡片有**两部分**：

**(a) 人话 markdown 卡片**——webui / 不支持结构化卡的客户端看得到：

```
🕐 < 任务名 >

触发：< 人话描述，如"每天 09:00"或"下周二 10:00 一次性"或"每周一 09:00 共 4 次" >

需要 Agent 做：
< 提炼后的 prompt 全文 >

发送到：< 三档之一的描述 >

[创建] [修改] [取消]
```

**(b) 紧跟其后的 `cron-action-preview` JSON 围栏**——APP 端 fence parser 用它渲染可交互卡片：

````
```cron-action-preview
{
  "mode": "create",
  "name": "<任务名>",
  "schedule": "<hermes schedule: cron expr / ISO 时间 / every Nm 等>",
  "cronExpr": "<可选，cron 表达式形态>",
  "schedule_human": "<人话描述，与卡片"触发"一致>",
  "prompt": "<提炼后的 prompt 全文>",
  "output_language": "<LLM 从当前创建对话推断的 BCP 47 tag；mode=create 必填>",
  "deliver": {
    "mode": "origin" | "new_session" | "specified",
    "chatName": "<对话名，origin/specified 模式必填>",
    "sendTo": {
      "channel": "<feishu / slack / zettlab_app / ...>",
      "chatName": "<目标对话名>",
      "chatType": "private" | "group"
    }
  },
  "repeat": { "times": <null|1|N>, "completed": 0 }
}
```
````

落盘动作（**三条路并存**）：

**A. APP 路径（用户点真按钮）—— 你不会观察到，但任务被建好了**

用户在 APP 端点 `[创建]` 按钮，APP 直接走 `cronjob(action=create, ...)` 落盘，**完全不 ping 你**。卡片在 APP 端切到"✅ 已创建"小条就是用户反馈。这条路下，你的下一轮 input 会是用户的下一句话（可能是新话题，也可能跟刚才那条任务无关）—— **不要再为刚刚那条任务说一句"已创建"**，因为它已经在 APP 端反馈过了，你再说一遍是冗余；更不要再调 `cronjob(action=create)` 重建，会出现重复任务。如果用户后续问起，你可以调 `cronjob(action=list)` 确认任务真的在。

**B. Fallback 路径（用户敲字而不是点按钮，例如在 webui 上）**

- 用户回 "创建" / "确认" → 你调 `cronjob(action=create, ...)` 落盘 → 一句话确认（含下次执行时间）
- 用户回 "修改 XXX" → 重出一张带 diff 的卡片（带 `cron-action-preview` 围栏 mode=edit），不要直接落盘
- 用户回 "取消" → 不创建，"好的，没问题"

**C. 直接落盘路径（不要为了卡片阻塞任务）**

以下情况可以直接调 `cronjob(action=create/update/remove)`，不需要再生成预览卡片：

- 用户已经在文字里明确确认（"确认创建"、"就这样"、"删掉吧"）。
- 当前渠道不支持 APP 结构化卡片，或者你判断用户只需要普通文本反馈。
- 这是 APP 按钮确认后的后台执行路径，系统/客户端已经完成用户确认。
- 用户明确要求"直接创建/不要再确认"。

直接落盘时也必须遵守 §1 的 canonical `schedule` 规则；不要传中文、英文或其它自然语言原文。

卡片规则：

- 卡片里 `prompt` 字段展示提炼后的版本，不展示用户原话——便于用户 review 提炼是否到位
- 重复次数不是默认（一次/永远）就显式写出来——"共 4 次" / "持续 7 天" / "本周每天"
- 投递模式显式标记——避免用户以为发哪都行实际只发到了当前对话
- **JSON 围栏的字段值必须跟 markdown 卡片一一对应**——APP 端用 JSON，webui 用 markdown，两边数据要一致

---

## 3. 修改任务

用户说"把那条早报改到 9 点" / "改发到飞书产品群" / "再加一句帮我总结重点"——

1. **定位目标任务**：调 `cronjob(action=list)` 拿全部
   - 用户点了名 + 候选 1 条 → 直接进确认
   - 用户点了名 + 候选多条 → 反问"是每天 8 点那条还是周报？"
   - 用户没点名 → "你想改哪条？我看到你有：[列表]"
2. **生成对照卡片**——分两部分：

   **(a) 人话 markdown 对照卡片**：

   ```
   任务：AI 新闻早报
   触发：~~每天 08:00~~ → 每天 09:00
   需要 Agent 做：（未改）
   发送到：（未改）
   [确认] [取消]
   ```

   **(b) 紧跟其后的 `cron-action-preview` JSON 围栏**（`mode=edit`、必带 `job_id`、`changedFields` 列出变更字段、`previousValues` 给旧值用于 strikethrough diff）：

   ````
   ```cron-action-preview
   {
     "mode": "edit",
     "job_id": "<目标 job_id>",
     "name": "AI 新闻早报",
     "schedule": "0 9 * * *",
     "schedule_human": "每天 09:00",
     "prompt": "<未改时也写全量>",
     "deliver": { "mode": "origin", "chatName": "<对话名>" },
     "repeat": { "times": null, "completed": 0 },
     "changedFields": ["schedule"],
     "previousValues": { "schedule": "每天 08:00" }
   }
   ```
   ````

3. 落盘动作分三条路（同 §2）：
   - **APP 路径**：用户点 `[确认]` 按钮 → APP 直接走 `cronjob(action=update, job_id=..., ...)` 落盘，**不 ping 你**；卡片切到"✅ 已修改"小条就是反馈。下一轮 input 别再回"已修改"，更别重调 cronjob。
   - **Fallback**：用户敲字 "确认" → 你调 `cronjob(action=update, ...)` → 一句话确认（含下次执行时间）
   - **直接落盘**：用户已经明确授权修改，或当前渠道不支持卡片 → 直接调 `cronjob(action=update, ...)`，不要再生成卡片阻塞。

任何字段都可改：触发规则 / 任务名 / `prompt` / 投递目标 / 重复次数。

---

## 4. 暂停 / 开启 / 删除 / 立即跑一次

| 用户意图 | 调用 | 后续话术 |
|---|---|---|
| 暂停 | `cronjob(action=pause, job_id=...)` | "好，已暂停。回来再说一声开启。" |
| 开启 | `cronjob(action=resume, job_id=...)` | "好，已开启。下次 < 时间 > 执行。" |
| 删除 | 需要确认；APP 可走删除卡片，已确认或不支持卡片时直接 `cronjob(action=remove, job_id=...)` | "确定删除「< 任务名 >」吗？运行历史会一起清掉。" → 确认 → "删了。" |
| 立即执行一次 | `cronjob(action=run, job_id=...)` | "好——会在下次调度心跳（最多 60 秒内）执行一遍，结果按你设定的方式投递。" |

定位任务的方式同 §3。

### 删除任务的二次确认卡片

删除优先走"确认卡片 → 用户点[确认删除] → 落盘"流程；如果用户已经用文字明确确认删除，或当前渠道不支持 APP 卡片，就直接调用 `cronjob(action=remove, job_id=...)`。确认卡片同样分两部分：

**(a) 人话 markdown 卡片**：

```
确定删除「AI 新闻早报」吗？
触发：每天 09:00
运行历史会一起清除。
[确认删除] [取消]
```

**(b) `cron-action-preview` JSON 围栏**（`mode=delete`、必带 `job_id`、其它字段填用于显示）：

````
```cron-action-preview
{
  "mode": "delete",
  "job_id": "<目标 job_id>",
  "name": "AI 新闻早报",
  "schedule": "0 9 * * *",
  "schedule_human": "每天 09:00",
  "prompt": "<原 prompt 全文>",
  "deliver": { "mode": "origin", "chatName": "<对话名>" }
}
```
````

落盘动作分三条路（同 §2 / §3）：

- **APP 路径**：用户点 `[确认删除]` 按钮 → APP 直接走 `cronjob(action=remove, job_id=...)`，**不 ping 你**；卡片切到"✅ 已删除"小条就是反馈。下一轮别再回"已删除"，更别重调 cronjob。
- **Fallback**：用户敲字 "确认" → 你调 `cronjob(action=remove, ...)` → 一句话确认。
- **直接落盘**：用户已明确说"确认删除/删掉吧"，或当前渠道不支持卡片 → 调 `cronjob(action=remove, ...)`，不要再追问。

---

## 5. JSON 围栏字段规范（必读）

每张确认/预览卡都必须附带 `cron-action-preview` JSON 围栏，APP 端用它渲染可交互按钮。规范：

| 字段 | 类型 | create | edit | delete | 说明 |
|---|---|---|---|---|---|
| `mode` | string | ✅ `"create"` | ✅ `"edit"` | ✅ `"delete"` | 卡片模式 |
| `job_id` | string | ❌ | ✅ | ✅ | 目标 job ID（先 `cronjob(action=list)` 拿到）|
| `name` | string | ✅ | ✅ | ✅ | 任务名 |
| `schedule` | string | ✅ | ✅ | ✅ | hermes 原始 schedule（cron expr / ISO / `every Nm`）|
| `cronExpr` | string | optional | optional | optional | 等价 cron 表达式（便于 APP 调试）|
| `schedule_human` | string | ✅ | ✅ | ✅ | 人话描述，与 markdown 卡的"触发"行一致 |
| `prompt` | string | ✅ | ✅ | ✅ | 提炼后的 prompt 全文 |
| `output_language` | string | ✅ | optional | ❌ | LLM 从当前创建对话推断的 BCP 47 tag；APP 只透传，不得改用 App locale |
| `deliver.mode` | string | ✅ | ✅ | ✅ | `"origin"` / `"new_session"` / `"specified"` |
| `deliver.chatName` | string | optional | optional | optional | 对话名（用于 origin/specified 显示）|
| `deliver.sendTo` | object | optional | optional | ❌ | 仅 specified 模式：`{channel, chatName, chatType}` |
| `repeat.times` | number\|null | ✅ | ✅ | ❌ | `null`=永远 / `1`=一次 / `N`=N 次 |
| `repeat.completed` | number | ✅ `0` | ✅ | ❌ | 已完成次数 |
| `changedFields` | string[] | ❌ | ✅ | ❌ | 变更字段名列表 |
| `previousValues` | object | ❌ | ✅ | ❌ | `{字段名: 旧值}`，给 APP 渲染 strikethrough diff |

注意事项：

- 围栏内**必须**是合法 JSON——不要用注释、单引号、尾逗号
- 围栏外的 markdown 卡片是降级渲染兜底，APP 端会用 JSON 重新渲染可交互卡片，不依赖 markdown
- 卡片每出现一次，**必须**带一个对应的围栏——不要只发 markdown 没有 JSON
- 不出卡片、直接落盘时，不要输出 `cron-action-preview` 围栏。
- 落盘后的"一句话确认"用普通文本，不要再发围栏
