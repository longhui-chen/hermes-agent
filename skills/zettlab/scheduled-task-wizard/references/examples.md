# 对话例子

主入口：`../SKILL.md`

7 个 end-to-end 对话场景，覆盖典型表达形态。创建/修改落盘后用普通文本回一句确认，不出围栏、不出卡片；只有删除出确认卡。

---

## 例 1：标准早报（默认形态，无 clarify）

> 用户："每天早上给我发 AI 新闻摘要"
>
> Agent：（提炼 + 推断"每天"=`0 8 * * *` + 默认 `origin` + 默认 forever，巡检类任务自动加 `[SILENT]` 提示并创建）
>
> ```
> cronjob(action=create,
>         name="AI 新闻早报",
>         schedule="0 8 * * *",
>         prompt="用 web_search 搜索最近 24 小时内的全球 AI 重要新闻，按重要性排前 5 条，每条一句话总结+链接。用中文输出。如本周期没有值得汇报的新内容，请只回复 [SILENT] 单独一行。",
>         output_language="zh-CN")
> ```
>
> 工具返回：`{"success": true, "job_id": "…", "next_run_at": "2026-04-16T08:00:00+08:00"}`
>
> Agent："建好了：每天早上 08:00 给你发 AI 新闻早报，发在当前对话，第一次是明天 08:00。"
>
> ——"明天 08:00"来自返回的 `next_run_at`，不是自己推的。其余例子同理，不再重复标注。

---

## 例 2：需要 clarify 的有限期

信息不齐时先问再建——clarify 是唯一会挡在落盘前面的东西。

> 用户："今天每隔 2 小时提醒我起来活动"
>
> Agent："好——今天一天大约 5 次左右就停？还是你想一直循环到手动停？"
>
> 用户："今天一天就行"
>
> Agent：（推断今天剩余约 5 个 2 小时间隔 → `cronjob(action=create, schedule="every 2h", repeat=5, ...)`）
>
> Agent："好，每 2 小时提醒你起来活动一次，今天共 5 次，下一次 15:00。"

---

## 例 3：需要 clarify 模糊时间

> 用户："下周提醒我去家长会"
>
> Agent："好——下周哪一天？几点？"
>
> 用户："周三下午 3 点"
>
> Agent：（`cronjob(action=create, schedule="2026-04-22T15:00", repeat=1, ...)`）
>
> Agent："记下了：下周三（4 月 22 日）15:00 提醒你去家长会，一次性。"

---

## 例 4：用户主动选其它会话

> 用户："每周一早上 9 点生成上周项目周报，发到飞书产品群"
>
> Agent：（调 `channel_directory_lookup` → 命中"飞书 - 产品群" → `cronjob(action=create, deliver="feishu:产品群", ...)`）
>
> Agent："建好了：每周一 09:00 生成上周项目周报，发到飞书的产品群，下次周一 09:00。"

---

## 例 5：每次新对话承接

> 用户："每天早上发个 AI 新闻早报，每天给我一个独立的新对话方便我归档"
>
> Agent：（`cronjob(action=create, deliver="new_session", ...)`）
>
> Agent："建好了：每天 08:00 发 AI 新闻早报，每次执行都新开一个对话承接，方便你归档。第一次明天 08:00。"

---

## 例 6：建完立刻改

> 用户："每天早上给我发 AI 新闻摘要"
>
> Agent："建好了：每天早上 08:00 …"（例 1）
>
> 用户："8 点太早了，改 9 点"
>
> Agent：（`cronjob(action=list)` 定位 → 唯一候选 → `cronjob(action=update, job_id=..., schedule="0 9 * * *")`）
>
> Agent："好，「AI 新闻早报」从 08:00 改到 09:00 了，下次明天 09:00 执行。"

---

## 例 7：删除——唯一出确认卡的动作

> 用户："把那个新闻早报删了"
>
> Agent：（`cronjob(action=list)` 定位 → 唯一候选，**先不调 remove**，出确认卡）
>
> ```
> 确定删除「AI 新闻早报」吗？
> 触发：每天 09:00
> 运行历史会一起清除。
> [确认删除] [取消]
> ```
>
> 紧跟其后的 JSON 围栏（APP 端解析渲染可交互按钮）：
>
> ````
> ```cron-action-preview
> {
>   "mode": "delete",
>   "job_id": "job-7f3a",
>   "name": "AI 新闻早报",
>   "schedule": "0 9 * * *",
>   "cronExpr": "0 9 * * *",
>   "schedule_human": "每天 09:00",
>   "prompt": "用 web_search 搜索最近 24 小时内的全球 AI 重要新闻，按重要性排前 5 条，每条一句话总结+链接。用中文输出。如本周期没有值得汇报的新内容，请只回复 [SILENT] 单独一行。",
>   "deliver": { "mode": "origin", "chatName": "与 ZettClaw 助手的对话" }
> }
> ```
> ````
>
> 但如果用户一开始就说"**确认**把新闻早报删了"——文字里已经明确确认，跳过卡片直接 `cronjob(action=remove, job_id=...)` → "删了。"
