# AC-365：按 Agent Connector 开关加载 Skill 的 Hermes 能力调研

> 调研日期：2026-09-04
>
> 范围：`hermes-agent`、`zettlab-presets`，并追踪其在 `zettlab-local-server` 的现有接入点。
>
> 调研基线：Hermes `0fc9551e`、Presets `c42497f`。实现工作在 `codex/ac-365-connector-skill-visibility` 隔离 worktree 中推进；原工作区未跟踪的 `hermes-agent/zpk/lib/` 与 `zettlab-presets/build/` 排除且未改动。

## 结论

AC-365 不需要把 Hermes 改造成“每轮读取所有 Connector Skill”。当前 Hermes 已经是两阶段加载：系统提示词只放 Skill 的名称与描述，完整 `SKILL.md` 仅在 `skill_view`、显式 `/skill`、预加载或 App `metadata.skill_slug` 时读取。真正的问题是 **Connector Skill 的候选索引仍按一次 profile 配置生成，缺少 Agent Connector policy 这一层可见性条件**（`agent/prompt_builder.py:2039-2091,2171-2195,2423-2450`；`tools/skills_tool.py:781-844,957-976`）。

建议采用“控制面编译、Hermes 消费”的 profile 级方案：

1. `zettlab-presets` 在 `manifest.yaml` 增加显式、可校验的 Connector 可见性元数据，把 Skill 映射到稳定的 `target_kind + target_id`；不要从 `required_scopes`、目录名或描述推断。
2. `zettlab-local-server` 只使用**完整且版本可信的 Agent policy snapshot**，计算该 Agent 的 Connector Skill policy-disabled 集合，写入 profile 的机器管理配置层；复用现有 per-Agent visibility gate、持久化 generation marker、失败重试和热重载。
3. Hermes 把机器管理的 `connector_policy_disabled` 与既有用户 `skills.disabled` 合并为统一 disabled set。这样可直接复用当前 prompt index、`skills_list`、slash/quick-pick、`skill_view`、preload 的过滤路径，无需再造一套 Connector 扫描器。
4. policy generation 变化后只调用 profile-scoped `POST /p/{profile}/v1/skills/reload`（单进程模式为 `/v1/skills/reload`）。该入口已经清理 prompt/snapshot、slash 表、持久化 system prompt 和缓存 Agent，使新可见性在下一轮生效（`gateway/platforms/zet_agent.py:8671-8707,8721-8812,9595,9649`）。
5. **不要复活 `/mcp/connectors`，也不要复用 `/v1/connectors/reload`。** 当前 Connector 执行面是 Chat 的 session invoke 与 Application/Core/Cron 的 direct invoke；Hermes 仅保留只读 `list_my_connectors`。后者明确返回无凭据列表，并说明“已连接不等于本会话可执行”（`tools/list_my_connectors_tool.py:1-8,20-31,69-105,108-116`）。`internal/skills/receiver/service.go:140-184` 与 Hermes `connectors/reload` 中仍有旧 MCP 注释/调用，应视作遗留而非 AC-365 设计依据。

交互 Chat 的常驻可见性可概括为：`已安装 ∩ Agent policy enabled ∩ 现有 platform/environment 条件 - 用户 disabled`。账号连接/过期状态不直接改变 profile 索引，而在 readiness 与 session invoke 阶段返回连接/重连引导；`connector-setup` 等恢复入口始终可见。Application / Cron 的任务绑定例外见下一节：它们在实际运行的 Agent 中按本次 workload 显式依赖临时补入索引，不受 Agent Connector 开关限制。

## 必须区分：Agent 级启用与 Chat override

| 状态 | 生命周期 / 作用域 | 是否改变 Skill 索引 | 用途 |
|---|---|---:|---|
| Agent Connector policy | Agent/profile 级，跨 Chat 持久 | 是 | 决定这个 Agent 能看到、列出和显式加载哪些 Connector Skill |
| Chat override | conversation/turn 级 | 否 | 只在本 Chat 内进一步允许或拒绝执行；不得影响同一 Agent 的其他 Chat |
| Application Connector binding | app instance / execution 级 | 仅补入本次运行 | App 显式依赖的 Connector Skill 在所用 Agent 中自动加载，不要求 Agent 开关同时开启 |
| Cron `job.skills` / Connector binding | job / execution 级 | 仅补入本次运行 | 定时任务显式依赖的 Connector Skill 在任务 Agent 中自动加载，不要求 Agent 开关同时开启 |

local-server 的请求结构已经能承载 `PolicyRevision`、`PolicyGeneration`、`PolicySchemaVersion`、`PolicySnapshotComplete`、Agent policy 与 Chat overrides（`zettlab-local-server/internal/chat/handler/chat_ws_v1.go:6719-6769`）。完整快照只有在 schema、complete、owner/account/agent/conversation/source/revision 均可信且每条 policy 的 `disabled_tools` 存在时才可用（同文件 `6874-6890`）。backend 还会把 policy 与 override 一起哈希为 fingerprint（`zettlab-local-server/internal/backend/hermes/chat.go:1026-1100`）。

但 App 与 Web/Desktop 当前 wire type 只发送 `agent_id`、`conversation_id`、schema/complete、policies 和 overrides，并没有发送可信的 source/revision/generation（`zettlab-app/types/chatproto.ts:47-54`；`zettlab-app/stores/chat/turn-dispatch.ts:908-946`；`zettlab-web/src/types/chatproto.ts:55-62`；`zettlab-web/src/utils/connectors/effective-policy.ts:200-220`）。因此 AC-365 **不能直接把客户端 Chat 快照当成 profile 可见性的权威源**。它最多作为“需要刷新”的提示；local-server 应通过现有 Server `GET /connectors/policies/:agentID` 读取权威 Agent policy（`zettlab-server/internal/api/router.go:853-855`），或消费新增的 Server→设备 policy-changed 控制事件。

AC-365 应复用这些**校验、revision/generation 和归一化结构**，但计算 profile 可见性时必须丢弃 `ConversationID` 与 `Overrides`，只取 Agent policies。当前每轮 prompt 注入同时包含两类数据（`zettlab-local-server/internal/backend/hermes/chat.go:882-954`），适合执行路由，不适合直接作为 profile 索引规则。否则一个 Chat 的临时 override 会污染同 Agent 其他会话，而 Hermes 的 system prompt 本身又是 session 持久缓存（`agent/system_prompt.py:196-212,629-665`）。

### Application / Cron：按 workload 自动补入，不写回 Agent 开关

Application / Cron 是与交互 Chat 不同的授权域。任务定义中显式选择 Connector/Skill，本身就是该 workload 的依赖声明；运行时应在它绑定的 Agent 中自动加载对应索引和正文，不要求用户再去 Agent 设置打开同一 Connector，也不能把这次临时补入写回 profile，污染普通 Chat。

Hermes 已有大部分 Cron 接缝：`job.skills` 是持久的显式 Skill 列表，scheduler 在构造任务 prompt 时逐个 `skill_view` 加载正文（`cron/scheduler.py:2552-2740`）；执行期间还会把列表绑定到 `cron_attached_skills` 的 ContextVar（`cron/scheduler.py:3421-3438`；`gateway/session_context.py:121-143`），模型 tool-schema cache 已包含该列表和 manifest fingerprint（`model_tools.py:85-103,135-171`）。AC-365 应把这套机制泛化为 workload-scoped attached Skill overlay：

```text
WorkloadSkillScope {
  source_kind: application | cron
  source_id: app_instance_id | job_id
  execution_id
  agent_id
  skill_ids[]
  connector_targets[]
  grant_revision
}
```

- Cron 直接复用 `job.skills`、现有 ContextVar/cache scope 和 `prepare-connector-execution` 的 agent/job/execution/provider 绑定；只需让“被当前 job 显式 attached 的 Skill”绕过 `connector_policy_disabled`，不绕过 platform/environment、用户 `skills.disabled`、路径/manifest/readiness 校验。
- Application 复用 apphost 已有 `app instance ↔ dedicated_agent_id` 和 task/execution 绑定，新增同构的 attached Skill scope；不要修改 profile 常驻集合，也不要为一次 app 调用触发全 profile `/skills/reload`。
- scoped overlay 只在当前 execution 的 prompt/index/cache key 中存在，`finally` 必须清理 ContextVar；未知 source、过期 revision、Agent/source 不匹配一律 fail closed。

仅让 Hermes 看见 Skill 还不够：当前 Server direct execution 明确仍强制检查 Agent policy（`zettlab-server/internal/service/connector_cron_execution.go:67-100,130-150`）。若 Application / Cron 按产品口径确实“不受 Agent Connector 开关限制”，Server 还需要一个独立的、Server-owned `WorkloadConnectorGrant`，在 App/Cron 配置确认时写入，运行时用 `owner + device + agent + source_kind + source_id + provider/connection + revision` 复核；它替代 direct lane 的 Agent policy 条件，但不替代连接状态、scope/action、风险审批与审计。现有 AC-local Cron `route_capability` 只有 128 条、2 小时、绑定 agent/job/execution/provider，且不出设备（`zettlab-local-server/internal/connectors/bridge/cron_execution.go:22-41,48-94,108-138`），可继续作为设备内路由凭据，但不能冒充 Server 权威 workload grant。

## 用户可见行为与验收口径

截图中的 Chat「添加内容与能力 → Skill」以及 Agent 的 Skill 管理列表都属于交互可见性表面，必须遵守以下规则：

| 表面 | Connector Skill 展示条件 | 开关关闭时 |
|---|---|---|
| Agent Skill 列表 / 搜索 / 计数 | 对应 `AgentConnectorPolicy` 已开启 | 完全不返回、不展示、搜索不到，不用置灰占位 |
| Chat Skill 快捷列表 / 搜索 / 已选项 | Agent policy 已开启，且当前 Chat override 没有关闭 | 完全不返回、不展示、搜索不到；已选项即时失效并在发送前移除 |
| Skill slug / quick-pick / `skill_view` | 与当前 Chat 的有效可见性相同 | fail closed，不能通过直接 slug 绕过，也不返回 Skill 正文 |
| 全局 Store / SkillHub 发现页 | 不属于某个 Agent/Chat 的运行列表 | 可以展示，但应标注需要连接并在 Agent 中开启；不得混入当前 Agent 的“可用 Skill”计数 |
| Application / Cron execution | workload 显式绑定且 grant 有效 | 仅在本 execution 自动补入；不因此出现在普通 Agent/Chat 列表 |

Chat 的有效展示集合为：

```text
interactive_skill_visibility
  = installed
  ∩ agent_connector_policy_enabled
  ∩ chat_override_effective_enabled
  ∩ platform_environment_allowed
  - user_disabled
```

“看不到”必须从数据源开始保证，不能只在 App/Web 组件里 `filter()`：Hermes/local-server 返回的 prompt index、`skills_list`、slash/quick-pick 和 `skill_view` 都要应用相同 predicate，客户端再做同源结果渲染。响应应携带有界的 `visibility_generation`/fingerprint；Agent 或 Chat Connector 开关变化后，使 Hermes cache 与 App/Web 查询 cache 同时失效。面板打开期间发生关闭时，列表立即移除对应项；若请求已在发送途中，session invoke 再次校验并拒绝，不能依赖 UI 时序。

App 与 Web/Desktop 必须消费同一契约并覆盖同样测试。当前 Web Chat picker 只是接收上层传入的 `skills` 数组（`zettlab-web/src/components/workspace/chat/chat-capability-picker.tsx:28-61,156-174`）；App picker 还支持用 `available=false` 渲染灰态（`zettlab-app/components/workspace/chat/input/capability-popover.tsx:389-423`）。对 Connector policy 关闭场景，上层数据源必须直接排除该 Skill，不得复用普通不可用 Skill 的灰态。

## 当前能力盘点

### 1. Hermes 已有渐进加载，而非完整正文一次性加载

- 冷扫描读取 frontmatter/description 并建立 metadata snapshot；snapshot 只保留 name、description、platform、conditions 等字段（`agent/prompt_builder.py:2039-2121`）。
- 系统提示词只渲染 compact Skill index，并指导模型按需调用 `skill_view`（`agent/prompt_builder.py:2423-2450`）。
- 完整正文在 `skill_view` 后才解析链接文件与 readiness（`tools/skills_tool.py:957-976,1078-1258,1322-1350,1467-1721`）。
- 缓存已经有界：prompt snapshot LRU 最大 8；Skill list cache TTL 30 秒（`agent/prompt_builder.py:1911-1916`；`tools/skills_tool.py:91-104`）。

因此应把 AC-365 定义为“Agent-scoped offer/load gate”，而不是再次实现 lazy body loading。

### 2. 过滤基础设施已覆盖大部分表面

Hermes 已有 shared platform/environment/disabled 语义：

- prompt index：`agent/prompt_builder.py:2124-2152,2196-2220,2222-2374`；
- `skills_list` / `_find_all_skills`：`tools/skills_tool.py:665-844`；
- 显式 `skill_view`：`tools/skills_tool.py:1322-1350`；
- slash 与预加载：`agent/skill_commands.py:442-580,612-680,880-940`；
- App quick-pick 的 `metadata.skill_slug`：`gateway/platforms/zet_agent.py:2544-2652`；
- gateway 菜单 / autocomplete：`hermes_cli/commands.py:874-997,1039-1066`。

仅过滤系统提示词会留下可枚举或可显式加载的旁路。最小 Hermes 改动应是让上述路径继续消费同一个合并后的 disabled set，而不是分别增加条件。

### 3. local-server 已有可直接复用的可见性协调器

`zettlab-local-server/internal/agent/registry/connector_visibility_reconcile.go` 已实现：

- Connector 授权完成后的 visibility reconcile 入口（`19-55`）；
- generation marker 的校验、临时文件写入、`fsync + rename`（`58-137`）；
- 先发布 marker、再进入写 gate、reconcile、reload 的顺序（`140-175`）；
- `GetOrSpawn` 返回运行实例前消费失败遗留 marker（`191-267`）；
- process 与 multiplex 两种 reload（`270-294`）。

`connector_visibility_gate.go:12-95` 还提供 64-shard、有引用计数回收的 per-Agent 读写 gate。registry 的 process/multiplex `ReloadSkills` 已分别调用 Hermes `/v1/skills/reload` 或 profile 路由（`registry.go:1249-1280`；`multiplex.go:767-790`）。这些正是 AC-365 的并发、崩溃恢复和 next-turn 生效机制，应直接扩展。

现有 Feishu/WeCom reconcile 只是 direct-run 与 cloud Skill 冲突的 stopgap：它用静态 provider 表并往 `skills.disabled` 单向追加（`connector_preset_skills.go:317-357,380-410,413-499`），不能直接推广为通用 Agent policy，因为：

- 它只覆盖两个 provider；
- 旧逻辑故意 sticky，不做 policy 开关的双向恢复；
- `skills.disabled` 同时容纳用户与系统写入，启用时无法证明某条禁用由谁创建。

local-server 已有通用 Skill 开关写法：`applyVisibility` 通过 profileconfig 锁和 YAML-preserving serializer 调用 `EnableSkills` / `DisableSkills`（`internal/skills/service/service.go:2284-2312`）；Hermes profileconfig 也能保留、去重 disabled 列表（`internal/agent/profileconfig/profileconfig.go:1610-1736`）。可复用其写入机制，但通用 Connector policy 必须有独立机器管理层，不能删除用户自己的禁用项。

### 4. Presets 缺少稳定的 Connector 映射契约

当前 manifest schema 要求 id/version/name/description、资源预算、scope、fs/shell/worst-case；支持 `runtime_capabilities` 与 `connector_action_manifest`，但没有“哪个 Agent Connector 开关控制此 Skill”的字段，且 schema `additionalProperties: false`（`zettlab-presets/schema/skill.schema.json:6-17,33-150`）。例如 Gmail 只声明 scopes 与固定 runner（`zettlab-presets/skills/intl/gmail/manifest.yaml:1-38`）；`connector-setup` 本身不要求 Connector scope，应该始终可见（`skills/common/connector-setup/manifest.yaml:1-24`）。

建议新增可选字段（名称可在 spec 阶段定稿）：

```yaml
connector_visibility:
  mode: any_enabled          # any_enabled | all_enabled
  targets:
    - target_kind: app
      target_id: gmail
```

语义：缺字段即非 Connector gate、始终按既有规则可见；provider-specific Skill 用 `any_enabled` + 单 target；`authorized-connectors` 等 umbrella Skill 可列多个 target 并用 `any_enabled`；`connector-setup` 不声明此字段。custom connector 必须使用稳定 `custom_mcp/custom_api + target_id`，不靠显示名匹配。构建脚本已经把 manifest 放进区域 catalog 与 all-catalog（`zettlab-presets/scripts/build-bundle.py:9-40,250-260`），local-server 可复用这份注册表，不在 Hermes 硬编码 provider 列表。

## 推荐状态模型与时序

建议在 profile config 中新增机器拥有的独立集合，例如：

```yaml
skills:
  disabled: [user-disabled-skill]       # 现有用户/产品显式开关
  connector_policy_disabled: [gmail]   # AC-365 机器管理，禁止手改
  connector_policy_generation: 42
```

Hermes 的 effective disabled set 为两者并集。配置只保存 Skill ID 与 generation，不保存 token、connection ID、URL、账号别名或 Chat override。集合须限量、ID 须校验；写入继续走 profileconfig 的按路径锁与原子写。

```text
App / Web 修改 Agent Connector policy
              │
              ▼
完整 Agent policy snapshot（revision/generation）
              │  忽略 Chat override
              ▼
local-server：manifest 映射 → connector_policy_disabled
              │  per-Agent write gate + durable marker + atomic write
              ▼
POST /p/{profile}/v1/skills/reload
              │
              ▼
Hermes 清 prompt/snapshot/session/agent cache
              │
              ▼
下一轮所有 Skill 表面消费 effective disabled set
```

Application / Cron 不走上面的持久投影，而走短生命周期 overlay：

```text
App/Cron 配置时显式选择 Connector Skill
              │
              ├─ Server-owned WorkloadConnectorGrant
              └─ app binding / cron job.skills
                         │
                         ▼
运行时校验 source + agent + execution + revision
                         │
                         ▼
attached Skill overlay → scoped index/cache → skill_view 预加载正文
                         │
                         ▼
direct invoke 再校验 connection + scope/action + risk + audit
```

触发优先级建议如下：

1. 目标态由 Server 在 Agent policy CAS 写成功后发送带 revision 的 device control event，local-server 收到后读取权威 policy 并立即 reconcile；当前代码尚未发现这一事件，需要作为控制面小切片补齐。
2. AC-365 Demo 可先在 profile 首次 spawn / 首轮 Chat 前调用现有 `GET /connectors/policies/:agentID`；App/Web 发来的 policy fingerprint 只作为刷新提示，不能直接落盘。读取要有短 timeout、有限重试和 last-known-good 降级，避免云端瞬时不可用阻断普通 Chat。
3. 若暂时只能消费请求内快照，也必须等 local-server 补齐并验证 owner/source/revision/generation 后才允许 reconcile；**不得**用不完整或仅由客户端声明 complete 的快照把所有 Connector Skill 误开启。

## 兼容、失败与安全策略

- feature flag 默认关闭时维持现状，支持灰度；开启后，完整快照缺失/版本不支持时对 connector-specific Skill fail closed，但 `connector-setup` 等恢复入口保持可见。
- reload 失败保留 generation marker，由下次 `GetOrSpawn` 重试；沿用现有 `ForwardOrStop` 的 critical 5xx stop fallback，而不是让陈旧索引永久驻留（`zettlab-local-server/internal/skills/receiver/service.go:29-46,140-160`）。
- Skill 可见只影响模型候选面，**绝不是执行授权**。Chat 由 session invoke 校验 owner、Agent、Chat、policy/scope，Application/Cron 由 direct invoke 校验 workload grant 与 Connector 权限；禁止 provider REST、shell/curl、token fallback。
- Application / Cron 的“不受 Agent 开关限制”只表示由 `WorkloadConnectorGrant` 替代 Agent policy；绝不表示无授权调用。grant 必须绑定 owner/device/agent/source/execution/target/revision，并在 direct invoke 时重新校验连接、scope/action、风险审批与审计。
- `list_my_connectors` 可保留作只读解释/恢复工具，但不能作为 prompt 构建前的动态授权来源：它在工具执行阶段才查询，而且文档明确连接状态不代表本轮可调用。
- Agent 级索引保持 App 与 Web/Desktop 一致；Chat override 只影响该 conversation 的执行路由，两端发送同一契约。

## 分阶段实施建议

1. **契约层**：Presets schema + manifest 补 `connector_visibility`，catalog 校验 target 唯一性/上限；定义 meta Skill 例外；App/Cron 定义显式 `skill_ids + connector_targets` 绑定。
2. **控制面**：local-server 复用 policy snapshot 校验、manifest catalog、visibility gate、marker 与 profileconfig 原子写，生成交互 Chat 的 `connector_policy_disabled`；去掉 AC-365 路径对 `/v1/connectors/reload` 的依赖。
3. **Workload 授权**：Server 增加有 revision 的 `WorkloadConnectorGrant`；Cron 复用现有 agent/job/execution/provider route，Application 复用 app instance/dedicated Agent binding；direct lane 不再要求同一 Agent 开关，但仍逐次校验 workload grant 与 Connector 权限。
4. **Hermes 小改**：配置解析时合并两类 disabled；泛化现有 Cron attached Skill ContextVar/cache scope，让 Application/Cron 的显式 Skill 仅在本 execution 绕过 `connector_policy_disabled`；`/v1/skills/reload` 无需新增 endpoint。
5. **验证**：覆盖 off→on、on→off、用户 disabled 不被系统 enable 覆盖、Chat override 不改 profile、并发 Chat + policy update、reload 5xx/重启补偿、multiplex profile 隔离、App quick-pick/slash/`skill_view` 无旁路，以及 Agent 开关关闭时 Application/Cron 仍只能加载并执行自己显式绑定的 Skill。

## Engineering Hard Rules 影响

- **HR1 内存**：只增加有界 ID 集与 generation；复用 LRU/TTL/64-shard gate，不引入按 Chat 无界缓存或重复读取正文。
- **HR2 HA**：复用 timeout、marker、原子写、重试与 stop fallback；失败时不破坏 chat，下一次 spawn/turn 自愈。
- **HR3 安全**：状态文件无凭据；visibility 与 authorization 分层；Chat 仍经 session invoke，Application/Cron 经 workload grant + direct invoke；禁止恢复 MCP bridge 或 provider 直连。
- **HR4 API Compat**：manifest/config 均为可选新增字段，feature flag 默认 legacy；不删旧字段、错误码或路由。App 与 Web/Desktop 同步消费同一 Agent policy 契约。
- **HR5 优先级**：优先保证 profile 状态一致和失败可恢复，其次 fail-closed，再优化 prompt token/扫描次数。
- **HR6 配置唯一源**：若引入设备 feature flag，只能进入对应 `zpk/config/<repo>.yaml`；不能依赖 example/root config。
- **HR-T1**：不新增、升级或降级 Microsoft/Azure 依赖。

## 定向验证建议

本次为只读调研，只新增此文档，未运行任何本地全量 QA。实施时只运行涉及 package/file 的定向测试，例如：

```bash
# Hermes：统一 disabled 过滤与 reload（按新增测试文件收窄）
cd /Users/zettlab/project/zettleb-memo-workspace/hermes-agent
pytest -q tests/test_skills_tool.py -k 'disabled or reload'
pytest -q tests/cron/test_skill_operation_scope_flow.py tests/cron/test_connector_execution_lease_flow.py

# local-server：visibility reconcile、profile config 与 chat policy 归一化
cd /Users/zettlab/project/zettleb-memo-workspace/zettlab-local-server
go test ./internal/agent/registry -run 'ConnectorVisibility|ReloadSkills'
go test ./internal/agent/profileconfig -run 'DisableSkills|EnableSkills'
go test ./internal/backend/hermes -run 'Connector.*(Context|Fingerprint|Override)'
go test ./internal/connectors/bridge -run 'CronExecution'

# server：只跑 workload direct authorization 的目标 package/case
cd /Users/zettlab/project/zettleb-memo-workspace/zettlab-server
go test ./internal/service -run 'Connector.*Direct|Connector.*Cron'

# Presets：只跑 manifest/catalog 的定向 schema 校验（以仓库现有脚本参数为准）
cd /Users/zettlab/project/zettleb-memo-workspace/zettlab-presets
python3 scripts/validate.py --skills-only
```

上述只是实施阶段的候选定向命令；全量 QA 留给 CI 或明确隔离环境。
