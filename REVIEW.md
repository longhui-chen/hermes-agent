# Review instructions (Zettlab)

> 本文件被 Claude Code Review 注入到**每个评审 agent 的最高优先级**，覆盖默认评审口径。
> 配套 `/code-review` 自托管 workflow 也以本文件为准。保持精简——只放「改变评审行为」的规则，
> 通用项目背景留在 `CLAUDE.md` / `AGENTS.md`。

## 「Important（🔴 合并前必须修）」在本仓的定义

只把会**坏行为、泄数据、破坏兼容、或违反工程硬约束**的问题标 Important。风格/命名/重构建议最多 Nit。
以下违反 **Engineering Hard Rules**（全文见 `AGENTS.md`）一律按 **Important**：

1. **内存预算（端侧 2GB 硬上限）**：新增常驻无界 cache、未关闭的 goroutine、重复加载模型、一次性 dump 大文件、
   无 LRU/TTL 的内存索引。
2. **稳定性 & 高可用**：外部依赖缺 timeout / retry / 降级；长驻进程不可被 supervisor 拉起；
   state mutation 不是 atomic-rename + backup（不可回滚）。
3. **Agent 安全**：cloud token 透传给 shell；LLM 输出直接拼接 sql / shell / path；connector / tool / skill 默认拿全量 scope；
   token / shell / sql / fs 操作缺 allowlist。
4. **API 向后兼容**（server / local-server / hermes / WS 控制面）：删字段 / 改类型 / 改 enum / 改 status / 改错误码；
   新增字段非可选；客户端解析不 tolerant；契约改动未同步所有消费方（app + web + 设备）。
5. **冲突取舍**：性能优化偷偷牺牲了稳定 / 安全而未在 PR 显式声明（顺序必须是 HA > 安全 > 性能）。
6. **配置唯一生效位置**：新增 / 依赖 `config.example.yaml` / `config.board.yaml` / 仓库根 `config.yaml` 等不打包配置
   （只有 `{应用仓库}/zpk/config/<repo>.yaml` 生效）。
7. **HR-T1 依赖冻结**：升级 / 降级 / 新增任何 Microsoft / Azure 系依赖
   （`github.com/Azure|AzureAD|Microsoft/*`、`@azure/*`、`azure-*`、`msal*`）。

## 始终检查

- 新增外部调用（HTTP / DB / IoT / OSS / LLM provider）是否带 timeout + retry + 失败降级。
- 把不可信输入（PR 内容、用户输入、LLM 输出、connector 返回）拼进 shell / sql / path 前是否校验/转义。
- 跨端契约改动（proto / JSON / WS 消息 / 错误码）是否保持向后兼容、是否走 feature flag / capability negotiation。
- 新增配置是否落在 `zpk/config/<repo>.yaml`；dev 差异是否走 `ZLS_*` 环境变量而非改打包配置。
- 新增 connector / tool / skill 的 PR 是否回答了「最坏情况能造成什么伤害」。

## 不要报（降噪）

- CI 已覆盖的：lint、格式化、类型错误、`just qa` 已门禁的项。
- 生成物 / 锁文件 / vendored 依赖 / `node_modules` / `*.lock` / 快照测试基线。
- 测试代码里**有意**违反生产规则的部分（mock / fixture）。

## 节流与收敛

- 每轮评审最多贴 **5 条 Nit**；超出在 summary 里写「另有 N 条同类」，不逐条刷屏。
- 复评（同一 PR 已评过）：只报新增的 Important，抑制重复 Nit，避免同一处反复打扰。
- summary 第一行给结论 tally（如 `2 Important · 1 Nit · 0 Pre-existing`）；无 Important 时首句写「无阻断问题」。

## 取证门槛

- 行为类判断要有 `file:line` 证据，不要仅凭命名/猜测下 Important（降低误报，省作者一次往返）。
