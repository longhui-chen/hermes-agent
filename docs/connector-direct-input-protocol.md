# Connector Chat direct input 协议切片

状态：结构化请求、解析和恢复基础已实现；实际连接控制器与 skill 接线尚未完成，客户端不发送能力开关，因此默认关闭。

复用 clarify 的 pending / response / timeout 生命周期。新增可选 connector_setup，只携带受校验的资源类型、模板或提供方 ID，以及不含凭据的 HTTPS MCP 地址。客户端完成时仅返回状态和目标 ID；Hermes 必须重新确认连接和当前会话授权，客户端回执不构成授权。旧客户端请求被明确拒绝，不回退到普通聊天收集密钥。

新 UI 必须在普通 composer 卸载后接管输入位置，不渲染 clarify 卡片，不把输入写入草稿或历史。尚未接线，不能据此宣称真实连接已完成。

验证：新增意图单元测试、现有 clarify 请求与 HTTP 回应流程、App/Web 实时事件及历史恢复测试、设备翻译和能力默认关闭测试。Hermes 4 个定向文件 105 项通过；Web 3 个文件 20 项通过；App 4 个文件 33 项通过。设备对应 package 仅运行 ConnectorSetup / TranslateClarify 匹配用例。无真实设备或 Electron 完整连接 E2E；本地全量 QA/build 未运行。

跨仓库契约当前剩余 Web 基线差异：message.appended.reason / truncated 已存在于 e06750906 的类型中，但设备 golden 未声明。vendored/live 各报 2 项，不属于本次新增字段；不得当作全量契约通过。

后续 E2E 清单：同一会话触发 → 只显示缺失输入 → 凭据直达可信接口 → 验证/创建/授权读回 → 原任务继续；断网重连恢复同一请求；取消/超时/切账号设备与 Agent 清空输入；旧客户端不收到密钥提示。

| 硬规则 | 本切片影响 |
|---|---|
| HR1 | 无新增常驻缓存或模型，单请求 metadata 有界，未实测 RSS |
| HR2 | 复用既有 pending 到期、响应及重放机制，不建第二套恢复状态 |
| HR3 | 不接收 secret 字段，不新增执行或授权权限；最坏影响为请求被拒绝或延迟连接，回执不放行工具 |
| HR4 | 可选字段和每请求显式能力；客户端尚不声明能力，默认关闭 |
| HR5 | 可恢复拒绝优先于绕过验证 |
| HR6 | 不改设备配置 |
| HR7 | 不新增共享预算、CAS、队列或数据库；复用既有生命周期 |
| HR-T1 | 无依赖变更 |

## Reuse known public API configuration

Optional `custom_api.variables` carries only base_url, endpoint_path, method, tool_name and header_name (at most five strings of 2048 characters). URLs reject userinfo/query/fragment; paths reject query/fragment; control characters and backslashes are rejected. Clients additionally require each key in the selected trusted template and prefer an explicit correction over the intent. Credentials and hardware details remain excluded. Existing clients can ignore this optional object.

Validation: App/Web parser and preparation suites each 23 tests passed; Hermes trusted clarify suite 24 passed; Go TestConnectorSetup in chatproto/backend translation passed. Manual acceptance: known URL/path → only credential requested; rejected token variable → no input dispatch or create. GUI/live service E2E and local full QA/build unrun. HR1 bounded payload/no background memory; HR2 existing lifecycle/retry; HR3 strict public-key validation; HR4 additive nested object; HR5 fail closed; HR6 no config; HR7 no additional state mechanism; HR-T1 no dependencies.

PR #1052 quality-tests (4) repair: synchronized snapshot 34669c8bd776. The root contract explicitly registers only existing optional Web message.appended reason/truncated compatibility reads as warnings, with closure in chat-ui-b1-lifecycle-error. No producer fields are invented; unknown extras still fail. Web plan-ack capability test now includes connector_direct_input. Web three affected files: 9 tests passed; root scoped exception test and cross-repo verify passed (0 errors, 90 explicit warnings); LS manifest and Hermes pinned manifest passed. App contract suite passed; combined App contract/flow run had 9 pass / 1 pre-existing failure in reverse context.compaction producer coverage. Its test and compare.mjs are byte-identical to HEAD and unchanged by this repair. No claim of App full flow success. HR1–7/T1 unchanged: metadata/pins only, no runtime/API/permission/config/dependency changes. Local full QA/build unrun; CI shard rerun pending.


## Jira 引导修复（2026-09-09）

已知 base_url 复用 HTTPS 或 RFC1918 私有 IPv4 HTTP（无 userinfo/query/fragment）；只在 catalog 的 connector_url 类型允许私网 HTTP，https_url 保持 HTTPS。远程 MCP 仍只允许 HTTPS。元数据不授予目标访问能力，运行时仍走既有模板校验、私网设备执行和明文传输确认。

普通 clarify 不得以“在安全连接卡中完成配置”代替结构化 connector_setup；运行时会在发布空泛澄清前拒绝。双端凭据提示包含可信模板名称，Jira 示例显示 PAT；具体缺失字段由现有输入模型决定。Skill 0.6.1 提供复用已知地址的真实调用例子。

验证：两端类型/准备单测和 Jira 地址→凭据→创建流程、Hermes callback 与 metadata 单测、LS clarify JSON roundtrip、presets 引导与 catalog check。未运行本地全量 QA/build；本轮未部署，不能将上一轮 AC 部署结果当成本轮真实 GUI/模型验证。

HR1 无常驻内存变化；HR2 无新增请求/重试机制；HR3 无凭据字段/新增 scope，不允许公网 HTTP、loopback 或 link-local HTTP；HR4 wire 字段保持不变，四端同步校验，旧端拒绝未支持地址；HR5 保留传输确认；HR6 无配置修改；HR7 无新增共享状态；HR-T1 无依赖变更。GitNexus 因缺少 libssl.3.dylib 未完成。Hermes upstream-pr: none。
