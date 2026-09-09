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
