# Connector 受保护输入与普通澄清边界

普通 clarify 的回答会进入模型和历史；在 question 中称其为“安全卡片”不会改变传输语义。只有通过校验的 connector_setup metadata 才能进入受保护输入，凭据直接提交既有 Connector 接口，模型仅接收有界状态回执。

本次修复：

- Hermes 在普通澄清发布前拒绝明确要求输入 PAT、密码、密钥等的中英文问题/选项，返回 connector_setup_required，引导重试结构化输入。该规则是拒绝守卫，不从文案猜测模板或转换请求；它不声称能识别所有语言和改写方式，工具 schema 同时明确禁止普通澄清索取凭据。
- submitted 的 next_step 明确回执不包含凭据，不能仅因回执到达就判断泄密或要求撤销。仍需验证可用性与会话授权，回执不是 grant。
- App/Web 在修改 pending/history 或发送 WS/HTTP 前校验受保护回复，只允许 status 和合法 target_id，拒绝原始凭据、额外字段和错误类型。正常澄清保留原行为。
- 不修改既有历史记录，不将已进入普通澄清历史的原始凭据假称为安全提交；历史清理和真实设备部署不在此次执行范围。

验证：Hermes refusal 单测与实际 callback/pending/HTTP response 流程；双端 receipt 单测、store 拒绝流程、定向 ESLint/TypeScript。没有运行本地全量 QA/build，也未进行真实设备、模型、OAuth 或 GUI E2E。GitNexus impact/detect-changes 因 libssl.3.dylib 缺失不可用。

| Hard Rule | 影响 |
|---|---|
| HR1 | 无新增持久状态，校验输入有界；无板端常驻 RSS 增量机制 |
| HR2 | 误用在发布前可恢复拒绝，沿用现有交互重试与取消 |
| HR3 | 收紧凭据进入模型/历史的路径；最坏风险仍是模型绕过规则索取原始密钥，不将文案识别当作安全输入授权 |
| HR4 | 不改变 wire 字段、枚举或端点；双端遵循现有 metadata 与回执契约 |
| HR5 | 保留可恢复失败，不以关闭安全检查换取连接成功 |
| HR6 | 无设备配置、部署或 OTA 修改 |
| HR7 | 无新增共享状态、预算、后台对账或第二套授权机制 |
| HR-T1 | 无依赖变更 |

Hermes upstream-pr: none；工具 schema/回执说明为最小 overlay，业务拒绝逻辑留在 zet_agent 平台目录。
