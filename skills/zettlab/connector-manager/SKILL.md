---
name: connector-manager
description: "管理 Zettlab Connector 授权：查看连接状态、发起 OAuth 授权、为当前 Agent 开关服务访问权限。不处理第三方 API 调用本身，只管连接与权限层。"
version: 1.0.0
author: zettlab
license: proprietary
metadata:
  hermes:
    tags: [connectors, oauth, zettlab, builtin, authorization]
  zettlab:
    connector_skill: true
    temporary_location: hermes-agent
    migration_target: dedicated-connector-skills-repo
    migration_task: connector-v1-t13
---

# Connector 管理

帮用户查看和管理 Zettlab 外部服务授权（GitHub · Notion · Linear）。

此 skill 负责**连接层**——授权是否建立、状态是否有效、当前 Agent 是否有权限使用。  
实际调用第三方 API（list issues、create page 等）不在此 skill 范围内。

## 环境要求

运行前确认以下环境变量已配置（在 `~/.hermes/.env` 或 profile 级 env 里设置）：

| 变量 | 说明 |
|---|---|
| `ZETTLAB_SERVER_URL` | Zettlab 云服务地址，如 `https://api.zettlab.com` |
| `ZETTLAB_USER_TOKEN` | 当前用户的 Bearer token（登录后从 App/Web 获取） |
| `ZETTLAB_AGENT_ID` | 当前 Agent 的 ID，用于查询和更新 Agent 级权限策略 |

缺少上述变量时，直接告诉用户需要在 `.env` 里补充哪个，不要继续尝试调用。

## 使用场景

- 用户问"我连接了哪些服务"→ 查询连接列表
- 用户问"帮我连接 Linear"→ 生成授权 URL，引导去浏览器完成 OAuth
- 用户问"这个 Agent 能用 GitHub 吗"→ 查询当前 Agent 的 connector policy
- 用户说"给这个 Agent 开启 Notion 权限"→ 更新 Agent connector policy
- 用户说"断开 GitHub 连接"→ 撤销连接

## 工作流

### 1. 查询连接列表

```bash
python3 ~/.hermes/skills/zettlab/connector-manager/scripts/connector_api.py list
```

输出示例：
```
GitHub     ✓ connected (wolfhunter)    active
Notion     ✗ not connected
Linear     ✓ connected (Wolf Hunter)   active
```

以用户语言展示结果：已连接的说"已连接，账号：xxx"，未连接的说"未连接，可以帮你发起授权"。

### 2. 发起 OAuth 授权

```bash
python3 ~/.hermes/skills/zettlab/connector-manager/scripts/connector_api.py authorize <provider>
# provider: github | notion | linear
```

输出一个授权 URL。把 URL 展示给用户，说"请在浏览器中打开这个链接完成授权"。  
用户完成后，告知他们可以再问"我连接了哪些服务"来确认连接已生效。

### 3. 查询 Agent connector 权限

```bash
python3 ~/.hermes/skills/zettlab/connector-manager/scripts/connector_api.py agent-policies
```

展示当前 Agent 对每个已连接服务的开关状态。

### 4. 更新 Agent connector 权限

```bash
python3 ~/.hermes/skills/zettlab/connector-manager/scripts/connector_api.py set-policy <provider> <enabled>
# enabled: true | false
```

操作后告诉用户"已为该 Agent 开启/关闭 xxx 访问权限"，并说明生效时机（下一次对话即生效）。

### 5. 撤销连接

```bash
python3 ~/.hermes/skills/zettlab/connector-manager/scripts/connector_api.py revoke <connection_id>
```

先用 `list` 拿到要撤销的 connection_id，再执行 revoke。  
撤销前跟用户确认，撤销后告知结果。

## 错误处理

| 错误 | 处理 |
|---|---|
| 环境变量缺失 | 告诉用户缺哪个变量，如何设置 |
| HTTP 401 | 告诉用户 `ZETTLAB_USER_TOKEN` 已过期，需要重新登录获取新 token |
| HTTP 400 `unsupported connector provider` | 告知该 provider 暂不支持 |
| 网络不通 | 检查 `ZETTLAB_SERVER_URL` 是否正确，服务是否可达 |

## 红线

- 不要把 token 打印出来展示给用户
- 不要替用户自动决定是否撤销连接，必须先确认
- 不要修改其他 Agent 的 connector policy，只操作 `ZETTLAB_AGENT_ID` 指向的当前 Agent
