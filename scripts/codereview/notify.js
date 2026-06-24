// Claude /code-review → 飞书：编排层（取 token → 定位 PR → 起话题/回复话题 → 回写锚点）。
//
// 两个入口，共用同一套话题/卡片/发送逻辑：
//   notifyFromCheckRun({github,context,core})    —— 托管 Code Review 的 check_run 路径
//   notifyFromActionResult({github,context,core,result,pr}) —— 自托管 Action 的 JSON 路径
//
// 由各 workflow 的 github-script 步骤 require 后调用。判定 + 卡片真源 =
// scripts/codereview/report.js（本文件只做 IO 编排）。发送机制与 linearb-feishu-report 一致：
// 同一只应用机器人 (cli_a97acaec84389cc0)，「每 PR 一话题」——首次不通过发根卡片并 @ 作者一次，
// 把飞书 message_id 回写成 PR 隐藏标记评论 <!-- claude-review-feishu-thread:<mid> -->；后续复评
// 不通过用 reply_in_thread 收进同话题，不重复 @。只处理 base=main、只在不通过时通知。
// 用与 linearb 不同的话题锚点前缀，两套评审各自独立成话题，互不污染。

const crypto = require('crypto');
const FEISHU = 'https://open.feishu.cn/open-apis';
const MARK = '<!-- claude-review-feishu-thread:';

function validFeishuMessageId(value) {
  return typeof value === 'string' && /^om_[A-Za-z0-9_-]{16,80}$/.test(value);
}

function feishuFailureSummary(result) {
  const status = result && typeof result.status !== 'undefined' ? result.status : 'UNKNOWN';
  const json = (result && result.json) || {};
  const code = typeof json.code !== 'undefined' ? json.code : 'UNKNOWN';
  const requestId = json.request_id || (json.data && json.data.request_id);
  return `HTTP ${status}, code=${code}${requestId ? `, request_id=${requestId}` : ''}`;
}

function shouldRecreateRootOnReplyFailure(result) {
  const status = result && result.status;
  const code = result && result.json ? String(result.json.code || '') : '';
  const msg = result && result.json ? String(result.json.msg || '') : '';
  return status === 404 ||
    /message.*(not found|deleted|recalled)|not found|deleted|recalled|不存在|找不到|已删除|已撤回/i.test(`${code} ${msg}`);
}

async function feishu(path, method, body, token) {
  const headers = { 'Content-Type': 'application/json' };
  if (token) headers.Authorization = `Bearer ${token}`;
  // HR-2：外部依赖必配 timeout —— 否则飞书 API 挂起会让 github-script 步骤永不 resolve、整个 job 卡到超时。
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 15000);
  try {
    const resp = await fetch(`${FEISHU}${path}`, { method, headers, body: body ? JSON.stringify(body) : undefined, signal: ctrl.signal });
    const text = await resp.text();
    let json; try { json = JSON.parse(text); } catch (_e) { json = {}; }
    return { ok: resp.ok, status: resp.status, json, text };
  } catch (e) {
    const msg = e && e.name === 'AbortError' ? 'feishu API timeout (15s)' : (e && e.message) || String(e);
    return { ok: false, status: 0, json: { code: 'NETWORK_ERROR', msg }, text: msg };
  } finally {
    clearTimeout(timer);
  }
}

function readFeishuEnv(core) {
  const appId = process.env.CODEREVIEW_FEISHU_APP_ID;
  const appSecret = process.env.CODEREVIEW_FEISHU_APP_SECRET;
  const chatId = process.env.CODEREVIEW_FEISHU_CHAT_ID;
  for (const s of [appId, appSecret, chatId]) if (s) core.setSecret(s);
  if (!appId || !appSecret || !chatId) {
    // 未配置（如刚铺到新仓、FEISHU_* 尚未接好）→ 安静跳过，不把 PR check 标红（fail）。
    // 托管路径的 reusable workflow 已把这三个 secret 声明为 required:true，由 GitHub 在 caller 侧强校验。
    core.warning('未配置 Feishu secret（CODEREVIEW_FEISHU_APP_ID / _APP_SECRET / _CHAT_ID），跳过通知');
    return null;
  }
  return { appId, appSecret, chatId };
}

// 共用核心：给定已判定为 fail 的 cls + PR 元数据，按「每 PR 一话题」发/回复飞书。
// dedupeKey 用于让同一次评审事件的重试幂等（飞书 uuid）。
async function postToThread({ github, context, core }, { cls, prData, env, dedupeKey }) {
  const { classifyCheckRun } = require('./report'); // 仅为触发 require 缓存；卡片函数下面单独取
  const { buildCard, interactiveCardContent } = require('./report');
  void classifyCheckRun;

  const tokResp = await feishu('/auth/v3/tenant_access_token/internal', 'POST', { app_id: env.appId, app_secret: env.appSecret });
  const token = tokResp.json.tenant_access_token;
  if (!token) { core.setFailed(`取 tenant_access_token 失败: HTTP ${tokResp.status}, code=${tokResp.json.code || 'UNKNOWN'}`); return; }
  core.setSecret(token);

  const prNum = prData.number;
  const reviewEventKey = `${context.repo.owner}/${context.repo.repo}#${prNum}:${dedupeKey}`;
  const feishuUuid = (kind) => crypto.createHash('sha256').update(`${reviewEventKey}:${kind}`).digest('hex').slice(0, 32);

  // 查该 PR 的话题根 message_id（隐藏标记评论）
  let rootMid = null, markerCommentId = null;
  try {
    const comments = await github.paginate(github.rest.issues.listComments,
      { owner: context.repo.owner, repo: context.repo.repo, issue_number: prNum, per_page: 100 });
    for (const c of comments) {
      const i = (c.body || '').indexOf(MARK);
      if (i >= 0) {
        const candidate = c.body.slice(i + MARK.length).split('-->')[0].trim();
        if (!validFeishuMessageId(candidate)) { core.warning(`忽略格式非法的话题锚点 comment_id=${c.id}`); continue; }
        rootMid = candidate; markerCommentId = c.id; break;
      }
    }
  } catch (e) { core.warning('读取 PR 评论失败（按首次发送处理）: ' + e.message); }

  const cardContentOrFail = (card, label) => {
    try { return interactiveCardContent(card); }
    catch (e) { core.setFailed(`生成飞书卡片失败(${label}): ${(e && e.message) || e}`); return null; }
  };

  let rootSendUuidKind = 'root';
  if (rootMid) {
    const content = cardContentOrFail(buildCard(context.repo.repo, prData, cls, { atAuthor: false, isReply: true }), 'reply');
    if (!content) return;
    const result = await feishu(`/im/v1/messages/${encodeURIComponent(rootMid)}/reply`, 'POST',
      { msg_type: 'interactive', content, reply_in_thread: true, uuid: feishuUuid(`reply:${rootMid}`) }, token);
    if (result.ok && result.json.code === 0) { core.info('话题回复成功'); return; }
    if (!shouldRecreateRootOnReplyFailure(result)) { core.setFailed(`话题回复失败，未重发根消息: ${feishuFailureSummary(result)}`); return; }
    core.warning(`话题根消息不可用(${feishuFailureSummary(result)})，回退重发根消息`);
    rootSendUuidKind = `root-recreate:${rootMid}`;
    rootMid = null;
  }

  const content = cardContentOrFail(buildCard(context.repo.repo, prData, cls, { atAuthor: true, isReply: false }), 'root');
  if (!content) return;
  const result = await feishu('/im/v1/messages?receive_id_type=chat_id', 'POST',
    { receive_id: env.chatId, msg_type: 'interactive', content, uuid: feishuUuid(rootSendUuidKind) }, token);
  if (!(result.ok && result.json.code === 0)) { core.setFailed(`Feishu 根消息发送失败: ${feishuFailureSummary(result)}`); return; }
  const newMid = result.json.data && result.json.data.message_id;
  if (!validFeishuMessageId(newMid)) { core.setFailed(`Feishu 根消息缺少有效 message_id: ${JSON.stringify(result.json.data || {})}`); return; }
  core.info(`根消息发送成功 message_id=${newMid}`);
  const markBody = `${MARK}${newMid} -->\n<sub>Claude /code-review 评审话题锚点（自动维护，请勿删除）</sub>`;
  try {
    if (markerCommentId) await github.rest.issues.updateComment({ owner: context.repo.owner, repo: context.repo.repo, comment_id: markerCommentId, body: markBody });
    else await github.rest.issues.createComment({ owner: context.repo.owner, repo: context.repo.repo, issue_number: prNum, body: markBody });
  } catch (e) { core.warning('回写话题标记评论失败（下次会重发根消息）: ' + e.message); }
}

async function resolvePr(github, context, hint) {
  let pr = hint || null;
  if (!pr) {
    const sha = context.payload.check_run && context.payload.check_run.head_sha;
    if (sha) {
      const { data } = await github.rest.repos.listPullRequestsAssociatedWithCommit(
        { owner: context.repo.owner, repo: context.repo.repo, commit_sha: sha });
      pr = data && data[0];
    }
  }
  if (!pr || !pr.number) return null;
  const { data: full } = await github.rest.pulls.get({ owner: context.repo.owner, repo: context.repo.repo, pull_number: pr.number });
  return { number: full.number, title: full.title, url: full.html_url, author: full.user.login, base: full.base.ref, head: full.head.ref };
}

/** 入口 A：托管 Code Review 的 `Claude Code Review` check_run 完成事件。 */
async function notifyFromCheckRun({ github, context, core }) {
  const { classifyCheckRun, shouldNotify } = require('./report');
  const env = readFeishuEnv(core); if (!env) return;
  const cr = context.payload.check_run;
  const cls = classifyCheckRun(cr, process.env);
  if (!shouldNotify(cls)) { core.info(`评审 verdict=${cls.verdict}（reason=${cls.reason}），不通知`); return; }
  const prData = await resolvePr(github, context, (cr.pull_requests && cr.pull_requests[0]) || null);
  if (!prData) { core.info('该 check 未关联 PR，跳过'); return; }
  if (prData.base !== 'main') { core.info(`base=${prData.base} 非 main，跳过`); return; }
  const dedupeKey = `${cr.id || cr.head_sha}:${cr.head_sha}:${(cr.completed_at || cr.updated_at || '')}`;
  await postToThread({ github, context, core }, { cls, prData, env, dedupeKey });
}

/** 入口 B：自托管 Action 评审产出的 codereview-result.json + 当前 PR 上下文。 */
async function notifyFromActionResult({ github, context, core, result }) {
  const { classifyActionResult, shouldNotify } = require('./report');
  const env = readFeishuEnv(core); if (!env) return;
  const cls = classifyActionResult(result, process.env);
  if (!shouldNotify(cls)) { core.info(`评审 verdict=${cls.verdict}（reason=${cls.reason}），不通知`); return; }
  const eventPr = context.payload.pull_request;
  const prData = await resolvePr(github, context, eventPr ? { number: eventPr.number } : null);
  if (!prData) { core.info('无法定位 PR，跳过'); return; }
  if (prData.base !== 'main') { core.info(`base=${prData.base} 非 main，跳过`); return; }
  const dedupeKey = `action:${context.sha}:${result && result.run_id ? result.run_id : process.env.GITHUB_RUN_ID || ''}`;
  await postToThread({ github, context, core }, { cls, prData, env, dedupeKey });
}

module.exports = { notifyFromCheckRun, notifyFromActionResult, validFeishuMessageId, shouldRecreateRootOnReplyFailure, MARK };
