#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const {
  CODEX_CHECK_APP_SLUG,
  CODEX_REVIEW_LOGIN,
  CODEX_REVIEW_USER_ID,
  DEFINITE_SEND_MAX_ATTEMPTS,
  DELIVERY_MARK,
  DELIVERY_REPAIR_MARK,
  DRAIN_BATCH_SIZE,
  EVENT_MARK,
  GITHUB_ATTEMPT_BUDGET_MS,
  GITHUB_REQUEST_TIMEOUT_MS,
  HISTORY_MAX_PAGES,
  HISTORY_PAGE_SIZE,
  HISTORY_MAX_ATTEMPTS,
  HISTORY_MAX_AGE_MS,
  MARK,
  REPAIR_MARK,
  STATE_MARK,
  appendDeliveryAndConfirm,
  canonicalEventRef,
  contentWithDeliveryToken,
  decodeDeliveryRecord,
  decodeEventRef,
  decodeRepairRecord,
  decodeThreadState,
  deliveryToken,
  drainQueuedCodexEvents,
  encodeDeliveryRecord,
  encodeEventRef,
  encodeThreadState,
  enqueueEventRef,
  enqueueOfficialCodexEvent,
  findDeliveryInFeishuHistory,
  isTrustedMarkerComment,
  parseRetryAfterMs,
  readDeliveryLedger,
  readEventQueue,
  readThreadState,
  repairDelivery,
  repairLegacyConflict,
  resolvePrNumber,
  scheduleQueuedCodexDrain,
  sweepQueuedCodexReviews,
  withGithubAttemptBudget,
  withGithubRetry,
  MAX_GITHUB_RETRY_DELAY_MS,
  WATCHDOG_MAX_DISPATCHES,
  WATCHDOG_MAX_PRS_PER_RUN,
  WATCHDOG_SHARDS,
  decodeDeliveryRepairRecord,
  historyRecoveryExhausted,
  watchdogWindow,
} = require('./notify');

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function response(json, status = 200) {
  return { ok: status >= 200 && status < 300, status, text: async () => JSON.stringify(json) };
}

function makeCore() {
  return {
    failures: [], warnings: [], infos: [], outputs: {},
    setFailed(message) { this.failures.push(String(message)); },
    warning(message) { this.warnings.push(String(message)); },
    info(message) { this.infos.push(String(message)); },
    setSecret() {},
    setOutput(key, value) { this.outputs[key] = value; },
  };
}

function botComment(id, body) {
  return { id, body, author_association: 'NONE', user: { login: 'github-actions[bot]', type: 'Bot' } };
}

const HEAD = 'a'.repeat(40);
const CREATED = '2026-08-05T12:00:00Z';

function makeCheck(id, options = {}) {
  const important = options.important === false ? 0 : 1;
  return {
    id,
    name: 'Codex Code Review',
    status: 'completed',
    conclusion: 'success',
    head_sha: options.headSha || HEAD,
    completed_at: options.createdAt || CREATED,
    pull_requests: [{ number: 42 }],
    app: { name: 'ChatGPT Codex Connector', slug: CODEX_CHECK_APP_SLUG, owner: { login: 'openai' } },
    check_suite: { app: { slug: CODEX_CHECK_APP_SLUG } },
    output: {
      title: important ? 'Review found issues' : 'No issues',
      summary: '',
      text: `<!-- bughunter-severity: {"normal":${important},"nit":0,"pre_existing":0} -->`,
    },
  };
}

function makeReview(id = 701) {
  return {
    id,
    body: 'Codex review',
    state: 'COMMENTED',
    submitted_at: CREATED,
    html_url: 'https://github.com/zettlab/demo/pull/42#pullrequestreview-701',
    user: { id: Number(CODEX_REVIEW_USER_ID), login: CODEX_REVIEW_LOGIN, type: 'Bot' },
  };
}

function makeGithub({
  comments = [], checks = new Map(), reviews = new Map(), reviewComments = [], associated = [], openPulls = [],
  checkpointCreateLosesResponse = false,
  beforeCheckpointCreate = null,
} = {}) {
  const listComments = async (options) => {
    assert(options.request.timeout === GITHUB_REQUEST_TIMEOUT_MS, 'comment list API timeout');
    assert(!Object.prototype.hasOwnProperty.call(options, 'sort') &&
      !Object.prototype.hasOwnProperty.call(options, 'direction'),
    'single-issue listComments 不发送未支持的 sort/direction 参数');
    if (Number(options.page || 1) === 1 && !options.since) github.commentReads += 1;
    let listed = comments.slice().sort((a, b) => Number(a.id) - Number(b.id));
    if (options.since) listed = listed.filter((comment) =>
      !comment.created_at || Date.parse(comment.created_at) > Date.parse(options.since));
    const size = Number(options.per_page || 100);
    const start = (Number(options.page || 1) - 1) * size;
    return { data: listed.slice(start, start + size) };
  };
  const listReviewComments = async () => {};
  const dispatches = [];
  let checkpointResponseLost = false;
  let checkpointInterleaved = false;
  const github = {
    dispatches,
    commentReads: 0,
    checkReads: new Map(),
    graphql: async (_query, variables) => {
      const listed = comments.slice().sort((a, b) => Number(a.id) - Number(b.id));
      const end = variables.before === null || typeof variables.before === 'undefined'
        ? listed.length : Number(variables.before);
      const start = Math.max(0, end - 100);
      return { repository: { pullRequest: { comments: {
        pageInfo: { hasPreviousPage: start > 0, startCursor: String(start) },
        nodes: listed.slice(start, end).map((comment) => ({
          databaseId: Number(comment.id), body: comment.body,
          createdAt: comment.created_at || CREATED,
          authorAssociation: comment.author_association || 'NONE',
          author: { login: comment.user && comment.user.login, __typename: comment.user && comment.user.type || 'User' },
        })),
      } } } };
    },
    rest: {
      repos: {
        listPullRequestsAssociatedWithCommit: async (options) => {
          assert(options.request.timeout === GITHUB_REQUEST_TIMEOUT_MS, 'associated API timeout');
          return { data: associated };
        },
        createDispatchEvent: async (options) => {
          assert(options.request.timeout === GITHUB_REQUEST_TIMEOUT_MS, 'dispatch API timeout');
          dispatches.push(options.client_payload);
          return { data: {} };
        },
      },
      pulls: {
        list: async (options) => {
          assert(options.request.timeout === GITHUB_REQUEST_TIMEOUT_MS, 'pull list API timeout');
          const start = (Number(options.page || 1) - 1) * Number(options.per_page || openPulls.length || 1);
          return { data: openPulls.slice(start, start + Number(options.per_page || openPulls.length || 1)) };
        },
        get: async (options) => ({ data: {
          number: options.pull_number,
          title: '测试 PR',
          html_url: `https://github.com/zettlab/demo/pull/${options.pull_number}`,
          user: { login: 'owner' },
          base: { ref: 'main' },
          head: { ref: 'feature/test', sha: HEAD },
          state: 'open',
        } }),
        getReview: async (options) => ({ data: reviews.get(String(options.review_id)) }),
        listReviewComments,
      },
      checks: {
        get: async (options) => {
          const key = String(options.check_run_id);
          github.checkReads.set(key, (github.checkReads.get(key) || 0) + 1);
          return { data: checks.get(key) };
        },
      },
      issues: {
        listComments,
        createComment: async (options) => {
          assert(options.request.timeout === GITHUB_REQUEST_TIMEOUT_MS, 'comment API timeout');
          if (!checkpointInterleaved && beforeCheckpointCreate &&
              String(options.body).includes('codex-review-feishu-checkpoint:')) {
            checkpointInterleaved = true;
            beforeCheckpointCreate(comments);
          }
          const nextCommentId = comments.reduce((max, item) => Math.max(max, Number(item.id) || 0), 999) + 1;
          const comment = botComment(nextCommentId, options.body);
          const latestCreatedAt = comments.reduce((max, item) =>
            item.created_at ? Math.max(max, Date.parse(item.created_at)) : max, Date.parse(CREATED));
          comment.created_at = new Date(latestCreatedAt + 1000).toISOString();
          comments.push(comment);
          if (checkpointCreateLosesResponse && !checkpointResponseLost &&
              String(options.body).includes('codex-review-feishu-checkpoint:')) {
            checkpointResponseLost = true;
            throw new Error('simulated lost checkpoint response');
          }
          return { data: comment };
        },
        updateComment: async (options) => {
          const comment = comments.find((item) => Number(item.id) === Number(options.comment_id));
          if (!comment) throw new Error('comment not found');
          comment.body = options.body;
          return { data: comment };
        },
        getComment: async (options) => ({ data: comments.find((item) => Number(item.id) === Number(options.comment_id)) }),
      },
    },
    paginate: async (fn, options) => {
      assert(options.request.timeout === GITHUB_REQUEST_TIMEOUT_MS, 'paginate timeout');
      if (fn === listComments) { github.commentReads += 1; return comments.slice(); }
      if (fn === listReviewComments) return reviewComments.slice();
      throw new Error('unexpected paginate target');
    },
  };
  return github;
}

function context(eventName = 'check_run', payload = {}) {
  return { eventName, repo: { owner: 'zettlab', repo: 'demo' }, payload, sha: HEAD };
}

function eventRef(id = '501', createdAt = CREATED) {
  return canonicalEventRef({
    version: 1,
    eventType: 'check_run',
    eventId: id,
    repo: 'zettlab/demo',
    pr: 42,
    headSha: HEAD,
    createdAt,
  });
}

function legacyBody(mid) {
  return `${MARK}${mid} -->\n<sub>Codex 代码评审话题锚点（自动维护，请勿删除）</sub>`;
}

async function main() {
  const oldFetch = global.fetch;
  const oldEnv = { ...process.env };
  process.env.CODEREVIEW_FEISHU_APP_ID = 'app-id';
  process.env.CODEREVIEW_FEISHU_APP_SECRET = 'app-secret';
  process.env.CODEREVIEW_FEISHU_CHAT_ID = 'chat-id';
  process.env.RESOLVED_PR_NUMBER = '42';
  process.env.CODEREVIEW_NOTIFY_ON = 'important';
  const keyPair = crypto.generateKeyPairSync('ed25519');
  const repairPrivateKey = keyPair.privateKey.export({ type: 'pkcs8', format: 'pem' });
  const repairPublicKey = keyPair.publicKey.export({ type: 'spki', format: 'pem' });
  const currentKeyPair = crypto.generateKeyPairSync('ed25519');
  const currentRepairPrivateKey = currentKeyPair.privateKey.export({ type: 'pkcs8', format: 'pem' });
  const currentRepairPublicKey = currentKeyPair.publicKey.export({ type: 'spki', format: 'pem' });
  const repairKeyring = { old: repairPublicKey, current: currentRepairPublicKey };
  process.env.CODEX_FEISHU_REPAIR_PUBLIC_KEYS = JSON.stringify(repairKeyring);
  try {
    assert(GITHUB_ATTEMPT_BUDGET_MS === 180000, '共享 GitHub attempt budget 固定 180s');
    assert(GITHUB_REQUEST_TIMEOUT_MS === 15000, 'GitHub API timeout 固定 15s');
    assert(DRAIN_BATCH_SIZE > 0 && DRAIN_BATCH_SIZE <= 10, 'drain batch 必须有小型上限');
    assert(HISTORY_PAGE_SIZE === 50 && HISTORY_MAX_PAGES > 0, 'Feishu history 有固定页数/消息数上限');
    assert(HISTORY_MAX_ATTEMPTS === 8 && HISTORY_MAX_AGE_MS === 24 * 60 * 60 * 1000,
      'history 恢复以 8 次或 24h 为硬上限');
    assert(DEFINITE_SEND_MAX_ATTEMPTS === 3, 'definitely-not-sent 最多发送三次');
    const retryError = new Error('limited');
    retryError.response = { headers: { 'Retry-After': '90' } };
    assert(parseRetryAfterMs(retryError, 0) === MAX_GITHUB_RETRY_DELAY_MS, 'Retry-After clamp 30s');
    let attempts = 0;
    const retried = await withGithubRetry({
      core: makeCore(), label: 'retry', sleepFn: async () => {},
      operation: async () => { attempts += 1; if (attempts < 3) throw new Error('transient'); return 'ok'; },
    });
    assert(retried.ok && attempts === 3, 'GitHub API 最多三次并可恢复');
    let nestedOperations = 0;
    await withGithubAttemptBudget(async () => {
      await withGithubAttemptBudget(async () => {
        const result = await withGithubRetry({
          core: makeCore(), label: 'budget', nowFn: () => 1001,
          operation: async () => { nestedOperations += 1; },
        });
        assert(!result.ok, '嵌套调用不可延长外层 deadline');
      }, { deadlineMs: 999999, nowFn: () => 1001 });
    }, { deadlineMs: GITHUB_REQUEST_TIMEOUT_MS + 1000, nowFn: () => 0 });
    assert(nestedOperations === 0, '预算不足时不发 GitHub 请求');

    const workflow = fs.readFileSync(path.join(__dirname, '../../.github/workflows/codex-review-feishu.yml'), 'utf8');
    assert(workflow.includes('repository_dispatch:') && workflow.includes('codex-review-feishu-drain'), 'workflow 接收 trusted self-dispatch');
    const enqueueIndex = workflow.indexOf('  enqueue:');
    const reportIndex = workflow.indexOf('  report:');
    assert(enqueueIndex > 0 && enqueueIndex < reportIndex, 'immutable event 在并发 report 前入队');
    const reportJob = workflow.slice(reportIndex, workflow.indexOf('  schedule_continuation:'));
    assert(workflow.includes('  kick_consumer:') && workflow.includes('always()') &&
      workflow.includes('Kick default-branch queue consumer even when enqueue failed'), 'enqueue 任意结果后都 kick durable consumer');
    assert(reportJob.includes("github.event_name == 'repository_dispatch'") &&
      reportJob.includes('environment: codex-review-feishu-production') &&
      !reportJob.includes('contents: write'), 'secret worker 仅默认分支 repository_dispatch 且挂 protected environment');
    assert(workflow.includes('  schedule_continuation:') && workflow.includes('fresh budget') &&
      workflow.includes('  queue_watchdog:'), 'remaining queue 用独立预算续批并由可信 schedule watchdog 扫描');
    assert(workflow.includes("needs.report.outputs.continuation_mode == 'backlog'"),
      '只有 backlog 可 immediate dispatch；retry 等 watchdog 到期');
    assert(workflow.includes('  kick_after_repair:') && workflow.includes('Kick a fresh default-branch consumer after repair'), 'repair 后恢复 fresh consumer');
    assert(workflow.includes('repairLegacyConflict') && workflow.includes('CODEX_FEISHU_REPAIR_PRIVATE_KEY') &&
      workflow.includes('CODEX_FEISHU_REPAIR_PUBLIC_KEYS') && workflow.includes('CODEX_FEISHU_REPAIR_KEY_ID'),
    'repair 使用 environment 私钥签名、keyId + public keyring 验签');
    assert(workflow.includes('repair_action:') && workflow.includes('select_mid') &&
      workflow.includes('retry') && workflow.includes('discard'),
    'workflow repair 显式选择 select_mid/retry/discard action');
    const repairJobIndex = workflow.indexOf('  repair_pending:');
    const repairJob = workflow.slice(repairJobIndex, workflow.indexOf('  kick_after_repair:'));
    const adminGateIndex = repairJob.indexOf('id: admin_gate');
    const repairSecretIndex = repairJob.indexOf('CODEX_FEISHU_REPAIR_PRIVATE_KEY:');
    assert(repairJobIndex > 0 && adminGateIndex > 0 && repairSecretIndex > adminGateIndex &&
      repairJob.includes('github.rest.repos.getCollaboratorPermissionLevel') &&
      repairJob.includes("permission !== 'admin'") &&
      repairJob.includes('username: actor'),
    'repair secret 前用 pinned github-script 精确验证 workflow actor 的 repository admin permission');
    assert(repairJob.includes("github.event_name == 'workflow_dispatch'") &&
      repairJob.includes('github.ref_name == github.event.repository.default_branch') &&
      repairJob.includes('environment: codex-review-feishu-repair'),
    'repair job 自身仅允许 default-branch workflow_dispatch，并保留 repair environment branch policy');
    assert(repairJob.includes('REPAIR_ACTOR: ${{ steps.admin_gate.outputs.actor }}') &&
      repairJob.includes("createHash('sha256')") && repairJob.includes('repairAuditBinding') &&
      repairJob.includes('runId: auditRunId, operator: actor') &&
      repairJob.includes('runId: auditRunId,'),
    '所有 signed repair 用 auditRunId 绑定 admin actor、run 与完整 workflow_dispatch inputs');
    assert(workflow.includes('github.event.review.user.id == 199175422') &&
      workflow.includes("github.event.review.user.login == 'chatgpt-codex-connector[bot]'") &&
      workflow.includes("github.event.check_run.app.slug == 'chatgpt-codex-connector'"), 'workflow actor 使用精确 ID/login/app slug');
    assert(!/uses:\s+actions\/[^@\s]+@v\d+\b/.test(workflow), 'Actions 全部 pin 完整 SHA');

    const ref = eventRef();
    assert(decodeEventRef(encodeEventRef(ref)).eventKey === ref.eventKey, 'immutable event ref 严格 round-trip');
    assert(encodeEventRef(ref).startsWith(EVENT_MARK), 'event ref 使用独立 marker');
    const token = deliveryToken(ref.eventKey);
    const rawCard = JSON.stringify({ header: { title: { tag: 'plain_text', content: 'x' } }, body: { elements: [] } });
    const card = JSON.parse(contentWithDeliveryToken(rawCard, token));
    assert(card.header.subtitle.tag === 'plain_text' && card.header.subtitle.content === token, 'root/reply 卡片嵌入 exact plain_text delivery token');

    const queueComments = [];
    const queueGithub = makeGithub({ comments: queueComments });
    await enqueueEventRef({ github: queueGithub, context: context(), core: makeCore(), ref });
    await enqueueEventRef({ github: queueGithub, context: context(), core: makeCore(), ref });
    const queue = await readEventQueue({ github: queueGithub, context: context(), core: makeCore(), prNum: 42 });
    assert(queue.ok && queue.events.length === 1 && queueComments.length === 2, '重复 delivery 的 immutable ref 折叠且不丢失');

    const eventPayload = {
      check_run: makeCheck(502),
    };
    const officialComments = [];
    const officialGithub = makeGithub({ comments: officialComments });
    await enqueueOfficialCodexEvent({
      github: officialGithub,
      context: context('check_run', eventPayload),
      core: makeCore(),
    });
    const officialQueue = await readEventQueue({ github: officialGithub, context: context(), core: makeCore(), prNum: 42 });
    assert(officialQueue.events.length === 1 && officialQueue.events[0].ref.eventId === '502', '官方事件只存 immutable ID/ref，不存 cls 或 PR body');

    const legacyComments = [
      { id: 1, body: legacyBody('om_1111111111111111'), author_association: 'MEMBER', user: { login: 'one', type: 'User' } },
      { id: 2, body: legacyBody('om_2222222222222222'), author_association: 'MEMBER', user: { login: 'two', type: 'User' } },
    ];
    const legacyGithub = makeGithub({ comments: legacyComments });
    const beforeRepair = await readThreadState({
      github: legacyGithub, context: context(), core: makeCore(), prNum: 42, allowLegacyConflict: true,
    });
    assert(beforeRepair.kind === 'legacy-conflict', '不同 legacy message_id 显式进入 generation-0 conflict');
    const repaired = await repairLegacyConflict({
      github: legacyGithub, context: context(), core: makeCore(), prNum: 42,
      messageId: 'om_3333333333333333', runId: '12345', privateKey: repairPrivateKey,
      keyring: repairKeyring, keyId: 'old',
    });
    assert(repaired.ok, 'operator canonical mid 生成更高 generation 签名 final');
    const afterRepair = await readThreadState({ github: legacyGithub, context: context(), core: makeCore(), prNum: 42 });
    assert(afterRepair.ok && afterRepair.kind === 'final' && afterRepair.generation === 1 &&
      afterRepair.messageId === 'om_3333333333333333', '签名 repair 仅迁移 legacy generation-0 conflict');
    const repairComment = legacyComments.find((comment) => String(comment.body).includes(REPAIR_MARK));
    assert(decodeRepairRecord(repairComment.body, repairKeyring).legacyHash === beforeRepair.legacyHash, 'repair 签名绑定 legacy snapshot 与 keyId');
    assert(decodeRepairRecord(repairComment.body, { current: currentRepairPublicKey }) === null,
      'keyring 移除旧 key 后旧签名不可验证');
    assert(decodeRepairRecord(repairComment.body, { old: repairPublicKey, current: currentRepairPublicKey }).keyId === 'old',
      '轮换后保留旧公钥仍可验证历史 repair');
    legacyComments.push(botComment(9999, encodeThreadState({
      version: 2, state: 'final', repo: 'zettlab/demo', pr: 42, generation: 1,
      claimId: 'ffffffffffffffffffffffffffffffff', messageId: 'om_4444444444444444',
    })));
    const v2Conflict = await readThreadState({ github: legacyGithub, context: context(), core: makeCore(), prNum: 42 });
    assert(!v2Conflict.ok && v2Conflict.kind === 'conflict', 'signed repair 不得覆盖任何 v2 conflict');

    const historyRecord = {
      version: 1, state: 'uncertain', repo: 'zettlab/demo', pr: 42, eventKey: ref.eventKey,
      attempt: 1, mode: 'reply', targetRoot: 'om_aaaaaaaaaaaaaaaa',
      threadGeneration: null, threadClaimId: null, token, sentAt: CREATED,
      messageId: null, historyAttempts: 0, nextCheckAt: '2026-08-05T12:00:30Z', reason: 'ambiguous',
    };
    assert(!historyRecoveryExhausted(historyRecord, 7, Date.parse(CREATED) + HISTORY_MAX_AGE_MS - 1),
      'history 第 7 次且未满 24h 仍可退避');
    assert(historyRecoveryExhausted(historyRecord, 8, Date.parse(CREATED) + 1) &&
      historyRecoveryExhausted(historyRecord, 1, Date.parse(CREATED) + HISTORY_MAX_AGE_MS),
    'history 达 8 次或 24h 任一边界即转 manual');
    assert(decodeDeliveryRecord(encodeDeliveryRecord(historyRecord)).state === 'uncertain', 'delivery ledger 严格 round-trip');
    let historyCalls = 0;
    global.fetch = async (url) => {
      historyCalls += 1;
      assert(String(url).includes('container_id_type=chat') && String(url).includes('page_size=50'), 'history 使用 bounded chat query');
      const historyCard = JSON.stringify({ header: { subtitle: { tag: 'plain_text', content: token } }, body: { elements: [] } });
      return response({ code: 0, data: { has_more: false, items: [{
        message_id: 'om_5555555555555555', msg_type: 'interactive', deleted: false,
        sender: { sender_type: 'app', id: 'app-id' }, body: { content: historyCard },
      }] } });
    };
    const exact = await findDeliveryInFeishuHistory({
      env: { appId: 'app-id', chatId: 'chat-id' }, token: 'tenant', record: historyRecord,
    });
    assert(exact.ok && exact.complete && exact.matches.length === 1 && historyCalls === 1, 'history 仅接受 exact token + exact app sender 唯一匹配');
    global.fetch = async () => response({ code: 0, data: { has_more: true, page_token: 'next', items: [] } });
    const incomplete = await findDeliveryInFeishuHistory({
      env: { appId: 'app-id', chatId: 'chat-id' }, token: 'tenant', record: historyRecord,
    });
    assert(!incomplete.complete && historyCalls <= HISTORY_MAX_PAGES + 1, 'history 达页数上限时 fail closed，不把 0 当未发送证明');

    const flowComments = [];
    const checks = new Map([['601', makeCheck(601)]]);
    const flowGithub = makeGithub({ comments: flowComments, checks });
    const flowContext = context('repository_dispatch', { client_payload: { pr_number: '42' } });
    const flowRef = eventRef('601');
    await enqueueEventRef({ github: flowGithub, context: flowContext, core: makeCore(), ref: flowRef });
    let rootPosts = 0;
    global.fetch = async (url) => {
      const value = String(url);
      if (value.endsWith('/auth/v3/tenant_access_token/internal')) return response({ code: 0, tenant_access_token: 'tenant-token' });
      if (value.includes('receive_id_type=chat_id')) {
        rootPosts += 1;
        throw new Error('network reset after accept');
      }
      if (value.includes('/im/v1/messages?container_id_type=chat')) {
        return response({ code: 0, data: { has_more: false, items: [] } });
      }
      throw new Error(`unexpected request ${value}`);
    };
    const firstCore = makeCore();
    await drainQueuedCodexEvents({ github: flowGithub, context: flowContext, core: firstCore });
    assert(rootPosts === 1 && firstCore.failures.length > 0, '首次 ambiguous send 持久化后显式失败');
    const preparingIndex = flowComments.findIndex((comment) => {
      const record = decodeDeliveryRecord(comment.body);
      return record && record.eventKey === flowRef.eventKey && record.state === 'preparing';
    });
    const pendingIndex = flowComments.findIndex((comment) => {
      const state = decodeThreadState(comment.body);
      return state && state.state === 'pending';
    });
    assert(preparingIndex >= 0 && pendingIndex > preparingIndex, 'root 先 durable preparing，再写 thread pending，崩溃可 deterministic claim 接管');
    const firstDeliveryCore = makeCore();
    const firstDelivery = await readDeliveryLedger({ github: flowGithub, context: flowContext, core: firstDeliveryCore, prNum: 42 });
    const firstDeliveryState = firstDelivery.latest.get(flowRef.eventKey);
    assert(firstDeliveryState && firstDeliveryState.state === 'uncertain',
      `429/5xx/disconnect 进入 durable uncertain: ${JSON.stringify(firstDeliveryCore.failures)}`);
    const immediateCore = makeCore();
    await drainQueuedCodexEvents({ github: flowGithub, context: flowContext, core: immediateCore });
    assert(rootPosts === 1 && immediateCore.failures.length > 0, '未到 history retry 时间绝不重发');

    const oldDateNow = Date.now;
    Date.now = () => Date.parse(firstDelivery.latest.get(flowRef.eventKey).nextCheckAt) + 1;
    process.env.DRAIN_SCHEDULE_UNCERTAIN = 'true';
    const zeroCore = makeCore();
    await drainQueuedCodexEvents({ github: flowGithub, context: flowContext, core: zeroCore });
    Date.now = oldDateNow;
    assert(rootPosts === 1 && zeroCore.outputs.needs_dispatch === 'true' && flowGithub.dispatches.length === 0,
      'history 0 match 保持非终态并请求独立续调度，绝不重发');
    await scheduleQueuedCodexDrain({ github: flowGithub, context: flowContext, core: makeCore(), prNum: 42 });
    assert(flowGithub.dispatches.length === 1, '续调度在独立 fresh-budget 调用完成');

    const afterZero = await readDeliveryLedger({ github: flowGithub, context: flowContext, core: makeCore(), prNum: 42 });
    const oldDateNowMultiple = Date.now;
    Date.now = () => Date.parse(afterZero.latest.get(flowRef.eventKey).nextCheckAt) + 1;
    global.fetch = async (url) => {
      const value = String(url);
      if (value.endsWith('/auth/v3/tenant_access_token/internal')) return response({ code: 0, tenant_access_token: 'tenant-token' });
      if (value.includes('/im/v1/messages?container_id_type=chat')) {
        const historyCard = JSON.stringify({ header: { subtitle: { tag: 'plain_text', content: deliveryToken(flowRef.eventKey) } }, body: { elements: [] } });
        return response({ code: 0, data: { has_more: false, items: [
          { message_id: 'om_6666666666666666', msg_type: 'interactive', deleted: false, sender: { sender_type: 'app', id: 'app-id' }, body: { content: historyCard } },
          { message_id: 'om_7777777777777777', msg_type: 'interactive', deleted: false, sender: { sender_type: 'app', id: 'app-id' }, body: { content: historyCard } },
        ] } });
      }
      throw new Error(`unexpected request ${value}`);
    };
    const multipleCore = makeCore();
    await drainQueuedCodexEvents({ github: flowGithub, context: flowContext, core: multipleCore });
    Date.now = oldDateNowMultiple;
    const manualLedger = await readDeliveryLedger({ github: flowGithub, context: flowContext, core: makeCore(), prNum: 42 });
    assert(manualLedger.latest.get(flowRef.eventKey).state === 'manual' && multipleCore.outputs.needs_dispatch === 'false',
      'history 多匹配进入 manual，事件仍在 queue 且避免热循环');
    assert(JSON.stringify(manualLedger.latest.get(flowRef.eventKey).candidateMessageIds) ===
      JSON.stringify(['om_6666666666666666', 'om_7777777777777777']),
    'manual ledger 持久化排序后的 exact-match candidate IDs');
    const manualSequence = flowComments.map((comment) => decodeDeliveryRecord(comment.body))
      .filter((record) => record && record.eventKey === flowRef.eventKey);
    const barrierRef = eventRef('602', '2026-08-05T12:00:01Z');
    checks.set('602', makeCheck(602, { important: false, createdAt: barrierRef.createdAt }));
    await enqueueEventRef({ github: flowGithub, context: flowContext, core: makeCore(), ref: barrierRef });
    const barrierCore = makeCore();
    await drainQueuedCodexEvents({ github: flowGithub, context: flowContext, core: barrierCore });
    const barrierLedger = await readDeliveryLedger({ github: flowGithub, context: flowContext, core: makeCore(), prNum: 42 });
    assert(!barrierLedger.latest.has(barrierRef.eventKey) && barrierCore.outputs.needs_dispatch === 'false' &&
      barrierCore.outputs.continuation_mode === 'watchdog',
    'manual root + matching pending 是硬 barrier，后续事件不处理且只交 watchdog/manual repair');
    const rejectedRetry = await repairDelivery({
      github: flowGithub, context: flowContext, core: makeCore(), prNum: 42,
      eventKey: flowRef.eventKey, messageId: null, action: 'retry', runId: 'run-retry-rejected', operator: 'maintainer',
      privateKey: currentRepairPrivateKey, keyring: repairKeyring, keyId: 'current',
    });
    assert(!rejectedRetry.ok, 'ambiguous/history_multiple manual 不得 operator retry，避免重复发送');
    const rejectedSelection = await repairDelivery({
      github: flowGithub, context: flowContext, core: makeCore(), prNum: 42,
      eventKey: flowRef.eventKey, messageId: 'om_8888888888888888', action: 'select_mid', runId: 'run-1', operator: 'maintainer',
      privateKey: currentRepairPrivateKey, keyring: repairKeyring, keyId: 'current',
    });
    assert(!rejectedSelection.ok, 'delivery repair selected mid 必须属于持久化 candidate set');
    const deliveryRepaired = await repairDelivery({
      github: flowGithub, context: flowContext, core: makeCore(), prNum: 42,
      eventKey: flowRef.eventKey, messageId: 'om_6666666666666666',
      action: 'select_mid', runId: 'run-2', operator: 'maintainer', privateKey: currentRepairPrivateKey,
      keyring: repairKeyring, keyId: 'current',
    });
    assert(deliveryRepaired.ok, 'delivery repair 精确绑定 eventKey + canonical message_id');
    const repairedDeliveryLedger = await readDeliveryLedger({ github: flowGithub, context: flowContext, core: makeCore(), prNum: 42 });
    assert(repairedDeliveryLedger.latest.get(flowRef.eventKey).state === 'done', 'manual event 仅在显式 repair 后完成，不吞事件');
    const deliveryRepairComment = flowComments.find((comment) => String(comment.body).includes(DELIVERY_REPAIR_MARK));
    const deliveryRepair = decodeDeliveryRepairRecord(deliveryRepairComment.body, repairKeyring);
    assert(deliveryRepair && deliveryRepair.eventKey === flowRef.eventKey &&
      deliveryRepair.messageId === 'om_6666666666666666' && deliveryRepair.keyId === 'current' &&
      deliveryRepair.action === 'select_mid' && deliveryRepair.priorReason === 'history_multiple' &&
      typeof deliveryRepair.priorManualHash === 'string' && deliveryRepair.priorManualHash.length === 64,
    'manual -> done companion 签名绑定 action/eventKey/prior hash+reason/candidates/selected/run/operator/keyId');
    await drainQueuedCodexEvents({ github: flowGithub, context: flowContext, core: makeCore() });
    const afterBarrierLedger = await readDeliveryLedger({ github: flowGithub, context: flowContext, core: makeCore(), prNum: 42 });
    assert(afterBarrierLedger.latest.get(barrierRef.eventKey).state === 'skipped',
      'select_mid 完成 root 后解除 pending barrier，后续 FIFO 可继续');

    const cloneManualSequence = (targetRef, mode) => manualSequence
      .filter((record) => mode === 'root' || record.state !== 'preparing')
      .map((record) => ({
      ...record,
      eventKey: targetRef.eventKey,
      token: deliveryToken(targetRef.eventKey),
      mode,
      targetRoot: mode === 'reply' ? 'om_6666666666666666' : null,
      threadGeneration: mode === 'reply' ? null : record.threadGeneration,
      threadClaimId: mode === 'reply' ? null : record.threadClaimId,
      }));

    const passReplyRef = eventRef('611', '2026-08-05T12:01:01Z');
    const passReleasedRef = eventRef('612', '2026-08-05T12:01:02Z');
    const passNextRef = eventRef('613', '2026-08-05T12:01:03Z');
    const passReplySequence = cloneManualSequence(passReplyRef, 'reply');
    const passReleasedSequence = cloneManualSequence(passReleasedRef, 'root');
    const releasedManual = passReleasedSequence[passReleasedSequence.length - 1];
    let passCommentId = 1;
    const passComments = [passReplyRef, passReleasedRef, passNextRef]
      .map((queuedRef) => botComment(passCommentId++, encodeEventRef(queuedRef)));
    for (const record of passReplySequence) passComments.push(botComment(passCommentId++, encodeDeliveryRecord(record)));
    for (const record of passReleasedSequence) passComments.push(botComment(passCommentId++, encodeDeliveryRecord(record)));
    passComments.push(botComment(passCommentId++, encodeThreadState({
      version: 2, state: 'pending', repo: 'zettlab/demo', pr: 42,
      generation: releasedManual.threadGeneration, claimId: releasedManual.threadClaimId, messageId: null,
    })));
    passComments.push(botComment(passCommentId++, encodeThreadState({
      version: 2, state: 'released', repo: 'zettlab/demo', pr: 42,
      generation: releasedManual.threadGeneration, claimId: releasedManual.threadClaimId, messageId: null,
    })));
    const passChecks = new Map([['613', makeCheck(613, { important: false, createdAt: passNextRef.createdAt })]]);
    const passGithub = makeGithub({ comments: passComments, checks: passChecks });
    await drainQueuedCodexEvents({ github: passGithub, context: flowContext, core: makeCore() });
    const passLedger = await readDeliveryLedger({ github: passGithub, context: flowContext, core: makeCore(), prNum: 42 });
    assert(passLedger.latest.get(passNextRef.eventKey).state === 'skipped',
      'manual reply 与 released root 均不构成 barrier，后续 FIFO 可继续');
    const repairedReplyAfterCursor = await repairDelivery({
      github: passGithub, context: flowContext, core: makeCore(), prNum: 42,
      eventKey: passReplyRef.eventKey, messageId: null, action: 'discard',
      runId: 'run-reply-after-cursor', operator: 'maintainer',
      privateKey: currentRepairPrivateKey, keyring: repairKeyring, keyId: 'current',
    });
    const repairedReleasedRootAfterCursor = await repairDelivery({
      github: passGithub, context: flowContext, core: makeCore(), prNum: 42,
      eventKey: passReleasedRef.eventKey, messageId: null, action: 'discard',
      runId: 'run-released-root-after-cursor', operator: 'maintainer',
      privateKey: currentRepairPrivateKey, keyring: repairKeyring, keyId: 'current',
    });
    const passRepairedLedger = await readDeliveryLedger({
      github: passGithub, context: flowContext, core: makeCore(), prNum: 42,
    });
    assert(repairedReplyAfterCursor.ok && repairedReleasedRootAfterCursor.ok &&
      passRepairedLedger.latest.get(passReplyRef.eventKey).state === 'skipped' &&
      passRepairedLedger.latest.get(passReleasedRef.eventKey).state === 'skipped',
    'manual reply 与 released root 越过 cursor 后仍保留 event/ref/repair chain，可执行 signed repair');

    const discardRef = eventRef('621', '2026-08-05T12:02:01Z');
    const discardSequence = cloneManualSequence(discardRef, 'reply').map((record, index, records) =>
      index === records.length - 1 ? { ...record, reason: 'history_exhausted', candidateMessageIds: [] } : record);
    let discardCommentId = 1;
    const discardComments = [botComment(discardCommentId++, encodeEventRef(discardRef))];
    for (const record of discardSequence) discardComments.push(botComment(discardCommentId++, encodeDeliveryRecord(record)));
    const discardGithub = makeGithub({ comments: discardComments });
    const discarded = await repairDelivery({
      github: discardGithub, context: flowContext, core: makeCore(), prNum: 42,
      eventKey: discardRef.eventKey, messageId: null, action: 'discard', runId: 'run-discard', operator: 'maintainer',
      privateKey: currentRepairPrivateKey, keyring: repairKeyring, keyId: 'current',
    });
    assert(discarded.ok, 'zero-candidate manual 可无需 message_id 执行 signed discard');
    const discardedLedger = await readDeliveryLedger({ github: discardGithub, context: flowContext, core: makeCore(), prNum: 42 });
    const discardRepairComment = discardComments.find((comment) => String(comment.body).includes(DELIVERY_REPAIR_MARK));
    const discardRepair = decodeDeliveryRepairRecord(discardRepairComment.body, repairKeyring);
    assert(discardedLedger.latest.get(discardRef.eventKey).state === 'skipped' && discardRepair.action === 'discard' &&
      discardRepair.messageId === null && discardRepair.priorReason === 'history_exhausted',
    'discard 签名绑定 prior manual hash/reason 并产生 terminal skipped');

    const retryRef = eventRef('622', '2026-08-05T12:02:02Z');
    const retryBase = cloneManualSequence(retryRef, 'root');
    const retryPreparing = retryBase.find((record) => record.state === 'preparing');
    const retrySending = retryBase.find((record) => record.state === 'sending');
    const retrySequence = [];
    for (let attempt = 1; attempt <= 1; attempt += 1) {
      const threadGeneration = retryPreparing.threadGeneration + attempt - 1;
      const threadClaimId = String(attempt).padStart(32, '0');
      retrySequence.push({ ...retryPreparing, attempt, state: 'preparing', sentAt: null, reason: null,
        threadGeneration, threadClaimId, historyAttempts: 0, nextCheckAt: null, candidateMessageIds: [] });
      retrySequence.push({ ...retrySending, attempt, state: 'sending', reason: null,
        threadGeneration, threadClaimId, historyAttempts: 0, nextCheckAt: null, candidateMessageIds: [] });
      retrySequence.push({ ...retrySending, attempt, state: 'not_sent', messageId: null,
        threadGeneration, threadClaimId, historyAttempts: 0,
        nextCheckAt: null, reason: 'definitely_not_sent', candidateMessageIds: [] });
    }
    retrySequence.push({ ...retrySequence[retrySequence.length - 1], state: 'manual',
      reason: 'definitely_not_sent_exhausted', candidateMessageIds: [] });
    let retryCommentId = 1;
    const retryNextRef = eventRef('623', '2026-08-05T12:02:03Z');
    const retryComments = [
      botComment(retryCommentId++, encodeEventRef(retryRef)),
      botComment(retryCommentId++, encodeEventRef(retryNextRef)),
    ];
    for (const record of retrySequence) retryComments.push(botComment(retryCommentId++, encodeDeliveryRecord(record)));
    const retryManual = retrySequence[retrySequence.length - 1];
    retryComments.push(botComment(retryCommentId++, encodeThreadState({
      version: 2, state: 'pending', repo: 'zettlab/demo', pr: 42,
      generation: retryManual.threadGeneration, claimId: retryManual.threadClaimId, messageId: null,
    })));
    retryComments.push(botComment(retryCommentId++, encodeThreadState({
      version: 2, state: 'released', repo: 'zettlab/demo', pr: 42,
      generation: retryManual.threadGeneration, claimId: retryManual.threadClaimId, messageId: null,
    })));
    const retryChecks = new Map([
      [retryRef.eventId, makeCheck(Number(retryRef.eventId), { createdAt: retryRef.createdAt })],
      [retryNextRef.eventId, makeCheck(Number(retryNextRef.eventId), {
        important: false, createdAt: retryNextRef.createdAt,
      })],
    ]);
    const retryGithub = makeGithub({ comments: retryComments, checks: retryChecks });
    await drainQueuedCodexEvents({ github: retryGithub, context: flowContext, core: makeCore() });
    const retriable = await repairDelivery({
      github: retryGithub, context: flowContext, core: makeCore(), prNum: 42,
      eventKey: retryRef.eventKey, messageId: null, action: 'retry', runId: 'run-retry', operator: 'maintainer',
      privateKey: currentRepairPrivateKey, keyring: repairKeyring, keyId: 'current',
    });
    assert(retriable.ok, '仅 definitely_not_sent_exhausted manual 可 signed retry 且无需 message_id');
    const retryLedger = await readDeliveryLedger({ github: retryGithub, context: flowContext, core: makeCore(), prNum: 42 });
    const retryRepairComment = retryComments.find((comment) => String(comment.body).includes(DELIVERY_REPAIR_MARK));
    const retryRepair = decodeDeliveryRepairRecord(retryRepairComment.body, repairKeyring);
    assert(retryLedger.latest.get(retryRef.eventKey).state === 'retrying' && retryRepair.action === 'retry' &&
      retryRepair.messageId === null && retryRepair.priorReason === 'definitely_not_sent_exhausted',
    'retry 签名绑定 definitely-not-sent prior manual 并进入 retrying');
    let replaySends = 0;
    global.fetch = async (url) => {
      if (String(url).includes('/auth/v3/tenant_access_token/internal')) {
        return response({ code: 0, tenant_access_token: 'tenant-token' });
      }
      if (String(url).includes('/im/v1/messages')) {
        replaySends += 1;
        if (replaySends === 1) return response({ code: 230001 }, 400);
        return response({ code: 0, data: { message_id: 'om_9999999999999999' } });
      }
      return response({ code: 0 });
    };
    await drainQueuedCodexEvents({ github: retryGithub, context: flowContext, core: makeCore() });
    const firstReplayLedger = await readDeliveryLedger({
      github: retryGithub, context: flowContext, core: makeCore(), prNum: 42,
    });
    assert(firstReplayLedger.latest.get(retryRef.eventKey).state === 'not_sent',
      'signed retry 第一次 replay 明确未发送后保留 cursor 前 actionable not_sent');
    await drainQueuedCodexEvents({ github: retryGithub, context: flowContext, core: makeCore() });
    const replayedLedger = await readDeliveryLedger({
      github: retryGithub, context: flowContext, core: makeCore(), prNum: 42,
    });
    const readsBeforeReplayRedelivery = retryGithub.checkReads.get(retryRef.eventId);
    await enqueueEventRef({ github: retryGithub, context: flowContext, core: makeCore(), ref: retryRef });
    await drainQueuedCodexEvents({ github: retryGithub, context: flowContext, core: makeCore() });
    assert(replayedLedger.latest.get(retryRef.eventKey).state === 'done' && replaySends === 2 &&
      retryGithub.checkReads.get(retryRef.eventId) === readsBeforeReplayRedelivery,
    'cursor 前 signed retry 派生 not_sent 会在下一 run 继续 replay 并终结；terminal tombstone 阻止 redelivery 重发');

    const {
      CHECKPOINT_MARK: checkpointMark,
      THREAD_REPAIR_MARK: threadRepairMark,
      decodeThreadRepairRecord,
      repairOrphanThread,
    } = require('./notify');
    const ownedRef = eventRef('623', '2026-08-05T12:02:03Z');
    const ownedSequence = cloneManualSequence(ownedRef, 'root');
    const ownedManual = ownedSequence[ownedSequence.length - 1];
    let ownedCommentId = 1;
    const ownedComments = [botComment(ownedCommentId++, encodeEventRef(ownedRef))];
    for (const record of ownedSequence) ownedComments.push(botComment(ownedCommentId++, encodeDeliveryRecord(record)));
    ownedComments.push(botComment(ownedCommentId++, encodeThreadState({
      version: 2, state: 'pending', repo: 'zettlab/demo', pr: 42,
      generation: ownedManual.threadGeneration, claimId: ownedManual.threadClaimId, messageId: null,
    })));
    const ownedGithub = makeGithub({ comments: ownedComments });
    const ownedCurrent = await readThreadState({
      github: ownedGithub, context: flowContext, core: makeCore(), prNum: 42,
    });
    const ownedBypass = await repairOrphanThread({
      github: ownedGithub, context: flowContext, core: makeCore(), prNum: 42, current: ownedCurrent,
      messageId: 'om_7777777777777777', release: false, runId: 'run-owned', operator: 'maintainer',
      privateKey: currentRepairPrivateKey, keyring: repairKeyring, keyId: 'current',
    });
    assert(!ownedBypass.ok && ownedBypass.ownerEventKey === ownedRef.eventKey &&
      !ownedComments.some((comment) => String(comment.body).includes(threadRepairMark)),
    '空 event_key 不得旁路修复 owned pending，错误返回 owner eventKey 要求 signed delivery repair');

    const orphanComments = [botComment(1, encodeThreadState({
      version: 2, state: 'pending', repo: 'zettlab/demo', pr: 42, generation: 9,
      claimId: '99999999999999999999999999999999', messageId: null,
    }))];
    const orphanGithub = makeGithub({ comments: orphanComments });
    const orphanCurrent = await readThreadState({ github: orphanGithub, context: flowContext, core: makeCore(), prNum: 42 });
    const orphanRepair = await repairOrphanThread({
      github: orphanGithub, context: flowContext, core: makeCore(), prNum: 42, current: orphanCurrent,
      messageId: null, release: true, runId: 'run-orphan', operator: 'maintainer',
      privateKey: currentRepairPrivateKey, keyring: repairKeyring, keyId: 'current',
    });
    const orphanAuditComment = orphanComments.find((comment) => String(comment.body).includes(threadRepairMark));
    const orphanAudit = orphanAuditComment && decodeThreadRepairRecord(orphanAuditComment.body, repairKeyring);
    assert(orphanRepair.ok && orphanAudit && orphanAudit.action === 'release' && orphanAudit.messageId === null &&
      orphanAudit.runId === 'run-orphan' && orphanAudit.operator === 'maintainer' && orphanAudit.keyId === 'current' &&
      /^[a-f0-9]{64}$/.test(orphanAudit.priorThreadHash),
    '仅 orphan pending 可修复，thread reducer 验签并绑定 prior snapshot/action/mid/run/operator/keyId');

    const dispatchCore = makeCore();
    const dispatchPr = await resolvePrNumber(
      flowGithub,
      context('repository_dispatch', { client_payload: { pr_number: '42' } }),
      dispatchCore,
    );
    assert(dispatchPr === 42 && dispatchCore.failures.length === 0, 'self-dispatch 只接受 canonical safe PR number');
    const badDispatchCore = makeCore();
    const badDispatch = await resolvePrNumber(
      flowGithub,
      context('repository_dispatch', { client_payload: { pr_number: '042' } }),
      badDispatchCore,
    );
    assert(badDispatch === null && badDispatchCore.failures.length === 1, '非 canonical dispatch payload fail closed');

    const mismatchComments = [];
    const mismatchChecks = new Map([['801', makeCheck(801, { headSha: 'b'.repeat(40) })]]);
    const mismatchGithub = makeGithub({ comments: mismatchComments, checks: mismatchChecks });
    await enqueueEventRef({ github: mismatchGithub, context: flowContext, core: makeCore(), ref: eventRef('801') });
    let mismatchSends = 0;
    global.fetch = async () => { mismatchSends += 1; return response({ code: 0 }); };
    const mismatchCore = makeCore();
    await drainQueuedCodexEvents({ github: mismatchGithub, context: flowContext, core: mismatchCore });
    assert(mismatchCore.failures.length > 0 && mismatchSends === 0 && mismatchCore.outputs.needs_dispatch === 'false' &&
      mismatchCore.outputs.continuation_mode !== 'backlog',
    `worker 按 immutable ID 重取并拒绝不一致；retry=false 永久错误不报告 backlog: ${JSON.stringify({ failures: mismatchCore.failures, sends: mismatchSends, outputs: mismatchCore.outputs })}`);

    const deadComments = [];
    const deadChecks = new Map([['902', makeCheck(902, { important: false })]]);
    const deadGithub = makeGithub({ comments: deadComments, checks: deadChecks });
    const deadRef = eventRef('901');
    const nextRef = eventRef('902');
    await enqueueEventRef({ github: deadGithub, context: flowContext, core: makeCore(), ref: deadRef });
    await enqueueEventRef({ github: deadGithub, context: flowContext, core: makeCore(), ref: nextRef });
    deadComments.push(botComment(1002, encodeDeliveryRecord({
      version: 1, state: 'failed', repo: 'zettlab/demo', pr: 42, eventKey: deadRef.eventKey,
      attempt: 0, mode: 'none', targetRoot: null, threadGeneration: null, threadClaimId: null,
      token: deliveryToken(deadRef.eventKey), sentAt: null, messageId: null, historyAttempts: 0,
      nextCheckAt: null, reason: 'definitely_not_sent', candidateMessageIds: [],
    })));
    let deadSends = 0;
    global.fetch = async () => { deadSends += 1; return response({ code: 0 }); };
    const deadReadsBefore = deadGithub.commentReads;
    await drainQueuedCodexEvents({ github: deadGithub, context: flowContext, core: makeCore() });
    assert(deadGithub.commentReads - deadReadsBefore === 1, '单批 drain 只读取一次 queue/thread/delivery comment snapshot');
    const deadLedger = await readDeliveryLedger({ github: deadGithub, context: flowContext, core: makeCore(), prNum: 42 });
    assert(deadLedger.latest.get(deadRef.eventKey).state === 'skipped' &&
      deadLedger.latest.get(nextRef.eventKey).state === 'skipped' && deadSends === 0,
    'failed 明确 dead-letter 为 skipped，不阻塞后续 FIFO');

    const cursorComments = [];
    const cursorChecks = new Map();
    let cursorCommentId = 1;
    const terminalFailed = decodeDeliveryRecord(deadComments.find((comment) => {
      const record = decodeDeliveryRecord(comment.body);
      return record && record.eventKey === deadRef.eventKey && record.state === 'failed';
    }).body);
    const terminalSkipped = deadLedger.latest.get(deadRef.eventKey);
    const cursorRefs = Array.from({ length: 60 }, (_value, index) =>
      eventRef(String(2000 + index), new Date(Date.parse(CREATED) + index * 1000).toISOString()));
    for (const cursorRef of cursorRefs) cursorComments.push(botComment(cursorCommentId++, encodeEventRef(cursorRef)));
    for (const cursorRef of cursorRefs.slice(0, 59)) {
      cursorComments.push(botComment(cursorCommentId++, encodeDeliveryRecord({
        ...terminalFailed, eventKey: cursorRef.eventKey, token: deliveryToken(cursorRef.eventKey),
      })));
      cursorComments.push(botComment(cursorCommentId++, encodeDeliveryRecord({
        ...terminalSkipped, eventKey: cursorRef.eventKey, token: deliveryToken(cursorRef.eventKey),
      })));
    }
    const cursorLast = cursorRefs[cursorRefs.length - 1];
    cursorChecks.set(cursorLast.eventId, makeCheck(Number(cursorLast.eventId), {
      important: false, createdAt: cursorLast.createdAt,
    }));
    const cursorGithub = makeGithub({ comments: cursorComments, checks: cursorChecks });
    const cursorFirstCore = makeCore();
    await drainQueuedCodexEvents({ github: cursorGithub, context: flowContext, core: cursorFirstCore });
    assert(cursorGithub.commentReads === 1 && cursorFirstCore.outputs.needs_dispatch === 'true' &&
      cursorFirstCore.outputs.continuation_mode === 'backlog',
    '单批最多扫描 50 个 terminal/manual，持久化 cursor 并请求 fresh backlog batch');
    const cursorSecondCore = makeCore();
    await drainQueuedCodexEvents({ github: cursorGithub, context: flowContext, core: cursorSecondCore });
    const cursorLedger = await readDeliveryLedger({ github: cursorGithub, context: flowContext, core: makeCore(), prNum: 42 });
    assert(cursorGithub.commentReads === 3 && cursorLedger.latest.get(cursorLast.eventKey).state === 'skipped',
      '第二批从 durable cursor 继续，越过 50 条头部 terminal，队尾事件不饥饿');
    assert(cursorComments.filter((comment) => String(comment.body).includes(checkpointMark)).length >= 2,
      '超过单页的账本由 tail-located append-only checkpoint revisions 持续压缩');
    const tailRef = eventRef('2999', new Date(Date.parse(CREATED) - 120000).toISOString());
    cursorChecks.set(tailRef.eventId, makeCheck(Number(tailRef.eventId), {
      important: false, createdAt: tailRef.createdAt,
    }));
    await enqueueEventRef({ github: cursorGithub, context: flowContext, core: makeCore(), ref: tailRef });
    await drainQueuedCodexEvents({ github: cursorGithub, context: flowContext, core: makeCore() });
    const tailLedger = await readDeliveryLedger({ github: cursorGithub, context: flowContext, core: makeCore(), prNum: 42 });
    assert(tailLedger.latest.get(tailRef.eventKey).state === 'skipped',
      'cursor 按 enqueue comment ID 排空后保持 tail；createdAt 更旧但 comment ID 更高的新事件仍会消费');

    const manyPulls = Array.from({ length: 400 }, (_value, index) => ({ number: index + 1 }));
    const slotTime = 1000 * 30 * 60 * 1000;
    const windowA = watchdogWindow(manyPulls, 'zettlab/demo', slotTime);
    const windowSame = watchdogWindow(manyPulls, 'zettlab/demo', slotTime + 1000);
    const windowRotated = watchdogWindow(manyPulls, 'zettlab/demo', slotTime + WATCHDOG_SHARDS * 30 * 60 * 1000);
    assert(windowA.shard === windowSame.shard && JSON.stringify(windowA.pulls) === JSON.stringify(windowSame.pulls),
      'watchdog 同一 time-slot 稳定分片');
    assert(windowA.pulls.length <= WATCHDOG_MAX_PRS_PER_RUN && WATCHDOG_MAX_DISPATCHES === 10 &&
      JSON.stringify(windowA.pulls) !== JSON.stringify(windowRotated.pulls),
    'watchdog 轮转窗口且显式限制 PR/dispatch cap，避免固定头部饥饿');
    for (const size of [40, 20, 32]) {
      const sameShardPulls = [];
      for (let number = 1; sameShardPulls.length < size; number += 1) {
        const candidate = { number };
        if (watchdogWindow([candidate], 'zettlab/demo', slotTime).pulls.length === 1) sameShardPulls.push(candidate);
      }
      const before = watchdogWindow(sameShardPulls, 'zettlab/demo', slotTime).pulls;
      const after = watchdogWindow(sameShardPulls, 'zettlab/demo', slotTime + WATCHDOG_SHARDS * 30 * 60 * 1000).pulls;
      assert(JSON.stringify(before) !== JSON.stringify(after),
        `watchdog 同 shard 每周期前移一位，${size} 条时也不会因 cap 公因数永久饥饿`);
    }
    const manualOnlyRef = eventRef('631', '2026-08-05T12:03:01Z');
    let manualOnlyId = 1;
    const manualOnlyComments = [botComment(manualOnlyId++, encodeEventRef(manualOnlyRef))];
    for (const record of cloneManualSequence(manualOnlyRef, 'reply')) {
      manualOnlyComments.push(botComment(manualOnlyId++, encodeDeliveryRecord(record)));
    }
    const manualOnlyGithub = makeGithub({ comments: manualOnlyComments, openPulls: [{ number: 42 }] });
    let manualSlot = 0;
    while (watchdogWindow([{ number: 42 }], 'zettlab/demo', manualSlot).pulls.length === 0) {
      manualSlot += 30 * 60 * 1000;
    }
    await sweepQueuedCodexReviews({ github: manualOnlyGithub, context: flowContext, core: makeCore(), nowMs: manualSlot });
    assert(manualOnlyGithub.dispatches.length === 0, 'watchdog 排除仅剩 manual 的 PR，避免无效热循环');
    const barrierWatchRef = eventRef('641', '2026-08-05T12:04:01Z');
    const barrierLaterRef = eventRef('642', '2026-08-05T12:04:02Z');
    const barrierWatchSequence = cloneManualSequence(barrierWatchRef, 'root');
    const barrierWatchManual = barrierWatchSequence[barrierWatchSequence.length - 1];
    let barrierWatchId = 1;
    const barrierWatchComments = [
      botComment(barrierWatchId++, encodeEventRef(barrierWatchRef)),
      botComment(barrierWatchId++, encodeEventRef(barrierLaterRef)),
    ];
    for (const record of barrierWatchSequence) {
      barrierWatchComments.push(botComment(barrierWatchId++, encodeDeliveryRecord(record)));
    }
    barrierWatchComments.push(botComment(barrierWatchId++, encodeThreadState({
      version: 2, state: 'pending', repo: 'zettlab/demo', pr: 42,
      generation: barrierWatchManual.threadGeneration, claimId: barrierWatchManual.threadClaimId, messageId: null,
    })));
    const barrierWatchGithub = makeGithub({ comments: barrierWatchComments, openPulls: [{ number: 42 }] });
    await sweepQueuedCodexReviews({ github: barrierWatchGithub, context: flowContext, core: makeCore(), nowMs: manualSlot });
    assert(barrierWatchGithub.dispatches.length === 0,
      'manual root + matching pending 是 watchdog barrier，即使队尾另有 actionable event 也只等 signed repair kick');

    const actorPayload = makeReview();
    assert(actorPayload.user.id === 199175422 && actorPayload.user.login === CODEX_REVIEW_LOGIN,
      'review actor fixture 绑定精确官方 ID/login');

    const trustedBody = encodeThreadState({
      version: 2, state: 'pending', repo: 'zettlab/demo', pr: 42, generation: 1,
      claimId: '0123456789abcdef0123456789abcdef', messageId: null,
    });
    assert(trustedBody.startsWith(STATE_MARK) && isTrustedMarkerComment(botComment(1, trustedBody)), 'v2 thread state 仅信任 github-actions bot');
    assert(!isTrustedMarkerComment({ body: trustedBody, author_association: 'OWNER', user: { login: 'owner', type: 'User' } }), 'OWNER 不能伪造 v2 state');
    const hostileBody = encodeThreadState({
      version: 2, state: 'final', repo: 'zettlab/demo', pr: 42, generation: 77,
      claimId: '77777777777777777777777777777777', messageId: 'om_7777777777777777',
    });
    const hostileComments = [{
      id: 1, body: hostileBody, created_at: '2026-08-05T12:10:00Z',
      author_association: 'OWNER', user: { login: 'owner', type: 'User' },
    }];
    const hostileGithub = makeGithub({ comments: hostileComments });
    await drainQueuedCodexEvents({ github: hostileGithub, context: flowContext, core: makeCore() });
    const hostileAfterCheckpoint = await readThreadState({
      github: hostileGithub, context: flowContext, core: makeCore(), prNum: 42,
    });
    assert(hostileAfterCheckpoint.ok && hostileAfterCheckpoint.kind === 'none',
      'checkpoint 保留原 author provenance，恶意 OWNER v2 marker 穿过 compaction 后仍不提升为 bot');
    const garbageComments = Array.from({ length: 241 }, (_value, index) => ({
      id: index + 1,
      body: encodeThreadState({
        version: 2, state: 'final', repo: 'zettlab/demo', pr: 42, generation: index + 1,
        claimId: String(index + 1).padStart(32, '0'), messageId: 'om_7777777777777777',
      }),
      created_at: new Date(Date.parse(CREATED) + index * 1000).toISOString(),
      author_association: 'OWNER', user: { login: `owner-${index}`, type: 'User' },
    }));
    const garbageGithub = makeGithub({ comments: garbageComments });
    const garbageCore = makeCore();
    await drainQueuedCodexEvents({ github: garbageGithub, context: flowContext, core: garbageCore });
    assert(garbageCore.outputs.needs_dispatch !== 'true' &&
      garbageComments.filter((comment) => String(comment.body).includes(checkpointMark)).length === 2,
    '241 个非可信 marker 完全不进入 compact state cap，不产生 overflow/backlog 热循环');

    const lostComments = [];
    const lostGithub = makeGithub({ comments: lostComments, checkpointCreateLosesResponse: true });
    const lostCore = makeCore();
    await drainQueuedCodexEvents({ github: lostGithub, context: flowContext, core: lostCore });
    assert(lostCore.failures.length === 0 &&
      lostComments.filter((comment) => String(comment.body).includes(checkpointMark)).length === 2,
    '首次 checkpoint POST 单次执行；响应丢失后 tail 按 canonical hash 恢复，随后仅追加下一 revision');

    const interleavedRef = eventRef('7099', '2026-08-05T12:19:59Z');
    const interleavedComments = Array.from({ length: 499 }, (_value, index) => ({
      id: index + 1, body: `ordinary-${index}`, created_at: new Date(Date.parse(CREATED) + index * 1000).toISOString(),
      author_association: 'NONE', user: { login: 'user', type: 'User' },
    }));
    const interleavedGithub = makeGithub({
      comments: interleavedComments,
      checks: new Map([['7099', makeCheck(7099, { important: false, createdAt: interleavedRef.createdAt })]]),
      beforeCheckpointCreate: (items) => items.push({
        ...botComment(500, encodeEventRef(interleavedRef)), created_at: '2026-08-05T12:19:59Z',
      }),
    });
    await drainQueuedCodexEvents({ github: interleavedGithub, context: flowContext, core: makeCore() });
    await drainQueuedCodexEvents({ github: interleavedGithub, context: flowContext, core: makeCore() });
    const interleavedLedger = await readDeliveryLedger({
      github: interleavedGithub, context: flowContext, core: makeCore(), prNum: 42,
    });
    const interleavedState = interleavedLedger.latest.get(interleavedRef.eventKey);
    assert(interleavedState && interleavedState.state === 'skipped' &&
      interleavedComments.findIndex((comment) => String(comment.body).includes(checkpointMark)) >
        interleavedComments.findIndex((comment) => decodeEventRef(comment.body)?.eventKey === interleavedRef.eventKey),
    '499-prefix 后并发 enqueue 抢在 POST 前成为第500条，append-only checkpoint 位于501仍可从 tail 定位');

    const sameSecond = '2026-08-05T12:20:00Z';
    const sameFirst = eventRef('7101', sameSecond);
    const sameSecondRef = eventRef('7102', sameSecond);
    const sameComments = [botComment(1, encodeEventRef(sameFirst))];
    sameComments[0].created_at = sameSecond;
    const sameChecks = new Map([
      ['7101', makeCheck(7101, { important: false, createdAt: sameSecond })],
      ['7102', makeCheck(7102, { important: false, createdAt: sameSecond })],
    ]);
    const sameGithub = makeGithub({ comments: sameComments, checks: sameChecks });
    const sameFirstCore = makeCore();
    await drainQueuedCodexEvents({ github: sameGithub, context: flowContext, core: sameFirstCore });
    sameComments.push({ ...botComment(9000, encodeEventRef(sameSecondRef)), created_at: sameSecond });
    const sameSecondCore = makeCore();
    await drainQueuedCodexEvents({ github: sameGithub, context: flowContext, core: sameSecondCore });
    const sameLedger = await readDeliveryLedger({ github: sameGithub, context: flowContext, core: makeCore(), prNum: 42 });
    const sameSecondState = sameLedger.latest.get(sameSecondRef.eventKey);
    assert(sameSecondState && sameSecondState.state === 'skipped' && sameGithub.checkReads.get('7102') === 1,
      `incremental since 回退1秒并以 id>high 去重，同秒高ID enqueue 不漏且只处理一次: ${JSON.stringify({ first: sameFirstCore.failures, second: sameSecondCore.failures, outputs: sameSecondCore.outputs, reads: [...sameGithub.checkReads] })}`);

    const tombstoneComments = [];
    const completedRef = eventRef('7201', '2026-08-05T12:21:00Z');
    const afterCompletedRef = eventRef('7202', '2026-08-05T12:21:01Z');
    const tombstoneChecks = new Map([
      ['7201', makeCheck(7201, { important: false, createdAt: completedRef.createdAt })],
      ['7202', makeCheck(7202, { important: false, createdAt: afterCompletedRef.createdAt })],
    ]);
    const tombstoneGithub = makeGithub({ comments: tombstoneComments, checks: tombstoneChecks });
    await enqueueEventRef({ github: tombstoneGithub, context: flowContext, core: makeCore(), ref: completedRef });
    await drainQueuedCodexEvents({ github: tombstoneGithub, context: flowContext, core: makeCore() });
    await enqueueEventRef({ github: tombstoneGithub, context: flowContext, core: makeCore(), ref: afterCompletedRef });
    await enqueueEventRef({ github: tombstoneGithub, context: flowContext, core: makeCore(), ref: completedRef });
    await drainQueuedCodexEvents({ github: tombstoneGithub, context: flowContext, core: makeCore() });
    const tombstoneLedger = await readDeliveryLedger({
      github: tombstoneGithub, context: flowContext, core: makeCore(), prNum: 42,
    });
    assert(tombstoneLedger.latest.get(completedRef.eventKey).state === 'skipped' &&
      tombstoneLedger.latest.get(afterCompletedRef.eventKey).state === 'skipped' &&
      tombstoneGithub.checkReads.get('7201') === 1 && tombstoneGithub.checkReads.get('7202') === 1,
    '已消费 terminal event 保留最小幂等 tombstone；新 enqueue 与同 GitHub event redelivery 不会重复 hydrate/send');

    const bulkComments = [];
    const bulkChecks = new Map();
    const bulkRefs = Array.from({ length: 350 }, (_value, index) => {
      const createdAt = new Date(Date.parse(CREATED) + index * 1000).toISOString();
      const ref = eventRef(String(8000 + index), createdAt);
      const comment = botComment(index + 1, encodeEventRef(ref));
      comment.created_at = createdAt;
      bulkComments.push(comment);
      bulkChecks.set(ref.eventId, makeCheck(Number(ref.eventId), { important: false, createdAt }));
      return ref;
    });
    const bulkGithub = makeGithub({ comments: bulkComments, checks: bulkChecks });
    const bulkFailures = [];
    for (let batch = 0; batch < 120; batch += 1) {
      const bulkCore = makeCore();
      await drainQueuedCodexEvents({ github: bulkGithub, context: flowContext, core: bulkCore });
      bulkFailures.push(...bulkCore.failures);
    }
    const bulkLedger = await readDeliveryLedger({ github: bulkGithub, context: flowContext, core: makeCore(), prNum: 42 });
    const bulkTerminalCount = bulkRefs.filter((ref) => bulkLedger.latest.get(ref.eventKey) &&
      bulkLedger.latest.get(ref.eventKey).state === 'skipped').length;
    const bulkMaxReads = Math.max(0, ...bulkGithub.checkReads.values());
    assert(bulkLedger.latest.get(bulkRefs[bulkRefs.length - 1].eventKey) &&
      bulkLedger.latest.get(bulkRefs[bulkRefs.length - 1].eventKey).state === 'skipped' &&
      bulkRefs.every((ref) => bulkGithub.checkReads.get(ref.eventId) === 1),
    `350 refs 先确认 checkpoint ingest 再分批推进 cursor；连续批次不重复 hydrate/send 且最终全部 terminal: ${JSON.stringify({ bulkTerminalCount, bulkMaxReads, failures: bulkFailures.slice(0, 3), readCount: bulkGithub.checkReads.size })}`);
    assert(String(flowComments.map((comment) => comment.body).join('\n')).includes(EVENT_MARK) &&
      String(flowComments.map((comment) => comment.body).join('\n')).includes(DELIVERY_MARK), 'queue 与 delivery ledger 均已落 PR comment');
  } finally {
    global.fetch = oldFetch;
    process.env = oldEnv;
  }
}

main().catch((error) => {
  console.error(`FAIL: ${error.stack || error}`);
  process.exitCode = 1;
});
