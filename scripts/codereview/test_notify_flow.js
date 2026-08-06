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
  BOOTSTRAP_MARK,
  BOOTSTRAP_PROGRESS_RESERVE_MS,
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
  decodeBootstrapProgress,
  decodeCheckpoint,
  decodeRepairRecord,
  decodeThreadState,
  deliveryToken,
  drainQueuedCodexEvents,
  encodeDeliveryRecord,
  encodeDeliveryRepairRecord,
  encodeEventRef,
  encodeCheckpoint,
  encodeBootstrapProgress,
  encodeThreadState,
  enqueueEventRef,
  enqueueOfficialCodexEvent,
  notifyFromCheckRun,
  consumeCodexWorkflowRunWakeup,
  findDeliveryInFeishuHistory,
  isTrustedMarkerComment,
  parseRetryAfterMs,
  readDeliveryLedger,
  readEventQueue,
  readThreadMarkerComments,
  readThreadState,
  repairDelivery,
  repairLegacyConflict,
  resolveOfficialCodexEventRefs,
  resolveCodexWorkflowRunRefs,
  resolvePrNumber,
  scheduleQueuedCodexDrain,
  sweepQueuedCodexReviews,
  withGithubAttemptBudget,
  withGithubRetry,
  githubGraphqlWithTimeout,
  MAX_GITHUB_RETRY_DELAY_MS,
  FEISHU_RATE_LIMIT_MAX_ATTEMPTS,
  FEISHU_RATE_LIMIT_MAX_INLINE_DELAY_MS,
  FEISHU_RATE_LIMIT_MAX_RETRY_AFTER_MS,
  MAX_RESOLVED_EVENT_REFS,
  CODEX_CAPTURE_WORKFLOW_NAME,
  CODEX_CAPTURE_WORKFLOW_PATH,
  WATCHDOG_MAX_DISPATCHES,
  WATCHDOG_MAX_PRS_PER_RUN,
  WATCHDOG_SHARDS,
  WATCHDOG_SLOT_MS,
  decodeDeliveryRepairRecord,
  historyRecoveryExhausted,
  watchdogWindow,
} = require('./notify');

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function response(json, status = 200, headers = {}) {
  const normalizedHeaders = Object.fromEntries(
    Object.entries(headers).map(([name, value]) => [name.toLowerCase(), String(value)]),
  );
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: { get: (name) => normalizedHeaders[String(name).toLowerCase()] || null },
    text: async () => JSON.stringify(json),
  };
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
    commit_id: HEAD,
    html_url: 'https://github.com/zettlab/demo/pull/42#pullrequestreview-701',
    user: { id: Number(CODEX_REVIEW_USER_ID), login: CODEX_REVIEW_LOGIN, type: 'Bot' },
  };
}

function makeGithub({
  comments = [], checks = new Map(), reviews = new Map(), reviewComments = [], associated = null, openPulls = [],
  associationFailure = null,
  workflowRun = null, workflowPath = CODEX_CAPTURE_WORKFLOW_PATH,
  dispatchFails = false, callOrder = null,
  checkpointCreateLosesResponse = false,
  checkpointCreateLosesResponseAt = null,
  deliveryCreateLosesResponseState = null,
  threadReleaseFails = false,
  beforeCheckpointCreate = null,
  onListComments = null,
  onGraphql = null,
  onCreateComment = null,
  bootstrapCreateLosesResponse = false,
} = {}) {
  const listComments = async (options) => {
    assert(options.request.timeout === GITHUB_REQUEST_TIMEOUT_MS, 'comment list API timeout');
    assert(!Object.prototype.hasOwnProperty.call(options, 'sort') &&
      !Object.prototype.hasOwnProperty.call(options, 'direction'),
    'single-issue listComments 不发送未支持的 sort/direction 参数');
    if (onListComments) onListComments(options);
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
  let checkpointCreateCount = 0;
  let checkpointInterleaved = false;
  const github = {
    dispatches,
    pullListPages: [],
    prReads: [],
    commentReads: 0,
    checkReads: new Map(),
    reviewReads: [],
    associationReads: [],
    failThreadRelease: threadReleaseFails,
    graphql: async (_query, variables) => {
      assert(variables.request && variables.request.signal instanceof AbortSignal,
        'GraphQL locator 必须透传 AbortController signal');
      if (onGraphql) onGraphql(variables);
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
      actions: {
        getWorkflowRun: async (options) => ({ data: workflowRun || {
          id: options.run_id, workflow_id: 77, status: 'completed', conclusion: 'success',
          event: 'pull_request_review', display_title: 'codex-feishu-capture:review:42:701',
          repository: { full_name: 'zettlab/demo' },
        } }),
        getWorkflow: async (options) => ({ data: {
          id: options.workflow_id, name: CODEX_CAPTURE_WORKFLOW_NAME, path: workflowPath,
        } }),
        listJobsForWorkflowRun: async () => ({ data: { jobs: [{
          name: 'capture', status: 'completed', conclusion: 'success',
        }] } }),
      },
      repos: {
        listPullRequestsAssociatedWithCommit: async (options) => {
          assert(options.request.timeout === GITHUB_REQUEST_TIMEOUT_MS, 'associated API timeout');
          github.associationReads.push(options.commit_sha);
          if (associationFailure) {
            const error = new Error(associationFailure === 'timeout'
              ? 'simulated association timeout' : 'simulated association 503');
            error.status = associationFailure === 'timeout' ? 0 : 503;
            if (associationFailure === 'timeout') error.name = 'AbortError';
            error.response = { headers: { 'Retry-After': '0' } };
            throw error;
          }
          if (associated !== null) return { data: associated };
          const unique = new Map();
          for (const checkRun of checks.values()) {
            if (checkRun.head_sha !== options.commit_sha) continue;
            for (const pr of checkRun.pull_requests || []) unique.set(Number(pr.number), pr);
          }
          return { data: [...unique.values()] };
        },
        createDispatchEvent: async (options) => {
          assert(options.request.timeout === GITHUB_REQUEST_TIMEOUT_MS, 'dispatch API timeout');
          if (callOrder) callOrder.push('dispatch');
          if (dispatchFails) {
            const error = new Error('dispatch forbidden');
            error.status = 403;
            throw error;
          }
          dispatches.push(options.client_payload);
          return { data: {} };
        },
      },
      pulls: {
        list: async (options) => {
          assert(options.request.timeout === GITHUB_REQUEST_TIMEOUT_MS, 'pull list API timeout');
          github.pullListPages.push(Number(options.page || 1));
          const start = (Number(options.page || 1) - 1) * Number(options.per_page || openPulls.length || 1);
          return { data: openPulls.slice(start, start + Number(options.per_page || openPulls.length || 1)) };
        },
        get: async (options) => {
          github.prReads.push(options.pull_number);
          return { data: {
            number: options.pull_number,
            title: '测试 PR',
            html_url: `https://github.com/zettlab/demo/pull/${options.pull_number}`,
            user: { login: 'owner' },
            base: { ref: 'main' },
            head: { ref: 'feature/test', sha: HEAD },
            state: 'open',
          } };
        },
        getReview: async (options) => {
          github.reviewReads.push({ pullNumber: options.pull_number, reviewId: options.review_id });
          return { data: reviews.get(String(options.review_id)) };
        },
        listReviews: async (options) => {
          if (callOrder) callOrder.push(`reviews:${options.page || 1}`);
          const items = [...reviews.values()];
          const start = (Number(options.page || 1) - 1) * Number(options.per_page || 100);
          return { data: items.slice(start, start + Number(options.per_page || 100)) };
        },
        listReviewComments,
      },
      checks: {
        get: async (options) => {
          const key = String(options.check_run_id);
          github.checkReads.set(key, (github.checkReads.get(key) || 0) + 1);
          return { data: checks.get(key) };
        },
        listForRef: async (options) => {
          if (callOrder) callOrder.push(`checks:${options.page || 1}`);
          const items = [...checks.values()];
          const start = (Number(options.page || 1) - 1) * Number(options.per_page || 100);
          return { data: {
            total_count: items.length,
            check_runs: items.slice(start, start + Number(options.per_page || 100)),
          } };
        },
      },
      issues: {
        listComments,
        createComment: async (options) => {
          assert(options.request.timeout === GITHUB_REQUEST_TIMEOUT_MS, 'comment API timeout');
          if (onCreateComment) onCreateComment(options);
          const requestedThreadState = decodeThreadState(options.body);
          if (github.failThreadRelease && requestedThreadState && requestedThreadState.state === 'released') {
            throw new Error('simulated thread release failure');
          }
          if (!checkpointInterleaved && beforeCheckpointCreate &&
              String(options.body).includes('codex-review-feishu-checkpoint:')) {
            checkpointInterleaved = true;
            beforeCheckpointCreate(comments, options.body);
          }
          const nextCommentId = comments.reduce((max, item) => Math.max(max, Number(item.id) || 0), 999) + 1;
          const comment = botComment(nextCommentId, options.body);
          const latestCreatedAt = comments.reduce((max, item) =>
            item.created_at ? Math.max(max, Date.parse(item.created_at)) : max, Date.parse(CREATED));
          comment.created_at = new Date(latestCreatedAt + 1000).toISOString();
          comments.push(comment);
          if (bootstrapCreateLosesResponse && String(options.body).includes(BOOTSTRAP_MARK)) {
            throw new Error('simulated lost bootstrap progress response');
          }
          const requestedDelivery = decodeDeliveryRecord(options.body);
          if (deliveryCreateLosesResponseState && requestedDelivery &&
              requestedDelivery.state === deliveryCreateLosesResponseState) {
            throw new Error(`simulated lost ${deliveryCreateLosesResponseState} response`);
          }
          if (String(options.body).includes('codex-review-feishu-checkpoint:')) checkpointCreateCount += 1;
          if (!checkpointResponseLost && String(options.body).includes('codex-review-feishu-checkpoint:') &&
              (checkpointCreateLosesResponse || checkpointCreateCount === checkpointCreateLosesResponseAt)) {
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
    assert(BOOTSTRAP_PROGRESS_RESERVE_MS > 2 * GITHUB_REQUEST_TIMEOUT_MS,
      'bootstrap 为 create + response-loss tail confirm 预留两个 GitHub 请求与 runner overhead');
    assert(DRAIN_BATCH_SIZE > 0 && DRAIN_BATCH_SIZE <= 10, 'drain batch 必须有小型上限');
    assert(HISTORY_PAGE_SIZE === 50 && HISTORY_MAX_PAGES > 0, 'Feishu history 有固定页数/消息数上限');
    assert(HISTORY_MAX_ATTEMPTS === 8 && HISTORY_MAX_AGE_MS === 24 * 60 * 60 * 1000,
      'history 恢复以 8 次或 24h 为硬上限');
    assert(DEFINITE_SEND_MAX_ATTEMPTS === 3, 'definitely-not-sent 最多发送三次');
    assert(FEISHU_RATE_LIMIT_MAX_ATTEMPTS === 3 && FEISHU_RATE_LIMIT_MAX_INLINE_DELAY_MS === 30000 &&
      FEISHU_RATE_LIMIT_MAX_RETRY_AFTER_MS === 60 * 60 * 1000,
    'Feishu 429 同请求最多3次、单轮最多等30s，服务端Retry-After持久化上限1h');
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
    const consumer = fs.readFileSync(path.join(__dirname, '../../.github/workflows/codex-review-feishu-consumer.yml'), 'utf8');
    assert(workflow.includes('name: Codex Review -> Feishu Capture') &&
      workflow.includes('run-name: codex-feishu-capture:') && workflow.includes('permissions: {}') &&
      !workflow.includes('actions/checkout') && !workflow.includes('actions/upload-artifact') &&
      !workflow.includes('issues: write') && !workflow.includes('secrets.') && !workflow.includes('environment:'),
    'source workflow 只产生严格 run-name wake-up，无 checkout/artifact/write/secrets/environment');
    assert(consumer.includes('workflow_run:') && consumer.includes('Codex Review -> Feishu Capture') &&
      consumer.includes('consumeCodexWorkflowRunWakeup') && consumer.includes('actions: read') &&
      consumer.includes('ref: ${{ github.event.repository.default_branch }}'),
    'consumer 仅在默认分支通过 workflow_run API 重验后写 durable queue');
    const consumeCaptureJob = consumer.slice(consumer.indexOf('  consume_capture:'), consumer.indexOf('  resolve_dispatch_pr:'));
    const watchdogJob = consumer.slice(consumer.indexOf('  queue_watchdog:'), consumer.indexOf('  normalize_repair:'));
    assert(consumeCaptureJob.includes('contents: write') && consumeCaptureJob.includes('issues: write') &&
      consumeCaptureJob.includes('checks: read'),
    'trusted consume_capture 具备 durable enqueue 与 repository_dispatch 的精确权限');
    assert(watchdogJob.includes('contents: write') && watchdogJob.includes('issues: write') &&
      watchdogJob.includes('checks: read') && watchdogJob.includes('pull-requests: read') &&
      watchdogJob.includes('CODEX_FEISHU_REPAIR_PUBLIC_KEYS: ${{ vars.CODEX_FEISHU_REPAIR_PUBLIC_KEYS }}') &&
      !watchdogJob.includes('CODEX_FEISHU_REPAIR_PRIVATE_KEY'),
    'watchdog 具备 source reconciliation/durable enqueue/dispatch 精确权限及仅公钥 repair 验签能力');
    assert(consumer.includes('repository_dispatch:') && consumer.includes('codex-review-feishu-drain'), 'consumer 接收 trusted self-dispatch');
    const reportIndex = consumer.indexOf('  report:');
    const reportJob = consumer.slice(reportIndex, consumer.indexOf('  schedule_continuation:'));
    assert(reportJob.includes("github.event_name == 'repository_dispatch'") &&
      reportJob.includes('environment: codex-review-feishu-production') &&
      reportJob.includes('ref: ${{ github.event.repository.default_branch }}') &&
      reportJob.includes('group: codex-review-feishu-${{ github.repository }}-pr-${{ needs.resolve_dispatch_pr.outputs.pr_number }}') &&
      reportJob.includes('cancel-in-progress: false') &&
      !reportJob.includes('github.event.pull_request.head.sha') &&
      !reportJob.includes('contents: write'), 'secret worker 仅默认分支 repository_dispatch、checkout default 且挂 protected environment');
    const repairJob = consumer.slice(consumer.indexOf('  repair_pending:'), consumer.indexOf('  kick_after_repair:'));
    assert(repairJob.includes('environment: codex-review-feishu-repair') &&
      repairJob.includes('ref: ${{ github.event.repository.default_branch }}') &&
      !repairJob.includes('github.event.pull_request.head.sha'),
    'repair secret job 明确 checkout default branch，绝不执行 PR head');
    assert(consumer.includes('  schedule_continuation:') && consumer.includes('fresh budget') &&
      consumer.includes('  queue_watchdog:'), 'remaining queue 用独立预算续批并由可信 schedule watchdog 扫描');
    assert(consumer.includes("needs.report.outputs.continuation_mode == 'backlog'") &&
      consumer.includes("cron: '*/5 * * * *'"),
    '只有 backlog 可 immediate dispatch；retry 最迟由 5min watchdog 到期');
    assert(consumer.includes('  kick_after_repair:') && consumer.includes('Kick a fresh default-branch consumer after repair'), 'repair 后恢复 fresh consumer');
    assert(consumer.includes('repairLegacyConflict') && consumer.includes('CODEX_FEISHU_REPAIR_PRIVATE_KEY') &&
      consumer.includes('CODEX_FEISHU_REPAIR_PUBLIC_KEYS') && consumer.includes('CODEX_FEISHU_REPAIR_KEY_ID'),
    'repair 使用 environment 私钥签名、keyId + public keyring 验签');
    assert(consumer.includes('repair_action:') && consumer.includes('select_mid') &&
      consumer.includes('retry') && consumer.includes('discard'),
    'workflow repair 显式选择 select_mid/retry/discard action');
    const repairJobIndex = consumer.indexOf('  repair_pending:');
    const repairAdminJob = consumer.slice(repairJobIndex, consumer.indexOf('  kick_after_repair:'));
    const adminGateIndex = repairAdminJob.indexOf('id: admin_gate');
    const repairSecretIndex = repairAdminJob.indexOf('CODEX_FEISHU_REPAIR_PRIVATE_KEY:');
    assert(repairJobIndex > 0 && adminGateIndex > 0 && repairSecretIndex > adminGateIndex &&
      repairAdminJob.includes('github.rest.repos.getCollaboratorPermissionLevel') &&
      repairAdminJob.includes("permission !== 'admin'") &&
      repairAdminJob.includes('username: actor'),
    'repair secret 前用 pinned github-script 精确验证 workflow actor 的 repository admin permission');
    assert(repairAdminJob.includes("github.event_name == 'workflow_dispatch'") &&
      repairAdminJob.includes('github.ref_name == github.event.repository.default_branch') &&
      repairAdminJob.includes('environment: codex-review-feishu-repair'),
    'repair job 自身仅允许 default-branch workflow_dispatch，并保留 repair environment branch policy');
    assert(repairAdminJob.includes('REPAIR_ACTOR: ${{ steps.admin_gate.outputs.actor }}') &&
      repairAdminJob.includes("createHash('sha256')") && repairAdminJob.includes('repairAuditBinding') &&
      repairJob.includes('runId: auditRunId, operator: actor') &&
      repairJob.includes('runId: auditRunId,'),
    '所有 signed repair 用 auditRunId 绑定 admin actor、run 与完整 workflow_dispatch inputs');
    assert(workflow.includes('github.event.review.user.id == 199175422') &&
      workflow.includes("github.event.review.user.login == 'chatgpt-codex-connector[bot]'") &&
      workflow.includes("github.event.check_run.app.slug == 'chatgpt-codex-connector'"), 'workflow actor 使用精确 ID/login/app slug');
    assert(!/uses:\s+actions\/[^@\s]+@v\d+\b/.test(`${workflow}\n${consumer}`), 'Actions 全部 pin 完整 SHA');

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
    assert(queue.ok && queue.events.length === 1 && queueComments.length === 1, '重复 delivery 的 immutable ref 折叠且不丢失');

    const eventPayload = {
      check_run: makeCheck(502),
    };
    const officialComments = [];
    const officialGithub = makeGithub({
      comments: officialComments,
      checks: new Map([[String(eventPayload.check_run.id), eventPayload.check_run]]),
    });
    await enqueueOfficialCodexEvent({
      github: officialGithub,
      context: context('check_run', eventPayload),
      core: makeCore(),
    });
    const officialQueue = await readEventQueue({ github: officialGithub, context: context(), core: makeCore(), prNum: 42 });
    assert(officialQueue.events.length === 1 && officialQueue.events[0].ref.eventId === '502', '官方事件只存 immutable ID/ref，不存 cls 或 PR body');

    const multiCheck = makeCheck(550);
    multiCheck.pull_requests = [{ number: 42 }, { number: 43 }];
    const multiGithub = makeGithub({
      checks: new Map([['550', multiCheck]]),
      associated: [{ number: 42 }, { number: 43 }, { number: 42 }, { number: 44 }],
    });
    const multiCore = makeCore();
    const multiRefs = await resolveOfficialCodexEventRefs(
      multiGithub, context('check_run', { check_run: multiCheck }), multiCore,
    );
    assert(MAX_RESOLVED_EVENT_REFS === 20 && multiCore.failures.length === 0 &&
      JSON.stringify(multiRefs.map((ref) => ref.pr)) === JSON.stringify([42, 43, 44]) &&
      new Set(multiRefs.map((ref) => ref.eventKey)).size === 3,
    'check_run 仅信任 commit association，去重后输出有序有界独立 refs');
    for (const ref of multiRefs) {
      await scheduleQueuedCodexDrain({ github: multiGithub, context: context(), core: makeCore(), ref });
    }
    assert(multiGithub.dispatches.length === 3 && multiGithub.dispatches.every((payload, index) =>
      payload.pr_number === String(multiRefs[index].pr) &&
      payload.event_ref.eventKey === multiRefs[index].eventKey),
    'resolver refs 即使不依赖 enqueue output 也逐 PR 原样闭环到 dispatch payload');

    const trustedDirectCheck = makeCheck(551, { headSha: 'c'.repeat(40) });
    const rawSpoofedCheck = {
      id: 551, name: 'attacker', status: 'completed', head_sha: 'd'.repeat(40),
      completed_at: CREATED, pull_requests: [{ number: 999 }],
      app: { slug: 'attacker' }, check_suite: { app: { slug: 'attacker' } },
    };
    const directSpoofComments = [];
    const directSpoofGithub = makeGithub({
      comments: directSpoofComments, checks: new Map([['551', trustedDirectCheck]]),
      associationFailure: '5xx',
    });
    const directSpoofContext = context('check_run', {
      check_run: rawSpoofedCheck, pull_request: { number: 999 },
    });
    const directSpoofCore = makeCore();
    const directSpoofRefs = await resolveOfficialCodexEventRefs(
      directSpoofGithub, directSpoofContext, directSpoofCore,
    );
    const directSpoofPr = await resolvePrNumber(directSpoofGithub, directSpoofContext, directSpoofCore);
    await enqueueOfficialCodexEvent({
      github: directSpoofGithub, context: directSpoofContext, core: directSpoofCore,
    });
    const directSpoofQueue = await readEventQueue({
      github: directSpoofGithub, context: context(), core: makeCore(), prNum: 42,
    });
    let legacyStrictPosts = 0;
    global.fetch = async (url) => {
      const value = String(url);
      if (value.endsWith('/auth/v3/tenant_access_token/internal')) {
        return response({ code: 0, tenant_access_token: 'tenant-token' });
      }
      if (value.includes('receive_id_type=chat_id')) {
        legacyStrictPosts += 1;
        return response({ code: 0, data: { message_id: 'om_3535353535353535' } });
      }
      throw new Error(`unexpected legacy strict check request ${value}`);
    };
    await notifyFromCheckRun({
      github: directSpoofGithub, context: directSpoofContext, core: directSpoofCore,
    });
    const legacyStrictLedger = await readDeliveryLedger({
      github: directSpoofGithub, context: context(), core: makeCore(), prNum: 42,
    });
    assert(directSpoofCore.failures.length === 0 && directSpoofPr === 42 &&
      directSpoofRefs.length === 1 && directSpoofRefs[0].pr === 42 &&
      directSpoofRefs[0].headSha === trustedDirectCheck.head_sha &&
      directSpoofQueue.events.length === 1 && directSpoofQueue.events[0].ref.pr === 42 &&
      directSpoofQueue.events[0].ref.headSha === trustedDirectCheck.head_sha &&
      directSpoofGithub.prReads.every((number) => number === 42) && legacyStrictPosts === 1 &&
      legacyStrictLedger.latest.get(directSpoofRefs[0].eventKey).state === 'done' &&
      directSpoofCore.warnings.some((message) => message.includes('降级使用')),
    'association 503 时 direct/legacy check 降级到 fresh embedded；raw 伪 head/identity/PR 仍无法入队或投递');

    const wakeReview = makeReview(801);
    wakeReview.commit_id = 'b'.repeat(40);
    const wakeRun = {
      id: 9001, workflow_id: 77, status: 'completed', conclusion: 'success',
      event: 'pull_request_review', display_title: 'codex-feishu-capture:review:42:801',
      repository: { full_name: 'zettlab/demo' },
    };
    const wakeComments = [];
    const wakeGithub = makeGithub({
      comments: wakeComments, reviews: new Map([['801', wakeReview]]), workflowRun: wakeRun,
    });
    const wakeContext = context('workflow_run', { workflow_run: { id: 9001, workflow_id: 77 } });
    const wakeRefs = await consumeCodexWorkflowRunWakeup({ github: wakeGithub, context: wakeContext, core: makeCore() });
    const wakeRef = decodeEventRef(wakeComments[0].body);
    const expectedWakeRef = canonicalEventRef({
      version: 1, eventType: 'pull_request_review', eventId: '801', repo: 'zettlab/demo', pr: 42,
      headSha: wakeReview.commit_id, createdAt: wakeReview.submitted_at,
    });
    assert(wakeRefs && wakeRefs.length === 1 && wakeGithub.dispatches.length === 1 &&
      wakeRef.eventKey === expectedWakeRef.eventKey && wakeRef.headSha === wakeReview.commit_id &&
      wakeGithub.reviewReads.length === 2 && wakeGithub.reviewReads.every((read) => read.pullNumber === 42),
    'workflow_run hint 无 head；scoped API 重验 hint PR 的 official review 后按 fetched commit/id 构造 immutable ref，current head 前进仍入队');
    const forbiddenComments = [];
    const forbiddenGithub = makeGithub({
      comments: forbiddenComments, reviews: new Map([['801', wakeReview]]), workflowRun: wakeRun,
      dispatchFails: true,
    });
    const forbiddenCore = makeCore();
    const forbiddenRefs = await consumeCodexWorkflowRunWakeup({
      github: forbiddenGithub, context: wakeContext, core: forbiddenCore,
    });
    assert(forbiddenRefs === null && forbiddenGithub.dispatches.length === 0 &&
      forbiddenComments.some((comment) => decodeEventRef(comment.body)?.eventId === '801') &&
      forbiddenCore.failures.some((message) => message.includes('调度下一批')),
    'dispatch 403 fail closed 且不回滚已确认 durable ref；权限错误不会伪报 continuation 成功');
    const wrongPathGithub = makeGithub({
      comments: [], reviews: new Map([['801', wakeReview]]), workflowRun: wakeRun,
      workflowPath: '.github/workflows/attacker.yml',
    });
    const wrongPathCore = makeCore();
    const wrongPathRefs = await resolveCodexWorkflowRunRefs({
      github: wrongPathGithub, context: wakeContext, core: wrongPathCore,
    });
    assert(wrongPathRefs === null && wrongPathCore.failures.some((message) => message.includes('id/name/path')),
      'workflow_run name 不构成信任；workflow_id 必须经 API 映射到固定 capture path');

    const oldCheckSha = 'c'.repeat(40);
    const oldCheck = makeCheck(803, { headSha: oldCheckSha });
    oldCheck.pull_requests = [{ number: 42 }];
    const oldCheckRun = {
      id: 9002, workflow_id: 77, status: 'completed', conclusion: 'success',
      event: 'check_run', display_title: 'codex-feishu-capture:check:0:803',
      repository: { full_name: 'zettlab/demo' },
    };
    const oldCheckComments = [];
    const oldCheckGithub = makeGithub({
      comments: oldCheckComments, checks: new Map([['803', oldCheck]]),
      associationFailure: 'timeout', workflowRun: oldCheckRun,
    });
    const oldCheckContext = context('workflow_run', { workflow_run: { id: 9002, workflow_id: 77 } });
    const oldCheckCore = makeCore();
    const oldCheckRefs = await consumeCodexWorkflowRunWakeup({
      github: oldCheckGithub, context: oldCheckContext, core: oldCheckCore,
    });
    assert(oldCheckRefs && oldCheckRefs.length === 1 && oldCheckRefs[0].pr === 42 &&
      oldCheckRefs[0].headSha === oldCheckSha && oldCheckGithub.dispatches.length === 1 &&
      oldCheckGithub.associationReads.length === 6 && oldCheckCore.failures.length === 0 &&
      oldCheckCore.warnings.some((message) => message.includes('降级使用')) &&
      oldCheckGithub.associationReads.every((sha) => sha === oldCheckSha),
    'association timeout 三次耗尽时，workflow source/rehydrate 均降级到 fresh embedded，旧 head 仍入队并 dispatch');
    const spoofedCheck = makeCheck(804, { headSha: oldCheckSha });
    spoofedCheck.pull_requests = [];
    const spoofedCheckCore = makeCore();
    const spoofedCheckRefs = await resolveCodexWorkflowRunRefs({
      github: makeGithub({
        checks: new Map([['804', spoofedCheck]]), associated: [],
        workflowRun: { ...oldCheckRun, id: 9003, display_title: 'codex-feishu-capture:check:0:804' },
      }),
      context: context('workflow_run', { workflow_run: { id: 9003, workflow_id: 77 } }),
      core: spoofedCheckCore,
    });
    assert(spoofedCheckRefs && spoofedCheckRefs.length === 0 && spoofedCheckCore.failures.length === 0,
      'check_run embedded pull_requests 不构成关联证明，commit association API 无结果时不可伪造入队');

    const unavailableCheck = makeCheck(805, { headSha: oldCheckSha });
    unavailableCheck.pull_requests = [];
    const unavailableRun = {
      ...oldCheckRun, id: 9004, display_title: 'codex-feishu-capture:check:0:805',
    };
    const unavailableComments = [];
    const unavailableGithub = makeGithub({
      comments: unavailableComments, checks: new Map([['805', unavailableCheck]]),
      associationFailure: 'timeout', workflowRun: unavailableRun,
    });
    const unavailableCore = makeCore();
    const unavailableRefs = await consumeCodexWorkflowRunWakeup({
      github: unavailableGithub,
      context: context('workflow_run', { workflow_run: { id: 9004, workflow_id: 77 } }),
      core: unavailableCore,
    });
    assert(unavailableRefs === null && unavailableComments.length === 0 &&
      unavailableGithub.dispatches.length === 0 && unavailableGithub.associationReads.length === 3 &&
      unavailableCore.failures.some((message) => message.includes('fresh embedded PR 为空') && message.includes('retry')),
    'association timeout 且 fresh embedded 为空时显式 retry failure，不静默成功或丢事件');

    const oldReview = makeReview(802);
    oldReview.commit_id = 'b'.repeat(40);
    const oldReviewRef = canonicalEventRef({
      version: 1, eventType: 'pull_request_review', eventId: '802', repo: 'zettlab/demo', pr: 42,
      headSha: oldReview.commit_id, createdAt: CREATED,
    });
    const oldReviewComments = [];
    const oldReviewGithub = makeGithub({ comments: oldReviewComments, reviews: new Map([['802', oldReview]]) });
    const oldReviewCore = makeCore();
    process.env.RESOLVED_EVENT_REFS = JSON.stringify([oldReviewRef]);
    await enqueueOfficialCodexEvent({
      github: oldReviewGithub, context: context('repository_dispatch', { client_payload: { pr_number: '42' } }),
      core: oldReviewCore,
    });
    delete process.env.RESOLVED_EVENT_REFS;
    assert(oldReviewCore.failures.length === 0 &&
      oldReviewComments.some((comment) => decodeEventRef(comment.body)?.eventKey === oldReviewRef.eventKey),
    '旧 review rehydrate 验 fetched review.commit_id===immutable ref，PR current head 前进不丢事件');
    const tamperedOldReviewCore = makeCore();
    process.env.RESOLVED_EVENT_REFS = JSON.stringify([canonicalEventRef({
      version: oldReviewRef.version, eventType: oldReviewRef.eventType, eventId: oldReviewRef.eventId,
      repo: oldReviewRef.repo, pr: oldReviewRef.pr, headSha: HEAD, createdAt: oldReviewRef.createdAt,
    })]);
    await enqueueOfficialCodexEvent({
      github: oldReviewGithub, context: context('repository_dispatch', { client_payload: { pr_number: '42' } }),
      core: tamperedOldReviewCore,
    });
    delete process.env.RESOLVED_EVENT_REFS;
    assert(tamperedOldReviewCore.failures.some((message) => message.includes('绑定不一致')),
      '旧 review immutable ref 的 commit 被篡改时仍 fail closed');

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
    const flowGithub = makeGithub({
      comments: flowComments, checks, openPulls: [{ number: 42 }], deliveryCreateLosesResponseState: 'sending',
    });
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
    assert(rootPosts === 1 && firstCore.failures.length > 0,
      'sending comment create response loss 经 fresh ledger exact 确认后只 POST 一次；ambiguous send 持久化后显式失败');
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
      `5xx/disconnect 进入 durable uncertain: ${JSON.stringify(firstDeliveryCore.failures)}`);
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
    const discardChecks = new Map([[
      discardRef.eventId, makeCheck(Number(discardRef.eventId), { createdAt: discardRef.createdAt }),
    ]]);
    const discardGithub = makeGithub({
      comments: discardComments, checks: discardChecks, openPulls: [{ number: 42 }],
      deliveryCreateLosesResponseState: 'skipped',
    });
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
    const duplicateDiscardRepairs = discardComments.filter((comment) =>
      String(comment.body).includes(DELIVERY_REPAIR_MARK));
    assert(discarded.ok && duplicateDiscardRepairs.length === 3,
      'signed repair POST 响应丢失后，完全相同 canonical repair 重试可重读确认且幂等归约');
    const conflictingRepairBody = `${encodeDeliveryRecord(discardedLedger.latest.get(discardRef.eventKey))}\n${encodeDeliveryRepairRecord({
      ...discardRepair, runId: 'run-conflicting-repair',
    }, currentRepairPrivateKey)}`;
    const conflictingRepairCore = makeCore();
    const conflictingRepairLedger = await readDeliveryLedger({
      github: makeGithub({ comments: [
        ...discardComments,
        botComment(99999, conflictingRepairBody),
      ] }),
      context: flowContext, core: conflictingRepairCore, prNum: 42,
    });
    assert(!conflictingRepairLedger.ok && conflictingRepairCore.failures.some((message) =>
      message.includes('delivery 状态转换冲突')),
    '同 terminal record 绑定不同 signed repair audit 内容仍冲突，不能借幂等吞掉分叉');
    const signedDiscardComments = discardComments.map((comment) => ({ ...comment }));
    await drainQueuedCodexEvents({ github: discardGithub, context: flowContext, core: makeCore() });

    const historyRetryRef = eventRef('624', '2026-08-05T12:02:04Z');
    const historyRetrySequence = cloneManualSequence(historyRetryRef, 'root').map((record, index, records) =>
      index === records.length - 1 ? { ...record, reason: 'history_exhausted', candidateMessageIds: [] } : record);
    const historyRetryManual = historyRetrySequence[historyRetrySequence.length - 1];
    let historyRetryCommentId = 1;
    const historyRetryComments = [botComment(historyRetryCommentId++, encodeEventRef(historyRetryRef))];
    for (const record of historyRetrySequence) {
      historyRetryComments.push(botComment(historyRetryCommentId++, encodeDeliveryRecord(record)));
    }
    historyRetryComments.push(botComment(historyRetryCommentId++, encodeThreadState({
      version: 2, state: 'pending', repo: 'zettlab/demo', pr: 42,
      generation: historyRetryManual.threadGeneration, claimId: historyRetryManual.threadClaimId, messageId: null,
    })));
    const historyRetryGithub = makeGithub({
      comments: historyRetryComments,
      checks: new Map([[historyRetryRef.eventId, makeCheck(Number(historyRetryRef.eventId), {
        createdAt: historyRetryRef.createdAt,
      })]]),
    });
    const confirmedNotSent = await repairDelivery({
      github: historyRetryGithub, context: flowContext, core: makeCore(), prNum: 42,
      eventKey: historyRetryRef.eventKey, messageId: null, action: 'retry',
      runId: 'run-history-confirmed-not-sent', operator: 'maintainer',
      privateKey: currentRepairPrivateKey, keyring: repairKeyring, keyId: 'current',
    });
    const releasedHistoryThread = await readThreadState({
      github: historyRetryGithub, context: flowContext, core: makeCore(), prNum: 42,
    });
    let historyRetryPosts = 0;
    global.fetch = async (url) => {
      const value = String(url);
      if (value.endsWith('/auth/v3/tenant_access_token/internal')) {
        return response({ code: 0, tenant_access_token: 'tenant-token' });
      }
      if (value.includes('receive_id_type=chat_id')) {
        historyRetryPosts += 1;
        return response({ code: 0, data: { message_id: 'om_2424242424242424' } });
      }
      throw new Error(`unexpected history retry request ${value}`);
    };
    await drainQueuedCodexEvents({ github: historyRetryGithub, context: flowContext, core: makeCore() });
    const historyRetryLedger = await readDeliveryLedger({
      github: historyRetryGithub, context: flowContext, core: makeCore(), prNum: 42,
    });
    assert(confirmedNotSent.ok && releasedHistoryThread.kind === 'released' && historyRetryPosts === 1 &&
      historyRetryLedger.latest.get(historyRetryRef.eventKey).state === 'done',
    'history_exhausted 严格0候选时经 signed retry 确认未发送，释放 root barrier 并让同 event 安全重发完成');

    for (const [index, candidates] of [
      ['one', ['om_2727272727272727']],
      ['multi', ['om_2828282828282828', 'om_2929292929292929']],
    ]) {
      const candidateRef = eventRef(index === 'one' ? '626' : '627',
        index === 'one' ? '2026-08-05T12:02:06Z' : '2026-08-05T12:02:07Z');
      const candidateSequence = cloneManualSequence(candidateRef, 'reply').map((record, recordIndex, records) =>
        recordIndex === records.length - 1
          ? { ...record, reason: 'history_exhausted', candidateMessageIds: candidates }
          : record);
      let candidateCommentId = 1;
      const candidateComments = [botComment(candidateCommentId++, encodeEventRef(candidateRef))];
      for (const record of candidateSequence) {
        candidateComments.push(botComment(candidateCommentId++, encodeDeliveryRecord(record)));
      }
      const candidateGithub = makeGithub({ comments: candidateComments });
      const rejectedCandidateRetry = await repairDelivery({
        github: candidateGithub, context: flowContext, core: makeCore(), prNum: 42,
        eventKey: candidateRef.eventKey, messageId: null, action: 'retry',
        runId: `run-history-${index}-retry`, operator: 'maintainer',
        privateKey: currentRepairPrivateKey, keyring: repairKeyring, keyId: 'current',
      });
      const selectedCandidate = await repairDelivery({
        github: candidateGithub, context: flowContext, core: makeCore(), prNum: 42,
        eventKey: candidateRef.eventKey, messageId: candidates[0], action: 'select_mid',
        runId: `run-history-${index}-select`, operator: 'maintainer',
        privateKey: currentRepairPrivateKey, keyring: repairKeyring, keyId: 'current',
      });
      const candidateLedger = await readDeliveryLedger({
        github: candidateGithub, context: flowContext, core: makeCore(), prNum: 42,
      });
      assert(!rejectedCandidateRetry.ok && selectedCandidate.ok &&
        candidateLedger.latest.get(candidateRef.eventKey).state === 'done',
      `history_exhausted ${index} exact-token 候选禁止 retry 重发，只允许 select_mid 收敛`);
    }

    const replyRetryRef = eventRef('625', '2026-08-05T12:02:05Z');
    const replyRetryComments = [
      botComment(1, encodeEventRef(replyRetryRef)),
      botComment(2, encodeThreadState({
        version: 2, state: 'final', repo: 'zettlab/demo', pr: 42, generation: 1,
        claimId: '25252525252525252525252525252525', messageId: 'om_2525252525252525',
      })),
    ];
    const replyRetryGithub = makeGithub({
      comments: replyRetryComments,
      checks: new Map([[replyRetryRef.eventId, makeCheck(Number(replyRetryRef.eventId), {
        createdAt: replyRetryRef.createdAt,
      })]]),
      openPulls: [{ number: 42 }],
    });
    let replyRetryPosts = 0;
    global.fetch = async (url) => {
      const value = String(url);
      if (value.endsWith('/auth/v3/tenant_access_token/internal')) {
        return response({ code: 0, tenant_access_token: 'tenant-token' });
      }
      if (value.endsWith('/reply')) {
        replyRetryPosts += 1;
        if (replyRetryPosts === 1) return response({ code: 230001 }, 400);
        return response({ code: 0, data: { message_id: 'om_2626262626262626' } });
      }
      throw new Error(`unexpected reply retry request ${value}`);
    };
    await drainQueuedCodexEvents({ github: replyRetryGithub, context: flowContext, core: makeCore() });
    const replyNotSentLedger = await readDeliveryLedger({
      github: replyRetryGithub, context: flowContext, core: makeCore(), prNum: 42,
    });
    await drainQueuedCodexEvents({ github: replyRetryGithub, context: flowContext, core: makeCore() });
    const replyDoneLedger = await readDeliveryLedger({
      github: replyRetryGithub, context: flowContext, core: makeCore(), prNum: 42,
    });
    assert(replyNotSentLedger.latest.get(replyRetryRef.eventKey).state === 'not_sent' &&
      replyDoneLedger.latest.get(replyRetryRef.eventKey).state === 'done' && replyRetryPosts === 2,
    '已有 root 的 reply 明确未发送 400 保持 retryable，同 root 有界重试成功而不落 failed/skipped');

    const rootMissingRef = eventRef('628', '2026-08-05T12:02:08Z');
    const rootMissingComments = [
      botComment(1, encodeEventRef(rootMissingRef)),
      botComment(2, encodeThreadState({
        version: 2, state: 'final', repo: 'zettlab/demo', pr: 42, generation: 1,
        claimId: '28282828282828282828282828282828', messageId: 'om_2828282828282828',
      })),
    ];
    const rootMissingGithub = makeGithub({
      comments: rootMissingComments,
      checks: new Map([[rootMissingRef.eventId, makeCheck(Number(rootMissingRef.eventId), {
        createdAt: rootMissingRef.createdAt,
      })]]),
    });
    let missingReplyPosts = 0;
    let replacementRootPosts = 0;
    global.fetch = async (url) => {
      const value = String(url);
      if (value.endsWith('/auth/v3/tenant_access_token/internal')) {
        return response({ code: 0, tenant_access_token: 'tenant-token' });
      }
      if (value.endsWith('/reply')) {
        missingReplyPosts += 1;
        return response({ code: 230011 }, 400);
      }
      if (value.includes('receive_id_type=chat_id')) {
        replacementRootPosts += 1;
        return response({ code: 0, data: { message_id: 'om_2929292929292929' } });
      }
      throw new Error(`unexpected root-missing request ${value}`);
    };
    await drainQueuedCodexEvents({ github: rootMissingGithub, context: flowContext, core: makeCore() });
    const rootMissingLedger = await readDeliveryLedger({
      github: rootMissingGithub, context: flowContext, core: makeCore(), prNum: 42,
    });
    const replacementThread = await readThreadState({
      github: rootMissingGithub, context: flowContext, core: makeCore(), prNum: 42,
    });
    const replacementRecord = rootMissingLedger.latest.get(rootMissingRef.eventKey);
    assert(missingReplyPosts === 1 && replacementRootPosts === 1 && replacementRecord.state === 'done' &&
      replacementRecord.mode === 'root' && replacementRecord.attempt === 2 &&
      replacementThread.kind === 'final' && replacementThread.generation === 2,
    '可信 root_missing 唯一允许 reply/not_sent 跨kind到 attempt+1 root/preparing，并reserve替代话题发送完成');

    const transient429Ref = eventRef('629', '2026-08-05T12:02:09Z');
    const transient429Comments = [botComment(1, encodeEventRef(transient429Ref))];
    const transient429Github = makeGithub({
      comments: transient429Comments,
      checks: new Map([[transient429Ref.eventId, makeCheck(Number(transient429Ref.eventId), {
        createdAt: transient429Ref.createdAt,
      })]]),
    });
    const transient429Uuids = [];
    global.fetch = async (url, options = {}) => {
      const value = String(url);
      if (value.endsWith('/auth/v3/tenant_access_token/internal')) {
        return response({ code: 0, tenant_access_token: 'tenant-token' });
      }
      if (value.includes('receive_id_type=chat_id')) {
        transient429Uuids.push(JSON.parse(options.body).uuid);
        if (transient429Uuids.length === 1) return response({ code: 99991429 }, 429, { 'Retry-After': '0' });
        return response({ code: 0, data: { message_id: 'om_3030303030303030' } });
      }
      throw new Error(`unexpected transient 429 request ${value}`);
    };
    await drainQueuedCodexEvents({ github: transient429Github, context: flowContext, core: makeCore() });
    const transient429Ledger = await readDeliveryLedger({
      github: transient429Github, context: flowContext, core: makeCore(), prNum: 42,
    });
    const transient429States = transient429Comments.map((comment) => decodeDeliveryRecord(comment.body)).filter(Boolean);
    assert(transient429Uuids.length === 2 && transient429Uuids[0] === transient429Uuids[1] &&
      transient429Ledger.latest.get(transient429Ref.eventKey).state === 'done' &&
      transient429States.every((record) => !['uncertain', 'manual'].includes(record.state)),
    'HTTP 429→success 在同root/token/uuid内有界重试，不进入history/manual');

    const persistent429Ref = eventRef('630', '2026-08-05T12:02:10Z');
    const persistent429Comments = [botComment(1, encodeEventRef(persistent429Ref))];
    const persistent429Github = makeGithub({
      comments: persistent429Comments,
      checks: new Map([[persistent429Ref.eventId, makeCheck(Number(persistent429Ref.eventId), {
        createdAt: persistent429Ref.createdAt,
      })]]),
    });
    let persistent429Posts = 0;
    let persistentHistoryCalls = 0;
    global.fetch = async (url) => {
      const value = String(url);
      if (value.endsWith('/auth/v3/tenant_access_token/internal')) {
        return response({ code: 0, tenant_access_token: 'tenant-token' });
      }
      if (value.includes('container_id_type=chat')) {
        persistentHistoryCalls += 1;
        return response({ code: 0, data: { has_more: false, items: [] } });
      }
      if (value.includes('receive_id_type=chat_id')) {
        persistent429Posts += 1;
        return response({ code: 99991429 }, 429, { 'Retry-After': '0' });
      }
      throw new Error(`unexpected persistent 429 request ${value}`);
    };
    await drainQueuedCodexEvents({ github: persistent429Github, context: flowContext, core: makeCore() });
    const persistent429Ledger = await readDeliveryLedger({
      github: persistent429Github, context: flowContext, core: makeCore(), prNum: 42,
    });
    const persisted429 = persistent429Ledger.latest.get(persistent429Ref.eventKey);
    const oldDateNow429 = Date.now;
    Date.now = () => Date.parse(persisted429.nextCheckAt) + 1;
    global.fetch = async (url) => {
      const value = String(url);
      if (value.endsWith('/auth/v3/tenant_access_token/internal')) {
        return response({ code: 0, tenant_access_token: 'tenant-token' });
      }
      if (value.includes('receive_id_type=chat_id')) {
        persistent429Posts += 1;
        return response({ code: 0, data: { message_id: 'om_3131313131313131' } });
      }
      throw new Error(`unexpected recovered 429 request ${value}`);
    };
    await drainQueuedCodexEvents({ github: persistent429Github, context: flowContext, core: makeCore() });
    Date.now = oldDateNow429;
    const recovered429Ledger = await readDeliveryLedger({
      github: persistent429Github, context: flowContext, core: makeCore(), prNum: 42,
    });
    const persistent429States = persistent429Comments.map((comment) => decodeDeliveryRecord(comment.body)).filter(Boolean);
    assert(persistent429Posts === FEISHU_RATE_LIMIT_MAX_ATTEMPTS + 1 && persistentHistoryCalls === 0 &&
      persisted429.state === 'not_sent' && persisted429.reason === 'rate_limited' && persisted429.nextCheckAt &&
      recovered429Ledger.latest.get(persistent429Ref.eventKey).state === 'done' &&
      persistent429States.every((record) => !['uncertain', 'manual'].includes(record.state)),
    '持续429耗尽后durable rate_limited/not_sent，watchdog到期跨run续投且绝不进入history/manual');

    const oversized429Ref = eventRef('632', '2026-08-05T12:02:12Z');
    const oversized429Comments = [botComment(1, encodeEventRef(oversized429Ref))];
    const oversized429Github = makeGithub({
      comments: oversized429Comments,
      checks: new Map([[oversized429Ref.eventId, makeCheck(Number(oversized429Ref.eventId), {
        createdAt: oversized429Ref.createdAt,
      })]]),
    });
    let oversized429Posts = 0;
    let oversized429ResponseAt = 0;
    global.fetch = async (url) => {
      const value = String(url);
      if (value.endsWith('/auth/v3/tenant_access_token/internal')) {
        return response({ code: 0, tenant_access_token: 'tenant-token' });
      }
      if (value.includes('receive_id_type=chat_id')) {
        oversized429Posts += 1;
        oversized429ResponseAt = Date.now();
        return response({ code: 99991429 }, 429, { 'Retry-After': '120' });
      }
      throw new Error(`unexpected oversized 429 request ${value}`);
    };
    const oversizedStart = Date.now();
    await drainQueuedCodexEvents({ github: oversized429Github, context: flowContext, core: makeCore() });
    const oversized429Ledger = await readDeliveryLedger({
      github: oversized429Github, context: flowContext, core: makeCore(), prNum: 42,
    });
    const oversized429 = oversized429Ledger.latest.get(oversized429Ref.eventKey);
    assert(oversized429Posts === 1 && oversized429ResponseAt >= oversizedStart &&
      oversized429.state === 'not_sent' &&
      Date.parse(oversized429.nextCheckAt) - oversized429ResponseAt >= 120000 &&
      Date.parse(oversized429.nextCheckAt) - oversized429ResponseAt <= 121000,
    'Retry-After=120s超过单轮inline cap时不sleep/提前POST，完整持久化nextCheckAt留给watchdog');

    const transientTokenRef = eventRef('633', '2026-08-05T12:02:13Z');
    const transientTokenComments = [botComment(1, encodeEventRef(transientTokenRef))];
    let transientTokenCalls = 0;
    let transientTokenMessagePosts = 0;
    let sendingObservedAfterTokenCalls = 0;
    const transientTokenGithub = makeGithub({
      comments: transientTokenComments,
      checks: new Map([[transientTokenRef.eventId, makeCheck(Number(transientTokenRef.eventId), {
        createdAt: transientTokenRef.createdAt,
      })]]),
      onCreateComment: ({ body }) => {
        if (decodeDeliveryRecord(body)?.state === 'sending') {
          sendingObservedAfterTokenCalls = transientTokenCalls;
        }
      },
    });
    global.fetch = async (url) => {
      const value = String(url);
      if (value.endsWith('/auth/v3/tenant_access_token/internal')) {
        transientTokenCalls += 1;
        if (transientTokenCalls === 1) return response({ code: 99991429 }, 429, { 'Retry-After': '0' });
        return response({ code: 0, tenant_access_token: 'tenant-token' });
      }
      if (value.includes('receive_id_type=chat_id')) {
        transientTokenMessagePosts += 1;
        return response({ code: 0, data: { message_id: 'om_3232323232323232' } });
      }
      throw new Error(`unexpected transient token 429 request ${value}`);
    };
    await drainQueuedCodexEvents({
      github: transientTokenGithub, context: flowContext, core: makeCore(),
    });
    const transientTokenLedger = await readDeliveryLedger({
      github: transientTokenGithub, context: flowContext, core: makeCore(), prNum: 42,
    });
    assert(transientTokenCalls === 2 && transientTokenMessagePosts === 1 &&
      sendingObservedAfterTokenCalls === 2 &&
      transientTokenLedger.latest.get(transientTokenRef.eventKey).state === 'done',
    'tenant token 429→success 在落 sending 前完成有界重试，消息只 POST 一次');

    const persistentTokenRef = eventRef('634', '2026-08-05T12:02:14Z');
    const persistentTokenComments = [botComment(1, encodeEventRef(persistentTokenRef))];
    let persistentTokenCalls = 0;
    let persistentTokenMessagePosts = 0;
    const persistentTokenGithub = makeGithub({
      comments: persistentTokenComments,
      checks: new Map([[persistentTokenRef.eventId, makeCheck(Number(persistentTokenRef.eventId), {
        createdAt: persistentTokenRef.createdAt,
      })]]),
    });
    global.fetch = async (url) => {
      const value = String(url);
      if (value.endsWith('/auth/v3/tenant_access_token/internal')) {
        persistentTokenCalls += 1;
        return response({ code: 99991429 }, 429, { 'Retry-After': '0' });
      }
      if (value.includes('receive_id_type=chat_id')) {
        persistentTokenMessagePosts += 1;
        throw new Error('token 429 后不应 POST 消息');
      }
      throw new Error(`unexpected persistent token 429 request ${value}`);
    };
    await drainQueuedCodexEvents({
      github: persistentTokenGithub, context: flowContext, core: makeCore(),
    });
    const persistentTokenLedger = await readDeliveryLedger({
      github: persistentTokenGithub, context: flowContext, core: makeCore(), prNum: 42,
    });
    const persistedToken429 = persistentTokenLedger.latest.get(persistentTokenRef.eventKey);
    const sendingBeforeTokenRecovery = persistentTokenComments.some((comment) =>
      decodeDeliveryRecord(comment.body)?.state === 'sending');
    const oldDateNowToken429 = Date.now;
    Date.now = () => Date.parse(persistedToken429.nextCheckAt) + 1;
    global.fetch = async (url) => {
      const value = String(url);
      if (value.endsWith('/auth/v3/tenant_access_token/internal')) {
        persistentTokenCalls += 1;
        return response({ code: 0, tenant_access_token: 'tenant-token' });
      }
      if (value.includes('receive_id_type=chat_id')) {
        persistentTokenMessagePosts += 1;
        return response({ code: 0, data: { message_id: 'om_3434343434343434' } });
      }
      throw new Error(`unexpected recovered token 429 request ${value}`);
    };
    await drainQueuedCodexEvents({
      github: persistentTokenGithub, context: flowContext, core: makeCore(),
    });
    Date.now = oldDateNowToken429;
    const recoveredTokenLedger = await readDeliveryLedger({
      github: persistentTokenGithub, context: flowContext, core: makeCore(), prNum: 42,
    });
    assert(persistentTokenCalls === FEISHU_RATE_LIMIT_MAX_ATTEMPTS + 1 &&
      persistentTokenMessagePosts === 1 && !sendingBeforeTokenRecovery &&
      persistedToken429.state === 'not_sent' && persistedToken429.reason === 'rate_limited' &&
      persistedToken429.historyAttempts === 0 &&
      recoveredTokenLedger.latest.get(persistentTokenRef.eventKey).state === 'done' &&
      persistentTokenComments.map((comment) => decodeDeliveryRecord(comment.body)).filter(Boolean)
        .every((record) => !['uncertain', 'manual'].includes(record.state)),
    '持续 tenant token 429 首轮持久化 not_sent/nextCheckAt 且零消息 POST，跨 run 到期后发送完成');

    const history429Ref = eventRef('635', '2026-08-05T12:02:15Z');
    const history429Claim = 'e'.repeat(32);
    const history429Now = Date.now();
    const history429SentAt = new Date(history429Now - 60000).toISOString();
    const history429Base = {
      version: 1, repo: history429Ref.repo, pr: history429Ref.pr,
      eventKey: history429Ref.eventKey, attempt: 1, mode: 'root', targetRoot: null,
      threadGeneration: 1, threadClaimId: history429Claim,
      token: deliveryToken(history429Ref.eventKey), messageId: null, candidateMessageIds: [],
    };
    const history429Comments = [
      botComment(1, encodeEventRef(history429Ref)),
      botComment(2, encodeDeliveryRecord({
        ...history429Base, state: 'preparing', sentAt: null, historyAttempts: 0,
        nextCheckAt: null, reason: null,
      })),
      botComment(3, encodeThreadState({
        version: 2, state: 'pending', repo: history429Ref.repo, pr: history429Ref.pr,
        generation: 1, claimId: history429Claim, messageId: null,
      })),
      botComment(4, encodeDeliveryRecord({
        ...history429Base, state: 'sending', sentAt: history429SentAt, historyAttempts: 0,
        nextCheckAt: null, reason: null,
      })),
      botComment(5, encodeDeliveryRecord({
        ...history429Base, state: 'uncertain', sentAt: history429SentAt, historyAttempts: 7,
        nextCheckAt: new Date(history429Now - 1000).toISOString(), reason: 'ambiguous',
      })),
    ];
    const history429Github = makeGithub({ comments: history429Comments });
    let history429Calls = 0;
    let history429ResponseAt = 0;
    global.fetch = async (url) => {
      const value = String(url);
      if (value.endsWith('/auth/v3/tenant_access_token/internal')) {
        return response({ code: 0, tenant_access_token: 'tenant-token' });
      }
      if (value.includes('container_id_type=chat')) {
        history429Calls += 1;
        history429ResponseAt = Date.now();
        return response({ code: 99991429 }, 429, { 'Retry-After': '120' });
      }
      throw new Error(`unexpected history 429 request ${value}`);
    };
    const history429Core = makeCore();
    await drainQueuedCodexEvents({
      github: history429Github, context: flowContext, core: history429Core,
    });
    const history429Ledger = await readDeliveryLedger({
      github: history429Github, context: flowContext, core: makeCore(), prNum: 42,
    });
    const deferredHistory429 = history429Ledger.latest.get(history429Ref.eventKey);
    assert(history429Calls === 1 && deferredHistory429.state === 'uncertain' &&
      deferredHistory429.reason === 'history_rate_limited' && deferredHistory429.historyAttempts === 7 &&
      Date.parse(deferredHistory429.nextCheckAt) - history429ResponseAt >= 120000 &&
      Date.parse(deferredHistory429.nextCheckAt) - history429ResponseAt <= 121000 &&
      history429Comments.map((comment) => decodeDeliveryRecord(comment.body)).filter(Boolean)
        .every((record) => record.state !== 'manual') &&
      history429Core.outputs.continuation_mode === 'retry',
    'history GET 429 不消耗 ambiguity attempt、不转 manual，仅按完整 Retry-After 延期');

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
    const ensuredRef = eventRef('8010', '2026-08-05T12:08:10Z');
    const ensuredComments = [];
    const ensuredGithub = makeGithub({ comments: ensuredComments, checks: new Map([
      [ensuredRef.eventId, makeCheck(Number(ensuredRef.eventId), { important: false, createdAt: ensuredRef.createdAt })],
    ]) });
    const ensuredContext = context('repository_dispatch', { client_payload: {
      pr_number: '42', event_ref: ensuredRef,
    } });
    const ensuredDrainCore = makeCore();
    await drainQueuedCodexEvents({ github: ensuredGithub, context: ensuredContext, core: ensuredDrainCore });
    const ensuredQueue = await readEventQueue({
      github: ensuredGithub, context: ensuredContext, core: makeCore(), prNum: 42,
    });
    assert(ensuredComments.some((comment) => decodeEventRef(comment.body)?.eventKey === ensuredRef.eventKey),
      `default report 对 dispatch ref strict rehydrate 后 idempotent ensure enqueue，关闭 enqueue 真失败窗口: ${JSON.stringify({ failures: ensuredDrainCore.failures, comments: ensuredComments.map((comment) => comment.body), queue: ensuredQueue.events })}`);
    const tamperedDispatchCore = makeCore();
    const tamperedDispatch = await resolvePrNumber(flowGithub, context('repository_dispatch', { client_payload: {
      pr_number: '42', event_ref: { ...ensuredRef, eventKey: '0'.repeat(64) },
    } }), tamperedDispatchCore);
    assert(tamperedDispatch === null && tamperedDispatchCore.failures.length > 0,
      'dispatch compact ref hash/pr/repo 任一不 canonical 时 fail closed');

    const unsentRef = eventRef('8020', '2026-08-05T12:08:20Z');
    const unsentComments = [];
    const unsentGithub = makeGithub({
      comments: unsentComments,
      checks: new Map([[unsentRef.eventId, makeCheck(Number(unsentRef.eventId), { createdAt: unsentRef.createdAt })]]),
      threadReleaseFails: true,
    });
    await enqueueEventRef({ github: unsentGithub, context: flowContext, core: makeCore(), ref: unsentRef });
    let unsentPosts = 0;
    global.fetch = async (url) => {
      if (String(url).includes('/auth/v3/tenant_access_token/internal')) {
        return response({ code: 0, tenant_access_token: 'tenant-token' });
      }
      if (String(url).includes('/im/v1/messages')) {
        unsentPosts += 1;
        return response({ code: 230001 }, 400);
      }
      return response({ code: 0 });
    };
    await drainQueuedCodexEvents({ github: unsentGithub, context: flowContext, core: makeCore() });
    const unsentLedger = await readDeliveryLedger({
      github: unsentGithub, context: flowContext, core: makeCore(), prNum: 42,
    });
    const unsentThread = await readThreadState({
      github: unsentGithub, context: flowContext, core: makeCore(), prNum: 42,
    });
    assert(unsentLedger.latest.get(unsentRef.eventKey).state === 'not_sent' && unsentThread.kind === 'pending',
      'definitely-not-sent exact delivery 先落盘；release 全部失败时保留 matching pending 给下 run 收敛');
    unsentGithub.failThreadRelease = false;
    global.fetch = async (url) => {
      if (String(url).includes('/auth/v3/tenant_access_token/internal')) {
        return response({ code: 0, tenant_access_token: 'tenant-token' });
      }
      if (String(url).includes('/im/v1/messages')) {
        unsentPosts += 1;
        return response({ code: 0, data: { message_id: 'om_8020802080208020' } });
      }
      return response({ code: 0 });
    };
    await drainQueuedCodexEvents({ github: unsentGithub, context: flowContext, core: makeCore() });
    const convergedLedger = await readDeliveryLedger({
      github: unsentGithub, context: flowContext, core: makeCore(), prNum: 42,
    });
    const notSentIndex = unsentComments.findIndex((comment) => {
      const record = decodeDeliveryRecord(comment.body);
      return record && record.eventKey === unsentRef.eventKey && record.state === 'not_sent';
    });
    const releasedIndex = unsentComments.findIndex((comment) => {
      const state = decodeThreadState(comment.body);
      return state && state.state === 'released';
    });
    assert(notSentIndex >= 0 && releasedIndex > notSentIndex &&
      convergedLedger.latest.get(unsentRef.eventKey).state === 'done' && unsentPosts === 2,
    '下 run 对 persisted not_sent 先 release 再开新 attempt；comment 顺序保证未发送证明先于 release');

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
    assert(cursorGithub.commentReads === 1 && cursorLedger.latest.get(cursorLast.eventKey).state === 'skipped',
      '第二批从 durable cursor 继续且有效 checkpoint 不重扫 legacy prefix，越过 50 条头部 terminal 后队尾事件不饥饿');
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
    const slotTime = 1000 * WATCHDOG_SLOT_MS;
    const windowA = watchdogWindow(manyPulls, 'zettlab/demo', slotTime);
    const windowSame = watchdogWindow(manyPulls, 'zettlab/demo', slotTime + 1000);
    const windowRotated = watchdogWindow(manyPulls, 'zettlab/demo', slotTime + WATCHDOG_SHARDS * WATCHDOG_SLOT_MS);
    assert(windowA.shard === windowSame.shard && JSON.stringify(windowA.pulls) === JSON.stringify(windowSame.pulls),
      'watchdog 同一 time-slot 稳定分片');
    assert(windowA.pulls.length <= WATCHDOG_MAX_PRS_PER_RUN && WATCHDOG_MAX_DISPATCHES === 10 &&
      WATCHDOG_SHARDS === 1 &&
      JSON.stringify(windowA.pulls) !== JSON.stringify(windowRotated.pulls),
    'watchdog 每5分钟轮转且显式限制 PR/dispatch cap，避免固定头部饥饿');
    for (const size of [40, 20, 32]) {
      const sameShardPulls = [];
      for (let number = 1; sameShardPulls.length < size; number += 1) {
        const candidate = { number };
        if (watchdogWindow([candidate], 'zettlab/demo', slotTime).pulls.length === 1) sameShardPulls.push(candidate);
      }
      const before = watchdogWindow(sameShardPulls, 'zettlab/demo', slotTime).pulls;
      const after = watchdogWindow(sameShardPulls, 'zettlab/demo', slotTime + WATCHDOG_SHARDS * WATCHDOG_SLOT_MS).pulls;
      assert(JSON.stringify(before) !== JSON.stringify(after),
        `watchdog 同 shard 每周期前移一位，${size} 条时也不会因 cap 公因数永久饥饿`);
    }
    const boundedSeen = new Set();
    const boundedRounds = Math.ceil(manyPulls.length / WATCHDOG_MAX_PRS_PER_RUN) + 1;
    for (let index = 0; index < boundedRounds; index += 1) {
      for (const pr of watchdogWindow(manyPulls, 'zettlab/demo', slotTime + index * WATCHDOG_SLOT_MS).pulls) {
        boundedSeen.add(pr.number);
      }
    }
    assert(boundedSeen.size === manyPulls.length,
      '400 PR 在保持每轮40/API预算上限时约11个5min轮次完整覆盖，不再叠加4-shard延迟');
    const manualOnlyRef = eventRef('631', '2026-08-05T12:03:01Z');
    let manualOnlyId = 1;
    const manualOnlyComments = [botComment(manualOnlyId++, encodeEventRef(manualOnlyRef))];
    for (const record of cloneManualSequence(manualOnlyRef, 'reply')) {
      manualOnlyComments.push(botComment(manualOnlyId++, encodeDeliveryRecord(record)));
    }
    const manualOnlyGithub = makeGithub({ comments: manualOnlyComments, openPulls: [{ number: 42 }] });
    let manualSlot = 0;
    while (watchdogWindow([{ number: 42 }], 'zettlab/demo', manualSlot).pulls.length === 0) {
      manualSlot += WATCHDOG_SLOT_MS;
    }
    await sweepQueuedCodexReviews({ github: manualOnlyGithub, context: flowContext, core: makeCore(), nowMs: manualSlot });
    assert(manualOnlyGithub.dispatches.length === 0, 'watchdog 排除仅剩 manual 的 PR，避免无效热循环');
    const reconciledCheck = makeCheck(632, { important: false });
    const reconciliationComments = [];
    const reconciliationGithub = makeGithub({
      comments: reconciliationComments, checks: new Map([['632', reconciledCheck]]),
      openPulls: [{ number: 42 }],
    });
    await sweepQueuedCodexReviews({
      github: reconciliationGithub, context: flowContext, core: makeCore(), nowMs: manualSlot,
    });
    assert(reconciliationGithub.dispatches.length === 1 &&
      reconciliationComments.some((comment) => decodeEventRef(comment.body)?.eventId === '632'),
    'schedule 在当前 head 有界枚举官方 reviews/check-runs，先 durable enqueue 缺失 ref 再 dispatch');
    const overflowSourceComments = [];
    const overflowSourceChecks = new Map(Array.from({ length: 25 }, (_value, index) => {
      const id = String(6400 + index);
      return [id, makeCheck(Number(id), {
        important: false, createdAt: new Date(Date.parse(CREATED) + index * 1000).toISOString(),
      })];
    }));
    const overflowSourceGithub = makeGithub({
      comments: overflowSourceComments, checks: overflowSourceChecks, openPulls: [{ number: 42 }],
    });
    await sweepQueuedCodexReviews({
      github: overflowSourceGithub, context: flowContext, core: makeCore(), nowMs: manualSlot,
    });
    const overflowFirstQueue = await readEventQueue({
      github: overflowSourceGithub, context: flowContext, core: makeCore(), prNum: 42,
    });
    await sweepQueuedCodexReviews({
      github: overflowSourceGithub, context: flowContext, core: makeCore(), nowMs: manualSlot,
    });
    const overflowSecondQueue = await readEventQueue({
      github: overflowSourceGithub, context: flowContext, core: makeCore(), prNum: 42,
    });
    assert(overflowFirstQueue.events.length === 20 && overflowSecondQueue.events.length === 25 &&
      overflowSourceGithub.dispatches.length === 2,
    '同一 head 超过20个官方 source 时按稳定 unknown 差集跨轮推进；第二轮覆盖剩余事件且每批 durable 后 dispatch');
    const pagedSourceComments = [];
    const pagedSourceChecks = new Map(Array.from({ length: 105 }, (_value, index) => {
      const id = String(6500 + index);
      return [id, makeCheck(Number(id), {
        important: false, createdAt: new Date(Date.parse(CREATED) + index * 1000).toISOString(),
      })];
    }));
    const pagedSourceGithub = makeGithub({
      comments: pagedSourceComments, checks: pagedSourceChecks, openPulls: [{ number: 42 }],
    });
    for (let round = 0; round < 6; round += 1) {
      await sweepQueuedCodexReviews({
        github: pagedSourceGithub, context: flowContext, core: makeCore(), nowMs: manualSlot,
      });
    }
    const pagedSourceQueue = await readEventQueue({
      github: pagedSourceGithub, context: flowContext, core: makeCore(), prNum: 42,
    });
    assert(pagedSourceQueue.events.length === 105 && pagedSourceGithub.dispatches.length === 6,
      '>100 official sources 通过显式有界分页可达，并按 durable unknown 每轮20最终完整覆盖');

    const overflowExistingRef = eventRef('6999', '2026-08-05T12:09:59Z');
    const pageCapComments = [];
    const pageCapChecks = new Map(Array.from({ length: 1001 }, (_value, index) => {
      const id = String(7000 + index);
      return [id, makeCheck(Number(id), {
        important: false, createdAt: new Date(Date.parse(CREATED) + index * 1000).toISOString(),
      })];
    }));
    const pageCapCallOrder = [];
    const pageCapGithub = makeGithub({
      comments: pageCapComments, checks: pageCapChecks, openPulls: [{ number: 42 }],
      callOrder: pageCapCallOrder,
    });
    await enqueueEventRef({
      github: pageCapGithub, context: flowContext, core: makeCore(), ref: overflowExistingRef,
    });
    const pageCapCore = makeCore();
    await sweepQueuedCodexReviews({
      github: pageCapGithub, context: flowContext, core: pageCapCore, nowMs: manualSlot,
    });
    assert(pageCapGithub.dispatches.length === 1 &&
      pageCapCore.warnings.some((message) => message.includes('checks 达分页 cap')) &&
      pageCapCallOrder[0] === 'dispatch' && pageCapCallOrder.indexOf('dispatch') < pageCapCallOrder.indexOf('reviews:1') &&
      pageCapCallOrder.filter((entry) => entry.startsWith('checks:')).length === 10,
    '已有 durable actionable queue 在任何 reconciliation API 前实际 dispatch；随后10页慢/耗预算 source 不会阻断 wake-up');
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

    const allOpenPulls = Array.from({ length: 251 }, (_value, index) => ({ number: index + 1 }));
    const allOpenGithub = makeGithub({ openPulls: allOpenPulls });
    await sweepQueuedCodexReviews({ github: allOpenGithub, context: flowContext, core: makeCore(), nowMs: slotTime });
    assert(JSON.stringify(allOpenGithub.pullListPages) === JSON.stringify([1, 2, 3]),
      'watchdog 从 page=1 流式读取全部 OPEN PR，不依赖固定 32 页窗口');
    const fairSeen = new Set();
    for (let index = 0; index < 2000 && fairSeen.size < manyPulls.length; index += 1) {
      for (const pr of watchdogWindow(manyPulls, 'zettlab/demo', index * WATCHDOG_SLOT_MS).pulls) fairSeen.add(pr.number);
    }
    assert(fairSeen.size === manyPulls.length, '5min watchdog 对完整 OPEN PR 集合保持可证明的轮转覆盖');

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

    let graphqlAbortObserved = false;
    let graphqlTimedOut = false;
    try {
      await githubGraphqlWithTimeout({
        graphql: async (_query, variables) => new Promise((_resolve, reject) => {
          assert(variables.request.signal instanceof AbortSignal, 'GraphQL abort test 收到 signal');
          variables.request.signal.addEventListener('abort', () => {
            graphqlAbortObserved = true;
            reject(new Error('mock aborted'));
          }, { once: true });
        }),
      }, 'query { viewer { login } }', {}, 5);
    } catch (error) {
      graphqlTimedOut = error.name === 'AbortError';
    }
    assert(graphqlAbortObserved && graphqlTimedOut,
      'GraphQL Octokit 请求由真实 AbortController signal 取消并进入 retryable timeout');

    const checkpointHighCreatedAt = new Date(Date.parse(CREATED) + 549000).toISOString();
    const validCheckpointComments = Array.from({ length: 550 }, (_value, index) => ({
      id: index + 1, body: `checkpoint-old-${index}`,
      created_at: new Date(Date.parse(CREATED) + index * 1000).toISOString(),
      author_association: 'NONE', user: { login: 'user', type: 'User' },
    }));
    validCheckpointComments.push({
      ...botComment(551, encodeCheckpoint({
        version: 1, revision: 1, repo: 'zettlab/demo', pr: 42,
        highCommentId: 550, highCreatedAt: checkpointHighCreatedAt, parentHash: null, entries: [],
      })),
      created_at: new Date(Date.parse(checkpointHighCreatedAt) + 1000).toISOString(),
    });
    const checkpointIncrementalRef = eventRef(
      '7299', new Date(Date.parse(checkpointHighCreatedAt) + 2000).toISOString(),
    );
    validCheckpointComments.push({
      ...botComment(552, encodeEventRef(checkpointIncrementalRef)),
      created_at: checkpointIncrementalRef.createdAt,
    });
    let validCheckpointNow = 0;
    let validCheckpointGraphqlCalls = 0;
    let validCheckpointRestCalls = 0;
    let validCheckpointPrefixCalls = 0;
    const validCheckpointGithub = makeGithub({
      comments: validCheckpointComments,
      onGraphql: () => {
        validCheckpointGraphqlCalls += 1;
        validCheckpointNow += GITHUB_REQUEST_TIMEOUT_MS;
      },
      onListComments: (options) => {
        validCheckpointRestCalls += 1;
        if (!options.since) validCheckpointPrefixCalls += 1;
        validCheckpointNow += GITHUB_REQUEST_TIMEOUT_MS;
      },
    });
    const validCheckpointSnapshot = await withGithubAttemptBudget(() => readThreadMarkerComments({
      github: validCheckpointGithub, context: flowContext, core: makeCore(), prNum: 42,
      nowFn: () => validCheckpointNow,
    }), { deadlineMs: GITHUB_ATTEMPT_BUDGET_MS, nowFn: () => validCheckpointNow });
    assert(validCheckpointSnapshot && validCheckpointSnapshot._checkpoint.checkpoint &&
      validCheckpointSnapshot.some((comment) => Number(comment.id) === 552) &&
      validCheckpointGraphqlCalls === 1 && validCheckpointRestCalls === 1 &&
      validCheckpointPrefixCalls === 0 && validCheckpointNow === 2 * GITHUB_REQUEST_TIMEOUT_MS,
    `valid checkpoint 只做 tail 定位和 high-watermark since 增量，不再读取 legacy REST prefix: ${JSON.stringify({ graphql: validCheckpointGraphqlCalls, rest: validCheckpointRestCalls, prefix: validCheckpointPrefixCalls, now: validCheckpointNow })}`);

    const bootstrapComments = Array.from({ length: 620 }, (_value, index) => ({
      id: index + 1,
      body: `ordinary-${index}`,
      created_at: new Date(Date.parse(CREATED) + index * 1000).toISOString(),
      author_association: 'NONE', user: { login: 'user', type: 'User' },
    }));
    const bootstrapRef = eventRef('7301', bootstrapComments[49].created_at);
    bootstrapComments[49] = { ...botComment(50, encodeEventRef(bootstrapRef)), created_at: bootstrapRef.createdAt };
    const bootstrapGithub = makeGithub({
      comments: bootstrapComments,
      checks: new Map([['7301', makeCheck(7301, { important: false, createdAt: bootstrapRef.createdAt })]]),
    });
    const bootstrapFirstCore = makeCore();
    await drainQueuedCodexEvents({
      github: bootstrapGithub, context: flowContext, core: bootstrapFirstCore, bootstrapPageLimit: 2,
    });
    const progressComment = bootstrapComments.find((comment) => String(comment.body).includes(BOOTSTRAP_MARK));
    const progress = progressComment && decodeBootstrapProgress(progressComment.body);
    assert(bootstrapFirstCore.outputs.continuation_mode === 'backlog' &&
      bootstrapGithub.checkReads.get('7301') === undefined && progress && progress.highCommentId === 200 &&
      progress.entries.length === 1,
    '无 checkpoint 的大 PR 分批 bootstrap；未完成绝不 hydrate/send，普通 comment 不物化进 progress');
    await drainQueuedCodexEvents({ github: bootstrapGithub, context: flowContext, core: makeCore() });
    const bootstrapLedger = await readDeliveryLedger({
      github: bootstrapGithub, context: flowContext, core: makeCore(), prNum: 42,
    });
    assert(bootstrapGithub.checkReads.get('7301') === 1 &&
      bootstrapLedger.latest.get(bootstrapRef.eventKey).state === 'skipped',
    '后续 run 从 trusted bootstrap-progress 增量续读，完整覆盖 620 comments 后才消费事件');

    const lostContinuationComments = Array.from({ length: 620 }, (_value, index) => ({
      id: index + 1,
      body: `lost-continuation-${index}`,
      created_at: new Date(Date.parse(CREATED) + index * 1000).toISOString(),
      author_association: 'NONE', user: { login: 'user', type: 'User' },
    }));
    const lostContinuationRef = eventRef('7302', lostContinuationComments[49].created_at);
    lostContinuationComments[49] = {
      ...botComment(50, encodeEventRef(lostContinuationRef)), created_at: lostContinuationRef.createdAt,
    };
    const lostContinuationGithub = makeGithub({
      comments: lostContinuationComments,
      checks: new Map([['7302', makeCheck(7302, {
        important: false, createdAt: lostContinuationRef.createdAt,
      })]]),
      openPulls: [{ number: 42 }],
    });
    const bootstrapHighWatermarks = [];
    let lastBootstrapRevision = 0;
    for (let round = 0; round < 4; round += 1) {
      lostContinuationGithub.dispatches.splice(0);
      const continuationCore = makeCore();
      await sweepQueuedCodexReviews({
        github: lostContinuationGithub, context: flowContext, core: continuationCore,
        nowMs: manualSlot, bootstrapPageLimit: 2,
      });
      const progress = lostContinuationComments
        .map((comment) => decodeBootstrapProgress(comment.body))
        .filter(Boolean)
        .sort((left, right) => left.revision - right.revision)
        .at(-1);
      if (progress && progress.revision > lastBootstrapRevision) {
        bootstrapHighWatermarks.push(progress.highCommentId);
        lastBootstrapRevision = progress.revision;
      }
      assert(continuationCore.failures.length === 0 && lostContinuationGithub.dispatches.length === 1,
        `bootstrap continuation 第${round + 1}轮即使前一dispatch丢失仍从schedule重新唤醒`);
    }
    assert(bootstrapHighWatermarks.length >= 2 &&
      bootstrapHighWatermarks.every((high, index) => index === 0 || high > bootstrapHighWatermarks[index - 1]),
    `丢失的 repository_dispatch 不会令 bootstrap 重头扫描，durable high-watermark 跨schedule推进: ${bootstrapHighWatermarks}`);
    await drainQueuedCodexEvents({
      github: lostContinuationGithub, context: flowContext, core: makeCore(),
    });
    const lostContinuationLedger = await readDeliveryLedger({
      github: lostContinuationGithub, context: flowContext, core: makeCore(), prNum: 42,
    });
    assert(lostContinuationLedger.latest.get(lostContinuationRef.eventKey).state === 'skipped',
      '任一后续 bootstrap continuation 最终送达后从 durable progress 完成扫描并消费原队列事件');

    const budgetComments = Array.from({ length: 1250 }, (_value, index) => ({
      id: index + 1,
      body: `budget-ordinary-${index}`,
      created_at: new Date(Date.parse(CREATED) + index * 1000).toISOString(),
      author_association: 'NONE', user: { login: 'user', type: 'User' },
    }));
    let budgetNow = 0;
    let budgetGraphqlCalls = 0;
    let budgetRestCalls = 0;
    let budgetPrefixCalls = 0;
    const budgetGithub = makeGithub({
      comments: budgetComments,
      onListComments: (options) => {
        budgetRestCalls += 1;
        if (!options.since) budgetPrefixCalls += 1;
        budgetNow += GITHUB_REQUEST_TIMEOUT_MS;
      },
      onCreateComment: (options) => {
        if (String(options.body).includes(BOOTSTRAP_MARK)) budgetNow += GITHUB_REQUEST_TIMEOUT_MS;
      },
      onGraphql: () => {
        budgetGraphqlCalls += 1;
        budgetNow += GITHUB_REQUEST_TIMEOUT_MS;
      },
      bootstrapCreateLosesResponse: true,
    });
    const budgetCore = makeCore();
    const budgetSnapshot = await withGithubAttemptBudget(() => readThreadMarkerComments({
      github: budgetGithub, context: flowContext, core: budgetCore, prNum: 42,
      nowFn: () => budgetNow,
    }), { deadlineMs: GITHUB_ATTEMPT_BUDGET_MS, nowFn: () => budgetNow });
    const budgetProgressComment = budgetComments.find((comment) =>
      String(comment.body).includes(BOOTSTRAP_MARK));
    const budgetProgress = budgetProgressComment && decodeBootstrapProgress(budgetProgressComment.body);
    assert(budgetCore.failures.length === 0 && budgetSnapshot &&
      budgetSnapshot._checkpoint.bootstrapIncomplete && budgetProgress &&
      budgetProgress.highCommentId === 400 && budgetNow === 165000 && budgetGraphqlCalls === 6 &&
      budgetRestCalls === 4 && budgetPrefixCalls === 4,
    `180s 共享预算仅在无 checkpoint bootstrap 扫 legacy prefix，并预留 create+response-loss confirm: ${JSON.stringify({ failures: budgetCore.failures, warnings: budgetCore.warnings, high: budgetProgress && budgetProgress.highCommentId, now: budgetNow, graphql: budgetGraphqlCalls, rest: budgetRestCalls, prefix: budgetPrefixCalls })}`);

    const siblingProgressA = {
      version: 1, revision: 1, repo: 'zettlab/demo', pr: 42,
      highCommentId: 100, highCreatedAt: CREATED, parentHash: null, entries: [],
    };
    const siblingProgressB = {
      ...siblingProgressA, highCommentId: 200,
      highCreatedAt: new Date(Date.parse(CREATED) + 1000).toISOString(),
    };
    const siblingProgressComments = [
      { ...botComment(1001, encodeBootstrapProgress(siblingProgressA)), created_at: CREATED },
      { ...botComment(1002, encodeBootstrapProgress(siblingProgressB)), created_at: CREATED },
    ];
    const siblingProgressCore = makeCore();
    const siblingProgressSnapshot = await readThreadMarkerComments({
      github: makeGithub({ comments: siblingProgressComments }), context: flowContext,
      core: siblingProgressCore, prNum: 42, bootstrapPageLimit: 0,
    });
    const joinedProgress = siblingProgressComments.map((comment) => decodeBootstrapProgress(comment.body))
      .find((progress) => progress && progress.revision === 2);
    const siblingAHash = crypto.createHash('sha256')
      .update(JSON.stringify(siblingProgressA, Object.keys(siblingProgressA).sort())).digest('hex');
    assert(siblingProgressCore.failures.length === 0 && siblingProgressSnapshot &&
      siblingProgressSnapshot._checkpoint.bootstrapIncomplete && joinedProgress &&
      joinedProgress.highCommentId === 100 && joinedProgress.parentHash === siblingAHash &&
      siblingProgressCore.warnings.some((message) => message.includes('并发 sibling')),
    `bootstrap 跨 worker 同 revision sibling 归约到最低 high-watermark 并续写 canonical parent: ${JSON.stringify({ failures: siblingProgressCore.failures, warnings: siblingProgressCore.warnings, joined: joinedProgress })}`);

    const lostComments = [];
    const lostGithub = makeGithub({ comments: lostComments, checkpointCreateLosesResponse: true });
    const lostCore = makeCore();
    await drainQueuedCodexEvents({ github: lostGithub, context: flowContext, core: lostCore });
    assert(lostCore.failures.length === 0 &&
      lostComments.filter((comment) => String(comment.body).includes(checkpointMark)).length === 2,
    '首次 checkpoint POST 单次执行；响应丢失后 tail 按 canonical hash 恢复，随后仅追加下一 revision');

    const laterLostComments = [];
    const laterLostGithub = makeGithub({ comments: laterLostComments, checkpointCreateLosesResponseAt: 2 });
    const laterLostCore = makeCore();
    await drainQueuedCodexEvents({ github: laterLostGithub, context: flowContext, core: laterLostCore });
    assert(laterLostCore.failures.length === 0,
      'ambiguous confirm 允许 tail 同时含旧 revision，只要求 expected revision 存在且唯一');

    const conflictComments = [];
    const conflictGithub = makeGithub({
      comments: conflictComments,
      checkpointCreateLosesResponse: true,
      beforeCheckpointCreate: (items, expectedBody) => {
        const expected = decodeCheckpoint(expectedBody);
        const conflicting = encodeCheckpoint({
          ...expected,
          highCommentId: expected.highCommentId + 1,
          highCreatedAt: CREATED,
        });
        items.push({ ...botComment(998, conflicting), created_at: CREATED });
      },
    });
    const conflictCore = makeCore();
    await drainQueuedCodexEvents({ github: conflictGithub, context: flowContext, core: conflictCore });
    assert(conflictCore.failures.some((message) => message.includes('checkpoint POST 未确认')) &&
      conflictCore.outputs.continuation_mode === 'retry',
    'ambiguous checkpoint 即使 expected hash 存在，same revision 不同 hash 也 fail closed');
    const conflictRecoveryCore = makeCore();
    await drainQueuedCodexEvents({ github: conflictGithub, context: flowContext, core: conflictRecoveryCore });
    const conflictRevisions = conflictComments.map((comment) => decodeCheckpoint(comment.body))
      .filter(Boolean).map((checkpoint) => checkpoint.revision);
    assert(conflictRecoveryCore.failures.length === 0 && conflictRevisions.includes(2) &&
      conflictRecoveryCore.warnings.some((message) => message.includes('并发 sibling')),
    `跨进程同 base/same revision 分支选择最低 high-watermark、source replay 后生成下一 revision，不永久冲突: ${JSON.stringify({ failures: conflictRecoveryCore.failures, warnings: conflictRecoveryCore.warnings, revisions: conflictRevisions })}`);

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
    const interleavedFirstCore = makeCore();
    const interleavedSecondCore = makeCore();
    await drainQueuedCodexEvents({ github: interleavedGithub, context: flowContext, core: interleavedFirstCore });
    await drainQueuedCodexEvents({ github: interleavedGithub, context: flowContext, core: interleavedSecondCore });
    const interleavedLedger = await readDeliveryLedger({
      github: interleavedGithub, context: flowContext, core: makeCore(), prNum: 42,
    });
    const interleavedState = interleavedLedger.latest.get(interleavedRef.eventKey);
    assert(interleavedState && interleavedState.state === 'skipped' &&
      interleavedComments.findIndex((comment) => String(comment.body).includes(checkpointMark)) >
        interleavedComments.findIndex((comment) => decodeEventRef(comment.body)?.eventKey === interleavedRef.eventKey),
    `499-prefix 后并发 enqueue 抢在 POST 前成为第500条，append-only checkpoint 位于501仍可从 tail 定位: ${JSON.stringify({ state: interleavedState, ledgerOk: interleavedLedger.ok, latest: [...interleavedLedger.latest.entries()], first: interleavedFirstCore.failures, second: interleavedSecondCore.failures, warnings: interleavedSecondCore.warnings, outputs: interleavedSecondCore.outputs, markerBodies: interleavedComments.filter((comment) => String(comment.body).includes('codex-review-feishu-')).map((comment) => ({ id: comment.id, body: String(comment.body).slice(0, 45) })) })}`);

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
    const tombstoneGithub = makeGithub({
      comments: tombstoneComments, checks: tombstoneChecks, openPulls: [{ number: 42 }],
    });
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
    const signedRepairWatchGithub = makeGithub({
      comments: signedDiscardComments,
      checks: discardChecks,
      openPulls: [{ number: 42 }],
    });
    const terminalCases = [
      { github: flowGithub, refs: [flowRef, barrierRef], label: 'done/skipped delivery ledger', expectAbsent: false },
      { github: signedRepairWatchGithub, refs: [discardRef], label: 'signed manual discard ledger', expectAbsent: false },
      { github: replyRetryGithub, refs: [replyRetryRef], label: 'compacted done tombstone', expectAbsent: true },
      { github: discardGithub, refs: [discardRef], label: 'compacted manual discard tombstone', expectAbsent: true },
      { github: tombstoneGithub, refs: [completedRef, afterCompletedRef], label: 'compacted skipped tombstones', expectAbsent: true },
    ];
    for (const terminalCase of terminalCases) {
      const beforeDispatches = terminalCase.github.dispatches.length;
      const beforeQueue = await readEventQueue({
        github: terminalCase.github, context: flowContext, core: makeCore(), prNum: 42,
      });
      const terminalCore = makeCore();
      await sweepQueuedCodexReviews({
        github: terminalCase.github, context: flowContext, core: terminalCore, nowMs: manualSlot,
      });
      const terminalQueue = await readEventQueue({
        github: terminalCase.github, context: flowContext, core: makeCore(), prNum: 42,
      });
      assert(terminalCore.failures.length === 0 && terminalCase.github.dispatches.length === beforeDispatches &&
        JSON.stringify(terminalQueue.events) === JSON.stringify(beforeQueue.events) &&
        (!terminalCase.expectAbsent || terminalCase.refs.every((terminalRef) =>
          !terminalQueue.events.some(({ ref: queuedRef }) => queuedRef.eventKey === terminalRef.eventKey))),
      `${terminalCase.label} 被 source reconciliation 识别为可信终态，不重建 queue/dispatch: ${JSON.stringify({
        failures: terminalCore.failures,
        warnings: terminalCore.warnings,
        beforeDispatches,
        afterDispatches: terminalCase.github.dispatches.length,
        queue: terminalQueue.events.map(({ ref }) => ({ eventId: ref.eventId, eventKey: ref.eventKey })),
      })}`);
    }

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
