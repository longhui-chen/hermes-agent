#!/usr/bin/env node

const fs = require('fs');
const os = require('os');
const path = require('path');
const {
  GITHUB_ATTEMPT_BUDGET_MS,
  GITHUB_REQUEST_TIMEOUT_MS,
  MARK,
  MAX_GITHUB_RETRY_DELAY_MS,
  STATE_MARK,
  decodeThreadState,
  encodeThreadState,
  finalizeThreadGeneration,
  isTrustedMarkerComment,
  notifyFromActionFailure,
  parseRetryAfterMs,
  readThreadState,
  releaseThreadGeneration,
  reserveThreadGeneration,
  resolvePr,
  resolvePrNumber,
  shouldRecreateRootOnReplyFailure,
  validFeishuMessageId,
  withGithubAttemptBudget,
  withGithubRetry,
} = require('./notify');

function assert(condition, message) {
  if (!condition) throw new Error(message);
  console.log(`PASS: ${message}`);
}

function response(json, status = 200) {
  return { ok: status >= 200 && status < 300, status, text: async () => JSON.stringify(json) };
}

function makeCore() {
  return {
    failures: [], infos: [], warnings: [],
    info(message) { this.infos.push(message); },
    warning(message) { this.warnings.push(message); },
    setFailed(message) { this.failures.push(message); },
    setSecret() {},
  };
}

function botComment(id, body) {
  return {
    id,
    body,
    author_association: 'NONE',
    user: { login: 'github-actions[bot]', type: 'Bot' },
    performed_via_github_app: { slug: 'github-actions' },
  };
}

function makeGithub({ comments = [], associated = [], createHook, updateHook } = {}) {
  const listComments = async () => {};
  const listReviewComments = async () => {};
  const github = {
    rest: {
      repos: {
        listPullRequestsAssociatedWithCommit: async (options) => {
          assert(options.request.timeout === GITHUB_REQUEST_TIMEOUT_MS, 'resolve API 使用 15s timeout');
          return { data: associated };
        },
      },
      pulls: {
        get: async (options) => {
          assert(options.request.timeout === GITHUB_REQUEST_TIMEOUT_MS, 'PR get 使用 15s timeout');
          return { data: {
            number: options.pull_number,
            title: '测试 PR',
            html_url: `https://github.com/zettlab/demo/pull/${options.pull_number}`,
            user: { login: 'owner' },
            base: { ref: 'main' },
            head: { ref: 'feature/test' },
            state: 'open',
          } };
        },
        listReviewComments,
      },
      issues: {
        listComments,
        createComment: async (options) => {
          assert(options.request.timeout === GITHUB_REQUEST_TIMEOUT_MS, 'comment create 使用 15s timeout');
          const id = 1000 + comments.length;
          comments.push(botComment(id, options.body));
          if (createHook) return createHook(options, comments);
          return { data: comments[comments.length - 1] };
        },
        updateComment: updateHook || (async () => ({ data: {} })),
      },
    },
    paginate: async (fn, options) => {
      assert(options.request.timeout === GITHUB_REQUEST_TIMEOUT_MS, 'paginate 使用 15s timeout');
      if (fn === listComments) return comments.slice();
      if (fn === listReviewComments) return [];
      throw new Error('unexpected paginate target');
    },
  };
  return github;
}

function context(payload = { pull_request: { number: 42 } }) {
  return { repo: { owner: 'zettlab', repo: 'demo' }, payload, sha: 'abc123' };
}

function state(overrides = {}) {
  return {
    version: 2,
    state: 'pending',
    repo: 'zettlab/demo',
    pr: 42,
    generation: 1,
    claimId: '0123456789abcdef0123456789abcdef',
    messageId: null,
    ...overrides,
  };
}

async function main() {
  const oldFetch = global.fetch;
  const oldEnv = { ...process.env };
  const runnerTemp = fs.mkdtempSync(path.join(os.tmpdir(), 'codex-feishu-state-'));
  process.env.RUNNER_TEMP = runnerTemp;
  process.env.CODEREVIEW_FEISHU_APP_ID = 'app-id';
  process.env.CODEREVIEW_FEISHU_APP_SECRET = 'app-secret';
  process.env.CODEREVIEW_FEISHU_CHAT_ID = 'chat-id';

  try {
    assert(GITHUB_ATTEMPT_BUDGET_MS === 180000, '单次通知 GitHub attempt budget 固定 180s');
    assert(GITHUB_REQUEST_TIMEOUT_MS === 15000, 'GitHub API timeout 固定 15s');
    const retryError = new Error('limited');
    retryError.response = { headers: { 'Retry-After': '90' } };
    assert(parseRetryAfterMs(retryError, 0) === MAX_GITHUB_RETRY_DELAY_MS, 'Retry-After 数字 clamp 30s');
    retryError.response.headers['Retry-After'] = new Date(90000).toUTCString();
    assert(parseRetryAfterMs(retryError, 0) === MAX_GITHUB_RETRY_DELAY_MS, 'Retry-After HTTP-date clamp 30s');
    let attempts = 0;
    const waits = [];
    const retried = await withGithubRetry({
      core: makeCore(), label: 'retry', sleepFn: async (ms) => waits.push(ms),
      operation: async () => { attempts += 1; if (attempts < 3) throw new Error('transient'); return 'ok'; },
    });
    assert(retried.ok && attempts === 3 && waits.length === 2, 'GitHub API 固定最多 3 次尝试');
    let budgetNow = 0;
    let budgetOperations = 0;
    let budgetSleeps = 0;
    const sharedDeadline = GITHUB_ATTEMPT_BUDGET_MS;
    const firstBudgetWrapper = await withGithubRetry({
      core: makeCore(), label: 'budget-first', deadlineMs: sharedDeadline, nowFn: () => budgetNow,
      operation: async () => {
        budgetOperations += 1;
        budgetNow = sharedDeadline - GITHUB_REQUEST_TIMEOUT_MS - 1;
        return 'ok';
      },
    });
    const secondBudgetWrapper = await withGithubRetry({
      core: makeCore(), label: 'budget-second', deadlineMs: sharedDeadline, nowFn: () => budgetNow,
      sleepFn: async () => { budgetSleeps += 1; },
      operation: async () => {
        budgetOperations += 1;
        budgetNow += 2;
        throw new Error('near deadline');
      },
    });
    assert(firstBudgetWrapper.ok && !secondBudgetWrapper.ok && budgetOperations === 2 && budgetSleeps === 0,
      '多个 retry wrapper 共享 deadline，近截止失败不 sleep 或追加 operation');
    let blockedOperations = 0;
    const blockedByBudget = await withGithubRetry({
      core: makeCore(), label: 'budget-blocked', deadlineMs: sharedDeadline, nowFn: () => budgetNow,
      sleepFn: async () => { budgetSleeps += 1; },
      operation: async () => { blockedOperations += 1; },
    });
    assert(!blockedByBudget.ok && blockedOperations === 0 && budgetSleeps === 0,
      '剩余预算不足单次 timeout 时 operation 立即显式失败');
    let nestedNow = 0;
    let nestedOperations = 0;
    let nestedResult;
    await withGithubAttemptBudget(async () => {
      nestedNow = 1001;
      await withGithubAttemptBudget(async () => {
        nestedResult = await withGithubRetry({
          core: makeCore(), label: 'nested-budget', nowFn: () => nestedNow,
          operation: async () => { nestedOperations += 1; },
        });
      }, { deadlineMs: 999999, nowFn: () => nestedNow });
    }, { deadlineMs: GITHUB_REQUEST_TIMEOUT_MS + 1000, nowFn: () => nestedNow });
    assert(!nestedResult.ok && nestedOperations === 0,
      '嵌套 GitHub attempt budget 复用外层 deadline，不被内层延长');

    const workflow = fs.readFileSync(path.join(__dirname, '../../.github/workflows/codex-review-feishu.yml'), 'utf8');
    const notifySource = fs.readFileSync(path.join(__dirname, 'notify.js'), 'utf8');
    for (const entry of [
      'notifyFromCheckRun',
      'notifyFromPullRequestReview',
      'notifyFromOfficialCodexEvent',
      'notifyFromActionResult',
      'notifyFromActionFailure',
    ]) {
      assert(new RegExp(`async function ${entry}\\(args\\) \\{\\n\\s+return withGithubAttemptBudget\\(`).test(notifySource),
        `${entry} 直调也受共享 GitHub attempt budget 保护`);
    }
    const reportJob = workflow.slice(workflow.indexOf('  report:'), workflow.indexOf('  normalize_repair:'));
    const repairJob = workflow.slice(workflow.indexOf('  repair_pending:'));
    assert(reportJob.includes('scripts/codereview') && reportJob.includes('.github/workflows/codex-review-feishu.yml'),
      '运行 shared flow test 的 report sparse checkout 包含 scripts 与 workflow');
    assert(workflow.includes('  normalize_repair:') && workflow.includes('/^[1-9][0-9]*$/') && workflow.includes('Number.isSafeInteger(prNumber)'),
      'workflow_dispatch 先无 secret canonicalize safe PR number');
    assert(repairJob.includes('needs: normalize_repair') &&
      repairJob.includes('group: codex-review-feishu-${{ github.repository }}-pr-${{ needs.normalize_repair.outputs.pr_number }}') &&
      repairJob.includes('REPAIR_PR_NUMBER: ${{ needs.normalize_repair.outputs.pr_number }}') &&
      !/group:.*inputs\.pr_number/.test(repairJob),
    'repair write job 仅使用 normalize output 且 concurrency 前缀与 report 一致');
    assert(repairJob.includes('withGithubAttemptBudget,') && repairJob.includes('await withGithubAttemptBudget(async () => {'),
      'repair resolve/read/finalize|release 链共享 180s GitHub attempt budget');
    assert(!/uses:\s+actions\/[^@\s]+@v\d+\b/.test(workflow), 'workflow 内 actions 均 pin 完整 SHA');

    const encoded = encodeThreadState(state());
    assert(encoded.startsWith(STATE_MARK), 'v2 state 使用独立 hidden marker');
    assert(JSON.stringify(decodeThreadState(encoded)) === JSON.stringify(state()), 'v2 state 严格 round-trip');
    assert(decodeThreadState(`${encoded}${encoded}`) === null, '重复 v2 marker 被拒绝');
    assert(decodeThreadState(encodeThreadState(state({ state: 'released' }))).state === 'released', 'released 要求 messageId=null');
    assert(isTrustedMarkerComment(botComment(1, encoded)), 'v2 仅信任 github-actions bot');
    assert(!isTrustedMarkerComment({ body: encoded, user: { login: 'owner', type: 'User' }, author_association: 'OWNER' }), 'OWNER 不能伪造 v2');
    const legacyBody = `${MARK}om_1234567890abcdef -->\n<sub>Codex 代码评审话题锚点（自动维护，请勿删除）</sub>`;
    assert(isTrustedMarkerComment({ body: legacyBody, user: { login: 'owner', type: 'User' }, author_association: 'OWNER' }), 'legacy exact final 兼容 OWNER');
    assert(validFeishuMessageId('om_1234567890abcdef'), '合法 Feishu message_id');
    assert(shouldRecreateRootOnReplyFailure({ status: 400, json: { code: 230011 } }), '仅明确 recalled code 重建 root');
    assert(!shouldRecreateRootOnReplyFailure({ status: 404, json: { code: 230001 } }), 'HTTP 404 本身不是 root missing 证据');

    const uniqueCore = makeCore();
    const uniqueGithub = makeGithub({ associated: [{ number: 77 }] });
    const unique = await resolvePrNumber(uniqueGithub, context({ check_run: { head_sha: 'sha', pull_requests: [] } }), uniqueCore);
    assert(unique === 77 && uniqueCore.failures.length === 0, 'check SHA 唯一解析 PR');
    const ambiguousCore = makeCore();
    const ambiguousGithub = makeGithub({ associated: [{ number: 77 }, { number: 78 }] });
    const ambiguous = await resolvePrNumber(ambiguousGithub, context({ check_run: { head_sha: 'sha', pull_requests: [] } }), ambiguousCore);
    assert(ambiguous === null && ambiguousCore.failures.length === 1, 'ambiguous PR 解析 fail closed');
    const stateFilteredCore = makeCore();
    const stateFilteredGithub = makeGithub({ associated: [{ number: 41, state: 'closed' }, { number: 42, state: 'open' }] });
    const stateFiltered = await resolvePrNumber(stateFilteredGithub, context({ check_run: { head_sha: 'sha', pull_requests: [] } }), stateFilteredCore);
    assert(stateFiltered === 42 && stateFilteredCore.failures.length === 0, 'associated API 排除 closed PR 后唯一解析 open PR');
    const resolved = await resolvePr(uniqueGithub, context(), { number: 42 }, makeCore());
    assert(resolved.number === 42 && resolved.state === 'open', 'resolvePr 返回 OPEN 状态并走 timeout/retry');
    let optionGetAttempts = 0;
    const optionGithub = {
      rest: {
        pulls: {
          get: async (params) => {
            optionGetAttempts += 1;
            assert(params.pull_number === 42, 'resolvePr options.prNum 传递 PR number');
            assert(params.request.timeout === GITHUB_REQUEST_TIMEOUT_MS, 'resolvePr pulls.get 使用 15s timeout');
            if (optionGetAttempts === 1) {
              const error = new Error('temporary upstream failure');
              error.status = 503;
              error.response = { status: 503, headers: { 'retry-after': '0' } };
              throw error;
            }
            return {
              data: {
                number: 42,
                state: 'open',
                title: 'Test PR',
                html_url: 'https://github.com/zettlab/repo/pull/42',
                user: { login: 'octocat' },
                base: { ref: 'main' },
                head: { ref: 'fix/codex-feishu-pr-thread' },
              },
            };
          },
        },
      },
    };
    const optionResolved = await resolvePr({ github: optionGithub, context: context(), core: makeCore(), prNum: 42 });
    assert(optionResolved.number === 42 && optionGetAttempts === 2, 'resolvePr options.prNum 调用 pulls.get 并重试瞬态失败');

    const comments = [];
    const github = makeGithub({ comments });
    const core = makeCore();
    const reviewEventKey = 'zettlab/demo#42:event-1';
    const recoveryFile = path.join(runnerTemp, 'reserve.json');
    const claimId = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa';
    const reserved = await reserveThreadGeneration({
      github, context: context(), core, prNum: 42, generation: 1, claimId,
      reviewEventKey, recoveryFile,
    });
    assert(reserved.ok, 'pending append 后重读确认才允许发送');
    const finalized = await finalizeThreadGeneration({
      github, context: context(), core, prNum: 42, generation: 1, claimId,
      messageId: 'om_1234567890abcdef',
    });
    assert(finalized.ok, 'final terminal append 覆盖同 claim pending');
    const finalState = await readThreadState({ github, context: context(), core, prNum: 42 });
    assert(finalState.kind === 'final' && finalState.messageId === 'om_1234567890abcdef', '最高 generation final 生效');
    assert(comments.some((comment) => comment.body.includes(`${MARK}om_1234567890abcdef -->`)), 'final 同时写 legacy exact marker');

    const releasedComments = [botComment(1, encodeThreadState(state({ claimId: 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb' })))];
    const releasedGithub = makeGithub({ comments: releasedComments });
    const released = await releaseThreadGeneration({
      github: releasedGithub, context: context(), core: makeCore(), prNum: 42,
      generation: 1, claimId: 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb', reason: 'known-failure',
    });
    assert(released.ok, 'known failure 可追加 released terminal');
    const releasedState = await readThreadState({ github: releasedGithub, context: context(), core: makeCore(), prNum: 42 });
    assert(releasedState.kind === 'released', 'released 最高代不回退旧 final');

    const duplicateClaim = 'cccccccccccccccccccccccccccccccc';
    const duplicateComments = [
      botComment(1, encodeThreadState(state({ claimId: duplicateClaim }))),
      botComment(2, encodeThreadState(state({ claimId: duplicateClaim }))),
    ];
    const duplicateState = await readThreadState({ github: makeGithub({ comments: duplicateComments }), context: context(), core: makeCore(), prNum: 42 });
    assert(duplicateState.ok && duplicateState.kind === 'pending', '同 claim 重复 pending 折叠');
    duplicateComments.push(botComment(3, encodeThreadState(state({ claimId: 'dddddddddddddddddddddddddddddddd' }))));
    const conflictCore = makeCore();
    const conflict = await readThreadState({ github: makeGithub({ comments: duplicateComments }), context: context(), core: conflictCore, prNum: 42 });
    assert(!conflict.ok && conflict.kind === 'conflict', '同 generation 不同 claim fail closed');

    const legacyComments = [{
      id: 9, body: legacyBody, author_association: 'MEMBER', user: { login: 'maintainer', type: 'User' },
    }];
    const legacyState = await readThreadState({ github: makeGithub({ comments: legacyComments }), context: context(), core: makeCore(), prNum: 42 });
    assert(legacyState.kind === 'final' && legacyState.generation === 0, 'legacy exact final 映射 generation=0');

    const rootRequests = [];
    const replyRequests = [];
    global.fetch = async (url, options) => {
      const value = String(url);
      if (value.endsWith('/auth/v3/tenant_access_token/internal')) return response({ code: 0, tenant_access_token: 'tenant-token' });
      const body = JSON.parse(options.body);
      if (value.includes('/reply')) { replyRequests.push(body); return response({ code: 0, data: { message_id: 'om_fedcba0987654321' } }); }
      if (value.includes('receive_id_type=chat_id')) { rootRequests.push(body); return response({ code: 0, data: { message_id: 'om_abcdef1234567890' } }); }
      throw new Error(`unexpected Feishu request ${value}`);
    };
    const normalComments = [];
    const normalGithub = makeGithub({ comments: normalComments });
    const normalCore = makeCore();
    await notifyFromActionFailure({ github: normalGithub, context: context(), core: normalCore, failure: { reason: 'runner', runId: 'run-1' } });
    await notifyFromActionFailure({ github: normalGithub, context: context(), core: normalCore, failure: { reason: 'runner', runId: 'run-2' } });
    assert(rootRequests.length === 1 && replyRequests.length === 1, '旧正常流程保持每 PR 一根、后续 thread reply');

    const reservingComments = [];
    let pendingCreateAttempts = 0;
    let reservingRootRequests = 0;
    const reservingGithub = makeGithub({
      comments: reservingComments,
      createHook: async (options, ledger) => {
        const createdState = decodeThreadState(options.body);
        if (createdState && createdState.state === 'pending') {
          pendingCreateAttempts += 1;
          if (pendingCreateAttempts <= 3) {
            ledger.pop();
            const error = new Error('pending comment definitely rejected');
            error.status = 422;
            error.response = { status: 422, headers: { 'retry-after': '0' } };
            throw error;
          }
        }
        return { data: ledger[ledger.length - 1] };
      },
    });
    global.fetch = async (url) => {
      if (String(url).endsWith('/auth/v3/tenant_access_token/internal')) return response({ code: 0, tenant_access_token: 'tenant-token' });
      if (String(url).includes('receive_id_type=chat_id')) {
        reservingRootRequests += 1;
        return response({ code: 0, data: { message_id: 'om_reserving12345678' } });
      }
      throw new Error(`unexpected Feishu request ${url}`);
    };
    const reservingFailure = { reason: 'runner', runId: 'run-reserving-retry' };
    await notifyFromActionFailure({ github: reservingGithub, context: context(), core: makeCore(), failure: reservingFailure });
    assert(pendingCreateAttempts === 3 && reservingRootRequests === 0 && reservingComments.length === 0,
      'pending create 三次明确失败时保留 reserving journal 且不发 root');
    const reservingRetryCore = makeCore();
    await notifyFromActionFailure({ github: reservingGithub, context: context(), core: reservingRetryCore, failure: reservingFailure });
    const reservingFinal = await readThreadState({ github: reservingGithub, context: context(), core: makeCore(), prNum: 42 });
    const reservingStates = reservingComments.map((comment) => decodeThreadState(comment.body)).filter(Boolean);
    assert(pendingCreateAttempts === 4 && reservingRootRequests === 1 && reservingFinal.kind === 'final' &&
      new Set(reservingStates.map((entry) => entry.claimId)).size === 1 && reservingRetryCore.failures.length === 0,
    '第二 workflow attempt 以同 event/claim 重做 reservation，确认 pending 后仅发一个 root');

    const timeoutComments = [];
    const timeoutGithub = makeGithub({ comments: timeoutComments });
    const timeoutCore = makeCore();
    let timedRootRequests = 0;
    global.fetch = async (url) => {
      if (String(url).endsWith('/auth/v3/tenant_access_token/internal')) return response({ code: 0, tenant_access_token: 'tenant-token' });
      timedRootRequests += 1;
      throw new Error('network reset after accept');
    };
    const timeoutFailure = { reason: 'ambiguous', runId: 'run-timeout' };
    await notifyFromActionFailure({ github: timeoutGithub, context: context(), core: timeoutCore, failure: timeoutFailure });
    await notifyFromActionFailure({ github: timeoutGithub, context: context(), core: timeoutCore, failure: timeoutFailure });
    const pendingAfterTimeout = await readThreadState({ github: timeoutGithub, context: context(), core: makeCore(), prNum: 42 });
    assert(timedRootRequests === 1 && pendingAfterTimeout.kind === 'pending', 'Feishu ambiguous timeout 保留 pending，same-job 第二步不重发');
  } finally {
    global.fetch = oldFetch;
    process.env = oldEnv;
    fs.rmSync(runnerTemp, { recursive: true, force: true });
  }
}

main().catch((error) => {
  console.error(`FAIL: ${error.stack || error}`);
  process.exitCode = 1;
});
