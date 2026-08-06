// Official Codex code review -> Feishu thread orchestration.
//
// A trusted, append-only PR comment ledger is the durable source of truth. A root send is
// allowed only after a pending generation has been confirmed. final/released terminal records
// supersede that pending claim; the highest generation always wins and never falls back.

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

const FEISHU = 'https://open.feishu.cn/open-apis';
const MARK = '<!-- codex-review-feishu-thread:';
const STATE_MARK = '<!-- codex-review-feishu-state:';
const THREAD_STATE_VERSION = 2;
const RECOVERY_STATE_VERSION = 2;
const MARKER_IO_ATTEMPTS = 3;
const GITHUB_REQUEST_TIMEOUT_MS = 15000;
const GITHUB_RETRY_BASE_MS = 1000;
const MAX_GITHUB_RETRY_DELAY_MS = 30000;
const GITHUB_ATTEMPT_BUDGET_MS = 180000;
const GITHUB_ACTIONS_BOT = 'github-actions[bot]';
const LEGACY_TRUSTED_ASSOCIATIONS = new Set(['OWNER', 'MEMBER', 'COLLABORATOR']);
const ROOT_MISSING_CODES = new Set(['230011', '230110']);
const ROOT_DEFINITELY_NOT_SENT_CODES = new Set([
  '230001', '230002', '230006', '230013', '230018', '230022', '230025',
  '230027', '230028', '230035', '230038', '230054', '230055', '232009',
]);
const STATE_KEYS = ['claimId', 'generation', 'messageId', 'pr', 'repo', 'state', 'version'];
let activeGithubAttemptDeadlineMs = null;

function noopCore() {
  return { info() {}, warning() {}, setFailed() {}, setSecret() {} };
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function errorMessage(error) {
  return (error && error.message) || String(error);
}

function parseRetryAfterMs(error, nowMs = Date.now()) {
  const headers = error && error.response && error.response.headers;
  if (!headers || typeof headers !== 'object') return null;
  const key = Object.keys(headers).find((name) => name.toLowerCase() === 'retry-after');
  let raw = key ? headers[key] : null;
  if (Array.isArray(raw)) [raw] = raw;
  if (raw === null || typeof raw === 'undefined' || String(raw).trim() === '') return null;
  const value = String(raw).trim();
  let delay;
  if (/^\d+(?:\.\d+)?$/.test(value)) delay = Math.max(0, Math.ceil(Number(value) * 1000));
  else {
    const retryAt = Date.parse(value);
    if (!Number.isFinite(retryAt)) return null;
    delay = Math.max(0, retryAt - nowMs);
  }
  return Math.min(delay, MAX_GITHUB_RETRY_DELAY_MS);
}

async function withGithubRetry({
  core = noopCore(),
  label,
  operation,
  attempts = MARKER_IO_ATTEMPTS,
  sleepFn = sleep,
  setFailedOnExhausted = true,
  deadlineMs = activeGithubAttemptDeadlineMs,
  nowFn = Date.now,
}) {
  let lastError = null;
  let stoppedForBudget = false;
  const hasDeadline = Number.isFinite(deadlineMs);
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    const remainingBeforeOperation = hasDeadline ? deadlineMs - nowFn() : Infinity;
    if (remainingBeforeOperation < GITHUB_REQUEST_TIMEOUT_MS) {
      lastError = new Error(`GitHub attempt budget 剩余 ${Math.max(0, remainingBeforeOperation)}ms，不足单次 ${GITHUB_REQUEST_TIMEOUT_MS}ms timeout`);
      stoppedForBudget = true;
      break;
    }
    try {
      return { ok: true, value: await operation(attempt) };
    } catch (error) {
      lastError = error;
      if (attempt < attempts) {
        const retryAfterMs = parseRetryAfterMs(error, nowFn());
        let delayMs = Math.min(
          retryAfterMs === null ? GITHUB_RETRY_BASE_MS * (2 ** (attempt - 1)) : retryAfterMs,
          MAX_GITHUB_RETRY_DELAY_MS,
        );
        if (hasDeadline) {
          const sleepBudgetMs = deadlineMs - nowFn() - GITHUB_REQUEST_TIMEOUT_MS;
          if (sleepBudgetMs <= 0) {
            lastError = new Error(`GitHub attempt budget 已无重试预算；最近错误: ${errorMessage(error)}`);
            stoppedForBudget = true;
            break;
          }
          delayMs = Math.min(delayMs, sleepBudgetMs);
        }
        core.warning(`${label}失败，第 ${attempt} 次重试将在 ${delayMs}ms 后执行: ${errorMessage(error)}`);
        await sleepFn(delayMs);
      }
    }
  }
  const message = stoppedForBudget
    ? `${label}失败，GitHub attempt budget 已截止: ${errorMessage(lastError)}`
    : `${label}失败，已耗尽 ${attempts} 次尝试: ${errorMessage(lastError)}`;
  if (setFailedOnExhausted) core.setFailed(message);
  else core.warning(message);
  return { ok: false, error: lastError };
}

function validFeishuMessageId(value) {
  return typeof value === 'string' && /^om_[A-Za-z0-9_-]{16,80}$/.test(value);
}

function validClaimId(value) {
  return typeof value === 'string' && /^[A-Za-z0-9_-]{16,64}$/.test(value);
}

function validateThreadState(state) {
  if (!state || typeof state !== 'object' || Array.isArray(state)) return false;
  const keys = Object.keys(state).sort();
  if (keys.length !== STATE_KEYS.length || keys.some((key, index) => key !== STATE_KEYS[index])) return false;
  if (state.version !== THREAD_STATE_VERSION) return false;
  if (!['pending', 'final', 'released'].includes(state.state)) return false;
  if (typeof state.repo !== 'string' || !/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(state.repo)) return false;
  if (!Number.isInteger(state.pr) || state.pr <= 0) return false;
  if (!Number.isInteger(state.generation) || state.generation < 1) return false;
  if (!validClaimId(state.claimId)) return false;
  if (state.state === 'final') return validFeishuMessageId(state.messageId);
  return state.messageId === null;
}

function canonicalThreadState(input) {
  const state = {
    version: input.version,
    state: input.state,
    repo: input.repo,
    pr: input.pr,
    generation: input.generation,
    claimId: input.claimId,
    messageId: typeof input.messageId === 'undefined' ? null : input.messageId,
  };
  if (!validateThreadState(state)) throw new Error('invalid codex-review-feishu thread state');
  return state;
}

function base64urlEncode(value) {
  return Buffer.from(value, 'utf8').toString('base64')
    .replace(/=/g, '').replace(/\+/g, '-').replace(/\//g, '_');
}

function base64urlDecode(value) {
  const padding = (4 - (value.length % 4)) % 4;
  return Buffer.from(value.replace(/-/g, '+').replace(/_/g, '/') + '='.repeat(padding), 'base64').toString('utf8');
}

function encodeThreadState(input) {
  const state = canonicalThreadState(input);
  return `${STATE_MARK}${base64urlEncode(JSON.stringify(state))} -->`;
}

function decodeThreadState(body) {
  if (typeof body !== 'string') return null;
  const matches = [...body.matchAll(/<!-- codex-review-feishu-state:([A-Za-z0-9_-]+) -->/g)];
  if (matches.length !== 1) return null;
  try {
    const decoded = JSON.parse(base64urlDecode(matches[0][1]));
    return validateThreadState(decoded) ? decoded : null;
  } catch (_error) {
    return null;
  }
}

function legacyMessageId(body) {
  if (typeof body !== 'string') return null;
  const match = body.match(/^<!-- codex-review-feishu-thread:(om_[A-Za-z0-9_-]{16,80}) -->\n<sub>Codex 代码评审话题锚点（(?:自动维护|根据最近一次成功根消息自动修复)，请勿删除）<\/sub>\s*$/);
  return match ? match[1] : null;
}

function isGithubActionsBot(comment) {
  const login = String(comment && comment.user && comment.user.login || '').toLowerCase();
  const type = String(comment && comment.user && comment.user.type || '').toLowerCase();
  return login === GITHUB_ACTIONS_BOT && type === 'bot';
}

function isTrustedMarkerComment(comment, options = {}) {
  const legacy = typeof options === 'boolean'
    ? options
    : Boolean(options.legacy || (!String(comment && comment.body || '').includes(STATE_MARK)));
  if (isGithubActionsBot(comment)) return true;
  if (!legacy) return false;
  return LEGACY_TRUSTED_ASSOCIATIONS.has(String(comment && comment.author_association || '').toUpperCase());
}

function repositoryName(context) {
  return `${context.repo.owner}/${context.repo.repo}`;
}

async function readThreadMarkerComments({ github, context, core = noopCore(), prNum }) {
  const result = await withGithubRetry({
    core,
    label: '读取 PR 话题状态',
    operation: () => github.paginate(github.rest.issues.listComments, {
      owner: context.repo.owner,
      repo: context.repo.repo,
      issue_number: prNum,
      per_page: 100,
      request: { timeout: GITHUB_REQUEST_TIMEOUT_MS },
    }),
  });
  return result.ok ? result.value : null;
}

function conflictState(core, message, comments = []) {
  core.setFailed(message);
  return { ok: false, kind: 'conflict', comments };
}

async function readThreadState({ github, context, core = noopCore(), prNum }) {
  const comments = await readThreadMarkerComments({ github, context, core, prNum });
  if (!comments) return { ok: false, kind: 'unavailable', comments: [] };
  const repo = repositoryName(context);
  const entries = [];
  for (const comment of comments) {
    const body = String(comment.body || '');
    if (body.includes(STATE_MARK)) {
      if (!isTrustedMarkerComment(comment, { legacy: false })) {
        core.warning(`忽略非 github-actions bot 的 v2 状态 comment_id=${comment.id}`);
        continue;
      }
      const decoded = decodeThreadState(body);
      if (!decoded) return conflictState(core, `可信 v2 状态格式损坏 comment_id=${comment.id}`, comments);
      if (decoded.repo !== repo || decoded.pr !== prNum) {
        return conflictState(core, `可信 v2 状态绑定到其他 repo/PR comment_id=${comment.id}`, comments);
      }
      entries.push({ ...decoded, commentId: comment.id });
      continue;
    }
    if (!body.includes(MARK)) continue;
    const mid = legacyMessageId(body);
    if (!mid) {
      if (isTrustedMarkerComment(comment, { legacy: true })) {
        return conflictState(core, `可信 legacy marker 格式损坏 comment_id=${comment.id}`, comments);
      }
      continue;
    }
    if (!isTrustedMarkerComment(comment, { legacy: true })) {
      core.warning(`忽略非受信作者的 legacy marker comment_id=${comment.id}`);
      continue;
    }
    entries.push({
      version: 1,
      state: 'final',
      repo,
      pr: prNum,
      generation: 0,
      claimId: `legacy-${mid}`,
      messageId: mid,
      commentId: comment.id,
    });
  }
  if (entries.length === 0) return { ok: true, kind: 'none', generation: 0, comments };

  const byGeneration = new Map();
  for (const entry of entries) {
    if (!byGeneration.has(entry.generation)) byGeneration.set(entry.generation, []);
    byGeneration.get(entry.generation).push(entry);
  }
  const reduced = [];
  for (const [generation, generationEntries] of byGeneration.entries()) {
    const claims = [...new Set(generationEntries.map((entry) => entry.claimId))];
    if (claims.length !== 1) {
      return conflictState(core, `generation=${generation} 存在不同 claim，停止通知`, comments);
    }
    const states = new Set(generationEntries.map((entry) => entry.state));
    const finalMids = [...new Set(generationEntries.filter((entry) => entry.state === 'final').map((entry) => entry.messageId))];
    if (finalMids.length > 1 || (states.has('final') && states.has('released'))) {
      return conflictState(core, `generation=${generation} terminal 冲突，停止通知`, comments);
    }
    let kind = 'pending';
    if (states.has('final')) kind = 'final';
    else if (states.has('released')) kind = 'released';
    const exemplar = generationEntries.find((entry) => entry.state === kind) || generationEntries[0];
    reduced.push({
      ...exemplar,
      state: kind,
      messageId: kind === 'final' ? finalMids[0] : null,
      commentIds: generationEntries.map((entry) => entry.commentId),
    });
  }
  reduced.sort((a, b) => b.generation - a.generation);
  const highest = reduced[0];
  return {
    ok: true,
    kind: highest.state,
    generation: highest.generation,
    claimId: highest.claimId,
    messageId: highest.messageId,
    rootMid: highest.messageId,
    state: highest,
    generations: reduced,
    comments,
  };
}

function stateCommentBody(state) {
  const marker = encodeThreadState(state);
  if (state.state === 'final') {
    return `${marker}\n${MARK}${state.messageId} -->\n<sub>Codex 代码评审话题锚点（自动维护，请勿删除）</sub>`;
  }
  return `${marker}\n<sub>Codex 代码评审话题状态：${state.state}（自动维护，请勿删除）</sub>`;
}

async function appendStateAndConfirm({ github, context, core, prNum, expected }) {
  const body = stateCommentBody(expected);
  await withGithubRetry({
    core,
    label: `追加 ${expected.state} 话题状态`,
    setFailedOnExhausted: false,
    operation: () => github.rest.issues.createComment({
      owner: context.repo.owner,
      repo: context.repo.repo,
      issue_number: prNum,
      body,
      request: { timeout: GITHUB_REQUEST_TIMEOUT_MS },
    }),
  });
  const confirmed = await readThreadState({ github, context, core, prNum });
  const matches = confirmed.ok &&
    confirmed.kind === expected.state &&
    confirmed.generation === expected.generation &&
    confirmed.claimId === expected.claimId &&
    (expected.state !== 'final' || confirmed.messageId === expected.messageId);
  if (!matches) {
    core.setFailed(`${expected.state} 状态写入未能通过重读确认`);
    return { ok: false, state: confirmed };
  }
  return { ok: true, state: confirmed };
}

function recoveryFileFor(reviewEventKey) {
  if (!process.env.RUNNER_TEMP) return null;
  const digest = crypto.createHash('sha256').update(reviewEventKey).digest('hex').slice(0, 32);
  return path.join(process.env.RUNNER_TEMP, `codex-review-feishu-${digest}.json`);
}

async function writeRecoveryStateAtomic(file, state) {
  if (!file) throw new Error('RUNNER_TEMP 未设置');
  const temporary = `${file}.${process.pid}.${crypto.randomBytes(6).toString('hex')}.tmp`;
  await fs.promises.mkdir(path.dirname(file), { recursive: true });
  try {
    await fs.promises.writeFile(temporary, `${JSON.stringify(state)}\n`, { encoding: 'utf8', mode: 0o600, flag: 'wx' });
    await fs.promises.rename(temporary, file);
  } finally {
    await fs.promises.unlink(temporary).catch((error) => {
      if (error && error.code !== 'ENOENT') throw error;
    });
  }
}

async function clearRecoveryState(file, core = noopCore()) {
  if (!file) return;
  try {
    await fs.promises.unlink(file);
  } catch (error) {
    if (!error || error.code !== 'ENOENT') core.warning(`清理通知恢复状态失败: ${errorMessage(error)}`);
  }
}

function recoveryRecord({ repo, prNum, reviewEventKey, generation, claimId, stage, messageId = null }) {
  return {
    version: RECOVERY_STATE_VERSION,
    repo,
    pr: prNum,
    reviewEventKey,
    generation,
    claimId,
    stage,
    messageId,
  };
}

async function loadRecoveryState(file, expected, core) {
  if (!file) return { found: false };
  let raw;
  try {
    raw = await fs.promises.readFile(file, 'utf8');
  } catch (error) {
    if (error && error.code === 'ENOENT') return { found: false };
    core.setFailed(`读取通知恢复状态失败: ${errorMessage(error)}`);
    return { found: true, valid: false };
  }
  let state;
  try {
    state = JSON.parse(raw);
  } catch (error) {
    core.setFailed(`通知恢复状态损坏: ${errorMessage(error)}`);
    return { found: true, valid: false };
  }
  const valid = state && state.version === RECOVERY_STATE_VERSION &&
    state.repo === expected.repo && state.pr === expected.pr &&
    state.reviewEventKey === expected.reviewEventKey &&
    Number.isInteger(state.generation) && state.generation >= 1 &&
    validClaimId(state.claimId) &&
    ['reserving', 'pending-confirmed', 'sending', 'root-sent', 'release-needed'].includes(state.stage) &&
    (state.messageId === null || validFeishuMessageId(state.messageId));
  if (!valid) {
    core.setFailed('通知恢复状态与当前事件不匹配');
    return { found: true, valid: false };
  }
  return { found: true, valid: true, state };
}

async function reserveThreadGeneration({
  github,
  context,
  core = noopCore(),
  prNum,
  generation,
  claimId,
  reviewEventKey,
  recoveryFile,
  allowFromFinal = false,
}) {
  if (!recoveryFile) {
    core.setFailed('RUNNER_TEMP 未设置，拒绝在无恢复日志时创建 pending');
    return { ok: false };
  }
  const repo = repositoryName(context);
  const current = await readThreadState({ github, context, core, prNum });
  if (!current.ok) return { ok: false, state: current };
  if (current.kind === 'pending') {
    if (current.generation === generation && current.claimId === claimId) {
      await writeRecoveryStateAtomic(recoveryFile, recoveryRecord({ repo, prNum, reviewEventKey, generation, claimId, stage: 'pending-confirmed' }));
      return { ok: true, state: current, recovered: true };
    }
    core.setFailed(`PR 已有 pending generation=${current.generation}，跨 run 不自动恢复`);
    return { ok: false, state: current };
  }
  if (current.kind === 'final' && !allowFromFinal) {
    core.setFailed('active final 未确认丢失，拒绝创建新 generation');
    return { ok: false, state: current };
  }
  const expectedGeneration = current.kind === 'none' ? 1 : current.generation + 1;
  if (generation !== expectedGeneration) {
    core.setFailed(`generation 非单调递增，expected=${expectedGeneration}, actual=${generation}`);
    return { ok: false, state: current };
  }
  const pending = canonicalThreadState({
    version: THREAD_STATE_VERSION,
    state: 'pending',
    repo,
    pr: prNum,
    generation,
    claimId,
    messageId: null,
  });
  try {
    await writeRecoveryStateAtomic(recoveryFile, recoveryRecord({ repo, prNum, reviewEventKey, generation, claimId, stage: 'reserving' }));
  } catch (error) {
    core.setFailed(`pending 前置恢复日志写入失败: ${errorMessage(error)}`);
    return { ok: false };
  }
  const appended = await appendStateAndConfirm({ github, context, core, prNum, expected: pending });
  if (!appended.ok) return appended;
  try {
    await writeRecoveryStateAtomic(recoveryFile, recoveryRecord({ repo, prNum, reviewEventKey, generation, claimId, stage: 'pending-confirmed' }));
  } catch (error) {
    core.setFailed(`pending 确认恢复日志写入失败: ${errorMessage(error)}`);
    return { ok: false, state: appended.state };
  }
  return appended;
}

async function finalizeThreadGeneration({ github, context, core = noopCore(), prNum, generation, claimId, messageId }) {
  if (!validFeishuMessageId(messageId)) {
    core.setFailed('final message_id 格式非法');
    return { ok: false };
  }
  const current = await readThreadState({ github, context, core, prNum });
  if (!current.ok) return { ok: false, state: current };
  if (current.kind === 'final' && current.generation === generation && current.claimId === claimId && current.messageId === messageId) {
    return { ok: true, state: current, recovered: true };
  }
  if (current.kind !== 'pending' || current.generation !== generation || current.claimId !== claimId) {
    core.setFailed('final 只能覆盖最高 generation 的同一 pending claim');
    return { ok: false, state: current };
  }
  const finalState = canonicalThreadState({
    version: THREAD_STATE_VERSION,
    state: 'final',
    repo: repositoryName(context),
    pr: prNum,
    generation,
    claimId,
    messageId,
  });
  return appendStateAndConfirm({ github, context, core, prNum, expected: finalState });
}

async function releaseThreadGeneration({ github, context, core = noopCore(), prNum, generation, claimId }) {
  const current = await readThreadState({ github, context, core, prNum });
  if (!current.ok) return { ok: false, state: current };
  if (current.kind === 'released' && current.generation === generation && current.claimId === claimId) {
    return { ok: true, state: current, recovered: true };
  }
  if (current.kind !== 'pending' || current.generation !== generation || current.claimId !== claimId) {
    core.setFailed('released 只能覆盖最高 generation 的同一 pending claim');
    return { ok: false, state: current };
  }
  const released = canonicalThreadState({
    version: THREAD_STATE_VERSION,
    state: 'released',
    repo: repositoryName(context),
    pr: prNum,
    generation,
    claimId,
    messageId: null,
  });
  return appendStateAndConfirm({ github, context, core, prNum, expected: released });
}

// Compatibility wrapper for existing callers. New production paths use append-only v2 state.
async function persistThreadMarker({ github, context, core = noopCore(), markerCommentId, prNum, markBody }) {
  const result = await withGithubRetry({
    core,
    label: '兼容回写 legacy 话题标记',
    operation: () => markerCommentId
      ? github.rest.issues.updateComment({
        owner: context.repo.owner,
        repo: context.repo.repo,
        comment_id: markerCommentId,
        body: markBody,
        request: { timeout: GITHUB_REQUEST_TIMEOUT_MS },
      })
      : github.rest.issues.createComment({
        owner: context.repo.owner,
        repo: context.repo.repo,
        issue_number: prNum,
        body: markBody,
        request: { timeout: GITHUB_REQUEST_TIMEOUT_MS },
      }),
  });
  return result.ok;
}

function feishuFailureSummary(result) {
  const status = result && typeof result.status !== 'undefined' ? result.status : 'UNKNOWN';
  const json = result && result.json || {};
  const code = typeof json.code !== 'undefined' ? json.code : 'UNKNOWN';
  const requestId = json.request_id || json.data && json.data.request_id;
  return `HTTP ${status}, code=${code}${requestId ? `, request_id=${requestId}` : ''}`;
}

function shouldRecreateRootOnReplyFailure(result) {
  const code = String(result && result.json && result.json.code || '');
  return ROOT_MISSING_CODES.has(code);
}

function isKnownRootSendFailure(result) {
  const status = Number(result && result.status);
  const code = String(result && result.json && result.json.code || '');
  return status === 400 && ROOT_DEFINITELY_NOT_SENT_CODES.has(code);
}

async function feishu(apiPath, method, body, token) {
  const headers = { 'Content-Type': 'application/json' };
  if (token) headers.Authorization = `Bearer ${token}`;
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 15000);
  try {
    const resp = await fetch(`${FEISHU}${apiPath}`, {
      method,
      headers,
      body: body ? JSON.stringify(body) : undefined,
      signal: ctrl.signal,
    });
    const text = await resp.text();
    let json;
    try { json = JSON.parse(text); } catch (_error) { json = {}; }
    return { ok: resp.ok, status: resp.status, json, text };
  } catch (error) {
    const message = error && error.name === 'AbortError' ? 'feishu API timeout (15s)' : errorMessage(error);
    return { ok: false, status: 0, json: { code: 'NETWORK_ERROR', msg: message }, text: message };
  } finally {
    clearTimeout(timer);
  }
}

function readFeishuEnv(core) {
  const appId = process.env.CODEREVIEW_FEISHU_APP_ID;
  const appSecret = process.env.CODEREVIEW_FEISHU_APP_SECRET;
  const chatId = process.env.CODEREVIEW_FEISHU_CHAT_ID;
  for (const secret of [appId, appSecret, chatId]) if (secret) core.setSecret(secret);
  if (!appId || !appSecret || !chatId) {
    core.warning('未配置 Feishu secret，跳过通知');
    return null;
  }
  return { appId, appSecret, chatId };
}

function deterministicClaimId(repo, prNum, generation, reviewEventKey) {
  return crypto.createHash('sha256').update(`claim:v2:${repo}#${prNum}:${generation}:${reviewEventKey}`).digest('hex').slice(0, 32);
}

function rootUuid(repo, prNum, generation, claimId) {
  return crypto.createHash('sha256').update(`root:v2:${repo}#${prNum}:${generation}:${claimId}`).digest('hex').slice(0, 32);
}

function eventUuid(reviewEventKey, kind) {
  return crypto.createHash('sha256').update(`${reviewEventKey}:${kind}`).digest('hex').slice(0, 32);
}

async function recoverSameJob({ github, context, core, prNum, reviewEventKey, recoveryFile }) {
  const repo = repositoryName(context);
  const loaded = await loadRecoveryState(recoveryFile, { repo, pr: prNum, reviewEventKey }, core);
  if (!loaded.found) return { handled: false };
  if (!loaded.valid) return { handled: true, ok: false };
  const journal = loaded.state;
  const current = await readThreadState({ github, context, core, prNum });
  if (!current.ok) return { handled: true, ok: false };
  const expectedClaimId = deterministicClaimId(repo, prNum, journal.generation, reviewEventKey);
  if (journal.claimId !== expectedClaimId) {
    core.setFailed('恢复日志 claim 与当前 event 不一致');
    return { handled: true, ok: false };
  }
  const safeReservingPredecessor = journal.stage === 'reserving' && (
    (current.kind === 'none' && current.generation === 0) ||
    (['final', 'released'].includes(current.kind) && current.generation === journal.generation - 1)
  );
  if (safeReservingPredecessor) {
    return { handled: false, retryReservation: journal, predecessor: current };
  }
  if (current.generation !== journal.generation || current.claimId !== journal.claimId) {
    core.setFailed('恢复日志与最高 generation/claim 不一致');
    return { handled: true, ok: false };
  }
  if (current.kind === 'final') {
    await clearRecoveryState(recoveryFile, core);
    return { handled: true, ok: true };
  }
  if (current.kind === 'released') {
    await clearRecoveryState(recoveryFile, core);
    core.setFailed('前次根消息明确未发送，generation 已 released');
    return { handled: true, ok: false };
  }
  if (current.kind !== 'pending') {
    core.setFailed('恢复日志存在但最高状态不是 pending');
    return { handled: true, ok: false };
  }
  if (journal.stage === 'root-sent') {
    const finalized = await finalizeThreadGeneration({
      github, context, core, prNum,
      generation: journal.generation,
      claimId: journal.claimId,
      messageId: journal.messageId,
    });
    if (finalized.ok) await clearRecoveryState(recoveryFile, core);
    return { handled: true, ok: finalized.ok };
  }
  if (journal.stage === 'release-needed') {
    const released = await releaseThreadGeneration({
      github, context, core, prNum,
      generation: journal.generation,
      claimId: journal.claimId,
    });
    if (released.ok) await clearRecoveryState(recoveryFile, core);
    core.setFailed('前次根消息明确发送失败');
    return { handled: true, ok: false };
  }
  if (journal.stage === 'sending') {
    core.setFailed('前次 Feishu 根消息结果不确定，保留 pending 并停止重发');
    return { handled: true, ok: false };
  }
  if (journal.stage === 'reserving') {
    await writeRecoveryStateAtomic(recoveryFile, { ...journal, stage: 'pending-confirmed' });
  }
  return { handled: false, resume: { ...journal, stage: 'pending-confirmed' } };
}

async function postToThread({ github, context, core }, { cls, prData, env, dedupeKey }) {
  const { buildCard, interactiveCardContent } = require('./report');
  const repo = repositoryName(context);
  const prNum = prData.number;
  const reviewEventKey = `${repo}#${prNum}:${dedupeKey}`;
  const recoveryFile = recoveryFileFor(reviewEventKey);
  const recovery = await recoverSameJob({ github, context, core, prNum, reviewEventKey, recoveryFile });
  if (recovery.handled) return;

  const cardContentOrFail = (card, label) => {
    try { return interactiveCardContent(card); }
    catch (error) { core.setFailed(`生成飞书卡片失败(${label}): ${errorMessage(error)}`); return null; }
  };

  let current;
  let generation;
  let claimId;
  let content;
  let allowFromFinal = false;
  if (recovery.resume) {
    current = await readThreadState({ github, context, core, prNum });
    if (!current.ok || current.kind !== 'pending' ||
        current.generation !== recovery.resume.generation || current.claimId !== recovery.resume.claimId) {
      core.setFailed('same-job pending 恢复与 PR 状态不一致');
      return;
    }
    generation = current.generation;
    claimId = current.claimId;
    content = cardContentOrFail(buildCard(context.repo.repo, prData, cls, { atAuthor: true, isReply: false }), 'root-recovery');
    if (!content) return;
  } else if (recovery.retryReservation) {
    current = recovery.predecessor;
    generation = recovery.retryReservation.generation;
    claimId = recovery.retryReservation.claimId;
    allowFromFinal = current.kind === 'final';
    content = cardContentOrFail(buildCard(context.repo.repo, prData, cls, { atAuthor: true, isReply: false }), 'root-reservation-retry');
    if (!content) return;
  } else {
    current = await readThreadState({ github, context, core, prNum });
    if (!current.ok) return;
    if (current.kind === 'pending') {
      core.setFailed(`检测到跨 run pending generation=${current.generation}，必须 trusted repair 后再通知`);
      return;
    }
  }

  const tokResp = await feishu('/auth/v3/tenant_access_token/internal', 'POST', { app_id: env.appId, app_secret: env.appSecret });
  const token = tokResp.json.tenant_access_token;
  if (!token) {
    core.setFailed(`取 tenant_access_token 失败: ${feishuFailureSummary(tokResp)}`);
    return;
  }
  core.setSecret(token);

  if (!recovery.resume && !recovery.retryReservation && current.kind === 'final') {
    const replyContent = cardContentOrFail(buildCard(context.repo.repo, prData, cls, { atAuthor: false, isReply: true }), 'reply');
    if (!replyContent) return;
    const reply = await feishu(`/im/v1/messages/${encodeURIComponent(current.messageId)}/reply`, 'POST', {
      msg_type: 'interactive',
      content: replyContent,
      reply_in_thread: true,
      uuid: eventUuid(reviewEventKey, `reply:${current.messageId}`),
    }, token);
    if (reply.ok && reply.json.code === 0) {
      core.info('话题回复成功');
      return;
    }
    if (!shouldRecreateRootOnReplyFailure(reply)) {
      core.setFailed(`话题回复失败，未创建新 generation: ${feishuFailureSummary(reply)}`);
      return;
    }
    core.warning(`根消息明确已撤回/删除，创建新 generation: ${feishuFailureSummary(reply)}`);
    allowFromFinal = true;
  }

  if (!recovery.resume) {
    if (!recovery.retryReservation) {
      generation = current.kind === 'none' ? 1 : current.generation + 1;
      claimId = deterministicClaimId(repo, prNum, generation, reviewEventKey);
      content = cardContentOrFail(buildCard(context.repo.repo, prData, cls, { atAuthor: true, isReply: false }), 'root');
      if (!content) return;
    }
    const reserved = await reserveThreadGeneration({
      github, context, core, prNum, generation, claimId, reviewEventKey, recoveryFile, allowFromFinal,
    });
    if (!reserved.ok) return;
  }

  try {
    await writeRecoveryStateAtomic(recoveryFile, recoveryRecord({
      repo, prNum, reviewEventKey, generation, claimId, stage: 'sending',
    }));
  } catch (error) {
    core.setFailed(`发送前原子恢复日志写入失败: ${errorMessage(error)}`);
    return;
  }

  const result = await feishu('/im/v1/messages?receive_id_type=chat_id', 'POST', {
    receive_id: env.chatId,
    msg_type: 'interactive',
    content,
    uuid: rootUuid(repo, prNum, generation, claimId),
  }, token);
  const newMid = result.json.data && result.json.data.message_id;
  if (result.ok && result.json.code === 0 && validFeishuMessageId(newMid)) {
    core.info(`根消息发送成功 message_id=${newMid}`);
    try {
      await writeRecoveryStateAtomic(recoveryFile, recoveryRecord({
        repo, prNum, reviewEventKey, generation, claimId, stage: 'root-sent', messageId: newMid,
      }));
    } catch (error) {
      core.setFailed(`根消息成功但 message_id 原子落盘失败，保留 pending: ${errorMessage(error)}`);
      return;
    }
    const finalized = await finalizeThreadGeneration({
      github, context, core, prNum, generation, claimId, messageId: newMid,
    });
    if (finalized.ok) await clearRecoveryState(recoveryFile, core);
    return;
  }

  if (isKnownRootSendFailure(result)) {
    try {
      await writeRecoveryStateAtomic(recoveryFile, recoveryRecord({
        repo, prNum, reviewEventKey, generation, claimId, stage: 'release-needed',
      }));
    } catch (error) {
      core.warning(`known failure 恢复日志写入失败: ${errorMessage(error)}`);
    }
    const released = await releaseThreadGeneration({ github, context, core, prNum, generation, claimId });
    if (released.ok) await clearRecoveryState(recoveryFile, core);
    core.setFailed(`Feishu 根消息明确未发送: ${feishuFailureSummary(result)}`);
    return;
  }

  core.setFailed(`Feishu 根消息结果不确定，保留 pending: ${feishuFailureSummary(result)}`);
}

function normalizeResolveArgs(githubOrOptions, contextArg, coreArg) {
  if (githubOrOptions && githubOrOptions.github) {
    return {
      github: githubOrOptions.github,
      context: githubOrOptions.context,
      core: githubOrOptions.core || noopCore(),
    };
  }
  return { github: githubOrOptions, context: contextArg, core: coreArg || noopCore() };
}

async function resolvePrNumber(githubOrOptions, contextArg, coreArg) {
  const { github, context, core } = normalizeResolveArgs(githubOrOptions, contextArg, coreArg);
  const direct = context.payload.pull_request && context.payload.pull_request.number;
  if (Number.isInteger(direct) && direct > 0) return direct;
  const checkRun = context.payload.check_run;
  if (!checkRun) {
    core.setFailed('事件不包含 pull_request 或 check_run');
    return null;
  }
  let candidates = [...new Set((checkRun.pull_requests || []).map((pr) => Number(pr.number)).filter((number) => Number.isInteger(number) && number > 0))];
  if (candidates.length === 0 && checkRun.head_sha) {
    const associated = await withGithubRetry({
      core,
      label: '按 check SHA 解析 PR',
      operation: () => github.rest.repos.listPullRequestsAssociatedWithCommit({
        owner: context.repo.owner,
        repo: context.repo.repo,
        commit_sha: checkRun.head_sha,
        request: { timeout: GITHUB_REQUEST_TIMEOUT_MS },
      }),
    });
    if (!associated.ok) return null;
    candidates = [...new Set((associated.value.data || [])
      .filter((pr) => pr.state == null || pr.state === 'open')
      .filter((pr) => !pr.head?.sha || pr.head.sha === checkRun.head_sha)
      .map((pr) => Number(pr.number))
      .filter((number) => Number.isInteger(number) && number > 0))];
  }
  if (candidates.length !== 1) {
    core.setFailed(`Codex 事件必须唯一关联一个 PR，实际=${candidates.join(',') || 'none'}`);
    return null;
  }
  return candidates[0];
}

async function resolvePr(github, context, hint, core = noopCore()) {
  if (github && github.github) {
    const options = github;
    return resolvePr(options.github, options.context, options.hint || options.pr || options.prNumber || options.prNum, options.core || noopCore());
  }
  let number = typeof hint === 'number' ? hint : hint && Number(hint.number || hint);
  if (!Number.isInteger(number) || number <= 0) number = await resolvePrNumber(github, context, core);
  if (!number) return null;
  const result = await withGithubRetry({
    core,
    label: `读取 PR #${number}`,
    operation: () => github.rest.pulls.get({
      owner: context.repo.owner,
      repo: context.repo.repo,
      pull_number: number,
      request: { timeout: GITHUB_REQUEST_TIMEOUT_MS },
    }),
  });
  if (!result.ok) return null;
  const full = result.value.data;
  return {
    number: full.number,
    title: full.title,
    url: full.html_url,
    author: full.user.login,
    base: full.base.ref,
    head: full.head.ref,
    state: full.state,
  };
}

function hasCodexName(value) {
  return /\bcodex\b/i.test(String(value || ''));
}

function isCodexUser(user) {
  const login = String(user && user.login || '').toLowerCase();
  const type = String(user && user.type || '').toLowerCase();
  return hasCodexName(login) && (!type || type === 'bot' || /\[bot\]$/.test(login));
}

function isCodexCheckRun(checkRun) {
  if (!checkRun) return false;
  return [
    checkRun.name,
    checkRun.check_suite && checkRun.check_suite.app && checkRun.check_suite.app.name,
    checkRun.app && checkRun.app.name,
    checkRun.app && checkRun.app.slug,
    checkRun.app && checkRun.app.owner && checkRun.app.owner.login,
  ].some(hasCodexName);
}

function isCodexPullRequestReview(review) {
  return Boolean(review && isCodexUser(review.user));
}

async function listCommentsForReview(github, context, core, prNum, reviewId) {
  const result = await withGithubRetry({
    core,
    label: '读取 Codex review comments',
    operation: () => github.paginate(github.rest.pulls.listReviewComments, {
      owner: context.repo.owner,
      repo: context.repo.repo,
      pull_number: prNum,
      per_page: 100,
      request: { timeout: GITHUB_REQUEST_TIMEOUT_MS },
    }),
  });
  if (!result.ok) return null;
  return result.value.filter((comment) => String(comment.pull_request_review_id || comment.review_id || '') === String(reviewId || ''));
}

async function resolvedPrForNotification(github, context, core) {
  const envNumber = Number(process.env.RESOLVED_PR_NUMBER);
  let number = Number.isInteger(envNumber) && envNumber > 0 ? envNumber : null;
  if (!number) number = await resolvePrNumber(github, context, core);
  if (!number) return null;
  const eventNumber = context.payload.pull_request && context.payload.pull_request.number;
  if (eventNumber && Number(eventNumber) !== number) {
    core.setFailed(`resolved PR #${number} 与 event PR #${eventNumber} 不一致`);
    return null;
  }
  return resolvePr(github, context, { number }, core);
}

async function notifyFromCheckRunWithinBudget({ github, context, core }) {
  const { classifyCheckRun, shouldNotify } = require('./report');
  const checkRun = context.payload.check_run;
  if (!isCodexCheckRun(checkRun)) { core.info('非 Codex check_run，跳过'); return; }
  const cls = classifyCheckRun(checkRun, process.env);
  if (!shouldNotify(cls)) { core.info(`评审 verdict=${cls.verdict}，不通知`); return; }
  const prData = await resolvedPrForNotification(github, context, core);
  if (!prData) return;
  if (prData.base !== 'main' || prData.state !== 'open') { core.info('PR 非 OPEN main，跳过'); return; }
  const env = readFeishuEnv(core); if (!env) return;
  const dedupeKey = `${checkRun.id || checkRun.head_sha}:${checkRun.head_sha}:${checkRun.completed_at || checkRun.updated_at || ''}`;
  await postToThread({ github, context, core }, { cls, prData, env, dedupeKey });
}

async function notifyFromPullRequestReviewWithinBudget({ github, context, core }) {
  const { classifyPullRequestReview, shouldNotify } = require('./report');
  const review = context.payload.review;
  if (!isCodexPullRequestReview(review)) { core.info('非 Codex review，跳过'); return; }
  const prData = await resolvedPrForNotification(github, context, core);
  if (!prData) return;
  if (prData.base !== 'main' || prData.state !== 'open') { core.info('PR 非 OPEN main，跳过'); return; }
  const comments = await listCommentsForReview(github, context, core, prData.number, review && review.id);
  if (!comments) return;
  const cls = classifyPullRequestReview(review, comments, process.env);
  if (!shouldNotify(cls)) { core.info(`评审 verdict=${cls.verdict}，不通知`); return; }
  const env = readFeishuEnv(core); if (!env) return;
  const dedupeKey = `review:${review.id || ''}:${review.submitted_at || review.updated_at || ''}:${comments.length}`;
  await postToThread({ github, context, core }, { cls, prData, env, dedupeKey });
}

async function withGithubAttemptBudget(operation, options = {}) {
  const previousDeadline = activeGithubAttemptDeadlineMs;
  const nowFn = options.nowFn || Date.now;
  const requestedDeadline = Number.isFinite(options.deadlineMs)
    ? options.deadlineMs
    : nowFn() + GITHUB_ATTEMPT_BUDGET_MS;
  activeGithubAttemptDeadlineMs = Number.isFinite(previousDeadline)
    ? previousDeadline
    : requestedDeadline;
  try {
    return await operation();
  } finally {
    activeGithubAttemptDeadlineMs = previousDeadline;
  }
}

async function notifyFromCheckRun(args) {
  return withGithubAttemptBudget(() => notifyFromCheckRunWithinBudget(args));
}

async function notifyFromPullRequestReview(args) {
  return withGithubAttemptBudget(() => notifyFromPullRequestReviewWithinBudget(args));
}

async function notifyFromOfficialCodexEvent(args) {
  return withGithubAttemptBudget(async () => {
    const { github, context, core } = args;
    if (context.eventName === 'check_run') return notifyFromCheckRun({ github, context, core });
    if (context.eventName === 'pull_request_review') return notifyFromPullRequestReview({ github, context, core });
    core.info(`event=${context.eventName} 不支持，跳过`);
  });
}

async function notifyFromActionResult(args) {
  return withGithubAttemptBudget(async () => {
    const { github, context, core, result } = args;
    const { classifyActionResult, shouldNotify } = require('./report');
    const cls = classifyActionResult(result, process.env);
    if (!shouldNotify(cls)) return;
    const prData = await resolvedPrForNotification(github, context, core);
    if (!prData || prData.base !== 'main' || prData.state !== 'open') return;
    const env = readFeishuEnv(core); if (!env) return;
    const dedupeKey = `action:${context.sha}:${result && result.run_id || process.env.GITHUB_RUN_ID || ''}`;
    await postToThread({ github, context, core }, { cls, prData, env, dedupeKey });
  });
}

async function notifyFromActionFailure(args) {
  return withGithubAttemptBudget(async () => {
    const { github, context, core, failure } = args;
    const { classifyActionFailure, shouldNotify } = require('./report');
    const cls = classifyActionFailure(failure || {});
    if (!shouldNotify(cls)) return;
    const prData = await resolvedPrForNotification(github, context, core);
    if (!prData || prData.base !== 'main' || prData.state !== 'open') return;
    const env = readFeishuEnv(core); if (!env) return;
    const runId = failure && failure.runId || process.env.GITHUB_RUN_ID || '';
    const dedupeKey = `action-failure:${context.sha}:${runId}:${cls.reason}`;
    await postToThread({ github, context, core }, { cls, prData, env, dedupeKey });
  });
}

module.exports = {
  notifyFromOfficialCodexEvent,
  notifyFromCheckRun,
  notifyFromPullRequestReview,
  notifyFromActionResult,
  notifyFromActionFailure,
  validFeishuMessageId,
  shouldRecreateRootOnReplyFailure,
  isCodexCheckRun,
  isCodexPullRequestReview,
  GITHUB_REQUEST_TIMEOUT_MS,
  GITHUB_ATTEMPT_BUDGET_MS,
  MAX_GITHUB_RETRY_DELAY_MS,
  parseRetryAfterMs,
  withGithubAttemptBudget,
  withGithubRetry,
  readThreadMarkerComments,
  persistThreadMarker,
  MARK,
  STATE_MARK,
  encodeThreadState,
  decodeThreadState,
  isTrustedMarkerComment,
  readThreadState,
  resolvePrNumber,
  resolvePr,
  reserveThreadGeneration,
  finalizeThreadGeneration,
  releaseThreadGeneration,
};
