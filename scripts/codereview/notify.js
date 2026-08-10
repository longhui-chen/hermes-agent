// Official Codex code review -> Feishu thread orchestration.
//
// Trusted PR comments provide three append-only ledgers:
// 1. immutable GitHub event references, written before per-PR concurrency;
// 2. Feishu thread state; and
// 3. per-event delivery state, including ambiguous-send recovery.

// The queue prevents GitHub Actions' single pending concurrency slot from dropping an event.
// The delivery token in every card lets a later run reconcile an accepted Feishu request
// without sending the event a second time.

const crypto = require('crypto');
const zlib = require('zlib');

const FEISHU = 'https://open.feishu.cn/open-apis';
const MARK = '<!-- codex-review-feishu-thread:';
const STATE_MARK = '<!-- codex-review-feishu-state:';
const REPAIR_MARK = '<!-- codex-review-feishu-repair:';
const THREAD_REPAIR_MARK = '<!-- codex-review-feishu-thread-repair:';
const DELIVERY_REPAIR_MARK = '<!-- codex-review-feishu-delivery-repair:';
const DELIVERY_TOMBSTONE_MARK = '<!-- codex-review-feishu-delivery-tombstone:';
const DRAIN_CURSOR_MARK = '<!-- codex-review-feishu-drain-cursor:';
const CHECKPOINT_MARK = '<!-- codex-review-feishu-checkpoint:';
const BOOTSTRAP_MARK = '<!-- codex-review-feishu-bootstrap-progress:';
const EVENT_MARK = '<!-- codex-review-feishu-event:';
const DELIVERY_MARK = '<!-- codex-review-feishu-delivery:';
const THREAD_STATE_VERSION = 2;
const REPAIR_VERSION = 1;
const EVENT_VERSION = 1;
const DELIVERY_VERSION = 1;
const MARKER_IO_ATTEMPTS = 3;
const GITHUB_REQUEST_TIMEOUT_MS = 15000;
const COMMENT_MINIMIZE_TIMEOUT_MS = 5000;
const GITHUB_RETRY_BASE_MS = 1000;
const MAX_GITHUB_RETRY_DELAY_MS = 30000;
const GITHUB_ATTEMPT_BUDGET_MS = 180000;
// A bootstrap scan must never spend the final two request slots: they are reserved for
// persisting the last completely absorbed page and confirming a lost create response.
// The extra 5s covers JS/runner overhead after full 15s request timeouts.
const BOOTSTRAP_PROGRESS_RESERVE_MS = (2 * GITHUB_REQUEST_TIMEOUT_MS) + 5000;
const FEISHU_REQUEST_TIMEOUT_MS = 15000;
const FEISHU_RATE_LIMIT_MAX_ATTEMPTS = 3;
const FEISHU_RATE_LIMIT_MAX_INLINE_DELAY_MS = 30000;
const FEISHU_RATE_LIMIT_MAX_RETRY_AFTER_MS = 60 * 60 * 1000;
const FEISHU_RATE_LIMIT_PERSIST_RESERVE_MS = 2 * GITHUB_REQUEST_TIMEOUT_MS + 5000;
const DRAIN_BATCH_SIZE = 5;
const DRAIN_SCAN_LIMIT = 50;
const COMMENT_PAGE_SIZE = 100;
const COMMENT_MIGRATION_MAX_PAGES = 5;
const COMMENT_INCREMENTAL_MAX_PAGES = 3;
const CHECKPOINT_MAX_ENTRIES = 2000;
const CHECKPOINT_MAX_BYTES = 60000;
const CHECKPOINT_MAX_INFLATED_BYTES = 2 * 1024 * 1024;
const HISTORY_PAGE_SIZE = 50;
const HISTORY_MAX_PAGES = 5;
const HISTORY_CLOCK_SKEW_SECONDS = 120;
const HISTORY_AFTER_SEND_SECONDS = 600;
const HISTORY_RETRY_DELAY_MS = 30000;
const HISTORY_MAX_BACKOFF_MS = 15 * 60 * 1000;
const HISTORY_MAX_ATTEMPTS = 8;
const HISTORY_MAX_AGE_MS = 24 * 60 * 60 * 1000;
const DEFINITE_SEND_MAX_ATTEMPTS = 3;
const WATCHDOG_PR_PAGE_SIZE = 100;
const WATCHDOG_MAX_PRS_PER_RUN = 40;
const WATCHDOG_MAX_DISPATCHES = 10;
const WATCHDOG_SHARDS = 1;
const WATCHDOG_SLOT_MS = 5 * 60 * 1000;
const GITHUB_ACTIONS_BOT = 'github-actions[bot]';
const CODEX_REVIEW_USER_ID = '199175422';
const CODEX_REVIEW_LOGIN = 'chatgpt-codex-connector[bot]';
const CODEX_CHECK_APP_SLUG = 'chatgpt-codex-connector';
const LEGACY_TRUSTED_ASSOCIATIONS = new Set(['OWNER', 'MEMBER', 'COLLABORATOR']);
const ROOT_MISSING_CODES = new Set(['230011', '230110']);
const ROOT_DEFINITELY_NOT_SENT_CODES = new Set([
  '230001', '230002', '230006', '230013', '230018', '230022', '230025',
  '230027', '230028', '230035', '230038', '230054', '230055', '232009',
]);
const STATE_KEYS = ['claimId', 'generation', 'messageId', 'pr', 'repo', 'state', 'version'];
const REPAIR_KEYS = ['claimId', 'generation', 'keyId', 'legacyHash', 'messageId', 'pr', 'repo', 'runId', 'signature', 'version'];
const EVENT_KEYS = ['createdAt', 'eventId', 'eventKey', 'eventType', 'headSha', 'pr', 'repo', 'version'];
const DELIVERY_KEYS = [
  'attempt', 'candidateMessageIds', 'eventKey', 'historyAttempts', 'messageId', 'mode', 'nextCheckAt', 'pr',
  'reason', 'repo', 'sentAt', 'state', 'targetRoot', 'threadClaimId',
  'threadGeneration', 'token', 'version',
];
const DELIVERY_REPAIR_KEYS = [
  'action', 'candidateHash', 'eventKey', 'keyId', 'messageId', 'operator', 'pr',
  'priorManualHash', 'priorReason', 'repo', 'runId', 'signature', 'version',
];
const DELIVERY_TOMBSTONE_KEYS = [
  'createdAt', 'eventCommentId', 'eventId', 'eventKey', 'eventType', 'headSha',
  'pr', 'repo', 'state', 'version',
];
const THREAD_REPAIR_KEYS = [
  'action', 'keyId', 'messageId', 'operator', 'pr', 'priorThreadHash', 'repo', 'runId', 'signature', 'version',
];
const DRAIN_CURSOR_KEYS = ['lastCommentId', 'nextEventKey', 'pr', 'repo', 'version'];
let activeGithubAttemptDeadlineMs = null;

function noopCore() {
  return { info() {}, warning() {}, setFailed() {}, setSecret() {}, setOutput() {} };
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function errorMessage(error) {
  return (error && error.message) || String(error);
}

function sortedKeysEqual(value, keys) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
  const actual = Object.keys(value).sort();
  return actual.length === keys.length && actual.every((key, index) => key === keys[index]);
}

function validRepo(value) {
  return typeof value === 'string' && /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(value);
}

function validIsoTime(value) {
  return typeof value === 'string' && /Z$/.test(value) && Number.isFinite(Date.parse(value));
}

function validSha(value) {
  return typeof value === 'string' && /^[a-fA-F0-9]{7,64}$/.test(value);
}

function validEventKey(value) {
  return typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);
}

function parseRetryAfterHeadersMs(headers, nowMs = Date.now(), maxDelayMs = MAX_GITHUB_RETRY_DELAY_MS) {
  if (!headers || typeof headers !== 'object') return null;
  let raw = typeof headers.get === 'function' ? headers.get('retry-after') : null;
  if (raw === null || typeof raw === 'undefined') {
    const key = Object.keys(headers).find((name) => name.toLowerCase() === 'retry-after');
    raw = key ? headers[key] : null;
  }
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
  return Math.min(delay, maxDelayMs);
}

function parseRetryAfterMs(error, nowMs = Date.now()) {
  return parseRetryAfterHeadersMs(error && error.response && error.response.headers, nowMs);
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
    const remaining = hasDeadline ? deadlineMs - nowFn() : Infinity;
    if (remaining < GITHUB_REQUEST_TIMEOUT_MS) {
      lastError = new Error(`GitHub attempt budget 剩余 ${Math.max(0, remaining)}ms，不足单次 ${GITHUB_REQUEST_TIMEOUT_MS}ms timeout`);
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

async function withGithubAttemptBudget(operation, options = {}) {
  const previousDeadline = activeGithubAttemptDeadlineMs;
  const nowFn = options.nowFn || Date.now;
  const requestedDeadline = Number.isFinite(options.deadlineMs)
    ? options.deadlineMs
    : nowFn() + GITHUB_ATTEMPT_BUDGET_MS;
  activeGithubAttemptDeadlineMs = Number.isFinite(previousDeadline) ? previousDeadline : requestedDeadline;
  try {
    return await operation();
  } finally {
    activeGithubAttemptDeadlineMs = previousDeadline;
  }
}

function hasGithubRequestBudget(nowMs = Date.now(), reserveMs = 0) {
  return !Number.isFinite(activeGithubAttemptDeadlineMs) ||
    activeGithubAttemptDeadlineMs - nowMs >= GITHUB_REQUEST_TIMEOUT_MS + reserveMs;
}

async function githubGraphqlWithTimeout(github, query, variables = {}, timeoutMs = GITHUB_REQUEST_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await github.graphql(query, {
      ...variables,
      request: { ...(variables.request || {}), timeout: timeoutMs, signal: controller.signal },
    });
  } catch (error) {
    if (!controller.signal.aborted) throw error;
    const timeoutError = new Error(`GitHub GraphQL timeout (${timeoutMs}ms)`);
    timeoutError.name = 'AbortError';
    timeoutError.cause = error;
    throw timeoutError;
  } finally {
    clearTimeout(timer);
  }
}

// PR comments are the durable source of truth for the notifier, but they are
// implementation details rather than review conversation. Minimize only after
// GitHub has confirmed the comment write. This mutation is deliberately
// best-effort and never retried: a presentation-layer failure must not replay a
// createComment operation or fail Feishu delivery.
async function minimizeLedgerCommentBestEffort({
  github, core = noopCore(), comment, label = '折叠 Codex 通知账本评论', nowFn = Date.now,
}) {
  const subjectId = String(comment && (comment.node_id || comment.nodeId) || '').trim();
  if (!subjectId || typeof github.graphql !== 'function') {
    core.warning(`${label}跳过：GitHub comment node_id 或 GraphQL client 不可用`);
    return false;
  }
  // Preserve enough outer attempt budget for one normal 15s state request.
  if (!hasGithubRequestBudget(nowFn(), COMMENT_MINIMIZE_TIMEOUT_MS)) {
    core.warning(`${label}跳过：GitHub attempt budget 不足`);
    return false;
  }
  try {
    const result = await githubGraphqlWithTimeout(github, `mutation($subjectId:ID!){
      minimizeComment(input:{subjectId:$subjectId,classifier:OUTDATED}){
        minimizedComment{isMinimized minimizedReason}
      }
    }`, { subjectId }, COMMENT_MINIMIZE_TIMEOUT_MS);
    const minimized = result && result.minimizeComment && result.minimizeComment.minimizedComment;
    if (!minimized || minimized.isMinimized !== true) {
      core.warning(`${label}未获 GitHub 最小化确认`);
      return false;
    }
    return true;
  } catch (error) {
    core.warning(`${label}失败，保留可见评论且继续主流程: ${errorMessage(error)}`);
    return false;
  }
}

function validFeishuMessageId(value) {
  return typeof value === 'string' && /^om_[A-Za-z0-9_-]{16,80}$/.test(value);
}

function validClaimId(value) {
  return typeof value === 'string' && /^[A-Za-z0-9_-]{16,64}$/.test(value);
}

function base64urlEncode(value) {
  return Buffer.from(value, 'utf8').toString('base64')
    .replace(/=/g, '').replace(/\+/g, '-').replace(/\//g, '_');
}

function base64urlDecode(value) {
  const padding = (4 - (value.length % 4)) % 4;
  return Buffer.from(value.replace(/-/g, '+').replace(/_/g, '/') + '='.repeat(padding), 'base64').toString('utf8');
}

function encodeMarker(prefix, value) {
  return `${prefix}${base64urlEncode(JSON.stringify(value))} -->`;
}

function decodeSingleMarker(body, markerName) {
  if (typeof body !== 'string') return null;
  const escaped = markerName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const matches = [...body.matchAll(new RegExp(`<!-- ${escaped}:([A-Za-z0-9_-]+) -->`, 'g'))];
  if (matches.length !== 1) return null;
  try {
    return JSON.parse(base64urlDecode(matches[0][1]));
  } catch (_error) {
    return null;
  }
}

function validateThreadState(state) {
  if (!sortedKeysEqual(state, STATE_KEYS)) return false;
  if (state.version !== THREAD_STATE_VERSION) return false;
  if (!['pending', 'final', 'released'].includes(state.state)) return false;
  if (!validRepo(state.repo) || !Number.isInteger(state.pr) || state.pr <= 0) return false;
  if (!Number.isInteger(state.generation) || state.generation < 1 || !validClaimId(state.claimId)) return false;
  return state.state === 'final' ? validFeishuMessageId(state.messageId) : state.messageId === null;
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

function encodeThreadState(input) {
  return encodeMarker(STATE_MARK, canonicalThreadState(input));
}

function decodeThreadState(body) {
  const decoded = decodeSingleMarker(body, 'codex-review-feishu-state');
  return validateThreadState(decoded) ? decoded : null;
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

function canonicalCheckpoint(input) {
  const checkpoint = {
    version: input.version,
    revision: input.revision,
    repo: input.repo,
    pr: input.pr,
    highCommentId: input.highCommentId,
    highCreatedAt: input.highCreatedAt,
    parentHash: typeof input.parentHash === 'undefined' ? null : input.parentHash,
    entries: input.entries,
  };
  if (checkpoint.version !== 1 || !Number.isInteger(checkpoint.revision) || checkpoint.revision < 1 ||
      !validRepo(checkpoint.repo) || !Number.isInteger(checkpoint.pr) || checkpoint.pr <= 0 ||
      !Number.isSafeInteger(checkpoint.highCommentId) || checkpoint.highCommentId < 0 ||
      !(checkpoint.parentHash === null || /^[a-f0-9]{64}$/.test(checkpoint.parentHash)) ||
      !validIsoTime(checkpoint.highCreatedAt) || !Array.isArray(checkpoint.entries) ||
      checkpoint.entries.length > CHECKPOINT_MAX_ENTRIES) return null;
  for (const entry of checkpoint.entries) {
    if (!entry || !Number.isSafeInteger(entry.id) || entry.id <= 0 || typeof entry.body !== 'string' ||
        entry.body.length > CHECKPOINT_MAX_BYTES || !validIsoTime(entry.createdAt) ||
        typeof entry.login !== 'string' || entry.login.length > 100 ||
        !['Bot', 'User'].includes(entry.type) || typeof entry.association !== 'string' ||
        entry.association.length > 40) return null;
  }
  return checkpoint;
}

function encodeCheckpoint(input) {
  const checkpoint = canonicalCheckpoint(input);
  if (!checkpoint) throw new Error('invalid checkpoint');
  const payload = Buffer.from(JSON.stringify(checkpoint.entries), 'utf8');
  if (payload.length > CHECKPOINT_MAX_INFLATED_BYTES) throw new Error('checkpoint inflated payload exceeds cap');
  const envelope = {
    version: 2,
    revision: checkpoint.revision,
    repo: checkpoint.repo,
    pr: checkpoint.pr,
    highCommentId: checkpoint.highCommentId,
    highCreatedAt: checkpoint.highCreatedAt,
    parentHash: checkpoint.parentHash,
    entryCount: checkpoint.entries.length,
    payloadHash: crypto.createHash('sha256').update(payload).digest('hex'),
    encoding: 'deflate-raw',
    blob: zlib.deflateRawSync(payload).toString('base64url'),
  };
  const body = encodeMarker(CHECKPOINT_MARK, envelope);
  if (Buffer.byteLength(body, 'utf8') > CHECKPOINT_MAX_BYTES) throw new Error('checkpoint exceeds size cap');
  return body;
}

function decodeCheckpoint(body) {
  const decoded = decodeSingleMarker(body, 'codex-review-feishu-checkpoint') || {};
  if (decoded.version !== 2) return canonicalCheckpoint(decoded);
  if (!sortedKeysEqual(decoded, [
    'blob', 'encoding', 'entryCount', 'highCommentId', 'highCreatedAt', 'parentHash',
    'payloadHash', 'pr', 'repo', 'revision', 'version',
  ]) || decoded.encoding !== 'deflate-raw' || !Number.isInteger(decoded.entryCount) ||
      decoded.entryCount < 0 || decoded.entryCount > CHECKPOINT_MAX_ENTRIES ||
      !/^[a-f0-9]{64}$/.test(decoded.payloadHash || '') || typeof decoded.blob !== 'string' ||
      decoded.blob.length > CHECKPOINT_MAX_BYTES) return null;
  try {
    const payload = zlib.inflateRawSync(Buffer.from(decoded.blob, 'base64url'), {
      maxOutputLength: CHECKPOINT_MAX_INFLATED_BYTES,
    });
    if (crypto.createHash('sha256').update(payload).digest('hex') !== decoded.payloadHash) return null;
    const entries = JSON.parse(payload.toString('utf8'));
    if (!Array.isArray(entries) || entries.length !== decoded.entryCount) return null;
    return canonicalCheckpoint({
      version: 1, revision: decoded.revision, repo: decoded.repo, pr: decoded.pr,
      highCommentId: decoded.highCommentId, highCreatedAt: decoded.highCreatedAt,
      parentHash: decoded.parentHash, entries,
    });
  } catch (_error) { return null; }
}

// Bootstrap progress uses the same bounded compressed payload as a checkpoint, but is not
// authoritative. A drain must finish the oldest-to-newest scan before it may hydrate/send.
function encodeBootstrapProgress(input) {
  return encodeCheckpoint(input).replace(CHECKPOINT_MARK, BOOTSTRAP_MARK);
}

function decodeBootstrapProgress(body) {
  if (typeof body !== 'string' || !body.includes(BOOTSTRAP_MARK) || body.includes(CHECKPOINT_MARK)) return null;
  return decodeCheckpoint(body.replace(BOOTSTRAP_MARK, CHECKPOINT_MARK));
}

function virtualCheckpointComment(entry) {
  return {
    id: entry.id,
    body: entry.body,
    created_at: entry.createdAt,
    author_association: entry.association,
    user: { login: entry.login, type: entry.type },
  };
}

async function listCommentPage({
  github, context, core, prNum, page, since,
  deadlineMs = activeGithubAttemptDeadlineMs, nowFn = Date.now,
}) {
  return withGithubRetry({
    core,
    label: `读取 PR 通知账本 page=${page}`,
    deadlineMs,
    nowFn,
    operation: () => github.rest.issues.listComments({
      owner: context.repo.owner, repo: context.repo.repo, issue_number: prNum,
      page, per_page: COMMENT_PAGE_SIZE,
      ...(since ? { since } : {}),
      request: { timeout: GITHUB_REQUEST_TIMEOUT_MS },
    }),
  });
}

async function readCheckpointTail({
  github, context, core, prNum, nowFn = Date.now,
  attempts = MARKER_IO_ATTEMPTS, maxPages = COMMENT_MIGRATION_MAX_PAGES,
}) {
  if (typeof github.graphql !== 'function') return [];
  let before = null;
  const found = [];
  for (let page = 0; page < maxPages; page += 1) {
    const result = await withGithubRetry({
      core, label: `从 comment tail 定位 checkpoint page=${page + 1}`,
      nowFn, attempts,
      operation: () => githubGraphqlWithTimeout(github, `query($owner:String!,$repo:String!,$pr:Int!,$before:String){
        repository(owner:$owner,name:$repo){pullRequest(number:$pr){comments(last:100,before:$before){
          pageInfo{hasPreviousPage startCursor}
          nodes{id databaseId body createdAt authorAssociation author{login __typename}}
        }}}
      }`, { owner: context.repo.owner, repo: context.repo.repo, pr: prNum, before }),
    });
    if (!result.ok) return null;
    const connection = result.value.repository.pullRequest.comments;
    for (const node of connection.nodes || []) {
      if (!(String(node.body || '').includes(CHECKPOINT_MARK) ||
            String(node.body || '').includes(BOOTSTRAP_MARK)) ||
          String(node.author && node.author.login || '').toLowerCase() !== GITHUB_ACTIONS_BOT) continue;
      found.push({
        id: Number(node.databaseId), node_id: node.id, body: node.body, created_at: node.createdAt,
        author_association: node.authorAssociation,
        user: { login: node.author.login, type: node.author.__typename === 'Bot' ? 'Bot' : 'User' },
      });
    }
    if (found.length || !connection.pageInfo.hasPreviousPage) return found;
    before = connection.pageInfo.startCursor;
  }
  // A bounded tail miss is not proof that a large legacy PR has no checkpoint. The forward
  // bootstrap below can safely establish that fact without materializing ordinary comments.
  return found;
}

function bootstrapMarkerEntry(comment) {
  const body = String(comment && comment.body || '');
  if (!body.includes('<!-- codex-review-feishu-') || body.includes(CHECKPOINT_MARK) ||
      body.includes(BOOTSTRAP_MARK)) return null;
  const trusted = isGithubActionsBot(comment) ||
    (legacyMessageId(body) && isTrustedMarkerComment(comment, { legacy: true }));
  const id = Number(comment && comment.id);
  if (!trusted || !Number.isSafeInteger(id) || id <= 0) return null;
  return {
    id, body,
    createdAt: validIsoTime(comment.created_at) ? comment.created_at : '1970-01-01T00:00:00.000Z',
    login: String(comment.user && comment.user.login || ''),
    type: String(comment.user && comment.user.type || 'User') === 'Bot' ? 'Bot' : 'User',
    association: String(comment.author_association || 'NONE'),
  };
}

// GitHub comments are the source of truth. Concurrent append-only checkpoint writers can
// legitimately create sibling snapshots from the same base. Select the sibling with the
// lowest high-watermark so every omitted comment is replayed, then follow only descendants
// of that deterministic branch. Orphan siblings remain harmless source comments and a later
// checkpoint joins them after replay; no process-local lock is required.
function selectSnapshotLineage(decoded, core, label, valueKey) {
  const byRevision = new Map();
  for (const entry of decoded) {
    const value = entry[valueKey];
    if (!byRevision.has(value.revision)) byRevision.set(value.revision, []);
    byRevision.get(value.revision).push(entry);
  }
  let selected = null;
  for (const [revision, entries] of [...byRevision.entries()].sort((a, b) => a[0] - b[0])) {
    if (selected && revision !== selected[valueKey].revision + 1) {
      core.warning(`${label} revision=${revision} 与 canonical lineage 不连续；从较早 high-watermark 重放`);
      break;
    }
    const unique = new Map();
    for (const entry of entries) {
      const hash = canonicalHash(entry[valueKey]);
      if (!unique.has(hash) || Number(entry.comment.id) < Number(unique.get(hash).entry.comment.id)) {
        unique.set(hash, { entry, hash });
      }
    }
    const parentHash = selected ? canonicalHash(selected[valueKey]) : null;
    const compatible = [...unique.values()].filter(({ entry }) =>
      !selected || entry[valueKey].parentHash === parentHash);
    if (!compatible.length) {
      core.warning(`${label} revision=${revision} 没有连接 canonical parent；从较早 high-watermark 重放`);
      break;
    }
    compatible.sort((left, right) => {
      const a = left.entry[valueKey];
      const b = right.entry[valueKey];
      return a.highCommentId - b.highCommentId ||
        Date.parse(a.highCreatedAt) - Date.parse(b.highCreatedAt) ||
        left.hash.localeCompare(right.hash) ||
        Number(left.entry.comment.id) - Number(right.entry.comment.id);
    });
    if (unique.size > 1) {
      core.warning(`${label} revision=${revision} 检测到并发 sibling；选择最低 high-watermark 并从 GitHub source replay`);
    }
    selected = compatible[0].entry;
  }
  return selected;
}

function selectBootstrapProgress(comments, repo, prNum, core) {
  const decoded = comments
    .filter((comment) => isGithubActionsBot(comment) && String(comment.body || '').includes(BOOTSTRAP_MARK))
    .map((comment) => ({ comment, progress: decodeBootstrapProgress(comment.body) }));
  if (decoded.some(({ progress }) => !progress || progress.repo !== repo || progress.pr !== prNum)) {
    core.setFailed('可信 bootstrap progress 损坏或绑定到其他 repo/PR');
    return { ok: false, progress: null };
  }
  const selected = selectSnapshotLineage(decoded, core, 'bootstrap progress', 'progress');
  return { ok: true, progress: selected ? selected.progress : null };
}

async function appendBootstrapProgress({
  github, context, core, prNum, previous, entries, highCommentId, highCreatedAt,
  nowFn = Date.now,
}) {
  const progress = canonicalCheckpoint({
    version: 1, revision: previous ? previous.revision + 1 : 1,
    repo: repositoryName(context), pr: prNum, highCommentId, highCreatedAt,
    parentHash: previous ? canonicalHash(previous) : null, entries,
  });
  let body;
  try { body = encodeBootstrapProgress(progress); } catch (error) {
    core.setFailed(`bootstrap progress 超过安全上限: ${errorMessage(error)}`);
    return false;
  }
  const result = await withGithubRetry({
    core, label: '持久化 bootstrap progress', attempts: 1, setFailedOnExhausted: false,
    nowFn,
    operation: () => github.rest.issues.createComment({
      owner: context.repo.owner, repo: context.repo.repo, issue_number: prNum, body,
      request: { timeout: GITHUB_REQUEST_TIMEOUT_MS },
    }),
  });
  if (result.ok) {
    await minimizeLedgerCommentBestEffort({
      github, core, comment: result.value.data, label: '折叠 bootstrap progress 评论', nowFn,
    });
    return true;
  }
  // createComment may have committed even though its response was lost. Spend exactly the
  // second reserved request on a one-page/one-attempt tail confirmation; further recovery is
  // delegated to the next run's source scan rather than exceeding the shared outer budget.
  const targetHash = canonicalHash(progress);
  const tail = await readCheckpointTail({
    github, context, core, prNum, nowFn, attempts: 1, maxPages: 1,
  });
  const confirmed = tail && tail.find((comment) => {
    const candidate = decodeBootstrapProgress(comment.body);
    return candidate && candidate.repo === repositoryName(context) && candidate.pr === prNum &&
      canonicalHash(candidate) === targetHash;
  });
  if (confirmed) {
    await minimizeLedgerCommentBestEffort({
      github, core, comment: confirmed, label: '折叠已恢复的 bootstrap progress 评论', nowFn,
    });
    return true;
  }
  core.setFailed('bootstrap progress POST 未确认；下轮从 GitHub source 重新扫描');
  return false;
}

async function readThreadMarkerComments({
  github, context, core = noopCore(), prNum, bootstrapPageLimit = Infinity,
  nowFn = Date.now,
}) {
  const repo = repositoryName(context);
  const tailMarkers = await readCheckpointTail({ github, context, core, prNum, nowFn });
  if (tailMarkers === null) return null;
  const checkpointComments = tailMarkers.filter((comment) =>
    isGithubActionsBot(comment) && String(comment.body || '').includes(CHECKPOINT_MARK));
  const decodedCheckpoints = checkpointComments.map((comment) => ({ comment, checkpoint: decodeCheckpoint(comment.body) }));
  if (decodedCheckpoints.some(({ checkpoint }) => !checkpoint || checkpoint.repo !== repo || checkpoint.pr !== prNum)) {
    core.setFailed('可信 checkpoint 损坏或绑定到其他 repo/PR');
    return null;
  }
  const selected = selectSnapshotLineage(decodedCheckpoints, core, 'checkpoint', 'checkpoint');
  const checkpointComment = selected ? selected.comment : null;
  const checkpoint = selected ? selected.checkpoint : null;
  let comments = [];
  let truncated = false;
  let bootstrapHigh = null;
  if (checkpoint) {
    comments = checkpoint.entries.map(virtualCheckpointComment);
    for (let page = 1; page <= COMMENT_INCREMENTAL_MAX_PAGES; page += 1) {
      const result = await listCommentPage({
        github, context, core, prNum, page,
        since: new Date(Date.parse(checkpoint.highCreatedAt) - 1000).toISOString(),
        nowFn,
      });
      if (!result.ok) return null;
      const fresh = result.value.data.filter((comment) => Number(comment.id) > checkpoint.highCommentId &&
        !String(comment.body || '').includes(CHECKPOINT_MARK));
      comments.push(...fresh);
      if (result.value.data.length < COMMENT_PAGE_SIZE) break;
      if (page === COMMENT_INCREMENTAL_MAX_PAGES) truncated = true;
    }
  } else {
    const selectedProgress = selectBootstrapProgress(tailMarkers, repo, prNum, core);
    if (!selectedProgress.ok) return null;
    const previousProgress = selectedProgress.progress;
    const retainedById = new Map((previousProgress ? previousProgress.entries : [])
      .map((entry) => [entry.id, entry]));
    const baseHighCommentId = previousProgress ? previousProgress.highCommentId : 0;
    let highCommentId = baseHighCommentId;
    let highCreatedAt = previousProgress ? previousProgress.highCreatedAt : '1970-01-01T00:00:00.000Z';
    const since = highCommentId ? new Date(Date.parse(highCreatedAt) - 1000).toISOString() : undefined;
    let complete = false;
    const scanDeadlineMs = Number.isFinite(activeGithubAttemptDeadlineMs)
      ? activeGithubAttemptDeadlineMs - BOOTSTRAP_PROGRESS_RESERVE_MS
      : activeGithubAttemptDeadlineMs;
    for (let page = 1; ; page += 1) {
      if (page > bootstrapPageLimit ||
          !hasGithubRequestBudget(nowFn(), BOOTSTRAP_PROGRESS_RESERVE_MS)) {
        truncated = true;
        break;
      }
      const result = await listCommentPage({
        github, context, core, prNum, page, since, deadlineMs: scanDeadlineMs, nowFn,
      });
      if (!result.ok) {
        // The scan budget is intentionally shorter than the outer budget. Persist all
        // fully absorbed pages with the reserved request slot even when a later page fails.
        truncated = true;
        break;
      }
      for (const comment of result.value.data) {
        const id = Number(comment.id);
        if (!Number.isSafeInteger(id) || id <= baseHighCommentId) continue;
        highCommentId = Math.max(highCommentId, id);
        if (validIsoTime(comment.created_at) && Date.parse(comment.created_at) > Date.parse(highCreatedAt)) {
          highCreatedAt = comment.created_at;
        }
        const entry = bootstrapMarkerEntry(comment);
        if (entry) retainedById.set(entry.id, entry);
      }
      if (result.value.data.length < COMMENT_PAGE_SIZE) {
        complete = true;
        break;
      }
    }
    const entries = [...retainedById.values()].sort((a, b) => a.id - b.id);
    if (!complete) {
      if (!await appendBootstrapProgress({
        github, context, core, prNum, previous: previousProgress, entries, highCommentId, highCreatedAt,
        nowFn,
      })) return null;
      comments = entries.map(virtualCheckpointComment);
      Object.defineProperty(comments, '_checkpoint', {
        value: {
          commentId: null, checkpoint: null, truncated: true, bootstrapIncomplete: true,
          bootstrapHighCommentId: highCommentId, bootstrapHighCreatedAt: highCreatedAt,
        }, enumerable: false,
      });
      return comments;
    }
    comments = entries.map(virtualCheckpointComment);
    bootstrapHigh = { highCommentId, highCreatedAt };
  }
  if (truncated) core.warning('incremental comment snapshot 达到有界页上限，仅推进已吸收 high-watermark 并续下一批');
  comments = [...new Map(comments.map((comment) => [Number(comment.id), comment])).values()]
    .sort((a, b) => Number(a.id) - Number(b.id));
  Object.defineProperty(comments, '_checkpoint', {
    value: {
      commentId: checkpointComment && Number(checkpointComment.id), checkpoint, truncated,
      bootstrapHighCommentId: bootstrapHigh && bootstrapHigh.highCommentId,
      bootstrapHighCreatedAt: bootstrapHigh && bootstrapHigh.highCreatedAt,
    }, enumerable: false,
  });
  return comments;
}

function conflictState(core, message, comments = [], options = {}) {
  if (!options.silent) core.setFailed(message);
  return { ok: false, kind: options.kind || 'conflict', comments, message, ...options.extra };
}

function unsignedRepairRecord(input) {
  return {
    version: input.version,
    repo: input.repo,
    pr: input.pr,
    legacyHash: input.legacyHash,
    generation: input.generation,
    keyId: input.keyId,
    claimId: input.claimId,
    messageId: input.messageId,
    runId: input.runId,
  };
}

function parseRepairPublicKeyring(value) {
  let parsed = value;
  if (typeof value === 'string') {
    try { parsed = JSON.parse(value); } catch (_error) { return null; }
  }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return null;
  const entries = Object.entries(parsed);
  if (entries.length < 1 || entries.length > 16) return null;
  const keyring = {};
  for (const [keyId, publicKey] of entries) {
    if (!/^[A-Za-z0-9_.-]{1,40}$/.test(keyId) || typeof publicKey !== 'string' || publicKey.length > 10000) return null;
    try {
      if (crypto.createPublicKey(publicKey).asymmetricKeyType !== 'ed25519') return null;
    } catch (_error) { return null; }
    keyring[keyId] = publicKey;
  }
  return keyring;
}

function repairPublicKey(keyring, keyId) {
  const parsed = parseRepairPublicKeyring(keyring);
  return parsed && parsed[keyId] || null;
}

function privateKeyMatchesKeyring(privateKey, keyring, keyId) {
  const expected = repairPublicKey(keyring, keyId);
  if (!privateKey || !expected) return false;
  try {
    const derivedKey = crypto.createPublicKey(privateKey);
    if (derivedKey.asymmetricKeyType !== 'ed25519') return false;
    const derived = derivedKey.export({ type: 'spki', format: 'der' });
    const configured = crypto.createPublicKey(expected).export({ type: 'spki', format: 'der' });
    return derived.length === configured.length && crypto.timingSafeEqual(derived, configured);
  } catch (_error) {
    return false;
  }
}

function repairPayload(input) {
  const payload = JSON.stringify(unsignedRepairRecord(input));
  return Buffer.from(`codex-review-feishu-legacy-repair:v1\n${payload}`, 'utf8');
}

function signRepairRecord(input, privateKey) {
  if (!privateKey) throw new Error('CODEX_FEISHU_REPAIR_PRIVATE_KEY 未设置');
  return crypto.sign(null, repairPayload(input), privateKey).toString('base64url');
}

function canonicalRepairRecord(input, key, verifyOnly = false) {
  const unsigned = unsignedRepairRecord(input);
  const signature = verifyOnly ? input.signature : signRepairRecord(unsigned, key);
  const record = { ...unsigned, signature };
  if (!sortedKeysEqual(record, REPAIR_KEYS) || record.version !== REPAIR_VERSION ||
      !validRepo(record.repo) || !Number.isInteger(record.pr) || record.pr <= 0 ||
      !/^[a-f0-9]{64}$/.test(record.legacyHash || '') ||
      !Number.isInteger(record.generation) || record.generation < 1 ||
      !/^[A-Za-z0-9_.-]{1,40}$/.test(record.keyId || '') ||
      !validClaimId(record.claimId) || !validFeishuMessageId(record.messageId) ||
      !/^[A-Za-z0-9_-]{1,100}$/.test(record.runId || '') ||
      !/^[A-Za-z0-9_-]{80,120}$/.test(record.signature || '')) return null;
  if (verifyOnly) {
    const publicKey = repairPublicKey(key, record.keyId);
    if (!publicKey) return null;
    try {
      if (!crypto.verify(null, repairPayload(unsigned), publicKey, Buffer.from(record.signature, 'base64url'))) return null;
    } catch (_error) {
      return null;
    }
  }
  return record;
}

function encodeRepairRecord(input, privateKey) {
  const record = canonicalRepairRecord(input, privateKey, false);
  if (!record) throw new Error('invalid legacy repair record');
  return encodeMarker(REPAIR_MARK, record);
}

function decodeRepairRecord(body, publicKey) {
  const decoded = decodeSingleMarker(body, 'codex-review-feishu-repair');
  return canonicalRepairRecord(decoded || {}, publicKey, true);
}

function unsignedThreadRepairRecord(input) {
  return {
    version: input.version,
    repo: input.repo,
    pr: input.pr,
    priorThreadHash: input.priorThreadHash,
    action: input.action,
    messageId: typeof input.messageId === 'undefined' ? null : input.messageId,
    runId: input.runId,
    operator: input.operator,
    keyId: input.keyId,
  };
}

function threadRepairPayload(input) {
  return Buffer.from(`codex-review-feishu-thread-repair:v1\n${JSON.stringify(unsignedThreadRepairRecord(input))}`, 'utf8');
}

function canonicalThreadRepairRecord(input, key, verifyOnly = false) {
  const unsigned = unsignedThreadRepairRecord(input);
  let signature;
  try {
    signature = verifyOnly ? input.signature : crypto.sign(null, threadRepairPayload(unsigned), key).toString('base64url');
  } catch (_error) { return null; }
  const record = { ...unsigned, signature };
  if (!sortedKeysEqual(record, THREAD_REPAIR_KEYS) || record.version !== 1 ||
      !validRepo(record.repo) || !Number.isInteger(record.pr) || record.pr <= 0 ||
      !/^[a-f0-9]{64}$/.test(record.priorThreadHash || '') ||
      !['finalize', 'release'].includes(record.action) ||
      (record.action === 'finalize' ? !validFeishuMessageId(record.messageId) : record.messageId !== null) ||
      !/^[A-Za-z0-9_-]{1,100}$/.test(record.runId || '') ||
      !/^[A-Za-z0-9_.-]{1,100}$/.test(record.operator || '') ||
      !/^[A-Za-z0-9_.-]{1,40}$/.test(record.keyId || '') ||
      !/^[A-Za-z0-9_-]{80,120}$/.test(record.signature || '')) return null;
  if (verifyOnly) {
    const publicKey = repairPublicKey(key, record.keyId);
    if (!publicKey) return null;
    try {
      if (!crypto.verify(null, threadRepairPayload(unsigned), publicKey, Buffer.from(record.signature, 'base64url'))) return null;
    } catch (_error) { return null; }
  }
  return record;
}

function encodeThreadRepairRecord(input, privateKey) {
  const record = canonicalThreadRepairRecord(input, privateKey, false);
  if (!record) throw new Error('invalid thread repair record');
  return encodeMarker(THREAD_REPAIR_MARK, record);
}

function decodeThreadRepairRecord(body, keyring) {
  const decoded = decodeSingleMarker(body, 'codex-review-feishu-thread-repair');
  return canonicalThreadRepairRecord(decoded || {}, keyring, true);
}

function legacyConflictHash(entries) {
  const snapshot = entries
    .map((entry) => `${String(entry.commentId)}:${entry.messageId}`)
    .sort();
  return crypto.createHash('sha256').update(JSON.stringify(snapshot)).digest('hex');
}

function reduceV2Entries(entries, core, comments) {
  const byGeneration = new Map();
  for (const entry of entries) {
    if (!byGeneration.has(entry.generation)) byGeneration.set(entry.generation, []);
    byGeneration.get(entry.generation).push(entry);
  }
  const reduced = [];
  for (const [generation, generationEntries] of byGeneration.entries()) {
    const claims = [...new Set(generationEntries.map((entry) => entry.claimId))];
    if (claims.length !== 1) return conflictState(core, `generation=${generation} 存在不同 v2 claim，停止通知`, comments);
    const states = new Set(generationEntries.map((entry) => entry.state));
    const finalMids = [...new Set(generationEntries.filter((entry) => entry.state === 'final').map((entry) => entry.messageId))];
    if (finalMids.length > 1 || (states.has('final') && states.has('released'))) {
      return conflictState(core, `generation=${generation} v2 terminal 冲突，停止通知`, comments);
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
  return { ok: true, reduced };
}

async function readThreadState({
  github, context, core = noopCore(), prNum, allowLegacyConflict = false,
  repairPublicKey = process.env.CODEX_FEISHU_REPAIR_PUBLIC_KEYS,
  comments: suppliedComments = null,
}) {
  const comments = suppliedComments || await readThreadMarkerComments({ github, context, core, prNum });
  if (!comments) return { ok: false, kind: 'unavailable', comments: [] };
  const repo = repositoryName(context);
  const v2Entries = [];
  const legacyEntries = [];
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
      let repair = null;
      if (body.includes(REPAIR_MARK)) {
        repair = decodeRepairRecord(body, repairPublicKey);
        if (!repair || repair.repo !== repo || repair.pr !== prNum ||
            repair.generation !== decoded.generation || repair.claimId !== decoded.claimId ||
            repair.messageId !== decoded.messageId || decoded.state !== 'final') {
          return conflictState(core, `可信 legacy repair 签名或绑定非法 comment_id=${comment.id}`, comments);
        }
      }
      let threadRepair = null;
      if (body.includes(THREAD_REPAIR_MARK)) {
        threadRepair = decodeThreadRepairRecord(body, repairPublicKey);
        const prior = [...v2Entries].reverse().find((entry) => entry.state === 'pending' &&
          entry.generation === decoded.generation && entry.claimId === decoded.claimId);
        const expectedState = threadRepair && threadRepair.action === 'finalize' ? 'final' : 'released';
        if (!threadRepair || !prior || threadRepair.repo !== repo || threadRepair.pr !== prNum ||
            threadRepair.priorThreadHash !== canonicalHash(canonicalThreadState(prior)) ||
            decoded.state !== expectedState || decoded.messageId !== threadRepair.messageId) {
          return conflictState(core, `可信 thread repair 签名或 prior snapshot 绑定非法 comment_id=${comment.id}`, comments);
        }
      }
      v2Entries.push({ ...decoded, commentId: comment.id, repair, threadRepair });
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
    legacyEntries.push({
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

  const v2 = reduceV2Entries(v2Entries, core, comments);
  if (!v2.ok) return v2;
  const legacyMids = [...new Set(legacyEntries.map((entry) => entry.messageId))];
  let reduced = v2.reduced.slice();
  if (legacyMids.length > 1) {
    const legacyHash = legacyConflictHash(legacyEntries);
    const signedMigration = v2Entries.some((entry) => entry.repair && entry.repair.legacyHash === legacyHash);
    if (!signedMigration) {
      return conflictState(core, 'generation=0 legacy marker 冲突，需签名 repair', comments, {
        silent: allowLegacyConflict,
        kind: 'legacy-conflict',
        extra: { legacyHash, legacyMessageIds: legacyMids.sort(), legacyEntries, v2Generations: reduced },
      });
    }
  } else if (legacyMids.length === 1) {
    const exemplar = legacyEntries[0];
    reduced.push({ ...exemplar, commentIds: legacyEntries.map((entry) => entry.commentId) });
  }

  if (reduced.length === 0) return { ok: true, kind: 'none', generation: 0, comments };
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

function stateCommentBody(state, repair = null, repairPrivateKey = null, threadRepair = null) {
  const parts = [encodeThreadState(state)];
  if (repair) parts.push(encodeRepairRecord(repair, repairPrivateKey));
  if (threadRepair) parts.push(encodeThreadRepairRecord(threadRepair, repairPrivateKey));
  if (state.state === 'final') {
    parts.push(`${MARK}${state.messageId} -->`);
    parts.push('<sub>Codex 代码评审话题锚点（自动维护，请勿删除）</sub>');
  } else {
    parts.push(`<sub>Codex 代码评审话题状态：${state.state}（自动维护，请勿删除）</sub>`);
  }
  return parts.join('\n');
}

async function createComment({ github, context, core, prNum, body, label }) {
  const created = await withGithubRetry({
    core,
    label,
    setFailedOnExhausted: false,
    operation: () => github.rest.issues.createComment({
      owner: context.repo.owner,
      repo: context.repo.repo,
      issue_number: prNum,
      body,
      request: { timeout: GITHUB_REQUEST_TIMEOUT_MS },
    }),
  });
  if (created.ok) {
    await minimizeLedgerCommentBestEffort({
      github, core, comment: created.value.data, label: `折叠 ${label}评论`,
    });
  }
  return created;
}

async function appendStateAndConfirm({
  github, context, core = noopCore(), prNum, expected, repair = null,
  repairPrivateKey = null, repairPublicKey = null, threadRepair = null, session = null,
}) {
  const created = await createComment({
    github, context, core, prNum,
    body: stateCommentBody(expected, repair, repairPrivateKey, threadRepair),
    label: `追加 ${expected.state} 话题状态`,
  });
  if (session) {
    if (!created.ok) {
      core.setFailed(`${expected.state} 状态写入未获 GitHub create response 确认`);
      return { ok: false, state: session.threadState };
    }
    session.threadState = {
      ok: true,
      kind: expected.state,
      generation: expected.generation,
      claimId: expected.claimId,
      messageId: expected.messageId,
      rootMid: expected.messageId,
      state: expected,
    };
    return { ok: true, state: session.threadState };
  }
  const confirmed = await readThreadState({ github, context, core, prNum, repairPublicKey: repairPublicKey || undefined });
  const matches = confirmed.ok && confirmed.kind === expected.state &&
    confirmed.generation === expected.generation && confirmed.claimId === expected.claimId &&
    (expected.state !== 'final' || confirmed.messageId === expected.messageId);
  if (!matches) {
    core.setFailed(`${expected.state} 状态写入未能通过重读确认`);
    return { ok: false, state: confirmed };
  }
  return { ok: true, state: confirmed };
}

async function reserveThreadGeneration({
  github, context, core = noopCore(), prNum, generation, claimId, allowFromFinal = false, session = null,
}) {
  const current = session ? session.threadState : await readThreadState({ github, context, core, prNum });
  if (!current.ok) return { ok: false, state: current };
  if (current.kind === 'pending') {
    if (current.generation === generation && current.claimId === claimId) return { ok: true, state: current, recovered: true };
    core.setFailed(`PR 已有其他 pending generation=${current.generation}`);
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
    version: THREAD_STATE_VERSION, state: 'pending', repo: repositoryName(context), pr: prNum,
    generation, claimId, messageId: null,
  });
  return appendStateAndConfirm({ github, context, core, prNum, expected: pending, session });
}

async function finalizeThreadGeneration({ github, context, core = noopCore(), prNum, generation, claimId, messageId, session = null }) {
  if (!validFeishuMessageId(messageId)) {
    core.setFailed('final message_id 格式非法');
    return { ok: false };
  }
  const current = session ? session.threadState : await readThreadState({ github, context, core, prNum });
  if (!current.ok) return { ok: false, state: current };
  if (current.kind === 'final' && current.generation === generation &&
      current.claimId === claimId && current.messageId === messageId) {
    return { ok: true, state: current, recovered: true };
  }
  if (current.kind !== 'pending' || current.generation !== generation || current.claimId !== claimId) {
    core.setFailed('final 只能覆盖最高 generation 的同一 pending claim');
    return { ok: false, state: current };
  }
  const finalState = canonicalThreadState({
    version: THREAD_STATE_VERSION, state: 'final', repo: repositoryName(context), pr: prNum,
    generation, claimId, messageId,
  });
  return appendStateAndConfirm({ github, context, core, prNum, expected: finalState, session });
}

async function releaseThreadGeneration({ github, context, core = noopCore(), prNum, generation, claimId, session = null }) {
  const current = session ? session.threadState : await readThreadState({ github, context, core, prNum });
  if (!current.ok) return { ok: false, state: current };
  if (current.kind === 'released' && current.generation === generation && current.claimId === claimId) {
    return { ok: true, state: current, recovered: true };
  }
  if (current.kind !== 'pending' || current.generation !== generation || current.claimId !== claimId) {
    core.setFailed('released 只能覆盖最高 generation 的同一 pending claim');
    return { ok: false, state: current };
  }
  const released = canonicalThreadState({
    version: THREAD_STATE_VERSION, state: 'released', repo: repositoryName(context), pr: prNum,
    generation, claimId, messageId: null,
  });
  return appendStateAndConfirm({ github, context, core, prNum, expected: released, session });
}

async function repairLegacyConflict({
  github, context, core = noopCore(), prNum, messageId, runId, privateKey, keyring, keyId,
}) {
  if (!validFeishuMessageId(messageId) || !/^[A-Za-z0-9_-]{1,100}$/.test(String(runId || '')) ||
      !privateKey || !privateKeyMatchesKeyring(privateKey, keyring, keyId)) {
    core.setFailed('legacy repair 参数非法，或 private key 与 keyring/keyId 不匹配');
    return { ok: false };
  }
  core.setSecret(privateKey);
  const current = await readThreadState({
    github, context, core, prNum, allowLegacyConflict: true, repairPublicKey: keyring,
  });
  if (current.kind !== 'legacy-conflict') {
    core.setFailed('仅允许 repair generation-0 legacy marker 冲突');
    return { ok: false, state: current };
  }
  const v2 = current.v2Generations || [];
  if (v2.some((entry) => entry.state === 'pending')) {
    core.setFailed('存在 v2 pending，拒绝用 legacy repair 绕过');
    return { ok: false, state: current };
  }
  const generation = v2.length ? Math.max(...v2.map((entry) => entry.generation)) + 1 : 1;
  const repo = repositoryName(context);
  const claimId = crypto.createHash('sha256')
    .update(`legacy-repair:v1:${repo}#${prNum}:${current.legacyHash}:${generation}:${messageId}:${runId}`)
    .digest('hex').slice(0, 32);
  const expected = canonicalThreadState({
    version: THREAD_STATE_VERSION, state: 'final', repo, pr: prNum, generation, claimId, messageId,
  });
  const repair = {
    version: REPAIR_VERSION, repo, pr: prNum, legacyHash: current.legacyHash,
    generation, claimId, messageId, runId, keyId,
  };
  return appendStateAndConfirm({
    github, context, core, prNum, expected, repair,
    repairPrivateKey: privateKey, repairPublicKey: keyring,
  });
}

// Compatibility wrapper retained for callers that still write an exact legacy marker.
async function persistThreadMarker({ github, context, core = noopCore(), markerCommentId, prNum, markBody }) {
  const result = await withGithubRetry({
    core,
    label: '兼容回写 legacy 话题标记',
    operation: () => markerCommentId
      ? github.rest.issues.updateComment({
        owner: context.repo.owner, repo: context.repo.repo, comment_id: markerCommentId,
        body: markBody, request: { timeout: GITHUB_REQUEST_TIMEOUT_MS },
      })
      : github.rest.issues.createComment({
        owner: context.repo.owner, repo: context.repo.repo, issue_number: prNum,
        body: markBody, request: { timeout: GITHUB_REQUEST_TIMEOUT_MS },
      }),
  });
  if (result.ok && !markerCommentId) {
    await minimizeLedgerCommentBestEffort({
      github, core, comment: result.value && result.value.data, label: '折叠 legacy 话题标记评论',
    });
  }
  return result.ok;
}

function eventKeyFor(input) {
  return crypto.createHash('sha256').update(JSON.stringify([
    input.version, input.eventType, input.eventId, input.repo, input.pr, input.headSha, input.createdAt,
  ])).digest('hex');
}

function validateEventRef(ref) {
  return sortedKeysEqual(ref, EVENT_KEYS) && ref.version === EVENT_VERSION &&
    ['check_run', 'pull_request_review'].includes(ref.eventType) && /^\d{1,30}$/.test(ref.eventId || '') &&
    validRepo(ref.repo) && Number.isInteger(ref.pr) && ref.pr > 0 && validSha(ref.headSha) &&
    validIsoTime(ref.createdAt) && validEventKey(ref.eventKey) && ref.eventKey === eventKeyFor(ref);
}

function canonicalEventRef(input) {
  const ref = {
    version: input.version,
    eventType: input.eventType,
    eventId: String(input.eventId),
    repo: input.repo,
    pr: input.pr,
    headSha: input.headSha,
    createdAt: input.createdAt,
    eventKey: input.eventKey || '',
  };
  if (!ref.eventKey) ref.eventKey = eventKeyFor(ref);
  if (!validateEventRef(ref)) throw new Error('invalid codex review event ref');
  return ref;
}

function encodeEventRef(input) {
  return encodeMarker(EVENT_MARK, canonicalEventRef(input));
}

function decodeEventRef(body) {
  const decoded = decodeSingleMarker(body, 'codex-review-feishu-event');
  return validateEventRef(decoded) ? decoded : null;
}

function canonicalDrainCursor(input) {
  const cursor = {
    version: input.version,
    repo: input.repo,
    pr: input.pr,
    nextEventKey: typeof input.nextEventKey === 'undefined' ? null : input.nextEventKey,
    lastCommentId: typeof input.lastCommentId === 'undefined' ? null : input.lastCommentId,
  };
  if (!sortedKeysEqual(cursor, DRAIN_CURSOR_KEYS) || ![1, 2, 3].includes(cursor.version) || !validRepo(cursor.repo) ||
      !Number.isInteger(cursor.pr) || cursor.pr <= 0 ||
      !(cursor.nextEventKey === null || validEventKey(cursor.nextEventKey)) ||
      (cursor.version >= 3
        ? !(Number.isSafeInteger(cursor.lastCommentId) && cursor.lastCommentId > 0 && validEventKey(cursor.nextEventKey))
        : cursor.lastCommentId !== null)) {
    throw new Error('invalid drain cursor');
  }
  return cursor;
}

function encodeDrainCursor(input) {
  return encodeMarker(DRAIN_CURSOR_MARK, canonicalDrainCursor(input));
}

function decodeDrainCursor(body) {
  const decoded = decodeSingleMarker(body, 'codex-review-feishu-drain-cursor');
  try { return canonicalDrainCursor(decoded || {}); } catch (_error) { return null; }
}

function readDrainCursor({ comments, context, core = noopCore(), prNum }) {
  const repo = repositoryName(context);
  const entries = [];
  for (const comment of comments || []) {
    const body = String(comment.body || '');
    if (!body.includes(DRAIN_CURSOR_MARK)) continue;
    if (!isGithubActionsBot(comment)) continue;
    const cursor = decodeDrainCursor(body);
    if (!cursor || cursor.repo !== repo || cursor.pr !== prNum) {
      core.setFailed(`可信 drain cursor 格式或绑定非法 comment_id=${comment.id}`);
      return { ok: false, nextEventKey: null };
    }
    entries.push({ cursor, commentId: comment.id });
  }
  entries.sort((a, b) => String(a.commentId).localeCompare(String(b.commentId), undefined, { numeric: true }));
  const latest = entries.length ? entries[entries.length - 1].cursor : null;
  return {
    ok: true,
    version: latest ? latest.version : 3,
    nextEventKey: latest ? latest.nextEventKey : null,
    lastCommentId: latest ? latest.lastCommentId : null,
  };
}

async function appendDrainCursor({ github, context, core, prNum, nextEventKey, lastCommentId }) {
  const cursor = canonicalDrainCursor({
    version: 3, repo: repositoryName(context), pr: prNum, nextEventKey, lastCommentId,
  });
  const created = await createComment({
    github, context, core, prNum,
    body: `${encodeDrainCursor(cursor)}\n<sub>Codex 队列扫描游标（自动维护，请勿删除）</sub>`,
    label: '推进 Codex 队列扫描游标',
  });
  return created.ok;
}

async function readEventQueue({ github, context, core = noopCore(), prNum, comments: suppliedComments = null }) {
  const comments = suppliedComments || await readThreadMarkerComments({ github, context, core, prNum });
  if (!comments) return { ok: false, events: [] };
  const repo = repositoryName(context);
  const byKey = new Map();
  for (const comment of comments) {
    const body = String(comment.body || '');
    if (!body.includes(EVENT_MARK)) continue;
    if (!isGithubActionsBot(comment)) {
      core.warning(`忽略非 github-actions bot 的 event ref comment_id=${comment.id}`);
      continue;
    }
    const ref = decodeEventRef(body);
    if (!ref || ref.repo !== repo || ref.pr !== prNum) {
      return conflictState(core, `可信 event ref 格式或绑定非法 comment_id=${comment.id}`, comments, {
        extra: { events: [] },
      });
    }
    const prior = byKey.get(ref.eventKey);
    if (prior && JSON.stringify(prior.ref) !== JSON.stringify(ref)) {
      return conflictState(core, `eventKey=${ref.eventKey} 存在不同 immutable ref`, comments, {
        extra: { events: [] },
      });
    }
    if (!prior) byKey.set(ref.eventKey, { ref, commentId: comment.id });
  }
  const events = [...byKey.values()].sort((a, b) =>
    String(a.commentId).localeCompare(String(b.commentId), undefined, { numeric: true }));
  return { ok: true, events, comments };
}

async function enqueueEventRef({ github, context, core = noopCore(), ref }) {
  const canonical = canonicalEventRef(ref);
  const existing = await readEventQueue({ github, context, core, prNum: canonical.pr });
  if (!existing.ok) return { ok: false, queue: existing };
  const alreadyQueued = existing.events.find((entry) => entry.ref.eventKey === canonical.eventKey);
  if (alreadyQueued) {
    if (JSON.stringify(alreadyQueued.ref) !== JSON.stringify(canonical)) {
      core.setFailed(`eventKey=${canonical.eventKey} 已绑定不同 immutable ref`);
      return { ok: false, queue: existing };
    }
    return { ok: true, ref: canonical, queue: existing, existing: true };
  }
  await createComment({
    github, context, core, prNum: canonical.pr,
    body: `${encodeEventRef(canonical)}\n<sub>Codex 代码评审事件队列（自动维护，请勿删除）</sub>`,
    label: '追加 Codex 评审事件队列',
  });
  const queue = await readEventQueue({ github, context, core, prNum: canonical.pr });
  const found = queue.ok && queue.events.some((entry) => JSON.stringify(entry.ref) === JSON.stringify(canonical));
  if (!found) {
    core.setFailed('事件队列写入未能通过重读确认');
    return { ok: false, queue };
  }
  return { ok: true, ref: canonical, queue };
}

function deliveryToken(eventKey) {
  return `ZETTLAB-CODEX-DELIVERY:${crypto.createHash('sha256').update(`delivery:v1:${eventKey}`).digest('hex')}`;
}

function validateDeliveryRecord(record) {
  if (!sortedKeysEqual(record, DELIVERY_KEYS) || record.version !== DELIVERY_VERSION ||
      !validRepo(record.repo) || !Number.isInteger(record.pr) || record.pr <= 0 ||
      !validEventKey(record.eventKey) || record.token !== deliveryToken(record.eventKey) ||
      !['preparing', 'sending', 'uncertain', 'manual', 'retrying', 'done', 'not_sent', 'skipped', 'failed'].includes(record.state) ||
      !Number.isInteger(record.attempt) || record.attempt < 0 ||
      !Number.isInteger(record.historyAttempts) || record.historyAttempts < 0 ||
      !(record.nextCheckAt === null || validIsoTime(record.nextCheckAt)) ||
      !(record.sentAt === null || validIsoTime(record.sentAt)) ||
      !(record.messageId === null || validFeishuMessageId(record.messageId)) ||
      !(record.targetRoot === null || validFeishuMessageId(record.targetRoot)) ||
      !(record.threadClaimId === null || validClaimId(record.threadClaimId)) ||
      !(record.threadGeneration === null || (Number.isInteger(record.threadGeneration) && record.threadGeneration >= 1)) ||
      !(record.reason === null || /^[a-z0-9_-]{1,64}$/.test(record.reason))) return false;
  if (!Array.isArray(record.candidateMessageIds) || record.candidateMessageIds.length > 10 ||
      record.candidateMessageIds.some((mid) => !validFeishuMessageId(mid)) ||
      JSON.stringify(record.candidateMessageIds) !== JSON.stringify([...new Set(record.candidateMessageIds)].sort())) return false;
  if (['skipped', 'failed'].includes(record.state) && record.attempt === 0) {
    return record.mode === 'none' && record.sentAt === null && record.messageId === null &&
      record.targetRoot === null && record.threadClaimId === null && record.threadGeneration === null &&
      record.nextCheckAt === null && record.historyAttempts === 0 && Boolean(record.reason) && record.candidateMessageIds.length === 0;
  }
  if (!['root', 'reply'].includes(record.mode) || record.attempt < 1) return false;
  if (record.mode === 'root') {
    if (record.targetRoot !== null || !validClaimId(record.threadClaimId) || !Number.isInteger(record.threadGeneration)) return false;
  } else if (!validFeishuMessageId(record.targetRoot) || record.threadClaimId !== null || record.threadGeneration !== null) return false;
  if (record.state === 'preparing') {
    return record.mode === 'root' && record.sentAt === null && record.messageId === null &&
      record.nextCheckAt === null && record.historyAttempts === 0 && record.reason === null && record.candidateMessageIds.length === 0;
  }
  if (record.state === 'skipped') {
    return record.messageId === null && record.nextCheckAt === null && Boolean(record.reason);
  }
  if (!record.sentAt) return false;
  if (record.state === 'done') return validFeishuMessageId(record.messageId) && record.nextCheckAt === null && record.reason === null;
  if (record.messageId !== null) return false;
  if (record.state === 'uncertain') {
    return record.nextCheckAt !== null &&
      ['ambiguous', 'history_zero', 'history_error', 'history_incomplete', 'history_rate_limited'].includes(record.reason);
  }
  if (record.state === 'manual') {
    if (record.nextCheckAt !== null || !['history_multiple', 'history_exhausted', 'definitely_not_sent_exhausted'].includes(record.reason)) return false;
    return record.reason !== 'history_multiple' || record.candidateMessageIds.length >= 2;
  }
  if (record.state === 'retrying') return record.nextCheckAt === null && record.reason === 'operator_retry';
  if (record.state === 'not_sent') {
    if (record.reason === 'rate_limited') return record.nextCheckAt !== null;
    if (record.reason === 'root_missing') return record.mode === 'reply' && record.nextCheckAt === null;
    return record.reason === 'definitely_not_sent' && record.nextCheckAt === null;
  }
  if (record.state === 'failed') return record.nextCheckAt === null && Boolean(record.reason);
  return record.state === 'sending' && record.nextCheckAt === null && record.reason === null;
}

function canonicalDeliveryRecord(input) {
  const record = {
    version: input.version,
    state: input.state,
    repo: input.repo,
    pr: input.pr,
    eventKey: input.eventKey,
    candidateMessageIds: [...new Set(input.candidateMessageIds || [])].sort(),
    attempt: input.attempt,
    mode: input.mode,
    targetRoot: typeof input.targetRoot === 'undefined' ? null : input.targetRoot,
    threadGeneration: typeof input.threadGeneration === 'undefined' ? null : input.threadGeneration,
    threadClaimId: typeof input.threadClaimId === 'undefined' ? null : input.threadClaimId,
    token: input.token || deliveryToken(input.eventKey),
    sentAt: typeof input.sentAt === 'undefined' ? null : input.sentAt,
    messageId: typeof input.messageId === 'undefined' ? null : input.messageId,
    historyAttempts: input.historyAttempts || 0,
    nextCheckAt: typeof input.nextCheckAt === 'undefined' ? null : input.nextCheckAt,
    reason: typeof input.reason === 'undefined' ? null : input.reason,
  };
  if (!validateDeliveryRecord(record)) throw new Error('invalid codex review delivery record');
  return record;
}

function encodeDeliveryRecord(input) {
  return encodeMarker(DELIVERY_MARK, canonicalDeliveryRecord(input));
}

function decodeDeliveryRecord(body) {
  const decoded = decodeSingleMarker(body, 'codex-review-feishu-delivery');
  return validateDeliveryRecord(decoded) ? decoded : null;
}

function canonicalDeliveryTombstone(input) {
  let ref;
  try {
    ref = canonicalEventRef({
      version: EVENT_VERSION, repo: input.repo, pr: input.pr, eventKey: input.eventKey,
      eventType: input.eventType, eventId: input.eventId, headSha: input.headSha, createdAt: input.createdAt,
    });
  } catch (_error) { return null; }
  const tombstone = {
    version: 1, repo: ref.repo, pr: ref.pr, eventKey: ref.eventKey,
    eventType: ref.eventType, eventId: ref.eventId, headSha: ref.headSha, createdAt: ref.createdAt,
    eventCommentId: Number(input.eventCommentId), state: input.state,
  };
  if (!sortedKeysEqual(tombstone, DELIVERY_TOMBSTONE_KEYS) ||
      !Number.isSafeInteger(tombstone.eventCommentId) || tombstone.eventCommentId <= 0 ||
      !['done', 'skipped'].includes(tombstone.state)) return null;
  return tombstone;
}

function encodeDeliveryTombstone(input) {
  const tombstone = canonicalDeliveryTombstone(input);
  if (!tombstone) throw new Error('invalid delivery tombstone');
  return encodeMarker(DELIVERY_TOMBSTONE_MARK, tombstone);
}

function decodeDeliveryTombstone(body) {
  return canonicalDeliveryTombstone(
    decodeSingleMarker(body, 'codex-review-feishu-delivery-tombstone') || {},
  );
}

function sameDeliveryIdentity(a, b) {
  return a.repo === b.repo && a.pr === b.pr && a.eventKey === b.eventKey &&
    a.attempt === b.attempt && a.mode === b.mode && a.targetRoot === b.targetRoot &&
    a.threadGeneration === b.threadGeneration && a.threadClaimId === b.threadClaimId &&
    a.token === b.token && a.sentAt === b.sentAt;
}

function sameDeliveryStaticIdentity(a, b) {
  return a.repo === b.repo && a.pr === b.pr && a.eventKey === b.eventKey &&
    a.attempt === b.attempt && a.mode === b.mode && a.targetRoot === b.targetRoot &&
    a.threadGeneration === b.threadGeneration && a.threadClaimId === b.threadClaimId && a.token === b.token;
}

function isRateLimitedNotSent(record) {
  return record.state === 'not_sent' && record.reason === 'rate_limited';
}

function validDeliveryTransition(previous, next) {
  if (!previous) return ['preparing', 'sending', 'skipped', 'failed'].includes(next.state) ||
    isRateLimitedNotSent(next);
  if (JSON.stringify(previous) === JSON.stringify(next)) return true;
  if (['done', 'skipped'].includes(previous.state)) return false;
  if (previous.state === 'failed') return next.state === 'skipped' && sameDeliveryIdentity(previous, next);
  if (previous.state === 'manual') return false;
  if (previous.state === 'retrying') {
    if (next.state === 'skipped') return sameDeliveryIdentity(previous, next);
    if (previous.mode === 'reply') {
      return ['sending', 'not_sent'].includes(next.state) &&
        (next.state !== 'not_sent' || isRateLimitedNotSent(next)) && next.attempt === previous.attempt + 1 &&
        sameDeliveryStaticIdentity({ ...previous, attempt: next.attempt }, next);
    }
    return next.state === 'preparing' && next.attempt === previous.attempt + 1 &&
      sameDeliveryStaticIdentity({
        ...previous, attempt: next.attempt, mode: 'root', targetRoot: null,
        threadGeneration: next.threadGeneration, threadClaimId: next.threadClaimId,
      }, next);
  }
  if (previous.state === 'not_sent') {
    if (next.state === 'skipped') return sameDeliveryIdentity(previous, next);
    if (next.state === 'manual') return sameDeliveryIdentity(previous, next);
    if (previous.reason === 'rate_limited') {
      return ['sending', 'not_sent'].includes(next.state) &&
        (next.state !== 'not_sent' || isRateLimitedNotSent(next)) && next.attempt === previous.attempt + 1 &&
        sameDeliveryStaticIdentity({ ...previous, attempt: next.attempt }, next);
    }
    if (previous.reason === 'root_missing') {
      return previous.mode === 'reply' && next.state === 'preparing' &&
        next.attempt === previous.attempt + 1 &&
        sameDeliveryStaticIdentity({
          ...previous, attempt: next.attempt, mode: 'root', targetRoot: null,
          threadGeneration: next.threadGeneration, threadClaimId: next.threadClaimId,
        }, next);
    }
    if (previous.mode === 'reply') {
      return ['sending', 'not_sent'].includes(next.state) &&
        (next.state !== 'not_sent' || isRateLimitedNotSent(next)) && next.attempt === previous.attempt + 1 &&
        sameDeliveryStaticIdentity({ ...previous, attempt: next.attempt }, next);
    }
    return next.state === 'preparing' && next.attempt === previous.attempt + 1 && sameDeliveryStaticIdentity({ ...previous, attempt: next.attempt, mode: 'root', targetRoot: null, threadGeneration: next.threadGeneration, threadClaimId: next.threadClaimId }, next);
  }
  if (previous.state === 'preparing') {
    return ['sending', 'skipped', 'manual', 'not_sent'].includes(next.state) &&
      (next.state !== 'not_sent' || isRateLimitedNotSent(next)) &&
      (next.state !== 'sending' || next.sentAt !== null) && sameDeliveryStaticIdentity(previous, next);
  }
  if (!sameDeliveryIdentity(previous, next)) return false;
  if (previous.state === 'sending') return ['uncertain', 'manual', 'done', 'not_sent', 'failed'].includes(next.state);
  if (previous.state === 'uncertain') {
    return ['done', 'manual', 'failed'].includes(next.state) ||
      (next.state === 'uncertain' && (next.historyAttempts > previous.historyAttempts ||
        (next.historyAttempts === previous.historyAttempts && next.reason === 'history_rate_limited' &&
         Date.parse(next.nextCheckAt) > Date.parse(previous.nextCheckAt))));
  }
  return false;
}

function canonicalJson(value) {
  if (Array.isArray(value)) return `[${value.map((entry) => canonicalJson(entry)).join(',')}]`;
  if (value && typeof value === 'object') {
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`).join(',')}}`;
  }
  return JSON.stringify(value);
}

function canonicalHash(value) {
  return crypto.createHash('sha256').update(canonicalJson(value)).digest('hex');
}

function candidateSetHash(candidateMessageIds) {
  return canonicalHash([...new Set(candidateMessageIds || [])].sort());
}

function unsignedDeliveryRepairRecord(input) {
  return {
    version: input.version,
    action: input.action,
    repo: input.repo,
    pr: input.pr,
    eventKey: input.eventKey,
    priorManualHash: input.priorManualHash,
    priorReason: input.priorReason,
    candidateHash: input.candidateHash,
    messageId: input.messageId,
    runId: input.runId,
    operator: input.operator,
    keyId: input.keyId,
  };
}

function deliveryRepairPayload(input) {
  return Buffer.from(`codex-review-feishu-delivery-repair:v1\n${JSON.stringify(unsignedDeliveryRepairRecord(input))}`, 'utf8');
}

function canonicalDeliveryRepairRecord(input, key, verifyOnly = false) {
  const unsigned = unsignedDeliveryRepairRecord(input);
  const signature = verifyOnly
    ? input.signature
    : crypto.sign(null, deliveryRepairPayload(unsigned), key).toString('base64url');
  const record = { ...unsigned, signature };
  if (!sortedKeysEqual(record, DELIVERY_REPAIR_KEYS) || record.version !== 1 ||
      !validRepo(record.repo) || !Number.isInteger(record.pr) || record.pr <= 0 ||
      !validEventKey(record.eventKey) || !/^[a-f0-9]{64}$/.test(record.priorManualHash || '') ||
      !/^[a-f0-9]{64}$/.test(record.candidateHash || '') ||
      !['select_mid', 'retry', 'discard'].includes(record.action) ||
      !(record.messageId === null || validFeishuMessageId(record.messageId)) ||
      !/^[a-z0-9_-]{1,64}$/.test(record.priorReason || '') ||
      !/^[A-Za-z0-9_-]{1,100}$/.test(record.runId || '') ||
      !/^[A-Za-z0-9_.-]{1,100}$/.test(record.operator || '') ||
      !/^[A-Za-z0-9_.-]{1,40}$/.test(record.keyId || '') ||
      !/^[A-Za-z0-9_-]{80,120}$/.test(record.signature || '')) return null;
  if ((record.action === 'select_mid') !== validFeishuMessageId(record.messageId)) return null;
  if (verifyOnly) {
    const publicKey = repairPublicKey(key, record.keyId);
    if (!publicKey) return null;
    try {
      if (!crypto.verify(null, deliveryRepairPayload(unsigned), publicKey, Buffer.from(record.signature, 'base64url'))) return null;
    } catch (_error) { return null; }
  }
  return record;
}

function encodeDeliveryRepairRecord(input, privateKey) {
  const record = canonicalDeliveryRepairRecord(input, privateKey, false);
  if (!record) throw new Error('invalid signed delivery repair record');
  return encodeMarker(DELIVERY_REPAIR_MARK, record);
}

function decodeDeliveryRepairRecord(body, keyring) {
  const decoded = decodeSingleMarker(body, 'codex-review-feishu-delivery-repair');
  return canonicalDeliveryRepairRecord(decoded || {}, keyring, true);
}

function validSignedManualRepair(previous, next, repair) {
  if (!repair || previous.state !== 'manual' || !sameDeliveryIdentity(previous, next) ||
      repair.repo !== previous.repo || repair.pr !== previous.pr || repair.eventKey !== previous.eventKey ||
      repair.priorManualHash !== canonicalHash(previous) || repair.priorReason !== previous.reason ||
      repair.candidateHash !== candidateSetHash(previous.candidateMessageIds) ||
      JSON.stringify(previous.candidateMessageIds) !== JSON.stringify(next.candidateMessageIds)) return false;
  if (repair.action === 'select_mid') {
    return next.state === 'done' && repair.messageId === next.messageId &&
      previous.candidateMessageIds.includes(next.messageId);
  }
  if (repair.action === 'retry') {
    return ['definitely_not_sent_exhausted', 'history_exhausted'].includes(previous.reason) &&
      previous.candidateMessageIds.length === 0 && next.state === 'retrying' &&
      next.messageId === null && repair.messageId === null;
  }
  return Boolean(repair.action === 'discard' && next.state === 'skipped' &&
    next.messageId === null && repair.messageId === null);
}

async function readDeliveryLedger({
  github, context, core = noopCore(), prNum,
  repairPublicKeys = process.env.CODEX_FEISHU_REPAIR_PUBLIC_KEYS,
  comments: suppliedComments = null,
}) {
  const comments = suppliedComments || await readThreadMarkerComments({ github, context, core, prNum });
  if (!comments) return { ok: false, latest: new Map() };
  const repo = repositoryName(context);
  const grouped = new Map();
  const tombstones = new Map();
  for (const comment of comments) {
    const body = String(comment.body || '');
    if (body.includes(DELIVERY_TOMBSTONE_MARK)) {
      if (!isGithubActionsBot(comment)) {
        core.warning(`忽略非 github-actions bot 的 delivery tombstone comment_id=${comment.id}`);
        continue;
      }
      const tombstone = decodeDeliveryTombstone(body);
      if (!tombstone || tombstone.repo !== repo || tombstone.pr !== prNum) {
        return conflictState(core, `可信 delivery tombstone 格式或绑定非法 comment_id=${comment.id}`, comments, {
          extra: { latest: new Map() },
        });
      }
      const previous = tombstones.get(tombstone.eventKey);
      if (previous && canonicalHash(previous) !== canonicalHash(tombstone)) {
        return conflictState(core, `eventKey=${tombstone.eventKey} delivery tombstone 冲突`, comments, {
          extra: { latest: new Map() },
        });
      }
      tombstones.set(tombstone.eventKey, tombstone);
      continue;
    }
    if (!body.includes(DELIVERY_MARK)) continue;
    if (!isGithubActionsBot(comment)) {
      core.warning(`忽略非 github-actions bot 的 delivery record comment_id=${comment.id}`);
      continue;
    }
    const record = decodeDeliveryRecord(body);
    if (!record || record.repo !== repo || record.pr !== prNum) {
      return conflictState(core, `可信 delivery record 格式或绑定非法 comment_id=${comment.id}`, comments, {
        extra: { latest: new Map() },
      });
    }
    let repair = null;
    if (body.includes(DELIVERY_REPAIR_MARK)) {
      repair = decodeDeliveryRepairRecord(body, repairPublicKeys);
      if (!repair) {
        return conflictState(core, `可信 delivery repair 签名非法 comment_id=${comment.id}`, comments, {
          extra: { latest: new Map() },
        });
      }
    }
    if (!grouped.has(record.eventKey)) grouped.set(record.eventKey, []);
    grouped.get(record.eventKey).push({ record, repair, commentId: comment.id });
  }
  const latest = new Map();
  for (const [eventKey, entries] of grouped.entries()) {
    entries.sort((a, b) => String(a.commentId).localeCompare(String(b.commentId), undefined, { numeric: true }));
    let previous = null;
    let previousRepair = null;
    for (const entry of entries) {
      const exactDuplicate = previous && canonicalHash(previous) === canonicalHash(entry.record) &&
        canonicalHash(previousRepair) === canonicalHash(entry.repair);
      if (exactDuplicate) continue;
      const validTransition = previous && previous.state === 'manual' && ['done', 'retrying', 'skipped'].includes(entry.record.state)
        ? validSignedManualRepair(previous, entry.record, entry.repair)
        : validDeliveryTransition(previous, entry.record) && !entry.repair;
      if (!validTransition) {
        return conflictState(core, `eventKey=${eventKey} delivery 状态转换冲突`, comments, {
          extra: { latest: new Map() },
        });
      }
      previous = entry.record;
      previousRepair = entry.repair;
    }
    latest.set(eventKey, previous);
  }
  for (const [eventKey, tombstone] of tombstones.entries()) {
    const current = latest.get(eventKey);
    if (current && (!['done', 'skipped'].includes(current.state) || current.state !== tombstone.state)) {
      return conflictState(core, `eventKey=${eventKey} delivery tombstone 与账本冲突`, comments, {
        extra: { latest: new Map() },
      });
    }
    if (!current) latest.set(eventKey, tombstone);
  }
  return { ok: true, latest, comments };
}

async function appendDeliveryAndConfirm({
  github, context, core = noopCore(), prNum, record,
  repair = null, repairPrivateKey = null, repairPublicKeys = null, session = null,
}) {
  const canonical = canonicalDeliveryRecord(record);
  const markers = [encodeDeliveryRecord(canonical)];
  let signedRepair = null;
  if (repair) {
    const encodedRepair = encodeDeliveryRepairRecord(repair, repairPrivateKey);
    markers.push(encodedRepair);
    signedRepair = decodeDeliveryRepairRecord(encodedRepair, repairPublicKeys);
  }
  if (session) {
    const previous = session.latest.get(canonical.eventKey);
    const validTransition = previous && previous.state === 'manual' && ['done', 'retrying', 'skipped'].includes(canonical.state)
      ? validSignedManualRepair(previous, canonical, signedRepair)
      : validDeliveryTransition(previous, canonical) && !signedRepair;
    if (!validTransition) {
      core.setFailed(`eventKey=${canonical.eventKey} 本地 delivery transition 非法`);
      return { ok: false };
    }
  }
  const created = await createComment({
    github, context, core, prNum,
    body: `${markers.join('\n')}\n<sub>Codex 飞书投递状态：${canonical.state}（自动维护，请勿删除）</sub>`,
    label: `追加 ${canonical.state} 投递状态`,
  });
  if (session) {
    if (!created.ok) {
      // A create response can be lost after GitHub persisted the comment. Re-read only on
      // this exceptional path; accepting the exact canonical record is safe because the
      // Feishu POST always happens after the sending transition returns successfully.
      const ledger = await readDeliveryLedger({
        github, context, core, prNum, repairPublicKeys: repairPublicKeys || undefined,
      });
      const confirmed = ledger.ok &&
        JSON.stringify(ledger.latest.get(canonical.eventKey)) === JSON.stringify(canonical);
      if (!confirmed) {
        core.setFailed(`${canonical.state} delivery 写入未获 GitHub create response 或重读确认`);
        return { ok: false, ledger };
      }
      session.latest = ledger.latest;
      return { ok: true, record: canonical, ledger, recovered: true };
    }
    session.latest.set(canonical.eventKey, canonical);
    return { ok: true, record: canonical };
  }
  const ledger = await readDeliveryLedger({
    github, context, core, prNum, repairPublicKeys: repairPublicKeys || undefined,
  });
  const confirmed = ledger.ok && JSON.stringify(ledger.latest.get(canonical.eventKey)) === JSON.stringify(canonical);
  if (!confirmed) {
    core.setFailed(`${canonical.state} 投递状态写入未能通过重读确认`);
    return { ok: false, ledger };
  }
  return { ok: true, record: canonical, ledger };
}

function terminalNoSendRecord(ref, state, reason) {
  return canonicalDeliveryRecord({
    version: DELIVERY_VERSION, state, repo: ref.repo, pr: ref.pr, eventKey: ref.eventKey,
    attempt: 0, mode: 'none', token: deliveryToken(ref.eventKey), reason,
  });
}

function feishuFailureSummary(result) {
  const status = result && typeof result.status !== 'undefined' ? result.status : 'UNKNOWN';
  const json = result && result.json || {};
  const code = typeof json.code !== 'undefined' ? json.code : 'UNKNOWN';
  const requestId = json.request_id || json.data && json.data.request_id;
  return `HTTP ${status}, code=${code}${requestId ? `, request_id=${requestId}` : ''}`;
}

function shouldRecreateRootOnReplyFailure(result) {
  return ROOT_MISSING_CODES.has(String(result && result.json && result.json.code || ''));
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
  const timer = setTimeout(() => ctrl.abort(), FEISHU_REQUEST_TIMEOUT_MS);
  try {
    const resp = await fetch(`${FEISHU}${apiPath}`, {
      method, headers, body: body ? JSON.stringify(body) : undefined, signal: ctrl.signal,
    });
    const text = await resp.text();
    let json;
    try { json = JSON.parse(text); } catch (_error) { json = {}; }
    return {
      ok: resp.ok,
      status: resp.status,
      json,
      retryAfterMs: parseRetryAfterHeadersMs(resp.headers, Date.now(), FEISHU_RATE_LIMIT_MAX_RETRY_AFTER_MS),
    };
  } catch (error) {
    const message = error && error.name === 'AbortError' ? 'feishu API timeout (15s)' : errorMessage(error);
    return { ok: false, status: 0, json: { code: 'NETWORK_ERROR', msg: message } };
  } finally {
    clearTimeout(timer);
  }
}

async function feishuWithRateLimit({
  apiPath, method, body, token, core = noopCore(), sleepFn = sleep, nowFn = Date.now,
}) {
  let result = null;
  for (let attempt = 1; attempt <= FEISHU_RATE_LIMIT_MAX_ATTEMPTS; attempt += 1) {
    result = await feishu(apiPath, method, body, token);
    if (Number(result.status) !== 429) return result;
    const retryAfterMs = result.retryAfterMs === null
      ? Math.min(GITHUB_RETRY_BASE_MS * (2 ** (attempt - 1)), FEISHU_RATE_LIMIT_MAX_INLINE_DELAY_MS)
      : result.retryAfterMs;
    const remainingMs = Number.isFinite(activeGithubAttemptDeadlineMs)
      ? activeGithubAttemptDeadlineMs - nowFn()
      : Infinity;
    const canRetry = attempt < FEISHU_RATE_LIMIT_MAX_ATTEMPTS &&
      retryAfterMs <= FEISHU_RATE_LIMIT_MAX_INLINE_DELAY_MS &&
      remainingMs >= retryAfterMs + FEISHU_REQUEST_TIMEOUT_MS + FEISHU_RATE_LIMIT_PERSIST_RESERVE_MS;
    if (!canRetry) return { ...result, rateLimited: true, retryAfterMs };
    core.warning(`Feishu HTTP 429，第 ${attempt} 次有界重试将在 ${retryAfterMs}ms 后执行`);
    await sleepFn(retryAfterMs);
  }
  return { ...result, rateLimited: true, retryAfterMs: FEISHU_RATE_LIMIT_MAX_INLINE_DELAY_MS };
}

function readFeishuEnv(core, required = false) {
  const appId = process.env.CODEREVIEW_FEISHU_APP_ID;
  const appSecret = process.env.CODEREVIEW_FEISHU_APP_SECRET;
  const chatId = process.env.CODEREVIEW_FEISHU_CHAT_ID;
  for (const secret of [appId, appSecret, chatId]) if (secret) core.setSecret(secret);
  if (!appId || !appSecret || !chatId) {
    if (required) core.setFailed('Feishu secret 未完整配置，队列保持待处理');
    else core.warning('未配置 Feishu secret，跳过通知');
    return null;
  }
  return { appId, appSecret, chatId };
}

async function tenantToken(env, core) {
  const response = await feishuWithRateLimit({
    apiPath: '/auth/v3/tenant_access_token/internal', method: 'POST',
    body: { app_id: env.appId, app_secret: env.appSecret }, core,
  });
  const token = response.ok && response.json.code === 0 && response.json.tenant_access_token;
  if (token) {
    core.setSecret(token);
    return { ok: true, token, rateLimited: false, retryAfterMs: null };
  }
  if (!response.rateLimited) {
    core.setFailed(`取 tenant_access_token 失败: ${feishuFailureSummary(response)}`);
  }
  return {
    ok: false, token: null, rateLimited: Boolean(response.rateLimited),
    retryAfterMs: response.rateLimited ? response.retryAfterMs : null,
  };
}

function contentWithDeliveryToken(content, token) {
  const card = JSON.parse(content);
  if (!card.header || !card.body || !Array.isArray(card.body.elements)) throw new Error('invalid interactive card content');
  card.header.subtitle = { tag: 'plain_text', content: token };
  return JSON.stringify(card);
}

function containsExactToken(value, token) {
  if (value === token) return true;
  if (Array.isArray(value)) return value.some((entry) => containsExactToken(entry, token));
  if (value && typeof value === 'object') return Object.values(value).some((entry) => containsExactToken(entry, token));
  return false;
}

function messageContainsToken(message, token) {
  let content = message && message.body && message.body.content;
  if (typeof content !== 'string') return false;
  try { content = JSON.parse(content); } catch (_error) { return false; }
  return containsExactToken(content, token);
}

async function findDeliveryInFeishuHistory({ env, token, record, core = noopCore(), nowMs = Date.now() }) {
  const sentSeconds = Math.floor(Date.parse(record.sentAt) / 1000);
  const startTime = sentSeconds - HISTORY_CLOCK_SKEW_SECONDS;
  const endTime = sentSeconds + HISTORY_AFTER_SEND_SECONDS;
  let pageToken = null;
  const matches = new Map();
  for (let page = 0; page < HISTORY_MAX_PAGES; page += 1) {
    const params = new URLSearchParams({
      container_id_type: 'chat',
      container_id: env.chatId,
      sort_type: 'ByCreateTimeDesc',
      page_size: String(HISTORY_PAGE_SIZE),
      start_time: String(startTime),
      end_time: String(endTime),
      card_msg_content_type: 'user_card_content',
    });
    if (pageToken) params.set('page_token', pageToken);
    const response = await feishuWithRateLimit({
      apiPath: `/im/v1/messages?${params.toString()}`, method: 'GET', body: null, token, core,
    });
    if (response.rateLimited) {
      return {
        ok: false, complete: false, matches: [...matches.values()],
        reason: feishuFailureSummary(response), rateLimited: true, retryAfterMs: response.retryAfterMs,
      };
    }
    if (!response.ok || response.json.code !== 0) {
      return { ok: false, complete: false, matches: [], reason: feishuFailureSummary(response) };
    }
    const data = response.json.data || {};
    for (const message of data.items || []) {
      const sender = message.sender || {};
      const exactSender = sender.sender_type === 'app' && sender.id === env.appId;
      if (exactSender && message.msg_type === 'interactive' && message.deleted !== true &&
          messageContainsToken(message, record.token) && validFeishuMessageId(message.message_id)) {
        matches.set(message.message_id, message);
        if (matches.size >= 2) {
          return { ok: true, complete: false, matches: [...matches.values()] };
        }
      }
    }
    if (!data.has_more) return { ok: true, complete: true, matches: [...matches.values()] };
    pageToken = data.page_token;
    if (!pageToken) return { ok: false, complete: false, matches: [...matches.values()], reason: 'has_more without page_token' };
  }
  return { ok: true, complete: false, matches: [...matches.values()], reason: 'history scan bound reached' };
}

function deterministicClaimId(repo, prNum, generation, eventKey) {
  return crypto.createHash('sha256').update(`claim:v3:${repo}#${prNum}:${generation}:${eventKey}`).digest('hex').slice(0, 32);
}

function requestUuid(eventKey, mode, attempt) {
  return crypto.createHash('sha256').update(`feishu:v3:${eventKey}:${mode}:${attempt}`).digest('hex').slice(0, 32);
}

function deliveryBase(ref, attempt, mode, sentAt, thread = {}) {
  return {
    version: DELIVERY_VERSION,
    repo: ref.repo,
    pr: ref.pr,
    eventKey: ref.eventKey,
    attempt,
    mode,
    targetRoot: mode === 'reply' ? thread.rootMid : null,
    threadGeneration: mode === 'root' ? thread.generation : null,
    threadClaimId: mode === 'root' ? thread.claimId : null,
    token: deliveryToken(ref.eventKey),
    sentAt,
  };
}

async function markDeliveryFailure(args, base, reason) {
  return appendDeliveryAndConfirm({
    ...args,
    record: canonicalDeliveryRecord({
      ...base,
      state: 'failed',
      messageId: null,
      nextCheckAt: null,
      reason,
    }),
  });
}

function historyRecoveryExhausted(record, nextAttempts, nowMs = Date.now()) {
  return nextAttempts >= HISTORY_MAX_ATTEMPTS || nowMs - Date.parse(record.sentAt) >= HISTORY_MAX_AGE_MS;
}

function rateLimitNextCheckAt(retryAfterMs, nowMs = Date.now()) {
  return new Date(nowMs + Math.max(1000, retryAfterMs || 0)).toISOString();
}

async function persistRateLimitedNotSent({
  github, context, core, ref, base, retryAfterMs, session = null, scope,
}) {
  const nextCheckAt = rateLimitNextCheckAt(retryAfterMs);
  const record = canonicalDeliveryRecord({
    ...base, state: 'not_sent', nextCheckAt, reason: 'rate_limited',
  });
  const persisted = await appendDeliveryAndConfirm({
    github, context, core, prNum: ref.pr, record, session,
  });
  core.setFailed(`Feishu ${scope} HTTP 429 有界重试耗尽，已持久化到 ${nextCheckAt} 由 watchdog 续投`);
  return { complete: false, retry: true, rateLimited: persisted.ok };
}

async function deferAmbiguousForRateLimit({
  github, context, core, ref, record, retryAfterMs, nowMs, session = null, scope,
}) {
  const nextCheckAt = rateLimitNextCheckAt(retryAfterMs, nowMs);
  const deferred = canonicalDeliveryRecord({
    ...record, state: 'uncertain', historyAttempts: record.historyAttempts,
    nextCheckAt, reason: 'history_rate_limited',
  });
  const persisted = await appendDeliveryAndConfirm({
    github, context, core, prNum: ref.pr, record: deferred, session,
  });
  core.setFailed(`Feishu ${scope} HTTP 429；historyAttempts 保持 ${record.historyAttempts}，延期到 ${nextCheckAt}`);
  return { complete: false, retry: true, rateLimited: persisted.ok };
}

async function recoverAmbiguousDelivery({ github, context, core, ref, record, env, nowMs = Date.now(), session = null }) {
  const dueAt = record.state === 'uncertain' ? Date.parse(record.nextCheckAt) : Date.parse(record.sentAt) + HISTORY_RETRY_DELAY_MS;
  if (nowMs < dueAt) {
    core.setFailed('Feishu 历史查询尚未到延迟重试时间；保持 uncertain 且不重发');
    return { complete: false, retry: true };
  }
  const tokenResult = await tenantToken(env, core);
  if (!tokenResult.ok) {
    if (tokenResult.rateLimited) {
      return deferAmbiguousForRateLimit({
        github, context, core, ref, record, retryAfterMs: tokenResult.retryAfterMs,
        nowMs: Date.now(), session, scope: 'tenant token',
      });
    }
    return { complete: false, retry: true };
  }
  const history = await findDeliveryInFeishuHistory({
    env, token: tokenResult.token, record, core, nowMs,
  });
  if (history.rateLimited) {
    return deferAmbiguousForRateLimit({
      github, context, core, ref, record, retryAfterMs: history.retryAfterMs,
      nowMs: Date.now(), session, scope: 'history',
    });
  }
  const candidates = [...new Set([
    ...record.candidateMessageIds,
    ...history.matches.map((message) => message.message_id),
  ])].sort();
  if (candidates.length > 1) {
    const manual = canonicalDeliveryRecord({
      ...record, state: 'manual', candidateMessageIds: candidates,
      nextCheckAt: null, reason: 'history_multiple',
    });
    await appendDeliveryAndConfirm({ github, context, core, prNum: ref.pr, record: manual, session });
    core.setFailed('Feishu 历史 exact token 多匹配，进入人工 delivery repair；事件保持未完成');
    return { complete: false, retry: false, manual: true };
  }
  if (!history.ok || !history.complete || history.matches.length === 0) {
    const attempts = record.historyAttempts + 1;
    const exhausted = historyRecoveryExhausted(record, attempts, nowMs);
    if (exhausted) {
      const manual = canonicalDeliveryRecord({
        ...record, state: 'manual', candidateMessageIds: candidates,
        historyAttempts: attempts, nextCheckAt: null, reason: 'history_exhausted',
      });
      await appendDeliveryAndConfirm({ github, context, core, prNum: ref.pr, record: manual, session });
      core.setFailed('Feishu 历史恢复达到 8 次或 24h 上限，转 manual 且不重发');
      return { complete: false, retry: false, manual: true };
    }
    const uncertain = canonicalDeliveryRecord({
      ...record,
      state: 'uncertain',
      candidateMessageIds: candidates,
      historyAttempts: attempts,
      nextCheckAt: new Date(nowMs + Math.min(
        HISTORY_RETRY_DELAY_MS * (2 ** Math.min(record.historyAttempts, 5)),
        HISTORY_MAX_BACKOFF_MS,
      )).toISOString(),
      reason: !history.ok ? 'history_error' : !history.complete ? 'history_incomplete' : 'history_zero',
    });
    await appendDeliveryAndConfirm({ github, context, core, prNum: ref.pr, record: uncertain, session });
    core.setFailed(`Feishu 历史暂不可确认(${uncertain.reason})；保持非终态退避且不会重发`);
    return { complete: false, retry: true };
  }
  const messageId = history.matches[0].message_id;
  if (record.mode === 'root') {
    const finalized = await finalizeThreadGeneration({
      github, context, core, prNum: ref.pr,
      generation: record.threadGeneration, claimId: record.threadClaimId, messageId, session,
    });
    if (!finalized.ok) return { complete: false, retry: false };
  }
  const done = canonicalDeliveryRecord({
    ...record, state: 'done', messageId, nextCheckAt: null, reason: null,
  });
  const appended = await appendDeliveryAndConfirm({ github, context, core, prNum: ref.pr, record: done, session });
  if (appended.ok) core.info('Feishu 历史 exact-token 唯一匹配，投递恢复完成');
  return { complete: appended.ok, retry: !appended.ok };
}

async function sendAttempt({ github, context, core, ref, prData, cls, env, attempt, mode, thread, session = null }) {
  const { buildCard, interactiveCardContent } = require('./report');
  let content;
  try {
    const card = buildCard(context.repo.repo, prData, cls, { atAuthor: mode === 'root', isReply: mode === 'reply' });
    content = contentWithDeliveryToken(interactiveCardContent(card), deliveryToken(ref.eventKey));
  } catch (error) {
    core.setFailed(`生成飞书卡片失败: ${errorMessage(error)}`);
    return { complete: false, retry: false };
  }
  const tokenResult = await tenantToken(env, core);
  if (!tokenResult.ok) {
    if (tokenResult.rateLimited) {
      return persistRateLimitedNotSent({
        github, context, core, ref,
        base: deliveryBase(ref, attempt, mode, new Date().toISOString(), thread),
        retryAfterMs: tokenResult.retryAfterMs, session, scope: 'tenant token',
      });
    }
    return { complete: false, retry: true };
  }
  const token = tokenResult.token;
  const sentAt = new Date().toISOString();
  const base = deliveryBase(ref, attempt, mode, sentAt, thread);
  const sending = canonicalDeliveryRecord({ ...base, state: 'sending' });
  const recorded = await appendDeliveryAndConfirm({ github, context, core, prNum: ref.pr, record: sending, session });
  if (!recorded.ok) return { complete: false, retry: true };

  const apiPath = mode === 'reply'
    ? `/im/v1/messages/${encodeURIComponent(thread.rootMid)}/reply`
    : '/im/v1/messages?receive_id_type=chat_id';
  const body = mode === 'reply'
    ? { msg_type: 'interactive', content, reply_in_thread: true, uuid: requestUuid(ref.eventKey, mode, attempt) }
    : { receive_id: env.chatId, msg_type: 'interactive', content, uuid: requestUuid(ref.eventKey, mode, attempt) };
  const result = await feishuWithRateLimit({ apiPath, method: 'POST', body, token, core });
  const messageId = result.json.data && result.json.data.message_id;
  if (result.ok && result.json.code === 0 && validFeishuMessageId(messageId)) {
    if (mode === 'root') {
      const finalized = await finalizeThreadGeneration({
        github, context, core, prNum: ref.pr,
        generation: thread.generation, claimId: thread.claimId, messageId, session,
      });
      if (!finalized.ok) return { complete: false, retry: true };
    }
    const done = canonicalDeliveryRecord({ ...base, state: 'done', messageId });
    const appended = await appendDeliveryAndConfirm({ github, context, core, prNum: ref.pr, record: done, session });
    if (appended.ok) core.info(`${mode === 'root' ? '根消息' : '话题回复'}发送成功`);
    return { complete: appended.ok, retry: !appended.ok };
  }

  if (result.rateLimited) {
    return persistRateLimitedNotSent({
      github, context, core, ref, base, retryAfterMs: result.retryAfterMs,
      session, scope: 'message',
    });
  }

  if (mode === 'reply' && shouldRecreateRootOnReplyFailure(result)) {
    const notSent = canonicalDeliveryRecord({ ...base, state: 'not_sent', reason: 'root_missing' });
    const appended = await appendDeliveryAndConfirm({ github, context, core, prNum: ref.pr, record: notSent, session });
    if (!appended.ok) return { complete: false, retry: false };
    core.warning(`原根消息明确已撤回/删除: ${feishuFailureSummary(result)}`);
    return { complete: false, retry: false, rootMissing: true, nextAttempt: attempt + 1 };
  }

  if (isKnownRootSendFailure(result)) {
    const notSent = canonicalDeliveryRecord({
      ...base, state: 'not_sent', reason: 'definitely_not_sent',
    });
    const persisted = await appendDeliveryAndConfirm({
      github, context, core, prNum: ref.pr, record: notSent, session,
    });
    if (!persisted.ok) return { complete: false, retry: true };
    if (mode === 'root') {
      const released = await releaseThreadGeneration({
        github, context, core, prNum: ref.pr,
        generation: thread.generation, claimId: thread.claimId, session,
      });
      if (!released.ok) return { complete: false, retry: true };
    }
    core.setFailed(`Feishu ${mode === 'root' ? '根消息' : '话题回复'}明确未发送，将由 watchdog 有界重试: ${feishuFailureSummary(result)}`);
    return { complete: false, retry: true };
  }

  // The POST result is ambiguous. The batch snapshot is intentionally stale here: a
  // concurrent/recovered comment may already have settled this exact sending attempt.
  // Re-read before appending uncertain, and never issue another POST for persisted sending.
  const freshLedger = await readDeliveryLedger({ github, context, core, prNum: ref.pr });
  if (!freshLedger.ok) return { complete: false, retry: true };
  const fresh = freshLedger.latest.get(ref.eventKey);
  if (session) session.latest = freshLedger.latest;
  if (!fresh || !sameDeliveryIdentity(fresh, sending)) {
    core.setFailed('Feishu POST 歧义后 delivery fresh reread 与本次 sending identity 不一致');
    return { complete: false, retry: true };
  }
  if (fresh.state === 'done' || fresh.state === 'skipped') {
    return { complete: true, retry: false };
  }
  if (fresh.state === 'manual' || fresh.state === 'failed') {
    return { complete: false, retry: false, manual: true };
  }
  if (fresh.state === 'not_sent') return { complete: false, retry: true };
  if (fresh.state === 'uncertain') return { complete: false, retry: true };
  if (fresh.state !== 'sending') {
    core.setFailed(`Feishu POST 歧义后出现非法 fresh state=${fresh.state}`);
    return { complete: false, retry: true };
  }
  const uncertain = canonicalDeliveryRecord({
    ...base,
    state: 'uncertain',
    historyAttempts: 0,
    nextCheckAt: new Date(Date.parse(sentAt) + HISTORY_RETRY_DELAY_MS).toISOString(),
    reason: 'ambiguous',
  });
  await appendDeliveryAndConfirm({ github, context, core, prNum: ref.pr, record: uncertain, session });
  core.setFailed(`Feishu 发送结果不确定，已转历史 exact-token 恢复且不会重发: ${feishuFailureSummary(result)}`);
  return { complete: false, retry: true };
}

async function prepareRootAttempt({ github, context, core, ref, attempt, current, allowFromFinal = false, session = null }) {
  const generation = current.kind === 'none' ? 1 : current.generation + 1;
  const claimId = deterministicClaimId(ref.repo, ref.pr, generation, ref.eventKey);
  const preparing = canonicalDeliveryRecord({
    ...deliveryBase(ref, attempt, 'root', null, { generation, claimId }),
    state: 'preparing',
  });
  const recorded = await appendDeliveryAndConfirm({ github, context, core, prNum: ref.pr, record: preparing, session });
  if (!recorded.ok) return { ok: false };
  const reserved = await reserveThreadGeneration({
    github, context, core, prNum: ref.pr, generation, claimId, allowFromFinal, session,
  });
  return reserved.ok ? { ok: true, generation, claimId, preparing } : { ok: false };
}

async function deliverClassifiedEvent({ github, context, core, ref, prData, cls, env, latest, session = null }) {
  if (latest && ['done', 'skipped'].includes(latest.state)) return { complete: true, retry: false };
  if (latest && ['manual', 'failed'].includes(latest.state)) {
    core.setFailed(`eventKey=${ref.eventKey} 等待人工 delivery repair`);
    return { complete: false, retry: false, manual: true };
  }
  if (latest && ['sending', 'uncertain'].includes(latest.state)) {
    return recoverAmbiguousDelivery({ github, context, core, ref, record: latest, env, session });
  }

  let current = session ? session.threadState : await readThreadState({ github, context, core, prNum: ref.pr });
  if (!current.ok) return { complete: false, retry: false };
  let attempt = latest && latest.state === 'not_sent' ? latest.attempt + 1 : 1;
  if (latest && latest.state === 'retrying') {
    if (latest.mode === 'reply') {
      if (current.kind !== 'final' || current.messageId !== latest.targetRoot) {
        core.setFailed('operator retry 与当前 reply root 不一致');
        return { complete: false, retry: false, manual: true };
      }
      return sendAttempt({
        github, context, core, ref, prData, cls, env, attempt: latest.attempt + 1,
        mode: 'reply', thread: { rootMid: latest.targetRoot }, session,
      });
    }
    if (latest.mode !== 'root' || current.kind !== 'released' || current.generation !== latest.threadGeneration ||
        current.claimId !== latest.threadClaimId) {
      core.setFailed('operator retry 与 released root thread state 不一致');
      return { complete: false, retry: false, manual: true };
    }
    attempt = latest.attempt + 1;
    const prepared = await prepareRootAttempt({ github, context, core, ref, attempt, current, session });
    if (!prepared.ok) return { complete: false, retry: true };
    return sendAttempt({
      github, context, core, ref, prData, cls, env, attempt, mode: 'root',
      thread: { generation: prepared.generation, claimId: prepared.claimId }, session,
    });
  }
  if (latest && latest.state === 'preparing') {
    const matchesPending = current.kind === 'pending' && current.generation === latest.threadGeneration &&
      current.claimId === latest.threadClaimId;
    if (!matchesPending) {
      const predecessorGeneration = current.kind === 'none' ? 0 : current.generation;
      if (predecessorGeneration !== latest.threadGeneration - 1 || current.kind === 'pending') {
        core.setFailed('preparing delivery 无法接管不匹配的 thread state');
        return { complete: false, retry: false, manual: true };
      }
      const reserved = await reserveThreadGeneration({
        github, context, core, prNum: ref.pr,
        generation: latest.threadGeneration, claimId: latest.threadClaimId,
        allowFromFinal: current.kind === 'final', session,
      });
      if (!reserved.ok) return { complete: false, retry: true };
    }
    return sendAttempt({
      github, context, core, ref, prData, cls, env, attempt: latest.attempt, mode: 'root',
      thread: { generation: latest.threadGeneration, claimId: latest.threadClaimId }, session,
    });
  }
  if (latest && latest.state === 'not_sent') {
    if (latest.reason === 'rate_limited') {
      if (Date.now() < Date.parse(latest.nextCheckAt)) {
        core.setFailed('Feishu 429 retry 尚未到 nextCheckAt；保持 durable not_sent');
        return { complete: false, retry: true };
      }
      if (latest.mode === 'reply') {
        if (current.kind !== 'final' || current.messageId !== latest.targetRoot) {
          core.setFailed('rate_limited reply 与当前 thread state 不一致');
          return { complete: false, retry: false, manual: true };
        }
        return sendAttempt({
          github, context, core, ref, prData, cls, env, attempt,
          mode: 'reply', thread: { rootMid: latest.targetRoot }, session,
        });
      }
      if (latest.mode !== 'root' || current.kind !== 'pending' ||
          current.generation !== latest.threadGeneration || current.claimId !== latest.threadClaimId) {
        core.setFailed('rate_limited root 与 matching pending thread state 不一致');
        return { complete: false, retry: false, manual: true };
      }
      return sendAttempt({
        github, context, core, ref, prData, cls, env, attempt,
        mode: 'root', thread: { generation: latest.threadGeneration, claimId: latest.threadClaimId }, session,
      });
    }
    if (latest.reason === 'definitely_not_sent') {
      if (latest.attempt >= DEFINITE_SEND_MAX_ATTEMPTS) {
        const manual = canonicalDeliveryRecord({
          ...latest, state: 'manual', nextCheckAt: null, reason: 'definitely_not_sent_exhausted',
        });
        const appended = await appendDeliveryAndConfirm({
          github, context, core, prNum: ref.pr, record: manual, session,
        });
        core.setFailed('Feishu 明确未发送已重试 3 次，转 manual dead-letter');
        return { complete: false, retry: !appended.ok, manual: appended.ok };
      }
      if (latest.mode === 'reply') {
        if (current.kind !== 'final' || current.messageId !== latest.targetRoot) {
          core.setFailed('definitely_not_sent reply 与当前 thread state 不一致');
          return { complete: false, retry: false, manual: true };
        }
        return sendAttempt({
          github, context, core, ref, prData, cls, env, attempt,
          mode: 'reply', thread: { rootMid: latest.targetRoot }, session,
        });
      }
      if (latest.mode !== 'root') {
        core.setFailed('definitely_not_sent delivery mode 非法');
        return { complete: false, retry: false, manual: true };
      }
      const matchingPending = current.kind === 'pending' && current.generation === latest.threadGeneration &&
        current.claimId === latest.threadClaimId;
      if (matchingPending) {
        const released = await releaseThreadGeneration({
          github, context, core, prNum: ref.pr,
          generation: latest.threadGeneration, claimId: latest.threadClaimId, session,
        });
        if (!released.ok) return { complete: false, retry: true };
        current = session ? session.threadState : await readThreadState({ github, context, core, prNum: ref.pr });
      }
      if (current.kind !== 'released' || current.generation !== latest.threadGeneration ||
          current.claimId !== latest.threadClaimId) {
        core.setFailed('definitely_not_sent delivery 无法收敛到 matching released thread state');
        return { complete: false, retry: false, manual: true };
      }
    } else if (latest.mode !== 'reply' || current.kind !== 'final' || current.messageId !== latest.targetRoot) {
      core.setFailed('root_missing delivery 与当前 thread state 不一致');
      return { complete: false, retry: false, manual: true };
    }
    const prepared = await prepareRootAttempt({
      github, context, core, ref, attempt, current, allowFromFinal: current.kind === 'final', session,
    });
    if (!prepared.ok) return { complete: false, retry: true };
    return sendAttempt({
      github, context, core, ref, prData, cls, env, attempt, mode: 'root',
      thread: { generation: prepared.generation, claimId: prepared.claimId }, session,
    });
  }

  if (current.kind === 'pending') {
    core.setFailed(`检测到无 delivery owner 的 pending generation=${current.generation}，需 trusted repair`);
    return { complete: false, retry: false };
  }
  if (current.kind === 'final') {
    const reply = await sendAttempt({
      github, context, core, ref, prData, cls, env, attempt, mode: 'reply',
      thread: { rootMid: current.messageId }, session,
    });
    if (!reply.rootMissing) return reply;
    current = session ? session.threadState : await readThreadState({ github, context, core, prNum: ref.pr });
    if (!current.ok || current.kind !== 'final') return { complete: false, retry: false };
    const prepared = await prepareRootAttempt({
      github, context, core, ref, attempt: reply.nextAttempt, current, allowFromFinal: true, session,
    });
    if (!prepared.ok) return { complete: false, retry: true };
    return sendAttempt({
      github, context, core, ref, prData, cls, env, attempt: reply.nextAttempt, mode: 'root',
      thread: { generation: prepared.generation, claimId: prepared.claimId }, session,
    });
  }
  const prepared = await prepareRootAttempt({ github, context, core, ref, attempt, current, session });
  if (!prepared.ok) return { complete: false, retry: true };
  return sendAttempt({
    github, context, core, ref, prData, cls, env, attempt, mode: 'root',
    thread: { generation: prepared.generation, claimId: prepared.claimId }, session,
  });
}

function normalizeResolveArgs(githubOrOptions, contextArg, coreArg) {
  if (githubOrOptions && githubOrOptions.github) {
    return { github: githubOrOptions.github, context: githubOrOptions.context, core: githubOrOptions.core || noopCore() };
  }
  return { github: githubOrOptions, context: contextArg, core: coreArg || noopCore() };
}

function safePositiveInteger(value) {
  const raw = String(value == null ? '' : value);
  if (!/^[1-9][0-9]*$/.test(raw)) return null;
  const number = Number(raw);
  return Number.isSafeInteger(number) ? number : null;
}

const MAX_RESOLVED_EVENT_REFS = 20;
const CODEX_CAPTURE_WORKFLOW_NAME = 'Codex Review -> Feishu Capture';
const CODEX_CAPTURE_WORKFLOW_PATH = '.github/workflows/codex-review-feishu.yml';
const SOURCE_RECONCILE_MAX_EVENTS = 20;
const SOURCE_RECONCILE_PAGE_SIZE = 100;
const SOURCE_RECONCILE_MAX_PAGES = 10;

function parseCompactEventRefs(value, core = noopCore()) {
  let input = value;
  if (typeof value === 'string') {
    try { input = JSON.parse(value); } catch (_error) { core.setFailed('event_refs_json 不是合法 JSON'); return null; }
  }
  if (!Array.isArray(input) || input.length === 0 || input.length > MAX_RESOLVED_EVENT_REFS) {
    core.setFailed(`event refs 数量必须为 1..${MAX_RESOLVED_EVENT_REFS}`);
    return null;
  }
  try {
    const refs = input.map((entry) => canonicalEventRef(entry));
    if (new Set(refs.map((ref) => `${ref.pr}:${ref.eventKey}`)).size !== refs.length) throw new Error('duplicate ref');
    return refs;
  } catch (error) {
    core.setFailed(`compact immutable event refs 非 canonical: ${errorMessage(error)}`);
    return null;
  }
}

function dispatchEventRef(context, core = noopCore()) {
  const payload = context.payload.client_payload || {};
  if (!payload.event_ref) return null;
  const refs = parseCompactEventRefs([payload.event_ref], core);
  if (!refs) return null;
  const ref = refs[0];
  if (ref.repo !== repositoryName(context) || String(payload.pr_number) !== String(ref.pr)) {
    core.setFailed('repository_dispatch compact ref 的 repo/pr 绑定非法');
    return null;
  }
  return ref;
}

async function resolvePrNumber(githubOrOptions, contextArg, coreArg) {
  const { github, context, core } = normalizeResolveArgs(githubOrOptions, contextArg, coreArg);
  if (context.eventName === 'repository_dispatch') {
    const number = safePositiveInteger(context.payload.client_payload && context.payload.client_payload.pr_number);
    if (!number) core.setFailed('repository_dispatch pr_number 非 canonical safe integer');
    if (number && context.payload.client_payload && context.payload.client_payload.event_ref) {
      const ref = dispatchEventRef(context, core);
      if (!ref || ref.pr !== number) return null;
    }
    return number;
  }
  const direct = context.payload.pull_request && context.payload.pull_request.number;
  if (context.eventName !== 'check_run' && Number.isInteger(direct) && direct > 0) return direct;
  const checkRun = context.payload.check_run;
  if (!checkRun) {
    core.setFailed('事件不包含 pull_request、check_run 或受信 drain payload');
    return null;
  }
  const refs = await resolveOfficialCodexEventRefs(github, context, core);
  if (!refs) return null;
  if (refs.length !== 1) {
    core.setFailed(`Codex 事件必须唯一关联一个 PR，实际=${refs.map((ref) => ref.pr).join(',') || 'none'}`);
    return null;
  }
  return refs[0].pr;
}

async function resolveOfficialCodexEventRefs(githubOrOptions, contextArg, coreArg) {
  const { github, context, core } = normalizeResolveArgs(githubOrOptions, contextArg, coreArg);
  if (!['check_run', 'pull_request_review'].includes(context.eventName)) {
    core.setFailed(`event=${context.eventName} 不能解析 source immutable refs`);
    return null;
  }
  if (context.eventName === 'pull_request_review') {
    const number = safePositiveInteger(context.payload.pull_request && context.payload.pull_request.number);
    if (!number) { core.setFailed('review event PR number 非 canonical'); return null; }
    const prData = await resolvePr(github, context, { number }, core);
    if (!prData || prData.state !== 'open' || prData.base !== 'main') return [];
    const ref = eventRefFromContext(context, prData, core);
    return ref && ref.headSha === prData.headSha ? [ref] : null;
  }
  const hintedId = safePositiveInteger(context.payload.check_run && context.payload.check_run.id);
  if (!hintedId) {
    core.setFailed('check_run hint 仅允许 canonical immutable ID');
    return null;
  }
  const fetched = await withGithubRetry({
    core, label: `按 source immutable ID 重取 check_run ${hintedId}`,
    operation: () => github.rest.checks.get({
      owner: context.repo.owner, repo: context.repo.repo, check_run_id: hintedId,
      request: { timeout: GITHUB_REQUEST_TIMEOUT_MS },
    }),
  });
  if (!fetched.ok) return null;
  return resolveFreshCheckRunRefs({
    github, context, core, checkRun: fetched.value.data, expectedId: hintedId,
  });
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
      owner: context.repo.owner, repo: context.repo.repo, pull_number: number,
      request: { timeout: GITHUB_REQUEST_TIMEOUT_MS },
    }),
  });
  if (!result.ok) return null;
  const full = result.value.data;
  if (!full || Number(full.number) !== number) {
    core.setFailed(`读取 PR #${number} 返回了不匹配的 PR number`);
    return null;
  }
  return {
    number: full.number,
    title: full.title,
    url: full.html_url,
    author: full.user.login,
    base: full.base.ref,
    head: full.head.ref,
    headSha: full.head.sha,
    state: full.state,
  };
}

function hasCodexName(value) {
  return /\bcodex\b/i.test(String(value || ''));
}

function isCodexUser(user) {
  const login = String(user && user.login || '').toLowerCase();
  return String(user && user.id || '') === CODEX_REVIEW_USER_ID &&
    login === CODEX_REVIEW_LOGIN && String(user && user.type || '').toLowerCase() === 'bot';
}

function isCodexCheckRun(checkRun) {
  return Boolean(checkRun && checkRun.app && checkRun.app.slug === CODEX_CHECK_APP_SLUG &&
    (!checkRun.check_suite || !checkRun.check_suite.app || checkRun.check_suite.app.slug === CODEX_CHECK_APP_SLUG));
}

async function resolveFreshCheckRunRefs({ github, context, core = noopCore(), checkRun, expectedId = null }) {
  const createdAt = checkRun && (checkRun.completed_at || checkRun.updated_at);
  if (!checkRun || (expectedId !== null && String(checkRun.id) !== String(expectedId)) ||
      !safePositiveInteger(checkRun.id) || !isCodexCheckRun(checkRun) || checkRun.status !== 'completed' ||
      !validSha(checkRun.head_sha) || !createdAt) {
    core.setFailed('fresh check_run immutable ID/Codex identity/status/head/time 非法');
    return null;
  }
  const embeddedPulls = Array.isArray(checkRun.pull_requests) ? checkRun.pull_requests : [];
  const associated = await withGithubRetry({
    core, label: `按 fresh check SHA ${checkRun.head_sha} 解析全部关联 PR`,
    setFailedOnExhausted: false,
    operation: () => github.rest.repos.listPullRequestsAssociatedWithCommit({
      owner: context.repo.owner, repo: context.repo.repo, commit_sha: checkRun.head_sha,
      page: 1, per_page: MAX_RESOLVED_EVENT_REFS + 1, request: { timeout: GITHUB_REQUEST_TIMEOUT_MS },
    }),
  });
  let associatedPulls = [];
  if (!associated.ok) {
    const embeddedNumbers = [...new Set(embeddedPulls
      .map((pr) => Number(pr && pr.number))
      .filter((number) => Number.isSafeInteger(number) && number > 0))];
    if (embeddedNumbers.length === 0) {
      core.setFailed('check_run association API 不可用且 fresh embedded PR 为空；保持 source retry');
      return null;
    }
    core.warning(`check_run association API 不可用，降级使用 ${embeddedNumbers.length} 个 fresh embedded PR`);
  } else {
    associatedPulls = Array.isArray(associated.value.data) ? associated.value.data : [];
  }
  if (associatedPulls.length > MAX_RESOLVED_EVENT_REFS) {
    core.setFailed(`check_run association API 结果超过 cap=${MAX_RESOLVED_EVENT_REFS}`);
    return null;
  }
  const numbers = [...new Set([...embeddedPulls, ...associatedPulls]
    .map((pr) => Number(pr && pr.number))
    .filter((number) => Number.isSafeInteger(number) && number > 0))].sort((a, b) => a - b);
  if (numbers.length > MAX_RESOLVED_EVENT_REFS) {
    core.setFailed(`check_run fresh embedded + association union 超过 cap=${MAX_RESOLVED_EVENT_REFS}`);
    return null;
  }
  const refs = [];
  for (const number of numbers) {
    const prData = await resolvePr(github, context, { number }, core);
    if (!prData) return null;
    if (prData.state !== 'open' || prData.base !== 'main') continue;
    refs.push(canonicalEventRef({
      version: EVENT_VERSION, eventType: 'check_run', eventId: String(checkRun.id),
      repo: repositoryName(context), pr: prData.number, headSha: checkRun.head_sha, createdAt,
    }));
  }
  return refs;
}

function isCodexPullRequestReview(review) {
  return Boolean(review && isCodexUser(review.user));
}

async function listCommentsForReview(github, context, core, prNum, reviewId) {
  const result = await withGithubRetry({
    core,
    label: '读取 Codex review comments',
    operation: () => github.paginate(github.rest.pulls.listReviewComments, {
      owner: context.repo.owner, repo: context.repo.repo, pull_number: prNum,
      per_page: 100, request: { timeout: GITHUB_REQUEST_TIMEOUT_MS },
    }),
  });
  if (!result.ok) return null;
  return result.value.filter((comment) => String(comment.pull_request_review_id || comment.review_id || '') === String(reviewId || ''));
}

async function resolvedPrForNotification(github, context, core) {
  const envNumber = Number(process.env.RESOLVED_PR_NUMBER);
  let number = Number.isSafeInteger(envNumber) && envNumber > 0 ? envNumber : null;
  if (!number) number = await resolvePrNumber(github, context, core);
  if (!number) return null;
  const eventNumber = context.payload.pull_request && context.payload.pull_request.number;
  if (eventNumber && Number(eventNumber) !== number) {
    core.setFailed(`resolved PR #${number} 与 event PR #${eventNumber} 不一致`);
    return null;
  }
  return resolvePr(github, context, { number }, core);
}

function eventRefFromContext(context, prData, core = noopCore()) {
  const repo = repositoryName(context);
  let input;
  if (context.eventName === 'check_run') {
    core.setFailed('raw check_run 禁止直接构造 ref；必须使用 checks.get fresh resolver');
    return null;
  } else if (context.eventName === 'pull_request_review') {
    const review = context.payload.review;
    if (!isCodexPullRequestReview(review)) {
      core.setFailed('拒绝入队非 Codex pull_request_review');
      return null;
    }
    const eventHead = context.payload.pull_request && context.payload.pull_request.head && context.payload.pull_request.head.sha;
    if (!eventHead || review.commit_id !== eventHead) {
      core.setFailed('review commit_id 与 event PR head 不一致');
      return null;
    }
    input = {
      version: EVENT_VERSION, eventType: 'pull_request_review', eventId: String(review.id || ''),
      repo, pr: prData.number,
      headSha: eventHead,
      createdAt: review.submitted_at || review.updated_at,
    };
  } else {
    core.setFailed(`event=${context.eventName} 不能入队`);
    return null;
  }
  try { return canonicalEventRef(input); }
  catch (error) { core.setFailed(`事件引用非法: ${errorMessage(error)}`); return null; }
}

async function enqueueOfficialCodexEventWithinBudget({ github, context, core }) {
  const serialized = process.env.RESOLVED_EVENT_REFS;
  if (serialized) {
    const refs = parseCompactEventRefs(serialized, core);
    if (!refs) return null;
    const confirmed = [];
    for (const ref of refs) {
      const hydrated = await rehydrateEvent({ github, context, core, ref });
      if (!hydrated.ok) continue;
      const result = await enqueueEventRef({ github, context, core, ref });
      if (result.ok) confirmed.push(ref);
    }
    return confirmed;
  }
  if (context.eventName === 'check_run') {
    const refs = await resolveOfficialCodexEventRefs(github, context, core);
    if (!refs) return null;
    if (refs.length !== 1) {
      core.setFailed(`direct check_run 入队必须唯一关联一个 OPEN main PR，实际=${refs.map((ref) => ref.pr).join(',') || 'none'}`);
      return null;
    }
    const result = await enqueueEventRef({ github, context, core, ref: refs[0] });
    return result.ok ? refs[0] : null;
  }
  const prData = await resolvedPrForNotification(github, context, core);
  if (!prData) return null;
  if (prData.base !== 'main' || prData.state !== 'open') {
    core.info('PR 非 OPEN main，跳过入队');
    return null;
  }
  const ref = eventRefFromContext(context, prData, core);
  if (!ref || prData.headSha !== ref.headSha) {
    if (ref) core.setFailed('事件 head SHA 与 canonical PR head 不一致');
    return null;
  }
  const result = await enqueueEventRef({ github, context, core, ref });
  return result.ok ? ref : null;
}

async function rehydrateEvent({ github, context, core, ref }) {
  const { classifyCheckRun, classifyPullRequestReview, shouldNotify } = require('./report');
  if (ref.repo !== repositoryName(context)) {
    core.setFailed('event ref repo 与 workflow repo 不一致');
    return { ok: false };
  }
  const prData = await resolvePr(github, context, { number: ref.pr }, core);
  if (!prData) return { ok: false, retry: true };
  if (prData.state !== 'open' || prData.base !== 'main') {
    return { ok: true, notify: false, reason: 'stale_pr' };
  }
  if (ref.eventType === 'check_run') {
    const id = safePositiveInteger(ref.eventId);
    if (!id) { core.setFailed('check_run immutable ID 超出 safe integer'); return { ok: false, retry: false, manualReason: 'invalid_event_id' }; }
    const fetched = await withGithubRetry({
      core,
      label: `按 immutable ID 重取 check_run ${ref.eventId}`,
      operation: () => github.rest.checks.get({
        owner: context.repo.owner, repo: context.repo.repo, check_run_id: id,
        request: { timeout: GITHUB_REQUEST_TIMEOUT_MS },
      }),
    });
    if (!fetched.ok) return { ok: false, retry: true };
    const checkRun = fetched.value.data;
    const createdAt = checkRun.completed_at || checkRun.updated_at;
    if (!checkRun || String(checkRun.id) !== ref.eventId || checkRun.head_sha !== ref.headSha ||
        createdAt !== ref.createdAt || !isCodexCheckRun(checkRun) || checkRun.status !== 'completed') {
      core.setFailed('immutable check_run 与队列 ref 的 Codex/head/time 绑定不一致');
      return { ok: false, retry: false, manualReason: 'immutable_mismatch' };
    }
    const associated = await withGithubRetry({
      core, label: `重验 check_run ${ref.eventId} 关联 PR`,
      setFailedOnExhausted: false,
      operation: () => github.rest.repos.listPullRequestsAssociatedWithCommit({
        owner: context.repo.owner, repo: context.repo.repo, commit_sha: checkRun.head_sha,
        page: 1, per_page: MAX_RESOLVED_EVENT_REFS + 1, request: { timeout: GITHUB_REQUEST_TIMEOUT_MS },
      }),
    });
    const embeddedPulls = Array.isArray(checkRun.pull_requests) ? checkRun.pull_requests : [];
    const embeddedNumbers = new Set(embeddedPulls.map((pr) => Number(pr && pr.number))
      .filter((number) => Number.isSafeInteger(number) && number > 0));
    let associatedPulls = [];
    if (!associated.ok) {
      if (!embeddedNumbers.has(ref.pr)) {
        core.setFailed('check_run association API 不可用且 fresh embedded 无法证明目标 PR；保持 rehydrate retry');
        return { ok: false, retry: true };
      }
      core.warning(`check_run ${ref.eventId} association API 不可用，降级使用 fresh embedded PR #${ref.pr}`);
    } else {
      associatedPulls = Array.isArray(associated.value.data) ? associated.value.data : [];
    }
    if (associatedPulls.length > MAX_RESOLVED_EVENT_REFS) {
      core.setFailed(`check_run rehydrate 关联 PR 超过 cap=${MAX_RESOLVED_EVENT_REFS}`);
      return { ok: false, retry: false, manualReason: 'association_overflow' };
    }
    const linked = new Set([...embeddedPulls, ...associatedPulls].map((pr) => Number(pr && pr.number))
      .filter((number) => Number.isSafeInteger(number) && number > 0));
    if (linked.size > MAX_RESOLVED_EVENT_REFS) {
      core.setFailed(`check_run rehydrate fresh embedded + association union 超过 cap=${MAX_RESOLVED_EVENT_REFS}`);
      return { ok: false, retry: false, manualReason: 'association_overflow' };
    }
    if (!linked.has(ref.pr)) {
      core.setFailed('immutable check_run 与队列 ref 的 Codex/repo/PR/head/time 绑定不一致');
      return { ok: false, retry: false, manualReason: 'immutable_mismatch' };
    }
    const cls = classifyCheckRun(checkRun, process.env);
    return { ok: true, notify: shouldNotify(cls), reason: 'not_notifiable', cls, prData };
  }
  const id = safePositiveInteger(ref.eventId);
  if (!id) { core.setFailed('review immutable ID 超出 safe integer'); return { ok: false, retry: false, manualReason: 'invalid_event_id' }; }
  const fetched = await withGithubRetry({
    core,
    label: `按 immutable ID 重取 review ${ref.eventId}`,
    operation: () => github.rest.pulls.getReview({
      owner: context.repo.owner, repo: context.repo.repo, pull_number: ref.pr, review_id: id,
      request: { timeout: GITHUB_REQUEST_TIMEOUT_MS },
    }),
  });
  if (!fetched.ok) return { ok: false, retry: true };
  const review = fetched.value.data;
  if (String(review.id) !== ref.eventId || review.submitted_at !== ref.createdAt ||
      review.commit_id !== ref.headSha || !isCodexPullRequestReview(review)) {
    core.setFailed('immutable review 与队列 ref 的 Codex/repo/PR/head/time 绑定不一致');
    return { ok: false, retry: false, manualReason: 'immutable_mismatch' };
  }
  const comments = await listCommentsForReview(github, context, core, ref.pr, review.id);
  if (!comments) return { ok: false, retry: true };
  const cls = classifyPullRequestReview(review, comments, process.env);
  return { ok: true, notify: shouldNotify(cls), reason: 'not_notifiable', cls, prData };
}

async function scheduleDrain({ github, context, core, prNum, ref = null }) {
  const clientPayload = { pr_number: String(prNum) };
  if (ref) clientPayload.event_ref = canonicalEventRef(ref);
  const result = await withGithubRetry({
    core,
    label: '调度下一批 Codex 飞书队列消费',
    operation: () => github.rest.repos.createDispatchEvent({
      owner: context.repo.owner,
      repo: context.repo.repo,
      event_type: 'codex-review-feishu-drain',
      client_payload: clientPayload,
      request: { timeout: GITHUB_REQUEST_TIMEOUT_MS },
    }),
  });
  return result.ok;
}

async function repairDelivery({
  github, context, core = noopCore(), prNum, eventKey, messageId,
  action = 'select_mid', runId, operator, privateKey, keyring, keyId,
}) {
  if (!validEventKey(eventKey) || !['select_mid', 'retry', 'discard'].includes(action) ||
      (action === 'select_mid' ? !validFeishuMessageId(messageId) : Boolean(messageId))) {
    core.setFailed('delivery repair action/eventKey/message_id 组合非法');
    return { ok: false };
  }
  const queue = await readEventQueue({ github, context, core, prNum });
  if (!queue.ok || !queue.events.some(({ ref }) => ref.eventKey === eventKey)) {
    core.setFailed('delivery repair eventKey 不在受信 durable queue');
    return { ok: false };
  }
  const ledger = await readDeliveryLedger({ github, context, core, prNum });
  const latest = ledger.ok && ledger.latest.get(eventKey);
  if (!latest || latest.state !== 'manual') {
    core.setFailed('delivery repair 仅允许完成 history_multiple manual 状态');
    return { ok: false };
  }
  if (action === 'select_mid' && !latest.candidateMessageIds.includes(messageId)) {
    core.setFailed('delivery repair selected message_id 不在 manual candidate set');
    return { ok: false };
  }
  if (action === 'retry' &&
      (!['definitely_not_sent_exhausted', 'history_exhausted'].includes(latest.reason) ||
       latest.candidateMessageIds.length !== 0)) {
    core.setFailed('retry 仅允许无候选的明确未发送/history_exhausted；存在 exact-token 候选时禁止重发');
    return { ok: false };
  }
  if (!/^[A-Za-z0-9_-]{1,100}$/.test(String(runId || '')) ||
      !/^[A-Za-z0-9_.-]{1,100}$/.test(String(operator || '')) ||
      !privateKeyMatchesKeyring(privateKey, keyring, keyId)) {
    core.setFailed('delivery repair signer/run/operator 或 keyring/keyId 非法');
    return { ok: false };
  }
  core.setSecret(privateKey);
  if (latest.mode === 'root' && action === 'select_mid') {
    const finalized = await finalizeThreadGeneration({
      github, context, core, prNum,
      generation: latest.threadGeneration, claimId: latest.threadClaimId, messageId,
    });
    if (!finalized.ok) return { ok: false };
  }
  if (latest.mode === 'root' && ['discard', 'retry'].includes(action)) {
    const thread = await readThreadState({ github, context, core, prNum });
    if (thread.kind === 'pending' && thread.generation === latest.threadGeneration &&
        thread.claimId === latest.threadClaimId) {
      const released = await releaseThreadGeneration({
        github, context, core, prNum,
        generation: latest.threadGeneration, claimId: latest.threadClaimId,
      });
      if (!released.ok) return { ok: false };
    }
  }
  const repair = {
    version: 1,
    repo: latest.repo,
    pr: latest.pr,
    eventKey: latest.eventKey,
    priorManualHash: canonicalHash(latest),
    priorReason: latest.reason,
    candidateHash: candidateSetHash(latest.candidateMessageIds),
    action,
    messageId: action === 'select_mid' ? messageId : null,
    runId: String(runId),
    operator: String(operator),
    keyId,
  };
  return appendDeliveryAndConfirm({
    github, context, core, prNum,
    record: canonicalDeliveryRecord({
      ...latest,
      state: action === 'select_mid' ? 'done' : action === 'retry' ? 'retrying' : 'skipped',
      messageId: action === 'select_mid' ? messageId : null,
      nextCheckAt: null,
      reason: action === 'select_mid' ? null : action === 'retry' ? 'operator_retry' : 'operator_discard',
    }),
    repair,
    repairPrivateKey: privateKey,
    repairPublicKeys: keyring,
  });
}

async function repairOrphanThread({
  github, context, core = noopCore(), prNum, current, messageId, release,
  runId, operator, privateKey, keyring, keyId,
}) {
  const comments = current && current.comments;
  const thread = current || await readThreadState({ github, context, core, prNum, comments });
  if (!thread.ok || thread.kind !== 'pending') {
    core.setFailed('thread repair 仅允许最高 generation 为 pending');
    return { ok: false };
  }
  const ledger = await readDeliveryLedger({ github, context, core, prNum, comments });
  if (!ledger.ok) return { ok: false };
  const owner = [...ledger.latest.values()].find((record) =>
    record.mode === 'root' &&
    record.threadGeneration === thread.generation && record.threadClaimId === thread.claimId);
  if (owner) {
    core.setFailed(`pending generation/claim 由 delivery eventKey=${owner.eventKey} 持有；必须填写 event_key 走 signed delivery repair`);
    return { ok: false, ownerEventKey: owner.eventKey };
  }
  const action = release ? 'release' : 'finalize';
  if ((action === 'finalize' ? !validFeishuMessageId(messageId) : Boolean(messageId)) ||
      !/^[A-Za-z0-9_-]{1,100}$/.test(String(runId || '')) ||
      !/^[A-Za-z0-9_.-]{1,100}$/.test(String(operator || '')) ||
      !privateKeyMatchesKeyring(privateKey, keyring, keyId)) {
    core.setFailed('orphan thread repair action/message_id/signer/run/operator 非法');
    return { ok: false };
  }
  core.setSecret(privateKey);
  const prior = canonicalThreadState(thread.state);
  const expected = canonicalThreadState({
    ...prior,
    state: action === 'finalize' ? 'final' : 'released',
    messageId: action === 'finalize' ? messageId : null,
  });
  return appendStateAndConfirm({
    github, context, core, prNum, expected,
    threadRepair: {
      version: 1, repo: expected.repo, pr: expected.pr,
      priorThreadHash: canonicalHash(prior), action,
      messageId: expected.messageId, runId: String(runId), operator: String(operator), keyId,
    },
    repairPrivateKey: privateKey,
    repairPublicKey: keyring,
  });
}

function setDrainOutputs(core, needsDispatch, mode) {
  core.setOutput('needs_dispatch', needsDispatch ? 'true' : 'false');
  core.setOutput('continuation_mode', mode || 'none');
}

async function drainQueuedCodexEventsWithinBudget({ github, context, core, batchSize = DRAIN_BATCH_SIZE }) {
  setDrainOutputs(core, false, 'none');
  const prNum = safePositiveInteger(process.env.RESOLVED_PR_NUMBER) || await resolvePrNumber(github, context, core);
  if (!prNum) return;
  const queue = await readEventQueue({ github, context, core, prNum });
  if (!queue.ok) return;
  const env = readFeishuEnv(core, queue.events.length > 0);
  if (!env && queue.events.length) {
    setDrainOutputs(core, true, 'retry');
    return;
  }
  let processed = 0;
  let blockedRetry = false;
  let manualSeen = false;
  for (const queued of queue.events) {
    const ledger = await readDeliveryLedger({ github, context, core, prNum });
    if (!ledger.ok) return;
    const latest = ledger.latest.get(queued.ref.eventKey);
    if (latest && ['done', 'skipped'].includes(latest.state)) continue;
    if (latest && latest.state === 'manual') {
      manualSeen = true;
      core.warning(`eventKey=${queued.ref.eventKey} 位于 manual dead-letter，继续后续 FIFO`);
      continue;
    }
    if (latest && latest.state === 'failed') {
      const skipped = canonicalDeliveryRecord({
        ...latest, state: 'skipped', messageId: null, nextCheckAt: null, reason: 'dead_letter',
      });
      await appendDeliveryAndConfirm({ github, context, core, prNum, record: skipped });
      continue;
    }
    if (processed >= batchSize) break;
    processed += 1;
    if (latest && ['sending', 'uncertain'].includes(latest.state)) {
      const recovered = await recoverAmbiguousDelivery({
        github, context, core, ref: queued.ref, record: latest, env,
      });
      if (!recovered.complete) {
        blockedRetry = recovered.retry;
        manualSeen = recovered.manual === true;
        if (manualSeen) continue;
        break;
      }
      continue;
    }
    const hydrated = await rehydrateEvent({ github, context, core, ref: queued.ref });
    if (!hydrated.ok) {
      blockedRetry = hydrated.retry !== false;
      if (hydrated.manualReason) {
        if (latest && latest.state === 'preparing') {
          const thread = await readThreadState({ github, context, core, prNum });
          if (thread.kind === 'pending' && thread.generation === latest.threadGeneration && thread.claimId === latest.threadClaimId) {
            const released = await releaseThreadGeneration({
              github, context, core, prNum,
              generation: latest.threadGeneration, claimId: latest.threadClaimId,
            });
            if (!released.ok) { blockedRetry = true; break; }
          }
        }
        const record = latest
          ? canonicalDeliveryRecord({ ...latest, state: 'skipped', messageId: null, nextCheckAt: null, reason: 'dead_letter' })
          : terminalNoSendRecord(queued.ref, 'skipped', 'dead_letter');
        await appendDeliveryAndConfirm({ github, context, core, prNum, record });
        blockedRetry = false;
        continue;
      }
      break;
    }
    if (!hydrated.notify) {
      if (latest && latest.state === 'preparing') {
        const thread = await readThreadState({ github, context, core, prNum });
        if (thread.kind === 'pending' && thread.generation === latest.threadGeneration &&
            thread.claimId === latest.threadClaimId) {
          const released = await releaseThreadGeneration({
            github, context, core, prNum,
            generation: latest.threadGeneration, claimId: latest.threadClaimId,
          });
          if (!released.ok) break;
        }
      }
      await appendDeliveryAndConfirm({
        github, context, core, prNum,
        record: latest
          ? canonicalDeliveryRecord({ ...latest, state: 'skipped', messageId: null, nextCheckAt: null, reason: hydrated.reason })
          : terminalNoSendRecord(queued.ref, 'skipped', hydrated.reason),
      });
      continue;
    }
    const delivered = await deliverClassifiedEvent({
      github, context, core, ref: queued.ref, prData: hydrated.prData,
      cls: hydrated.cls, env, latest,
    });
    if (!delivered.complete) {
      blockedRetry = delivered.retry;
      manualSeen = delivered.manual === true;
      if (manualSeen) continue;
      break;
    }
  }

  const finalLedger = await readDeliveryLedger({ github, context, core, prNum });
  if (!finalLedger.ok) return;
  const remaining = queue.events.filter(({ ref }) => {
    const state = finalLedger.latest.get(ref.eventKey);
    return !state || !['done', 'skipped'].includes(state.state);
  });
  if (remaining.length > 0) {
    const remainingActionable = remaining.some(({ ref }) => {
      const state = finalLedger.latest.get(ref.eventKey);
      return !state || state.state !== 'manual';
    });
    const mode = blockedRetry ? 'retry' : remainingActionable ? 'backlog' : 'watchdog';
    setDrainOutputs(core, mode !== 'watchdog', mode);
  }
}

function isManualRootPendingBarrier(record, threadState) {
  return Boolean(record && record.state === 'manual' && record.mode === 'root' &&
    threadState && threadState.kind === 'pending' &&
    threadState.generation === record.threadGeneration && threadState.claimId === record.threadClaimId);
}

function persistedDrainCursorVersion(comments, cursor, context, prNum) {
  const repo = repositoryName(context);
  const records = comments
    .filter((comment) => isTrustedMarkerComment(comment, { legacy: false }))
    .map((comment) => ({ id: Number(comment.id), record: decodeDrainCursor(comment.body) }))
    .filter(({ record }) => record && record.repo === repo && record.pr === prNum &&
      record.nextEventKey === cursor.nextEventKey)
    .sort((a, b) => b.id - a.id);
  return records.length ? records[0].record.version : 2;
}

function compactCheckpointEntries(comments, queue, cursor, latest) {
  const retainedKeys = new Set(queue.events
    .filter(({ ref }) => {
      const record = latest.get(ref.eventKey);
      return !record || !['done', 'skipped'].includes(record.state);
    })
    .map(({ ref }) => ref.eventKey));
  const threadEntries = [];
  const cursorEntries = [];
  const otherEntries = [];
  const tombstoneEntries = new Map();
  for (const comment of comments) {
    const body = String(comment.body || '');
    const entry = {
      id: Number(comment.id), body,
      createdAt: validIsoTime(comment.created_at) ? comment.created_at : '1970-01-01T00:00:00.000Z',
      login: String(comment.user && comment.user.login || ''),
      type: String(comment.user && comment.user.type || 'User'),
      association: String(comment.author_association || 'NONE'),
    };
    if (!Number.isSafeInteger(entry.id) || entry.id <= 0 || body.includes(CHECKPOINT_MARK)) continue;
    const tombstone = decodeDeliveryTombstone(body);
    if (body.includes(DELIVERY_TOMBSTONE_MARK)) {
      if (isGithubActionsBot(comment) && tombstone) {
        const previous = tombstoneEntries.get(tombstone.eventKey);
        if (!previous || entry.id < previous.entry.id) tombstoneEntries.set(tombstone.eventKey, { tombstone, entry });
      }
      continue;
    }
    if (body.includes(STATE_MARK)) {
      if (isGithubActionsBot(comment) && decodeThreadState(body)) threadEntries.push(entry);
      continue;
    }
    if (body.includes(MARK) && !body.includes(EVENT_MARK)) {
      if (isTrustedMarkerComment(comment, { legacy: true }) && legacyMessageId(body)) threadEntries.push(entry);
      continue;
    }
    if (body.includes(DRAIN_CURSOR_MARK)) {
      if (isGithubActionsBot(comment) && decodeDrainCursor(body)) cursorEntries.push(entry);
      continue;
    }
    const event = decodeEventRef(body);
    if (isGithubActionsBot(comment) && event && retainedKeys.has(event.eventKey)) {
      otherEntries.push(entry);
      continue;
    }
    const delivery = decodeDeliveryRecord(body);
    if (isGithubActionsBot(comment) && delivery && retainedKeys.has(delivery.eventKey)) otherEntries.push(entry);
  }
  for (const event of queue.events) {
    const record = latest.get(event.ref.eventKey);
    if (!record || !['done', 'skipped'].includes(record.state) || tombstoneEntries.has(event.ref.eventKey)) continue;
    const source = comments.find((comment) => isGithubActionsBot(comment) &&
      decodeEventRef(comment.body)?.eventKey === event.ref.eventKey);
    const tombstone = canonicalDeliveryTombstone({
      ...event.ref, eventCommentId: Number(event.commentId), state: record.state,
    });
    if (!tombstone) continue;
    tombstoneEntries.set(event.ref.eventKey, {
      tombstone,
      entry: {
        id: Number(event.commentId), body: encodeDeliveryTombstone(tombstone),
        createdAt: source && validIsoTime(source.created_at) ? source.created_at : event.ref.createdAt,
        login: GITHUB_ACTIONS_BOT, type: 'Bot', association: 'NONE',
      },
    });
  }
  const pendingStates = threadEntries.map((entry) => decodeThreadState(entry.body))
    .filter((state) => state && state.state === 'pending')
    .sort((a, b) => b.generation - a.generation);
  const pending = pendingStates[0];
  const pendingOwnerEventKeys = new Set();
  if (pending) {
    for (const comment of comments) {
      const delivery = decodeDeliveryRecord(comment.body);
      if (!isGithubActionsBot(comment) || !delivery || delivery.mode !== 'root' ||
          delivery.threadGeneration !== pending.generation ||
          delivery.threadClaimId !== pending.claimId) continue;
      pendingOwnerEventKeys.add(delivery.eventKey);
      const body = String(comment.body || '');
      const entry = {
        id: Number(comment.id), body,
        createdAt: validIsoTime(comment.created_at) ? comment.created_at : '1970-01-01T00:00:00.000Z',
        login: String(comment.user && comment.user.login || ''),
        type: String(comment.user && comment.user.type || 'User'),
        association: String(comment.author_association || 'NONE'),
      };
      if (!otherEntries.some((candidate) => candidate.id === entry.id)) otherEntries.push(entry);
    }
    for (const comment of comments) {
      if (!isGithubActionsBot(comment)) continue;
      const event = decodeEventRef(comment.body);
      if (!event || !pendingOwnerEventKeys.has(event.eventKey)) continue;
      const entry = {
        id: Number(comment.id), body: String(comment.body || ''),
        createdAt: validIsoTime(comment.created_at) ? comment.created_at : '1970-01-01T00:00:00.000Z',
        login: String(comment.user && comment.user.login || ''),
        type: String(comment.user && comment.user.type || 'User'),
        association: String(comment.author_association || 'NONE'),
      };
      if (!otherEntries.some((candidate) => candidate.id === entry.id)) otherEntries.push(entry);
    }
  }
  const terminalEntries = [...tombstoneEntries.entries()]
    .filter(([eventKey]) => !pendingOwnerEventKeys.has(eventKey))
    .map(([, value]) => value.entry);
  const entries = [...threadEntries.slice(-20), ...terminalEntries, ...otherEntries, ...cursorEntries.slice(-1)]
    .sort((a, b) => a.id - b.id);
  if (entries.length > CHECKPOINT_MAX_ENTRIES) return null;
  return entries;
}

async function persistCheckpoint({ github, context, core, prNum, comments, queue, cursor, latest = null }) {
  let effectiveLatest = latest;
  if (!(effectiveLatest instanceof Map)) {
    const ledger = await readDeliveryLedger({ github, context, core, prNum, comments });
    if (!ledger.ok) return false;
    effectiveLatest = ledger.latest;
  }
  const entries = compactCheckpointEntries(comments, queue, cursor, effectiveLatest);
  if (!entries) {
    core.warning('compact checkpoint 超过 state entry cap；保留旧 checkpoint 并立即续 backlog');
    return null;
  }
  const metadata = comments._checkpoint || {};
  const previous = metadata.checkpoint;
  const highCommentId = comments.reduce((max, comment) =>
    String(comment.body || '').includes(CHECKPOINT_MARK) ? max : Math.max(max, Number(comment.id) || 0),
  previous ? previous.highCommentId : Number(metadata.bootstrapHighCommentId) || 0);
  let highCreatedAt = previous ? previous.highCreatedAt :
    (validIsoTime(metadata.bootstrapHighCreatedAt) ? metadata.bootstrapHighCreatedAt : '1970-01-01T00:00:00.000Z');
  for (const comment of comments) {
    if (Number(comment.id) <= (previous ? previous.highCommentId : 0)) continue;
    if (validIsoTime(comment.created_at) && Date.parse(comment.created_at) > Date.parse(highCreatedAt)) {
      highCreatedAt = comment.created_at;
    }
  }
  const checkpoint = canonicalCheckpoint({
    version: 1, revision: previous ? previous.revision + 1 : 1,
    repo: repositoryName(context), pr: prNum, highCommentId, highCreatedAt, entries,
    parentHash: previous ? canonicalHash(previous) : null,
  });
  let body;
  try { body = encodeCheckpoint(checkpoint); } catch (error) {
    core.warning(`compact checkpoint 超过 comment size cap；保留旧 checkpoint 并立即续 backlog: ${errorMessage(error)}`);
    return null;
  }
  // Checkpoints are append-only: created position is always at the comment tail and each
  // revision cryptographically names its parent canonical snapshot.
  if (true) {
    try {
      const created = await github.rest.issues.createComment({
        owner: context.repo.owner, repo: context.repo.repo, issue_number: prNum, body,
        request: { timeout: GITHUB_REQUEST_TIMEOUT_MS },
      });
      metadata.commentId = Number(created.data.id);
      metadata.checkpoint = checkpoint;
      await minimizeLedgerCommentBestEffort({
        github, core, comment: created.data, label: '折叠 compact checkpoint 评论',
      });
      return true;
    } catch (error) {
      core.warning(`checkpoint POST 响应不确定，执行有界 prefix 核对: ${errorMessage(error)}`);
      const targetHash = canonicalHash(checkpoint);
      const tail = await readCheckpointTail({ github, context, core, prNum });
      if (!tail) return false;
      const found = tail.map((comment) => {
        const candidate = decodeCheckpoint(comment.body);
        return candidate && candidate.repo === repositoryName(context) && candidate.pr === prNum
          ? { hash: canonicalHash(candidate), commentId: Number(comment.id), nodeId: comment.node_id, checkpoint: candidate }
          : null;
      }).filter((candidate) => candidate && candidate.checkpoint.revision === checkpoint.revision);
      const expected = found.filter(({ hash }) => hash === targetHash);
      const revisionHashes = new Set(found.map(({ hash }) => hash));
      if (expected.length > 0 && revisionHashes.size === 1 &&
          expected.every(({ checkpoint: candidate }) => candidate.parentHash === checkpoint.parentHash)) {
        metadata.commentId = expected[0].commentId;
        metadata.checkpoint = expected[0].checkpoint;
        await minimizeLedgerCommentBestEffort({
          github, core, comment: { nodeId: expected[0].nodeId }, label: '折叠已恢复的 compact checkpoint 评论',
        });
        return true;
      }
      core.setFailed('checkpoint POST 未确认或发现不一致 lineage，保留旧 high-watermark');
      return false;
    }
  }
  const updated = await withGithubRetry({
    core, label: '更新 compact checkpoint', setFailedOnExhausted: false,
    operation: () => github.rest.issues.updateComment({
      owner: context.repo.owner, repo: context.repo.repo, comment_id: metadata.commentId, body,
      request: { timeout: GITHUB_REQUEST_TIMEOUT_MS },
    }),
  });
  if (updated.ok) {
    metadata.checkpoint = checkpoint;
    return true;
  }
  const reread = await withGithubRetry({
    core, label: '核对不确定 checkpoint update', setFailedOnExhausted: false,
    operation: () => github.rest.issues.getComment({
      owner: context.repo.owner, repo: context.repo.repo, comment_id: metadata.commentId,
      request: { timeout: GITHUB_REQUEST_TIMEOUT_MS },
    }),
  });
  const confirmed = reread.ok && decodeCheckpoint(reread.value.data.body);
  if (confirmed && confirmed.revision === checkpoint.revision &&
      canonicalHash(confirmed) === canonicalHash(checkpoint)) {
    metadata.checkpoint = checkpoint;
    return true;
  }
  core.setFailed('checkpoint update 未确认，保留旧 high-watermark 重放');
  return false;
}

async function drainQueuedCodexEventsSnapshotWithinBudget({
  github, context, core, batchSize = DRAIN_BATCH_SIZE, bootstrapPageLimit = Infinity,
}) {
  setDrainOutputs(core, false, 'none');
  const prNum = safePositiveInteger(process.env.RESOLVED_PR_NUMBER) || await resolvePrNumber(github, context, core);
  if (!prNum) return;
  let comments = await readThreadMarkerComments({ github, context, core, prNum, bootstrapPageLimit });
  if (!comments) {
    setDrainOutputs(core, true, 'retry');
    return;
  }
  if (comments._checkpoint && comments._checkpoint.bootstrapIncomplete) {
    setDrainOutputs(core, true, 'backlog');
    return;
  }
  let queue = await readEventQueue({ github, context, core, prNum, comments });
  let ledger = await readDeliveryLedger({ github, context, core, prNum, comments });
  let threadState = await readThreadState({ github, context, core, prNum, comments });
  let cursor = readDrainCursor({ comments, context, core, prNum });
  if (!queue.ok || !ledger.ok || !threadState.ok || !cursor.ok) return;
  const checkpointMetadata = comments._checkpoint || {};
  const priorHighCommentId = checkpointMetadata.checkpoint ? checkpointMetadata.checkpoint.highCommentId : 0;
  const snapshotHighCommentId = comments.reduce((max, comment) =>
    String(comment.body || '').includes(CHECKPOINT_MARK) ? max : Math.max(max, Number(comment.id) || 0), 0);
  let freshOverflow = false;
  if (!checkpointMetadata.checkpoint || snapshotHighCommentId > priorHighCommentId) {
    const ingested = await persistCheckpoint({
      github, context, core, prNum, comments, queue, cursor, latest: ledger.latest,
    });
    if (ingested === null) {
      if (!checkpointMetadata.checkpoint) {
        setDrainOutputs(core, true, 'backlog');
        return;
      }
      comments = checkpointMetadata.checkpoint.entries.map(virtualCheckpointComment);
      Object.defineProperty(comments, '_checkpoint', {
        value: { checkpoint: checkpointMetadata.checkpoint, commentId: checkpointMetadata.commentId, truncated: false },
        enumerable: false,
      });
      queue = await readEventQueue({ github, context, core, prNum, comments });
      ledger = await readDeliveryLedger({ github, context, core, prNum, comments });
      threadState = await readThreadState({ github, context, core, prNum, comments });
      cursor = readDrainCursor({ comments, context, core, prNum });
      if (!queue.ok || !ledger.ok || !threadState.ok || !cursor.ok) return;
      freshOverflow = true;
    } else if (!ingested) {
      setDrainOutputs(core, true, 'retry');
      return;
    }
    if (checkpointMetadata.truncated) {
      setDrainOutputs(core, true, 'backlog');
      return;
    }
  }
  if (queue.events.length === 0) {
    const checkpointed = await persistCheckpoint({
      github, context, core, prNum, comments, queue, cursor, latest: ledger.latest,
    });
    if (checkpointed === null) setDrainOutputs(core, true, 'backlog');
    else if (!checkpointed) setDrainOutputs(core, true, 'retry');
    else if (comments._checkpoint && comments._checkpoint.truncated) setDrainOutputs(core, true, 'backlog');
    return;
  }
  const env = readFeishuEnv(core, true);
  if (!env) {
    setDrainOutputs(core, true, 'retry');
    return;
  }

  const session = { latest: new Map(ledger.latest), threadState };
  const events = queue.events;
  const cursorIndex = cursor.nextEventKey
    ? events.findIndex(({ ref }) => ref.eventKey === cursor.nextEventKey)
    : -1;
  // v1 field name is retained for marker compatibility, but its value is now a monotonic
  // high-watermark: the last event absorbed by the consumer, never the next circular slot.
  const cursorVersion = cursor.version || persistedDrainCursorVersion(comments, cursor, context, prNum);
  const startIndex = cursorVersion >= 3 && cursor.lastCommentId
    ? events.findIndex((event) => Number(event.commentId) > cursor.lastCommentId)
    : cursorIndex >= 0 ? cursorIndex + (cursorVersion >= 2 ? 1 : 0) : 0;
  const normalizedStartIndex = startIndex < 0 ? events.length : startIndex;
  const replayActionableStates = new Set([
    'retrying', 'preparing', 'sending', 'uncertain', 'not_sent', 'failed',
  ]);
  const replayIndexes = events
    .map((event, index) => ({ event, index }))
    .filter(({ event }) => Number(event.commentId) <= cursor.lastCommentId &&
      replayActionableStates.has(session.latest.get(event.ref.eventKey)?.state))
    .map(({ index }) => index);
  if (replayIndexes.length === 0 && normalizedStartIndex >= events.length) {
    const checkpointed = await persistCheckpoint({
      github, context, core, prNum, comments, queue, cursor, latest: ledger.latest,
    });
    if (checkpointed === null) setDrainOutputs(core, true, 'backlog');
    else if (!checkpointed) setDrainOutputs(core, true, 'retry');
    else if (comments._checkpoint && comments._checkpoint.truncated) setDrainOutputs(core, true, 'backlog');
    return;
  }
  const forwardIndexes = Array.from(
    { length: Math.max(0, events.length - normalizedStartIndex) },
    (_value, offset) => normalizedStartIndex + offset,
  );
  const scanIndexes = [...replayIndexes, ...forwardIndexes].slice(0, DRAIN_SCAN_LIMIT);
  let scanned = 0;
  let processed = 0;
  let nextEventKey = cursor.nextEventKey;
  let lastCommentId = cursor.lastCommentId;
  let blockedMode = null;

  const advanceAfter = (index) => {
    const commentId = Number(events[index].commentId);
    if (commentId > lastCommentId) {
      nextEventKey = events[index].ref.eventKey;
      lastCommentId = commentId;
    }
  };
  const stopAt = (ref, mode) => {
    blockedMode = mode;
  };
  const skipDeadLetter = async (ref, latest, reason = 'dead_letter') => {
    if (latest && latest.state === 'preparing' && isManualRootPendingBarrier({ ...latest, state: 'manual' }, session.threadState)) {
      const released = await releaseThreadGeneration({
        github, context, core, prNum,
        generation: latest.threadGeneration, claimId: latest.threadClaimId, session,
      });
      if (!released.ok) return false;
    }
    const skipped = latest
      ? canonicalDeliveryRecord({ ...latest, state: 'skipped', messageId: null, nextCheckAt: null, reason })
      : terminalNoSendRecord(ref, 'skipped', reason);
    return (await appendDeliveryAndConfirm({
      github, context, core, prNum, record: skipped, session,
    })).ok;
  };

  for (const index of scanIndexes) {
    const ref = events[index].ref;
    scanned += 1;
    let latest = session.latest.get(ref.eventKey);
    if (latest && ['done', 'skipped'].includes(latest.state)) {
      advanceAfter(index);
      continue;
    }
    if (latest && latest.state === 'manual') {
      if (isManualRootPendingBarrier(latest, session.threadState)) {
        stopAt(ref, 'watchdog');
        break;
      }
      core.warning(`eventKey=${ref.eventKey} manual dead-letter 可越过，继续 FIFO`);
      advanceAfter(index);
      continue;
    }
    if (latest && latest.state === 'failed') {
      if (!await skipDeadLetter(ref, latest)) {
        stopAt(ref, 'retry');
        break;
      }
      advanceAfter(index);
      continue;
    }
    if (processed >= batchSize) {
      stopAt(ref, 'backlog');
      break;
    }
    processed += 1;

    if (latest && ['sending', 'uncertain'].includes(latest.state)) {
      const recovered = await recoverAmbiguousDelivery({
        github, context, core, ref, record: latest, env, session,
      });
      if (recovered.complete) {
        advanceAfter(index);
        continue;
      }
      latest = session.latest.get(ref.eventKey);
      if (latest && latest.state === 'manual' && !isManualRootPendingBarrier(latest, session.threadState)) {
        advanceAfter(index);
        continue;
      }
      stopAt(ref, recovered.retry ? 'retry' : 'watchdog');
      break;
    }

    const hydrated = await rehydrateEvent({ github, context, core, ref });
    if (!hydrated.ok) {
      if (hydrated.manualReason) {
        if (!await skipDeadLetter(ref, latest)) {
          stopAt(ref, 'retry');
          break;
        }
        advanceAfter(index);
        continue;
      }
      stopAt(ref, hydrated.retry === false ? 'watchdog' : 'retry');
      break;
    }
    if (!hydrated.notify) {
      if (!await skipDeadLetter(ref, latest, hydrated.reason)) {
        stopAt(ref, 'retry');
        break;
      }
      advanceAfter(index);
      continue;
    }

    const delivered = await deliverClassifiedEvent({
      github, context, core, ref, prData: hydrated.prData, cls: hydrated.cls,
      env, latest, session,
    });
    if (delivered.complete) {
      advanceAfter(index);
      continue;
    }
    latest = session.latest.get(ref.eventKey);
    if (latest && latest.state === 'failed') {
      if (!await skipDeadLetter(ref, latest)) {
        stopAt(ref, 'retry');
        break;
      }
      advanceAfter(index);
      continue;
    }
    if (latest && latest.state === 'manual' && !isManualRootPendingBarrier(latest, session.threadState)) {
      advanceAfter(index);
      continue;
    }
    stopAt(ref, delivered.retry ? 'retry' : 'watchdog');
    break;
  }

  if (scanned > 0 && (nextEventKey !== cursor.nextEventKey || lastCommentId !== cursor.lastCommentId) &&
      !await appendDrainCursor({ github, context, core, prNum, nextEventKey, lastCommentId })) {
    blockedMode = blockedMode || 'retry';
  }
  const checkpointed = await persistCheckpoint({
    github, context, core, prNum, comments, queue,
    cursor: { ...cursor, version: 3, nextEventKey, lastCommentId },
    latest: session.latest,
  });
  if (checkpointed === null) blockedMode = blockedMode || 'backlog';
  else if (!checkpointed) blockedMode = blockedMode || 'retry';
  if (comments._checkpoint && comments._checkpoint.truncated) blockedMode = blockedMode || 'backlog';
  const remaining = events.filter(({ ref }) => {
    const state = session.latest.get(ref.eventKey);
    return !state || !['done', 'skipped'].includes(state.state);
  });
  if (remaining.length === 0) return;
  const actionable = remaining.some(({ ref }) => {
    const state = session.latest.get(ref.eventKey);
    return !state || state.state !== 'manual';
  });
  const mode = blockedMode || (freshOverflow ? 'backlog' : actionable ? 'backlog' : 'watchdog');
  setDrainOutputs(core, mode !== 'watchdog', mode);
}

async function scheduleQueuedCodexDrain(args) {
  return withGithubAttemptBudget(async () => {
    const { github, context, core } = args;
    let ref = args.ref || null;
    if (ref) {
      const refs = parseCompactEventRefs([ref], core);
      if (!refs) return false;
      [ref] = refs;
    }
    const prNum = safePositiveInteger(args.prNum || ref && ref.pr || process.env.RESOLVED_PR_NUMBER) || await resolvePrNumber(github, context, core);
    if (!prNum) return false;
    return scheduleDrain({ github, context, core, prNum, ref });
  });
}

function parseCodexCaptureRunName(value, core = noopCore()) {
  const raw = String(value || '');
  if (raw.length > 100) { core.setFailed('capture run name 超长'); return null; }
  const match = /^codex-feishu-capture:(review|check):(0|[1-9][0-9]*):([1-9][0-9]*)$/.exec(raw);
  if (!match) { core.setFailed('capture run name 格式非法'); return null; }
  const pr = safePositiveInteger(match[2]);
  const eventId = safePositiveInteger(match[3]);
  if ((!pr && match[2] !== '0') || !eventId || (match[1] === 'review' && !pr)) {
    core.setFailed('capture run name 数字超出 safe integer');
    return null;
  }
  return { type: match[1], pr: pr || 0, eventId };
}

async function resolveCodexWorkflowRunRefs({ github, context, core = noopCore() }) {
  if (context.eventName !== 'workflow_run') { core.setFailed('仅 workflow_run 可消费 capture wake-up'); return null; }
  const hint = context.payload.workflow_run || {};
  const runId = safePositiveInteger(hint.id);
  if (!runId) { core.setFailed('workflow_run id 非 canonical safe integer'); return null; }
  const runResult = await withGithubRetry({
    core, label: `重取 source workflow run ${runId}`,
    operation: () => github.rest.actions.getWorkflowRun({
      owner: context.repo.owner, repo: context.repo.repo, run_id: runId,
      request: { timeout: GITHUB_REQUEST_TIMEOUT_MS },
    }),
  });
  if (!runResult.ok) return null;
  const run = runResult.value.data || {};
  const workflowId = safePositiveInteger(run.workflow_id);
  if (!workflowId || String(run.id) !== String(runId) || run.status !== 'completed' || run.conclusion !== 'success' ||
      !['check_run', 'pull_request_review'].includes(run.event) ||
      String(run.repository && run.repository.full_name || '') !== repositoryName(context) ||
      (hint.workflow_id && String(hint.workflow_id) !== String(workflowId))) {
    core.setFailed('source workflow run repo/id/event/status/conclusion 绑定非法');
    return null;
  }
  const workflowResult = await withGithubRetry({
    core, label: `验证 source workflow id=${workflowId}`,
    operation: () => github.rest.actions.getWorkflow({
      owner: context.repo.owner, repo: context.repo.repo, workflow_id: workflowId,
      request: { timeout: GITHUB_REQUEST_TIMEOUT_MS },
    }),
  });
  if (!workflowResult.ok) return null;
  const workflow = workflowResult.value.data || {};
  if (String(workflow.id) !== String(workflowId) || workflow.name !== CODEX_CAPTURE_WORKFLOW_NAME ||
      workflow.path !== CODEX_CAPTURE_WORKFLOW_PATH) {
    core.setFailed('source workflow id/name/path 不是固定 capture workflow');
    return null;
  }
  const displayTitle = String(run.display_title || '');
  if (!/^codex-feishu-capture:(review|check):(0|[1-9][0-9]*):([1-9][0-9]*)$/.test(displayTitle)) {
    core.warning('legacy capture workflow_run display_title 非 canonical，安全 no-op');
    return [];
  }
  const capture = parseCodexCaptureRunName(displayTitle, core);
  if (!capture) return null;
  if ((capture.type === 'review') !== (run.event === 'pull_request_review')) {
    core.setFailed('canonical capture title type 与 fetched workflow_run event 不一致');
    return null;
  }
  const jobsResult = await withGithubRetry({
    core, label: `验证 source workflow run ${runId} capture job`,
    operation: () => github.rest.actions.listJobsForWorkflowRun({
      owner: context.repo.owner, repo: context.repo.repo, run_id: runId,
      filter: 'latest', page: 1, per_page: 2,
      request: { timeout: GITHUB_REQUEST_TIMEOUT_MS },
    }),
  });
  if (!jobsResult.ok) return null;
  const jobs = jobsResult.value.data && jobsResult.value.data.jobs || [];
  if (jobs.length !== 1 || !safePositiveInteger(jobs[0].id) || jobs[0].name !== 'capture' ||
      jobs[0].status !== 'completed') {
    core.setFailed('canonical source workflow capture job 数量/id/name/status 非法');
    return null;
  }
  if (jobs[0].conclusion === 'skipped') {
    core.warning('canonical capture job 被官方 source filter 正常跳过，安全 no-op');
    return [];
  }
  if (jobs[0].conclusion !== 'success') {
    core.setFailed(`canonical source workflow capture job conclusion=${String(jobs[0].conclusion)} 非法`);
    return null;
  }
  if (capture.type === 'review') {
    const prData = await resolvePr(github, context, { number: capture.pr }, core);
    if (!prData || prData.state !== 'open' || prData.base !== 'main') return [];
    const fetched = await withGithubRetry({
      core, label: `按 wake-up immutable ID 重取 review ${capture.eventId}`,
      operation: () => github.rest.pulls.getReview({
        owner: context.repo.owner, repo: context.repo.repo, pull_number: capture.pr, review_id: capture.eventId,
        request: { timeout: GITHUB_REQUEST_TIMEOUT_MS },
      }),
    });
    if (!fetched.ok) return null;
    const review = fetched.value.data;
    if (!review || String(review.id) !== String(capture.eventId) || !isCodexPullRequestReview(review) ||
        !validSha(review.commit_id) || !review.submitted_at) {
      core.setFailed('wake-up review 官方身份/commit/time 绑定非法');
      return null;
    }
    return [canonicalEventRef({
      version: EVENT_VERSION, eventType: 'pull_request_review', eventId: String(review.id),
      repo: repositoryName(context), pr: prData.number, headSha: review.commit_id, createdAt: review.submitted_at,
    })];
  }
  const fetched = await withGithubRetry({
    core, label: `按 wake-up immutable ID 重取 check_run ${capture.eventId}`,
    operation: () => github.rest.checks.get({
      owner: context.repo.owner, repo: context.repo.repo, check_run_id: capture.eventId,
      request: { timeout: GITHUB_REQUEST_TIMEOUT_MS },
    }),
  });
  if (!fetched.ok) return null;
  const checkRun = fetched.value.data;
  if (!checkRun || String(checkRun.id) !== String(capture.eventId) || !isCodexCheckRun(checkRun) ||
      checkRun.status !== 'completed' || !validSha(checkRun.head_sha) || !(checkRun.completed_at || checkRun.updated_at)) {
    core.setFailed('wake-up check_run 官方身份/status/head/time 绑定非法');
    return null;
  }
  return resolveFreshCheckRunRefs({
    github, context, core, checkRun, expectedId: capture.eventId,
  });
}

async function consumeCodexWorkflowRunWakeup(args) {
  return withGithubAttemptBudget(async () => {
    const { github, context, core } = args;
    const refs = await resolveCodexWorkflowRunRefs({ github, context, core });
    if (!refs) return null;
    const confirmed = [];
    for (const ref of refs) {
      const hydrated = await rehydrateEvent({ github, context, core, ref });
      if (!hydrated.ok || hydrated.reason === 'stale_pr') return null;
      const queued = await enqueueEventRef({ github, context, core, ref });
      if (!queued.ok) return null;
      confirmed.push(ref);
    }
    for (const ref of confirmed) {
      if (!await scheduleDrain({ github, context, core, prNum: ref.pr })) return null;
    }
    return confirmed;
  });
}

async function reconcileOfficialSourcesForPr({ github, context, core, prData, knownEventKeys = new Set() }) {
  const reviewItems = [];
  const checkItems = [];
  for (let page = 1; page <= SOURCE_RECONCILE_MAX_PAGES; page += 1) {
    if (!hasGithubRequestBudget()) return null;
    const result = await withGithubRetry({
      core, label: `枚举 PR #${prData.number} 当前 head reviews page=${page}`,
      operation: () => github.rest.pulls.listReviews({
        owner: context.repo.owner, repo: context.repo.repo, pull_number: prData.number,
        page, per_page: SOURCE_RECONCILE_PAGE_SIZE,
        request: { timeout: GITHUB_REQUEST_TIMEOUT_MS },
      }),
    });
    if (!result.ok) return null;
    const pageItems = result.value.data || [];
    reviewItems.push(...pageItems);
    if (pageItems.length < SOURCE_RECONCILE_PAGE_SIZE) break;
    if (page === SOURCE_RECONCILE_MAX_PAGES) {
      core.warning(`PR #${prData.number} reviews 达分页 cap=${SOURCE_RECONCILE_MAX_PAGES}`);
    }
  }
  for (let page = 1; page <= SOURCE_RECONCILE_MAX_PAGES; page += 1) {
    if (!hasGithubRequestBudget()) return null;
    const result = await withGithubRetry({
      core, label: `枚举 PR #${prData.number} 当前 head checks page=${page}`,
      operation: () => github.rest.checks.listForRef({
        owner: context.repo.owner, repo: context.repo.repo, ref: prData.headSha, filter: 'all',
        page, per_page: SOURCE_RECONCILE_PAGE_SIZE,
        request: { timeout: GITHUB_REQUEST_TIMEOUT_MS },
      }),
    });
    if (!result.ok) return null;
    const pageItems = result.value.data && result.value.data.check_runs || [];
    checkItems.push(...pageItems);
    if (pageItems.length < SOURCE_RECONCILE_PAGE_SIZE) break;
    if (page === SOURCE_RECONCILE_MAX_PAGES) {
      core.warning(`PR #${prData.number} checks 达分页 cap=${SOURCE_RECONCILE_MAX_PAGES}`);
    }
  }
  const reviews = reviewItems.filter((review) =>
    isCodexPullRequestReview(review) && review.commit_id === prData.headSha && review.submitted_at);
  const checks = checkItems.filter((checkRun) =>
    isCodexCheckRun(checkRun) && checkRun.status === 'completed' && checkRun.head_sha === prData.headSha &&
    (checkRun.completed_at || checkRun.updated_at));
  const all = [
    ...reviews.map((review) => canonicalEventRef({
      version: EVENT_VERSION, eventType: 'pull_request_review', eventId: String(review.id),
      repo: repositoryName(context), pr: prData.number, headSha: prData.headSha, createdAt: review.submitted_at,
    })),
    ...checks.map((checkRun) => canonicalEventRef({
      version: EVENT_VERSION, eventType: 'check_run', eventId: String(checkRun.id),
      repo: repositoryName(context), pr: prData.number, headSha: prData.headSha,
      createdAt: checkRun.completed_at || checkRun.updated_at,
    })),
  ].sort((a, b) => a.createdAt.localeCompare(b.createdAt) || a.eventKey.localeCompare(b.eventKey));
  const unknown = all.filter((ref) => !knownEventKeys.has(ref.eventKey));
  if (unknown.length > SOURCE_RECONCILE_MAX_EVENTS) {
    core.info(`PR #${prData.number} source reconciliation unknown=${unknown.length}，本轮稳定推进 ${SOURCE_RECONCILE_MAX_EVENTS}`);
  }
  return unknown.slice(0, SOURCE_RECONCILE_MAX_EVENTS);
}

function watchdogWindow(pulls, repo, nowMs = Date.now()) {
  const slot = Math.floor(nowMs / WATCHDOG_SLOT_MS);
  const shard = slot % WATCHDOG_SHARDS;
  const cycle = Math.floor(slot / WATCHDOG_SHARDS);
  const inShard = pulls
    .filter((pr) => {
      const digest = crypto.createHash('sha256').update(`${repo}#${pr.number}`).digest();
      return digest.readUInt32BE(0) % WATCHDOG_SHARDS === shard;
    })
    .sort((a, b) => Number(a.number) - Number(b.number));
  if (inShard.length === 0) return { slot, shard, pulls: [] };
  // For the same shard, cycle increments by exactly one. Advancing one position is coprime
  // with every non-empty set length, unlike a cap-sized step (40 starves lengths 20/32/40).
  let step = Math.min(WATCHDOG_MAX_PRS_PER_RUN, Math.max(1, inShard.length - 1));
  const gcd = (left, right) => right === 0 ? left : gcd(right, left % right);
  while (step > 1 && gcd(step, inShard.length) !== 1) step -= 1;
  const offset = (cycle * step) % inShard.length;
  const rotated = inShard.slice(offset).concat(inShard.slice(0, offset));
  return { slot, shard, pulls: rotated.slice(0, WATCHDOG_MAX_PRS_PER_RUN) };
}

async function sweepQueuedCodexReviews(args) {
  return withGithubAttemptBudget(async () => {
    const { github, context, core } = args;
    const nowMs = Number.isFinite(args.nowMs) ? args.nowMs : Date.now();
    const pulls = [];
    let page = 1;
    for (;;) {
      if (!hasGithubRequestBudget()) {
        core.warning('watchdog 在读取完全部 OPEN PR 前到达 attempt deadline，留待下一轮');
        return;
      }
      const result = await withGithubRetry({
        core,
        label: `列出 watchdog PR page=${page}`,
        operation: () => github.rest.pulls.list({
          owner: context.repo.owner, repo: context.repo.repo, state: 'open', base: 'main',
          sort: 'created', direction: 'asc', page, per_page: WATCHDOG_PR_PAGE_SIZE,
          request: { timeout: GITHUB_REQUEST_TIMEOUT_MS },
        }),
      });
      if (!result.ok) return;
      pulls.push(...result.value.data);
      if (result.value.data.length < WATCHDOG_PR_PAGE_SIZE) break;
      page += 1;
    }
    const window = watchdogWindow(pulls, repositoryName(context), nowMs);
    let dispatches = 0;
    for (const pr of window.pulls) {
      if (dispatches >= WATCHDOG_MAX_DISPATCHES) break;
      const prNum = Number(pr.number);
      if (!Number.isSafeInteger(prNum) || prNum <= 0) continue;
      const prData = await resolvePr(github, context, { number: prNum }, core);
      if (!prData || prData.state !== 'open' || prData.base !== 'main') continue;
      const comments = await readThreadMarkerComments({
        github, context, core, prNum, bootstrapPageLimit: args.bootstrapPageLimit,
      });
      if (!comments) continue;
      if (comments._checkpoint && comments._checkpoint.bootstrapIncomplete) {
        if (await scheduleDrain({ github, context, core, prNum })) {
          dispatches += 1;
        } else {
          core.warning(`watchdog PR #${prNum} bootstrap progress 已持久化但 continuation dispatch 失败，留待下一轮`);
        }
        continue;
      }
      const queue = await readEventQueue({ github, context, core, prNum, comments });
      if (!queue.ok) continue;
      const ledger = await readDeliveryLedger({ github, context, core, prNum, comments });
      if (!ledger.ok) continue;
      let barrier = false;
      let actionable = false;
      if (queue.events.length > 0) {
        const thread = await readThreadState({ github, context, core, prNum, comments });
        if (!thread.ok) continue;
        barrier = queue.events.some(({ ref }) =>
          isManualRootPendingBarrier(ledger.latest.get(ref.eventKey), thread));
        actionable = queue.events.some(({ ref }) => {
          const state = ledger.latest.get(ref.eventKey);
          if (!state || ['retrying', 'failed'].includes(state.state)) return true;
          if (['done', 'skipped', 'manual'].includes(state.state)) return false;
          return !state.nextCheckAt || Date.parse(state.nextCheckAt) <= nowMs;
        });
      }
      let existingDispatched = false;
      if (!barrier && actionable) {
        existingDispatched = await scheduleDrain({ github, context, core, prNum });
        if (!existingDispatched) {
          core.warning(`watchdog PR #${prNum} 已有 durable queue dispatch 失败，跳过 source reconciliation`);
          continue;
        }
        dispatches += 1;
      }
      const sources = await reconcileOfficialSourcesForPr({
        github, context, core, prData,
        knownEventKeys: new Set([
          ...queue.events.map(({ ref }) => ref.eventKey),
          ...[...ledger.latest.entries()]
            .filter(([, record]) => ['done', 'skipped'].includes(record.state))
            .map(([eventKey]) => eventKey),
        ]),
      });
      let sourcesDurable = true;
      for (const ref of sources || []) {
        const queued = await enqueueEventRef({ github, context, core, ref });
        if (!queued.ok) { sourcesDurable = false; break; }
      }
      if (barrier) continue;
      const newDurableSources = sourcesDurable && sources && sources.length > 0;
      if (!existingDispatched && newDurableSources && dispatches < WATCHDOG_MAX_DISPATCHES &&
          await scheduleDrain({ github, context, core, prNum })) dispatches += 1;
    }
    core.info(`watchdog slot=${window.slot} shard=${window.shard} open_prs=${pulls.length} prs=${window.pulls.length} dispatches=${dispatches}`);
  });
}

async function notifyFromCheckRunWithinBudget({ github, context, core }) {
  const refs = await resolveOfficialCodexEventRefs(github, context, core);
  if (!refs) return;
  let env = null;
  for (const ref of refs) {
    const hydrated = await rehydrateEvent({ github, context, core, ref });
    if (!hydrated.ok) return;
    if (!hydrated.notify) {
      core.info(`fresh check_run #${ref.eventId} 不满足通知条件`);
      continue;
    }
    env = env || readFeishuEnv(core);
    if (!env) return;
    const ledger = await readDeliveryLedger({ github, context, core, prNum: ref.pr });
    if (!ledger.ok) return;
    await deliverClassifiedEvent({
      github, context, core, ref, prData: hydrated.prData, cls: hydrated.cls,
      env, latest: ledger.latest.get(ref.eventKey),
    });
  }
}

async function notifyFromPullRequestReviewWithinBudget({ github, context, core }) {
  const { classifyPullRequestReview, shouldNotify } = require('./report');
  const review = context.payload.review;
  if (!isCodexPullRequestReview(review)) { core.info('非 Codex review，跳过'); return; }
  const prData = await resolvedPrForNotification(github, context, core);
  if (!prData || prData.base !== 'main' || prData.state !== 'open') return;
  const comments = await listCommentsForReview(github, context, core, prData.number, review.id);
  if (!comments) return;
  const cls = classifyPullRequestReview(review, comments, process.env);
  if (!shouldNotify(cls)) return;
  const ref = eventRefFromContext(context, prData, core);
  if (!ref) return;
  const env = readFeishuEnv(core); if (!env) return;
  const ledger = await readDeliveryLedger({ github, context, core, prNum: ref.pr });
  if (!ledger.ok) return;
  await deliverClassifiedEvent({ github, context, core, ref, prData, cls, env, latest: ledger.latest.get(ref.eventKey) });
}

async function postSynthetic({ github, context, core, cls, prData, dedupeKey }) {
  const env = readFeishuEnv(core); if (!env) return;
  const createdAt = new Date().toISOString();
  const ref = canonicalEventRef({
    version: EVENT_VERSION,
    eventType: 'check_run',
    eventId: String(Math.max(1, parseInt(crypto.createHash('sha256').update(dedupeKey).digest('hex').slice(0, 12), 16))),
    repo: repositoryName(context),
    pr: prData.number,
    headSha: validSha(context.sha) ? context.sha : crypto.createHash('sha256').update(String(context.sha || dedupeKey)).digest('hex'),
    createdAt,
  });
  const ledger = await readDeliveryLedger({ github, context, core, prNum: ref.pr });
  if (!ledger.ok) return;
  await deliverClassifiedEvent({ github, context, core, ref, prData, cls, env, latest: ledger.latest.get(ref.eventKey) });
}

async function enqueueOfficialCodexEvent(args) {
  return withGithubAttemptBudget(() => enqueueOfficialCodexEventWithinBudget(args));
}

async function ensureDispatchedEventEnqueued({ github, context, core }) {
  if (context.eventName !== 'repository_dispatch' ||
      !context.payload.client_payload || !context.payload.client_payload.event_ref) return true;
  const ref = dispatchEventRef(context, core);
  if (!ref) return false;
  const hydrated = await rehydrateEvent({ github, context, core, ref });
  if (!hydrated.ok) {
    core.setFailed('dispatch immutable ref strict rehydrate 失败，拒绝补入队');
    return false;
  }
  const ensured = await enqueueEventRef({ github, context, core, ref });
  if (!ensured.ok) core.setFailed('dispatch immutable ref 未能 idempotent ensure enqueue');
  return ensured.ok;
}

async function drainQueuedCodexEvents(args) {
  return withGithubAttemptBudget(async () => {
    if (!await ensureDispatchedEventEnqueued(args)) return;
    return drainQueuedCodexEventsSnapshotWithinBudget(args);
  });
}

async function notifyFromCheckRun(args) {
  return withGithubAttemptBudget(() => notifyFromCheckRunWithinBudget(args));
}

async function notifyFromPullRequestReview(args) {
  return withGithubAttemptBudget(() => notifyFromPullRequestReviewWithinBudget(args));
}

async function notifyFromOfficialCodexEvent(args) {
  return withGithubAttemptBudget(async () => {
    await enqueueOfficialCodexEventWithinBudget(args);
    await drainQueuedCodexEventsSnapshotWithinBudget(args);
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
    await postSynthetic({ github, context, core, cls, prData, dedupeKey: `action:${context.sha}:${result && result.run_id || ''}` });
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
    const runId = failure && failure.runId || process.env.GITHUB_RUN_ID || '';
    await postSynthetic({ github, context, core, cls, prData, dedupeKey: `action-failure:${context.sha}:${runId}:${cls.reason}` });
  });
}

module.exports = {
  notifyFromOfficialCodexEvent,
  notifyFromCheckRun,
  notifyFromPullRequestReview,
  notifyFromActionResult,
  notifyFromActionFailure,
  enqueueOfficialCodexEvent,
  drainQueuedCodexEvents,
  enqueueEventRef,
  readEventQueue,
  readDeliveryLedger,
  appendDeliveryAndConfirm,
  findDeliveryInFeishuHistory,
  deliveryToken,
  contentWithDeliveryToken,
  canonicalEventRef,
  encodeEventRef,
  decodeEventRef,
  encodeDeliveryRecord,
  decodeDeliveryRecord,
  encodeDeliveryRepairRecord,
  decodeDeliveryRepairRecord,
  validFeishuMessageId,
  shouldRecreateRootOnReplyFailure,
  isCodexCheckRun,
  isCodexPullRequestReview,
  GITHUB_REQUEST_TIMEOUT_MS,
  COMMENT_MINIMIZE_TIMEOUT_MS,
  GITHUB_ATTEMPT_BUDGET_MS,
  BOOTSTRAP_PROGRESS_RESERVE_MS,
  MAX_GITHUB_RETRY_DELAY_MS,
  FEISHU_RATE_LIMIT_MAX_ATTEMPTS,
  FEISHU_RATE_LIMIT_MAX_INLINE_DELAY_MS,
  FEISHU_RATE_LIMIT_MAX_RETRY_AFTER_MS,
  DRAIN_BATCH_SIZE,
  HISTORY_MAX_PAGES,
  HISTORY_PAGE_SIZE,
  HISTORY_MAX_ATTEMPTS,
  HISTORY_MAX_AGE_MS,
  DEFINITE_SEND_MAX_ATTEMPTS,
  WATCHDOG_PR_PAGE_SIZE,
  WATCHDOG_MAX_PRS_PER_RUN,
  WATCHDOG_MAX_DISPATCHES,
  WATCHDOG_SHARDS,
  WATCHDOG_SLOT_MS,
  CODEX_REVIEW_USER_ID,
  CODEX_REVIEW_LOGIN,
  CODEX_CHECK_APP_SLUG,
  parseRetryAfterMs,
  withGithubAttemptBudget,
  withGithubRetry,
  githubGraphqlWithTimeout,
  minimizeLedgerCommentBestEffort,
  readThreadMarkerComments,
  persistThreadMarker,
  MARK,
  STATE_MARK,
  REPAIR_MARK,
  EVENT_MARK,
  DELIVERY_MARK,
  DELIVERY_REPAIR_MARK,
  encodeThreadState,
  decodeThreadState,
  encodeRepairRecord,
  decodeRepairRecord,
  isTrustedMarkerComment,
  readThreadState,
  resolvePrNumber,
  resolveOfficialCodexEventRefs,
  resolveCodexWorkflowRunRefs,
  consumeCodexWorkflowRunWakeup,
  parseCodexCaptureRunName,
  parseCompactEventRefs,
  MAX_RESOLVED_EVENT_REFS,
  CODEX_CAPTURE_WORKFLOW_NAME,
  CODEX_CAPTURE_WORKFLOW_PATH,
  SOURCE_RECONCILE_MAX_EVENTS,
  resolvePr,
  reserveThreadGeneration,
  finalizeThreadGeneration,
  releaseThreadGeneration,
  repairLegacyConflict,
  repairDelivery,
  repairOrphanThread,
  encodeThreadRepairRecord,
  decodeThreadRepairRecord,
  parseRepairPublicKeyring,
  historyRecoveryExhausted,
  scheduleQueuedCodexDrain,
  sweepQueuedCodexReviews,
  watchdogWindow,
  THREAD_REPAIR_MARK,
  CHECKPOINT_MARK,
  BOOTSTRAP_MARK,
  encodeCheckpoint,
  decodeCheckpoint,
  encodeBootstrapProgress,
  decodeBootstrapProgress,
};
