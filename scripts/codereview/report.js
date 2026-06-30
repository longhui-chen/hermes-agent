// Official Codex code review → 飞书报告：判定 + 卡片的唯一真源（single source of truth）。
//
// 既被本地回归测试（scripts/codereview/test_report.js）直接 require，
// 也经 scripts/codereview/notify.js 被官方 Codex Review 的 pull_request_review
// / check_run 监听路径消费。
//
// 背景：把 PR 评审结论统一成 pass / fail / skip，再决定是否给飞书发卡片。
// 评审结论有三种来源，本文件给每种来源各一个 adapter，下游卡片/判定逻辑共用：
//
//   A) 官方 Codex Review 的 check_run。
//      如果 GitHub 上出现 Codex check_run，结论从 check 的 output.text 里读。
//      兼容旧托管路径末尾的机器可读标记：
//        <!-- bughunter-severity: {"normal":2,"nit":1,"pre_existing":0} -->
//      normal=🔴 Important（合并前应修的真 bug）、nit=🟡 次要、pre_existing=🟣 存量；
//      output.text 里若有「严重度 | 文件:行 | 问题」表，best-effort 解析成缺陷清单。
//      → classifyCheckRun(checkRun)
//
//   B) 官方 Codex Review 的 pull_request_review。
//      Codex 像 reviewer 一样提交 PR review；行内评论里的 P0/P1/P2 badge 是优先级真源，
//      飞书卡片按同一 badge 展示。若 review body 明确显示报错 / 超时 / 额度问题，则按基础设施失败通知。
//      → classifyPullRequestReview(review, reviewComments)
//
//   C) 历史自托管 GitHub Action（openai/codex-action@v1）。
//      评审在 CI 里跑，prompt 要求 Codex GPT-5.5 额外吐一个 codereview-result.json，
//      形如 {"verdict":"fail","important":2,"nit":1,"issues":[{severity,file,line,title}]}。
//      当前 workflow 已移除；保留 adapter 只为旧测试/兼容历史通知数据。
//      → classifyActionResult(json)
//
// 判定二元化（与 linearb 流程一致）：有「合并前应修的真 bug」=不通过，否则=通过。
// 官方 Codex 报错/超时/额度问题=基础设施失败（发群排查）；拿不到结论=skip（不误报）。
// 历史自托管 Action 连 codereview-result.json 都没产出时，也按 CI/鉴权/基础设施故障通知。
// 「只在不通过时通知」：通过不打扰。默认门槛 = normal(Important) > 0，可用环境变量
// CODEREVIEW_NOTIFY_ON=any 放宽到「含 nit 也通知」。

/**
 * 通知门槛：默认只在出现 Important（合并前应修）时通知；设 CODEREVIEW_NOTIFY_ON=any
 * 时，nit 也算「有问题」。pre_existing（存量 bug，非本 PR 引入）从不单独触发通知。
 * @returns {'important'|'any'}
 */
function notifyThreshold(env) {
  const v = String((env || process.env || {}).CODEREVIEW_NOTIFY_ON || 'important').toLowerCase();
  return v === 'any' ? 'any' : 'important';
}

/**
 * 从托管 Code Review 的 check_run.output.text 里解析「严重度表」。best-effort：
 * 解析不到就返回空数组，count 仍以机器可读标记为准（清单只是给人看的明细）。
 * 表行形如： | 🔴 Important | `src/auth/session.ts:142` | Token refresh races... |
 * @param {string} text
 * @returns {Array<{sev:'important'|'nit'|'pre_existing', loc:string, title:string}>}
 */
function parseSeverityTable(text) {
  const out = [];
  const body = text || '';
  const re = /^\s*\|\s*(🔴|🟡|🟣)[^|]*\|\s*`?([^|`]+?)`?\s*\|\s*([^|]+?)\s*\|\s*$/gmu;
  let m;
  while ((m = re.exec(body)) !== null) {
    const sev = m[1] === '🔴' ? 'important' : m[1] === '🟡' ? 'nit' : 'pre_existing';
    const loc = m[2].trim();
    const title = m[3].trim();
    if (/^[-:\s]+$/.test(loc) || /^file:?line$/i.test(loc)) continue; // 跳过表头/分隔行
    out.push({ sev, loc, title });
  }
  return out;
}

function isInfrastructureFailureText(text) {
  return /encountered an error|timed out|spend cap|rate limit|quota|usage limit|was skipped|failed to run|could not run|internal error/i.test(String(text || ''));
}

function cleanIssueTitle(value) {
  const s = String(value || '')
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/[#>*_~]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
  return s.length > 180 ? `${s.slice(0, 177)}...` : s;
}

const PRIORITY = {
  p0: {
    label: 'P0',
    badge: '🔴 P0',
    sev: 'important',
  },
  p1: {
    label: 'P1',
    badge: '🟠 P1',
    sev: 'important',
  },
  p2: {
    label: 'P2',
    badge: '🟡 P2',
    sev: 'nit',
  },
};

function priorityLabel(priority) {
  return PRIORITY[priority] ? PRIORITY[priority].badge : '';
}

function parseCodexPriority(text) {
  const s = String(text || '');
  const m = s.match(/!\[\s*(P[0-2])\s+Badge\s*\]\([^)]*\)/i) ||
    s.match(/img\.shields\.io\/badge\/(P[0-2])-/i) ||
    s.match(/\b(P[0-2])\s*[:：-]/i);
  return m ? m[1].toLowerCase() : null;
}

function stripCodexPriorityBadge(text) {
  return String(text || '')
    .replace(/<\/?sub>/gi, ' ')
    .replace(/!\[\s*P[0-2]\s+Badge\s*\]\([^)]*\)/gi, ' ')
    .replace(/https:\/\/img\.shields\.io\/badge\/P[0-2]-[^)\s]+/gi, ' ')
    .replace(/\bP[0-2]\s+Badge\b/gi, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function extractCodexIssueTitle(body) {
  const raw = String(body || '');
  const firstBold = raw.match(/^\s*\*\*([\s\S]*?)\*\*/);
  const firstLine = raw.split(/\r?\n/).map((line) => line.trim()).find(Boolean) || raw;
  const title = firstBold ? firstBold[1] : firstLine;
  return cleanIssueTitle(stripCodexPriorityBadge(title));
}

function priorityCountsForIssues(issues) {
  const hasPriority = (issues || []).some((issue) => issue && issue.priority);
  if (!hasPriority) return null;
  const counts = { p0: 0, p1: 0, p2: 0 };
  for (const issue of issues || []) {
    const priority = issue && issue.priority ? issue.priority : 'p1';
    if (!Object.prototype.hasOwnProperty.call(counts, priority)) continue;
    counts[priority] += 1;
  }
  return counts;
}

function infrastructureFailure(source, failure = {}) {
  const reason = oneLine(failure.reason, 'unknown error');
  const status = oneLine(failure.status || '');
  const runUrl = oneLine(failure.runUrl || '');
  const titleParts = [source || 'Codex review infrastructure failure'];
  if (status) titleParts.push(status);
  if (reason) titleParts.push(`原因：${reason}`);
  if (runUrl) titleParts.push(`链接：${runUrl}`);
  return {
    verdict: 'fail',
    reason: 'infra_failure',
    kind: 'infra_failure',
    counts: { important: 1, nit: 0, pre_existing: 0 },
    issues: [{
      sev: 'important',
      loc: failure.loc || 'Codex Review',
      title: titleParts.join(' · '),
    }],
    count: 1,
  };
}

/**
 * 把「托管 Code Review」的 check_run 判定为 pass / fail / skip。
 * @param {{name?:string, output?:{title?:string, summary?:string, text?:string}}} checkRun
 * @param {object} [env]
 * @returns {{verdict:'pass'|'fail'|'skip', reason:string, counts:{important:number,nit:number,pre_existing:number}, issues:Array, count:number}}
 */
function classifyCheckRun(checkRun, env) {
  const output = (checkRun && checkRun.output) || {};
  const title = String(output.title || '');
  const summary = String(output.summary || '');
  const text = String(output.text || output.summary || '');
  const empty = { important: 0, nit: 0, pre_existing: 0 };

  // 评审报错 / 超时（基础设施没跑完）→ 基础设施失败通知；但只扫 title + summary（短状态字段）。
  // ⚠️ 只扫 title + summary（短状态字段），**绝不**扫 findings 正文 text——否则一条形如
  //   「请求 timed out 时会一直 hang」的真不通过缺陷会被误判为 infra failure。
  //   真正的基础设施信号（'Code review encountered an error / timed out'）只出现在 title。
  if (isInfrastructureFailureText(title + ' ' + summary)) {
    return infrastructureFailure('Codex check_run 未完成评审', {
      status: checkRun && checkRun.conclusion ? `conclusion=${checkRun.conclusion}` : '',
      reason: title || summary || 'Codex check_run failed',
      runUrl: checkRun && checkRun.html_url,
    });
  }

  // 机器可读严重度标记（官方稳定契约）。
  const sm = text.match(/bughunter-severity:\s*(\{[\s\S]*?\})\s*-->/);
  let sev = null;
  if (sm) { try { sev = JSON.parse(sm[1]); } catch (_e) { sev = null; } }

  if (!sev) {
    // 没拿到机器可读标记。标题明确「无问题」才判通过，否则判 skip（不可判定，不误报）。
    if (/no issues|no problems|looks good|lgtm/i.test(title)) {
      return { verdict: 'pass', reason: 'no_issues', counts: empty, issues: [], count: 0 };
    }
    return { verdict: 'skip', reason: 'inconclusive', counts: empty, issues: [], count: 0 };
  }

  const counts = {
    important: Number(sev.normal || 0) || 0,
    nit: Number(sev.nit || 0) || 0,
    pre_existing: Number(sev.pre_existing || 0) || 0,
  };
  const issues = parseSeverityTable(text);
  const threshold = notifyThreshold(env);
  const trigger = threshold === 'any' ? counts.important + counts.nit : counts.important;

  if (trigger > 0) {
    return { verdict: 'fail', reason: 'issues', counts, issues, count: trigger };
  }
  return { verdict: 'pass', reason: 'clean', counts, issues, count: 0 };
}

/**
 * 把官方 Codex 的 pull_request_review 判定为 pass / fail / skip。
 * 官方集成会像普通 reviewer 一样在 PR 里留下 review + 行内评论；当前没有
 * codereview-result.json 这类结构化文件，所以用保守策略：
 * - 有 P0/P1 行内 review comment：按 Important 通知；只有 P2 及以下时视为非阻塞，不发飞书。
 * - review body 明确报错/超时/额度：基础设施失败通知。
 * - 明确 clean：pass，不发。
 * - 其它：skip，避免把普通状态文本误当代码问题。
 *
 * @param {{body?:string,state?:string,html_url?:string,submitted_at?:string}} review
 * @param {Array<{path?:string,line?:number,original_line?:number,body?:string}>} reviewComments
 * @param {object} [env]
 */
function classifyPullRequestReview(review, reviewComments, env) {
  const body = String((review && review.body) || '');
  const comments = Array.isArray(reviewComments) ? reviewComments : [];
  const statusText = body.slice(0, 2000);
  if (isInfrastructureFailureText(statusText)) {
    return infrastructureFailure('Codex PR review 未完成评审', {
      status: review && review.state ? `state=${review.state}` : '',
      reason: cleanIssueTitle(statusText) || 'Codex review failed',
      runUrl: review && review.html_url,
    });
  }
  let issues = comments
    .filter((c) => c && String(c.body || '').trim())
    .map((c) => {
      const priority = parseCodexPriority(c.body);
      return {
        sev: priority && PRIORITY[priority] ? PRIORITY[priority].sev : 'important',
        priority,
        loc: [c.path, c.line || c.original_line].filter((x) => x != null && x !== '').join(':'),
        title: extractCodexIssueTitle(c.body),
      };
    });
  if (issues.length) {
    if (issues.some((issue) => issue.priority)) {
      issues = issues.map((issue) => issue.priority ? issue : { ...issue, priority: 'p1' });
    }
    const priorityCounts = priorityCountsForIssues(issues);
    const counts = priorityCounts
      ? {
        important: priorityCounts.p0 + priorityCounts.p1,
        nit: priorityCounts.p2,
        pre_existing: 0,
      }
      : { important: issues.length, nit: 0, pre_existing: 0 };
    const threshold = notifyThreshold(env);
    const trigger = threshold === 'any' ? counts.important + counts.nit : counts.important;
    if (trigger <= 0) {
      return {
        verdict: 'pass',
        reason: 'non_blocking',
        counts,
        priorityCounts,
        issues,
        count: 0,
      };
    }
    return {
      verdict: 'fail',
      reason: 'issues',
      counts,
      priorityCounts,
      issues,
      count: trigger,
    };
  }
  if (/no issues|no problems|looks good|lgtm|no blocking/i.test(body)) {
    return {
      verdict: 'pass',
      reason: 'clean',
      counts: { important: 0, nit: 0, pre_existing: 0 },
      issues: [],
      count: 0,
    };
  }
  return {
    verdict: 'skip',
    reason: 'inconclusive',
    counts: { important: 0, nit: 0, pre_existing: 0 },
    issues: [],
    count: 0,
  };
}

/**
 * 把自托管 Action 产出的 codereview-result.json 判定为 pass / fail / skip。
 * @param {{verdict?:string, important?:number, nit?:number, pre_existing?:number,
 *          issues?:Array<{severity?:string, file?:string, line?:(number|string), title?:string}>}} json
 * @param {object} [env]
 */
function classifyActionResult(json, env) {
  const empty = { important: 0, nit: 0, pre_existing: 0 };
  if (!json || typeof json !== 'object') {
    return { verdict: 'skip', reason: 'inconclusive', counts: empty, issues: [], count: 0 };
  }
  const issuesRaw = Array.isArray(json.issues) ? json.issues : [];
  const counts = {
    important: Number(json.important || 0) || issuesRaw.filter((i) => /important|high|critical|bug/i.test(String(i.severity))).length,
    nit: Number(json.nit || 0) || issuesRaw.filter((i) => /nit|minor|low/i.test(String(i.severity))).length,
    pre_existing: Number(json.pre_existing || 0) || 0,
  };
  const issues = issuesRaw.map((i) => ({
    sev: /important|high|critical|bug/i.test(String(i.severity)) ? 'important'
      : /nit|minor|low/i.test(String(i.severity)) ? 'nit' : 'pre_existing',
    loc: [i.file, i.line].filter((x) => x != null && x !== '').join(':'),
    title: String(i.title || ''),
  }));
  const threshold = notifyThreshold(env);
  const trigger = threshold === 'any' ? counts.important + counts.nit : counts.important;
  if (trigger > 0) return { verdict: 'fail', reason: 'issues', counts, issues, count: trigger };
  return { verdict: 'pass', reason: 'clean', counts, issues, count: 0 };
}

function oneLine(value, fallback = '') {
  const s = String(value == null ? fallback : value).replace(/\s+/g, ' ').trim();
  return s.length > 260 ? `${s.slice(0, 257)}...` : s;
}

/**
 * 自托管 Action 未产出 codereview-result.json 时的基础设施故障分类。
 * 这不是“评审发现代码问题”，但它会让主干保护失去可信评审结果，必须通知群里排查。
 * @param {{reason?:string, stepOutcome?:string, stepConclusion?:string, runUrl?:string}} failure
 */
function classifyActionFailure(failure = {}) {
  const reason = oneLine(failure.reason, 'unknown error');
  const step = oneLine(failure.stepOutcome || failure.stepConclusion || '');
  const runUrl = oneLine(failure.runUrl || '');
  const titleParts = ['未产出 codereview-result.json'];
  if (step) titleParts.push(`review step=${step}`);
  if (reason) titleParts.push(`原因：${reason}`);
  if (runUrl) titleParts.push(`Actions：${runUrl}`);
  return infrastructureFailure('历史自托管 Codex Action 未产出结果', {
    loc: '.github/workflows/codex-code-review.yml',
    reason: titleParts.join(' · '),
  });
}

/**
 * 是否应该给飞书发报告。与 linearb 流程一致：只有 fail 才推；pass / skip 都不发（不打扰）。
 * @param {{verdict:string}} cls
 */
function shouldNotify(cls) {
  return !!cls && cls.verdict === 'fail';
}

// GitHub 登录名 → 飞书 open_id（用于「不通过」卡片里 @ 出 PR 作者）。唯一真源 =
// scripts/test-harness/feishu/feishu_openid_registry.json 的 at_by_github_active。
//
// ⚠️ open_id 按 app 维度隔离：本流程与 linearb-feishu-report 同一只应用机器人
//   (cli_a97acaec84389cc0) 发送 + 起话题。必须用「该应用机器人维度」的 open_id——
//   用错维度会被飞书判为 cross-app 整卡拒收。所以这里**只**读 at_by_github_active；
//   映射缺失时退化为不 @、不报错（best-effort），补全走 build_registry.py。
const OPENID_BY_GH = (() => {
  try {
    const reg = require('../test-harness/feishu/feishu_openid_registry.json');
    return reg.at_by_github_active || {};
  } catch (_e) {
    return {};
  }
})();

/**
 * 给 PR 作者生成飞书 @ 标签；映射不到则返回空串（退化为不 @）。
 * lark_md 用 `<at id=ou_xxx></at>` @ 人（应用机器人卡片支持）。
 * @param {string} login GitHub 登录名
 */
function atTag(login) {
  const id = OPENID_BY_GH[login];
  return id ? `<at id=${id}></at>` : '';
}

// lark_md 转义：PR 标题 / 分支 / 作者 / Codex issue 文案均为用户可控，不转义可注入伪造
// 可点击链接，或用换行塞入假「**结论**：✅ 通过」行伪造结论。pr.number 为整数、
// pr.url 取自 GitHub html_url（可信）不转义。
// ⚠️ 必须中和尖括号 < > —— 否则攻击者用 PR 标题/分支/文件名塞入 `<at id=all></at>`（伪造 @、甚至 @所有人）
//   或 `<a href=http://evil>x</a>`（在可信评审群卡片里注入钓鱼链接），因为卡片按 lark_md 渲染、
//   <at>/<a> 正是飞书自己的标签语法。用形近字符替换，保证绝不构成真标签（飞书是否渲染 HTML 实体不确定）。
function esc(s) {
  return String(s == null ? '' : s)
    .replace(/[\r\n]+/g, ' ')
    .replace(/</g, '‹').replace(/>/g, '›') // ‹ › 形近，杜绝 <at>/<a> 标签注入
    .replace(/\\/g, '\\\\')
    .replace(/([\[\]()*_~`])/g, '\\$1');
}

const SEV_ICON = { important: '🔴', nit: '🟡', pre_existing: '🟣' };

function priorityTally(priorityCounts) {
  if (!priorityCounts) return '';
  return [
    `${priorityLabel('p0')} ${priorityCounts.p0 || 0}`,
    `${priorityLabel('p1')} ${priorityCounts.p1 || 0}`,
    `${priorityLabel('p2')} ${priorityCounts.p2 || 0}`,
  ].join(' · ');
}

function issueMarker(issue) {
  if (issue && issue.priority) return priorityLabel(issue.priority) || '•';
  return SEV_ICON[issue && issue.sev] || '•';
}

function displayIssueCount(cls) {
  const pc = cls && cls.priorityCounts;
  if (pc) return (pc.p0 || 0) + (pc.p1 || 0) + (pc.p2 || 0);
  return cls ? cls.count : 0;
}

/**
 * 组飞书交互卡片。
 * @param {string} repo 仓库名
 * @param {{number:number,title:string,url:string,author:string,base:string,head:string}} pr
 * @param {ReturnType<classifyCheckRun>} cls
 * @param {{atAuthor?:boolean, isReply?:boolean}} [opts]
 */
function buildCard(repo, pr, cls, opts = {}) {
  const atAuthor = opts.atAuthor !== false;
  const ok = cls.verdict === 'pass';
  const infra = cls.reason === 'infra_failure' || cls.kind === 'infra_failure';
  const c = cls.counts || { important: 0, nit: 0, pre_existing: 0 };
  const issueCount = displayIssueCount(cls);
  const head = ok
    ? '✅ Codex 代码评审通过'
    : infra
      ? '🚨 Codex 代码评审基础设施失败'
    : `❌ Codex 代码评审不通过 — ${issueCount} 个问题`;
  const authorAt = !ok && atAuthor ? atTag(pr.author) : '';
  const tally = cls.priorityCounts
    ? priorityTally(cls.priorityCounts)
    : `🔴 ${c.important} · 🟡 ${c.nit} · 🟣 ${c.pre_existing}`;

  // 常显区（= 截图里红框）：仓库 / PR / 提交者，扫一眼就知道是谁的哪个 PR。
  const headLines = [
    `**仓库**：${esc(repo)}`,
    `**PR**：[#${pr.number} ${esc(pr.title)}](${pr.url})`,
    `**提交者**：${authorAt ? authorAt + ' ' : ''}${esc(pr.author)}`,
  ];
  // 折叠区：分支 / 结论 / 严重度 / 问题清单，点「展开详情」才显示，不刷屏。
  const detailLines = [
    `**分支**：${esc(pr.base)} ← ${esc(pr.head)}`,
    `**结论**：${ok ? '✅ 通过（无需合并前修复的问题）' : infra ? '🚨 评审基础设施失败（未产出 codereview-result.json）' : `❌ 不通过（${issueCount} 个问题）`}`,
    `**${cls.priorityCounts ? '优先级' : '严重度'}**：${infra ? '基础设施故障（需排查 Action / Codex 鉴权 / 超时）' : tally}`,
  ];
  if (!ok && cls.issues && cls.issues.length) {
    detailLines.push(infra ? '\n**故障线索**：' : '\n**问题清单**：');
    cls.issues.slice(0, 10).forEach((it, i) => {
      const icon = issueMarker(it);
      const loc = it.loc ? ` \`${esc(it.loc)}\`` : '';
      detailLines.push(`${i + 1}. ${icon}${loc}${it.title ? ' — ' + esc(it.title) : ''}`);
    });
    if (cls.issues.length > 10) detailLines.push(`… 另有 ${cls.issues.length - 10} 项，详见 PR 行内评论`);
  } else if (!ok) {
    detailLines.push('\n详见 PR 的 **Codex Code Review** 评论。');
  }
  const note = opts.isReply
    ? infra ? 'Codex · 同一 PR 复评 · 基础设施故障回复' : 'Codex · 同一 PR 复评 · 话题回复'
    : infra ? 'Codex · PR → main · 基础设施故障自动告警' : 'Codex · PR → main · 自动触发';

  // 飞书 card schema 2.0：用 collapsible_panel 做「红框常显 + 其余下拉展开」。
  return {
    msg_type: 'interactive',
    card: {
      schema: '2.0',
      config: { wide_screen_mode: true, update_multi: true },
      header: { template: ok ? 'green' : 'red', title: { tag: 'plain_text', content: head } },
      body: {
        elements: [
          { tag: 'markdown', content: headLines.join('\n') },
          {
            tag: 'collapsible_panel',
            expanded: false,
            header: {
              title: { tag: 'markdown', content: '**展开详情**（分支 · 结论 · 严重度 · 问题清单）' },
              vertical_align: 'center',
              icon: { tag: 'standard_icon', token: 'down-small-ccm_outlined', size: '16px 16px' },
              icon_position: 'right',
              icon_expanded_angle: -180,
            },
            elements: [
              { tag: 'markdown', content: detailLines.join('\n') },
            ],
          },
          { tag: 'hr' },
          // card 2.0 不支持 note 标签：脚注改用灰色 markdown；按钮跳转改用 behaviors.open_url。
          { tag: 'markdown', content: `<font color='grey'>${note}</font>` },
          { tag: 'button', text: { tag: 'plain_text', content: '打开 PR' }, type: 'primary', width: 'default', behaviors: [{ type: 'open_url', default_url: pr.url }] },
        ],
      },
    },
  };
}

// Feishu im/v1/messages（msg_type=interactive）要求 content 是卡片 body 的 JSON 字符串，
// 而 incoming webhook 要求 {msg_type, card}。把这层 API 形状转换收在一处，避免发送/回复
// 路径与 buildCard 返回形状漂移。
function interactiveCardContent(card) {
  const body = card && card.card && card.card.body;
  if (!card || card.msg_type !== 'interactive' || !body || !Array.isArray(body.elements)) {
    throw new Error('invalid interactive card payload');
  }
  return JSON.stringify(card.card);
}

module.exports = {
  classifyCheckRun,
  classifyPullRequestReview,
  classifyActionResult,
  classifyActionFailure,
  isInfrastructureFailureText,
  parseSeverityTable,
  shouldNotify,
  buildCard,
  interactiveCardContent,
  atTag,
  notifyThreshold,
  OPENID_BY_GH,
};
