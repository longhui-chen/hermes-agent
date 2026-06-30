// Codex GPT-5.5 code review → 飞书报告：判定 + 卡片的唯一真源（single source of truth）。
//
// 既被本地回归测试（scripts/codereview/test_report.js）直接 require，
// 也经 scripts/codereview/notify.js 被托管 Code Review 的 check_run 路径
// 和自托管 Codex GitHub Action 路径消费。
//
// 背景：把 PR 评审结论统一成 pass / fail / skip，再决定是否给飞书发卡片。
// 评审结论有两种来源，本文件给两种来源各一个 adapter，下游卡片/判定逻辑共用：
//
//   A) 托管 Code Review。
//      它在 GitHub 上以 check_run 出现，结论从 check 的
//      output.text 里读。末尾带一行机器可读标记（官方文档承诺的稳定契约）：
//        <!-- bughunter-severity: {"normal":2,"nit":1,"pre_existing":0} -->
//      normal=🔴 Important（合并前应修的真 bug）、nit=🟡 次要、pre_existing=🟣 存量。
//      output.text 里还有一张「严重度 | 文件:行 | 问题」表，best-effort 解析成缺陷清单。
//      → classifyCheckRun(checkRun)
//
//   B) 自托管 GitHub Action（openai/codex-action@v1）。
//      评审在 CI 里跑，prompt 要求 Codex GPT-5.5 额外吐一个 codereview-result.json，
//      形如 {"verdict":"fail","important":2,"nit":1,"issues":[{severity,file,line,title}]}。
//      → classifyActionResult(json)
//
// 判定二元化（与 linearb 流程一致）：有「合并前应修的真 bug」=不通过，否则=通过。
// 托管 check_run 报错/超时/拿不到结论=skip（不发，避免误报「0 个问题也不通过」）；
// 自托管 Action 连 codereview-result.json 都没产出时，是 CI/鉴权/基础设施故障，单独发故障卡片。
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

  // 评审报错 / 超时（基础设施没跑完）→ skip，绝不当成「0 问题也不通过」误报。
  // ⚠️ 只扫 title + summary（短状态字段），**绝不**扫 findings 正文 text——否则一条形如
  //   「请求 timed out 时会一直 hang」的真不通过缺陷会被误判为 skip 而不通知（false-negative）。
  //   真正的基础设施信号（'Code review encountered an error / timed out'）只出现在 title。
  if (/encountered an error|timed out|spend cap|was skipped/i.test(title + ' ' + summary)) {
    return { verdict: 'skip', reason: 'error_or_timeout', counts: empty, issues: [], count: 0 };
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
  return {
    verdict: 'fail',
    reason: 'infra_failure',
    kind: 'infra_failure',
    counts: { important: 1, nit: 0, pre_existing: 0 },
    issues: [{
      sev: 'important',
      loc: '.github/workflows/codex-code-review.yml',
      title: titleParts.join(' · '),
    }],
    count: 1,
  };
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
  const head = ok
    ? '✅ Codex GPT-5.5 代码评审通过'
    : infra
      ? '🚨 Codex GPT-5.5 代码评审基础设施失败'
    : `❌ Codex GPT-5.5 代码评审不通过 — ${cls.count} 个问题`;
  const authorAt = !ok && atAuthor ? atTag(pr.author) : '';
  const tally = `🔴 ${c.important} · 🟡 ${c.nit} · 🟣 ${c.pre_existing}`;

  // 常显区（= 截图里红框）：仓库 / PR / 提交者，扫一眼就知道是谁的哪个 PR。
  const headLines = [
    `**仓库**：${esc(repo)}`,
    `**PR**：[#${pr.number} ${esc(pr.title)}](${pr.url})`,
    `**提交者**：${authorAt ? authorAt + ' ' : ''}${esc(pr.author)}`,
  ];
  // 折叠区：分支 / 结论 / 严重度 / 问题清单，点「展开详情」才显示，不刷屏。
  const detailLines = [
    `**分支**：${esc(pr.base)} ← ${esc(pr.head)}`,
    `**结论**：${ok ? '✅ 通过（无需合并前修复的问题）' : infra ? '🚨 评审基础设施失败（未产出 codereview-result.json）' : `❌ 不通过（${cls.count} 个问题）`}`,
    `**严重度**：${infra ? '基础设施故障（需排查 Action / Codex 鉴权 / 超时）' : tally}`,
  ];
  if (!ok && cls.issues && cls.issues.length) {
    detailLines.push(infra ? '\n**故障线索**：' : '\n**问题清单**：');
    cls.issues.slice(0, 10).forEach((it, i) => {
      const icon = SEV_ICON[it.sev] || '•';
      const loc = it.loc ? ` \`${esc(it.loc)}\`` : '';
      detailLines.push(`${i + 1}. ${icon}${loc}${it.title ? ' — ' + esc(it.title) : ''}`);
    });
    if (cls.issues.length > 10) detailLines.push(`… 另有 ${cls.issues.length - 10} 项，详见 PR 行内评论`);
  } else if (!ok) {
    detailLines.push('\n详见 PR 的 **Codex GPT-5.5 Code Review** 评论。');
  }
  const note = opts.isReply
    ? infra ? 'Codex GPT-5.5 · 同一 PR 复评 · 基础设施故障回复' : 'Codex GPT-5.5 · 同一 PR 复评 · 话题回复'
    : infra ? 'Codex GPT-5.5 · PR → main · 基础设施故障自动告警' : 'Codex GPT-5.5 · PR → main · 自动触发';

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
  classifyActionResult,
  classifyActionFailure,
  parseSeverityTable,
  shouldNotify,
  buildCard,
  interactiveCardContent,
  atTag,
  notifyThreshold,
  OPENID_BY_GH,
};
