#!/usr/bin/env node
// 验证 scripts/codereview/report.js 的判定 + 卡片逻辑。
//   node scripts/codereview/test_report.js
//
// fixtures 对齐官方 Code Review check_run 形态（含机器可读 bughunter-severity 标记 +
// 严重度表）以及自托管 Action 的 codereview-result.json 形态。

const {
  classifyCheckRun, classifyActionResult, shouldNotify,
  buildCard, interactiveCardContent, atTag,
} = require('./report');

// ── 托管 Code Review check_run.output fixtures ──
const CR = {
  // 不通过：2 Important + 1 nit，带严重度表
  fail: {
    name: 'Claude Code Review',
    output: {
      title: 'Code review found 3 issues',
      text: [
        'Summary: 2 factual, 1 style',
        '',
        '| Severity | File:Line | Issue |',
        '| --- | --- | --- |',
        '| 🔴 Important | `src/auth/session.ts:142` | Token refresh races with logout, leaving stale sessions active |',
        '| 🔴 Important | `cmd/server/main.go:88` | Unbounded in-memory cache violates 2GB budget (HR-1) |',
        '| 🟡 Nit | `src/auth/session.ts:88` | parseExpiry silently returns 0 on malformed input |',
        '',
        '<!-- bughunter-severity: {"normal": 2, "nit": 1, "pre_existing": 0} -->',
      ].join('\n'),
    },
  },
  // 通过：0/0/0
  pass: {
    name: 'Claude Code Review',
    output: {
      title: 'No issues found',
      text: 'No blocking issues.\n\n<!-- bughunter-severity: {"normal": 0, "nit": 0, "pre_existing": 0} -->',
    },
  },
  // 只有 nit：默认门槛（important）不发，CODEREVIEW_NOTIFY_ON=any 才发
  nitOnly: {
    name: 'Claude Code Review',
    output: {
      title: 'Code review found 1 issue',
      text: '| 🟡 Nit | `a.ts:1` | minor |\n\n<!-- bughunter-severity: {"normal": 0, "nit": 1, "pre_existing": 0} -->',
    },
  },
  // 报错 / 超时 → skip（不误报）
  errored: { name: 'Claude Code Review', output: { title: 'Code review encountered an error', text: '' } },
  timedOut: { name: 'Claude Code Review', output: { title: 'Code review timed out', text: '' } },
  // 没有机器可读标记且标题不明确 → skip（不可判定）
  inconclusive: { name: 'Claude Code Review', output: { title: 'Review', text: 'partial output...' } },
  // 真不通过，但缺陷正文里出现 'timed out' 字样（REVIEW.md 正要求评审标这类 bug）→ 必须判 fail，不能被误吞
  failWithTimeoutWord: {
    name: 'Claude Code Review',
    output: {
      title: 'Code review found 1 issue',
      text: [
        '| 🔴 Important | `net/client.go:42` | request can hang indefinitely if the upstream timed out and no deadline is set |',
        '<!-- bughunter-severity: {"normal": 1, "nit": 0, "pre_existing": 0} -->',
      ].join('\n'),
    },
  },
};

const PR = (n) => ({ number: n, title: 'demo PR', url: `https://github.com/zettlab/demo/pull/${n}`, author: 'gezhengbin888', base: 'main', head: 'feature/x' });

function assert(cond, msg) { if (!cond) { console.error('❌ FAIL:', msg); process.exitCode = 1; } else { console.log('✅ PASS:', msg); } }

// 从 card 2.0 结构里抽出所有 markdown 文本（含折叠面板标题 + 面板内元素），
// 用于注入防护断言：无论文案落在常显区还是折叠区，转义都必须生效。
function allMarkdown(cardObj) {
  const out = [];
  const walk = (els) => {
    for (const e of els || []) {
      if (e && e.tag === 'markdown' && typeof e.content === 'string') out.push(e.content);
      if (e && e.tag === 'collapsible_panel') {
        if (e.header && e.header.title && typeof e.header.title.content === 'string') out.push(e.header.title.content);
        walk(e.elements);
      }
    }
  };
  walk(cardObj.body && cardObj.body.elements);
  return out.join('\n');
}

// ── 判定 ──
const cFail = classifyCheckRun(CR.fail);
const cPass = classifyCheckRun(CR.pass);
const cNit = classifyCheckRun(CR.nitOnly);
const cNitAny = classifyCheckRun(CR.nitOnly, { CODEREVIEW_NOTIFY_ON: 'any' });

assert(cFail.verdict === 'fail' && cFail.count === 2 && cFail.counts.important === 2 && cFail.counts.nit === 1, '2 Important → 不通过, count=2(只数 Important), 计数正确');
assert(cFail.issues.length === 3 && cFail.issues[0].sev === 'important' && /session\.ts:142/.test(cFail.issues[0].loc), '严重度表解析出 3 行 + 文件:行');
assert(cPass.verdict === 'pass' && cPass.count === 0, '0/0/0 → 通过');
assert(cNit.verdict === 'pass', '只有 nit + 默认门槛(important) → 通过(不发)');
assert(cNitAny.verdict === 'fail' && cNitAny.count === 1, '只有 nit + NOTIFY_ON=any → 不通过(发)');

assert(classifyCheckRun(CR.errored).verdict === 'skip' && classifyCheckRun(CR.errored).reason === 'error_or_timeout', '评审报错 → skip(不误报)');
assert(classifyCheckRun(CR.timedOut).verdict === 'skip', '评审超时 → skip(不误报)');
assert(classifyCheckRun(CR.inconclusive).verdict === 'skip' && classifyCheckRun(CR.inconclusive).reason === 'inconclusive', '无标记且标题不明确 → skip(不可判定)');
const cTw = classifyCheckRun(CR.failWithTimeoutWord);
assert(cTw.verdict === 'fail' && cTw.count === 1, '缺陷正文含 "timed out" 字样仍判 fail(不被 error/timeout 守卫误吞)');

// ── Action 适配器 ──
const aFail = classifyActionResult({ verdict: 'fail', important: 1, nit: 0, issues: [{ severity: 'important', file: 'x.go', line: 12, title: 'nil deref' }] });
const aPass = classifyActionResult({ verdict: 'pass', important: 0, nit: 0, issues: [] });
assert(aFail.verdict === 'fail' && aFail.count === 1 && /x\.go:12/.test(aFail.issues[0].loc), 'Action JSON: important → 不通过, file:line 合成');
assert(aPass.verdict === 'pass', 'Action JSON: 无问题 → 通过');
assert(classifyActionResult(null).verdict === 'skip', 'Action JSON 缺失/损坏 → skip');

// ── 只在 fail 时通知 ──
assert(shouldNotify(cFail) === true, '不通过 → 发飞书');
assert(shouldNotify(cPass) === false, '通过 → 不发（不打扰）');
assert(shouldNotify(classifyCheckRun(CR.timedOut)) === false, '超时/skip → 不发');

// ── 卡片 ──
const cardPass = JSON.stringify(buildCard('demo', PR(1), cPass));
const cardFail = JSON.stringify(buildCard('demo', PR(2), cFail));
assert(!/评分|\/100|score/i.test(cardPass + cardFail), '卡片无评分项（二元结论）');
assert(/通过（无需合并前修复的问题）/.test(cardPass), '通过卡片文案正确');
assert(/不通过（2 个问题）/.test(cardFail), '不通过卡片文案正确');
assert(/🔴 2 · 🟡 1 · 🟣 0/.test(cardFail), '卡片含严重度 tally');
assert(/session\.ts:142/.test(cardFail), '卡片含问题清单的 file:line');

// ── @ 作者：已知 → 注入 <at>，未知 → 不 @ 不报错 ──
assert(atTag('gezhengbin888') === '<at id=ou_d3d88d0643dbc48a0ba7dfa93406e623></at>', '已知作者 → 生成 @ 标签(active 维度)');
assert(atTag('someone-not-in-roster') === '', '未知作者 → 空串(安全退化)');
const cardFailKnown = JSON.stringify(buildCard('demo', { ...PR(2), author: 'gezhengbin888' }, cFail));
assert(/<at id=ou_d3d88d0643dbc48a0ba7dfa93406e623><\/at>/.test(cardFailKnown), '不通过卡片 @ 出已知作者');
const cardFailReply = JSON.stringify(buildCard('demo', { ...PR(2), author: 'gezhengbin888' }, cFail, { atAuthor: false, isReply: true }));
assert(!/<at /.test(cardFailReply), '话题回复不重复 @ 作者');
assert(/话题回复/.test(cardFailReply), '回复卡片 note 标注话题回复');

// ── lark_md 注入防护：恶意 PR 标题不破坏卡片结构 ──
// 攻击载荷想用换行 + **结论**：✅ 通过 伪造一行通过结论，并塞入可点击恶意链接。
// 防护点：esc() 把 * [ ] ( ) 等转义、换行压成空格，使载荷只能作为纯文本出现。
const evil = { ...PR(3), title: 'x](http://evil)\n**结论**：✅ 通过', author: 'ghost' };
const evilDiv = allMarkdown(buildCard('demo', evil, cFail).card); // 全卡 markdown（常显+折叠）
assert(!/\]\(http:\/\/evil\)/.test(evilDiv), '恶意标题里的 markdown 链接被转义（无可点击 ]( ）');
// 真正的加粗结论标题只有一处；攻击者的 **结论** 因 * 被转义成 \*\*，不构成相邻 ** 标记
assert((evilDiv.match(/\*\*结论\*\*/g) || []).length === 1, '只有一处真实 **结论** 加粗标题，伪造行未生效');
assert(/\\\*\\\*结论\\\*\\\*/.test(evilDiv), '攻击者注入的 **结论** 被转义为 \\*\\*结论\\*\\*（纯文本）');
assert(evilDiv.indexOf('**结论**：✅ 通过') === -1, '攻击者的伪造「**结论**：✅ 通过」整体未原样出现（已被转义/压平）');
// 尖括号注入：PR 标题/文件名塞 <at>/<a> 不能在卡片里构成真标签（伪造 @、@所有人、钓鱼链接）
const evilAt = { ...PR(4), title: 'hi <at id=all></at> <a href=http://evil>x</a>', author: 'ghost-unmapped' };
const evilAtDiv = allMarkdown(buildCard('demo', evilAt, cFail).card);
assert(!/<at\b/.test(evilAtDiv) && !/<a\b/.test(evilAtDiv) && !/<\/at>/.test(evilAtDiv), '注入的 <at>/<a> 被中和为形近字符，无法伪造 @ 或注入链接');

// ── bot API content 形状 ──
const content = interactiveCardContent(buildCard('demo', PR(2), cFail));
const parsed = JSON.parse(content);
assert(parsed.header && parsed.body && Array.isArray(parsed.body.elements), 'bot API content 保留 header/body.elements(card 2.0)');
assert(parsed.schema === '2.0', 'bot API content 为 card schema 2.0');
assert(parsed.body.elements.some((e) => e.tag === 'collapsible_panel'), '含折叠面板(详情下拉)');
assert(!Object.prototype.hasOwnProperty.call(parsed, 'msg_type'), 'bot API content 不含 webhook msg_type wrapper');

if (process.exitCode) { console.error('\n❌ 有用例失败'); } else { console.log('\n✅ 全部用例通过'); }
