/* 无头 Node 自检：node test/run.js */
const AI = require('../src/ai-forge.js');
const { selftest } = AI;

const t0 = Date.now();
const r = selftest();
const ms = Date.now() - t0;

console.log(`\n星核 AI-Forge 自检  v${AI.VERSION}   (${ms} ms)\n`);
let fails = 0;
for (const t of r.results) {
  const tag = t.pass ? '\u2705 PASS' : '\u274C FAIL';
  if (!t.pass) fails++;
  console.log(`  ${tag}  ${t.name}   [${t.detail}]`);
}
console.log(`\n结果：${r.passed}/${r.total} 通过  ${r.allPass ? '\u2705 全绿' : '\u26A0 有失败项 ' + fails}`);
process.exit(r.allPass ? 0 : 1);
