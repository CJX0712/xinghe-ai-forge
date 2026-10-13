/* 构建脚本：把 src/ai-forge.js 内联进模板，产出零外部依赖的单文件 index.html
 * 用法：node tools/build-html.js  （保证 HTML 与源码不漂移，可复现） */
const fs = require('fs');
const path = require('path');

const root = path.join(__dirname, '..');
const src = fs.readFileSync(path.join(root, 'src', 'ai-forge.js'), 'utf8');
const tpl = fs.readFileSync(path.join(root, 'tools', 'index.template.html'), 'utf8');

if (!tpl.includes('<!--AIFORGE_SOURCE-->')) { console.error('模板缺少 <!--AIFORGE_SOURCE--> 占位符'); process.exit(1); }
// 用 <script> 包裹注入；避免源码里出现 </script> 破坏结构
const safe = src.replace(/<\/script>/gi, '<\\/script>');
const out = tpl.replace('<!--AIFORGE_SOURCE-->', '<script>\n' + safe + '\n</script>');

fs.writeFileSync(path.join(root, 'index.html'), out);
console.log('index.html 生成完毕，大小', (out.length / 1024).toFixed(1), 'KB，内联源码', (src.length / 1024).toFixed(1), 'KB');
