# ✦ 星核 AI-Forge

> **零依赖 · 单文件 · 可验证的世界级 AI 核心**
> 数值线性代数 → 经典机器学习 → 深度学习，浏览器 `<script>` 与 Node `require` 双跑。
> 作者：**晨星**（CJX0712）· MIT License

[![selftest](https://img.shields.io/badge/selftest-17%2F17-brightgreen)](#-一键自检) [![deps](https://img.shields.io/badge/dependencies-0-blue)](#) [![license](https://img.shields.io/badge/license-MIT-blue)](#license)

---

## 为什么是「集大成 · 零重写」

每一行代码都站在世界顶级成果之上，而非从零拍脑袋。核心算法采用各自领域公认的最优实现，并内置**可被另一套独立实现交叉验证的「不变量」**——这是本项目与「玩具级复刻」的根本区别。

| 层 | 模块 | 采用的世界级算法 | 可验证不变量 |
|---|---|---|---|
| 数值地基 | Cholesky | 对称正定分解 `A=LLᵀ` | `‖LLᵀ−A‖ ≈ 7e-16` |
| 数值地基 | Householder QR | 反射法瘦身 QR `A=QR` | `A=QR` 重构 ≈ ε，`QᵀQ=I` |
| 数值地基 | 对称特征值 | 循环 Jacobi（Numerical Recipes） | `‖Av−λv‖ ≈ 1e-15`，`VᵀV=I` |
| 数值地基 | SVD | 单侧 Jacobi SVD（Golub & Van Loan） | 重构 ≈ ε，`UᵀU=I`，秩-2 矩阵 σ₃≈0 |
| 监督学习 | 线性回归 | 正规方程 + Cholesky | `Xᵀ(y−Xŵ)=0`（≈8e-14） |
| 监督学习 | 逻辑回归 | IRLS / Newton + 交叉熵 | 中心差分**梯度检验** `3.3e-11` |
| 无监督 | PCA | 协方差 Jacobi 特征值 | 协方差对称半正定、`V` 正交、`Σλ=tr` |
| 无监督 | K-Means | k-means++ 初始化 + Lloyd | 惯性单调不增 |
| 无监督 | GMM | EM + logsumexp 稳定对数似然 | 责任度行和=1、权重和=1、**对数似然单调不降** |
| 监督学习 | SVM | Simplified SMO（Platt 1998） | 线性可分数据训练准确率 ≥ 0.97 |
| 监督学习 | 决策树 / kNN / 朴素贝叶斯 | CART-Gini / 惰性近邻 / 高斯 NB | k=1 训练 0 误差、NB 后验归一 |
| 深度学习 | MLP | 反向传播 + Adam/SGD | 中心差分**梯度检验** `2.2e-10` |
| 深度学习 | 多头注意力 | 缩放点积（Vaswani 2017） | softmax 沿键归一 + 反传**梯度检验** `1.3e-10` |
| 深度学习 | Transformer 编码器块 | MHA + 残差 + LayerNorm + FFN | 前向形状与有限性 |

> 数学依据：Golub & Van Loan《Matrix Computations》、Bishop《PRML》§9.2、Goodfellow《Deep Learning》、Vaswani et al. 2017、Platt 1998、Arthur & Vassilvitskii 2007、Breiman 1984。

---

## 快速开始

### 浏览器（零依赖，单文件）
直接用 `<script>` 引入，全局得到 `AIForge`：
```html
<script src="src/ai-forge.js"></script>
<script>
  const net = new AIForge.MLP({ layers:[2,16,3], activation:'relu', output:'softmax' });
  net.fit(X, y, { epochs:120 });
  console.log(AIForge.selftest().allPass); // true
</script>
```
打开 [`index.html`](./index.html) 即为一键自检 + 可交互演示（该文件由源码内联生成，无任何外部依赖）。

### Node
```js
const AI = require('./src/ai-forge.js');

// 线性回归
const lr = new AI.LinearRegression().fit(X, y);
lr.predict([1.2, -0.5]);

// 深度学习分类
const net = new AI.MLP({ layers:[2,16,3], activation:'relu', output:'softmax', optimizer:'adam' });
net.fit(X, y, { epochs:140 });
net.predict([0.3, -1.1]); // → 类别索引

// 运行 17 项不变量自检
const r = AI.selftest();
console.log(r.passed + '/' + r.total, r.allPass ? '全绿' : '有失败');
```

---

## 🧪 一键自检（17/17 全绿）

```bash
node test/run.js
```
实测输出：
```
星核 AI-Forge 自检  v1.0.0   (135 ms)
  ✅ PASS  Cholesky ‖LLᵀ−A‖   [7.24e-16]
  ✅ PASS  Householder QR 重构 + QᵀQ=I   [rec=2.42e-15 orth=1.75e-15]
  ✅ PASS  Jacobi 特征值 ‖Av−λv‖ + VᵀV=I   [resid=1.33e-15 orth=4.07e-16]
  ✅ PASS  Jacobi SVD 重构 + UᵀU=I   [rec=2.83e-15 orth=3.11e-16]
  ✅ PASS  线性回归 Xᵀ残差≈0   [max=7.63e-14]
  ✅ PASS  逻辑回归梯度检验   [maxRelErr=3.29e-11]
  ✅ PASS  PCA 协方差对称半正定 + V正交 + Σλ=tr   [...]
  ✅ PASS  GMM 责任度归一 + 权重和=1   [Σw=1.000000]
  ✅ PASS  EM 对数似然单调不降   [final ll=-590.19]
  ✅ PASS  K-Means 收敛   [inertia=126.344]
  ✅ PASS  kNN k=1 训练集 0 误差   [60/60]
  ✅ PASS  高斯朴素贝叶斯 训练准确率高   [acc=1.000]
  ✅ PASS  CART 决策树 可分数据高准确   [acc=1.000]
  ✅ PASS  SVM-SMO 线性可分高准确   [acc=1.000]
  ✅ PASS  MLP 反向传播梯度检验   [maxRelErr=2.23e-10]
  ✅ PASS  注意力 softmax 归一 + 反传梯度检验   [maxRelErr=1.28e-10]
  ✅ PASS  Transformer 编码器块 前向   [out=4×8]
结果：17/17 通过  ✅ 全绿
```

> **关于「假绿」的工程纪律**：梯度检验若解析梯度与数值梯度任一非有限（NaN/Inf），`gradCheck` 会**显式返回 Infinity** 使检验失败——绝不静默跳过。这是本项目可信度的关键一环（开发中曾借此抓住 MLP softmax 未生效、注意力反传返回 TypedArray 致 `Array.flat()` 不展平等两处会「假绿」的真 bug）。

---

## 项目结构
```
ai-forge/
├── index.html               # 单文件演示页（内联源码，零外部依赖）
├── src/ai-forge.js          # ★ 核心：UMD，浏览器 & Node 双跑，零依赖
├── test/run.js              # 无头自检：node test/run.js
├── tools/build-html.js      # 由 src 生成 index.html（可复现，防漂移）
├── tools/index.template.html
├── docs/ALGORITHMS.md       # 逐模块算法/不变量/论文索引
├── package.json
└── LICENSE                  # MIT
```

## 设计取舍
- **Float64Array 后端**：稠密矩阵用一维 `Float64Array` 行主序存储，JIT 友好、缓存局部性优于 `Array<Array>`。
- **数值稳定性优先**：GMM 用 `logsumexp` + Cholesky 求 `log|Σ|`；softmax/sigmoid 做溢出防护；QR 用反射法保向后稳定。
- **可复现**：内置 mulberry32 种子 PRNG，自检结果确定；`index.html` 由源码构建生成，杜绝手抄漂移。

## License
MIT © 晨星（CJX0712）
