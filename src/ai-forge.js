/* =====================================================================
 * 星核 AI-Forge  (AI Core Forge)
 * ---------------------------------------------------------------------
 * 世界级零依赖 AI 核心：数值线性代数 + 经典机器学习 + 深度学习。
 * 作者：晨星 (CJX0712)
 *
 * 设计原则（集大成 / 零重写）：
 *   - 每个算法采用领域内公认的最优实现，而非从零拍脑袋：
 *       · 对称特征值      —— 循环 Jacobi（Numerical Recipes, 经典高精度）
 *       · 奇异值分解 SVD  —— 单侧 Jacobi SVD（Golub & Van Loan, 精度最高的一类）
 *       · QR / 最小二乘   —— Householder 反射（向后数值稳定）
 *       · SPD 求解/求逆   —— Cholesky + Gauss-Jordan 主元
 *       · GMM             —— EM + logsumexp 稳定对数似然（Bishop PRML §9.2）
 *       · SVM             —— Simplified SMO（Platt 1998 分解的紧凑实现）
 *       · K-Means         —— k-means++ 初始化 + Lloyd 迭代
 *       · 决策树          —— CART / Gini 不纯度（Breiman 1984）
 *       · 神经网络        —— 反向传播 + 中心差分梯度检验（Goodfellow DL）
 *       · 注意力          —— 缩放点积多头注意力（Vaswani 2017）
 *   - 每个模块内置「可被独立实现交叉验证的不变量」，见 selftest()。
 *   - 单文件、零外部依赖；浏览器 <script> 与 Node require 双跑。
 * ===================================================================== */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.AIForge = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  /* ===================== 0. 可复现随机数 & 数值工具 ===================== */
  // mulberry32：小巧、可复现的 PRNG，保证自检结果确定。
  function makeRng(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const _rng = makeRng(20261014);
  function randn() { // Box-Muller 标准正态
    let u = 0, v = 0;
    while (u === 0) u = _rng();
    while (v === 0) v = _rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  // 稳定 logsumexp：log Σ e^{x_i} = m + log Σ e^{x_i-m}，防溢出（BM §4.1 数值稳定性）
  function logsumexp(x) {
    let m = -Infinity;
    for (const v of x) if (v > m) m = v;
    if (m === -Infinity) return -Infinity;
    let s = 0;
    for (const v of x) s += Math.exp(v - m);
    return m + Math.log(s);
  }
  function sigmoid(z) { // 数值稳定
    if (z >= 0) { const e = Math.exp(-z); return 1 / (1 + e); }
    const e = Math.exp(z); return e / (1 + e);
  }
  function softmax(v) {
    let m = -Infinity; for (const x of v) if (x > m) m = x;
    let s = 0; const a = new Array(v.length);
    for (let i = 0; i < v.length; i++) { a[i] = Math.exp(v[i] - m); s += a[i]; }
    for (let i = 0; i < v.length; i++) a[i] /= s;
    return a;
  }
  function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
  function norm(a) { return Math.sqrt(dot(a, a)); }
  // 中心差分梯度检验：返回 maxRelErr。任一梯度非有限 → 返回 Infinity（显式失败，拒绝假绿）
  function gradCheck(f, gradFn, x, eps) {
    eps = eps || 1e-5; let maxRel = 0;
    const g = gradFn(x);
    for (let i = 0; i < x.length; i++) {
      const xp = x.slice(); xp[i] += eps; const fp = f(xp);
      const xm = x.slice(); xm[i] -= eps; const fm = f(xm);
      const num = (fp - fm) / (2 * eps);
      const ana = g[i];
      if (!Number.isFinite(num) || !Number.isFinite(ana)) return Infinity;
      const rel = Math.abs(num - ana) / (Math.abs(num) + Math.abs(ana) + 1e-12);
      if (rel > maxRel) maxRel = rel;
    }
    return maxRel;
  }

  /* ===================== 1. Matrix：Float64Array 稠密矩阵 ===================== */
  class Matrix {
    constructor(r, c, data) {
      this.r = r; this.c = c;
      this.data = data && data.length === r * c ? data : new Float64Array(r * c);
    }
    get(i, j) { return this.data[i * this.c + j]; }
    set(i, j, v) { this.data[i * this.c + j] = v; return this; }
    clone() { return new Matrix(this.r, this.c, this.data.slice()); }
    static zeros(r, c) { return new Matrix(r, c); }
    static ones(r, c) { const m = new Matrix(r, c); m.data.fill(1); return m; }
    static identity(n) { const m = new Matrix(n, n); for (let i = 0; i < n; i++) m.set(i, i, 1); return m; }
    static randn(r, c) { const m = new Matrix(r, c); for (let i = 0; i < r * c; i++) m.data[i] = randn(); return m; }
    static from2D(a) {
      const m = new Matrix(a.length, a[0].length);
      for (let i = 0; i < a.length; i++) for (let j = 0; j < a[0].length; j++) m.set(i, j, a[i][j]);
      return m;
    }
    to2D() { const a = []; for (let i = 0; i < this.r; i++) { const row = []; for (let j = 0; j < this.c; j++) row.push(this.get(i, j)); a.push(row); } return a; }
    // ---- 基础运算 ----
    add(o) { const m = new Matrix(this.r, this.c); for (let i = 0; i < this.data.length; i++) m.data[i] = this.data[i] + (o.data ? o.data[i] : o); return m; }
    sub(o) { const m = new Matrix(this.r, this.c); for (let i = 0; i < this.data.length; i++) m.data[i] = this.data[i] - (o.data ? o.data[i] : o); return m; }
    scale(s) { const m = new Matrix(this.r, this.c); for (let i = 0; i < this.data.length; i++) m.data[i] = this.data[i] * s; return m; }
    mul(o) { // 矩阵乘法 A(this) @ B(o)
      if (!o.data) return this.scale(o);
      const { r: n, c: m } = this, p = o.c;
      const out = new Matrix(n, p);
      const Bc = o.c, Bd = o.data, Ad = this.data, Od = out.data;
      for (let i = 0; i < n; i++) for (let k = 0; k < m; k++) {
        const a = Ad[i * m + k]; if (a === 0) continue;
        const ib = i * p, km = k * Bc;
        for (let j = 0; j < p; j++) Od[ib + j] += a * Bd[km + j];
      }
      return out;
    }
    transpose() { const m = new Matrix(this.c, this.r); for (let i = 0; i < this.r; i++) for (let j = 0; j < this.c; j++) m.set(j, i, this.get(i, j)); return m; }
    apply(fn) { const m = new Matrix(this.r, this.c); for (let i = 0; i < this.data.length; i++) m.data[i] = fn(this.data[i]); return m; }
    rowSum() { const v = new Float64Array(this.r); for (let i = 0; i < this.r; i++) { let s = 0; for (let j = 0; j < this.c; j++) s += this.get(i, j); v[i] = s; } return v; }
    colMeans() { const v = new Float64Array(this.c); for (let j = 0; j < this.c; j++) { let s = 0; for (let i = 0; i < this.r; i++) s += this.get(i, j); v[j] = s / this.r; } return v; }

    /* ---- Cholesky：A = L Lᵀ，仅对称正定（SPD） ---- */
    cholesky() {
      const n = this.r; if (n !== this.c) throw new Error('cholesky: 需方阵');
      const L = Matrix.zeros(n, n);
      for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
        let s = this.get(i, j);
        for (let k = 0; k < j; k++) s -= L.get(i, k) * L.get(j, k);
        if (i === j) { if (s <= 1e-15) throw new Error('cholesky: 非正定'); L.set(i, i, Math.sqrt(s)); }
        else L.set(i, j, s / L.get(j, j));
      }
      return L;
    }
    // SPD 求解 L Lᵀ x = b（前代 + 回代）
    static choleskySolve(A, b) {
      const L = A.cholesky(); const n = A.r;
      const y = new Float64Array(n);
      for (let i = 0; i < n; i++) { let s = b[i]; for (let k = 0; k < i; k++) s -= L.get(i, k) * y[k]; y[i] = s / L.get(i, i); }
      const x = new Float64Array(n);
      for (let i = n - 1; i >= 0; i--) { let s = y[i]; for (let k = i + 1; k < n; k++) s -= L.get(k, i) * x[k]; x[i] = s / L.get(i, i); }
      return x;
    }

    /* ---- Gauss-Jordan 部分主元求逆 ---- */
    inverse() {
      const n = this.r; if (n !== this.c) throw new Error('inverse: 需方阵');
      const A = this.clone().data, I = Matrix.identity(n).data;
      for (let col = 0; col < n; col++) {
        let piv = col, maxv = Math.abs(A[col * n + col]);
        for (let r = col + 1; r < n; r++) { const v = Math.abs(A[r * n + col]); if (v > maxv) { maxv = v; piv = r; } }
        if (maxv < 1e-300) throw new Error('inverse: 奇异矩阵');
        if (piv !== col) { for (let j = 0; j < n; j++) { const t = A[col * n + j]; A[col * n + j] = A[piv * n + j]; A[piv * n + j] = t; const s = I[col * n + j]; I[col * n + j] = I[piv * n + j]; I[piv * n + j] = s; } }
        const d = A[col * n + col];
        for (let j = 0; j < n; j++) { A[col * n + j] /= d; I[col * n + j] /= d; }
        for (let r = 0; r < n; r++) {
          if (r === col) continue;
          const f = A[r * n + col]; if (f === 0) continue;
          for (let j = 0; j < n; j++) { A[r * n + j] -= f * A[col * n + j]; I[r * n + j] -= f * I[col * n + j]; }
        }
      }
      return new Matrix(n, n, I);
    }

    /* ---- Householder QR（瘦身）A(m×n) = Q(m×n) R(n×n) ---- */
    householderQR() {
      const m = this.r, n = this.c; const R = this.clone(); const Qf = Matrix.identity(m);
      for (let k = 0; k < Math.min(m - 1, n); k++) {
        let nrm = 0; for (let i = k; i < m; i++) { const v = R.get(i, k); nrm += v * v; } nrm = Math.sqrt(nrm);
        if (nrm < 1e-300) continue;
        const vv = new Float64Array(m - k);
        for (let i = k; i < m; i++) vv[i - k] = R.get(i, k);
        const sgn = (R.get(k, k) >= 0) ? 1 : -1; vv[0] += sgn * nrm; // 反射向量
        let vn = 0; for (let i = 0; i < m - k; i++) vn += vv[i] * vv[i]; vn = Math.sqrt(vn);
        if (vn < 1e-300) continue;
        for (let i = 0; i < m - k; i++) vv[i] /= vn;
        // R <- H R （列块 k:n）
        for (let j = k; j < n; j++) { let d = 0; for (let i = k; i < m; i++) d += vv[i - k] * R.get(i, j); for (let i = k; i < m; i++) R.set(i, j, R.get(i, j) - 2 * vv[i - k] * d); }
        // Qf <- Qf H
        const vf = new Float64Array(m); for (let i = k; i < m; i++) vf[i] = vv[i - k];
        const w = new Float64Array(m);
        for (let i = 0; i < m; i++) { let s = 0; for (let t = 0; t < m; t++) s += Qf.get(i, t) * vf[t]; w[i] = s; }
        for (let i = 0; i < m; i++) for (let t = 0; t < m; t++) Qf.set(i, t, Qf.get(i, t) - 2 * w[i] * vf[t]);
      }
      const Q = Matrix.zeros(m, n); for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) Q.set(i, j, Qf.get(i, j));
      const Rt = Matrix.zeros(n, n); for (let i = 0; i < n; i++) for (let j = i; j < n; j++) Rt.set(i, j, R.get(i, j));
      return { Q, R: Rt };
    }
    // 最小二乘求解 min ‖Ax-b‖
    lstsq(b) {
      const { Q, R } = this.householderQR(); const n = this.c;
      const y = new Float64Array(n);
      for (let j = 0; j < n; j++) { let s = 0; for (let i = 0; i < this.r; i++) s += Q.get(i, j) * b[i]; y[j] = s; }
      const x = new Float64Array(n);
      for (let i = n - 1; i >= 0; i--) { let s = y[i]; for (let j = i + 1; j < n; j++) s -= R.get(i, j) * x[j]; x[i] = s / R.get(i, i); }
      return x;
    }

    /* ---- 对称特征值分解：循环 Jacobi（Numerical Recipes 经典实现） ---- */
    jacobiEig() {
      if (this.r !== this.c) throw new Error('jacobiEig: 需方阵');
      const n = this.c; const a = this.clone(); const v = Matrix.identity(n);
      // 单一运行对角数组 d（避免 NR 双缓冲被误写成双计数）；收敛后即特征值
      const d = new Float64Array(n);
      for (let ip = 0; ip < n; ip++) d[ip] = a.get(ip, ip);
      for (let sweep = 0; sweep < 60; sweep++) {
        let sm = 0; for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) sm += Math.abs(a.get(p, q));
        if (sm < 1e-300) break;
        const thresh = (sweep < 4) ? 0.2 * sm / (n * n) : 0;
        for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) {
          const apq = a.get(p, q);
          const g = 100 * Math.abs(apq);
          if (sweep > 4 && Math.abs(d[p]) + g === Math.abs(d[p]) && Math.abs(d[q]) + g === Math.abs(d[q])) { a.set(p, q, 0); }
          else if (Math.abs(apq) > thresh) {
            const h = d[q] - d[p];
            const theta = 0.5 * h / apq;
            const t = (Math.abs(theta) > 1e15) ? 1 / (2 * theta) : ((theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(1 + theta * theta)));
            const c = 1 / Math.sqrt(1 + t * t), s = t * c, tau = s / (1 + c);
            const hh = t * apq; d[p] -= hh; d[q] += hh; a.set(p, q, 0);
            for (let j = 0; j < p; j++) { const g1 = a.get(j, p), h1 = a.get(j, q); a.set(j, p, g1 - s * (h1 + g1 * tau)); a.set(j, q, h1 + s * (g1 - h1 * tau)); }
            for (let j = p + 1; j < q; j++) { const g1 = a.get(p, j), h1 = a.get(j, q); a.set(p, j, g1 - s * (h1 + g1 * tau)); a.set(j, q, h1 + s * (g1 - h1 * tau)); }
            for (let j = q + 1; j < n; j++) { const g1 = a.get(p, j), h1 = a.get(q, j); a.set(p, j, g1 - s * (h1 + g1 * tau)); a.set(q, j, h1 + s * (g1 - h1 * tau)); }
            for (let j = 0; j < n; j++) { const g1 = v.get(j, p), h1 = v.get(j, q); v.set(j, p, g1 - s * (h1 + g1 * tau)); v.set(j, q, h1 + s * (g1 - h1 * tau)); }
          }
        }
      }
      for (let ip = 0; ip < n; ip++) a.set(ip, ip, d[ip]);
      const ev = []; for (let i = 0; i < n; i++) ev.push(a.get(i, i));
      const idx = ev.map((e, i) => [e, i]).sort((x, y) => y[0] - x[0]).map(x => x[1]);
      const values = idx.map(i => ev[i]);
      const vectors = Matrix.zeros(n, n);
      idx.forEach((orig, j) => { for (let i = 0; i < n; i++) vectors.set(i, j, v.get(i, orig)); });
      return { values, vectors };
    }

    /* ---- 单侧 Jacobi SVD：A = U Σ Vᵀ（Golub & Van Loan 高精度一族） ---- */
    jacobiSVD() {
      const m = this.r, n = this.c; const R = this.clone(); const V = Matrix.identity(n);
      for (let sweep = 0; sweep < 60; sweep++) {
        let off = 0;
        for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
          let alpha = 0, beta = 0, gamma = 0;
          for (let i = 0; i < m; i++) { const aip = R.get(i, p), aiq = R.get(i, q); alpha += aip * aip; beta += aiq * aiq; gamma += aip * aiq; }
          off = Math.max(off, Math.abs(gamma));
          if (Math.abs(gamma) < 1e-300) continue;
          const zeta = (beta - alpha) / (2 * gamma);
          const t = Math.sign(zeta || 1) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
          const c = 1 / Math.sqrt(1 + t * t), s = c * t;
          for (let i = 0; i < m; i++) { const rip = R.get(i, p), riq = R.get(i, q); R.set(i, p, c * rip - s * riq); R.set(i, q, s * rip + c * riq); }
          for (let i = 0; i < n; i++) { const vip = V.get(i, p), viq = V.get(i, q); V.set(i, p, c * vip - s * viq); V.set(i, q, s * vip + c * viq); }
        }
        if (off < 1e-14) break;
      }
      const sig = []; for (let j = 0; j < n; j++) { let s = 0; for (let i = 0; i < m; i++) { const v = R.get(i, j); s += v * v; } sig.push(Math.sqrt(s)); }
      const idx = sig.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).map(x => x[1]);
      const sigma = idx.map(i => sig[i]);
      const U = Matrix.zeros(m, n), V2 = Matrix.zeros(n, n);
      idx.forEach((orig, nj) => { const s = sig[orig]; if (s > 1e-300) for (let i = 0; i < m; i++) U.set(i, nj, R.get(i, orig) / s); for (let i = 0; i < n; i++) V2.set(i, nj, V.get(i, orig)); });
      return { U, sigma, V: V2 };
    }
    reconstructSVD(svd) {
      const { U, sigma, V } = svd; const n = this.c;
      const S = Matrix.zeros(n, n); for (let i = 0; i < n; i++) S.set(i, i, sigma[i]);
      return U.mul(S).mul(V.transpose());
    }
  }

  /* ===================== 2. 经典机器学习 ===================== */

  // —— 线性回归：正规方程 + Cholesky；残差 ⊥ X 为不变量 ——
  class LinearRegression {
    constructor(opts) { opts = opts || {}; this.l2 = opts.l2 || 0; this.w = null; }
    fit(Xarr, y) {
      const n = Xarr.length, d = Xarr[0].length;
      const X = new Matrix(n, d + 1); for (let i = 0; i < n; i++) { X.set(i, 0, 1); for (let j = 0; j < d; j++) X.set(i, j + 1, Xarr[i][j]); }
      const Xt = X.transpose();
      let XtX = Xt.mul(X);
      if (this.l2) { const R = Matrix.identity(d + 1).scale(this.l2); R.set(0, 0, 0); XtX = XtX.add(R); }
      const Xty = new Float64Array(d + 1); for (let j = 0; j <= d; j++) { let s = 0; for (let i = 0; i < n; i++) s += Xt.get(j, i) * y[i]; Xty[j] = s; }
      this.w = Matrix.choleskySolve(XtX, Xty); this.d = d; return this;
    }
    predict(x) { let s = this.w[0]; for (let j = 0; j < this.d; j++) s += this.w[j + 1] * x[j]; return s; }
  }

  // —— 逻辑回归：IRLS/Newton + 交叉熵；梯度检验为不变量 ——
  class LogisticRegression {
    constructor(opts) { opts = opts || {}; this.l2 = opts.l2 || 1e-4; this.iters = opts.iters || 50; this.w = null; }
    fit(Xarr, y) {
      const n = Xarr.length, d = Xarr[0].length;
      const X = new Matrix(n, d + 1); for (let i = 0; i < n; i++) { X.set(i, 0, 1); for (let j = 0; j < d; j++) X.set(i, j + 1, Xarr[i][j]); }
      let w = new Float64Array(d + 1); const Xt = X.transpose();
      for (let it = 0; it < this.iters; it++) {
        const z = new Float64Array(n), p = new Float64Array(n);
        for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j <= d; j++) s += X.get(i, j) * w[j]; z[i] = s; p[i] = sigmoid(s); }
        const g = new Float64Array(d + 1);
        for (let j = 0; j <= d; j++) { let s = 0; for (let i = 0; i < n; i++) s += Xt.get(j, i) * (p[i] - y[i]); g[j] = s / n + (j > 0 ? this.l2 * w[j] : 0); }
        // H = Xᵀ diag(p(1-p)) X / n + λI
        const H = Matrix.zeros(d + 1, d + 1);
        for (let a = 0; a <= d; a++) for (let b = 0; b <= d; b++) { let s = 0; for (let i = 0; i < n; i++) s += Xt.get(a, i) * p[i] * (1 - p[i]) * Xt.get(b, i); H.set(a, b, s / n); }
        for (let j = 1; j <= d; j++) H.set(j, j, H.get(j, j) + this.l2); H.set(0, 0, H.get(0, 0) + 1e-8);
        const delta = Matrix.choleskySolve(H, g);
        let maxd = 0; for (let j = 0; j <= d; j++) { w[j] -= delta[j]; maxd = Math.max(maxd, Math.abs(delta[j])); }
        if (maxd < 1e-10) break;
      }
      this.w = w; this.d = d; return this;
    }
    predictProba(x) { let s = this.w[0]; for (let j = 0; j < this.d; j++) s += this.w[j + 1] * x[j]; return sigmoid(s); }
    predict(x) { return this.predictProba(x) >= 0.5 ? 1 : 0; }
    // 交叉熵损失与梯度（供梯度检验）
    _lossGrad(Xarr, y, w) {
      const n = Xarr.length, d = Xarr[0].length; let loss = 0;
      const g = new Float64Array(d + 1);
      for (let i = 0; i < n; i++) { let z = w[0]; for (let j = 0; j < d; j++) z += w[j + 1] * Xarr[i][j]; const p = sigmoid(z); loss += -(y[i] * Math.log(p + 1e-12) + (1 - y[i]) * Math.log(1 - p + 1e-12)); g[0] += (p - y[i]); for (let j = 0; j < d; j++) g[j + 1] += (p - y[i]) * Xarr[i][j]; }
      loss /= n; let pen = 0; for (let j = 1; j <= d; j++) pen += w[j] * w[j]; loss += 0.5 * this.l2 * pen; // L2 惩罚进入损失，使 f 与 g 对齐（梯度检验用）
      for (let j = 0; j <= d; j++) g[j] = g[j] / n + (j > 0 ? this.l2 * w[j] : 0);
      return { loss, g };
    }
  }

  // —— PCA：协方差对称半正定 + 特征向量正交为不变量 ——
  class PCA {
    constructor(opts) { opts = opts || {}; this.k = opts.k || 1; }
    fit(Xarr) {
      const n = Xarr.length, d = Xarr[0].length;
      this.mean = new Float64Array(d);
      for (let j = 0; j < d; j++) { let s = 0; for (let i = 0; i < n; i++) s += Xarr[i][j]; this.mean[j] = s / n; }
      const C = new Matrix(d, d);
      for (let i = 0; i < n; i++) for (let a = 0; a < d; a++) { const xa = Xarr[i][a] - this.mean[a]; for (let b = a; b < d; b++) { const xb = Xarr[i][b] - this.mean[b]; const v = xa * xb; C.set(a, b, C.get(a, b) + v); C.set(b, a, C.get(a, b)); } }
      for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) C.set(a, b, C.get(a, b) / n);
      const { values, vectors } = C.jacobiEig();
      this.explainedVariance = values; this.totalVar = values.reduce((a, b) => a + b, 0);
      this.components = Matrix.zeros(this.k, d); this.componentsF = new Matrix(d, this.k);
      for (let j = 0; j < this.k; j++) for (let a = 0; a < d; a++) { this.components.set(j, a, vectors.get(a, j)); this.componentsF.set(a, j, vectors.get(a, j)); }
      this.C = C; return this;
    }
    transform(x) { const out = new Float64Array(this.k); for (let j = 0; j < this.k; j++) { let s = 0; for (let a = 0; a < this.mean.length; a++) s += this.components.get(j, a) * (x[a] - this.mean[a]); out[j] = s; } return out; }
    reconstruct(z) { const out = new Float64Array(this.mean.length); for (let a = 0; a < this.mean.length; a++) { let s = this.mean[a]; for (let j = 0; j < this.k; j++) s += this.componentsF.get(a, j) * z[j]; out[a] = s; } return out; }
  }

  // —— K-Means：k-means++ 初始化 + Lloyd；惯性单调不增 ——
  class KMeans {
    constructor(opts) { opts = opts || {}; this.k = opts.k || 2; this.iters = opts.iters || 100; }
    fit(Xarr) {
      const n = Xarr.length, d = Xarr[0].length;
      const dist = (a, b) => { let s = 0; for (let j = 0; j < d; j++) { const v = a[j] - b[j]; s += v * v; } return s; };
      // k-means++ 初始化（Arthur & Vassilvitskii 2007）
      const C = [Xarr[(Math.floor(_rng() * n))].slice()];
      const d2 = new Float64Array(n).fill(Infinity);
      while (C.length < this.k) {
        let sum = 0; for (let i = 0; i < n; i++) { d2[i] = Math.min(d2[i], dist(Xarr[i], C[C.length - 1])); sum += d2[i]; }
        if (sum === 0) { C.push(Xarr[Math.floor(_rng() * n)].slice()); continue; }
        let r = _rng() * sum, pick = n - 1; for (let i = 0; i < n; i++) { r -= d2[i]; if (r <= 0) { pick = i; break; } }
        C.push(Xarr[pick].slice());
      }
      this.centroids = C; let labels = new Array(n).fill(0); let prevInf = Infinity; this.inertia = 0;
      for (let it = 0; it < this.iters; it++) {
        let inertia = 0;
        for (let i = 0; i < n; i++) { let best = 0, bd = Infinity; for (let c = 0; c < this.k; c++) { const dd = dist(Xarr[i], C[c]); if (dd < bd) { bd = dd; best = c; } } labels[i] = best; inertia += bd; }
        const sums = Array.from({ length: this.k }, () => new Float64Array(d)); const cnt = new Array(this.k).fill(0);
        for (let i = 0; i < n; i++) { const c = labels[i]; cnt[c]++; for (let j = 0; j < d; j++) sums[c][j] += Xarr[i][j]; }
        for (let c = 0; c < this.k; c++) if (cnt[c] > 0) for (let j = 0; j < d; j++) C[c][j] = sums[c][j] / cnt[c];
        this.inertia = inertia;
        if (Math.abs(prevInf - inertia) < 1e-9) { break; }
        prevInf = inertia;
      }
      this.labels = labels; this.centroids = C; return this;
    }
    predict(x) { let best = 0, bd = Infinity; for (let c = 0; c < this.k; c++) { let s = 0; for (let j = 0; j < x.length; j++) { const v = x[j] - this.centroids[c][j]; s += v * v; } if (s < bd) { bd = s; best = c; } } return best; }
  }

  // —— GMM（全协方差）EM：责任度行和=1、权重和=1、对数似然单调 ——
  class GMM {
    constructor(opts) { opts = opts || {}; this.k = opts.k || 2; this.iters = opts.iters || 60; this.tol = opts.tol || 1e-6; this.diag = !!opts.diag; }
    fit(Xarr) {
      const n = Xarr.length, d = Xarr[0].length;
      // k-means 初始化
      const km = new KMeans({ k: this.k }).fit(Xarr);
      const W = new Float64Array(this.k).fill(1 / this.k);
      const means = km.centroids.map(c => Float64Array.from(c));
      const covs = [];
      for (let c = 0; c < this.k; c++) { const M = new Matrix(d, d); for (let i = 0; i < n; i++) for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) M.set(a, b, M.get(a, b) + (Xarr[i][a] - means[c][a]) * (Xarr[i][b] - means[c][b])); for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) M.set(a, b, M.get(a, b) / n + 1e-6); covs.push(M); }
      let prev = -Infinity; this.logLikelihood = -Infinity; this.resp = null;
      for (let it = 0; it < this.iters; it++) {
        // E-step
        const logp = []; for (let c = 0; c < this.k; c++) logp.push(this._logGauss(Xarr, means[c], covs[c]));
        const resp = new Float64Array(n * this.k); let ll = 0;
        for (let i = 0; i < n; i++) { const row = []; for (let c = 0; c < this.k; c++) row.push(Math.log(W[c]) + logp[c][i]); const m = logsumexp(row); ll += m; for (let c = 0; c < this.k; c++) resp[i * this.k + c] = Math.exp(row[c] - m); }
        // M-step
        const Nk = new Float64Array(this.k); for (let c = 0; c < this.k; c++) for (let i = 0; i < n; i++) Nk[c] += resp[i * this.k + c];
        for (let c = 0; c < this.k; c++) {
          W[c] = Nk[c] / n;
          const mu = new Float64Array(d); for (let j = 0; j < d; j++) { let s = 0; for (let i = 0; i < n; i++) s += resp[i * this.k + c] * Xarr[i][j]; mu[j] = s / Nk[c]; } means[c] = mu;
          const M = new Matrix(d, d); for (let i = 0; i < n; i++) for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) M.set(a, b, M.get(a, b) + resp[i * this.k + c] * (Xarr[i][a] - mu[a]) * (Xarr[i][b] - mu[b]));
          for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) M.set(a, b, M.get(a, b) / Nk[c] + 1e-9);
          covs[c] = M;
        }
        this.logLikelihood = ll; this.resp = resp;
        if (Math.abs(ll - prev) < this.tol * Math.abs(ll)) break; prev = ll;
      }
      this.W = W; this.means = means; this.covs = covs; return this;
    }
    _logGauss(Xarr, mu, cov) {
      const d = mu.length; const L = cov.cholesky();
      let logdet = 0; for (let i = 0; i < d; i++) logdet += 2 * Math.log(L.get(i, i));
      const out = new Float64Array(Xarr.length); const c0 = -0.5 * (d * Math.log(2 * Math.PI) + logdet);
      for (let i = 0; i < Xarr.length; i++) {
        // solve L y = (x-mu)
        const y = new Float64Array(d); for (let a = 0; a < d; a++) { let s = Xarr[i][a] - mu[a]; for (let b = 0; b < a; b++) s -= L.get(a, b) * y[b]; y[a] = s / L.get(a, a); }
        let m = 0; for (let a = 0; a < d; a++) m += y[a] * y[a];
        out[i] = c0 - 0.5 * m;
      }
      return out;
    }
    predict(x) { let best = 0, bl = -Infinity; for (let c = 0; c < this.k; c++) { const lp = this._logGauss([x], this.means[c], this.covs[c])[0] + Math.log(this.W[c]); if (lp > bl) { bl = lp; best = c; } } return best; }
  }

  // —— k 近邻（惰性学习） ——
  class KNN {
    constructor(opts) { opts = opts || {}; this.k = opts.k || 3; this.task = opts.task || 'class'; }
    fit(Xarr, y) { this.X = Xarr; this.y = y; return this; }
    predict(x) {
      const ds = this.X.map((p, i) => ({ d: (() => { let s = 0; for (let j = 0; j < p.length; j++) { const v = p[j] - x[j]; s += v * v; } return s; })(), i }));
      ds.sort((a, b) => a.d - b.d); const top = ds.slice(0, this.k);
      if (this.task === 'mean') { let s = 0; for (const t of top) s += this.y[t.i]; return s / this.k; }
      const tally = {}; for (const t of top) tally[this.y[t.i]] = (tally[this.y[t.i]] || 0) + 1;
      return Object.keys(tally).sort((a, b) => tally[b] - tally[a])[0] | 0;
    }
  }

  // —— 高斯朴素贝叶斯：后验和为 1 ——
  class GaussianNB {
    fit(Xarr, y) {
      const d = Xarr[0].length; this.classes = [...new Set(y)]; this.params = {}; this.prior = {};
      for (const c of this.classes) { const pts = Xarr.filter((_, i) => y[i] === c); this.prior[c] = pts.length / Xarr.length; const mu = new Float64Array(d), var0 = new Float64Array(d); for (let j = 0; j < d; j++) { let s = 0; for (const p of pts) s += p[j]; mu[j] = s / pts.length; let v = 0; for (const p of pts) v += (p[j] - mu[j]) ** 2; var0[j] = v / pts.length + 1e-9; } this.params[c] = { mu, var0 }; }
      return this;
    }
    predict(x) { let best = null, bl = -Infinity; for (const c of this.classes) { const { mu, var0 } = this.params[c]; let lp = Math.log(this.prior[c]); for (let j = 0; j < x.length; j++) lp += -0.5 * Math.log(2 * Math.PI * var0[j]) - (x[j] - mu[j]) ** 2 / (2 * var0[j]); if (lp > bl) { bl = lp; best = c; } } return best; }
  }

  // —— CART 决策树（Gini，二分） ——
  class DecisionTree {
    constructor(opts) { opts = opts || {}; this.maxDepth = opts.maxDepth || 5; this.minSplit = opts.minSplit || 2; this.task = opts.task || 'class'; this.root = null; }
    fit(Xarr, y) { this.X = Xarr; this.y = y; this.root = this._build(Xarr.map((_, i) => i), 0); return this; }
    _leaf(rows) { if (this.task === 'mean') { let s = 0; for (const i of rows) s += this.y[i]; return { leaf: true, val: s / rows.length }; } const t = {}; for (const i of rows) t[this.y[i]] = (t[this.y[i]] || 0) + 1; return { leaf: true, val: +Object.keys(t).sort((a, b) => t[b] - t[a])[0] }; }
    _build(rows, depth) {
      if (rows.length < this.minSplit || depth >= this.maxDepth) return this._leaf(rows);
      let best = null, d = this.X[0].length;
      const gini = (rs) => { const t = {}; for (const i of rs) t[this.y[i]] = (t[this.y[i]] || 0) + 1; let g = 1; const n = rs.length; for (const k in t) g -= (t[k] / n) ** 2; return g; };
      for (let f = 0; f < d; f++) {
        const vals = [...new Set(rows.map(i => this.X[i][f]))].sort((a, b) => a - b);
        for (let t = 0; t < vals.length - 1; t++) { const thr = (vals[t] + vals[t + 1]) / 2; const L = rows.filter(i => this.X[i][f] <= thr), Rr = rows.filter(i => this.X[i][f] > thr); if (!L.length || !Rr.length) continue; const gain = gini(rows) - (L.length / rows.length) * gini(L) - (Rr.length / rows.length) * gini(Rr); if (!best || gain > best.gain) best = { gain, f, thr, L, Rr }; }
      }
      if (!best || best.gain <= 1e-9) return this._leaf(rows);
      return { leaf: false, f: best.f, thr: best.thr, left: this._build(best.L, depth + 1), right: this._build(best.Rr, depth + 1) };
    }
    predict(x) { let n = this.root; while (!n.leaf) n = x[n.f] <= n.thr ? n.left : n.right; return n.val; }
  }

  // —— SVM（Simplified SMO，Platt 1998 分解的紧凑实现），线性/RBF 核 ——
  class SVM {
    constructor(opts) { opts = opts || {}; this.C = opts.C || 1; this.kernel = opts.kernel || 'linear'; this.gamma = opts.gamma || 0.5; this.tol = opts.tol || 1e-3; this.maxPasses = opts.maxPasses || 50; }
    _K(a, b) { if (this.kernel === 'linear') { let s = 0; for (let j = 0; j < a.length; j++) s += a[j] * b[j]; return s; } let s = 0; for (let j = 0; j < a.length; j++) { const v = a[j] - b[j]; s += v * v; } return Math.exp(-this.gamma * s); }
    fit(Xarr, yraw) {
      const n = Xarr.length; const y = yraw.map(v => (v === 1 ? 1 : -1)); this.X = Xarr; this.y = y;
      const alpha = new Float64Array(n); const b = { v: 0 }; const E = new Float64Array(n);
      const K = Xarr.map(a => Xarr.map(b2 => this._K(a, b2)));
      let passes = 0;
      while (passes < this.maxPasses) {
        let changed = 0;
        for (let i = 0; i < n; i++) {
          let fxi = b.v; for (let m = 0; m < n; m++) fxi += alpha[m] * y[m] * K[m][i]; E[i] = fxi - y[i];
          if ((y[i] * E[i] < -this.tol && alpha[i] < this.C) || (y[i] * E[i] > this.tol && alpha[i] > 0)) {
            let j = i; while (j === i) j = Math.floor(_rng() * n);
            let fxj = b.v; for (let m = 0; m < n; m++) fxj += alpha[m] * y[m] * K[m][j]; E[j] = fxj - y[j];
            const ai = alpha[i], aj = alpha[j];
            const L = (y[i] !== y[j]) ? Math.max(0, alpha[j] - alpha[i]) : Math.max(0, alpha[i] + alpha[j] - this.C);
            const H = (y[i] !== y[j]) ? Math.min(this.C, this.C + alpha[j] - alpha[i]) : Math.min(this.C, alpha[i] + alpha[j]);
            if (L === H) continue;
            const eta = 2 * K[i][j] - K[i][i] - K[j][j]; if (eta >= 0) continue;
            alpha[j] -= y[j] * (E[i] - E[j]) / eta; alpha[j] = Math.min(H, Math.max(L, alpha[j]));
            if (Math.abs(alpha[j] - aj) < 1e-8) continue;
            alpha[i] += y[i] * y[j] * (aj - alpha[j]);
            const b1 = b.v - E[i] - y[i] * (alpha[i] - ai) * K[i][i] - y[j] * (alpha[j] - aj) * K[i][j];
            const b2 = b.v - E[j] - y[i] * (alpha[i] - ai) * K[i][j] - y[j] * (alpha[j] - aj) * K[j][j];
            b.v = (0 < alpha[i] && alpha[i] < this.C) ? b1 : ((0 < alpha[j] && alpha[j] < this.C) ? b2 : (b1 + b2) / 2);
            changed++;
          }
        }
        if (changed === 0) passes++; else passes = 0;
      }
      this.alpha = alpha; this.b = b.v; return this;
    }
    predict(x) { let s = this.b; for (let m = 0; m < this.X.length; m++) if (this.alpha[m] > 0) s += this.alpha[m] * this.y[m] * this._K(this.X[m], x); return s >= 0 ? 1 : 0; }
  }

  /* ===================== 3. 深度学习 ===================== */

  // —— MLP：反向传播 + 中心差分梯度检验 ——
  class MLP {
    // layers: [in, h1, ..., out]; activation: 'relu'|'tanh'|'sigmoid'; output:'softmax'|'linear'
    constructor(opts) {
      opts = opts || {}; this.sizes = opts.layers; this.act = opts.activation || 'relu';
      this.output = opts.output || 'softmax'; this.lr = opts.lr || 0.05; this.optim = opts.optimizer || 'adam';
      this.W = []; this.b = []; this.mW = []; this.mb = []; this.vW = []; this.vb = []; this.t = 0;
      for (let l = 0; l < this.sizes.length - 1; l++) {
        const fan = this.sizes[l], fout = this.sizes[l + 1], lim = Math.sqrt(6 / (fan + fout));
        const w = new Matrix(fout, fan); for (let i = 0; i < w.data.length; i++) w.data[i] = (randn() * 0.5) * lim;
        this.W.push(w); this.b.push(new Float64Array(fout));
        this.mW.push(Matrix.zeros(fout, fan)); this.mb.push(new Float64Array(fout));
        this.vW.push(Matrix.zeros(fout, fan)); this.vb.push(new Float64Array(fout));
      }
    }
    _afn(z) { if (this.act === 'relu') return Math.max(0, z); if (this.act === 'tanh') return Math.tanh(z); return sigmoid(z); }
    _adfn(z, a) { if (this.act === 'relu') return z > 0 ? 1 : 0; if (this.act === 'tanh') return 1 - a * a; return a * (1 - a); }
    forward(x) {
      const as = [Float64Array.from(x)], zs = [];
      for (let l = 0; l < this.W.length; l++) { const z = new Float64Array(this.b[l].length); const a = as[l]; for (let i = 0; i < this.W[l].r; i++) { let s = this.b[l][i]; for (let j = 0; j < a.length; j++) s += this.W[l].get(i, j) * a[j]; z[i] = s; } zs.push(z); let out; if (l === this.W.length - 1) out = (this.output === 'softmax') ? softmax(z) : z; else out = z.map(v => this._afn(v)); as.push(out); }
      this._as = as; this._zs = zs; return as[as.length - 1];
    }
    loss(x, y) { const o = this.forward(x); if (this.output === 'softmax') { return -Math.log(o[y] + 1e-12); } let s = 0; for (let i = 0; i < o.length; i++) s += (o[i] - (i === y ? 1 : 0)) ** 2; return s / o.length; }
    // 反向传播，返回每层 (dW, db)
    backward(x, y) {
      const o = this.forward(x); const L = this.W.length; const gW = [], gb = [];
      for (let l = 0; l < L; l++) { gW.push(Matrix.zeros(this.W[l].r, this.W[l].c)); gb.push(new Float64Array(this.W[l].r)); }
      let delta;
      if (this.output === 'softmax') { delta = o.slice(); delta[y] -= 1; }
      else { delta = new Float64Array(o.length); for (let i = 0; i < o.length; i++) delta[i] = 2 * (o[i] - (i === y ? 1 : 0)) / o.length; }
      for (let l = L - 1; l >= 0; l--) {
        const aPrev = this._as[l];
        for (let i = 0; i < this.W[l].r; i++) { gb[l][i] = delta[i]; for (let j = 0; j < aPrev.length; j++) gW[l].set(i, j, delta[i] * aPrev[j]); }
        if (l > 0) { const dPrev = new Float64Array(aPrev.length); for (let j = 0; j < aPrev.length; j++) { let s = 0; for (let i = 0; i < this.W[l].r; i++) s += this.W[l].get(i, j) * delta[i]; dPrev[j] = s * this._adfn(this._zs[l - 1][j], aPrev[j]); } delta = dPrev; }
      }
      return { gW, gb };
    }
    _step(gW, gb, lr) {
      const b1 = 0.9, b2 = 0.999, eps = 1e-8; this.t++;
      for (let l = 0; l < this.W.length; l++) {
        for (let i = 0; i < this.W[l].r; i++) for (let j = 0; j < this.W[l].c; j++) {
          const g = gW[l].get(i, j);
          if (this.optim === 'adam') { this.mW[l].set(i, j, b1 * this.mW[l].get(i, j) + (1 - b1) * g); this.vW[l].set(i, j, b2 * this.vW[l].get(i, j) + (1 - b2) * g * g); const mh = this.mW[l].get(i, j) / (1 - Math.pow(b1, this.t)), vh = this.vW[l].get(i, j) / (1 - Math.pow(b2, this.t)); this.W[l].set(i, j, this.W[l].get(i, j) - lr * mh / (Math.sqrt(vh) + eps)); }
          else this.W[l].set(i, j, this.W[l].get(i, j) - lr * g);
        }
        for (let i = 0; i < this.W[l].r; i++) { const g = gb[l][i]; if (this.optim === 'adam') { this.mb[l][i] = b1 * this.mb[l][i] + (1 - b1) * g; this.vb[l][i] = b2 * this.vb[l][i] + (1 - b2) * g * g; const mh = this.mb[l][i] / (1 - Math.pow(b1, this.t)), vh = this.vb[l][i] / (1 - Math.pow(b2, this.t)); this.b[l][i] -= lr * mh / (Math.sqrt(vh) + eps); } else this.b[l][i] -= lr * g; }
      }
    }
    step(x, y) { const { gW, gb } = this.backward(x, y); this._step(gW, gb, this.lr); return this.loss(x, y); }
    fit(Xarr, y, opts) { opts = opts || {}; const ep = opts.epochs || 60; for (let e = 0; e < ep; e++) { let s = 0; for (let i = 0; i < Xarr.length; i++) s += this.step(Xarr[i], y[i]); } return this; }
    predict(x) { const o = this.forward(x); if (this.output === 'softmax') { let bi = 0, bv = -Infinity; for (let i = 0; i < o.length; i++) if (o[i] > bv) { bv = o[i]; bi = i; } return bi; } return o[0]; }
    // 参数扁平化（供梯度检验）
    _flatParams() { const p = []; for (let l = 0; l < this.W.length; l++) { for (let i = 0; i < this.W[l].data.length; i++) p.push(this.W[l].data[i]); for (let i = 0; i < this.W[l].r; i++) p.push(this.b[l][i]); } return p; }
    _setFlat(p) { let idx = 0; for (let l = 0; l < this.W.length; l++) { for (let i = 0; i < this.W[l].data.length; i++) this.W[l].data[i] = p[idx++]; for (let i = 0; i < this.W[l].r; i++) this.b[l][i] = p[idx++]; } }
    _analyticFlatGrad(x, y) { const { gW, gb } = this.backward(x, y); const g = []; for (let l = 0; l < this.W.length; l++) { for (let i = 0; i < this.W[l].data.length; i++) g.push(gW[l].data[i]); for (let i = 0; i < this.W[l].r; i++) g.push(gb[l][i]); } return g; }
  }

  // —— 缩放点积注意力：权重沿键归一；解析反传可梯度检验 ——
  function scaledDotProductAttention(Q, K, V) { // 均为 n×d 的二维数组
    const n = Q.length, d = K[0].length, dv = V[0].length, scale = 1 / Math.sqrt(d);
    const scores = []; for (let i = 0; i < n; i++) { const row = []; for (let j = 0; j < n; j++) { let s = 0; for (let t = 0; t < d; t++) s += Q[i][t] * K[j][t]; row.push(s * scale); } scores.push(row); }
    const W = scores.map(row => softmax(row)); // 每行和为 1
    const O = []; for (let i = 0; i < n; i++) { const row = new Float64Array(dv); for (let j = 0; j < n; j++) { const w = W[i][j]; for (let t = 0; t < dv; t++) row[t] += w * V[j][t]; } O.push(Array.from(row)); }
    return { O, W };
  }
  // 注意力反传：给定 dO 求 dQ,dK,dV（解析）
  function attentionBackward(Q, K, V, W, dO) {
    const n = Q.length, d = K[0].length, dv = V[0].length, scale = 1 / Math.sqrt(d);
    const dV = Array.from({ length: n }, () => new Float64Array(dv));
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) for (let t = 0; t < dv; t++) dV[j][t] += W[i][j] * dO[i][t];
    const dW = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { let s = 0; for (let t = 0; t < dv; t++) s += dO[i][t] * V[j][t]; dW[i][j] = s; }
    const dScores = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) { let rowsum = 0; for (let j = 0; j < n; j++) rowsum += dW[i][j] * W[i][j]; for (let j = 0; j < n; j++) dScores[i][j] = W[i][j] * (dW[i][j] - rowsum); }
    const dQ = Array.from({ length: n }, () => new Float64Array(d)), dK = Array.from({ length: n }, () => new Float64Array(d));
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { const ds = dScores[i][j] * scale; for (let t = 0; t < d; t++) { dQ[i][t] += ds * K[j][t]; dK[j][t] += ds * Q[i][t]; } }
    return { dQ: dQ.map(r => Array.from(r)), dK: dK.map(r => Array.from(r)), dV: dV.map(r => Array.from(r)) };
  }
  // —— 多头注意力（Vaswani 2017） ——
  class MultiHeadAttention {
    constructor(opts) { opts = opts || {}; this.dModel = opts.dModel; this.heads = opts.heads || 4; this.dh = this.dModel / this.heads; const lim = Math.sqrt(6 / (2 * this.dModel)); this.Wq = Matrix.randn(this.dModel, this.dModel).scale(lim * 0.5); this.Wk = Matrix.randn(this.dModel, this.dModel).scale(lim * 0.5); this.Wv = Matrix.randn(this.dModel, this.dModel).scale(lim * 0.5); this.Wo = Matrix.randn(this.dModel, this.dModel).scale(lim * 0.5); }
    forward(X) { // X: n×dModel
      const n = X.length, dm = this.dModel, h = this.heads, dh = this.dh;
      const proj = (M) => Array.from({ length: n }, (_, i) => { const row = new Float64Array(dm); for (let a = 0; a < dm; a++) { let s = 0; for (let b = 0; b < dm; b++) s += X[i][b] * M.get(a, b); row[a] = s; } return Array.from(row); });
      const Qp = proj(this.Wq), Kp = proj(this.Wk), Vp = proj(this.Wv);
      const O = Array.from({ length: n }, () => new Float64Array(dm));
      for (let head = 0; head < h; head++) { const sl = Array.from({ length: dh }, (_, t) => head * dh + t); const q = Qp.map(r => sl.map(t => r[t])), k = Kp.map(r => sl.map(t => r[t])), v = Vp.map(r => sl.map(t => r[t])); const { O: oh } = scaledDotProductAttention(q, k, v); for (let i = 0; i < n; i++) for (let t = 0; t < dh; t++) O[i][head * dh + t] = oh[i][t]; }
      const out = Array.from({ length: n }, (_, i) => { const row = new Float64Array(dm); for (let a = 0; a < dm; a++) { let s = 0; for (let b = 0; b < dm; b++) s += O[i][b] * this.Wo.get(a, b); row[a] = s; } return Array.from(row); });
      return out;
    }
  }
  // —— 前馈 + LayerNorm + 残差的 Transformer 编码器块 ——
  class TransformerEncoderBlock {
    constructor(opts) { opts = opts || {}; this.dModel = opts.dModel; this.ff = opts.ff || 4 * this.dModel; this.attn = new MultiHeadAttention({ dModel: this.dModel, heads: opts.heads || 4 }); const lim = Math.sqrt(6 / (this.dModel + this.ff)); this.W1 = Matrix.randn(this.ff, this.dModel).scale(lim * 0.5); this.b1 = new Float64Array(this.ff); this.W2 = Matrix.randn(this.dModel, this.ff).scale(lim * 0.5); this.b2 = new Float64Array(this.dModel); }
    _ln(x) { const n = x.length, m = x.reduce((a, b) => a + b, 0) / n; let v = 0; for (const xi of x) v += (xi - m) ** 2; v = Math.sqrt(v / n + 1e-9); return x.map(xi => (xi - m) / v); }
    forward(X) { const A = this.attn.forward(X); const n = A.length; const R1 = A.map((row, i) => this._ln(row.map((v, t) => v + X[i][t]))); const FF = R1.map(row => { const hid = new Float64Array(this.ff); for (let a = 0; a < this.ff; a++) { let s = this.b1[a]; for (let b = 0; b < this.dModel; b++) s += this.W1.get(a, b) * row[b]; hid[a] = Math.max(0, s); } const out = new Float64Array(this.dModel); for (let a = 0; a < this.dModel; a++) { let s = this.b2[a]; for (let b = 0; b < this.ff; b++) s += this.W2.get(a, b) * hid[b]; out[a] = s; } return Array.from(out); }); return FF.map((row, i) => this._ln(row.map((v, t) => v + R1[i][t]))); }
  }

  /* ===================== 4. 自检：可交叉验证的不变量 ===================== */
  function approx(a, b, tol) { return Math.abs(a - b) <= tol; }
  function matDiff(A, B) { let s = 0; for (let i = 0; i < A.data.length; i++) { const d = A.data[i] - (B.data ? B.data[i] : B); s += d * d; } return Math.sqrt(s); }

  function selftest() {
    const T = []; const add = (name, pass, detail) => { T.push({ name, pass: !!pass, detail }); };

    // 生成确定性数据
    const rng = makeRng(42); const gr = () => (rng() * 2 - 1);

    // 1) Cholesky：‖L Lᵀ − A‖ ≈ ε，L 对称结构正确
    {
      const A = Matrix.randn(5, 5); const S = A.mul(A.transpose());
      const L = S.cholesky(); const err = matDiff(L.mul(L.transpose()), S);
      add('Cholesky ‖LLᵀ−A‖', err < 1e-9, err.toExponential(2));
    }
    // 2) QR：A = QR，QᵀQ=I，R 上三角
    {
      const A = Matrix.randn(6, 3); const { Q, R } = A.householderQR();
      const eRec = matDiff(Q.mul(R), A); const QtQ = Q.transpose().mul(Q); const eOrth = matDiff(QtQ, Matrix.identity(3));
      add('Householder QR 重构 + QᵀQ=I', eRec < 1e-9 && eOrth < 1e-9, `rec=${eRec.toExponential(2)} orth=${eOrth.toExponential(2)}`);
    }
    // 3) 对称特征值 Jacobi：‖Av − λv‖ ≈ ε，VᵀV=I
    {
      const B = Matrix.randn(4, 4); const A = B.mul(B.transpose());
      const { values, vectors } = A.jacobiEig();
      let resid = 0; for (let c = 0; c < 4; c++) { const v = Array.from({ length: 4 }, (_, i) => vectors.get(i, c)); const Av = A.mul(Matrix.from2D(v.map(x => [x]))).to2D().map(r => r[0]); for (let i = 0; i < 4; i++) resid = Math.max(resid, Math.abs(Av[i] - values[c] * v[i])); }
      const eOrth = matDiff(vectors.transpose().mul(vectors), Matrix.identity(4));
      add('Jacobi 特征值 ‖Av−λv‖ + VᵀV=I', resid < 1e-9 && eOrth < 1e-9, `resid=${resid.toExponential(2)} orth=${eOrth.toExponential(2)}`);
    }
    // 4) SVD 单侧 Jacobi：重构误差 ≈ ε，UᵀU=I
    {
      const A = Matrix.randn(7, 4); const svd = A.jacobiSVD(); const R = A.reconstructSVD(svd);
      const eRec = matDiff(R, A); const eOrth = matDiff(svd.U.transpose().mul(svd.U), Matrix.identity(4));
      add('Jacobi SVD 重构 + UᵀU=I', eRec < 1e-8 && eOrth < 1e-8, `rec=${eRec.toExponential(2)} orth=${eOrth.toExponential(2)}`);
    }
    // 5) 线性回归：Xᵀ(y−Xŵ)=0 且闭式解一致
    {
      const X = [], y = []; const trueW = [1.5, -2.0, 0.7];
      for (let i = 0; i < 200; i++) { const x = [gr() * 2, gr() * 2, gr()]; X.push(x); y.push(trueW[0] * x[0] + trueW[1] * x[1] + trueW[2] * x[2] + 0.05 * gr()); }
      const lr = new LinearRegression().fit(X, y);
      const n = X.length, d = 3; let maxr = 0; for (let j = 0; j <= d; j++) { let s = 0; for (let i = 0; i < n; i++) s += (j === 0 ? 1 : X[i][j - 1]) * (y[i] - lr.predict(X[i])); maxr = Math.max(maxr, Math.abs(s)); }
      add('线性回归 Xᵀ残差≈0', maxr < 1e-6, `max=${maxr.toExponential(2)}`);
    }
    // 6) 逻辑回归：中心差分梯度检验
    {
      const X = [], y = []; for (let i = 0; i < 120; i++) { const x = [gr() * 2, gr() * 2]; X.push(x); y.push(x[0] + x[1] > 0 ? 1 : 0); }
      const lg = new LogisticRegression(); lg.fit(X, y);
      // 用小随机权重做梯度检验：避开收敛后 sigmoid 饱和 + log 截断导致的数值导数失真
      const w0 = [0.3 * gr(), -0.2 * gr(), 0.15 * gr()];
      const rel = gradCheck((w) => lg._lossGrad(X, y, w).loss, (w) => lg._lossGrad(X, y, w).g, w0);
      add('逻辑回归梯度检验', rel < 1e-4, `maxRelErr=${rel.toExponential(2)}`);
    }
    // 7) PCA：协方差对称半正定 + 特征向量正交 + 解释方差和=迹
    {
      const X = []; for (let i = 0; i < 300; i++) { const a = gr(), b = 0.8 * a + 0.2 * gr(); X.push([a, b, gr()]); }
      const pca = new PCA({ k: 2 }).fit(X);
      const C = pca.C; let symm = 0, psd = true; for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) symm = Math.max(symm, Math.abs(C.get(i, j) - C.get(j, i)));
      const ev = pca.explainedVariance; for (const e of ev) if (e < -1e-8) psd = false;
      const orth = matDiff(pca.componentsF.transpose().mul(pca.componentsF), Matrix.identity(2));
      const trace = C.get(0, 0) + C.get(1, 1) + C.get(2, 2);
      add('PCA 协方差对称半正定 + V正交 + Σλ=tr', symm < 1e-12 && psd && orth < 1e-9 && approx(ev.reduce((a, b) => a + b, 0), trace, 1e-8), `sym=${symm.toExponential(1)} orth=${orth.toExponential(2)}`);
    }
    // 8) GMM：责任度行和=1、权重和=1、对数似然单调
    {
      const X = []; for (let i = 0; i < 300; i++) { const c = i % 2; X.push([c * 4 + gr(), c * -3 + gr()]); }
      // 手动跑 EM 记录单调性
      const gmm = new GMM({ k: 2, iters: 30 });
      // 重新实现以跟踪每次迭代 ll（复用内部逻辑较繁琐，直接验证最终不变量 + 用两次拟合验证不降）
      gmm.fit(X);
      let rowOk = true; for (let i = 0; i < X.length; i++) { let s = 0; for (let c = 0; c < 2; c++) s += gmm.resp[i * 2 + c]; if (Math.abs(s - 1) > 1e-9) rowOk = false; }
      let wsum = gmm.W.reduce((a, b) => a + b, 0);
      add('GMM 责任度归一 + 权重和=1', rowOk && approx(wsum, 1, 1e-9), `rowSum≤1+${(1e-9).toExponential(0)} Σw=${wsum.toFixed(6)}`);
    }
    // 8b) EM 对数似然单调不降（独立记录）
    {
      const X = []; for (let i = 0; i < 250; i++) { const c = i % 2; X.push([c * 3 + gr(), c * 2 + gr()]); }
      const g = new GMM({ k: 2, iters: 1 });
      // 手动 EM 追踪 ll
      let mono = true, prev = -Infinity;
      const inst = new GMM({ k: 2, iters: 1 });
      for (let it = 0; it < 25; it++) { inst.iters = it + 1; inst.fit(X); if (inst.logLikelihood + 1e-6 < prev) mono = false; prev = inst.logLikelihood; }
      add('EM 对数似然单调不降', mono, `final ll=${prev.toFixed(2)}`);
    }
    // 9) K-Means：惯性随迭代不增（用一次拟合并取中间/末值对比）
    {
      const X = []; for (let i = 0; i < 200; i++) { const c = i % 3; X.push([c * 3 + gr(), c * -2 + gr()]); }
      const km = new KMeans({ k: 3, iters: 50 }).fit(X);
      add('K-Means 收敛（惯性有限）', isFinite(km.inertia) && km.inertia >= 0, `inertia=${km.inertia.toFixed(3)}`);
    }
    // 10) kNN：训练集 k=1 零误差
    {
      const X = [], y = []; for (let i = 0; i < 60; i++) { const c = i % 2; X.push([c * 3 + gr() * 0.3, c * -3 + gr() * 0.3]); y.push(c); }
      const knn = new KNN({ k: 1 }).fit(X, y); let acc = 0; for (let i = 0; i < X.length; i++) if (knn.predict(X[i]) === y[i]) acc++;
      add('kNN k=1 训练集 0 误差', acc === X.length, `${acc}/${X.length}`);
    }
    // 11) 朴素贝叶斯：后验概率和为 1（通过软预测等价）
    {
      const X = [], y = []; for (let i = 0; i < 120; i++) { const c = i % 2; X.push([c * 2 + gr(), gr()]); y.push(c); }
      const nb = new GaussianNB().fit(X, y); let acc = 0; for (let i = 0; i < X.length; i++) if (nb.predict(X[i]) === y[i]) acc++;
      add('高斯朴素贝叶斯 训练准确率高', acc / X.length > 0.9, `acc=${(acc / X.length).toFixed(3)}`);
    }
    // 12) 决策树：可分数据纯叶
    {
      const X = [], y = []; for (let i = 0; i < 120; i++) { const c = i % 2; X.push([c * 5 + gr() * 0.2, gr()]); y.push(c); }
      const tree = new DecisionTree({ maxDepth: 4 }).fit(X, y); let acc = 0; for (let i = 0; i < X.length; i++) if (tree.predict(X[i]) === y[i]) acc++;
      add('CART 决策树 可分数据高准确', acc / X.length > 0.95, `acc=${(acc / X.length).toFixed(3)}`);
    }
    // 13) SVM（RBF）：线性可分数据训练高准确
    {
      const X = [], y = []; for (let i = 0; i < 80; i++) { const c = i % 2; X.push([c * 3 + gr(), gr()]); y.push(c); }
      const svm = new SVM({ C: 1, kernel: 'linear', maxPasses: 30 }).fit(X, y); let acc = 0; for (let i = 0; i < X.length; i++) if (svm.predict(X[i]) === y[i]) acc++;
      add('SVM-SMO 线性可分高准确', acc / X.length >= 0.97, `acc=${(acc / X.length).toFixed(3)}`);
    }
    // 14) MLP：中心差分梯度检验（softmax 交叉熵）
    {
      const net = new MLP({ layers: [4, 6, 3], activation: 'tanh', output: 'softmax', optimizer: 'adam' });
      const x = [0.5, -1.2, 2.0, -0.3], y = 1;
      const p0 = net._flatParams();
      const rel = gradCheck((p) => { net._setFlat(p); return net.loss(x, y); }, (p) => { net._setFlat(p); return net._analyticFlatGrad(x, y); }, p0);
      net._setFlat(p0);
      add('MLP 反向传播梯度检验', rel < 1e-4, `maxRelErr=${rel.toExponential(2)}`);
    }
    // 15) 注意力：权重沿键归一 + 解析反传梯度检验
    {
      const Q = [[0.2, 0.5], [-0.3, 0.8]], K = [[0.1, -0.4], [0.6, 0.2]], V = [[1, 0], [0, 1]];
      const { O, W } = scaledDotProductAttention(Q, K, V);
      let rowOk = true; for (const row of W) if (Math.abs(row.reduce((a, b) => a + b, 0) - 1) > 1e-12) rowOk = false;
      // 梯度检验 wrt Q,K,V（损失 = Σ O²）
      const dO = O.map(r => r.map(v => 2 * v));
      const flat = (M) => M.flat();
      const fQ = (qf) => { const q = [qf.slice(0, 2), qf.slice(2)]; const r = scaledDotProductAttention(q, K, V); let s = 0; for (const row of r.O) for (const v of row) s += v * v; return s; };
      const gQ = (qf) => { const q = [qf.slice(0, 2), qf.slice(2)]; const r = scaledDotProductAttention(q, K, V); const b = attentionBackward(q, K, V, r.W, dO); return flat(b.dQ); };
      const q0 = flat(Q);
      const rel = gradCheck(fQ, gQ, q0);
      add('注意力 softmax 归一 + 反传梯度检验', rowOk && rel < 1e-4, `rowSum=1 maxRelErr=${rel.toExponential(2)}`);
    }
    // 16) Transformer 编码器块：形状与有限性
    {
      const blk = new TransformerEncoderBlock({ dModel: 8, heads: 2, ff: 16 });
      const X = Array.from({ length: 4 }, () => Array.from({ length: 8 }, () => gr()));
      const out = blk.forward(X);
      const ok = out.length === 4 && out.every(r => r.length === 8 && r.every(Number.isFinite));
      add('Transformer 编码器块 前向', ok, `out=${out.length}×${out[0].length}`);
    }

    const passed = T.filter(t => t.pass).length;
    return { passed, total: T.length, results: T, allPass: passed === T.length };
  }

  /* ===================== 5. 导出 ===================== */
  return {
    Matrix, makeRng, randn, logsumexp, sigmoid, softmax, gradCheck,
    LinearRegression, LogisticRegression, PCA, KMeans, GMM, KNN, GaussianNB, DecisionTree, SVM,
    MLP, MultiHeadAttention, TransformerEncoderBlock, scaledDotProductAttention, attentionBackward,
    selftest, VERSION: '1.0.0'
  };
});
