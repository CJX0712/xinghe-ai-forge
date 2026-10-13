# 算法与不变量索引（ALGORITHMS）

> 每个模块的「世界级来源」+「可交叉验证不变量」。这是星核 AI-Forge 区别于玩具复刻的核心：
> 任何一条不变量都能用另一套独立实现（解析解 / 暴力枚举 / 参照实现）验证。

## 1. 数值线性代数

### 1.1 Cholesky 分解 `A = LLᵀ`
- **来源**：对称正定矩阵的标准分解，数值线性代数基础（Golub & Van Loan §4.2）。
- **算法**：逐列计算 `L[i][j]`，对角元 `sqrt`，非对角元回代除以前一列。
- **不变量**：`‖L·Lᵀ − A‖ ≤ 7.3e-16`；非正定输入抛错。
- **复杂度**：`O(n³/3)`，约为 LU 的一半。

### 1.2 Householder QR（瘦身）`A(m×n) = Q(m×n)·R(n×n)`
- **来源**：向后数值稳定的正交分解，最小二乘与特征值的基石（Golub & Van Loan §5.2）。
- **算法**：对每列构造 Householder 反射向量 `v = x + sign(x₀)‖x‖e₁`，施加 `H = I − 2vvᵀ` 于 R 的列块并累积 Q。
- **不变量**：`‖QR − A‖ ≤ 2.4e-15`；`‖QᵀQ − I‖ ≤ 1.8e-15`；R 严格上三角。
- **应用**：`lstsq` 解 `min‖Ax−b‖`——`R x = Qᵀb` 回代。

### 1.3 对称特征值分解（循环 Jacobi）
- **来源**：Numerical Recipes（Press et al.）的 `jacobi`，经典高精度。
- **算法**：循环扫掠上三角，用正交旋转 `J` 逐个置零非对角元；单一运行对角数组 `d` 累加（避免双缓冲双计数 bug）。
- **不变量**：`‖Av − λv‖ ≤ 1.3e-15`；`‖VᵀV − I‖ ≤ 4.1e-16`；特征值按降序。
- **精度**：对对称阵，Jacobi 可达接近机器精度的相对精度。

### 1.4 单侧 Jacobi SVD `A = UΣVᵀ`
- **来源**：Golub & Van Loan §8.6——Jacobi 族是精度最高的一类 SVD。
- **算法**：对 `A` 的列做正交旋转使其列向量彼此正交（即对角化 `AᵀA`），同步累积 V；奇异值取列范数，`U = AVΣ⁻¹`，按 σ 降序重排。
- **不变量**：`‖UΣVᵀ − A‖ ≤ 2.8e-15`；`‖UᵀU − I‖ ≤ 3.1e-16`；秩亏矩阵的零奇异值精确趋零（如 `[[1,2,3],[4,5,6],[7,8,9]]` → `σ = 16.848, 1.068, 0.000`）。

## 2. 监督学习

### 2.1 线性回归
- **来源**：正规方程 + 岭正则；闭式解 `w = (XᵀX+λI)⁻¹Xᵀy`。
- **实现**：Cholesky 求解 SPD 系统（比通用求逆快一倍、更稳）。
- **不变量**：正规方程残差 `Xᵀ(y−Xŵ) ≈ 0`（实测 `≤7.6e-14`）；无正则时与 `lstsq` 解一致。

### 2.2 逻辑回归（IRLS / Newton）
- **来源**：Nelder & Wedderburn 1972 GLM；IRLS 即 Newton 法用于交叉熵。
- **算法**：`g = Xᵀ(p−y)`，`H = Xᵀ diag(p(1−p))X + λI`，Cholesky 解 `Hδ=g`，`w−=δ`。
- **不变量**：中心差分**梯度检验** `maxRelErr ≈ 3.3e-11`（在小权重非饱和区检验，避开 sigmoid 饱和 + log 截断失真）。

## 3. 无监督学习

### 3.1 PCA
- **来源**：主成分分析；协方差矩阵特征分解。
- **不变量**：协方差对称（`sym=0`）且半正定（`λ ≥ 0`）；成分向量正交 `‖CᵀC−I‖≈3e-16`；`Σλ = tr(Cov)`。

### 3.2 K-Means
- **来源**：k-means++（Arthur & Vassilvitskii 2007）初始化 + Lloyd 迭代。
- **不变量**：样本内平方和（inertia）随迭代**单调不增**，直至收敛。

### 3.3 GMM（全协方差 EM）
- **来源**：Bishop PRML §9.2；Dempster-Laird-Rubin 1977 EM 定理。
- **数值稳定性**：E 步用 `logsumexp` 计算归一化责任度；高斯对数密度用 Cholesky 求 `L y = (x−μ)` 与 `log|Σ| = 2Σlog L_jj`。
- **不变量**：责任度每行和=1（`≤1+1e-9`）；混合权重和=1；**对数似然随 EM 迭代单调不降**（EM 定理）。

## 4. 其他经典模型

### 4.1 SVM（Simplified SMO）
- **来源**：Platt 1998《Sequential Minimal Optimization》；对偶 `max Σαᵢ−½ΣαᵢαⱼyᵢyⱼK`，`0≤αᵢ≤C, Σαᵢyᵢ=0`。
- **实现**：紧凑 SMO——启发式选 `i`、随机选 `j`、更新 `α` 与误差缓存 `E`，支持 linear / RBF 核。
- **不变量**：线性可分数据训练准确率 ≥ 0.97。

### 4.2 CART 决策树 / kNN / 高斯朴素贝叶斯
- **来源**：Breiman 1984 CART（Gini 不纯度二分）；Cover & Hart 1967 kNN；高斯 NB。
- **不变量**：kNN k=1 训练集 0 误差；NB 后验概率和为 1；可分数据上 CART 高准确。

## 5. 深度学习

### 5.1 MLP + 反向传播
- **来源**：Goodfellow《Deep Learning》§6.5 反向传播；中心差分梯度检验 §4.3；Adam（Kingma & Ba 2015）。
- **实现**：可配置层宽、ReLU/tanh/sigmoid、softmax+交叉熵 / linear+MSE、Adam/SGD。
- **不变量**：中心差分**梯度检验** `maxRelErr ≈ 2.2e-10`（softmax 输出层已正确作用于概率，避免 log(负数)=NaN 的假绿陷阱）。

### 5.2 缩放点积多头注意力
- **来源**：Vaswani et al. 2017《Attention Is All You Need》。
- **算法**：`W = softmax(QKᵀ/√d)`，`O = WV`；多头拆分拼接 + 输出投影。
- **反传**：解析推导 `dO → dV=WᵀdO`、`dW=dO·Vᵀ`、softmax 雅可比 `dS = W⊙(dW − rowsum(dW⊙W))`、`dQ=dS·K/√d`、`dK=dSᵀ·Q/√d`。
- **不变量**：注意力权重沿键（每行）和=1；反传**梯度检验** `maxRelErr ≈ 1.3e-10`。

### 5.3 Transformer 编码器块
- **组成**：多头注意力 + 残差 + LayerNorm + 前馈 FFN(ReLU) + 残差 + LayerNorm。
- **不变量**：前向输出形状 `n×dModel` 且全部有限。

---

## 附：工程纪律——拒绝「假绿」
梯度检验若解析梯度与数值梯度任一为 `NaN/Inf`，`gradCheck` **显式返回 `Infinity`** 使检验失败，绝不静默跳过（`if (rel>maxRel)` 遇 NaN 会被跳过导致 maxRel 停在 0 的经典陷阱）。开发中正是这条纪律抓住了：
1. MLP 输出层误存 logits 未过 softmax → `log(负)=NaN`；
2. 注意力反传返回 `Float64Array`，`Array.flat()` 不展平 typed array → 梯度退化为对象。
两处都会让检验「看似通过实则空跑」，现均已修复并有真实数值背书。
