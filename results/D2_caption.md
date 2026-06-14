# D2 — Recoverability modulus vs clean budget（主图 caption）

文件：`D2_budget_curve.png` / `.pdf`（dim=8，σ_noise=0.5，ε=0.3，5 seeds，mean±std）。

**一图三事**（proposal §9 / primer §0）：

1. **可恢复性是连续的，不是二元的。** 横轴是线性腐蚀算子 `A` 从满秩 `degrade` 到近奇异
   的条件数 `κ(A)`。预算随 `κ` **平滑**变化，没有"可识别/不可识别"的硬跳变——这正是把
   OMNI 的二元 `T_r` 单射判据换成连续量的可视化。

2. **Ψ 预测预算（右panel）。** 噪声-only 预算 `n(κ)`（要达到目标 clean-space 误差 ε）随
   `κ` **发散**，且精确落在 `Ψ²` 渐近线 `∝ σ_noise²·κ²/ε²` 上（黑虚线）。放大因子
   `σ_noise²/σ_i² = σ_noise²·Ψ_I,i²` 就是 M3 的 Picard/Ψ——"单射却不可恢复"（SFBD 悲观）
   的定量来源。

3. **低秩 → 需要 clean，且被 clean "封顶"（左panel）。** 固定一份**充裕**噪声预算
   `n₀=2000`，所需 clean 样本 `m(κ)` 有三相：
   - **绿（κ≲34）**：`m=0`，纯腐蚀数据即可恢复；
   - **橙（34≲κ≲360）**：`m` 从 0 升起，clean 开始补偿病态方向；
   - **红（κ≳360）**：`m` **饱和**在 ≈7.3（"clean cap"），该方向完全由 clean 承担。

**诚实点（对 proposal §9 措辞的精确化）**：真正**发散**的是**噪声**预算（右panel）；而**clean**
预算是被**封顶**的（左panel 红区平台）。这反而比"clean 预算发散"更强、更贴两篇——它正是
OMNI "极少量 clean（如 50 张）即可大幅缓解" 的定量解释：clean 的价值在于给一个本会发散的
噪声需求**封顶**。

**方法**：全程高斯-线性闭式（`recover.py`/`budget.py`），无神经网络；闭式误差已用 Monte-Carlo
验证（`tests/test_recover.py`，5% 内）。Ψ 指无权 `Ψ_I`；实验不经过 OMNI 的 KL 率，故不受
散度桥接问题影响（red line 4）。
</content>
