# D3 — Ψ 预测样本复杂度（采样验证）caption

文件：`D3_sample_complexity_n.png`（噪声）、`D3_sample_complexity_m.png`（clean），dim=8，4 seeds，每点跨 3 个 ε 目标。

**做了什么**：D2 是闭式律；D3 用**真采样**走完整管线（抽样 → 估计 q̂/clean → 反演恢复 →
测 clean-space W2），反解"达到目标 ε 所需样本数"，与 `budget.predict_budget` 的 Ψ 预测对照。

**结果**：

| 散点 | 物理量 | slope | R² | 中位相对误差 |
|---|---|---|---|---|
| 噪声 `n` | injective 病态算子（gaussian/blur/degrade），Ψ 放大 | **1.001** | **1.0000** | **1.17%** |
| clean `m` | non-injective 算子零空间（grayscale/masking），O(1/√m) | **1.002** | **0.9997** | **1.00%** |

**读法**：
- 噪声散点跨 **5 个数量级**（n≈30 → 7×10⁶）全部落在 `y=x` 上；degrade(κ≈10³) 把所需 `n`
  推到 10⁶–10⁷，正是 Ψ²∝κ² 放大——"单射却不可恢复"的采样证据。
- clean 散点验证零空间方向由 clean 以 `m_pred=Σ_null s_i/ε²` 恢复（O(1/√m)）。
- 残余 ~1% 是 Monte-Carlo 抽样噪声，非系统偏差 → **未见 `Ψ_q` vs `Ψ_I` 的常数偏移**
  （prompt 04 gotcha 所列；若出现会记为问 Yu 项，此处无）。

**意义**：把 D2 的闭式律升级为**采样验证的预测律**——`Ψ_I` 谱直接、定量地预测端到端样本预算。
全程无神经网络；Ψ 指无权 `Ψ_I`，不经 OMNI 的 KL 率（red line 4 无关）。
</content>
