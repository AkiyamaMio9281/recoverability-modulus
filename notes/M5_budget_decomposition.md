# M5 — clean 预算分解（D7，线性算子）

> prompt 07 / primer §3, 决策 C（**限定线性/可线性化** `T_r`；一般非线性无良定义零空间）。
> 把总恢复误差拆成"可恢复部分（由模数控制）+ 不可恢复部分（零空间，由 clean 估）"，并把它
> 接到 OMNI 的 `h†`（I-projection）与 Tikhonov 正则化。命名：SFBD = SFBD (ICML'25)。

---

## 1. 命题（线性预算分解）

设线性腐蚀 `T_r`（奇异值 `σ_i`、右奇异基 `v_i`），clean 模型在该基中 `Σ_x=diag(s_i)`。用 `n`
个噪声样本（或等价地 OMNI 迭代 `K`）恢复 range 方向、`m` 个 clean 样本恢复零空间方向。则
clean-space 误差（均值部 W2²）正交分解为
```
err² =  Σ_{i: σ_i>0} Var_i              +   Σ_{i: σ_i=0} Var_i
     =  「可恢复部分」                      +   「不可恢复部分」
     ~  O(1/(γK))  或  O(1/n)            +   O(1/m).
```
- **可恢复部分**（`range(T_r*)`）：由模数在 range 上控制；噪声样本 `n`↑（或 OMNI 迭代 `K`↑）→ 0。
- **不可恢复部分**（`null(T_r)`）：`‖h − p_data‖` 在零空间上，仅 `m` 个 clean 样本能估，`~Σ_null s_i/m`。

---

## 2. 推导（SVD range⊕nullspace）

每方向 `i` 的逆方差精度（recover.py）：噪声 `a_i = σ_i²/(σ_i² s_i + σ_noise²)`、clean `b_i = 1/s_i`。
合并方差 `Var_i = 1/(n a_i + m b_i)`。
- `σ_i>0`：`a_i>0`，噪声路线供精度 → `n` 主导，`Var_i → s_i·(放大)/n`，放大 = `σ_noise²/σ_i² = σ_noise²·Ψ_I,i²`（与 M3 Picard 一致）。
- `σ_i=0`：`a_i=0`，噪声零信息 → `Var_i = s_i/m`（仅 clean）。

零空间正交于 range，故误差按方向相加 → §1 的分解，且 **两部分严格可加**。

---

## 3. Tikhonov 正则化与 `h†` 自洽（OMNI Prop 2）

augmented-KLAP `min KL(q‖T_r p) + λ KL(h‖p)` 的线性化解 = Tikhonov 滤波：方向 `i` 的滤波因子
```
f_i = σ_i² / (σ_i² + λ),
```
- 放大被**封顶**到 `~1/(2√λ)`（避免 `σ_i→0` 的爆炸），代价是零空间方向引入对先验 `h` 的偏差
  `(1−f_i)·(h − p_data)_i`。
- `λ→0`：`f_i→1`（range 方向无偏），但零空间 `f_i=0` 恒保留 `h` 分量 → 解 → `h†`，即 `h` 到解集
  `S(q)` 的 **I-projection**（**OMNI Prop 2**，`SFBD OMNI/contents/amb_KL_proj.tex:70-73`）。**自洽。**
- 偏差-方差权衡：`λ`↑ 减方差（封顶放大）增偏差（偏向 `h`）；`λ`↓ 反之。clean 样本 `m`↑ 使 `h` 更准 →
  偏差项随 `O(1/√m)` 减小，正是 §1 的不可恢复部分。

---

## 4. 数值核验（grayscale rank=5 + 噪声，dim=8）

`results` 之外的快速核验（与 recover.py 一致）：
```
total err²            = 0.06604
  recoverable(range)  = 0.01627   (n=500)   → 0.00175 (n=5000) → 0.00018 (n=50000)   ∝ 1/n
  unrecoverable(null) = 0.04977   = Σ_null s_i / m  (m=40)，对 n 不变
  additivity          = 0.06604   （两部分严格相加 ✓）
```
- 可恢复部分 `∝1/n` 衰减；不可恢复部分对 `n` 不变、只随 `m` 降（`∝1/m`）——**正是 §1 分解**。
- **不可恢复部分 = D3 的 clean-m 散点所验证的量**（`K²_clean=Σ_null s_i`，`results/D3_sample_complexity_m.png`，
  slope 1.002 / R²=0.9997）。即 **D3 已经实证验证了 M5 的不可恢复项**。

---

## 5. 与两篇图的对照

- **OMNI Fig 2 / 3a**（少量 clean 大幅改善）↔ **不可恢复部分** `O(1/√m)`：clean 样本把零空间偏差
  压下去；M5 给出它的定量形式 `Σ_null s_i/m`，并解释"为何 50 张就够"（=零空间维数 × 信号能量小）。
- **SFBD Fig 3**（噪声样本的悲观速率）↔ **可恢复部分**的慢率：range 上的放大 `σ_noise²·Ψ_I,i²`
  使噪声预算大（D2 右panel 的 Ψ² 发散、M4 §5 的 log 率）。
- **D2 主图**：左panel 的"clean 封顶"= §1 不可恢复部分的饱和；右panel 的发散 = 可恢复部分在
  噪声-only 下的 Ψ² 代价。M5 是 D2 的定理化。

---

## 6. 待 Yu 复核
1. 把"均值部 W2"的分解推广到**全 W2（含协方差 Bures 项）**是否同型（协方差放大是 `(σ_noise²/σ_i²)²`，平方量级）。
2. §3 的 Tikhonov ↔ augmented-KLAP 对应在非高斯先验 `h` 下的精确接口（source-condition 语言）。
3. 决策 C：可线性化但非严格线性的 `T_r`（如弱非线性腐蚀）能否保留该分解的近似版。
</content>
