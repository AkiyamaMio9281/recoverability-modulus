# M6 — SFBD-OMNI 真实算法动力学:Ψ 控制 clean 空间收敛(D8)

> 不再用理想估计量,而是跑**真实 SFBD-OMNI 交替最小化**(闭式高斯)。结论:`Ψ`(=条件数)
> 控制 clean 空间的可恢复性——*Identifiability ≠ Recoverability* 在算法**自己的迭代**里发生。
> 这把 D6 从"骨架"升级为"对着真算法验证过"。命名:SFBD = ICML'25。

---

## 1. 闭式 OMNI 更新(高斯-线性)

OMNI 交替更新(γ=1, λ=0,`SFBD OMNI/contents/SFBD.tex`):
$$u_y^k(x)=\frac{p^k(x)\,r(y\mid x)}{(T_r p^k)(y)},\qquad p^{k+1}(x)=\int q(y)\,u_y^k(x)\,dy.$$

设 $p^k=\mathcal N(m_k,S_k)$,腐蚀 $y=Ax+\varepsilon,\ \varepsilon\sim\mathcal N(0,\Gamma)$。

**第一步:$u_y^k$ 是线性-高斯后验**。联合 $p^k(x)\,r(y\mid x)=\mathcal N(x;m_k,S_k)\,\mathcal N(y;Ax,\Gamma)$,
后验 $x\mid y$ 为高斯:
$$x\mid y\sim\mathcal N\big(m_k+K_k(y-Am_k),\ (I-K_kA)S_k\big),\qquad K_k=S_kA^\top(AS_kA^\top+\Gamma)^{-1}.$$

**第二步:$p^{k+1}$ 是 $x$ 的边缘**。$y\sim q=\mathcal N(m_q,S_q)$($m_q=A\mu,\ S_q=A\Sigma A^\top+\Gamma$),
$x=m_k+K_k(y-Am_k)+\text{(indep noise }(I-K_kA)S_k)$,于是
$$\boxed{\;m_{k+1}=m_k+K_k(m_q-Am_k),\qquad S_{k+1}=K_kS_qK_k^\top+(I-K_kA)S_k.\;}$$

**这是卡尔曼更新式**——OMNI 一步 = 朝 $q$ 的一次 Kalman 校正。

**不动点 = $p_{\mathrm{data}}$**:代入 $m_k=\mu,S_k=\Sigma$,$K S_q=\Sigma A^\top$,可验 $m_{k+1}=\mu$、$S_{k+1}=\Sigma$。∎
(injective $A$ 时收敛到 $p_{\mathrm{data}}$,与 OMNI 收敛性一致。)

---

## 2. 核心发现:Ψ 控制 clean 空间收敛

跑 $d=5$、噪声 $\sigma=0.3$、条件数 $\kappa\in\{1,5,25,125,625\}$(degrade $A$),5 seeds,50 迭代。

**末迭代(seeds 中位数):**

| $\kappa=\Psi$ | 腐蚀空间 $\mathrm{KL}(q\Vert T_r p^k)$ | clean 空间 $W_2(p^k,p_{\mathrm{data}})$ |
|---|---|---|
| 1 | $0$ | $0$ |
| 5 | $5.2\times10^{-10}$ | $7.0\times10^{-5}$ |
| 25 | $3.4\times10^{-4}$ | $0.31$ |
| 125 | $7.1\times10^{-5}$ | $0.51$ |
| 625 | $\mathbf{3.9\times10^{-6}}$ | $\mathbf{0.53}$ |

**读法(这就是整个论点的最强一击):**
- **腐蚀空间(Panel A)**:所有 $\kappa$ 的 $\mathrm{KL}(q\Vert T_r p^k)$ 都收敛——算法"在腐蚀空间看起来成功"。
- **clean 空间(Panel B)**:$W_2(p^k,p_{\mathrm{data}})$ 随 $\kappa=\Psi$ 增大而**停滞**在越来越高的水平。
- **gap(Panel C)**:$\kappa=625$ 时腐蚀 KL 小到 $3.9\times10^{-6}$(看起来完全收敛!),clean $W_2$ 却 $=0.53$
  (彻底失败)。**算法在最病态时、在腐蚀空间看起来最"收敛",而在 clean 空间最离谱。**

这正是 OMNI 自己点出的 *Identifiability ≠ Recoverability*——但现在是在**它自己的迭代轨迹**里、
被 $\Psi$ 定量地展示出来,而非定性吐槽。

---

## 3. 与 D6 的关系

D6:$d_R(p_{\mathrm{data}},p^{k})\lesssim R\sqrt{\chi^2(T_r p^k\Vert q)}$,即 clean 误差 $\approx\Psi\times\sqrt{\text{腐蚀差}}$。
本实验正是它的**算法级证据**:腐蚀差($\mathrm{KL}$)很小,但乘上大 $\Psi$ 后 clean 误差仍大;$\Psi$ 越大,
同样的腐蚀收敛"翻译"到 clean 空间的误差越大。**D8 把 D6 的结构在真算法上验证了。**

---

## 4. 边界 / 待 Yu
- 仍是**高斯-线性、γ=1、λ=0、injective**。非高斯/非线性、`λ>0` 的非单射 + clean 先验(`h†`)是下一步。
- `γ<1` 的 online 变体会破坏高斯性(混合分布),需另作近似。
- corrupted KL 在大 $\kappa$ 处随 $\kappa$ **非单调**(算法把腐蚀分布拟合得很好、与 $\kappa$ 关系复杂);
  但"腐蚀小 / clean 大"的 gap 结论稳健。值得和 Yu 讨论这条 corrupted-KL 曲线的解释。
</content>
