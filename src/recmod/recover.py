"""高斯-线性设定下的闭式恢复与样本预算（Phase 2/3）。

设定（primer §4 + prompt 03）：
    clean:  x ~ N(μ_x, Σ_x)，在 A 的右奇异基中 Σ_x = diag(s_i)（各向异性）。
    腐蚀:   y = A x + ε，ε ~ N(0, σ_noise² I)，A = U diag(σ_i) Vᵀ。

在方向 i（A 的奇异方向）上估计 clean 参数（这里取**均值** μ_x,i，W2 的均值部分；
协方差估计有同型的 1/σ_i² 结构，平方量级）：

    噪声样本单样本精度  a_i = σ_i² / (σ_i² s_i + σ_noise²)
    clean 样本单样本精度 b_i = 1 / s_i

n 个噪声样本 + m 个 clean 样本，逆方差最优合并后该方向方差：
    Var_i(n, m) = 1 / (n a_i + m b_i).

clean-space 误差（均值部 W2）：err(n,m) = sqrt( Σ_i Var_i(n,m) )。

关键事实（驱动 D2）：
    噪声路线放大 = σ_noise²/σ_i² = σ_noise² · Ψ_I,i²（Picard/M3 一致）。σ_i→0 时
    噪声预算发散，而 clean 路线 b_i=1/s_i 与 σ_i 无关 → clean 预算对 κ 平坦。

全程闭式（不训练网络）。Ψ 指无权 Ψ_I（primer §3），实验线不经过 OMNI 的 KL 率，
故不受散度桥接问题影响（red line 4）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .operators import LinearCorruption, ZERO_TOL


# ----------------------------------------------------------------------------
# 高斯模型 + 闭式 W2
# ----------------------------------------------------------------------------
@dataclass
class GaussianModel:
    """各向异性高斯 N(mu, diag(s))，在 A 的右奇异基中对角。"""
    mu: np.ndarray      # (d,)
    s: np.ndarray       # (d,) 方差（diag(Σ_x)）

    @property
    def dim(self) -> int:
        return self.mu.shape[0]


def w2_gaussian_diag(mu1, s1, mu2, s2) -> float:
    """两个对角高斯之间的 W2 距离（闭式）：‖μ1−μ2‖² + Σ(√s1−√s2)²，再开方。"""
    mu1, s1, mu2, s2 = map(np.asarray, (mu1, s1, mu2, s2))
    mean_part = np.sum((mu1 - mu2) ** 2)
    cov_part = np.sum((np.sqrt(s1) - np.sqrt(s2)) ** 2)
    return float(np.sqrt(mean_part + cov_part))


# ----------------------------------------------------------------------------
# 每方向精度
# ----------------------------------------------------------------------------
def direction_precisions(op: LinearCorruption, model: GaussianModel):
    """返回 (a, b, sigmas)：噪声/ clean 单样本精度与 A 的奇异值（按 op 的右奇异基）。

    约定模型 Σ_x 已在 A 的右奇异基中对角（s 与 v_i 对齐）。σ_i≈0 的方向 a_i=0
    （噪声路线对该方向零信息 → 必须 clean）。
    """
    sig = op.singular_values()
    d = model.dim
    if sig.size < d:  # 补零空间方向
        sig = np.concatenate([sig, np.zeros(d - sig.size)])
    sig = sig[:d]
    s = model.s
    sn2 = op.noise_std ** 2
    denom = sig ** 2 * s + sn2
    # σ_i=0 且无噪声 → denom=0；该方向噪声路线零精度
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(denom > 0, sig ** 2 / denom, 0.0)
    b = 1.0 / s
    return a, b, sig


def achievable_error(op: LinearCorruption, model: GaussianModel,
                     n: float, m: float) -> float:
    """给定 n 噪声 + m clean，闭式 clean-space 误差（均值部 W2）。"""
    a, b, _ = direction_precisions(op, model)
    prec = n * a + m * b
    if np.any(prec <= 0):
        return np.inf  # 某方向无任何信息
    return float(np.sqrt(np.sum(1.0 / prec)))


# ----------------------------------------------------------------------------
# 样本预算（闭式 / 数值反解）
# ----------------------------------------------------------------------------
def required_noisy_only(op: LinearCorruption, model: GaussianModel, eps: float) -> float:
    """m=0 时达到误差 ε 所需噪声样本 n。σ_i→0 时发散（Ψ² 放大）。"""
    a, _, _ = direction_precisions(op, model)
    if np.any(a <= 0):
        return np.inf  # 存在噪声不可达方向 → 噪声单独无法恢复
    # err² = Σ 1/(n a_i) = (1/n) Σ 1/a_i ≤ ε²  → n ≥ Σ(1/a_i)/ε²
    return float(np.sum(1.0 / a) / eps ** 2)


def required_clean_only(op: LinearCorruption, model: GaussianModel, eps: float) -> float:
    """n=0 时达到误差 ε 所需 clean 样本 m = Σ s_i / ε²。与 κ 无关（对 κ 平坦）。"""
    _, b, _ = direction_precisions(op, model)
    return float(np.sum(1.0 / b) / eps ** 2)


def _required_noisy_given_clean(op, model, eps, m, n_hi=1e18) -> float:
    """固定 m clean，二分求达到 ε 所需噪声 n。不可达返回 inf。"""
    if achievable_error(op, model, 0.0, m) <= eps:
        return 0.0
    if achievable_error(op, model, n_hi, m) > eps:
        return np.inf
    lo, hi = 0.0, n_hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if achievable_error(op, model, mid, m) <= eps:
            hi = mid
        else:
            lo = mid
    return hi


def required_clean_given_noisy(op: LinearCorruption, model: GaussianModel,
                               eps: float, n: float, m_hi: float = 1e18) -> float:
    """固定**充裕**噪声预算 n，二分求达到 ε 所需 clean 样本 m。

    这是 D2 的核心量（SFBD/OMNI 前提：噪声充裕、clean 稀缺）：
    κ 小 → 噪声单独够 → m=0；κ↑ → 病态方向噪声路线饱和 → m 上升；
    κ→∞ → 该方向完全由 clean 承担 → m 饱和（不是发散到 ∞，clean 把发散"封顶"）。
    """
    if achievable_error(op, model, n, 0.0) <= eps:
        return 0.0
    if achievable_error(op, model, n, m_hi) > eps:
        return np.inf  # 即便 m 很大也不够（一般不会发生，clean-only 总可达）
    lo, hi = 0.0, m_hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if achievable_error(op, model, n, mid) <= eps:
            hi = mid
        else:
            lo = mid
    return hi


def optimal_mix(op: LinearCorruption, model: GaussianModel, eps: float,
                cost_noisy: float = 1.0, cost_clean: float = 1.0,
                m_grid: int = 400):
    """在 cost·样本数 最小下，求达到 ε 的最优 (n*, m*)。

    扫 m∈[0, m_clean_only]，每个 m 反解所需 n，取总成本最小者。返回 dict。
    """
    m_max = required_clean_only(op, model, eps)
    ms = np.linspace(0.0, m_max, m_grid)
    best = None
    for m in ms:
        n = _required_noisy_given_clean(op, model, eps, m)
        if not np.isfinite(n):
            continue
        cost = cost_noisy * n + cost_clean * m
        if best is None or cost < best["cost"]:
            best = {"n": n, "m": m, "cost": cost}
    if best is None:  # 噪声完全不可达，纯 clean
        best = {"n": np.inf, "m": m_max, "cost": cost_clean * m_max}
    return best


__all__ = [
    "GaussianModel", "w2_gaussian_diag", "direction_precisions",
    "achievable_error", "required_noisy_only", "required_clean_only",
    "required_clean_given_noisy", "optimal_mix",
]
