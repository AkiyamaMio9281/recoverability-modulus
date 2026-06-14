"""由可恢复度模数 Ψ_I 预测样本预算（Phase 2/3）。

核心律（primer §3 + recover.py）：在方向 i 上，噪声路线的估计方差放大正比于
Ψ_I,i² = 1/σ_i²（Picard）。因此

    噪声预算   n_pred = Σ_i (s_i + σ_noise² · Ψ_I,i²) / ε²        （随 κ 发散）
    clean 预算 m_pred = Σ_i s_i / ε²                             （与 κ 无关，平坦）

n_pred 与 recover.required_noisy_only 在闭式下恒等——这里以 Ψ 语言重写，使
"Ψ 预测预算" 显式化（D3 再用采样验证 predict vs measured）。
"""

from __future__ import annotations

import numpy as np

from .operators import LinearCorruption
from .recover import GaussianModel, direction_precisions


def predict_budget(op: LinearCorruption, model: GaussianModel, target_eps: float) -> dict:
    """由 Ψ_I 谱预测达到目标误差 ε 的样本预算。

    Returns
    -------
    dict 含:
        n_noisy : 噪声-only 预算（Σ(s_i + σ_noise²·Ψ_i²)/ε²；存在零空间方向→inf）
        m_clean : clean-only 预算（Σ s_i / ε²；与 κ 无关）
        psi2    : 每方向 Ψ_I,i² = 1/σ_i²（零方向为 inf）
        amplification : σ_noise² · Σ Ψ_i²（噪声预算中的放大项，发散来源）
    """
    a, b, sig = direction_precisions(op, model)
    s = model.s
    sn2 = op.noise_std ** 2
    with np.errstate(divide="ignore"):
        psi2 = np.where(sig > 0, 1.0 / sig ** 2, np.inf)

    # 噪声-only：Σ 1/a_i / ε² = Σ (s_i + σ_noise²/σ_i²)/ε²
    if np.any(a <= 0):
        n_noisy = np.inf
    else:
        n_noisy = float(np.sum(1.0 / a) / target_eps ** 2)

    m_clean = float(np.sum(s) / target_eps ** 2)
    amplification = float(sn2 * np.sum(psi2[np.isfinite(psi2)])) if sn2 > 0 else 0.0

    return {
        "n_noisy": n_noisy,
        "m_clean": m_clean,
        "psi2": psi2,
        "amplification": amplification,
    }


__all__ = ["predict_budget"]
