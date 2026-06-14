"""可恢复度模数 Ψ_I（无权 L²，primer §3）与经验代理。

记号：本仓库的 `Ψ` 默认指**无权** `Ψ_I`，闭式由腐蚀算子的 modulus 矩阵 M（见
operators.modulus_matrix）的 SVD 给出：

    Ψ_I(φ)² = Σ_i ⟨φ, v_i⟩² / σ_i²        （Picard 条件）

其中 σ_i, v_i 为 M = U Σ Vᵀ 的奇异值与右奇异向量。落在 σ_i=0 方向上的 φ（即 T_r 的
零空间方向）给 Ψ_I=∞，正是需要 clean 先验的方向。

q-加权 `Ψ_q`（理论 D6 用）需广义 SVD，本阶段不实现（primer §3，列为问 Yu）。
"""

from __future__ import annotations

import numpy as np

from .operators import LinearCorruption, ZERO_TOL


def _modulus_svd(op: LinearCorruption) -> tuple[np.ndarray, np.ndarray]:
    """返回 modulus 矩阵 M 的 (S, Vt)：奇异值与右奇异向量（行）。

    用 full_matrices=True 以保留完整右奇异基（含零空间方向），便于零空间检测。
    """
    M = op.modulus_matrix()
    _, S, Vt = np.linalg.svd(M, full_matrices=True)
    # Vt 行数 = dim_in；S 长度 = min(m,n)，零空间方向（多出的 Vt 行）补 0 奇异值。
    n = Vt.shape[0]
    if S.size < n:
        S = np.concatenate([S, np.zeros(n - S.size)])
    return S, Vt


def _zero_threshold(S: np.ndarray) -> float:
    smax = S.max() if S.size else 1.0
    return ZERO_TOL * max(smax, 1.0)


def psi_closed_form(op: LinearCorruption, phi: np.ndarray) -> float:
    """无权可恢复度模数 Ψ_I(φ)（primer §3 的 Picard 闭式）。

    Parameters
    ----------
    op : LinearCorruption
    phi : (dim_in,) ndarray
        clean-space 测试函数（线性测试函数的系数向量 / 信号）。

    Returns
    -------
    float
        Ψ_I(φ)。若 φ 在某零奇异值方向有非零分量 → np.inf。
    """
    S, Vt = _modulus_svd(op)
    phi = np.asarray(phi, dtype=float)
    coeffs = Vt @ phi  # ⟨φ, v_i⟩
    tol = _zero_threshold(S)
    zero = S <= tol
    phi_scale = max(1.0, float(np.linalg.norm(phi)))
    if np.any(np.abs(coeffs[zero]) > ZERO_TOL * phi_scale):
        return np.inf  # φ 触及零空间方向 → 不可恢复
    nz = ~zero
    return float(np.sqrt(np.sum((coeffs[nz] ** 2) / (S[nz] ** 2))))


def nullspace_mask(op: LinearCorruption) -> np.ndarray:
    """布尔数组：标出 modulus 矩阵的零奇异值方向（Ψ_I=∞ 的方向）。"""
    S, _ = _modulus_svd(op)
    return S <= _zero_threshold(S)


def psi_proxy(op: LinearCorruption) -> float:
    """经验条件数代理（供 §05 跨算子用）：非零谱上的 RMS 放大倍数

        proxy = sqrt( mean_{σ_i>0} 1/σ_i² ).

    只在 range 方向上度量放大（排除零空间，那部分由 clean 先验单独处理）。随病态程度
    单调增大；不必等于 `psi_closed_form`，只需与之单调（docstring 约定）。
    """
    S, _ = _modulus_svd(op)
    nz = S > _zero_threshold(S)
    if not np.any(nz):
        return np.inf
    return float(np.sqrt(np.mean(1.0 / S[nz] ** 2)))


def recoverability_index(op: LinearCorruption, lam: float) -> float:
    """Tikhonov 正则化放大指数（跨算子统一代理，primer §4.2）：

        R(op; λ) = sqrt( Σ_i 1/(σ_i² + λ) ).

    与 psi_proxy 不同，它对零空间方向**不排除**而是**封顶**到 1/λ（对应 augmented-KLAP
    的 σ_i/(σ_i²+λ) 行为），故 injective 与 non-injective 算子可放在同一把尺子上比较。
    纯算子谱量（给定 λ），不依赖样本——D5 跨算子律用它的 noise-加权版（见 e4）。
    """
    sig = op.singular_values()
    return float(np.sqrt(np.sum(1.0 / (sig ** 2 + lam))))


def psi_tier(value: float, mild: float = 1e1, huge: float = 1e6) -> str:
    """把 Ψ 值分成三档（primer §3.2）：有限温和 / 有限巨大 / 无穷。"""
    if not np.isfinite(value):
        return "∞ (need clean prior)"
    if value <= mild:
        return "finite-mild"
    if value <= huge:
        return "finite-huge"
    return "finite-extreme"


__all__ = ["psi_closed_form", "nullspace_mask", "psi_proxy",
           "recoverability_index", "psi_tier"]
