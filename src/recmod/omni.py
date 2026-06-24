"""SFBD-OMNI 交替最小化在高斯-线性设定下的**闭式**动力学（D8）。

OMNI 更新（γ=1, λ=0）对高斯 `p^k=N(m_k,S_k)` 与线性+高斯腐蚀 `y=Ax+ε, ε~N(0,Γ)` 保持高斯，
每步闭式（推导见 notes/M6）：

    K_k     = S_k Aᵀ (A S_k Aᵀ + Γ)^{-1}          # 卡尔曼增益式
    m_{k+1} = m_k + K_k (m_q − A m_k)
    S_{k+1} = K_k S_q K_kᵀ + (I − K_k A) S_k

其中 q = T_r p_data = N(m_q, S_q), m_q = Aμ, S_q = AΣAᵀ + Γ。不动点 = p_data（injective A）。

用途：把"理想估计量"换成**真实算法**，验证 Ψ(=条件数) 控制 clean 空间的收敛——
即 Identifiability ≠ Recoverability 在 OMNI 自己的迭代里发生。
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from .operators import LinearCorruption


def kl_gaussian(m0, S0, m1, S1) -> float:
    """KL( N(m0,S0) ‖ N(m1,S1) )，闭式。"""
    d = len(m0)
    S1i = np.linalg.inv(S1)
    dm = m1 - m0
    sign, logdet1 = np.linalg.slogdet(S1)
    _, logdet0 = np.linalg.slogdet(S0)
    return float(0.5 * (np.trace(S1i @ S0) + dm @ S1i @ dm - d + logdet1 - logdet0))


def w2_gaussian(m0, S0, m1, S1) -> float:
    """W2( N(m0,S0), N(m1,S1) )（Bures），闭式。"""
    s0 = sla.sqrtm(S0)
    inner = sla.sqrtm(s0 @ S1 @ s0)
    bures = np.trace(S0 + S1 - 2.0 * inner.real)
    return float(np.sqrt(max(np.sum((m0 - m1) ** 2) + bures, 0.0)))


def omni_step(m, S, A, Gamma, mq, Sq):
    """一步闭式 OMNI 更新，返回 (m', S')。"""
    inn = A @ S @ A.T + Gamma
    K = S @ A.T @ np.linalg.inv(inn)
    m_new = m + K @ (mq - A @ m)
    I = np.eye(S.shape[0])
    S_new = K @ Sq @ K.T + (I - K @ A) @ S
    S_new = 0.5 * (S_new + S_new.T)  # 数值对称化
    return m_new, S_new


def run_omni(op: LinearCorruption, mu, Sigma, n_iter=60, m0=None, S0=None):
    """跑闭式 OMNI，返回轨迹 dict：每步 corrupted-space KL(q‖T_r p^k) 与 clean-space W2。

    Parameters
    ----------
    op : LinearCorruption  (用 op.A 作 A，op.noise_std 作噪声)
    mu, Sigma : p_data 的均值与（完整）协方差
    m0, S0 : p^0 初始化（默认 N(0, I)）
    """
    A = op.A
    d = A.shape[1]
    Gamma = (op.noise_std ** 2) * np.eye(A.shape[0])
    mq = A @ mu
    Sq = A @ Sigma @ A.T + Gamma
    m = np.zeros(d) if m0 is None else np.asarray(m0, float)
    S = np.eye(d) if S0 is None else np.asarray(S0, float)

    kl_corr, w2_clean = [], []
    for _ in range(n_iter):
        m, S = omni_step(m, S, A, Gamma, mq, Sq)
        mc, Sc = A @ m, A @ S @ A.T + Gamma          # T_r p^k
        kl_corr.append(kl_gaussian(mq, Sq, mc, Sc))   # KL(q ‖ T_r p^k)
        w2_clean.append(w2_gaussian(m, S, mu, Sigma))  # clean-space error
    return {"kl_corrupted": np.array(kl_corr), "w2_clean": np.array(w2_clean),
            "m_final": m, "S_final": S}


__all__ = ["kl_gaussian", "w2_gaussian", "omni_step", "run_omni"]
