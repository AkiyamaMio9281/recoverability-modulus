"""理论辅助：KL→χ² 散度桥接常数（M4 §3）。

桥接引理（密度比有界 c≤m/q≤C 下）：
    χ²(m‖q) ≤ (1/β(c,C)) · KL(q‖m),
    β(c,C) = min_{t∈[c,C]} (−log t + t − 1)/(t−1)².

这是决策 A（χ²/L²）与 OMNI 的 KL(q‖·) 率之间的桥；C→∞ 时 β→0（桥接常数发散），故 BDR 必需。
"""

from __future__ import annotations

import numpy as np


def bridge_constant(c: float, C: float, grid: int = 200_000) -> float:
    """β(c,C) = min_{t∈[c,C]} (−log t + t − 1)/(t−1)²（t=1 处极限 1/2）。"""
    if not (0 < c <= 1 <= C):
        raise ValueError("need 0 < c <= 1 <= C (ratio bracket must contain 1)")
    ts = np.linspace(c, C, grid)
    ts = ts[np.abs(ts - 1) > 1e-7]
    g = -np.log(ts) + ts - 1.0
    return float(np.min(g / (ts - 1.0) ** 2))


def chi2_div(m: np.ndarray, q: np.ndarray, dx: float = 1.0) -> float:
    """χ²(m‖q) = ∫ (m−q)²/q（离散网格，间距 dx）。"""
    m, q = np.asarray(m), np.asarray(q)
    return float(np.sum((m - q) ** 2 / q) * dx)


def kl_div(p: np.ndarray, r: np.ndarray, dx: float = 1.0) -> float:
    """KL(p‖r) = ∫ p log(p/r)（离散网格，间距 dx）。"""
    p, r = np.asarray(p), np.asarray(r)
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / r[mask])) * dx)


__all__ = ["bridge_constant", "chi2_div", "kl_div"]
