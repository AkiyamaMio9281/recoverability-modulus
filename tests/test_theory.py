"""Phase 6 测试：KL→χ² 桥接引理（M4 §3）。"""

import numpy as np
import pytest

from recmod import theory as th


def test_bridge_constant_shrinks_with_wider_ratio():
    # C 越大 β 越小（桥接常数 1/β 发散）→ BDR 必需
    b1 = th.bridge_constant(0.5, 2.0)
    b2 = th.bridge_constant(0.2, 5.0)
    b3 = th.bridge_constant(0.1, 10.0)
    assert b1 > b2 > b3 > 0
    assert abs(b1 - 0.307) < 0.02   # 与 note 记录值一致


def test_bridge_inequality_holds():
    """随机有界比高斯对：χ²(m‖q) ≤ KL(q‖m)/β 全成立。"""
    rng = np.random.default_rng(0)
    grid = np.linspace(-5, 5, 400)
    dx = grid[1] - grid[0]

    def normal(mu, sd):
        p = np.exp(-0.5 * ((grid - mu) / sd) ** 2)
        return p / (p.sum() * dx)

    worst = 0.0
    n_checked = 0
    for _ in range(1500):
        q = normal(rng.uniform(-0.5, 0.5), rng.uniform(0.8, 1.5))
        m = normal(rng.uniform(-0.5, 0.5), rng.uniform(0.8, 1.5))
        t = m / q
        c, C = t.min(), t.max()
        if C > 10 or c < 0.1:
            continue
        chi2 = th.chi2_div(m, q, dx)
        kl = th.kl_div(q, m, dx)
        if kl < 1e-10:
            continue
        beta = th.bridge_constant(max(c, 0.05), min(C, 12))
        ratio = chi2 / (kl / beta)
        worst = max(worst, ratio)
        n_checked += 1
    assert n_checked > 100
    assert worst <= 1.0 + 1e-6   # 引理成立
