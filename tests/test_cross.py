"""Phase 4 测试：跨算子代理与测量（prompt 05 检查点 C4，快速版）。"""

import sys
import pathlib

import numpy as np
import pytest
from scipy.stats import spearmanr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "experiments"))

from recmod import operators as ops, modulus as mod, recover as rec
import e4_cross_operator as e4


def test_recoverability_index_monotone_and_caps_nullspace():
    # 病态加重 → 指数增大
    base = ops.gaussian_noise(8, 0.0)
    vals = [mod.recoverability_index(ops.degrade(base, t, 1e-7), lam=1e-3)
            for t in [0.0, 0.3, 0.6, 0.9]]
    assert all(b >= a for a, b in zip(vals, vals[1:]))
    # 非单射算子有限（被 λ 封顶），不是 inf
    assert np.isfinite(mod.recoverability_index(ops.grayscale(8, rank=4), lam=1e-3))


def test_measure_error_matches_closed_form_isotropic():
    # 各向同性下 MC 测量应与 achievable_error 闭式吻合
    model = rec.GaussianModel(mu=np.zeros(8), s=np.ones(8))
    op = ops.gaussian_noise(8, 0.4)
    closed = rec.achievable_error(op, model, 300, 50)
    meas = e4.measure_error(op, model, 300, 50, trials=3000, seed=0)
    assert abs(meas - closed) / closed < 0.05


def test_cross_operator_collapse_fast():
    """缩减版跨算子：代理与测量误差应强相关（坍缩）。"""
    rng = np.random.default_rng(0)
    s = np.sort(rng.uniform(0.5, 2.0, 8))[::-1]
    model = rec.GaussianModel(mu=rng.standard_normal(8), s=s)
    iso = rec.GaussianModel(mu=np.zeros(8), s=np.ones(8))
    zoo = [
        e4.with_noise(ops.gaussian_noise(8, 0.3), 0.3),
        e4.with_noise(ops.gaussian_noise(8, 0.8), 0.8),
        e4.with_noise(ops.blur(8, kernel_std=1.5), 0.4),
        e4.with_noise(ops.grayscale(8, rank=5), 0.4),
        e4.with_noise(ops.masking(8, alpha=0.5), 0.4),
    ]
    proxy = [rec.achievable_error(op, iso, 400, 60) for op in zoo]
    meas = [e4.measure_error(op, model, 400, 60, trials=800, seed=1) for op in zoo]
    rho, _ = spearmanr(proxy, meas)
    assert rho > 0.85
