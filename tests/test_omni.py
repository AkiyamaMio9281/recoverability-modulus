"""D8 测试:闭式 OMNI 动力学(prompt 外的新进展)。"""

import numpy as np
import pytest

from recmod import operators as ops, omni


def _model(d=4, seed=0):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    Sigma = Q @ np.diag(rng.uniform(0.5, 2.0, d)) @ Q.T
    mu = rng.standard_normal(d)
    return mu, Sigma, Q


def test_kl_w2_self_zero():
    m = np.array([1.0, -2.0]); S = np.array([[2.0, 0.3], [0.3, 1.0]])
    assert abs(omni.kl_gaussian(m, S, m, S)) < 1e-10
    assert omni.w2_gaussian(m, S, m, S) < 1e-7


def test_pdata_is_fixed_point():
    mu, Sigma, _ = _model()
    op = ops.gaussian_noise(len(mu), 0.4)            # A=I
    A = op.A; Gamma = op.noise_std ** 2 * np.eye(len(mu))
    mq, Sq = A @ mu, A @ Sigma @ A.T + Gamma
    m1, S1 = omni.omni_step(mu.copy(), Sigma.copy(), A, Gamma, mq, Sq)
    assert np.allclose(m1, mu, atol=1e-8)
    assert np.allclose(S1, Sigma, atol=1e-8)


def test_wellconditioned_converges_to_pdata():
    mu, Sigma, _ = _model()
    op = ops.gaussian_noise(len(mu), 0.3)            # A=I, well-conditioned
    tr = omni.run_omni(op, mu, Sigma, n_iter=40)
    assert tr["kl_corrupted"][-1] < 1e-8
    assert tr["w2_clean"][-1] < 1e-4


def test_gap_grows_with_conditioning():
    """核心发现:final clean W2 随条件数单调增(腐蚀空间却看起来收敛)。"""
    d = 5
    mu, Sigma, Q = _model(d, seed=1)
    w2_fin = []
    for kappa in [1.0, 10.0, 100.0]:
        sv = np.linspace(1.0, 1.0 / kappa, d)
        A = Q @ np.diag(sv) @ Q.T
        op = ops.LinearCorruption(A, noise_std=0.3)
        tr = omni.run_omni(op, mu, Sigma, n_iter=50)
        w2_fin.append(tr["w2_clean"][-1])
    assert w2_fin[0] < w2_fin[1] < w2_fin[2]          # gap 单调增
    assert w2_fin[0] < 1e-3                            # 良态几乎完美
    assert w2_fin[2] > 0.05                            # 病态明显失败
