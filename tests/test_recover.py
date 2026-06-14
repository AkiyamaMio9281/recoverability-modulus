"""Phase 2 测试：闭式恢复 / 预算的关键性质 + Monte-Carlo 验证（prompt 03 检查点 C2）。"""

import numpy as np
import pytest

from recmod import operators as ops, recover as rec, budget as bud


DIM = 6


def _model(seed=0):
    rng = np.random.default_rng(seed)
    s = np.sort(rng.uniform(0.5, 2.0, DIM))[::-1]
    mu = rng.standard_normal(DIM)
    return rec.GaussianModel(mu=mu, s=s)


# -- 噪声预算随 κ 发散，clean 预算平坦 --------------------------------------
def test_noisy_budget_diverges_clean_flat():
    model = _model()
    base = ops.gaussian_noise(DIM, 0.5)
    eps = 0.3
    ts = [0.0, 0.3, 0.6, 0.9]
    n_noisy = [rec.required_noisy_only(ops.degrade(base, t, 1e-6), model, eps) for t in ts]
    m_clean = [rec.required_clean_only(ops.degrade(base, t, 1e-6), model, eps) for t in ts]
    assert all(b > a for a, b in zip(n_noisy, n_noisy[1:])), n_noisy   # 严格发散
    assert np.allclose(m_clean, m_clean[0], rtol=1e-9)                 # 完全平坦


def test_required_clean_given_noisy_rises_then_saturates():
    model = _model()
    base = ops.gaussian_noise(DIM, 0.5)
    eps, n0 = 0.3, 2000
    ts = np.linspace(0, 0.95, 20)
    m = [rec.required_clean_given_noisy(ops.degrade(base, t, 1e-7), model, eps, n0) for t in ts]
    assert m[0] == 0.0                       # 低 κ：噪声单独够
    assert m[-1] > 0.0                       # 高 κ：需 clean
    assert all(b >= a - 1e-6 for a, b in zip(m, m[1:]))  # 单调不减
    assert m[-1] < rec.required_clean_only(base, model, eps) + 1  # 被封顶（≤ clean-only）


# -- Ψ 预测 == 闭式恢复 ------------------------------------------------------
def test_psi_prediction_matches_recover():
    model = _model(3)
    op = ops.degrade(ops.gaussian_noise(DIM, 0.5), 0.5, 1e-6)
    eps = 0.3
    p = bud.predict_budget(op, model, eps)
    assert np.isclose(p["n_noisy"], rec.required_noisy_only(op, model, eps), rtol=1e-9)
    assert np.isclose(p["m_clean"], rec.required_clean_only(op, model, eps), rtol=1e-9)


# -- 闭式误差的 Monte-Carlo 验证 --------------------------------------------
def test_achievable_error_matches_monte_carlo():
    """采样 n 噪声 + m clean，逆方差合并估计 μ_x，经验误差应与 achievable_error 吻合。"""
    rng = np.random.default_rng(7)
    model = _model(1)
    op = ops.gaussian_noise(DIM, 0.6)            # A=I，便于直接验证
    n, m = 300, 80
    a, b, sig = rec.direction_precisions(op, model)
    closed = rec.achievable_error(op, model, n, m)

    trials = 4000
    errs = np.zeros(trials)
    for k in range(trials):
        # clean 样本：直接观测 x
        xc = model.mu + np.sqrt(model.s) * rng.standard_normal((m, DIM))
        mu_clean = xc.mean(0)                     # 每方向方差 s_i/m
        # 噪声样本：y = x + ε（A=I）
        xn = model.mu + np.sqrt(model.s) * rng.standard_normal((n, DIM))
        y = xn + op.noise_std * rng.standard_normal((n, DIM))
        mu_noisy = y.mean(0)                      # 每方向方差 (s_i+σ²)/n
        # 逆方差合并
        var_clean = model.s / m
        var_noisy = (model.s + op.noise_std ** 2) / n
        w = (1 / var_noisy) / (1 / var_noisy + 1 / var_clean)
        mu_hat = w * mu_noisy + (1 - w) * mu_clean
        errs[k] = np.sum((mu_hat - model.mu) ** 2)
    mc_err = np.sqrt(errs.mean())
    assert abs(mc_err - closed) / closed < 0.05   # 5% 内


# -- D3：Ψ 预测的样本复杂度常数 K² 与 MC 测量一致 ---------------------------
def test_d3_noisy_K2_prediction_matches_mc():
    """噪声路线：K²_pred = Σ 1/a_i 应与 MC 测得 K²=n·err² 吻合（Ψ 预测验证）。"""
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "experiments"))
    import e3_sample_complexity as e3

    model = _model(2)
    op = ops.degrade(ops.gaussian_noise(DIM, 0.5), 0.4, 1e-6)  # 病态但满秩
    a, _, _ = rec.direction_precisions(op, model)
    K2_pred = float(np.sum(1.0 / a[a > 0]))
    K2_meas = e3.measure_noisy_K2(op, model, n_probe=800, trials=4000, seed=0)
    assert abs(K2_meas - K2_pred) / K2_pred < 0.05


def test_d3_clean_null_K2_prediction_matches_mc():
    """clean 零空间路线：K²_pred = Σ_null s_i 应与 MC 吻合（O(1/√m) 验证）。"""
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "experiments"))
    import e3_sample_complexity as e3
    from recmod.operators import ZERO_TOL

    model = _model(2)
    op = ops.grayscale(DIM, rank=DIM // 2)
    sig = op.singular_values()
    if sig.size < DIM:
        sig = np.concatenate([sig, np.zeros(DIM - sig.size)])
    null = sig <= ZERO_TOL * max(sig.max(), 1.0)
    K2_pred = float(np.sum(model.s[null]))
    K2_meas = e3.measure_clean_null_K2(op, model, m_probe=400, trials=4000, seed=0)
    assert abs(K2_meas - K2_pred) / K2_pred < 0.05
