"""Phase 1 测试：Ψ_I 闭式、M1 退回、零空间、代理单调（prompt 02 检查点 C1）。"""

import numpy as np
import pytest
from scipy.stats import spearmanr

from recmod import operators as ops, modulus as mod


DIM = 64


def _norm_cos_mode(u, dim):
    grid = np.arange(dim)
    phi = np.cos(u * grid)
    if np.linalg.norm(phi) < 1e-9:
        phi = np.ones(dim)
    return phi / np.linalg.norm(phi)


# -- M1：高斯上 Ψ_I = exp(σ²u²/2) -------------------------------------------
@pytest.mark.parametrize("sigma", [0.3, 0.6, 1.0])
def test_gaussian_psi_matches_prop1(sigma):
    op = ops.gaussian_noise(DIM, sigma)
    freqs = ops.angular_frequencies(DIM)
    for k in [0, 1, 3, 7, 15, DIM // 4]:
        u = freqs[k]
        phi = _norm_cos_mode(u, DIM)
        num = mod.psi_closed_form(op, phi)
        ana = np.exp(0.5 * sigma ** 2 * u ** 2)
        assert abs(num - ana) / ana < 0.05  # 检查点要求 <5%（实际 ~1e-15）


def test_gaussian_slope_is_sigma_sq_half():
    sigma = 0.7
    op = ops.gaussian_noise(DIM, sigma)
    freqs = ops.angular_frequencies(DIM)
    ks = np.arange(0, DIM // 2)
    u = np.abs(freqs[ks])
    psi = np.array([mod.psi_closed_form(op, _norm_cos_mode(freqs[k], DIM)) for k in ks])
    slope, _ = np.polyfit(u ** 2, np.log(psi), 1)
    assert abs(slope - sigma ** 2 / 2) / (sigma ** 2 / 2) < 1e-3


# -- 零空间方向 → Ψ = inf ----------------------------------------------------
def test_nullspace_directions_give_inf():
    for op in [ops.grayscale(DIM, rank=DIM // 2), ops.masking(DIM, alpha=0.5)]:
        nmask = mod.nullspace_mask(op)
        assert nmask.any(), "该算子应有零空间"
        # 取一个零空间右奇异向量作为 φ
        _, _, Vt = np.linalg.svd(op.modulus_matrix(), full_matrices=True)
        phi = Vt[np.where(nmask)[0][0]]
        assert mod.psi_closed_form(op, phi) == np.inf


def test_range_directions_are_finite():
    op = ops.grayscale(DIM, rank=DIM // 2)
    _, _, Vt = np.linalg.svd(op.modulus_matrix(), full_matrices=True)
    nmask = mod.nullspace_mask(op)
    phi = Vt[np.where(~nmask)[0][0]]  # range 方向
    assert np.isfinite(mod.psi_closed_form(op, phi))


# -- 代理与闭式单调一致 ------------------------------------------------------
def test_proxy_monotone_with_closed_form():
    # degrade 阶梯是干净的单调病态旋钮（smin 单调 →0）；代理与闭式应同向单调且强相关。
    base = ops.gaussian_noise(DIM, 0.0)  # A=I，满秩谱平
    ts = [0.0, 0.2, 0.4, 0.6, 0.8]
    seq = [ops.degrade(base, t, floor=1e-8) for t in ts]
    phi = np.ones(DIM) / np.sqrt(DIM)  # 宽谱，覆盖收缩方向
    proxy = [mod.psi_proxy(o) for o in seq]
    closed = [mod.psi_closed_form(o, phi) for o in seq]
    assert all(b >= a for a, b in zip(proxy, proxy[1:])), proxy
    assert all(b >= a for a, b in zip(closed, closed[1:])), closed
    rho, _ = spearmanr(proxy, closed)
    assert rho > 0.9


def test_psi_tier_labels():
    assert mod.psi_tier(2.0) == "finite-mild"
    assert mod.psi_tier(1e4) == "finite-huge"
    assert mod.psi_tier(np.inf).startswith("∞")
