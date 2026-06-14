"""Phase 0 测试：四类算子 + SVD + 伴随对偶 + 退化扫描（prompt 01 检查点 C0）。"""

import numpy as np
import pytest

from recmod import operators as ops


DIM = 16


# -- 形状 --------------------------------------------------------------------
@pytest.mark.parametrize("op", [
    ops.gaussian_noise(DIM, sigma=0.3),
    ops.blur(DIM, kernel_std=1.5),
    ops.grayscale(DIM, rank=8),
    ops.masking(DIM, alpha=0.4),
])
def test_apply_shape(op):
    x = np.ones(op.dim_in)
    y = op.apply(x, rng=np.random.default_rng(0))
    assert y.shape == (op.dim_out,)
    # batch
    X = np.ones((5, op.dim_in))
    Y = op.apply(X, rng=np.random.default_rng(0))
    assert Y.shape == (5, op.dim_out)


# -- SVD 重构 ----------------------------------------------------------------
@pytest.mark.parametrize("op", [
    ops.gaussian_noise(DIM, sigma=0.0),
    ops.blur(DIM, kernel_std=2.0),
    ops.grayscale(DIM, rank=10),
    ops.masking(DIM, alpha=0.5),
])
def test_svd_reconstruction(op):
    U, S, Vt = op.svd()
    A_rec = (U * S) @ Vt
    assert np.allclose(A_rec, op.A, atol=1e-10)


# -- 条件数 / 单射性 ---------------------------------------------------------
def test_injective_operators_have_finite_cond():
    # 加噪 (A=I) 与模糊 都是满秩 → 条件数有限
    assert np.isfinite(ops.gaussian_noise(DIM, 0.3).cond())
    assert np.isfinite(ops.blur(DIM, kernel_std=1.5).cond())
    assert ops.gaussian_noise(DIM, 0.3).is_injective()


def test_noninjective_operators_have_inf_cond():
    # 灰度 (低秩) 与 masking(α=1) 有零奇异值 → 条件数 inf、非单射
    assert ops.grayscale(DIM, rank=8).cond() == np.inf
    assert not ops.grayscale(DIM, rank=8).is_injective()
    full_mask = ops.masking(DIM, alpha=1.0)
    assert full_mask.cond() == np.inf
    assert not full_mask.is_injective()


def test_blur_is_illconditioned():
    # 模糊应比加噪病态得多（高频奇异值快衰减）
    assert ops.blur(DIM, kernel_std=2.5).cond() > ops.gaussian_noise(DIM, 0.3).cond()


# -- 伴随对偶检验 <Ax, ψ> = <x, Aᵀψ> ----------------------------------------
@pytest.mark.parametrize("op", [
    ops.gaussian_noise(DIM, sigma=0.0),  # 用无噪声版做确定性对偶检验
    ops.blur(DIM, kernel_std=1.5),
    ops.grayscale(DIM, rank=9),
    ops.masking(DIM, alpha=0.3),
])
def test_adjoint_duality(op):
    rng = np.random.default_rng(1)
    x = rng.standard_normal(op.dim_in)
    psi = rng.standard_normal(op.dim_out)
    Ax = op.apply(x)  # noise_std=0 时确定性
    lhs = Ax @ psi
    rhs = x @ op.adjoint(psi)
    assert np.isclose(lhs, rhs, atol=1e-10)


# -- 退化扫描 ----------------------------------------------------------------
def test_degrade_monotone_cond():
    base = ops.gaussian_noise(DIM, sigma=0.0)  # A=I，谱平
    ts = [0.0, 0.25, 0.5, 0.75, 1.0]
    conds = [ops.degrade(base, t, floor=1e-6).cond() for t in ts]
    # 条件数随 t 单调不减，且明显发散
    assert all(b >= a - 1e-9 for a, b in zip(conds, conds[1:]))
    assert conds[-1] > conds[0] * 100


def test_degrade_rejects_low_rank_base():
    with pytest.raises(ValueError):
        ops.degrade(ops.grayscale(DIM, rank=8), t=0.5)
