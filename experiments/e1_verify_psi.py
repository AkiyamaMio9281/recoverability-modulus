"""e1 (D1 / M1)：数值确认 Ψ_I(e_u) = exp(σ²‖u‖²/2)，即退回 SFBD (ICML'25) Prop 1。

跑法：
    python experiments/e1_verify_psi.py
产物：
    results/D1_M1_prop1_recovery.png   —— log Ψ vs u²，应是斜率 σ²/2 的直线
    stdout：每个频率的数值/解析对照 + 拟合斜率 vs σ²/2

记号：Ψ 指无权 Ψ_I（primer §3）。频率约定 u_k = 2π·fftfreq(dim)（operators.angular_frequencies），
与算子 modulus 一致，slope 才对得上。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from recmod import operators as ops, modulus as mod

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)


def fourier_mode(u: float, dim: int) -> np.ndarray:
    """归一化实余弦模 cos(u·grid)（u=0 时取常数模）。它是 C_ν 的特征向量。"""
    grid = np.arange(dim)
    phi = np.cos(u * grid)
    nrm = np.linalg.norm(phi)
    if nrm < 1e-9:
        phi = np.ones(dim)
        nrm = np.linalg.norm(phi)
    return phi / nrm


def run(dim: int = 128, sigma: float = 0.5) -> dict:
    op = ops.gaussian_noise(dim, sigma)
    freqs = ops.angular_frequencies(dim)
    ks = np.arange(0, dim // 2 + 1)  # 0 .. Nyquist

    u_vals, psi_num, psi_ana = [], [], []
    for k in ks:
        u = float(freqs[k]) if k < dim // 2 else float(np.pi)  # Nyquist
        phi = fourier_mode(u, dim)
        u_vals.append(abs(u))
        psi_num.append(mod.psi_closed_form(op, phi))
        psi_ana.append(np.exp(0.5 * sigma ** 2 * u ** 2))

    u_vals = np.array(u_vals)
    psi_num = np.array(psi_num)
    psi_ana = np.array(psi_ana)
    rel_err = np.abs(psi_num - psi_ana) / psi_ana

    # 拟合 log Ψ vs u²，斜率应 = σ²/2
    slope, intercept = np.polyfit(u_vals ** 2, np.log(psi_num), 1)

    # 报告
    print(f"M1 退回 SFBD (ICML'25) Prop 1 — dim={dim}, sigma={sigma}")
    print(f"{'k':>4} {'u':>9} {'Psi_num':>12} {'Psi_ana':>12} {'rel_err':>10}")
    for i in range(0, len(ks), max(1, len(ks) // 12)):
        print(f"{ks[i]:>4} {u_vals[i]:>9.4f} {psi_num[i]:>12.6g} "
              f"{psi_ana[i]:>12.6g} {rel_err[i]:>10.2e}")
    print(f"\nmax rel_err = {rel_err.max():.2e}")
    print(f"fitted slope (log Psi vs u^2) = {slope:.6f}   |   sigma^2/2 = {sigma**2/2:.6f}")
    print(f"slope rel_err = {abs(slope - sigma**2/2)/(sigma**2/2):.2e}")

    # 图
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(u_vals ** 2, np.log(psi_num), "o", ms=4, label="numeric $\\Psi_I$")
    ax[0].plot(u_vals ** 2, 0.5 * sigma ** 2 * u_vals ** 2, "-",
               label=f"analytic $\\sigma^2 u^2/2$ (slope={sigma**2/2:.3f})")
    ax[0].set_xlabel("$u^2$")
    ax[0].set_ylabel("$\\log \\Psi_I(e_u)$")
    ax[0].set_title("M1: $\\Psi_I(e_u)=\\exp(\\sigma^2 u^2/2)$ (SFBD Prop 1)")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].semilogy(u_vals, rel_err + 1e-18, "o-", ms=4)
    ax[1].set_xlabel("$u$")
    ax[1].set_ylabel("relative error")
    ax[1].set_title("numeric vs analytic (machine precision)")
    ax[1].grid(alpha=0.3, which="both")

    fig.suptitle(f"D1 / M1  (dim={dim}, $\\sigma$={sigma})")
    fig.tight_layout()
    out = RESULTS / "D1_M1_prop1_recovery.png"
    fig.savefig(out, dpi=130)
    print(f"\nsaved: {out}")

    return {"max_rel_err": float(rel_err.max()), "slope": float(slope),
            "slope_target": sigma ** 2 / 2}


if __name__ == "__main__":
    run()
