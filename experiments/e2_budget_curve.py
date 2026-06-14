"""e2 (D2, 主图)：clean 预算 vs 条件数 的发散/封顶曲线。

一张图同时展示三件事（primer §0 / proposal §9）：
  1) 可恢复性是**连续**的（预算随 κ 平滑变化）；
  2) **Ψ 预测预算**（噪声预算发散率 ∝ Ψ²=κ²，与解析渐近线吻合）；
  3) **低秩 → 需要 clean**（噪声-only 预算发散；固定充裕噪声下所需 clean 预算从 0 升起、
     并被 clean 封顶——对应 OMNI "少量 clean 即可"的经验）。

设定：clean = 各向异性高斯；腐蚀 = 线性 A（从满秩 degrade 到近奇异）+ 加性噪声 σ_noise。
全程闭式（recover.py / budget.py），Ψ 指无权 Ψ_I。≥3 seeds，mean±std。

跑法：python experiments/e2_budget_curve.py
产物：results/D2_budget_curve.png / .pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from recmod import operators as ops, recover as rec

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)


def run(dim: int = 8, sigma_noise: float = 0.5, eps: float = 0.3,
        n0: int = 2000, seeds: int = 5, floor: float = 1e-7) -> dict:
    ts = np.linspace(0.0, 0.95, 40)
    base = ops.gaussian_noise(dim, sigma_noise)
    kappa = np.array([ops.degrade(base, t, floor).cond() for t in ts])

    m_req = np.full((seeds, len(ts)), np.nan)     # 固定 n0 下所需 clean
    n_noisy = np.full((seeds, len(ts)), np.nan)   # 噪声-only 预算（发散）

    for si in range(seeds):
        rng = np.random.default_rng(si)
        s = np.sort(rng.uniform(0.5, 2.0, dim))[::-1]  # 各向异性方差
        mu = rng.standard_normal(dim)
        model = rec.GaussianModel(mu=mu, s=s)
        for ti, t in enumerate(ts):
            op = ops.degrade(base, t, floor)
            m_req[si, ti] = rec.required_clean_given_noisy(op, model, eps, n0)
            n_noisy[si, ti] = rec.required_noisy_only(op, model, eps)

    m_mean, m_std = m_req.mean(0), m_req.std(0)
    n_mean, n_std = n_noisy.mean(0), n_noisy.std(0)

    # 相位边界（基于均值曲线）
    pos = m_mean > 1e-6
    k1 = kappa[pos][0] if pos.any() else kappa[-1]                 # clean 开始需要
    plateau = m_mean[-1]
    near = m_mean > 0.99 * plateau
    k2 = kappa[near][0] if near.any() else kappa[-1]              # clean 饱和

    # Ψ² 渐近线（噪声预算 ∝ σ_noise²·κ²/ε²，σ_max=1）
    psi_asym = (np.sum([1]) * 0 + sigma_noise ** 2 * kappa ** 2) / eps ** 2

    # ---- plot (English labels: robust fonts + shareable with Yu) ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))

    # Panel A: required clean budget at fixed abundant noisy budget
    a = ax[0]
    a.axvspan(kappa[0], k1, color="tab:green", alpha=0.08)
    a.axvspan(k1, k2, color="tab:orange", alpha=0.08)
    a.axvspan(k2, kappa[-1], color="tab:red", alpha=0.08)
    a.plot(kappa, m_mean, "o-", ms=4, color="tab:blue",
           label=f"clean budget $m$ (fixed $n_0$={n0})")
    a.fill_between(kappa, m_mean - m_std, m_mean + m_std, color="tab:blue", alpha=0.2)
    a.axhline(plateau, ls="--", color="gray", lw=1, label=f"clean cap $\\approx${plateau:.1f}")
    a.set_xscale("log")
    a.set_xlabel("condition number $\\kappa(A)$")
    a.set_ylabel("clean samples $m$ to reach target $\\epsilon$")
    a.set_title("D2: clean budget vs condition number")
    a.text(np.sqrt(kappa[0] * k1), plateau * 0.55,
           "recoverable from\ncorrupted data\n($m$=0)", ha="center", va="center", fontsize=8)
    a.text(np.sqrt(k2 * kappa[-1]), plateau * 0.45, "low-rank\n$\\to$ need clean\n(capped)",
           ha="center", va="center", fontsize=8)
    a.legend(loc="lower right", fontsize=9)
    a.grid(alpha=0.3, which="both")

    # Panel B: noisy-only budget diverges + Psi^2 asymptote
    b = ax[1]
    b.loglog(kappa, n_mean, "s-", ms=4, color="tab:red", label="noisy-only budget $n$ ($m$=0)")
    b.fill_between(kappa, np.maximum(n_mean - n_std, 1e-9), n_mean + n_std,
                   color="tab:red", alpha=0.2)
    b.loglog(kappa, psi_asym + n_mean[0], "--", color="black", lw=1.2,
             label="$\\Psi^2$ asymptote $\\propto \\sigma^2\\kappa^2/\\epsilon^2$")
    b.set_xlabel("condition number $\\kappa(A)$")
    b.set_ylabel("noisy samples $n$ to reach target $\\epsilon$")
    b.set_title("noisy-only budget diverges with $\\kappa$ ($\\Psi$-driven)")
    b.legend(loc="upper left", fontsize=9)
    b.grid(alpha=0.3, which="both")

    fig.suptitle(f"Recoverability modulus vs clean budget  (dim={dim}, "
                 f"$\\sigma$={sigma_noise}, $\\epsilon$={eps}, {seeds} seeds)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(RESULTS / f"D2_budget_curve.{ext}", dpi=130)

    print(f"phases: κ1(clean 开始需要)={k1:.3g}, κ2(clean 饱和)={k2:.3g}, "
          f"clean 封顶={plateau:.2f}")
    print(f"noisy-only 预算: κ={kappa[0]:.1f}→{n_mean[0]:.0f}, "
          f"κ={kappa[-1]:.2g}→{n_mean[-1]:.3g}  (发散 ∝ κ²)")
    print(f"saved: {RESULTS/'D2_budget_curve.png'} (+ .pdf)")
    return {"kappa": kappa, "m_mean": m_mean, "n_mean": n_mean,
            "k1": float(k1), "k2": float(k2), "plateau": float(plateau)}


if __name__ == "__main__":
    run()
