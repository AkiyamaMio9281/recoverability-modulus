"""e4 (D5, 合成)：跨算子预测律 —— "是模数，不是腐蚀类型"。

主张：一个**单一、廉价的算子级代理**（可恢复度模数在等参考预算下的预测误差）跨四类腐蚀
（高斯/模糊/灰度/masking）预测真实恢复结果——不同腐蚀类型**坍缩到同一条曲线**。

代理（廉价、operator+budget-only，各向同性参考信号 s≡1）：
    proxy(op) = achievable_error(op, iso_model, n0, m0).
真实结果（贵、真各向异性模型 + 采样）：
    measured  = MC 恢复误差(op, true_model, n0, m0).

控制变量（隔离 Ψ 效应）：dim / n0 / m0 / 观测噪声 σ_obs / ε / 恢复方法 全部固定，**只变算子**。
≥3 seeds（变 true_model），报 Spearman ρ（整体 + 族内）。

跑法：python experiments/e4_cross_operator.py
产物：results/D5_cross_operator_synth.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from recmod import operators as ops, recover as rec
from recmod.operators import ZERO_TOL

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

# ---- 固定控制变量 ----
DIM = 10
SIGMA_OBS = 0.4      # 所有算子共同的观测噪声
N0 = 400             # 固定噪声预算
M0 = 60              # 固定 clean 预算
EPS = 0.5            # clean-budget 面板的目标误差
SEEDS = 4


def with_noise(op, sigma):
    op.noise_std = float(sigma)
    return op


def operator_zoo():
    """四族 × 多档强度，共同观测噪声 σ_obs（高斯族通过噪声本身变化）。"""
    zoo = []
    for s in [0.2, 0.4, 0.6, 0.8, 1.0]:
        zoo.append(("gaussian", f"σ={s}", with_noise(ops.gaussian_noise(DIM, s), s)))
    for std in [0.8, 1.2, 1.6, 2.0]:
        zoo.append(("blur", f"std={std}", with_noise(ops.blur(DIM, kernel_std=std), SIGMA_OBS)))
    for r in [8, 7, 6, 5]:
        zoo.append(("grayscale", f"r={r}", with_noise(ops.grayscale(DIM, rank=r), SIGMA_OBS)))
    for a in [0.2, 0.35, 0.5, 0.65]:
        zoo.append(("masking", f"α={a}", with_noise(ops.masking(DIM, alpha=a), SIGMA_OBS)))
    return zoo


def measure_error(op, model, n, m, trials=1500, seed=0):
    """MC：n 噪声(range, deconv) + m clean(全方向) 逆方差合并，返回 RMS clean-space 误差。"""
    rng = np.random.default_rng(seed)
    sig = op.singular_values()
    d = model.dim
    if sig.size < d:
        sig = np.concatenate([sig, np.zeros(d - sig.size)])
    rangemask = sig > ZERO_TOL * max(sig.max(), 1.0)
    s, mu, sn = model.s, model.mu, op.noise_std
    var_noisy = np.where(rangemask, (sig ** 2 * s + sn ** 2) /
                         np.where(rangemask, sig ** 2, 1.0) / n, np.inf)
    var_clean = s / m
    w = np.where(np.isfinite(var_noisy), (1 / var_noisy) / (1 / var_noisy + 1 / var_clean), 0.0)
    err2 = np.empty(trials)
    for k in range(trials):
        xn = mu + np.sqrt(s) * rng.standard_normal((n, d))
        yu = sig * xn + sn * rng.standard_normal((n, d))
        mu_noisy = np.where(rangemask, yu.mean(0) / np.where(sig > 0, sig, 1.0), 0.0)
        xc = mu + np.sqrt(s) * rng.standard_normal((m, d))
        mu_clean = xc.mean(0)
        mu_hat = w * mu_noisy + (1 - w) * mu_clean
        err2[k] = np.sum((mu_hat - mu) ** 2)
    return float(np.sqrt(err2.mean()))


def run():
    zoo = operator_zoo()
    iso = rec.GaussianModel(mu=np.zeros(DIM), s=np.ones(DIM))  # 各向同性参考

    rows = []  # (family, proxy, meas_err, m_budget)
    for seed in range(SEEDS):
        rng = np.random.default_rng(2000 + seed)
        s = np.sort(rng.uniform(0.5, 2.0, DIM))[::-1]
        mu = rng.standard_normal(DIM)
        model = rec.GaussianModel(mu=mu, s=s)
        for fam, _, op in zoo:
            proxy = rec.achievable_error(op, iso, N0, M0)          # 廉价代理
            meas = measure_error(op, model, N0, M0, seed=seed)     # 真实测量
            mbud = rec.required_clean_given_noisy(op, model, EPS, N0)  # clean 预算
            rows.append((fam, proxy, meas, mbud))

    fams = ["gaussian", "blur", "grayscale", "masking"]
    cmap = {f: plt.get_cmap("tab10")(i) for i, f in enumerate(fams)}

    proxy_all = np.array([r[1] for r in rows])
    meas_all = np.array([r[2] for r in rows])
    mbud_all = np.array([r[3] for r in rows])

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.4))

    # Panel A: proxy vs measured final error（坍缩）
    for f in fams:
        px = [r[1] for r in rows if r[0] == f]
        my = [r[2] for r in rows if r[0] == f]
        ax[0].scatter(px, my, s=30, color=cmap[f], label=f, alpha=0.8, edgecolor="k", lw=0.3)
    lo, hi = min(proxy_all.min(), meas_all.min()) * 0.8, max(proxy_all.max(), meas_all.max()) * 1.2
    ax[0].plot([lo, hi], [lo, hi], "k--", lw=1, label="$y=x$")
    rho_a, _ = spearmanr(proxy_all, meas_all)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("recoverability proxy (modulus-predicted error, isotropic)")
    ax[0].set_ylabel("measured recovery error (MC, anisotropic)")
    ax[0].set_title(f"D5: collapse across corruption types\nSpearman $\\rho$={rho_a:.4f} "
                    f"(it's the modulus, not the type)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, which="both")

    # Panel B: proxy vs clean budget m
    for f in fams:
        px = [r[1] for r in rows if r[0] == f]
        mb = [r[3] for r in rows if r[0] == f]
        ax[1].scatter(px, mb, s=30, color=cmap[f], label=f, alpha=0.8, edgecolor="k", lw=0.3)
    rho_b, _ = spearmanr(proxy_all, mbud_all)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("recoverability proxy (modulus-predicted error, isotropic)")
    ax[1].set_ylabel(f"clean budget $m$ to reach $\\epsilon$={EPS} (fixed $n_0$={N0})")
    ax[1].set_title(f"clean budget vs proxy, all families\nSpearman $\\rho$={rho_b:.4f}")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, which="both")

    fig.suptitle(f"Cross-operator recoverability law (synthetic)  "
                 f"dim={DIM}, $\\sigma_{{obs}}$={SIGMA_OBS}, $n_0$={N0}, $m_0$={M0}, {SEEDS} seeds")
    fig.tight_layout()
    fig.savefig(RESULTS / "D5_cross_operator_synth.png", dpi=130)

    # 族内 Spearman
    print(f"overall Spearman: error rho={rho_a:.4f}, clean-budget rho={rho_b:.4f}")
    for f in fams:
        px = [r[1] for r in rows if r[0] == f]
        my = [r[2] for r in rows if r[0] == f]
        rho, _ = spearmanr(px, my)
        print(f"  within {f:10s}: rho(proxy, error)={rho:.4f}")
    print("saved: results/D5_cross_operator_synth.png")
    return {"rho_error": rho_a, "rho_budget": rho_b}


if __name__ == "__main__":
    run()
