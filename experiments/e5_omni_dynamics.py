"""e5 (D8): SFBD-OMNI 真实算法动力学 —— Ψ 控制 clean 空间收敛。

不再是理想估计量:这里跑**真实 OMNI 交替最小化**(闭式高斯, src/recmod/omni.py),展示
"Identifiability ≠ Recoverability" 在算法**自己的迭代**里发生:

  - 腐蚀空间 KL(q‖T_r p^k) 对所有条件数都收敛(算法"在腐蚀空间看起来成功");
  - clean 空间 W2(p^k, p_data) 随条件数 κ(=Ψ) 增大而**变慢/停滞**;
  - 即:ill-posed 时 OMNI 在腐蚀空间已"收敛"、clean 空间却离得很远 —— gap 由 Ψ 决定。

跑法: python experiments/e5_omni_dynamics.py
产物: results/D8_omni_dynamics.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from recmod import operators as ops, omni

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)


def make_operator(Q, kappa, sigma_noise):
    """条件数 = kappa 的线性算子 A = Q diag(sv) Qᵀ(sv 从 1 到 1/kappa),加噪 σ。"""
    d = Q.shape[0]
    sv = np.linspace(1.0, 1.0 / kappa, d)
    A = Q @ np.diag(sv) @ Q.T
    return ops.LinearCorruption(A, noise_std=sigma_noise, name=f"kappa={kappa:g}")


def run(d=5, sigma_noise=0.3, n_iter=50, kappas=(1, 5, 25, 125, 625), seeds=5):
    kl = {k: [] for k in kappas}
    w2 = {k: [] for k in kappas}
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        Sigma = Q @ np.diag(rng.uniform(0.5, 2.0, d)) @ Q.T
        mu = rng.standard_normal(d)
        for k in kappas:
            op = make_operator(Q, k, sigma_noise)
            tr = omni.run_omni(op, mu, Sigma, n_iter=n_iter)
            kl[k].append(tr["kl_corrupted"])
            w2[k].append(tr["w2_clean"])
    kl = {k: np.array(v) for k, v in kl.items()}
    w2 = {k: np.array(v) for k, v in w2.items()}

    iters = np.arange(1, n_iter + 1)
    cmap = plt.get_cmap("viridis")
    colors = {k: cmap(i / (len(kappas) - 1)) for i, k in enumerate(kappas)}

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

    # Panel A: corrupted-space KL vs iter (all converge)
    for k in kappas:
        med = np.median(kl[k], 0)
        ax[0].semilogy(iters, med + 1e-18, "-o", ms=3, color=colors[k], label=f"$\\kappa$={k}")
    ax[0].set_xlabel("OMNI iteration $k$")
    ax[0].set_ylabel(r"corrupted-space $\mathrm{KL}(q\,\|\,T_r p^k)$")
    ax[0].set_title("A. In corrupted space: OMNI converges\n(looks successful for all $\\kappa$)")
    ax[0].legend(fontsize=8, title="$\\kappa=\\Psi$"); ax[0].grid(alpha=0.3, which="both")

    # Panel B: clean-space W2 vs iter (degrades with kappa)
    for k in kappas:
        med = np.median(w2[k], 0)
        ax[1].semilogy(iters, med + 1e-18, "-o", ms=3, color=colors[k], label=f"$\\kappa$={k}")
    ax[1].set_xlabel("OMNI iteration $k$")
    ax[1].set_ylabel(r"clean-space $W_2(p^k, p_{\mathrm{data}})$")
    ax[1].set_title("B. In clean space: convergence stalls\nas $\\kappa=\\Psi$ grows")
    ax[1].legend(fontsize=8, title="$\\kappa=\\Psi$"); ax[1].grid(alpha=0.3, which="both")

    # Panel C: the gap at final iterate vs kappa
    ks = np.array(kappas, float)
    kl_fin = np.array([np.median(kl[k][:, -1]) for k in kappas])
    w2_fin = np.array([np.median(w2[k][:, -1]) for k in kappas])
    ax[2].loglog(ks, w2_fin, "s-", color="tab:red", label=r"clean $W_2$ (final)")
    ax[2].loglog(ks, kl_fin, "o--", color="tab:blue", label=r"corrupted KL (final)")
    ax[2].set_xlabel("condition number $\\kappa = \\Psi$")
    ax[2].set_ylabel("error at final iterate")
    ax[2].set_title("C. The gap grows with $\\Psi$\n(corrupted small, clean large)")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3, which="both")

    fig.suptitle(f"SFBD-OMNI dynamics (closed-form, $d$={d}, $\\sigma$={sigma_noise}, "
                 f"{seeds} seeds): $\\Psi$ controls clean-space recoverability  —  "
                 "Identifiability $\\neq$ Recoverability, live in the algorithm")
    fig.tight_layout()
    fig.savefig(RESULTS / "D8_omni_dynamics.png", dpi=130)

    print("final iterate (median over seeds):")
    for k in kappas:
        print(f"  kappa={k:>4}: corrupted KL={kl_fin[list(kappas).index(k)]:.2e}  "
              f"clean W2={w2_fin[list(kappas).index(k)]:.2e}")
    print("saved: results/D8_omni_dynamics.png")
    return {"kl_fin": kl_fin, "w2_fin": w2_fin, "kappas": ks}


if __name__ == "__main__":
    run()
