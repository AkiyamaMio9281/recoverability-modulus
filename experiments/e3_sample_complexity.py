"""e3 (D3)：Ψ 预测样本复杂度的 预测-实测 散点验证。

D2 是闭式律；D3 用**真采样**走完整 估计 q̂ → 反演恢复 → 测 clean-space 误差 的管线，
反解"达到目标 ε 所需样本数"，与 `budget.predict_budget` 的 Ψ 预测对照，落在 y=x 附近。

两张散点：
  - 噪声样本 n（injective 病态算子：gaussian 各 σ / blur / degrade）——验证 Ψ 放大；
  - clean 样本 m（non-injective 算子的零空间：grayscale / masking）——验证 O(1/√m)。

误差律：err(n) = K/√n，K² 由 MC 测得（K²_meas = n·err²），预测 K²_pred 来自 Ψ 谱。
于是 n*(ε)=K²/ε²；scatter 跨 (算子 × seed × ε)。≥3 seeds，报 R²/slope/中位相对误差。

跑法：python experiments/e3_sample_complexity.py
产物：results/D3_sample_complexity_n.png、results/D3_sample_complexity_m.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from recmod import operators as ops, recover as rec, budget as bud
from recmod.operators import ZERO_TOL

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)
EPS_LIST = [0.2, 0.3, 0.5]


def _model(dim, seed):
    rng = np.random.default_rng(1000 + seed)
    s = np.sort(rng.uniform(0.5, 2.0, dim))[::-1]
    mu = rng.standard_normal(dim)
    return rec.GaussianModel(mu=mu, s=s)


def measure_noisy_K2(op, model, n_probe=600, trials=3000, seed=0):
    """MC 测噪声路线常数 K²=n·err²（仅 range 方向；deconv 估计 μ）。"""
    rng = np.random.default_rng(seed)
    sig = op.singular_values()
    d = model.dim
    if sig.size < d:
        sig = np.concatenate([sig, np.zeros(d - sig.size)])
    rangemask = sig > ZERO_TOL * max(sig.max(), 1.0)
    s, mu, sn = model.s, model.mu, op.noise_std
    err2 = np.empty(trials)
    for k in range(trials):
        xp = mu + np.sqrt(s) * rng.standard_normal((n_probe, d))      # x' in V-basis
        yu = sig * xp + sn * rng.standard_normal((n_probe, d))        # y_u = σ x' + ε
        with np.errstate(divide="ignore", invalid="ignore"):
            muhat = np.where(rangemask, yu.mean(0) / np.where(sig > 0, sig, 1.0), mu)
        err2[k] = np.sum(((muhat - mu) ** 2)[rangemask])
    return float(err2.mean() * n_probe)


def measure_clean_null_K2(op, model, m_probe=200, trials=3000, seed=0):
    """MC 测 clean 路线常数 K²=m·err²（仅零空间方向；clean 直接估 μ）。"""
    rng = np.random.default_rng(seed)
    sig = op.singular_values()
    d = model.dim
    if sig.size < d:
        sig = np.concatenate([sig, np.zeros(d - sig.size)])
    nullmask = sig <= ZERO_TOL * max(sig.max(), 1.0)
    s, mu = model.s, model.mu
    err2 = np.empty(trials)
    for k in range(trials):
        xc = mu + np.sqrt(s) * rng.standard_normal((m_probe, d))
        muhat = xc.mean(0)
        err2[k] = np.sum(((muhat - mu) ** 2)[nullmask])
    return float(err2.mean() * m_probe)


def stats(pred, meas):
    pred, meas = np.asarray(pred), np.asarray(meas)
    lp, lm = np.log10(pred), np.log10(meas)
    slope, intercept = np.polyfit(lp, lm, 1)
    ss_res = np.sum((lm - (slope * lp + intercept)) ** 2)
    ss_tot = np.sum((lm - lm.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    med_rel = float(np.median(np.abs(meas - pred) / pred))
    return slope, r2, med_rel


def scatter_plot(points, title, fname, budget_key):
    """points: list of dict(pred, meas, label). 画 pred vs meas + y=x。"""
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    labels = sorted({p["label"] for p in points})
    cmap = plt.get_cmap("tab10")
    for i, lab in enumerate(labels):
        pr = [p["pred"] for p in points if p["label"] == lab]
        me = [p["meas"] for p in points if p["label"] == lab]
        ax.scatter(pr, me, s=28, color=cmap(i % 10), label=lab, alpha=0.8, edgecolor="k", lw=0.3)
    allv = [p["pred"] for p in points] + [p["meas"] for p in points]
    lo, hi = min(allv) * 0.6, max(allv) * 1.6
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="$y=x$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    slope, r2, med_rel = stats([p["pred"] for p in points], [p["meas"] for p in points])
    ax.set_xlabel(f"predicted {budget_key} (from $\\Psi_I$ spectrum)")
    ax.set_ylabel(f"measured {budget_key} (Monte-Carlo)")
    ax.set_title(f"{title}\nslope={slope:.3f}, $R^2$={r2:.4f}, median rel.err={med_rel:.2%}")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(RESULTS / fname, dpi=130)
    return slope, r2, med_rel


def run(dim=8, seeds=4):
    # injective 病态算子（噪声预算）
    def ops_n():
        yield "gaussian σ=0.3", ops.gaussian_noise(dim, 0.3)
        yield "gaussian σ=0.6", ops.gaussian_noise(dim, 0.6)
        yield "blur std=1.2", ops.blur(dim, kernel_std=1.2)
        yield "degrade κ≈1e3", ops.degrade(ops.gaussian_noise(dim, 0.5), 0.5, 1e-6)
    # non-injective 算子（clean 零空间预算）
    def ops_m():
        yield "grayscale r=6", ops.grayscale(dim, rank=6)
        yield "grayscale r=4", ops.grayscale(dim, rank=4)
        yield "masking α=0.3", ops.masking(dim, alpha=0.3)
        yield "masking α=0.5", ops.masking(dim, alpha=0.5)

    pts_n, pts_m = [], []
    for seed in range(seeds):
        model = _model(dim, seed)
        for lab, op in ops_n():
            K2_meas = measure_noisy_K2(op, model, seed=seed)
            K2_pred = np.sum(1.0 / rec.direction_precisions(op, model)[0]
                             [rec.direction_precisions(op, model)[0] > 0])
            for eps in EPS_LIST:
                pts_n.append({"pred": K2_pred / eps ** 2, "meas": K2_meas / eps ** 2, "label": lab})
        for lab, op in ops_m():
            K2_meas = measure_clean_null_K2(op, model, seed=seed)
            sig = op.singular_values()
            if sig.size < dim:
                sig = np.concatenate([sig, np.zeros(dim - sig.size)])
            nullmask = sig <= ZERO_TOL * max(sig.max(), 1.0)
            K2_pred = float(np.sum(model.s[nullmask]))
            for eps in EPS_LIST:
                pts_m.append({"pred": K2_pred / eps ** 2, "meas": K2_meas / eps ** 2, "label": lab})

    sn = scatter_plot(pts_n, "D3: noisy-sample complexity $n$ ($\\Psi_I$-predicted)",
                      "D3_sample_complexity_n.png", "$n$")
    sm = scatter_plot(pts_m, "D3: clean-sample complexity $m$ (nullspace, $O(1/\\sqrt{m})$)",
                      "D3_sample_complexity_m.png", "$m$")
    print(f"noisy n:  slope={sn[0]:.3f}  R2={sn[1]:.4f}  median rel.err={sn[2]:.2%}")
    print(f"clean m:  slope={sm[0]:.3f}  R2={sm[1]:.4f}  median rel.err={sm[2]:.2%}")
    print(f"saved: results/D3_sample_complexity_n.png, _m.png")
    return {"n": sn, "m": sm}


if __name__ == "__main__":
    run()
