"""线性腐蚀算子 T_r 及其谱结构。

本文件实现 primer §4 的四类线性算子（加噪 / 模糊 / 灰度 / masking）以及退化扫描
工具。所有算子都暴露统一接口：

    apply(x)      : 作用 y = A x （加噪算子额外采样高斯噪声）
    svd()         : 返回 (U, S, Vt)，缓存
    cond()        : 条件数 σ_max / σ_min，σ_min≈0 时为 inf
    adjoint(psi)  : 伴随 T_r* ψ，线性确定性情形 = Aᵀ ψ

记号约定：本仓库的 Ψ 默认指无权 Ψ_I（primer §3），其闭式由这里的 SVD 提供，
见 modulus.py。这里只负责算子与谱，不算模数。
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import circulant

# σ_i 低于此阈值视为零（落入零空间 / 非单射方向）。
ZERO_TOL = 1e-12


class LinearCorruption:
    """确定性线性腐蚀 y = A x，可选加性高斯噪声。

    Parameters
    ----------
    A : (m, n) ndarray
        腐蚀矩阵（T_r 在向量表示下的作用）。
    noise_std : float, optional
        加性高斯噪声标准差（各向同性）。0 表示无噪声。
    name : str
        算子名（用于绘图 / 报表）。
    """

    def __init__(self, A: np.ndarray, noise_std: float = 0.0, name: str = "linear"):
        A = np.asarray(A, dtype=float)
        if A.ndim != 2:
            raise ValueError(f"A must be 2D, got shape {A.shape}")
        self.A = A
        self.noise_std = float(noise_std)
        self.name = name
        self._svd_cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    # -- 维度 --------------------------------------------------------------
    @property
    def dim_in(self) -> int:
        return self.A.shape[1]

    @property
    def dim_out(self) -> int:
        return self.A.shape[0]

    # -- 作用 --------------------------------------------------------------
    def apply(self, x: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        """y = A x (+ 噪声)。x 形状 (n,) 或 (batch, n)。"""
        x = np.asarray(x, dtype=float)
        y = x @ self.A.T  # 支持 (batch, n)
        if self.noise_std > 0:
            rng = np.random.default_rng() if rng is None else rng
            y = y + self.noise_std * rng.standard_normal(y.shape)
        return y

    def adjoint(self, psi: np.ndarray) -> np.ndarray:
        """伴随 T_r* ψ = Aᵀ ψ（确定性线性情形）。

        写成独立方法是为后续非线性算子预留接口；本阶段只做线性。
        """
        psi = np.asarray(psi, dtype=float)
        return psi @ self.A  # (Aᵀ ψ) 对 (batch, m) 也成立
        # 注：psi @ A == (A.T @ psi.T).T，等价于 Aᵀ 作用。

    # -- 谱 ----------------------------------------------------------------
    def svd(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回 (U, S, Vt)，full_matrices=False，结果缓存。"""
        if self._svd_cache is None:
            U, S, Vt = np.linalg.svd(self.A, full_matrices=False)
            self._svd_cache = (U, S, Vt)
        return self._svd_cache

    def modulus_matrix(self) -> np.ndarray:
        """驱动模数 Ψ_I 的算子矩阵 M = (噪声卷积 C_ν) ∘ A。

        - 确定性算子（blur/grayscale/masking，无噪声）→ M = A，直接 SVD。
        - 加噪算子 → M = C_ν · A；特别地 gaussian_noise 的 A=I ⟹ M = C_ν，
          其特征值即 Fourier 乘子 Φ_ν(u)=exp(−σ²u²/2)，故 Ψ_I(e_u)=1/Φ_ν=exp(σ²u²/2)（M1）。

        Note
        ----
        这是 **算子视图**（作用在 clean-space 测试函数 / 信号上），与 `apply`（采样视图，
        真正加随机噪声）刻意分开：模数只看算子的谱，不看噪声实现。
        """
        if self.noise_std > 0:
            C = gaussian_conv_matrix(self.dim_out, self.noise_std)
            return C @ self.A
        return self.A

    def singular_values(self) -> np.ndarray:
        return self.svd()[1]

    def cond(self) -> float:
        """条件数 σ_max / σ_min；存在零奇异值（非单射）时返回 inf。"""
        S = self.singular_values()
        if S.size == 0:
            return np.inf
        smin = S.min()
        if smin <= ZERO_TOL * max(S.max(), 1.0):
            return np.inf
        return float(S.max() / smin)

    def is_injective(self) -> bool:
        """列满秩 ⟺ 单射（primer §1.5）。"""
        S = self.singular_values()
        return bool(S.size == self.dim_in and S.min() > ZERO_TOL * max(S.max(), 1.0))

    def __repr__(self) -> str:
        return (f"LinearCorruption(name={self.name!r}, shape={self.A.shape}, "
                f"noise_std={self.noise_std}, cond={self.cond():.3g})")


# ----------------------------------------------------------------------------
# Fourier / 卷积工具（加性噪声的算子视图）
# ----------------------------------------------------------------------------
def angular_frequencies(dim: int) -> np.ndarray:
    """周期网格的角频率 u_k = 2π·fftfreq(dim) ∈ (−π, π]。

    M1 验证与 modulus 必须用**同一** u 约定，slope σ²/2 才对得上。
    """
    return 2.0 * np.pi * np.fft.fftfreq(dim)


def gaussian_conv_matrix(dim: int, sigma: float) -> np.ndarray:
    """加性高斯噪声的卷积算子 C_ν 的实循环矩阵。

    用 Fourier 乘子直接构造：特征值 = Φ_ν(u_k) = exp(−σ²u_k²/2)（精确，无混叠）。
    σ≤0 退化为单位阵。
    """
    if sigma <= 0:
        return np.eye(dim)
    mult = np.exp(-0.5 * sigma ** 2 * angular_frequencies(dim) ** 2)
    col = np.fft.ifft(mult).real  # 循环矩阵首列；mult 实偶 ⟹ col 实
    return circulant(col)


# ----------------------------------------------------------------------------
# 四类算子工厂（primer §4）
# ----------------------------------------------------------------------------
def gaussian_noise(dim: int, sigma: float) -> LinearCorruption:
    """加噪：A = I，加性高斯噪声 σ。Fourier 乘子，单射（退回 SFBD）。"""
    return LinearCorruption(np.eye(dim), noise_std=sigma, name=f"gaussian(σ={sigma})")


def blur(dim: int, kernel: np.ndarray | None = None, kernel_std: float = 1.5) -> LinearCorruption:
    """模糊：循环卷积矩阵。高频奇异值快衰减 → 病态但单射。

    kernel 给定则直接用；否则用宽度 kernel_std 的离散高斯核（已归一化）。
    """
    if kernel is None:
        half = max(1, int(np.ceil(3 * kernel_std)))
        t = np.arange(-half, half + 1)
        kernel = np.exp(-0.5 * (t / kernel_std) ** 2)
        kernel = kernel / kernel.sum()
    # 把核嵌入长度 dim 的第一列（循环），核中心对齐到 index 0。
    col = np.zeros(dim)
    k = len(kernel)
    c = k // 2
    for i, w in enumerate(kernel):
        col[(i - c) % dim] += w
    A = circulant(col)
    return LinearCorruption(A, name=f"blur(std={kernel_std})")


def grayscale(dim: int, rank: int, seed: int = 0) -> LinearCorruption:
    """灰度：秩 `rank` 的正交投影（dim - rank 个零奇异值）。

    非单射，零空间维数 = dim - rank（对应需要 clean 先验的方向）。
    """
    if not (0 < rank <= dim):
        raise ValueError(f"rank must be in (0, dim], got rank={rank}, dim={dim}")
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((dim, rank)))  # (dim, rank) 正交列
    A = Q @ Q.T  # 秩-rank 投影，奇异值为 rank 个 1 + (dim-rank) 个 0
    return LinearCorruption(A, name=f"grayscale(rank={rank}/{dim})")


def masking(dim: int, alpha: float, seed: int = 0) -> LinearCorruption:
    """masking：**固定**对角 0/1 算子，每坐标以概率 α 抽样一次决定是否置 0。

    注意（避免与 OMNI 混淆）：这是一个**确定性线性算子**，只要有任一坐标被 mask 即
    非单射（该坐标为零奇异值、属零空间）。这与 OMNI 所述"random dropout 在 α<1 时单射"
    不同——后者指**随机 mask 核**在分布层面单射。线性算子 wedge 用这个固定模式版本：
    它的零空间方向正对应"需要 clean 先验"的坐标。
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0,1], got {alpha}")
    rng = np.random.default_rng(seed)
    keep = (rng.random(dim) >= alpha).astype(float)
    A = np.diag(keep)
    return LinearCorruption(A, name=f"masking(α={alpha})")


# ----------------------------------------------------------------------------
# 退化扫描（primer §4 末）
# ----------------------------------------------------------------------------
def degrade(op: LinearCorruption, t: float, floor: float = 1e-8) -> LinearCorruption:
    """把一个**满秩**算子沿单一方向平滑推向非单射：用 t∈[0,1] 把最小奇异值几何压向 0。

    新的最小奇异值 σ_min(t) = σ_max · floor**t（其余奇异值不动）。当基算子谱较平
    （如 identity，σ≡σ_max）时，条件数 κ(t) = floor**(-t) 随 t 单调从 1 发散到 1/floor，
    正是 D2 主图 x 轴（条件数）所需的"沿某方向逼近不可恢复"。保留 U, Vt 与噪声，只改谱。

    Note
    ----
    要求基算子满秩（无零奇异值）；对已经低秩的算子（灰度 / masking(α=1)）退化无意义，
    会触发 ValueError。
    """
    if not (0.0 <= t <= 1.0):
        raise ValueError(f"t must be in [0,1], got {t}")
    U, S, Vt = op.svd()
    smax = S.max()
    if S.min() <= ZERO_TOL * max(smax, 1.0):
        raise ValueError("degrade expects a full-rank (injective) base operator; "
                         f"'{op.name}' already has a zero singular value.")
    S = S.copy()
    S[-1] = smax * (floor ** t)  # 只压最小奇异值，保持其为最小
    A_new = (U * S) @ Vt
    return LinearCorruption(A_new, noise_std=op.noise_std,
                            name=f"{op.name}|degrade(t={t:.3g})")


__all__ = [
    "LinearCorruption",
    "gaussian_noise",
    "blur",
    "grayscale",
    "masking",
    "degrade",
    "angular_frequencies",
    "gaussian_conv_matrix",
    "ZERO_TOL",
]
