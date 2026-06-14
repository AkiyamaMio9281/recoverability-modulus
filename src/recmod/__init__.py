"""recmod — recoverability modulus toolkit.

实现细节见 ../../prompts/。本包的算子与模数全程使用无权 Ψ_I（primer §3）。
"""

from . import operators
from . import modulus
from . import recover
from . import budget
from . import theory

__all__ = ["operators", "modulus", "recover", "budget", "theory"]
