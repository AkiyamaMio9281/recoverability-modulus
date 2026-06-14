"""让 pytest / 脚本无需安装即可 import recmod（src 布局）。"""
import sys
from pathlib import Path

SRC = Path(__file__).parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
