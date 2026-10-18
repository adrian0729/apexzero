"""Make ``import apexzero`` work even if the editable-install .pth file is skipped.

On some macOS setups the files inside .venv get the "hidden" flag, and Python >= 3.12
ignores hidden .pth files, which silently breaks the editable install. Scripts import
this module first so they never depend on that file. (Fix the venv itself with
``chflags -R nohidden .venv``.)
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
