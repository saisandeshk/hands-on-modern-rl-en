"""Keep sibling-script imports working for the pre-existing unit tests.

With ``__init__.py`` present, pytest imports test modules as
``chapter03_value_functions.test_*`` with ``code/`` on ``sys.path``;
insert this directory so ``import gridworld_q_learning`` keeps working.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
