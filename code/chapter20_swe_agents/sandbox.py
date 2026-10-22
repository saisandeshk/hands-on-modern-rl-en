"""Sandboxed execution of candidate solutions against a test suite.

The verifier is the heart of the RLVR loop from Chapter 20: a candidate
patch is applied in an isolated environment, the task tests are executed,
and the pass fraction becomes the reward. This module is a miniature of the
SWE-bench evaluator (restore snapshot -> apply patch -> run tests in an
isolated container -> record reward), with two simplifications honest to a
CPU-only textbook lab:

- "Container" = a fresh temporary directory used as the child's working
  directory, plus a per-batch wall-clock timeout. The candidates are our own
  fixed snippets (never model-generated code); ``tasks.FORBIDDEN_TOKENS``
  audits that they import nothing capable of network or filesystem access.
- One subprocess call executes a whole batch of test cases for one
  candidate, so training needs only one sandbox call per (task, candidate).

Timeouts, crashes, and per-test failures are reported separately, mirroring
the chapter's insistence on distinguishing "the test failed" from "the
environment did not start / timed out".
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# The driver is written into the temp dir and run as a separate process.
# It imports `solve` from the candidate file, runs every case, and records
# one boolean plus an optional error string per case.
_DRIVER_SOURCE = """\
import json
import traceback

from candidate import solve

with open("cases.json") as f:
    cases = json.load(f)

results = []
for case in cases:
    try:
        got = solve(*case["args"])
        ok = bool(got == case["expected"])
        results.append({"ok": ok, "error": None if ok else f"got {got!r}, expected {case['expected']!r}"})
    except Exception:
        results.append({"ok": False, "error": traceback.format_exc(limit=3)})

with open("result.json", "w") as f:
    json.dump(results, f)
"""


@dataclass
class RunResult:
    """Outcome of one sandboxed batch run."""

    passed: int
    total: int
    ok_per_case: list[bool] = field(default_factory=list)
    errors: list[str | None] = field(default_factory=list)
    timed_out: bool = False
    crashed: bool = False
    stderr: str = ""

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


def run_candidate(
    source: str,
    cases: list[tuple[tuple, object]],
    timeout: float = 10.0,
) -> RunResult:
    """Execute ``source`` (which must define ``solve``) on ``cases`` in a subprocess.

    Args:
        source: Python source defining a ``solve`` function.
        cases: list of ``(args_tuple, expected)`` pairs. Args and expected
            values must be JSON-serializable.
        timeout: wall-clock seconds for the whole batch.

    Returns:
        A :class:`RunResult` with per-case pass/fail. A timeout or a child
        crash marks every case failed and sets the corresponding flag --
        infrastructure failure is never silently counted as mere test failure.
    """
    payload = [{"args": list(args), "expected": expected} for args, expected in cases]
    with tempfile.TemporaryDirectory(prefix="swe_toy_") as tmpdir:
        tmp = Path(tmpdir)
        (tmp / "candidate.py").write_text(source)
        (tmp / "cases.json").write_text(json.dumps(payload))
        (tmp / "driver.py").write_text(_DRIVER_SOURCE)
        try:
            proc = subprocess.run(
                [sys.executable, "driver.py"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return RunResult(passed=0, total=len(cases), timed_out=True)
        if proc.returncode != 0 or not (tmp / "result.json").exists():
            return RunResult(
                passed=0,
                total=len(cases),
                crashed=True,
                stderr=proc.stderr[-2000:],
            )
        raw = json.loads((tmp / "result.json").read_text())
    ok_list = [bool(r["ok"]) for r in raw]
    return RunResult(
        passed=sum(ok_list),
        total=len(ok_list),
        ok_per_case=ok_list,
        errors=[r["error"] for r in raw],
    )


def pass_rate(source: str, cases: list[tuple[tuple, object]], timeout: float = 10.0) -> float:
    """Convenience wrapper returning just the pass fraction (the RL reward)."""
    return run_candidate(source, cases, timeout=timeout).pass_rate
