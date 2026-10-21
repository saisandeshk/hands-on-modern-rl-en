# environment.py
import os
import subprocess
import sys
import tempfile


class SandboxEnv:
    """Minimal code-execution environment: subprocess + timeout; this is not a security boundary."""

    def __init__(self, timeout=10):
        self.timeout = timeout

    def step(self, action_type: str, action_args: dict) -> dict:
        """Execute one action and return the observation and termination status."""
        if action_type == "execute_code":
            return self._exec_code(action_args["code"])
        elif action_type == "finish":
            return {"observation": "", "done": True}
        else:
            return {"observation": f"Unknown action: {action_type}", "done": False}

    def _exec_code(self, code: str) -> dict:
        """Execute the code in a subprocess of the current Python, with a bounded wait time."""
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                f.flush()
                temp_path = f.name
                result = subprocess.run(
                    [sys.executable, temp_path],
                    timeout=self.timeout,
                    capture_output=True,
                    text=True,
                )
                return {
                    "observation": (result.stdout + result.stderr)[-500:],  # truncate
                    "done": False,
                }
        except subprocess.TimeoutExpired:
            return {"observation": "TIMEOUT", "done": True}
        except Exception as e:
            return {"observation": f"ERROR: {e}", "done": False}
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

    def reset(self):
        """Reset the environment state (called at the start of a new episode)."""
        pass
