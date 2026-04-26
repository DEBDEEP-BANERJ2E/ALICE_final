"""
Programmatic Verifier — executes agent-generated code in a RestrictedPython sandbox.

Restrictions:
- No access to: open, exec, eval, __import__, compile
- Memory limit: 512 MB
- Timeout: 5 seconds
- No network or filesystem access
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 5
MEMORY_LIMIT_MB = 512
BLOCKED_BUILTINS = frozenset(["open", "exec", "eval", "__import__", "compile"])


class ProgrammaticVerifier:
    """Executes code in a RestrictedPython sandbox."""

    def __init__(
        self,
        timeout: int = TIMEOUT_SECONDS,
        memory_limit_mb: int = MEMORY_LIMIT_MB,
    ) -> None:
        self.timeout = timeout
        self.memory_limit_mb = memory_limit_mb
        self._violation_log: list = []

    def execute_code(self, code: str) -> Dict[str, Any]:
        """Execute code in RestrictedPython sandbox.

        Returns:
            {success, output, execution_time, error_message, error_type}
        """
        # Placeholder — full implementation in Task 8
        start = time.monotonic()
        try:
            result = self._run_restricted(code)
            elapsed = time.monotonic() - start
            return {
                "success": True,
                "output": result,
                "execution_time": elapsed,
                "error_message": None,
                "error_type": None,
            }
        except Exception as exc:
            elapsed = time.monotonic() - start
            logger.warning("Sandbox execution failed: %s", exc)
            return {
                "success": False,
                "output": None,
                "execution_time": elapsed,
                "error_message": str(exc),
                "error_type": type(exc).__name__,
            }

    def is_safe(self, code: str) -> bool:
        """Quick static check — returns False if obviously unsafe builtins are present."""
        for name in BLOCKED_BUILTINS:
            if name in code:
                return False
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_restricted(self, code: str) -> Any:
        """Run code inside RestrictedPython environment."""
        from RestrictedPython import compile_restricted, safe_globals  # type: ignore
        byte_code = compile_restricted(code, filename="<agent>", mode="exec")
        local_vars: Dict[str, Any] = {}
        exec(byte_code, safe_globals, local_vars)  # noqa: S102
        return local_vars.get("result")

    def _log_violation(self, code_snippet: str, violation_type: str) -> None:
        self._violation_log.append({
            "timestamp": time.time(),
            "violation_type": violation_type,
            "code_snippet": code_snippet[:200],
        })
        logger.warning("Sandbox violation [%s]: %s", violation_type, code_snippet[:80])
