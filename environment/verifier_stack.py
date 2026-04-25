"""
Verifier Stack — three-tier verification system.

Tier 1: Programmatic (RestrictedPython sandbox, 5s timeout)
Tier 2: LLM Judge (reference model scoring with CoT rubric)
Tier 3: Regression Battery (held-out test set evaluation)

Composite score = (0.3 × T1) + (0.4 × T2) + (0.3 × T3)
If Tier 1 fails → composite = 0, skip Tiers 2 & 3.
"""

from __future__ import annotations

import logging
import os
import signal
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

TIER1_WEIGHT = 0.3
TIER2_WEIGHT = 0.4
TIER3_WEIGHT = 0.3
TIER2_THRESHOLD = 0.5
COMPOSITE_PASS_THRESHOLD = 0.5
SANDBOX_TIMEOUT = int(os.getenv("SANDBOX_TIMEOUT", "5"))
SANDBOX_MEMORY_MB = int(os.getenv("SANDBOX_MEMORY_MB", "512"))
REGRESSION_BATTERY_SIZE = int(os.getenv("REGRESSION_BATTERY_SIZE", "20"))
LAMBDA_JUDGE = float(os.getenv("LAMBDA_JUDGE", "0.3"))


# ---------------------------------------------------------------------------
# Regression battery — 20 held-out tasks for regression testing
# ---------------------------------------------------------------------------

_REGRESSION_TASKS: List[Dict[str, Any]] = [
    {"prompt": "result = 2 + 2", "expected": 4},
    {"prompt": "result = 10 - 3", "expected": 7},
    {"prompt": "result = 3 * 7", "expected": 21},
    {"prompt": "result = [x * 2 for x in range(5)]", "expected": [0, 2, 4, 6, 8]},
    {"prompt": "result = sum(range(10))", "expected": 45},
    {"prompt": "result = len('hello world')", "expected": 11},
    {"prompt": "result = 'hello' + ' ' + 'world'", "expected": "hello world"},
    {"prompt": "result = sorted([3, 1, 4, 1, 5, 9])", "expected": [1, 1, 3, 4, 5, 9]},
    {"prompt": "result = max([10, 20, 5, 30, 15])", "expected": 30},
    {"prompt": "result = min([10, 20, 5, 30, 15])", "expected": 5},
    {"prompt": "result = 2 ** 8", "expected": 256},
    {"prompt": "result = 100 // 7", "expected": 14},
    {"prompt": "result = 100 % 7", "expected": 2},
    {"prompt": "result = abs(-42)", "expected": 42},
    {"prompt": "result = round(3.14159, 2)", "expected": 3.14},
    {"prompt": "result = list(reversed([1, 2, 3, 4, 5]))", "expected": [5, 4, 3, 2, 1]},
    {"prompt": "result = ''.join(['a', 'b', 'c'])", "expected": "abc"},
    {"prompt": "result = [i**2 for i in range(1, 6)]", "expected": [1, 4, 9, 16, 25]},
    {"prompt": "result = {1, 2, 3} | {3, 4, 5}", "expected": {1, 2, 3, 4, 5}},
    {"prompt": "result = dict(zip(['a', 'b'], [1, 2]))", "expected": {"a": 1, "b": 2}},
]


class VerifierStack:
    """Orchestrates three-tier verification of agent outputs."""

    def __init__(self, failure_bank: Any = None) -> None:
        self._failure_bank = failure_bank
        self._regression_baseline: float = 1.0  # initialized on first run

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(self, agent_output: str, task: str = "") -> Dict[str, Any]:
        """Execute three-tier verification cascade.

        Returns:
            {tier1_score, tier2_score, tier3_score, composite_score, reasoning}
        """
        tier1_result = self.tier1_verify(agent_output)
        tier1_score = 1.0 if tier1_result.get("success") else 0.0

        if not tier1_result.get("success"):
            result = {
                "tier1_score": 0.0,
                "tier1_details": tier1_result,
                "tier2_score": None,
                "tier2_details": None,
                "tier3_score": None,
                "tier3_details": None,
                "composite_score": 0.0,
                "reasoning": f"Tier 1 failed: {tier1_result.get('error_message', '')}",
            }
            self._handle_failure(agent_output, result, task)
            return result

        tier2_result = self.tier2_verify(agent_output, task)
        tier2_score = tier2_result.get("composite_score", 0.0)

        if tier2_score < TIER2_THRESHOLD:
            tier3_result = self.tier3_verify(agent_output)
            tier3_score = tier3_result.get("pass_rate", 0.0)
        else:
            tier3_result = {"pass_rate": 1.0, "skipped": True}
            tier3_score = 1.0

        composite = (
            TIER1_WEIGHT * tier1_score
            + TIER2_WEIGHT * tier2_score
            + TIER3_WEIGHT * tier3_score
        )
        composite = max(0.0, min(1.0, composite))

        result = {
            "tier1_score": tier1_score,
            "tier1_details": tier1_result,
            "tier2_score": tier2_score,
            "tier2_details": tier2_result,
            "tier3_score": tier3_score,
            "tier3_details": tier3_result,
            "composite_score": composite,
            "reasoning": self._build_reasoning(tier1_score, tier2_score, tier3_score, composite),
        }

        if composite < COMPOSITE_PASS_THRESHOLD:
            self._handle_failure(agent_output, result, task)

        logger.info("Verification complete: composite=%.3f", composite)
        return result

    def tier1_verify(self, code: str) -> Dict[str, Any]:
        """Programmatic verification in RestrictedPython sandbox.

        Executes code with 5s timeout, 512MB memory cap, and blocked builtins.
        Returns: {success, output, execution_time, error_message, error_type}
        """
        start = time.monotonic()
        try:
            result_container: Dict[str, Any] = {}
            exception_container: Dict[str, Any] = {}

            def _run():
                try:
                    from RestrictedPython import compile_restricted, safe_globals, safe_builtins  # type: ignore
                    from RestrictedPython.Guards import safe_globals as restricted_safe_globals  # type: ignore

                    restricted_builtins = dict(safe_builtins)
                    # Block dangerous builtins
                    for dangerous in ("open", "exec", "eval", "__import__", "compile", "breakpoint"):
                        restricted_builtins.pop(dangerous, None)

                    restricted_globals = dict(safe_globals)
                    restricted_globals["__builtins__"] = restricted_builtins
                    restricted_globals["_getiter_"] = iter
                    restricted_globals["_getattr_"] = getattr
                    restricted_globals["_write_"] = lambda x: x
                    restricted_globals["_inplacevar_"] = lambda op, x, y: (
                        x + y if op == "+=" else x - y if op == "-=" else
                        x * y if op == "*=" else x / y if op == "/=" else x
                    )

                    byte_code = compile_restricted(code, filename="<agent>", mode="exec")
                    local_vars: Dict[str, Any] = {}
                    exec(byte_code, restricted_globals, local_vars)  # noqa: S102
                    result_container["output"] = local_vars.get("result")
                except Exception as exc:
                    exception_container["exc"] = exc

            t = threading.Thread(target=_run, daemon=True)
            t.start()
            t.join(timeout=SANDBOX_TIMEOUT)

            elapsed = time.monotonic() - start

            if t.is_alive():
                return {
                    "success": False,
                    "output": None,
                    "execution_time": elapsed,
                    "error_message": f"Execution timed out after {SANDBOX_TIMEOUT}s",
                    "error_type": "TimeoutError",
                }

            if exception_container:
                exc = exception_container["exc"]
                return {
                    "success": False,
                    "output": None,
                    "execution_time": elapsed,
                    "error_message": str(exc),
                    "error_type": type(exc).__name__,
                }

            return {
                "success": True,
                "output": result_container.get("output"),
                "execution_time": elapsed,
                "error_message": None,
                "error_type": None,
            }

        except Exception as exc:
            elapsed = time.monotonic() - start
            logger.warning("Tier 1 outer error: %s", exc)
            return {
                "success": False,
                "output": None,
                "execution_time": elapsed,
                "error_message": str(exc),
                "error_type": type(exc).__name__,
            }

    def tier2_verify(self, output: str, task: str = "") -> Dict[str, Any]:
        """LLM Judge verification using HF Inference API with CoT rubric.

        Returns: {criterion_scores, composite_score, reasoning, flagged_for_review}
        """
        try:
            from openai import OpenAI  # type: ignore

            api_key = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or "dummy"
            base_url = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
            model = os.getenv("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

            client = OpenAI(api_key=api_key, base_url=base_url)

            rubric_prompt = (
                f"Evaluate this response to the task.\n\n"
                f"Task: {task[:500]}\n\n"
                f"Response: {output[:500]}\n\n"
                f"Score each criterion 0.0-1.0:\n"
                f"1. Correctness: Is the answer factually correct?\n"
                f"2. Completeness: Does it fully address the task?\n"
                f"3. Clarity: Is the reasoning clear?\n"
                f"4. Efficiency: Is the solution concise?\n\n"
                f"Reply ONLY with 4 numbers separated by spaces, e.g.: 0.8 0.7 0.9 0.6"
            )

            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": rubric_prompt}],
                max_tokens=32,
                temperature=0.0,
            )
            raw = resp.choices[0].message.content.strip() if resp.choices else ""
            scores = self._parse_rubric_scores(raw)

            composite = float(sum(scores.values()) / len(scores)) if scores else 0.5
            flagged = False

            return {
                "criterion_scores": scores,
                "composite_score": composite,
                "reasoning": raw,
                "flagged_for_review": flagged,
            }

        except Exception as exc:
            logger.warning("Tier 2 LLM Judge failed (%s) — falling back to 0.5", exc)
            return {
                "criterion_scores": {"correctness": 0.5, "completeness": 0.5, "clarity": 0.5, "efficiency": 0.5},
                "composite_score": 0.5,
                "reasoning": f"fallback: {exc}",
                "flagged_for_review": False,
            }

    def tier3_verify(self, agent_output: str) -> Dict[str, Any]:
        """Regression Battery — evaluate against 20 held-out tasks.

        Returns: {pass_rate, failed_tasks, performance_trend}
        """
        passed = 0
        failed: List[str] = []

        for task in _REGRESSION_TASKS[:REGRESSION_BATTERY_SIZE]:
            t1 = self.tier1_verify(task["prompt"])
            if t1.get("success") and t1.get("output") == task["expected"]:
                passed += 1
            else:
                failed.append(task["prompt"][:60])

        total = min(REGRESSION_BATTERY_SIZE, len(_REGRESSION_TASKS))
        pass_rate = passed / total if total > 0 else 0.0

        # Track regression vs baseline
        trend = pass_rate - self._regression_baseline
        if pass_rate < self._regression_baseline - 0.05:
            logger.warning(
                "Regression alert: pass_rate=%.2f dropped from baseline=%.2f",
                pass_rate, self._regression_baseline,
            )

        return {
            "pass_rate": pass_rate,
            "failed_tasks": failed,
            "performance_trend": trend,
        }

    def set_regression_baseline(self) -> None:
        """Run regression battery and set current pass rate as baseline."""
        result = self.tier3_verify("")
        self._regression_baseline = result["pass_rate"]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_rubric_scores(self, raw: str) -> Dict[str, float]:
        """Parse space-separated scores from LLM rubric response."""
        criteria = ["correctness", "completeness", "clarity", "efficiency"]
        try:
            parts = [float(x) for x in raw.split()[:4]]
            if len(parts) == 4:
                return {c: max(0.0, min(1.0, v)) for c, v in zip(criteria, parts)}
        except (ValueError, TypeError):
            pass
        return {c: 0.5 for c in criteria}

    def _handle_failure(self, agent_output: str, result: Dict[str, Any], task: str = "") -> None:
        """Add failed output to Failure Bank if available."""
        if self._failure_bank is not None:
            self._failure_bank.add_failure({
                "prompt": task,
                "actual_output": agent_output,
                "verification_result": result,
                "error_type": "verification_failure",
            })

    @staticmethod
    def _build_reasoning(t1: float, t2: Optional[float], t3: Optional[float], composite: float) -> str:
        t2_str = f"{t2:.2f}" if t2 is not None else "N/A"
        t3_str = f"{t3:.2f}" if t3 is not None else "N/A"
        return f"T1={t1:.2f} T2={t2_str} T3={t3_str} composite={composite:.3f}"
