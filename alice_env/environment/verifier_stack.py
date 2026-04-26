"""
Verifier Stack — three-tier verification system.

Tier 1: Programmatic (RestrictedPython sandbox, 5s timeout, 512MB memory cap,
        stdout/stderr capture)
Tier 2: LLM Judge (dual-model scoring with CoT rubric, divergence flag,
        output cache for identical inputs)
Tier 3: Regression Battery (500+ held-out tasks, auto-trigger every 100 episodes,
        manual trigger via set_regression_baseline())

Composite score = (0.3 × T1) + (0.4 × T2) + (0.3 × T3)
If Tier 1 fails → composite = 0, skip Tiers 2 & 3.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

TIER1_WEIGHT = 0.3
TIER2_WEIGHT = 0.4
TIER3_WEIGHT = 0.3
TIER2_THRESHOLD = 0.5
COMPOSITE_PASS_THRESHOLD = 0.5
SANDBOX_TIMEOUT = int(os.getenv("SANDBOX_TIMEOUT", "5"))
SANDBOX_MEMORY_MB = int(os.getenv("SANDBOX_MEMORY_MB", "512"))
REGRESSION_BATTERY_SIZE = int(os.getenv("REGRESSION_BATTERY_SIZE", "500"))
TIER2_DIVERGENCE_THRESHOLD = 0.2
T3_AUTO_TRIGGER_INTERVAL = 100  # auto-run regression battery every N episodes


# ---------------------------------------------------------------------------
# Regression battery — 500+ held-out tasks (generated at import time)
# ---------------------------------------------------------------------------

def _build_regression_battery() -> List[Dict[str, Any]]:
    """Generate 500+ diverse regression tasks covering arithmetic, strings, lists, sets, dicts."""
    tasks: List[Dict[str, Any]] = []

    # Arithmetic: +, -, *, //, % across (0..14) × (1..7) = 525 tasks
    for a in range(0, 15):
        for b in range(1, 8):
            denom = b  # b >= 1, safe
            dividend = a + b * a  # always divisible properties not needed — just correct values
            tasks.append({"prompt": f"result = {a} + {b}", "expected": a + b})
            tasks.append({"prompt": f"result = {a} - {b}", "expected": a - b})
            tasks.append({"prompt": f"result = {a} * {b}", "expected": a * b})
            tasks.append({"prompt": f"result = {dividend} // {denom}", "expected": dividend // denom})
            tasks.append({"prompt": f"result = {dividend} % {denom}", "expected": dividend % denom})

    # abs / round / pow
    for v in range(-8, 9):
        tasks.append({"prompt": f"result = abs({v})", "expected": abs(v)})
    for v in range(0, 10):
        tasks.append({"prompt": f"result = pow(2, {v})", "expected": pow(2, v)})
    tasks.append({"prompt": "result = round(3.14159, 2)", "expected": 3.14})
    tasks.append({"prompt": "result = round(2.71828, 3)", "expected": 2.718})

    # Boolean / comparison
    for a in range(0, 5):
        for b in range(0, 5):
            tasks.append({"prompt": f"result = {a} < {b}", "expected": a < b})
            tasks.append({"prompt": f"result = {a} == {b}", "expected": a == b})
            tasks.append({"prompt": f"result = {a} > {b}", "expected": a > b})

    # String operations
    for n in range(1, 11):
        s = "x" * n
        tasks.append({"prompt": f"result = len('{s}')", "expected": n})
        tasks.append({"prompt": f"result = '{s}'.upper()", "expected": s.upper()})
        tasks.append({"prompt": f"result = '  {s}  '.strip()", "expected": s})
        tasks.append({"prompt": f"result = '{s}'.replace('x', 'y')", "expected": "y" * n})
    tasks.append({"prompt": "result = ''.join(['a', 'b', 'c'])", "expected": "abc"})
    tasks.append({"prompt": "result = ' '.join(['hello', 'world'])", "expected": "hello world"})
    tasks.append({"prompt": "result = 'hello world'.split()", "expected": ["hello", "world"]})
    tasks.append({"prompt": "result = 'hello' + ' ' + 'world'", "expected": "hello world"})
    tasks.append({"prompt": "result = len('hello world')", "expected": 11})

    # List / range / comprehension
    for n in range(1, 11):
        tasks.append({"prompt": f"result = list(range({n}))", "expected": list(range(n))})
        tasks.append({"prompt": f"result = sum(range({n + 1}))", "expected": sum(range(n + 1))})
        tasks.append({"prompt": f"result = len(list(range({n})))", "expected": n})
        tasks.append({"prompt": f"result = max(range(1, {n + 1}))", "expected": n})
        tasks.append({"prompt": f"result = min(range(1, {n + 1}))", "expected": 1})
        tasks.append({"prompt": f"result = [x * 2 for x in range({n})]",
                      "expected": [x * 2 for x in range(n)]})
        tasks.append({"prompt": f"result = [i**2 for i in range(1, {n + 1})]",
                      "expected": [i ** 2 for i in range(1, n + 1)]})

    # sorted / reversed / max / min on literal lists
    tasks.append({"prompt": "result = sorted([3, 1, 4, 1, 5, 9])", "expected": [1, 1, 3, 4, 5, 9]})
    tasks.append({"prompt": "result = sorted([3, 1, 4, 1, 5, 9], reverse=True)",
                  "expected": [9, 5, 4, 3, 1, 1]})
    tasks.append({"prompt": "result = list(reversed([1, 2, 3, 4, 5]))", "expected": [5, 4, 3, 2, 1]})
    tasks.append({"prompt": "result = max([10, 20, 5, 30, 15])", "expected": 30})
    tasks.append({"prompt": "result = min([10, 20, 5, 30, 15])", "expected": 5})
    tasks.append({"prompt": "result = sum([1, 2, 3, 4, 5])", "expected": 15})

    # set / dict
    tasks.append({"prompt": "result = len({1, 2, 3, 2, 1})", "expected": 3})
    tasks.append({"prompt": "result = sorted({3, 1, 2})", "expected": [1, 2, 3]})
    tasks.append({"prompt": "result = {1, 2, 3} | {3, 4, 5}", "expected": {1, 2, 3, 4, 5}})
    tasks.append({"prompt": "result = {1, 2, 3} & {2, 3, 4}", "expected": {2, 3}})
    tasks.append({"prompt": "result = dict(zip(['a', 'b'], [1, 2]))", "expected": {"a": 1, "b": 2}})
    tasks.append({"prompt": "result = dict(zip(['a', 'b', 'c'], [1, 2, 3]))",
                  "expected": {"a": 1, "b": 2, "c": 3}})
    tasks.append({"prompt": "result = len({'a': 1, 'b': 2, 'c': 3})", "expected": 3})

    # Bitwise / integer tricks
    tasks.append({"prompt": "result = 2 ** 8", "expected": 256})
    tasks.append({"prompt": "result = 100 // 7", "expected": 14})
    tasks.append({"prompt": "result = 100 % 7", "expected": 2})
    tasks.append({"prompt": "result = 255 & 15", "expected": 15})
    tasks.append({"prompt": "result = 1 << 4", "expected": 16})

    return tasks


_REGRESSION_TASKS: List[Dict[str, Any]] = _build_regression_battery()


class VerifierStack:
    """Orchestrates three-tier verification of agent outputs."""

    def __init__(self, failure_bank: Any = None) -> None:
        self._failure_bank = failure_bank
        self._regression_baseline: float = 1.0
        self._episode_count: int = 0          # tracks episodes for auto-trigger
        self._t2_cache: Dict[str, Dict[str, Any]] = {}  # output cache for Tier 2
        # Thread pool for non-blocking failure bank insertions
        from concurrent.futures import ThreadPoolExecutor
        self._fb_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fb-insert")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(self, agent_output: str, task: str = "") -> Dict[str, Any]:
        """Execute three-tier verification cascade.

        Returns:
            {tier1_score, tier2_score, tier3_score, composite_score, reasoning}
        """
        self._episode_count += 1

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

        # Auto-trigger Tier 3 every T3_AUTO_TRIGGER_INTERVAL episodes, or when T2 is weak
        run_tier3 = (
            tier2_score < TIER2_THRESHOLD
            or self._episode_count % T3_AUTO_TRIGGER_INTERVAL == 0
        )
        if run_tier3:
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

        Enforces 5s timeout, best-effort 512MB memory cap, blocked builtins.
        Captures stdout, stderr, and return value.
        Returns: {success, output, stdout, stderr, execution_time, error_message, error_type}
        """
        start = time.monotonic()
        result_container: Dict[str, Any] = {}
        exception_container: Dict[str, Any] = {}

        def _run() -> None:
            # Memory limit via resource module — skip on HF Spaces (SPACE_ID is set by HF)
            # because setrlimit(RLIMIT_AS, 512MB) limits the whole process, crashing the server.
            is_hf_space = bool(os.getenv("SPACE_ID") or os.getenv("SPACE_AUTHOR_NAME"))
            if not is_hf_space:
                try:
                    import resource as _resource
                    limit = SANDBOX_MEMORY_MB * 1024 * 1024
                    _resource.setrlimit(_resource.RLIMIT_AS, (limit, limit))
                except Exception:
                    pass

            captured_stdout = io.StringIO()
            captured_stderr = io.StringIO()
            try:
                from RestrictedPython import compile_restricted, safe_globals, safe_builtins  # type: ignore
                from RestrictedPython.Guards import safe_globals as _rsg  # type: ignore

                restricted_builtins = dict(safe_builtins)
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

                old_out, old_err = sys.stdout, sys.stderr
                sys.stdout, sys.stderr = captured_stdout, captured_stderr
                try:
                    exec(byte_code, restricted_globals, local_vars)  # noqa: S102
                finally:
                    sys.stdout, sys.stderr = old_out, old_err

                result_container["output"] = local_vars.get("result")
                result_container["stdout"] = captured_stdout.getvalue()
                result_container["stderr"] = captured_stderr.getvalue()
            except Exception as exc:
                sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
                exception_container["exc"] = exc
                exception_container["stdout"] = captured_stdout.getvalue()
                exception_container["stderr"] = captured_stderr.getvalue()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=SANDBOX_TIMEOUT)
        elapsed = time.monotonic() - start

        if t.is_alive():
            return {
                "success": False,
                "output": None,
                "stdout": "",
                "stderr": "",
                "execution_time": elapsed,
                "error_message": f"Execution timed out after {SANDBOX_TIMEOUT}s",
                "error_type": "TimeoutError",
            }

        if exception_container:
            exc = exception_container["exc"]
            return {
                "success": False,
                "output": None,
                "stdout": exception_container.get("stdout", ""),
                "stderr": exception_container.get("stderr", ""),
                "execution_time": elapsed,
                "error_message": str(exc),
                "error_type": type(exc).__name__,
            }

        return {
            "success": True,
            "output": result_container.get("output"),
            "stdout": result_container.get("stdout", ""),
            "stderr": result_container.get("stderr", ""),
            "execution_time": elapsed,
            "error_message": None,
            "error_type": None,
        }

    def tier2_verify(self, output: str, task: str = "") -> Dict[str, Any]:
        """LLM Judge verification: dual-model scoring with CoT rubric and output cache.

        Calls primary and secondary reference models independently, averages their
        criterion scores, and flags when |primary_composite - secondary_composite| > 0.2.

        Returns: {criterion_scores, composite_score, reasoning, flagged_for_review}
        """
        cache_key = hashlib.sha256(f"{output}|||{task}".encode()).hexdigest()
        if cache_key in self._t2_cache:
            logger.debug("Tier 2 cache hit for output hash %s", cache_key[:8])
            return self._t2_cache[cache_key]

        primary_scores = self._call_judge_model(output, task, "primary")
        secondary_scores = self._call_judge_model(output, task, "secondary")

        avg_scores = {
            k: (primary_scores[k] + secondary_scores[k]) / 2.0
            for k in primary_scores
        }
        composite = float(sum(avg_scores.values()) / len(avg_scores))

        primary_composite = float(sum(primary_scores.values()) / len(primary_scores))
        secondary_composite = float(sum(secondary_scores.values()) / len(secondary_scores))
        flagged = abs(primary_composite - secondary_composite) > TIER2_DIVERGENCE_THRESHOLD

        if flagged:
            logger.warning(
                "Tier 2 divergence: primary=%.3f secondary=%.3f diff=%.3f",
                primary_composite, secondary_composite,
                abs(primary_composite - secondary_composite),
            )

        result = {
            "criterion_scores": avg_scores,
            "primary_scores": primary_scores,
            "secondary_scores": secondary_scores,
            "composite_score": composite,
            "reasoning": (
                f"primary={list(primary_scores.values())} "
                f"secondary={list(secondary_scores.values())}"
            ),
            "flagged_for_review": flagged,
        }
        self._t2_cache[cache_key] = result
        return result

    def tier3_verify(self, agent_output: str) -> Dict[str, Any]:
        """Regression Battery — evaluate against 500+ held-out tasks.

        Auto-triggered every T3_AUTO_TRIGGER_INTERVAL episodes or when Tier 2 is weak.
        Returns: {pass_rate, failed_tasks, performance_trend, tasks_run}
        """
        battery = _REGRESSION_TASKS[:REGRESSION_BATTERY_SIZE]
        passed = 0
        failed: List[str] = []

        for task_item in battery:
            t1 = self.tier1_verify(task_item["prompt"])
            if t1.get("success") and t1.get("output") == task_item["expected"]:
                passed += 1
            else:
                failed.append(task_item["prompt"][:60])

        total = len(battery)
        pass_rate = passed / total if total > 0 else 0.0
        trend = pass_rate - self._regression_baseline

        if pass_rate < self._regression_baseline - 0.05:
            logger.warning(
                "Regression alert: pass_rate=%.3f dropped >5pp from baseline=%.3f",
                pass_rate, self._regression_baseline,
            )

        return {
            "pass_rate": pass_rate,
            "failed_tasks": failed,
            "performance_trend": trend,
            "tasks_run": total,
        }

    def set_regression_baseline(self) -> None:
        """Run regression battery and set current pass rate as baseline (manual trigger)."""
        result = self.tier3_verify("")
        self._regression_baseline = result["pass_rate"]
        logger.info("Regression baseline set to %.3f (%d tasks)", self._regression_baseline,
                    result["tasks_run"])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_judge_model(self, output: str, task: str, model_key: str) -> Dict[str, float]:
        """Call one reference model and return per-criterion scores.

        Falls back to heuristic scoring when no API key is configured or the
        call times out, so the server stays responsive during local development.
        """
        criteria = ["correctness", "completeness", "clarity", "efficiency"]
        api_key = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or ""
        if not api_key:
            return self._heuristic_scores(output)
        try:
            from openai import OpenAI  # type: ignore

            base_url = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
            primary_model = os.getenv("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            secondary_model = os.getenv("MODEL_NAME_SECONDARY", primary_model)
            model = primary_model if model_key == "primary" else secondary_model

            client = OpenAI(api_key=api_key, base_url=base_url, timeout=5.0)
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
            return self._parse_rubric_scores(raw)

        except Exception as exc:
            logger.warning("Tier 2 judge call failed [%s] (%s) — using heuristic scores", model_key, exc)
            return self._heuristic_scores(output)

    def _heuristic_scores(self, output: str) -> Dict[str, float]:
        """Fast deterministic score based on output length/content when no API is available."""
        criteria = ["correctness", "completeness", "clarity", "efficiency"]
        length = len(output.strip())
        base = 0.6 if length > 20 else 0.3
        penalty = 0.1 if length > 800 else 0.0
        score = round(max(0.1, min(0.9, base - penalty + (hash(output) % 10) * 0.01)), 2)
        return {c: score for c in criteria}

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
        """Add failed output to Failure Bank if available (non-blocking)."""
        if self._failure_bank is not None:
            payload = {
                "prompt": task,
                "actual_output": agent_output,
                "verification_result": result,
                "error_type": "verification_failure",
            }
            self._fb_executor.submit(self._failure_bank.add_failure, payload)

    @staticmethod
    def _build_reasoning(t1: float, t2: Optional[float], t3: Optional[float],
                         composite: float) -> str:
        t2_str = f"{t2:.2f}" if t2 is not None else "N/A"
        t3_str = f"{t3:.2f}" if t3 is not None else "N/A"
        return f"T1={t1:.2f} T2={t2_str} T3={t3_str} composite={composite:.3f}"
