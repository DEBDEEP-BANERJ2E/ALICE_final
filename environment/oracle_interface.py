"""
Oracle Interface — calibrates benchmarks using reference models.

Uses the HF Inference API (OpenAI-compatible) to evaluate tasks and compute
discrimination scores. Results are cached with 30-day TTL to minimize API costs.
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

CACHE_TTL_DAYS = 30
DIVERGENCE_THRESHOLD = 0.3
DIFFICULTY_EASY_MAX = 0.4
DIFFICULTY_HARD_MIN = 0.7
CACHE_HIT_RATE_WARNING = 0.5

# HF Inference API config
_API_BASE = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
_API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or "dummy"
_PRIMARY_MODEL = os.getenv("REFERENCE_MODEL_PRIMARY", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
_SECONDARY_MODEL = os.getenv("REFERENCE_MODEL_SECONDARY", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")


class OracleInterface:
    """Calibrates task difficulty using reference model evaluations."""

    def __init__(self) -> None:
        self._cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._cache_hits: int = 0
        self._cache_misses: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calibrate_task(self, task: str) -> Dict[str, Any]:
        """Evaluate a task using primary and secondary reference models.

        Returns:
            {primary_score, secondary_score, reference_performance, difficulty,
             flagged_for_review}
        """
        task_hash = self._hash_task(task)
        primary_score = self._get_or_fetch_score(task_hash, "primary", task)
        secondary_score = self._get_or_fetch_score(task_hash, "secondary", task)

        reference_performance = (primary_score + secondary_score) / 2.0
        difficulty = self._assign_difficulty(reference_performance)
        flagged = abs(primary_score - secondary_score) > DIVERGENCE_THRESHOLD

        if flagged:
            logger.warning(
                "Task %s flagged: primary=%.2f secondary=%.2f",
                task_hash[:8], primary_score, secondary_score,
            )

        self._log_calibration(task_hash, primary_score, secondary_score, difficulty)
        return {
            "gpt4o_score": primary_score,
            "qwen72b_score": secondary_score,
            "reference_performance": reference_performance,
            "difficulty": difficulty,
            "flagged_for_review": flagged,
        }

    def get_cached_score(self, task_hash: str, model_name: str) -> Optional[float]:
        """Return cached score if available and not expired."""
        entry = self._cache.get((task_hash, model_name))
        if entry is None:
            return None
        age = datetime.now(timezone.utc) - entry["timestamp"]
        if age > timedelta(days=CACHE_TTL_DAYS):
            del self._cache[(task_hash, model_name)]
            return None
        return entry["score"]

    def invalidate_cache(self, task_hash: str) -> None:
        """Invalidate all cached scores for a specific task."""
        keys_to_delete = [k for k in self._cache if k[0] == task_hash]
        for k in keys_to_delete:
            del self._cache[k]

    def get_cache_hit_rate(self) -> float:
        """Return the cache hit rate metric."""
        total = self._cache_hits + self._cache_misses
        if total == 0:
            return 0.0
        rate = self._cache_hits / total
        if rate < CACHE_HIT_RATE_WARNING:
            logger.warning("Cache hit rate %.2f below threshold %.2f", rate, CACHE_HIT_RATE_WARNING)
        return rate

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_fetch_score(self, task_hash: str, model_key: str, task: str) -> float:
        cached = self.get_cached_score(task_hash, model_key)
        if cached is not None:
            self._cache_hits += 1
            return cached
        self._cache_misses += 1
        score = self._call_reference_model(model_key, task)
        self._cache[(task_hash, model_key)] = {
            "score": score,
            "timestamp": datetime.now(timezone.utc),
        }
        return score

    def _call_reference_model(self, model_key: str, task: str) -> float:
        """Call reference model via HF Inference API and return a difficulty score in [0, 1].

        Score represents how difficult the task is for the reference model.
        Higher score = model finds it harder (useful for discrimination zone).
        """
        try:
            from openai import OpenAI  # type: ignore

            model = _PRIMARY_MODEL if model_key == "primary" else _SECONDARY_MODEL
            client = OpenAI(api_key=_API_KEY, base_url=_API_BASE)

            eval_prompt = (
                f"Rate how difficult this task is for an AI assistant on a scale of 0.0 to 1.0.\n"
                f"0.0 = trivially easy, 1.0 = extremely hard.\n\n"
                f"Task: {task[:400]}\n\n"
                f"Reply with ONLY a single decimal number between 0.0 and 1.0."
            )

            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": eval_prompt}],
                max_tokens=10,
                temperature=0.0,
            )
            raw = resp.choices[0].message.content.strip() if resp.choices else "0.5"
            score = float(raw.split()[0])
            return max(0.0, min(1.0, score))

        except Exception as exc:
            logger.warning("Reference model call failed (%s) — using default 0.5", exc)
            return 0.5

    def _assign_difficulty(self, reference_performance: float) -> str:
        if reference_performance < DIFFICULTY_EASY_MAX:
            return "easy"
        elif reference_performance <= DIFFICULTY_HARD_MIN:
            return "medium"
        return "hard"

    @staticmethod
    def _hash_task(task: str) -> str:
        return hashlib.sha256(task.encode()).hexdigest()

    def _log_calibration(
        self, task_hash: str, primary_score: float, secondary_score: float, difficulty: str
    ) -> None:
        logger.info(
            "Calibrated %s: primary=%.2f secondary=%.2f difficulty=%s",
            task_hash[:8], primary_score, secondary_score, difficulty,
        )
