"""
Oracle Interface — calibrates benchmarks using reference models (GPT-4o, Qwen-72B)
with caching and rate-limit handling.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

CACHE_TTL_DAYS = 30
DIVERGENCE_THRESHOLD = 0.3
DIFFICULTY_EASY_MAX = 0.4
DIFFICULTY_HARD_MIN = 0.7
CACHE_HIT_RATE_WARNING = 0.5


class OracleInterface:
    """Calibrates task difficulty using reference model evaluations."""

    def __init__(self) -> None:
        # Cache: (task_hash, model_name) → {score, timestamp}
        self._cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._cache_hits: int = 0
        self._cache_misses: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calibrate_task(self, task: str) -> Dict[str, Any]:
        """Evaluate a task using GPT-4o and Qwen-72B reference models.

        Returns:
            {gpt4o_score, qwen72b_score, reference_performance, difficulty,
             flagged_for_review}
        """
        task_hash = self._hash_task(task)
        gpt4o_score = self._get_or_fetch_score(task_hash, "gpt4o", task)
        qwen_score = self._get_or_fetch_score(task_hash, "qwen72b", task)

        reference_performance = (gpt4o_score + qwen_score) / 2.0
        difficulty = self._assign_difficulty(reference_performance)
        flagged = abs(gpt4o_score - qwen_score) > DIVERGENCE_THRESHOLD

        if flagged:
            logger.warning(
                "Task %s flagged for manual review: gpt4o=%.2f qwen72b=%.2f",
                task_hash[:8],
                gpt4o_score,
                qwen_score,
            )

        self._log_calibration(task_hash, gpt4o_score, qwen_score, difficulty)
        return {
            "gpt4o_score": gpt4o_score,
            "qwen72b_score": qwen_score,
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
            logger.warning("Cache hit rate %.2f is below threshold %.2f", rate, CACHE_HIT_RATE_WARNING)
        return rate

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_fetch_score(self, task_hash: str, model_name: str, task: str) -> float:
        cached = self.get_cached_score(task_hash, model_name)
        if cached is not None:
            self._cache_hits += 1
            return cached
        self._cache_misses += 1
        score = self._call_reference_model(model_name, task)
        self._cache[(task_hash, model_name)] = {
            "score": score,
            "timestamp": datetime.now(timezone.utc),
        }
        return score

    def _call_reference_model(self, model_name: str, task: str) -> float:
        """Call a reference model API and return a score in [0, 1].

        Placeholder — real implementation uses OpenAI / HF Inference API.
        """
        return 0.5  # placeholder

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
        self, task_hash: str, gpt4o_score: float, qwen_score: float, difficulty: str
    ) -> None:
        logger.info(
            "Calibrated task %s: gpt4o=%.2f qwen72b=%.2f difficulty=%s",
            task_hash[:8],
            gpt4o_score,
            qwen_score,
            difficulty,
        )
