"""
Curriculum Manager — computes discrimination zones and manages task difficulty
progression via co-evolutionary escalation.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np


WINDOW_SIZE = 100
MIN_EPISODES_BETWEEN_ESCALATIONS = 50
DISCRIMINATION_LOW = 0.2
DISCRIMINATION_HIGH = 0.8
ESCALATION_THRESHOLD = 0.1
ZONE_COVERAGE_LOW = 0.3
ZONE_COVERAGE_HIGH = 0.7


class CurriculumManager:
    """Manages curriculum difficulty and co-evolutionary escalation."""

    def __init__(self) -> None:
        self.difficulty_tier: int = 1
        self.task_performance: Dict[str, Deque[float]] = {}  # task_id → recent success flags
        self.task_metadata: Dict[str, Dict[str, Any]] = {}
        self._agent_improvement_score: float = 0.0
        self._benchmark_improvement_score: float = 0.0
        self._episodes_since_escalation: int = 0
        self._change_log: List[Dict[str, Any]] = []
        self._manual_override: Optional[int] = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def compute_discrimination_zone(
        self, task_performance: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Categorize tasks as too_easy / discrimination_zone / too_hard.

        Args:
            task_performance: {task_id: {"success_rate": float, ...}}

        Returns:
            {discrimination_zone_tasks, too_easy, too_hard, coverage_pct}
        """
        discrimination_zone: List[str] = []
        too_easy: List[str] = []
        too_hard: List[str] = []

        for task_id, metrics in task_performance.items():
            sr = metrics.get("success_rate", 0.0)
            if sr < DISCRIMINATION_LOW:
                too_easy.append(task_id)
            elif sr > DISCRIMINATION_HIGH:
                too_hard.append(task_id)
            else:
                discrimination_zone.append(task_id)

        total = len(task_performance)
        coverage_pct = len(discrimination_zone) / total if total > 0 else 0.0

        return {
            "discrimination_zone_tasks": discrimination_zone,
            "too_easy": too_easy,
            "too_hard": too_hard,
            "coverage_pct": coverage_pct,
        }

    def should_escalate(self) -> bool:
        """Return True iff both agent and benchmark improvement exceed threshold."""
        if self._episodes_since_escalation < MIN_EPISODES_BETWEEN_ESCALATIONS:
            return False
        return (
            self._agent_improvement_score > ESCALATION_THRESHOLD
            and self._benchmark_improvement_score > ESCALATION_THRESHOLD
        )

    def escalate(self) -> None:
        """Increase difficulty by 1 level and reset improvement scores."""
        self.difficulty_tier += 1
        self._agent_improvement_score = 0.0
        self._benchmark_improvement_score = 0.0
        self._episodes_since_escalation = 0
        self._log_change("escalation", f"Escalated to difficulty tier {self.difficulty_tier}")

    def get_curriculum_heatmap(self) -> np.ndarray:
        """Return a heatmap array of task difficulty distribution (5 domains × 10 tiers)."""
        heatmap = np.zeros((5, 10), dtype=np.float32)
        for task_id, history in self.task_performance.items():
            if not history:
                continue
            sr = float(np.mean(list(history)))
            # Map task_id hash to a domain row (0-4) and use difficulty_tier as column
            domain_idx = hash(task_id) % 5
            tier_idx = min(self.difficulty_tier - 1, 9)
            heatmap[domain_idx, tier_idx] = max(heatmap[domain_idx, tier_idx], sr)
        return heatmap

    def detect_plateau(self) -> bool:
        """Return True if no improvement has occurred in the last 100 episodes."""
        if len(self._change_log) == 0:
            return self._episodes_since_escalation >= WINDOW_SIZE
        return False

    def update_task_performance(self, task_id: str, success: bool) -> None:
        """Record a task outcome and update sliding-window metrics."""
        if task_id not in self.task_performance:
            self.task_performance[task_id] = deque(maxlen=WINDOW_SIZE)
            self.task_metadata[task_id] = {
                "attempt_count": 0,
                "average_reward": 0.0,
                "last_attempted": None,
            }
        self.task_performance[task_id].append(1.0 if success else 0.0)
        meta = self.task_metadata[task_id]
        meta["attempt_count"] += 1
        meta["last_attempted"] = datetime.now(timezone.utc).isoformat()
        self._episodes_since_escalation += 1

    def get_task_success_rate(self, task_id: str) -> float:
        """Return the sliding-window success rate for a task."""
        history = self.task_performance.get(task_id)
        if not history:
            return 0.0
        return float(np.mean(list(history)))

    def set_improvement_scores(
        self, agent_score: float, benchmark_score: float
    ) -> None:
        """Update improvement scores used for escalation decisions."""
        self._agent_improvement_score = agent_score
        self._benchmark_improvement_score = benchmark_score

    def set_manual_override(self, difficulty_tier: Optional[int]) -> None:
        """Manually override the curriculum difficulty tier."""
        self._manual_override = difficulty_tier
        if difficulty_tier is not None:
            self.difficulty_tier = difficulty_tier
            self._log_change("manual_override", f"Manual override to tier {difficulty_tier}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_change(self, change_type: str, justification: str) -> None:
        self._change_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": change_type,
            "justification": justification,
            "difficulty_tier": self.difficulty_tier,
        })
