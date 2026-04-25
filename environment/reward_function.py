"""
Reward Function — Bellman-shaped composite scoring.

Per-turn rewards:
  Turn 1: R₁ = R_programmatic ∈ {0, 1}
  Turn 2: R₂ = λ_judge × R_judge - attempt_decay
  Turn 3: R₃ = R_programmatic × R_regression × (1 - 2×attempt_decay)
               - repetition_penalty × I(a_t == a_{t-1})

Potential-based shaping: R̃(s,a,s') = R(s,a,s') + γΦ(s') - Φ(s)
  where Φ(s) = discrimination_coverage(s)

Final reward clamped to [-1.0, 1.0].
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

GAMMA = 0.99
LAMBDA_JUDGE = 0.8


class RewardFunction:
    """Computes composite per-turn and episode rewards."""

    def __init__(
        self,
        attempt_decay_weight: float = 0.1,
        novelty_penalty_weight: float = 0.05,
        repetition_penalty_weight: float = 0.02,
        gamma: float = GAMMA,
    ) -> None:
        self.weights = {
            "attempt_decay": attempt_decay_weight,
            "novelty_penalty": novelty_penalty_weight,
            "repetition_penalty": repetition_penalty_weight,
        }
        self.gamma = gamma
        self._weight_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_reward(self, episode_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute per-turn shaped rewards and cumulative episode reward.

        Args:
            episode_data: {
                turns: [{turn_number, action, verification, task_in_failure_bank,
                         times_task_attempted, total_tasks, prev_action,
                         discrimination_coverage_before, discrimination_coverage_after}],
                ...
            }

        Returns:
            {per_turn_rewards, cumulative_reward, shaped_rewards}
        """
        turns = episode_data.get("turns", [])
        per_turn_rewards: List[float] = []
        shaped_rewards: List[float] = []

        prev_coverage = 0.0
        for turn_data in turns:
            raw = self._compute_turn_reward(turn_data)
            coverage_after = turn_data.get("discrimination_coverage_after", prev_coverage)
            shaping = self.gamma * coverage_after - prev_coverage
            shaped = raw + shaping
            shaped = max(-1.0, min(1.0, shaped))
            per_turn_rewards.append(raw)
            shaped_rewards.append(shaped)
            prev_coverage = coverage_after

        cumulative = sum(shaped_rewards)
        cumulative = max(-1.0, min(1.0, cumulative))

        return {
            "per_turn_rewards": per_turn_rewards,
            "shaped_rewards": shaped_rewards,
            "cumulative_reward": cumulative,
        }

    def set_weights(self, weights: Dict[str, float]) -> None:
        """Update penalty weights (all must be in [0, 1])."""
        for key, val in weights.items():
            if not (0.0 <= val <= 1.0):
                logger.error("Invalid weight %s=%.3f — must be in [0, 1]", key, val)
                raise ValueError(f"Weight '{key}' must be in [0, 1], got {val}")
        self.weights.update(weights)
        self._weight_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "weights": dict(self.weights),
        })
        logger.info("Reward weights updated: %s", self.weights)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_turn_reward(self, turn_data: Dict[str, Any]) -> float:
        """Compute raw (unshapped) reward for a single turn."""
        turn_number = turn_data.get("turn_number", 1)
        verification = turn_data.get("verification") or {}
        task_in_failure_bank = turn_data.get("task_in_failure_bank", False)
        times_attempted = turn_data.get("times_task_attempted", 1)
        total_tasks = max(turn_data.get("total_tasks", 1), 1)
        prev_action = turn_data.get("prev_action", "")
        action = turn_data.get("action", "")

        r_programmatic = float(verification.get("tier1_score", 0.0))
        r_judge = float(verification.get("tier2_score") or 0.0)
        r_regression = float(verification.get("tier3_score") or 1.0)

        attempt_decay = self.weights["attempt_decay"] * (turn_number - 1)
        novelty_penalty = self.weights["novelty_penalty"] if task_in_failure_bank else 0.0
        repetition_penalty = (
            self.weights["repetition_penalty"] * (times_attempted / total_tasks)
        )

        if turn_number == 1:
            raw = r_programmatic
        elif turn_number == 2:
            raw = LAMBDA_JUDGE * r_judge - attempt_decay
        else:
            repetition_flag = 1.0 if (action == prev_action and action) else 0.0
            raw = (
                r_programmatic * r_regression * (1.0 - 2.0 * attempt_decay)
                - repetition_penalty * repetition_flag
            )

        raw -= novelty_penalty
        return max(-1.0, min(1.0, raw))
