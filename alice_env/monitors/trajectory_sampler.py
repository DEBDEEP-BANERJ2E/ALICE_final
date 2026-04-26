"""
Trajectory Sampler — randomly samples 5% of trajectories for anomaly detection.

Detects: reward_hacking, exploration_collapse, output_repetition, policy_divergence.
"""

from __future__ import annotations

import logging
import math
import random
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATE = 0.05
ENTROPY_COLLAPSE_THRESHOLD = 0.5
REPETITION_THRESHOLD = 0.5
ANOMALY_SCORE_ALERT = 0.7
ANOMALY_RATE_ALERT = 0.1


class TrajectorySampler:
    """Samples and analyzes trajectories for adversarial behavior."""

    def __init__(
        self,
        sample_rate: float = DEFAULT_SAMPLE_RATE,
        entropy_threshold: float = ENTROPY_COLLAPSE_THRESHOLD,
        repetition_threshold: float = REPETITION_THRESHOLD,
    ) -> None:
        self.sample_rate = sample_rate
        self.entropy_threshold = entropy_threshold
        self.repetition_threshold = repetition_threshold
        self._sampled_count: int = 0
        self._anomaly_count: int = 0
        self._incident_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_sample(self) -> bool:
        """Return True with probability equal to sample_rate."""
        return random.random() < self.sample_rate

    def analyze_trajectory(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a trajectory for anomalies.

        Returns:
            {anomaly_type, anomaly_score, trajectory_entropy, flagged}
        """
        self._sampled_count += 1
        actions = [t.get("action", "") for t in trajectory.get("turns", [])]
        rewards = [t.get("reward", 0.0) for t in trajectory.get("turns", [])]

        entropy = self.compute_trajectory_entropy(actions)
        repetition_rate = self._compute_repetition_rate(actions)
        anomaly_type: Optional[str] = None
        anomaly_score = 0.0

        if entropy < self.entropy_threshold:
            anomaly_type = "mode_collapse"
            anomaly_score = max(anomaly_score, 1.0 - entropy / self.entropy_threshold)

        if repetition_rate > self.repetition_threshold:
            anomaly_type = anomaly_type or "output_repetition"
            anomaly_score = max(anomaly_score, repetition_rate)

        if self._detect_reward_hacking(rewards):
            anomaly_type = anomaly_type or "reward_hacking"
            anomaly_score = max(anomaly_score, 0.8)

        flagged = anomaly_score >= ANOMALY_SCORE_ALERT
        if flagged:
            self._anomaly_count += 1
            self._log_incident(trajectory.get("metadata", {}).get("episode_id", "unknown"), anomaly_type, anomaly_score)

        if self.get_anomaly_rate() > ANOMALY_RATE_ALERT:
            logger.warning("Anomaly rate %.2f exceeds threshold %.2f", self.get_anomaly_rate(), ANOMALY_RATE_ALERT)

        return {
            "anomaly_type": anomaly_type,
            "anomaly_score": anomaly_score,
            "trajectory_entropy": entropy,
            "repetition_rate": repetition_rate,
            "flagged": flagged,
        }

    def compute_trajectory_entropy(self, actions: List[str]) -> float:
        """Compute Shannon entropy of action distribution."""
        if not actions:
            return 0.0
        counts: Dict[str, int] = {}
        for a in actions:
            counts[a] = counts.get(a, 0) + 1
        total = len(actions)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        # Normalize to [0, 1] by dividing by log2(total)
        max_entropy = math.log2(total) if total > 1 else 1.0
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def get_anomaly_rate(self) -> float:
        """Return anomaly_rate = anomalies / sampled_trajectories."""
        if self._sampled_count == 0:
            return 0.0
        return self._anomaly_count / self._sampled_count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_repetition_rate(self, actions: List[str]) -> float:
        if not actions:
            return 0.0
        most_common = max(set(actions), key=actions.count)
        return actions.count(most_common) / len(actions)

    def _detect_reward_hacking(self, rewards: List[float]) -> bool:
        """Detect suspiciously high rewards without corresponding task success."""
        if not rewards:
            return False
        return all(r >= 0.9 for r in rewards)

    def _log_incident(self, trajectory_id: str, anomaly_type: Optional[str], score: float) -> None:
        import time
        self._incident_log.append({
            "timestamp": time.time(),
            "trajectory_id": trajectory_id,
            "anomaly_type": anomaly_type,
            "anomaly_score": score,
        })
        logger.warning("Anomaly detected [%s] in trajectory %s (score=%.2f)", anomaly_type, trajectory_id, score)
