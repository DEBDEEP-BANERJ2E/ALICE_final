"""
DAPO Entropy Monitor — detects policy collapse and adjusts learning rate.

Monitors policy entropy over a sliding window of 100 episodes.
Flags potential_collapse if entropy decreases > 20% over the window.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

WINDOW_SIZE = 100
COLLAPSE_DECREASE_THRESHOLD = 0.20   # 20% entropy decrease
DIVERSITY_ALERT_THRESHOLD = 0.3
LOG_INTERVAL = 10
LR_REDUCTION_FACTOR = 0.5


class EntropyMonitor:
    """Monitors policy entropy and detects mode collapse."""

    def __init__(
        self,
        initial_learning_rate: float = 1e-5,
        on_lr_change: Optional[Callable[[float], None]] = None,
    ) -> None:
        self.learning_rate = initial_learning_rate
        self._baseline_lr = initial_learning_rate
        self._on_lr_change = on_lr_change
        self._entropy_history: Deque[float] = deque(maxlen=WINDOW_SIZE)
        self._episode_count: int = 0
        self._collapsed: bool = False
        self._metrics_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_step(self, action_probs: List[float]) -> Dict[str, Any]:
        """Record policy entropy for a training step.

        Args:
            action_probs: Probability distribution over actions (must sum to ~1).

        Returns:
            {entropy, action_diversity, collapsed, learning_rate}
        """
        entropy = self.compute_policy_entropy(action_probs)
        self._entropy_history.append(entropy)
        self._episode_count += 1

        collapsed = self.detect_collapse()
        if collapsed and not self._collapsed:
            self._collapsed = True
            self.adjust_learning_rate(LR_REDUCTION_FACTOR)
            logger.warning("Policy collapse detected — learning rate reduced to %.2e", self.learning_rate)
        elif not collapsed and self._collapsed:
            self._collapsed = False
            self._restore_learning_rate()

        if self._episode_count % LOG_INTERVAL == 0:
            self._log_metrics(entropy)

        return {
            "entropy": entropy,
            "action_diversity": self._compute_action_diversity(action_probs),
            "collapsed": collapsed,
            "learning_rate": self.learning_rate,
        }

    def compute_policy_entropy(self, action_probs: List[float]) -> float:
        """Compute Shannon entropy of the action probability distribution."""
        entropy = 0.0
        for p in action_probs:
            if p > 0:
                entropy -= p * math.log(p)
        return entropy

    def detect_collapse(self) -> bool:
        """Return True if entropy has decreased > 20% over the sliding window."""
        if len(self._entropy_history) < WINDOW_SIZE:
            return False
        window = list(self._entropy_history)
        baseline = window[0]
        current = window[-1]
        if baseline <= 0:
            return False
        decrease = (baseline - current) / baseline
        return decrease > COLLAPSE_DECREASE_THRESHOLD

    def adjust_learning_rate(self, factor: float = LR_REDUCTION_FACTOR) -> None:
        """Multiply learning rate by factor."""
        self.learning_rate *= factor
        if self._on_lr_change:
            self._on_lr_change(self.learning_rate)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _restore_learning_rate(self) -> None:
        """Restore learning rate to baseline when entropy recovers."""
        self.learning_rate = self._baseline_lr
        if self._on_lr_change:
            self._on_lr_change(self.learning_rate)
        logger.info("Entropy recovered — learning rate restored to %.2e", self.learning_rate)

    def _compute_action_diversity(self, action_probs: List[float]) -> float:
        """Compute action diversity as fraction of non-negligible actions."""
        if not action_probs:
            return 0.0
        threshold = 1.0 / (len(action_probs) * 10)
        unique = sum(1 for p in action_probs if p > threshold)
        diversity = unique / len(action_probs)
        if diversity < DIVERSITY_ALERT_THRESHOLD:
            logger.warning("Low action diversity: %.2f", diversity)
        return diversity

    def _log_metrics(self, entropy: float) -> None:
        self._metrics_log.append({
            "episode": self._episode_count,
            "entropy": entropy,
            "learning_rate": self.learning_rate,
            "collapsed": self._collapsed,
        })
