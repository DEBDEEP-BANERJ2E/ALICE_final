# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
ALICE Environment Implementation.

ALICE (Adversarial Loop for Inter-model Co-evolutionary Environment) is a
closed-loop RL training environment implementing a hunt→repair→verify→escalate
cycle for training LLM agents.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import AliceAction, AliceObservation
    from ..environment.episode_handler import EpisodeHandler
    from ..environment.task_generator import TaskGenerator
    from ..environment.curriculum_manager import CurriculumManager
    from ..environment.reward_function import RewardFunction
    from ..environment.verifier_stack import VerifierStack
    from ..environment.failure_bank import FailureBank
except ImportError:
    from models import AliceAction, AliceObservation
    from environment.episode_handler import EpisodeHandler
    from environment.task_generator import TaskGenerator
    from environment.curriculum_manager import CurriculumManager
    from environment.reward_function import RewardFunction
    from environment.verifier_stack import VerifierStack
    from environment.failure_bank import FailureBank

logger = logging.getLogger(__name__)


class AliceEnvironment(Environment):
    """
    ALICE RL Environment implementing 3-turn episode structure with
    adversarial task generation, multi-tier verification, and curriculum learning.

    Each episode consists of exactly 3 turns:
    - Turn 1: Initial attempt with adversarially-generated task
    - Turn 2: CoT reflection + retry with failure feedback
    - Turn 3: Hint + final attempt

    The environment integrates:
    - Task Generator (Hunt/Repair modes)
    - Episode Handler (3-turn trajectory management)
    - Verifier Stack (3-tier verification)
    - Failure Bank (novelty indexing)
    - Reward Function (Bellman-shaped composite scoring)
    - Curriculum Manager (discrimination zone computation)
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, agent_version: str = "0.0.0") -> None:
        """Initialize the ALICE environment.

        Args:
            agent_version: Version identifier for the agent being trained
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._agent_version = agent_version
        self._benchmark_version = "0.0.0"

        # Initialize core components
        self._episode_handler = EpisodeHandler()
        self._task_generator = TaskGenerator()
        self._curriculum_manager = CurriculumManager()
        self._failure_bank = FailureBank()
        self._verifier_stack = VerifierStack(failure_bank=self._failure_bank)
        self._reward_function = RewardFunction()

        # Health metrics
        self._start_time = time.time()
        self._request_count = 0
        self._error_count = 0
        self._latencies: list[float] = []

        # Episode state
        self._current_task: str = ""
        self._current_episode_id: str = ""
        self._episode_start_time: float = 0.0

        logger.info("ALICE Environment initialized with agent_version=%s", agent_version)

    def reset(self) -> AliceObservation:
        """
        Reset the environment and initialize a new episode.

        Generates a new adversarial task via Hunt mode and initializes
        the 3-turn episode structure.

        Returns:
            AliceObservation with initial state containing task and episode metadata
        """
        start_time = time.time()
        self._request_count += 1

        try:
            # Generate new episode ID
            episode_id = str(uuid4())
            self._current_episode_id = episode_id
            self._episode_start_time = time.time()

            # Generate adversarial task via Hunt mode
            agent_performance = self._get_agent_performance()
            discrimination_zone = self._curriculum_manager.compute_discrimination_zone(
                agent_performance
            ).get("discrimination_zone_tasks", [])

            task_data = self._task_generator.hunt_mode(
                agent_performance=agent_performance,
                discrimination_zone=discrimination_zone,
            )
            self._current_task = task_data.get("prompt", "")

            # Initialize episode
            initial_state = self._episode_handler.initialize_episode(
                episode_id=episode_id,
                agent_version=self._agent_version,
                task=self._current_task,
                benchmark_version=self._benchmark_version,
                difficulty_level=self._curriculum_manager.difficulty_tier,
            )

            # Update internal state
            self._state = State(episode_id=episode_id, step_count=0)

            # Record latency
            latency = time.time() - start_time
            self._latencies.append(latency)

            logger.info(
                "Episode reset: episode_id=%s, task_difficulty=%.1f, turn=1",
                episode_id[:8],
                task_data.get("difficulty_score", 0.0),
            )

            return AliceObservation(
                task=self._current_task,
                turn_number=1,
                feedback="",
                hint=None,
                verification_result={},
                task_difficulty=task_data.get("difficulty_score", 0.0),
                discrimination_coverage=self._curriculum_manager.compute_discrimination_zone(
                    agent_performance
                ).get("coverage_pct", 0.0),
                done=False,
                reward=0.0,
                metadata={
                    "episode_id": episode_id,
                    "timestamp": initial_state.get("timestamp", ""),
                    "agent_version": self._agent_version,
                    "benchmark_version": self._benchmark_version,
                },
            )

        except Exception as e:
            self._error_count += 1
            logger.error("Error in reset: %s", e, exc_info=True)
            raise

    def step(self, action: AliceAction) -> AliceObservation:  # type: ignore[override]
        """
        Execute a step in the environment by processing agent's code solution.

        Processes the agent's code through:
        1. Verifier Stack (3-tier verification)
        2. Reward Function (composite scoring)
        3. Episode Handler (trajectory recording)
        4. Failure Bank (if verification fails)

        Args:
            action: AliceAction containing agent's code solution

        Returns:
            AliceObservation with verification results, reward, and next state
        """
        start_time = time.time()
        self._request_count += 1

        try:
            # Ensure episode has been initialized
            if not self._current_episode_id or self._episode_handler.current_turn == 0:
                logger.warning("Step called before reset - initializing new episode")
                self.reset()

            # Verify agent output
            verification_result = self._verifier_stack.verify(
                agent_output=action.code,
                task=self._current_task,
            )

            # Get current turn from episode handler
            current_turn = self._episode_handler.current_turn

            # Compute reward
            episode_data = self._build_episode_data(
                action=action.code,
                verification=verification_result,
                turn_number=current_turn,
            )
            reward_data = self._reward_function.compute_reward(episode_data)
            reward = reward_data.get("shaped_rewards", [0.0])[-1] if reward_data.get("shaped_rewards") else 0.0

            # Process step through episode handler
            state, _, done, info = self._episode_handler.step(action.code)

            # Update turn data with verification and reward
            if self._episode_handler.trajectory["turns"]:
                self._episode_handler.trajectory["turns"][-1].update({
                    "verification": verification_result,
                    "reward": reward,
                })

            # Update curriculum manager
            task_success = verification_result.get("composite_score", 0.0) >= 0.5
            self._curriculum_manager.update_task_performance(
                task_id=self._current_task[:50],  # Use truncated task as ID
                success=task_success,
            )

            # Handle episode completion
            if done:
                self._finalize_episode()

            # Update internal state
            self._state.step_count += 1

            # Build feedback for next turn
            feedback = self._build_feedback(verification_result, current_turn)
            hint = self._build_hint(current_turn + 1) if current_turn + 1 == 3 else None

            # Record latency
            latency = time.time() - start_time
            self._latencies.append(latency)

            logger.info(
                "Step completed: episode_id=%s, turn=%d, reward=%.3f, composite_score=%.3f, done=%s",
                self._current_episode_id[:8],
                current_turn,
                reward,
                verification_result.get("composite_score", 0.0),
                done,
            )

            return AliceObservation(
                task=self._current_task,
                turn_number=current_turn + 1 if not done else current_turn,
                feedback=feedback,
                hint=hint,
                verification_result=verification_result,
                task_difficulty=self._curriculum_manager.difficulty_tier * 10.0,
                discrimination_coverage=self._curriculum_manager.compute_discrimination_zone(
                    self._get_agent_performance()
                ).get("coverage_pct", 0.0),
                done=done,
                reward=reward,
                metadata={
                    "episode_id": self._current_episode_id,
                    "turn": current_turn,
                    "composite_score": verification_result.get("composite_score", 0.0),
                },
            )

        except Exception as e:
            self._error_count += 1
            logger.error("Error in step: %s", e, exc_info=True)
            raise

    @property
    def state(self) -> State:
        """
        Get the current environment state without side effects.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state

    def health(self) -> Dict[str, Any]:
        """
        Return system health metrics.

        Returns:
            Dictionary containing:
            - uptime: Seconds since environment initialization
            - error_rate: Ratio of errors to total requests
            - latency_p95: 95th percentile latency in seconds
            - memory_usage: Placeholder for memory usage (not implemented)
        """
        uptime = time.time() - self._start_time
        error_rate = self._error_count / max(self._request_count, 1)

        # Compute p95 latency
        if self._latencies:
            sorted_latencies = sorted(self._latencies)
            p95_index = int(len(sorted_latencies) * 0.95)
            latency_p95 = sorted_latencies[p95_index] if p95_index < len(sorted_latencies) else 0.0
        else:
            latency_p95 = 0.0

        return {
            "uptime": uptime,
            "error_rate": error_rate,
            "latency_p95": latency_p95,
            "memory_usage": 0.0,  # Placeholder
            "request_count": self._request_count,
            "error_count": self._error_count,
            "episode_id": self._current_episode_id,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_agent_performance(self) -> Dict[str, Any]:
        """Get agent performance metrics for task generation."""
        performance = {}
        for task_id, meta in self._curriculum_manager.task_metadata.items():
            success_rate = self._curriculum_manager.get_task_success_rate(task_id)
            performance[task_id] = {
                "success_rate": success_rate,
                "attempt_count": meta.get("attempt_count", 0),
                "average_reward": meta.get("average_reward", 0.0),
            }
        return performance

    def _build_episode_data(
        self,
        action: str,
        verification: Dict[str, Any],
        turn_number: int,
    ) -> Dict[str, Any]:
        """Build episode data for reward computation."""
        prev_action = ""
        if self._episode_handler.trajectory["turns"]:
            prev_action = self._episode_handler.trajectory["turns"][-1].get("action", "")

        agent_performance = self._get_agent_performance()
        total_tasks = len(agent_performance)
        times_attempted = agent_performance.get(
            self._current_task[:50], {}
        ).get("attempt_count", 1)

        # Check if task is in failure bank
        task_in_failure_bank = any(
            entry.prompt == self._current_task
            for entry in self._failure_bank._entries.values()
        )

        return {
            "turns": [{
                "turn_number": turn_number,
                "action": action,
                "verification": verification,
                "task_in_failure_bank": task_in_failure_bank,
                "times_task_attempted": times_attempted,
                "total_tasks": max(total_tasks, 1),
                "prev_action": prev_action,
                "discrimination_coverage_before": 0.0,
                "discrimination_coverage_after": 0.0,
            }]
        }

    def _build_feedback(self, verification: Dict[str, Any], turn_number: int) -> str:
        """Build feedback message based on verification results."""
        composite_score = verification.get("composite_score", 0.0)
        reasoning = verification.get("reasoning", "")

        if composite_score >= 0.5:
            return f"Turn {turn_number} succeeded! Score: {composite_score:.2f}"
        else:
            tier1_details = verification.get("tier1_details", {})
            if not tier1_details.get("success"):
                error_msg = tier1_details.get("error_message", "Unknown error")
                return f"Turn {turn_number} failed: {error_msg}. Please fix the code and try again."
            else:
                return f"Turn {turn_number} failed verification. Score: {composite_score:.2f}. {reasoning}"

    def _build_hint(self, turn_number: int) -> Optional[str]:
        """Build hint for turn 3."""
        if turn_number == 3:
            return "Hint: Consider edge cases carefully. Check for boundary conditions and error handling."
        return None

    def _finalize_episode(self) -> None:
        """Finalize episode and update curriculum."""
        trajectory = self._episode_handler.finalize_episode()

        # Check for escalation
        if self._curriculum_manager.should_escalate():
            self._curriculum_manager.escalate()
            logger.info(
                "Curriculum escalated to difficulty tier %d",
                self._curriculum_manager.difficulty_tier,
            )

        logger.info(
            "Episode finalized: episode_id=%s, total_reward=%.3f, success_rate=%.2f",
            self._current_episode_id[:8],
            trajectory["metadata"].get("total_reward", 0.0),
            trajectory["metadata"].get("success_rate", 0.0),
        )
