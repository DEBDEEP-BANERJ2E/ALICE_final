"""
Episode Handler — manages 3-turn episode structure with CoT reasoning.

Turn 1: initial attempt
Turn 2: CoT reflection + retry (failure feedback provided)
Turn 3: hint + final attempt
"""

from __future__ import annotations

import json
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple


MAX_TURNS = 3
TRAJECTORY_HISTORY_LIMIT = 1000


class EpisodeHandler:
    """Manages 3-turn episode lifecycle and trajectory storage."""

    def __init__(self) -> None:
        self.current_turn: int = 0
        self.max_turns: int = MAX_TURNS
        self.trajectory: Dict[str, Any] = {"turns": [], "metadata": {}}
        self._history: Deque[Dict[str, Any]] = deque(maxlen=TRAJECTORY_HISTORY_LIMIT)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize_episode(
        self,
        episode_id: str,
        agent_version: str,
        task: str,
        benchmark_version: str = "0.0.0",
        difficulty_level: int = 1,
    ) -> Dict[str, Any]:
        """Initialize a new episode with a pre-generated task.

        Returns the initial state dict.
        """
        self.current_turn = 1
        self.trajectory = {
            "turns": [],
            "metadata": {
                "episode_id": episode_id,
                "agent_version": agent_version,
                "benchmark_version": benchmark_version,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "difficulty_level": difficulty_level,
                "task": task,
            },
        }
        return {
            "episode_id": episode_id,
            "timestamp": self.trajectory["metadata"]["timestamp"],
            "task": task,
            "agent_version": agent_version,
            "turn_number": 1,
        }

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Process agent action for the current turn.

        Returns (state, reward, done, info).
        """
        if self.current_turn > self.max_turns:
            raise RuntimeError("Episode already completed — call initialize_episode first.")

        observation = self._build_observation(self.current_turn)
        reward = 0.0  # placeholder — computed by RewardFunction in Task 11
        done = self.current_turn >= self.max_turns

        turn_data: Dict[str, Any] = {
            "turn_number": self.current_turn,
            "observation": observation,
            "action": action,
            "cot_trace": "",
            "reward": reward,
            "done": done,
            "verification": None,
        }
        self.record_turn(turn_data)

        state = {
            "episode_id": self.trajectory["metadata"]["episode_id"],
            "turn_number": self.current_turn,
            "task": self.trajectory["metadata"]["task"],
            "agent_version": self.trajectory["metadata"]["agent_version"],
        }

        if done:
            self.finalize_episode()

        self.current_turn += 1
        return state, reward, done, {"turn": self.current_turn - 1}

    def record_turn(self, turn_data: Dict[str, Any]) -> None:
        """Record a turn with observation, action, reward, CoT trace, and verification."""
        self.trajectory["turns"].append(turn_data)

    def finalize_episode(self) -> Dict[str, Any]:
        """Return the complete trajectory with all metadata and statistics."""
        turns = self.trajectory["turns"]
        total_reward = sum(t.get("reward", 0.0) for t in turns)
        success_count = sum(
            1 for t in turns if t.get("verification") and t["verification"].get("composite_score", 0) >= 0.5
        )
        success_rate = success_count / len(turns) if turns else 0.0

        self.trajectory["metadata"].update({
            "total_reward": total_reward,
            "success_rate": success_rate,
            "reasoning_quality": 0.0,  # computed by downstream analysis
        })

        self._history.append(self.trajectory)
        return self.trajectory

    def get_trajectory(self, episode_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a stored trajectory by episode_id."""
        for traj in self._history:
            if traj["metadata"].get("episode_id") == episode_id:
                return traj
        return None

    def serialize_trajectory(self, trajectory: Optional[Dict[str, Any]] = None) -> str:
        """Serialize a trajectory (or current) to JSON."""
        target = trajectory or self.trajectory
        return json.dumps(target, default=str)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_observation(self, turn_number: int) -> str:
        """Build the observation string for the given turn."""
        task = self.trajectory["metadata"].get("task", "")
        if turn_number == 1:
            return f"Task: {task}"
        elif turn_number == 2:
            prev_action = self.trajectory["turns"][-1]["action"] if self.trajectory["turns"] else ""
            return f"Task: {task}\nYour previous attempt was: {prev_action}\nPlease reflect and retry."
        else:
            return f"Task: {task}\nHint: Consider edge cases carefully. Make your final attempt."
