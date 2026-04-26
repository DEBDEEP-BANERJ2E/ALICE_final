"""
MDP state representation for ALICE.

State vector: (task_embedding[768], agent_capability_vector[5], difficulty_tier,
               turn_number, failure_bank_snapshot[16×768], discrimination_coverage,
               cumulative_reward) — total 13,065 dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


STATE_DIM = 13_065
TASK_EMBED_DIM = 768
CAPABILITY_DIM = 5
FAILURE_SNAPSHOT_K = 16


@dataclass
class MDPState:
    """Fixed-dimension MDP state vector for ALICE."""

    task_embedding: np.ndarray = field(default_factory=lambda: np.zeros(TASK_EMBED_DIM))
    agent_capability_vector: np.ndarray = field(default_factory=lambda: np.zeros(CAPABILITY_DIM))
    difficulty_tier: int = 1
    turn_number: int = 1
    failure_bank_snapshot: np.ndarray = field(
        default_factory=lambda: np.zeros(FAILURE_SNAPSHOT_K * TASK_EMBED_DIM)
    )
    discrimination_coverage: float = 0.0
    cumulative_reward: float = 0.0

    def to_vector(self) -> np.ndarray:
        """Serialize state to a flat numpy array of shape (13065,)."""
        return np.concatenate([
            self.task_embedding,
            self.agent_capability_vector,
            np.array([self.difficulty_tier], dtype=np.float32),
            np.array([self.turn_number], dtype=np.float32),
            self.failure_bank_snapshot,
            np.array([self.discrimination_coverage], dtype=np.float32),
            np.array([self.cumulative_reward], dtype=np.float32),
        ]).astype(np.float32)

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> "MDPState":
        """Deserialize state from a flat numpy array of shape (13065,)."""
        if vec.shape[0] != STATE_DIM:
            raise ValueError(f"Expected vector of length {STATE_DIM}, got {vec.shape[0]}")
        offset = 0
        task_embedding = vec[offset: offset + TASK_EMBED_DIM]
        offset += TASK_EMBED_DIM
        capability = vec[offset: offset + CAPABILITY_DIM]
        offset += CAPABILITY_DIM
        difficulty_tier = int(vec[offset])
        offset += 1
        turn_number = int(vec[offset])
        offset += 1
        snapshot_len = FAILURE_SNAPSHOT_K * TASK_EMBED_DIM
        failure_snapshot = vec[offset: offset + snapshot_len]
        offset += snapshot_len
        discrimination_coverage = float(vec[offset])
        offset += 1
        cumulative_reward = float(vec[offset])
        return cls(
            task_embedding=task_embedding,
            agent_capability_vector=capability,
            difficulty_tier=difficulty_tier,
            turn_number=turn_number,
            failure_bank_snapshot=failure_snapshot,
            discrimination_coverage=discrimination_coverage,
            cumulative_reward=cumulative_reward,
        )

    @staticmethod
    def encode_task(task: str) -> np.ndarray:
        """Encode a task string to a 768-dim embedding using sentence-transformers."""
        # Lazy import to avoid loading model at module level
        from sentence_transformers import SentenceTransformer  # type: ignore
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embedding = model.encode(task, normalize_embeddings=True)
        # Pad or truncate to TASK_EMBED_DIM
        result = np.zeros(TASK_EMBED_DIM, dtype=np.float32)
        n = min(len(embedding), TASK_EMBED_DIM)
        result[:n] = embedding[:n]
        return result

    @staticmethod
    def encode_failure_bank_snapshot(top_k_failures: List[np.ndarray]) -> np.ndarray:
        """Encode top-k failure embeddings into a fixed-size snapshot with zero-padding."""
        snapshot = np.zeros(FAILURE_SNAPSHOT_K * TASK_EMBED_DIM, dtype=np.float32)
        for i, emb in enumerate(top_k_failures[:FAILURE_SNAPSHOT_K]):
            start = i * TASK_EMBED_DIM
            n = min(len(emb), TASK_EMBED_DIM)
            snapshot[start: start + n] = emb[:n]
        return snapshot
