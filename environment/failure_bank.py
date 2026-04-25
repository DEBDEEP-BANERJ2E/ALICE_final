"""
Failure Bank — indexes failed tasks by novelty and manages the repair queue.

Uses sentence-transformer embeddings and k-NN cosine similarity for novelty scoring.
"""

from __future__ import annotations

import uuid
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

MAX_ENTRIES = 10_000
NOVELTY_DUPLICATE_THRESHOLD = 0.8   # similarity > 0.8 → duplicate
NOVELTY_NOVEL_THRESHOLD = 0.5       # similarity < 0.5 → novel
REPAIR_QUEUE_NOVELTY_MIN = 0.7
KNN_K = 5


class FailureBankEntry:
    """Represents a single failure record."""

    def __init__(
        self,
        failure_id: str,
        timestamp: str,
        agent_version: str,
        error_type: str,
        prompt: str,
        expected_output: str,
        actual_output: str,
        cot_trace: str,
        trajectory: Any,
        novelty_score: float,
        semantic_embedding: np.ndarray,
        repair_priority: float = 0.0,
        repair_status: str = "pending",
    ) -> None:
        self.failure_id = failure_id
        self.timestamp = timestamp
        self.agent_version = agent_version
        self.error_type = error_type
        self.prompt = prompt
        self.expected_output = expected_output
        self.actual_output = actual_output
        self.cot_trace = cot_trace
        self.trajectory = trajectory
        self.novelty_score = novelty_score
        self.semantic_embedding = semantic_embedding
        self.repair_priority = repair_priority
        self.repair_status = repair_status
        self.failure_frequency: int = 1


class FailureBank:
    """Stores and indexes failed tasks for repair synthesis."""

    _st_model = None  # shared SentenceTransformer instance

    def __init__(self) -> None:
        self._entries: Dict[str, FailureBankEntry] = {}
        self._repair_queue: List[str] = []  # ordered failure_ids
        self._archived: List[FailureBankEntry] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_failure(self, failure: Dict[str, Any]) -> str:
        """Add a failed task with novelty scoring.

        Returns the assigned failure_id.
        """
        embedding = self._compute_embedding(failure.get("prompt", ""))
        novelty_score = self.compute_novelty_score({"embedding": embedding})

        failure_id = str(uuid.uuid4())
        entry = FailureBankEntry(
            failure_id=failure_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            agent_version=failure.get("agent_version", "0.0.0"),
            error_type=failure.get("error_type", "unknown"),
            prompt=failure.get("prompt", ""),
            expected_output=failure.get("expected_output", ""),
            actual_output=failure.get("actual_output", ""),
            cot_trace=failure.get("cot_trace", ""),
            trajectory=failure.get("trajectory"),
            novelty_score=novelty_score,
            semantic_embedding=embedding,
            repair_priority=novelty_score,  # updated below
        )

        self._entries[failure_id] = entry
        self._update_repair_queue(entry)
        self._maybe_archive()
        return failure_id

    def compute_novelty_score(self, failure: Dict[str, Any]) -> float:
        """Compute semantic similarity-based novelty score in [0, 1].

        1.0 = completely novel, 0.0 = exact duplicate.
        """
        embedding = failure.get("embedding")
        if embedding is None:
            return 1.0
        if not self._entries:
            return 1.0

        existing_embeddings = np.stack([e.semantic_embedding for e in self._entries.values()])
        similarities = self._cosine_similarity_batch(embedding, existing_embeddings)
        k = min(KNN_K, len(similarities))
        top_k_sim = float(np.mean(np.sort(similarities)[-k:]))

        if top_k_sim > NOVELTY_DUPLICATE_THRESHOLD:
            return 0.1
        elif top_k_sim < NOVELTY_NOVEL_THRESHOLD:
            return 0.9
        return 1.0 - top_k_sim

    def get_repair_candidates(self, num_pairs: int = 8) -> List[Dict[str, Any]]:
        """Return top-N failures ordered by repair_priority."""
        sorted_ids = sorted(
            self._repair_queue,
            key=lambda fid: self._entries[fid].repair_priority,
            reverse=True,
        )
        results = []
        for fid in sorted_ids[:num_pairs]:
            entry = self._entries[fid]
            results.append({
                "failure_id": entry.failure_id,
                "prompt": entry.prompt,
                "expected_output": entry.expected_output,
                "actual_output": entry.actual_output,
                "novelty_score": entry.novelty_score,
                "repair_priority": entry.repair_priority,
                "error_type": entry.error_type,
            })
        return results

    def query_failures(
        self,
        error_type: Optional[str] = None,
        agent_version: Optional[str] = None,
        time_range: Optional[Tuple[str, str]] = None,
        novelty_threshold: Optional[float] = None,
    ) -> List[FailureBankEntry]:
        """Query failures by multiple criteria."""
        results = list(self._entries.values())
        if error_type:
            results = [e for e in results if e.error_type == error_type]
        if agent_version:
            results = [e for e in results if e.agent_version == agent_version]
        if novelty_threshold is not None:
            results = [e for e in results if e.novelty_score >= novelty_threshold]
        return results

    def get_failure_distribution(self) -> Dict[str, Any]:
        """Return failure distribution metrics for dashboard display."""
        by_type: Dict[str, int] = defaultdict(int)
        for entry in self._entries.values():
            by_type[entry.error_type] += 1
        return {
            "total": len(self._entries),
            "by_error_type": dict(by_type),
            "repair_queue_size": len(self._repair_queue),
            "archived": len(self._archived),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_embedding(self, text: str) -> np.ndarray:
        """Compute sentence-transformer embedding for a text string."""
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            if not hasattr(FailureBank, "_st_model") or FailureBank._st_model is None:
                FailureBank._st_model = SentenceTransformer("all-MiniLM-L6-v2")
            emb = FailureBank._st_model.encode(text, normalize_embeddings=True)
            result = np.zeros(768, dtype=np.float32)
            n = min(len(emb), 768)
            result[:n] = emb[:n]
            return result
        except Exception:
            # Fallback to deterministic hash-based embedding when model unavailable
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            return rng.random(768).astype(np.float32)

    def _update_repair_queue(self, entry: FailureBankEntry) -> None:
        """Add entry to repair queue if novelty is high enough."""
        if entry.novelty_score >= REPAIR_QUEUE_NOVELTY_MIN:
            entry.repair_priority = entry.novelty_score * entry.failure_frequency
            self._repair_queue.append(entry.failure_id)

    def _maybe_archive(self) -> None:
        """Archive oldest entries when bank exceeds MAX_ENTRIES."""
        if len(self._entries) > MAX_ENTRIES:
            oldest_id = next(iter(self._entries))
            self._archived.append(self._entries.pop(oldest_id))
            if oldest_id in self._repair_queue:
                self._repair_queue.remove(oldest_id)
            logger.info("Archived failure %s; bank size=%d", oldest_id[:8], len(self._entries))

    @staticmethod
    def _cosine_similarity_batch(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between vec and each row of matrix."""
        vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
        matrix_norm = matrix / norms
        return matrix_norm @ vec_norm
