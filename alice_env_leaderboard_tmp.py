"""
ALICE Leaderboard — tracks RL benchmark scores for 5 reference models
plus any user-submitted models.

Scoring is a composite of:
  - avg_reward        (weight 0.5)
  - success_rate      (weight 0.3)
  - discrimination_coverage  (weight 0.2)

All values stored in memory; persisted to LEADERBOARD_PATH on every write.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

LEADERBOARD_PATH = Path(os.getenv("LEADERBOARD_PATH", "data/leaderboard.json"))

# ---------------------------------------------------------------------------
# Reference models seeded with initial estimates
# (will be overwritten as real training results come in)
# ---------------------------------------------------------------------------

BENCHMARK_MODELS: List[Dict] = [
    {
        "model_id":   "Qwen/Qwen2.5-0.5B-Instruct",
        "display_name": "Qwen2.5-0.5B",
        "params_b":   0.5,
        "avg_reward": 1.6549,
        "success_rate": 0.7375,
        "discrimination_coverage": 0.7362,
        "episodes_run": 150,
        "source":     "benchmark",
    },
    {
        "model_id":   "Qwen/Qwen2.5-1.5B-Instruct",
        "display_name": "Qwen2.5-1.5B",
        "params_b":   1.5,
        "avg_reward": 0.8515,
        "success_rate": 0.1775,
        "discrimination_coverage": 0.145,
        "episodes_run": 50,
        "source":     "benchmark",
    },
    {
        "model_id":   "Qwen/Qwen2.5-3B-Instruct",
        "display_name": "Qwen2.5-3B",
        "params_b":   3.0,
        "avg_reward": 1.4278,
        "success_rate": 0.58,
        "discrimination_coverage": 0.255,
        "episodes_run": 50,
        "source":     "benchmark",
    },
    {
        "model_id":   "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "display_name": "SmolLM2-1.7B",
        "params_b":   1.7,
        "avg_reward": 0.5279,
        "success_rate": 0.0825,
        "discrimination_coverage": 0.0625,
        "episodes_run": 50,
        "source":     "benchmark",
    },
    {
        "model_id":   "google/gemma-3-1b-it",
        "display_name": "Gemma-3-1B",
        "params_b":   1.0,
        "avg_reward": 0.0,
        "success_rate": 0.0,
        "discrimination_coverage": 0.0,
        "episodes_run": 50,
        "source":     "benchmark",
    },
]


@dataclass
class ModelEntry:
    model_id:                str
    display_name:            str
    params_b:                float
    avg_reward:              float
    success_rate:            float
    discrimination_coverage: float
    episodes_run:            int
    source:                  str        # "benchmark" | "user"
    submitted_at:            str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    rl_score:                float = 0.0

    def __post_init__(self):
        self.rl_score = self._compute_score()

    def _compute_score(self) -> float:
        return round(
            0.5 * self.avg_reward
            + 0.3 * self.success_rate
            + 0.2 * self.discrimination_coverage,
            4,
        )

    def update(self, avg_reward: float, success_rate: float,
               discrimination_coverage: float, episodes_run: int):
        self.avg_reward              = round(avg_reward, 4)
        self.success_rate            = round(success_rate, 4)
        self.discrimination_coverage = round(discrimination_coverage, 4)
        self.episodes_run            = episodes_run
        self.rl_score                = self._compute_score()


class Leaderboard:
    def __init__(self):
        self._lock    = threading.Lock()
        self._entries: Dict[str, ModelEntry] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self):
        LEADERBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Always seed benchmark defaults first so episodes_run is never 0
        _benchmark_defaults = {d["model_id"]: d for d in BENCHMARK_MODELS}
        for d in BENCHMARK_MODELS:
            self._entries[d["model_id"]] = ModelEntry(**d)
        # Merge persisted data on top, but keep the higher episodes_run
        if LEADERBOARD_PATH.exists():
            try:
                raw = json.loads(LEADERBOARD_PATH.read_text())
                for d in raw:
                    mid = d.get("model_id", "")
                    if mid in _benchmark_defaults:
                        d["episodes_run"] = max(
                            int(d.get("episodes_run", 0)),
                            _benchmark_defaults[mid]["episodes_run"],
                        )
                    self._entries[mid] = ModelEntry(**d)
            except Exception:
                pass
        self._save()

    def _save(self):
        LEADERBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
        LEADERBOARD_PATH.write_text(
            json.dumps([asdict(e) for e in self._entries.values()], indent=2)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_leaderboard(self, model_ids: Optional[List[str]] = None) -> List[dict]:
        """Return entries sorted by rl_score desc, optionally filtered."""
        with self._lock:
            entries = list(self._entries.values())
        if model_ids:
            entries = [e for e in entries if e.model_id in model_ids]
        entries.sort(key=lambda e: e.rl_score, reverse=True)
        return [
            {
                "rank":                    i + 1,
                "model_id":                e.model_id,
                "display_name":            e.display_name,
                "params_b":                e.params_b,
                "rl_score":                e.rl_score,
                "avg_reward":              e.avg_reward,
                "success_rate":            e.success_rate,
                "discrimination_coverage": e.discrimination_coverage,
                "episodes_run":            e.episodes_run,
                "source":                  e.source,
                "submitted_at":            e.submitted_at,
            }
            for i, e in enumerate(entries)
        ]

    def update_model_score(
        self,
        model_id: str,
        avg_reward: float,
        success_rate: float,
        discrimination_coverage: float,
        episodes_run: int,
    ):
        """Called by training scripts to push live results into the leaderboard."""
        with self._lock:
            if model_id in self._entries:
                self._entries[model_id].update(avg_reward, success_rate,
                                               discrimination_coverage, episodes_run)
            else:
                # Auto-register unknown model (e.g. user-submitted mid-training)
                e = ModelEntry(
                    model_id=model_id,
                    display_name=model_id.split("/")[-1],
                    params_b=0.0,
                    avg_reward=avg_reward,
                    success_rate=success_rate,
                    discrimination_coverage=discrimination_coverage,
                    episodes_run=episodes_run,
                    source="user",
                )
                self._entries[model_id] = e
            self._save()

    def submit_model(
        self,
        model_id: str,
        display_name: Optional[str] = None,
        params_b: float = 0.0,
    ) -> dict:
        """Register a user-submitted model for comparison."""
        with self._lock:
            if model_id in self._entries:
                return {"status": "already_exists", "model_id": model_id}
            e = ModelEntry(
                model_id=model_id,
                display_name=display_name or model_id.split("/")[-1],
                params_b=params_b,
                avg_reward=0.0,
                success_rate=0.0,
                discrimination_coverage=0.0,
                episodes_run=0,
                source="user",
            )
            self._entries[model_id] = e
            self._save()
        return {"status": "submitted", "model_id": model_id}

    def all_model_ids(self) -> List[str]:
        with self._lock:
            return list(self._entries.keys())

    def benchmark_model_ids(self) -> List[str]:
        with self._lock:
            return [e.model_id for e in self._entries.values() if e.source == "benchmark"]