"""
Task Generator — Hunt mode (adversarial prompt generation) and Repair mode
(training pair synthesis from Failure Bank).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


ADVERSARIAL_STRATEGIES = [
    "semantic_perturbation",
    "logical_contradiction",
    "context_confusion",
    "boundary_testing",
]


class TaskGenerator:
    """Generates adversarial prompts (Hunt) and training pairs (Repair)."""

    def __init__(self) -> None:
        self._prompt_history: List[str] = []
        self._strategy_effectiveness: Dict[str, float] = {s: 0.0 for s in ADVERSARIAL_STRATEGIES}

    # ------------------------------------------------------------------
    # Hunt mode
    # ------------------------------------------------------------------

    def hunt_mode(
        self,
        agent_performance: Dict[str, Any],
        discrimination_zone: List[str],
    ) -> Dict[str, Any]:
        """Generate an adversarial prompt targeting the discrimination zone.

        Args:
            agent_performance: Per-task success rates and attempt counts.
            discrimination_zone: Task IDs in the 20-80% success rate range.

        Returns:
            {prompt, difficulty_score, strategy, reasoning, cot_trace}
        """
        strategy = self._select_strategy(agent_performance)
        prompt = self._generate_prompt(strategy, agent_performance, discrimination_zone)
        difficulty_score = self._compute_difficulty(prompt, agent_performance)
        reasoning = self._build_reasoning(strategy, difficulty_score)
        cot_trace = self._build_cot_trace(strategy, prompt)

        self._prompt_history.append(prompt)
        return {
            "prompt": prompt,
            "difficulty_score": difficulty_score,
            "strategy": strategy,
            "reasoning": reasoning,
            "cot_trace": cot_trace,
        }

    # ------------------------------------------------------------------
    # Repair mode
    # ------------------------------------------------------------------

    def repair_mode(
        self,
        failure_bank: Any,  # FailureBank — imported lazily to avoid circular deps
        num_pairs: int = 8,
    ) -> List[Dict[str, Any]]:
        """Synthesize training pairs from high-novelty Failure Bank entries.

        Args:
            failure_bank: FailureBank instance.
            num_pairs: Number of training pairs to synthesize.

        Returns:
            List of {prompt, solution, reasoning, priority_score} dicts.
        """
        candidates = failure_bank.get_repair_candidates(num_pairs)
        pairs: List[Dict[str, Any]] = []
        for failure in candidates:
            solution = self._call_reference_model(failure.get("prompt", ""))
            if solution is None:
                continue
            pair = {
                "prompt": failure.get("prompt", ""),
                "solution": solution,
                "reasoning": f"Repair for failure {failure.get('failure_id', '')}",
                "priority_score": failure.get("repair_priority", 0.0),
            }
            pairs.append(pair)
        pairs.sort(key=lambda p: p["priority_score"], reverse=True)
        return pairs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_strategy(self, agent_performance: Dict[str, Any]) -> str:
        """Select adversarial strategy based on agent weakness profile."""
        # Placeholder: round-robin through strategies
        idx = len(self._prompt_history) % len(ADVERSARIAL_STRATEGIES)
        return ADVERSARIAL_STRATEGIES[idx]

    def _generate_prompt(
        self,
        strategy: str,
        agent_performance: Dict[str, Any],
        discrimination_zone: List[str],
    ) -> str:
        """Generate a prompt using the selected strategy."""
        base = discrimination_zone[0] if discrimination_zone else "default_task"
        return f"[{strategy}] {base}"

    def _compute_difficulty(self, prompt: str, agent_performance: Dict[str, Any]) -> float:
        """Assign a difficulty score 0-100 based on agent performance history."""
        return 50.0  # placeholder

    def _build_reasoning(self, strategy: str, difficulty_score: float) -> str:
        """Build reasoning string explaining strategy choice."""
        return (
            f"Strategy '{strategy}' selected to target agent weaknesses. "
            f"Estimated difficulty: {difficulty_score:.1f}/100."
        )

    def _build_cot_trace(self, strategy: str, prompt: str) -> str:
        """Build chain-of-thought trace for the generated prompt."""
        return f"CoT: Applied {strategy} to construct adversarial prompt: '{prompt}'"

    def _call_reference_model(self, prompt: str) -> Optional[str]:
        """Call reference model (GPT-4o / Qwen-72B) to generate corrected solution."""
        # Placeholder — real implementation in Task 5
        return f"[reference_solution for: {prompt[:40]}]"
