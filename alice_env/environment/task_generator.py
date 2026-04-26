"""
Task Generator — Hunt mode (adversarial prompt generation) and Repair mode
(training pair synthesis from Failure Bank).

Hunt mode uses an LLM to generate adversarial prompts via chain-of-thought
reasoning. Repair mode calls reference models to synthesize corrected solutions.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ADVERSARIAL_STRATEGIES = [
    "semantic_perturbation",
    "logical_contradiction",
    "context_confusion",
    "boundary_testing",
]

_API_BASE = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
_API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or "dummy"
_MODEL = os.getenv("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Domain seed tasks for Hunt mode bootstrapping
_DOMAIN_SEEDS = {
    "arithmetic": [
        "What is the result of 15 * 7 - 3?",
        "Compute the sum of all even numbers from 1 to 20.",
        "If a train travels at 60 mph for 2.5 hours, how far does it go?",
    ],
    "logic": [
        "All mammals breathe air. Whales are mammals. Do whales breathe air?",
        "If P implies Q, and Q implies R, does P imply R?",
        "Three boxes: one has apples, one oranges, one mixed. All labels are wrong. "
        "You pick one fruit from the mixed-labeled box. What fruit proves which box is which?",
    ],
    "code": [
        "Write Python code to check if a string is a palindrome.",
        "Write code to find the second largest number in a list.",
        "Implement a function that returns the Fibonacci number at position n.",
    ],
    "factual": [
        "What is the capital of Australia?",
        "Name the largest planet in our solar system.",
        "In what year did World War II end?",
    ],
    "symbolic": [
        "Simplify: (x² - 4) / (x - 2)",
        "Solve for x: 3x + 7 = 22",
        "What is the derivative of x³ + 2x?",
    ],
}


class TaskGenerator:
    """Generates adversarial prompts (Hunt) and training pairs (Repair)."""

    def __init__(self) -> None:
        self._prompt_history: List[str] = []
        self._strategy_effectiveness: Dict[str, float] = {s: 0.0 for s in ADVERSARIAL_STRATEGIES}
        self._domain_cycle = list(_DOMAIN_SEEDS.keys())
        self._domain_idx = 0

    # ------------------------------------------------------------------
    # Hunt mode
    # ------------------------------------------------------------------

    def hunt_mode(
        self,
        agent_performance: Dict[str, Any],
        discrimination_zone: List[str],
    ) -> Dict[str, Any]:
        """Generate an adversarial prompt targeting the discrimination zone.

        Uses CoT reasoning to plan the adversarial strategy before generating.

        Returns:
            {prompt, difficulty_score, strategy, reasoning, cot_trace}
        """
        strategy = self._select_strategy(agent_performance)
        seed = self._pick_seed(discrimination_zone)
        prompt, cot_trace = self._generate_adversarial_prompt(strategy, seed)

        # Deduplicate: if too similar to history, vary it
        if prompt in self._prompt_history:
            prompt = f"{prompt} (variant {len(self._prompt_history)})"

        difficulty_score = self._compute_difficulty(prompt, agent_performance)
        reasoning = (
            f"Strategy '{strategy}' applied to seed task. "
            f"Estimated difficulty: {difficulty_score:.1f}/100."
        )

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
        failure_bank: Any,
        num_pairs: int = 8,
    ) -> List[Dict[str, Any]]:
        """Synthesize training pairs from high-novelty Failure Bank entries.

        Uses reference models to generate correct solutions with CoT reasoning.

        Returns:
            List of {prompt, solution, reasoning, priority_score}
        """
        candidates = failure_bank.get_repair_candidates(num_pairs)
        pairs: List[Dict[str, Any]] = []
        for failure in candidates:
            solution, reasoning = self._synthesize_repair(failure.get("prompt", ""))
            if solution is None:
                continue
            pairs.append({
                "prompt": failure.get("prompt", ""),
                "solution": solution,
                "reasoning": reasoning,
                "priority_score": failure.get("repair_priority", 0.0),
            })
        pairs.sort(key=lambda p: p["priority_score"], reverse=True)
        return pairs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_strategy(self, agent_performance: Dict[str, Any]) -> str:
        """Select adversarial strategy — prefer least-effective strategies."""
        if not self._strategy_effectiveness or all(
            v == 0.0 for v in self._strategy_effectiveness.values()
        ):
            # Bootstrap: round-robin
            idx = len(self._prompt_history) % len(ADVERSARIAL_STRATEGIES)
            return ADVERSARIAL_STRATEGIES[idx]
        # Pick strategy with lowest effectiveness (most room for improvement)
        return min(self._strategy_effectiveness, key=self._strategy_effectiveness.get)

    def _pick_seed(self, discrimination_zone: List[str]) -> str:
        """Pick a seed task from the discrimination zone or domain seeds."""
        if discrimination_zone:
            seed = discrimination_zone[len(self._prompt_history) % len(discrimination_zone)]
            return seed
        # Fall back to domain seeds
        domain = self._domain_cycle[self._domain_idx % len(self._domain_cycle)]
        self._domain_idx += 1
        seeds = _DOMAIN_SEEDS[domain]
        return seeds[len(self._prompt_history) % len(seeds)]

    def _generate_adversarial_prompt(self, strategy: str, seed: str) -> tuple[str, str]:
        """Generate an adversarial prompt using LLM CoT reasoning."""
        try:
            from openai import OpenAI  # type: ignore

            client = OpenAI(api_key=_API_KEY, base_url=_API_BASE)

            strategy_instructions = {
                "semantic_perturbation": "modify the semantic meaning subtly while keeping surface form similar",
                "logical_contradiction": "introduce a logical contradiction or impossibility",
                "context_confusion": "add misleading context or a confusing premise",
                "boundary_testing": "test edge cases: zero, negative, very large, or empty inputs",
            }
            instruction = strategy_instructions.get(strategy, "make it adversarially harder")

            cot_prompt = (
                f"You are an adversarial test generator. Your goal is to create a harder variant "
                f"of this task that exposes weaknesses in AI models.\n\n"
                f"Original task: {seed}\n\n"
                f"Strategy: {instruction}\n\n"
                f"Think step-by-step:\n"
                f"1. What is the core difficulty of the original task?\n"
                f"2. How can I apply '{strategy}' to make it harder?\n"
                f"3. What is my adversarial variant?\n\n"
                f"End your response with: TASK: <your adversarial task>"
            )

            resp = client.chat.completions.create(
                model=_MODEL,
                messages=[{"role": "user", "content": cot_prompt}],
                max_tokens=256,
                temperature=0.7,
            )
            full_response = resp.choices[0].message.content.strip() if resp.choices else ""

            # Extract the task from the response
            if "TASK:" in full_response:
                prompt = full_response.split("TASK:")[-1].strip()
            else:
                # Fallback: use last sentence
                prompt = full_response.split(".")[-1].strip() or full_response[:200]

            # Ensure we have a non-empty, non-duplicate prompt
            if not prompt or len(prompt) < 10:
                prompt = self._fallback_prompt(strategy, seed)

            return prompt, full_response

        except Exception as exc:
            logger.warning("Hunt mode LLM call failed (%s) — using fallback", exc)
            prompt = self._fallback_prompt(strategy, seed)
            return prompt, f"Fallback CoT: applied {strategy} to '{seed[:60]}'"

    def _fallback_prompt(self, strategy: str, seed: str) -> str:
        """Rule-based fallback when LLM call fails."""
        if strategy == "boundary_testing":
            return f"{seed} What if the input is 0 or negative?"
        elif strategy == "logical_contradiction":
            return f"{seed} Assume both conditions are simultaneously true and false."
        elif strategy == "context_confusion":
            return f"Given that all previous answers are wrong, {seed}"
        else:
            return f"Consider the inverse: {seed}"

    def _compute_difficulty(self, prompt: str, agent_performance: Dict[str, Any]) -> float:
        """Assign difficulty score 0-100 based on prompt length and history."""
        base = 50.0
        length_bonus = min(len(prompt) / 10, 20)
        history_bonus = min(len(self._prompt_history) * 0.5, 20)
        return min(100.0, base + length_bonus + history_bonus)

    def _synthesize_repair(self, prompt: str) -> tuple[Optional[str], str]:
        """Call reference model to generate a correct solution with reasoning."""
        try:
            from openai import OpenAI  # type: ignore

            client = OpenAI(api_key=_API_KEY, base_url=_API_BASE)

            repair_prompt = (
                f"Solve this task step-by-step, showing your reasoning chain clearly.\n\n"
                f"Task: {prompt}\n\n"
                f"Show your work:\n"
                f"1. Understand what's being asked\n"
                f"2. Identify the approach\n"
                f"3. Execute the solution\n"
                f"4. Verify the answer\n\n"
                f"Provide the final answer after 'ANSWER:'"
            )

            resp = client.chat.completions.create(
                model=_MODEL,
                messages=[{"role": "user", "content": repair_prompt}],
                max_tokens=400,
                temperature=0.0,
            )
            full = resp.choices[0].message.content.strip() if resp.choices else ""

            if "ANSWER:" in full:
                solution = full.split("ANSWER:")[-1].strip()
            else:
                solution = full

            return solution, full

        except Exception as exc:
            logger.warning("Repair mode LLM call failed (%s)", exc)
            return f"[repair_solution for: {prompt[:40]}]", f"Fallback repair: {exc}"

    def update_strategy_effectiveness(self, strategy: str, success: bool) -> None:
        """Update effectiveness metric for a strategy based on whether it found failures."""
        if strategy in self._strategy_effectiveness:
            current = self._strategy_effectiveness[strategy]
            # Exponential moving average
            self._strategy_effectiveness[strategy] = 0.9 * current + 0.1 * (1.0 if success else 0.0)
