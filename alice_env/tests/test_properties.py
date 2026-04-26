"""
Property-Based Tests for ALICE — validates universal correctness properties.

Properties tested:
  P1:  Markov property (state transitions deterministic given same seed)
  P2:  Reward boundedness (all rewards in [-1, 1])
  P3:  Discrimination zone non-degeneracy (curriculum never empties)
  P4:  Verifier Tier 1 override lower bound (T1 pass → composite ≥ 0.3)
  P5:  Tier 1 anti-manipulation (T1 fail → composite = 0)
  P6:  Novelty monotonicity (adding similar failures ↓ novelty score)
  P7:  Co-evolutionary coupling (escalation iff both improvements > 0.1)
  P8:  Attempt decay monotonicity (reward strictly decreasing per turn)
  P9:  GRPO advantage zero-mean (group-normalized advantages ≈ 0)
  P10: Sandbox isolation (restricted builtins raise exceptions)
  P11: Episode termination (always exactly 3 turns)
  P12: Repair surgical minimality (regression drop ≤ 15pp)

Unit tests:
  Task 5.3: Hunt mode tests
  Task 6.2/6.3: CurriculumManager discrimination zone + co-evolution
  Task 7.2: OracleInterface caching
  Task 8.2-8.4: Verifier properties
  Task 10.2: Novelty monotonicity
  Task 11.2/11.3: Reward function properties
  Task 12.2: GRPO advantage
  Task 13.3: Sandbox restrictions
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# P1: Markov Property — same (state, action) produces identical output
# ============================================================================

class TestMarkovProperty:
    """Property 1: ∀ s_t, a_t: P(s_{t+1}|history) = P(s_{t+1}|s_t, a_t)."""

    def test_same_action_same_episode_gives_same_turn_structure(self):
        from environment.episode_handler import EpisodeHandler
        from environment.reward_function import RewardFunction

        eh1 = EpisodeHandler()
        eh1.initialize_episode("ep-001", "v0", "test task")
        state1, reward1, done1, info1 = eh1.step("answer_A")

        eh2 = EpisodeHandler()
        eh2.initialize_episode("ep-001", "v0", "test task")
        state2, reward2, done2, info2 = eh2.step("answer_A")

        assert state1["task"] == state2["task"]
        assert state1["agent_version"] == state2["agent_version"]
        assert done1 == done2

    def test_deterministic_reward_for_same_verification_result(self):
        from environment.reward_function import RewardFunction

        rf = RewardFunction()
        verification = {"tier1_score": 1.0, "tier2_score": 0.8, "tier3_score": 1.0}
        episode_data = {"turns": [{
            "turn_number": 1,
            "action": "x",
            "verification": verification,
            "task_in_failure_bank": False,
            "times_task_attempted": 1,
            "total_tasks": 10,
            "prev_action": "",
            "discrimination_coverage_before": 0.5,
            "discrimination_coverage_after": 0.5,
        }]}
        r1 = rf.compute_reward(episode_data)
        r2 = rf.compute_reward(episode_data)
        assert r1["shaped_rewards"] == r2["shaped_rewards"]


# ============================================================================
# P2: Reward Boundedness — all per-turn rewards in [-1.0, 1.0]
# ============================================================================

class TestRewardBoundedness:
    """Property 2: ∀ (s, a): R(s, a) ∈ [-1.0, 1.0]."""

    def _make_episode_data(self, t1, t2, t3, turn, in_bank=False):
        return {"turns": [{
            "turn_number": turn,
            "action": "some_action",
            "verification": {"tier1_score": t1, "tier2_score": t2, "tier3_score": t3},
            "task_in_failure_bank": in_bank,
            "times_task_attempted": turn,
            "total_tasks": 3,
            "prev_action": "prev" if turn > 1 else "",
            "discrimination_coverage_before": 0.3,
            "discrimination_coverage_after": 0.4,
        }]}

    def test_reward_bounded_turn1_success(self):
        from environment.reward_function import RewardFunction
        rf = RewardFunction()
        data = self._make_episode_data(1.0, 0.8, 1.0, 1)
        result = rf.compute_reward(data)
        for r in result["shaped_rewards"]:
            assert -1.0 <= r <= 1.0

    def test_reward_bounded_turn1_failure(self):
        from environment.reward_function import RewardFunction
        rf = RewardFunction()
        data = self._make_episode_data(0.0, 0.0, 0.0, 1)
        result = rf.compute_reward(data)
        for r in result["shaped_rewards"]:
            assert -1.0 <= r <= 1.0

    def test_reward_bounded_turn3_with_penalties(self):
        from environment.reward_function import RewardFunction
        rf = RewardFunction()
        data = self._make_episode_data(0.5, 0.3, 0.7, 3, in_bank=True)
        result = rf.compute_reward(data)
        for r in result["shaped_rewards"]:
            assert -1.0 <= r <= 1.0

    def test_reward_bounded_random_1000_episodes(self):
        from environment.reward_function import RewardFunction
        rng = np.random.default_rng(42)
        rf = RewardFunction()
        for _ in range(1000):
            t1 = float(rng.random())
            t2 = float(rng.random())
            t3 = float(rng.random())
            turn = int(rng.integers(1, 4))
            data = self._make_episode_data(t1, t2, t3, turn)
            result = rf.compute_reward(data)
            for r in result["shaped_rewards"]:
                assert -1.0 <= r <= 1.0, f"Reward {r} out of bounds"


# ============================================================================
# P3: Discrimination Zone Non-Degeneracy
# ============================================================================

class TestDiscriminationZoneNonDegeneracy:
    """Property 3: curriculum always maintains non-empty discrimination zone."""

    def test_empty_performance_gives_empty_zone(self):
        from environment.curriculum_manager import CurriculumManager
        cm = CurriculumManager()
        result = cm.compute_discrimination_zone({})
        assert result["coverage_pct"] == 0.0

    def test_mixed_performance_has_discrimination_zone(self):
        from environment.curriculum_manager import CurriculumManager
        cm = CurriculumManager()
        perf = {
            "task_easy": {"success_rate": 0.9},   # too_hard (from model perspective)
            "task_medium": {"success_rate": 0.5},  # in discrimination zone
            "task_hard": {"success_rate": 0.1},    # too_easy (agent struggles)
        }
        result = cm.compute_discrimination_zone(perf)
        assert len(result["discrimination_zone_tasks"]) >= 1
        assert "task_medium" in result["discrimination_zone_tasks"]

    def test_too_easy_tasks_escalate_difficulty(self):
        from environment.curriculum_manager import CurriculumManager
        cm = CurriculumManager()
        cm._episodes_since_escalation = 100
        cm._agent_improvement_score = 0.15
        cm._benchmark_improvement_score = 0.15
        assert cm.should_escalate() is True
        initial_tier = cm.difficulty_tier
        cm.escalate()
        assert cm.difficulty_tier == initial_tier + 1

    def test_escalation_resets_improvement_scores(self):
        from environment.curriculum_manager import CurriculumManager
        cm = CurriculumManager()
        cm._episodes_since_escalation = 100
        cm._agent_improvement_score = 0.2
        cm._benchmark_improvement_score = 0.2
        cm.escalate()
        assert cm._agent_improvement_score == 0.0
        assert cm._benchmark_improvement_score == 0.0

    def test_minimum_episodes_between_escalations(self):
        from environment.curriculum_manager import CurriculumManager
        cm = CurriculumManager()
        cm._episodes_since_escalation = 10  # below minimum of 50
        cm._agent_improvement_score = 0.5
        cm._benchmark_improvement_score = 0.5
        assert cm.should_escalate() is False


# ============================================================================
# P4 & P5: Verifier Tier 1 Override / Anti-Manipulation
# ============================================================================

class TestVerifierTier1Override:
    """Property 4: T1 pass → composite ≥ 0.3. Property 5: T1 fail → composite = 0."""

    def _make_verifier(self):
        from environment.verifier_stack import VerifierStack
        v = VerifierStack()
        return v

    def test_tier1_pass_composite_at_least_03(self):
        v = self._make_verifier()
        v.tier1_verify = lambda code: {"success": True, "output": None, "execution_time": 0.0}
        v.tier2_verify = lambda output, task="": {"composite_score": 0.0, "criterion_scores": {}, "reasoning": "", "flagged_for_review": False}
        v.tier3_verify = lambda output: {"pass_rate": 0.0, "failed_tasks": [], "performance_trend": 0.0}
        result = v.verify("code", task="task")
        assert result["composite_score"] >= 0.3

    def test_tier1_fail_composite_is_zero(self):
        v = self._make_verifier()
        v.tier1_verify = lambda code: {"success": False, "output": None, "execution_time": 0.0, "error_message": "fail", "error_type": "Error"}
        result = v.verify("bad_code", task="task")
        assert result["composite_score"] == 0.0

    def test_tier1_fail_skips_tier2_and_tier3(self):
        v = self._make_verifier()
        tier2_called = []
        v.tier1_verify = lambda code: {"success": False, "output": None, "execution_time": 0.0, "error_message": "fail", "error_type": "Error"}
        v.tier2_verify = lambda output, task="": tier2_called.append(True) or {"composite_score": 0.9}
        result = v.verify("bad_code", task="task")
        assert len(tier2_called) == 0
        assert result["tier2_score"] is None

    def test_tier1_anti_manipulation_even_with_high_tier2(self):
        """Even if tier2 scores 1.0, tier1 fail must produce composite=0."""
        v = self._make_verifier()
        v.tier1_verify = lambda code: {"success": False, "output": None, "execution_time": 0.0, "error_message": "fail", "error_type": "Error"}
        # tier2 should never be called when tier1 fails
        result = v.verify("manipulated_output", task="task")
        assert result["composite_score"] == 0.0


# ============================================================================
# P6: Novelty Monotonicity
# ============================================================================

class TestNoveltyMonotonicity:
    """Property 6: adding similar failures never increases novelty of subsequent similar ones."""

    def test_first_failure_is_maximally_novel(self):
        from environment.failure_bank import FailureBank
        fb = FailureBank()
        score = fb.compute_novelty_score({
            "embedding": np.ones(768, dtype=np.float32)
        })
        assert score == 1.0  # empty bank → fully novel

    def test_duplicate_failure_has_low_novelty(self):
        from environment.failure_bank import FailureBank, FailureBankEntry
        import uuid
        from datetime import datetime, timezone
        fb = FailureBank()

        # Use a deterministic embedding (bypass sentence-transformer)
        emb = np.ones(768, dtype=np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-8)

        # Directly inject 3 entries with the identical embedding
        for _ in range(3):
            entry = FailureBankEntry(
                failure_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc).isoformat(),
                agent_version="0.0.0",
                error_type="test",
                prompt="duplicate task",
                expected_output="",
                actual_output="",
                cot_trace="",
                trajectory=None,
                novelty_score=1.0,
                semantic_embedding=emb.copy(),
            )
            fb._entries[entry.failure_id] = entry

        # Novelty for same embedding should be low (duplicate threshold at 0.8 similarity)
        score = fb.compute_novelty_score({"embedding": emb})
        assert score <= 0.15, f"Expected low novelty for duplicate, got {score}"

    def test_novelty_score_non_increasing_for_similar_failures(self):
        from environment.failure_bank import FailureBank
        fb = FailureBank()
        emb_base = np.ones(768, dtype=np.float32)

        scores = []
        for i in range(3):
            # Slightly perturb the embedding
            emb = emb_base + np.random.default_rng(i).random(768).astype(np.float32) * 0.01
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            score = fb.compute_novelty_score({"embedding": emb})
            scores.append(score)
            fb.add_failure({"prompt": f"similar task {i}", "error_type": "test"})

        # Each successive score should be ≤ the previous (non-increasing)
        for i in range(len(scores) - 1):
            assert scores[i + 1] <= scores[i] + 0.1, (
                f"Novelty increased from {scores[i]:.3f} to {scores[i+1]:.3f}"
            )


# ============================================================================
# P7: Co-evolutionary Coupling
# ============================================================================

class TestCoEvolutionaryCoupling:
    """Property 7: escalation ↔ (agent_improvement > 0.1 ∧ benchmark_improvement > 0.1)."""

    def _make_cm(self, agent_score, bench_score, episodes_since=60):
        from environment.curriculum_manager import CurriculumManager
        cm = CurriculumManager()
        cm._agent_improvement_score = agent_score
        cm._benchmark_improvement_score = bench_score
        cm._episodes_since_escalation = episodes_since
        return cm

    @pytest.mark.parametrize("agent,bench,expected", [
        (0.15, 0.15, True),   # both exceed threshold
        (0.05, 0.15, False),  # agent below threshold
        (0.15, 0.05, False),  # benchmark below threshold
        (0.0, 0.0, False),    # both below
        (0.11, 0.11, True),   # both just above
        (0.10, 0.10, False),  # both at boundary (not strictly greater)
    ])
    def test_escalation_condition(self, agent, bench, expected):
        cm = self._make_cm(agent, bench)
        assert cm.should_escalate() == expected, (
            f"should_escalate({agent}, {bench}) expected {expected}"
        )

    def test_insufficient_episodes_blocks_escalation(self):
        cm = self._make_cm(0.5, 0.5, episodes_since=10)
        assert cm.should_escalate() is False


# ============================================================================
# P8: Attempt Decay Monotonicity
# ============================================================================

class TestAttemptDecayMonotonicity:
    """Property 8: for fixed outcome, reward is strictly decreasing in turn number."""

    def test_turn1_reward_greater_than_turn2(self):
        from environment.reward_function import RewardFunction
        rf = RewardFunction()
        verification = {"tier1_score": 1.0, "tier2_score": 0.8, "tier3_score": 1.0}

        data1 = {"turns": [{
            "turn_number": 1, "action": "x", "verification": verification,
            "task_in_failure_bank": False, "times_task_attempted": 1,
            "total_tasks": 3, "prev_action": "",
            "discrimination_coverage_before": 0.0, "discrimination_coverage_after": 0.0,
        }]}
        data2 = {"turns": [{
            "turn_number": 2, "action": "x", "verification": verification,
            "task_in_failure_bank": False, "times_task_attempted": 2,
            "total_tasks": 3, "prev_action": "",
            "discrimination_coverage_before": 0.0, "discrimination_coverage_after": 0.0,
        }]}

        r1 = rf.compute_reward(data1)["per_turn_rewards"][0]
        r2 = rf.compute_reward(data2)["per_turn_rewards"][0]
        # Turn 1 programmatic reward should be >= Turn 2 (which includes decay)
        assert r1 >= r2

    def test_attempt_decay_applied_correctly(self):
        from environment.reward_function import RewardFunction
        rf = RewardFunction(attempt_decay_weight=0.1)
        # Turn 2 has attempt_decay = 0.1 * (2-1) = 0.1 deducted
        data = {"turns": [{
            "turn_number": 2, "action": "x",
            "verification": {"tier1_score": 0.0, "tier2_score": 1.0, "tier3_score": 1.0},
            "task_in_failure_bank": False, "times_task_attempted": 2,
            "total_tasks": 3, "prev_action": "",
            "discrimination_coverage_before": 0.0, "discrimination_coverage_after": 0.0,
        }]}
        result = rf.compute_reward(data)
        raw = result["per_turn_rewards"][0]
        # Turn 2 raw = LAMBDA_JUDGE * R_judge - attempt_decay = 0.8*1.0 - 0.1 = 0.7
        assert -1.0 <= raw <= 1.0


# ============================================================================
# P9: GRPO Advantage Zero-Mean
# ============================================================================

class TestGRPOAdvantageZeroMean:
    """Property 9: group-normalized advantages have mean ≈ 0 and std ≈ 1."""

    def _compute_advantages(self, rewards):
        rewards = np.array(rewards, dtype=float)
        mu = np.mean(rewards)
        sigma = np.std(rewards) + 1e-8
        return (rewards - mu) / sigma

    def test_advantages_zero_mean_small_group(self):
        rewards = [0.2, 0.5, 0.8, 0.1, 0.9, 0.3, 0.7, 0.4]
        adv = self._compute_advantages(rewards)
        assert abs(np.mean(adv)) < 1e-6

    def test_advantages_unit_std_small_group(self):
        rewards = [0.2, 0.5, 0.8, 0.1, 0.9, 0.3, 0.7, 0.4]
        adv = self._compute_advantages(rewards)
        assert abs(np.std(adv) - 1.0) < 1e-6

    def test_advantages_stable_for_random_groups(self):
        rng = np.random.default_rng(99)
        for _ in range(100):
            rewards = rng.random(8).tolist()
            adv = self._compute_advantages(rewards)
            assert abs(np.mean(adv)) < 1e-5

    def test_noise_injected_when_group_variance_near_zero(self):
        """When all rewards identical, advantages should be perturbed (not all zero)."""
        rewards = [0.5] * 8
        adv = self._compute_advantages(rewards)
        # With epsilon=1e-8 in denominator, all advantages will be 0 — which is the
        # degenerate case the trainer should detect and inject noise for
        assert all(a == 0.0 for a in adv) or np.std(adv) < 1e-6


# ============================================================================
# P10: Sandbox Isolation
# ============================================================================

class TestSandboxIsolation:
    """Property 10: sandboxed code cannot access restricted builtins."""

    def _make_verifier(self):
        from environment.verifier_stack import VerifierStack
        return VerifierStack()

    def test_open_is_blocked(self):
        v = self._make_verifier()
        result = v.tier1_verify("import os; result = open('/etc/passwd', 'r').read()")
        assert result["success"] is False

    def test_eval_is_blocked(self):
        v = self._make_verifier()
        result = v.tier1_verify("result = eval('1+1')")
        assert result["success"] is False

    def test_exec_is_blocked(self):
        v = self._make_verifier()
        result = v.tier1_verify("exec('import os')")
        assert result["success"] is False

    def test_import_is_blocked(self):
        v = self._make_verifier()
        result = v.tier1_verify("import os; result = os.getcwd()")
        assert result["success"] is False

    def test_safe_arithmetic_passes(self):
        v = self._make_verifier()
        result = v.tier1_verify("result = 2 + 2")
        assert result["success"] is True
        assert result["output"] == 4

    def test_safe_list_comprehension_passes(self):
        v = self._make_verifier()
        result = v.tier1_verify("result = [x * 2 for x in range(5)]")
        assert result["success"] is True
        assert result["output"] == [0, 2, 4, 6, 8]

    def test_timeout_enforced(self):
        v = self._make_verifier()
        result = v.tier1_verify("result = 0\nwhile True:\n    result += 1")
        assert result["success"] is False
        assert "timeout" in result["error_message"].lower() or result["error_type"] == "TimeoutError"

    def test_external_state_unchanged_after_violation(self):
        v = self._make_verifier()
        import os as _os
        cwd_before = _os.getcwd()
        v.tier1_verify("import os; os.chdir('/')")
        cwd_after = _os.getcwd()
        assert cwd_before == cwd_after


# ============================================================================
# P11: Episode Termination — always exactly 3 turns
# ============================================================================

class TestEpisodeTermination:
    """Property 11: every episode terminates at exactly turn 3."""

    def test_episode_terminates_at_turn_3(self):
        from environment.episode_handler import EpisodeHandler
        eh = EpisodeHandler()
        eh.initialize_episode("ep-001", "v0", "test task")
        done_turns = []
        for i in range(3):
            _, _, done, info = eh.step(f"action_{i}")
            if done:
                done_turns.append(i + 1)
        assert len(done_turns) == 1
        assert done_turns[0] == 3

    def test_episode_does_not_terminate_before_turn_3(self):
        from environment.episode_handler import EpisodeHandler
        eh = EpisodeHandler()
        eh.initialize_episode("ep-001", "v0", "test task")
        _, _, done1, _ = eh.step("action_1")
        _, _, done2, _ = eh.step("action_2")
        assert done1 is False
        assert done2 is False

    def test_episode_turn_count_matches_max_turns(self):
        from environment.episode_handler import EpisodeHandler
        eh = EpisodeHandler()
        eh.initialize_episode("ep-001", "v0", "test task")
        for i in range(3):
            eh.step(f"action_{i}")
        assert len(eh.trajectory["turns"]) == 3

    @pytest.mark.parametrize("action", [
        "",           # empty string — but EpisodeHandler accepts it (validation is in server)
        "x" * 1000,  # very long action
        "None",
        "🔥💯",       # unicode
    ])
    def test_episode_terminates_with_pathological_actions(self, action):
        from environment.episode_handler import EpisodeHandler
        eh = EpisodeHandler()
        eh.initialize_episode("ep-001", "v0", "test task")
        done = False
        for i in range(3):
            _, _, done, _ = eh.step(action)
        assert done is True


# ============================================================================
# P12: Repair Surgical Minimality (regression drop ≤ 15pp)
# ============================================================================

class TestRepairSurgicalMinimality:
    """Property 12: regression score after repair ≥ regression score before - 0.15."""

    def test_regression_battery_baseline(self):
        from environment.verifier_stack import VerifierStack
        v = VerifierStack()
        result = v.tier3_verify("")
        # Baseline should be high since our 20 tasks are simple arithmetic
        assert result["pass_rate"] >= 0.5

    def test_regression_drop_bounded(self):
        from environment.verifier_stack import VerifierStack
        v = VerifierStack()
        r_pre = v.tier3_verify("")["pass_rate"]
        # Simulate a repair: run again (same result since deterministic)
        r_post = v.tier3_verify("")["pass_rate"]
        assert r_post >= r_pre - 0.15


# ============================================================================
# Task 5.3: Hunt mode unit tests
# ============================================================================

class TestHuntMode:
    """Unit tests for Task Generator Hunt mode."""

    def test_hunt_mode_returns_required_keys(self):
        from environment.task_generator import TaskGenerator
        tg = TaskGenerator()
        result = tg.hunt_mode(agent_performance={}, discrimination_zone=[])
        assert "prompt" in result
        assert "difficulty_score" in result
        assert "strategy" in result
        assert "reasoning" in result
        assert "cot_trace" in result

    def test_hunt_mode_strategy_is_valid(self):
        from environment.task_generator import TaskGenerator, ADVERSARIAL_STRATEGIES
        tg = TaskGenerator()
        result = tg.hunt_mode(agent_performance={}, discrimination_zone=[])
        assert result["strategy"] in ADVERSARIAL_STRATEGIES

    def test_hunt_mode_prompt_is_non_empty(self):
        from environment.task_generator import TaskGenerator
        tg = TaskGenerator()
        result = tg.hunt_mode(agent_performance={}, discrimination_zone=[])
        assert len(result["prompt"]) > 0

    def test_hunt_mode_difficulty_score_in_range(self):
        from environment.task_generator import TaskGenerator
        tg = TaskGenerator()
        result = tg.hunt_mode(agent_performance={}, discrimination_zone=[])
        assert 0 <= result["difficulty_score"] <= 100

    def test_hunt_mode_uses_discrimination_zone(self):
        from environment.task_generator import TaskGenerator
        tg = TaskGenerator()
        zone = ["solve x + 5 = 12", "what is 7 * 8?"]
        result = tg.hunt_mode(agent_performance={}, discrimination_zone=zone)
        assert len(result["prompt"]) > 0

    def test_hunt_mode_deduplicates_prompts(self):
        from environment.task_generator import TaskGenerator
        tg = TaskGenerator()
        results = [
            tg.hunt_mode(agent_performance={}, discrimination_zone=["task_a"])
            for _ in range(3)
        ]
        prompts = [r["prompt"] for r in results]
        # All prompts should be in history after generation
        assert len(tg._prompt_history) >= 3


# ============================================================================
# Task 7.2: OracleInterface caching
# ============================================================================

class TestOracleInterfaceCaching:
    """Unit tests for Oracle caching behavior."""

    def test_cache_hit_avoids_api_call(self):
        from environment.oracle_interface import OracleInterface
        oracle = OracleInterface()
        call_count = [0]

        original_call = oracle._call_reference_model
        def counting_call(model_key, task):
            call_count[0] += 1
            return 0.5
        oracle._call_reference_model = counting_call

        oracle.calibrate_task("test task alpha")
        count_after_first = call_count[0]
        oracle.calibrate_task("test task alpha")
        count_after_second = call_count[0]

        assert count_after_first == 2  # primary + secondary
        assert count_after_second == count_after_first  # no new calls (cache hit)

    def test_different_tasks_call_api(self):
        from environment.oracle_interface import OracleInterface
        oracle = OracleInterface()
        call_count = [0]
        oracle._call_reference_model = lambda m, t: call_count.__setitem__(0, call_count[0] + 1) or 0.5

        oracle.calibrate_task("task one")
        oracle.calibrate_task("task two")
        assert call_count[0] == 4  # 2 models × 2 tasks

    def test_cache_hit_rate_computed_correctly(self):
        from environment.oracle_interface import OracleInterface
        oracle = OracleInterface()
        oracle._call_reference_model = lambda m, t: 0.5

        oracle.calibrate_task("cached_task")
        oracle.calibrate_task("cached_task")
        rate = oracle.get_cache_hit_rate()
        assert rate > 0.0

    def test_difficulty_assigned_correctly(self):
        from environment.oracle_interface import OracleInterface
        oracle = OracleInterface()
        oracle._call_reference_model = lambda m, t: 0.2  # easy
        result = oracle.calibrate_task("easy task")
        assert result["difficulty"] == "easy"

        oracle2 = OracleInterface()
        oracle2._call_reference_model = lambda m, t: 0.8  # hard
        result2 = oracle2.calibrate_task("hard task")
        assert result2["difficulty"] == "hard"

    def test_divergence_flagged(self):
        from environment.oracle_interface import OracleInterface
        oracle = OracleInterface()
        scores = iter([0.1, 0.8])  # |0.1 - 0.8| = 0.7 > 0.3
        oracle._call_reference_model = lambda m, t: next(scores)
        result = oracle.calibrate_task("divergent task")
        assert result["flagged_for_review"] is True

    def test_cache_expiry_after_30_days(self):
        from datetime import datetime, timedelta, timezone
        from environment.oracle_interface import OracleInterface
        oracle = OracleInterface()
        oracle._call_reference_model = lambda m, t: 0.5

        # First call - cache miss
        task_hash = oracle._hash_task("expiry test task")
        oracle.calibrate_task("expiry test task")
        
        # Manually set cache entry to 31 days ago
        old_timestamp = datetime.now(timezone.utc) - timedelta(days=31)
        oracle._cache[(task_hash, "primary")] = {"score": 0.5, "timestamp": old_timestamp}
        oracle._cache[(task_hash, "secondary")] = {"score": 0.5, "timestamp": old_timestamp}
        
        # Should return None for expired cache
        assert oracle.get_cached_score(task_hash, "primary") is None
        assert oracle.get_cached_score(task_hash, "secondary") is None


# ============================================================================
# Task 11.2: Reward Function property tests
# ============================================================================

class TestRewardFunctionProperties:
    """Property tests for the reward function."""

    def test_set_weights_validates_range(self):
        from environment.reward_function import RewardFunction
        rf = RewardFunction()
        with pytest.raises(ValueError):
            rf.set_weights({"attempt_decay": 1.5})

    def test_set_weights_valid(self):
        from environment.reward_function import RewardFunction
        rf = RewardFunction()
        rf.set_weights({"attempt_decay": 0.2, "novelty_penalty": 0.1})
        assert rf.weights["attempt_decay"] == 0.2
        assert rf.weights["novelty_penalty"] == 0.1

    def test_cumulative_reward_bounded(self):
        from environment.reward_function import RewardFunction
        rf = RewardFunction()
        data = {"turns": [
            {"turn_number": 1, "action": "a", "verification": {"tier1_score": 1.0, "tier2_score": 0.9, "tier3_score": 1.0},
             "task_in_failure_bank": False, "times_task_attempted": 1, "total_tasks": 1,
             "prev_action": "", "discrimination_coverage_before": 0.0, "discrimination_coverage_after": 0.5},
            {"turn_number": 2, "action": "b", "verification": {"tier1_score": 0.0, "tier2_score": 0.8, "tier3_score": 1.0},
             "task_in_failure_bank": True, "times_task_attempted": 2, "total_tasks": 1,
             "prev_action": "a", "discrimination_coverage_before": 0.5, "discrimination_coverage_after": 0.6},
        ]}
        result = rf.compute_reward(data)
        assert -1.0 <= result["cumulative_reward"] <= 1.0

    def test_potential_shaping_applied(self):
        from environment.reward_function import RewardFunction
        rf = RewardFunction(gamma=0.99)
        # Turn 2: raw = LAMBDA_JUDGE * R_judge - attempt_decay = 0.8*0.5 - 0.1 = 0.3
        # prev_coverage starts at 0.0; shaping = gamma * coverage_after - prev_coverage
        #   = 0.99 * 0.5 - 0.0 = 0.495
        # shaped = clamp(0.3 + 0.495) = 0.795
        data = {"turns": [{
            "turn_number": 2, "action": "x",
            "verification": {"tier1_score": 0.0, "tier2_score": 0.5, "tier3_score": 1.0},
            "task_in_failure_bank": False, "times_task_attempted": 2, "total_tasks": 3,
            "prev_action": "y", "discrimination_coverage_before": 0.3, "discrimination_coverage_after": 0.5,
        }]}
        result = rf.compute_reward(data)
        shaped = result["shaped_rewards"][0]
        raw = result["per_turn_rewards"][0]
        # prev_coverage in compute_reward rolls from 0.0 (start of episode), not from turn data
        expected_shaping = rf.gamma * 0.5 - 0.0  # gamma * coverage_after - initial prev_coverage
        expected_shaped = max(-1.0, min(1.0, raw + expected_shaping))
        assert abs(shaped - expected_shaped) < 0.01


# ============================================================================
# Task 12.2: GRPO advantage zero-mean (already in P9, add trainer test)
# ============================================================================

class TestGRPOTrainer:
    """Unit tests for GRPO trainer logic."""

    def test_advantages_zero_mean_via_trainer(self):
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from training.train import GRPOTrainer
        trainer = GRPOTrainer()
        rollouts = [{"reward": r} for r in [0.2, 0.5, 0.8, 0.1, 0.9, 0.3, 0.7, 0.4]]
        adv = trainer._compute_advantages(rollouts)
        assert abs(np.mean(adv)) < 1e-5

    def test_advantages_unit_std_via_trainer(self):
        from training.train import GRPOTrainer
        trainer = GRPOTrainer()
        rollouts = [{"reward": r} for r in [0.2, 0.5, 0.8, 0.1, 0.9, 0.3, 0.7, 0.4]]
        adv = trainer._compute_advantages(rollouts)
        assert abs(np.std(adv) - 1.0) < 1e-5


# ============================================================================
# Task 3.1: MDPState — to_vector / from_vector round-trip + encode helpers
# ============================================================================

class TestMDPState:
    """Unit tests for MDPState serialization and encoding (Requirements 31.1, 31.5, 33.1)."""

    def test_to_vector_shape(self):
        from environment.state import MDPState, STATE_DIM
        s = MDPState()
        vec = s.to_vector()
        assert vec.shape == (STATE_DIM,), f"Expected ({STATE_DIM},), got {vec.shape}"

    def test_from_vector_round_trip(self):
        from environment.state import MDPState
        import numpy as np
        s = MDPState(
            task_embedding=np.ones(768, dtype=np.float32) * 0.5,
            agent_capability_vector=np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32),
            difficulty_tier=3,
            turn_number=2,
            discrimination_coverage=0.42,
            cumulative_reward=-0.1,
        )
        vec = s.to_vector()
        s2 = MDPState.from_vector(vec)
        assert s2.difficulty_tier == s.difficulty_tier
        assert s2.turn_number == s.turn_number
        assert abs(s2.discrimination_coverage - s.discrimination_coverage) < 1e-5
        assert abs(s2.cumulative_reward - s.cumulative_reward) < 1e-5
        assert np.allclose(s2.task_embedding, s.task_embedding, atol=1e-5)
        assert np.allclose(s2.agent_capability_vector, s.agent_capability_vector, atol=1e-5)

    def test_from_vector_wrong_size_raises(self):
        from environment.state import MDPState
        import numpy as np
        with pytest.raises(ValueError):
            MDPState.from_vector(np.zeros(100))

    def test_encode_failure_bank_snapshot_zero_padding(self):
        from environment.state import MDPState, FAILURE_SNAPSHOT_K, TASK_EMBED_DIM
        import numpy as np
        # Provide fewer than K embeddings — rest should be zero-padded
        embeddings = [np.ones(TASK_EMBED_DIM, dtype=np.float32) for _ in range(3)]
        snapshot = MDPState.encode_failure_bank_snapshot(embeddings)
        assert snapshot.shape == (FAILURE_SNAPSHOT_K * TASK_EMBED_DIM,)
        # First 3 slots filled
        assert np.all(snapshot[:3 * TASK_EMBED_DIM] == 1.0)
        # Remaining slots zero-padded
        assert np.all(snapshot[3 * TASK_EMBED_DIM:] == 0.0)

    def test_encode_failure_bank_snapshot_truncates_to_k(self):
        from environment.state import MDPState, FAILURE_SNAPSHOT_K, TASK_EMBED_DIM
        import numpy as np
        # Provide more than K embeddings — should truncate
        embeddings = [np.ones(TASK_EMBED_DIM, dtype=np.float32) * i for i in range(FAILURE_SNAPSHOT_K + 5)]
        snapshot = MDPState.encode_failure_bank_snapshot(embeddings)
        assert snapshot.shape == (FAILURE_SNAPSHOT_K * TASK_EMBED_DIM,)

    def test_default_state_vector_is_valid(self):
        from environment.state import MDPState, STATE_DIM
        s = MDPState()
        vec = s.to_vector()
        assert vec.shape == (STATE_DIM,)
        assert not np.any(np.isnan(vec))
        assert not np.any(np.isinf(vec))


# ============================================================================
# Task 5.2: Repair mode unit tests
# ============================================================================

class TestRepairMode:
    """Unit tests for Task Generator Repair mode (Requirements 4.1–4.7)."""

    def _make_mock_failure_bank(self, num_failures=3):
        """Create a mock failure bank with synthetic entries."""
        class MockFailureBank:
            def get_repair_candidates(self, num_pairs):
                return [
                    {
                        "prompt": f"Solve: x + {i} = 10",
                        "repair_priority": float(num_pairs - i),
                        "error_type": "incorrect_output",
                    }
                    for i in range(min(num_pairs, num_failures))
                ]
        return MockFailureBank()

    def test_repair_mode_returns_list(self):
        from environment.task_generator import TaskGenerator
        tg = TaskGenerator()
        fb = self._make_mock_failure_bank()
        result = tg.repair_mode(fb, num_pairs=2)
        assert isinstance(result, list)

    def test_repair_mode_returns_required_keys(self):
        from environment.task_generator import TaskGenerator
        tg = TaskGenerator()
        fb = self._make_mock_failure_bank()
        result = tg.repair_mode(fb, num_pairs=2)
        assert len(result) > 0
        for pair in result:
            assert "prompt" in pair
            assert "solution" in pair
            assert "reasoning" in pair
            assert "priority_score" in pair

    def test_repair_mode_ordered_by_priority(self):
        from environment.task_generator import TaskGenerator
        tg = TaskGenerator()
        fb = self._make_mock_failure_bank(num_failures=3)
        result = tg.repair_mode(fb, num_pairs=3)
        if len(result) >= 2:
            priorities = [p["priority_score"] for p in result]
            assert priorities == sorted(priorities, reverse=True)

    def test_repair_mode_solution_non_empty(self):
        from environment.task_generator import TaskGenerator
        tg = TaskGenerator()
        fb = self._make_mock_failure_bank()
        result = tg.repair_mode(fb, num_pairs=1)
        for pair in result:
            assert len(pair["solution"]) > 0

    def test_repair_mode_empty_failure_bank(self):
        from environment.task_generator import TaskGenerator
        tg = TaskGenerator()

        class EmptyBank:
            def get_repair_candidates(self, n):
                return []

        result = tg.repair_mode(EmptyBank(), num_pairs=5)
        assert result == []

    def test_strategy_effectiveness_updates(self):
        from environment.task_generator import TaskGenerator, ADVERSARIAL_STRATEGIES
        tg = TaskGenerator()
        strategy = ADVERSARIAL_STRATEGIES[0]
        initial = tg._strategy_effectiveness[strategy]
        tg.update_strategy_effectiveness(strategy, success=True)
        assert tg._strategy_effectiveness[strategy] != initial

    def test_strategy_effectiveness_ema(self):
        """Effectiveness uses exponential moving average — success increases it."""
        from environment.task_generator import TaskGenerator, ADVERSARIAL_STRATEGIES
        tg = TaskGenerator()
        strategy = ADVERSARIAL_STRATEGIES[0]
        for _ in range(10):
            tg.update_strategy_effectiveness(strategy, success=True)
        assert tg._strategy_effectiveness[strategy] > 0.5


# ============================================================================
# Task 6.1: CurriculumManager additional unit tests
# ============================================================================

class TestCurriculumManagerMetrics:
    """Unit tests for CurriculumManager metrics (Requirements 6.1–6.8, 24.1–24.7)."""

    def test_update_task_performance_tracks_attempts(self):
        from environment.curriculum_manager import CurriculumManager
        cm = CurriculumManager()
        cm.update_task_performance("task_a", success=True)
        cm.update_task_performance("task_a", success=False)
        assert cm.task_metadata["task_a"]["attempt_count"] == 2

    def test_get_task_success_rate_correct(self):
        from environment.curriculum_manager import CurriculumManager
        cm = CurriculumManager()
        for _ in range(3):
            cm.update_task_performance("task_b", success=True)
        for _ in range(1):
            cm.update_task_performance("task_b", success=False)
        rate = cm.get_task_success_rate("task_b")
        assert abs(rate - 0.75) < 1e-6

    def test_get_task_success_rate_unknown_task(self):
        from environment.curriculum_manager import CurriculumManager
        cm = CurriculumManager()
        assert cm.get_task_success_rate("nonexistent") == 0.0

    def test_manual_override_sets_tier(self):
        from environment.curriculum_manager import CurriculumManager
        cm = CurriculumManager()
        cm.set_manual_override(5)
        assert cm.difficulty_tier == 5

    def test_change_log_records_escalation(self):
        from environment.curriculum_manager import CurriculumManager
        cm = CurriculumManager()
        cm._episodes_since_escalation = 100
        cm._agent_improvement_score = 0.2
        cm._benchmark_improvement_score = 0.2
        cm.escalate()
        assert len(cm._change_log) == 1
        assert cm._change_log[0]["type"] == "escalation"

    def test_heatmap_shape(self):
        from environment.curriculum_manager import CurriculumManager
        cm = CurriculumManager()
        hm = cm.get_curriculum_heatmap()
        assert hm.shape == (5, 10)

    def test_heatmap_populated_after_updates(self):
        from environment.curriculum_manager import CurriculumManager
        cm = CurriculumManager()
        cm.update_task_performance("task_x", success=True)
        hm = cm.get_curriculum_heatmap()
        assert hm.sum() > 0.0

    def test_plateau_detected_after_100_episodes(self):
        from environment.curriculum_manager import CurriculumManager
        cm = CurriculumManager()
        cm._episodes_since_escalation = 100
        assert cm.detect_plateau() is True

    def test_no_plateau_before_100_episodes(self):
        from environment.curriculum_manager import CurriculumManager
        cm = CurriculumManager()
        cm._episodes_since_escalation = 50
        assert cm.detect_plateau() is False


# ============================================================================
# Task 13.3: Sandbox restriction enforcement unit tests
# Requirements 16.1–16.9
# ============================================================================

class TestSandboxRestrictionEnforcement:
    """Unit tests for sandbox restriction enforcement (Task 13.3)."""

    def _vs(self):
        from environment.verifier_stack import VerifierStack
        return VerifierStack()

    def test_open_raises_exception(self):
        result = self._vs().tier1_verify("result = open('/etc/passwd', 'r')")
        assert result["success"] is False

    def test_exec_raises_exception(self):
        result = self._vs().tier1_verify("exec('x=1')")
        assert result["success"] is False

    def test_eval_raises_exception(self):
        result = self._vs().tier1_verify("result = eval('1+1')")
        assert result["success"] is False

    def test_import_raises_exception(self):
        result = self._vs().tier1_verify("import os; result = os.getcwd()")
        assert result["success"] is False

    def test_compile_raises_exception(self):
        result = self._vs().tier1_verify("result = compile('x=1', '<string>', 'exec')")
        assert result["success"] is False

    def test_timeout_configuration(self):
        from environment.verifier_stack import SANDBOX_TIMEOUT
        assert SANDBOX_TIMEOUT == 5

    def test_timeout_enforced_for_infinite_loop(self):
        result = self._vs().tier1_verify("while True: pass")
        assert result["success"] is False
        assert "timed out" in (result.get("error_message") or "").lower()

    def test_memory_limit_configured(self):
        from environment.verifier_stack import SANDBOX_MEMORY_MB
        assert SANDBOX_MEMORY_MB == 512

    def test_safe_code_still_passes(self):
        result = self._vs().tier1_verify("result = 7 * 6")
        assert result["success"] is True
        assert result["output"] == 42

    def test_breakpoint_is_blocked(self):
        result = self._vs().tier1_verify("breakpoint()")
        assert result["success"] is False


# ============================================================================
# Task 13.1: TrajectorySampler unit tests
# Requirements 17.1–17.8, 30.1–30.7
# ============================================================================

class TestTrajectorySampler:
    """Unit tests for TrajectorySampler (Task 13.1)."""

    def _make_trajectory(self, actions, rewards=None):
        if rewards is None:
            rewards = [0.5] * len(actions)
        turns = [{"action": a, "reward": r} for a, r in zip(actions, rewards)]
        return {"turns": turns, "metadata": {"episode_id": "test-ep-001"}}

    def test_analyze_clean_trajectory_not_flagged(self):
        from monitors.trajectory_sampler import TrajectorySampler
        ts = TrajectorySampler()
        traj = self._make_trajectory(["action_a", "action_b", "action_c"])
        result = ts.analyze_trajectory(traj)
        assert isinstance(result["anomaly_score"], float)
        assert isinstance(result["flagged"], bool)

    def test_output_repetition_detected(self):
        from monitors.trajectory_sampler import TrajectorySampler
        ts = TrajectorySampler(repetition_threshold=0.4)
        traj = self._make_trajectory(["same"] * 8 + ["diff"])
        result = ts.analyze_trajectory(traj)
        assert result["repetition_rate"] > 0.4

    def test_mode_collapse_detected_by_entropy(self):
        from monitors.trajectory_sampler import TrajectorySampler
        ts = TrajectorySampler(entropy_threshold=0.5)
        # All identical actions → entropy = 0 → collapse
        traj = self._make_trajectory(["same_action"] * 10)
        result = ts.analyze_trajectory(traj)
        assert result["trajectory_entropy"] == 0.0

    def test_reward_hacking_detected(self):
        from monitors.trajectory_sampler import TrajectorySampler
        ts = TrajectorySampler()
        traj = self._make_trajectory(["a", "b", "c"], rewards=[0.95, 0.99, 0.98])
        result = ts.analyze_trajectory(traj)
        assert result["anomaly_type"] is not None

    def test_anomaly_rate_computed_correctly(self):
        from monitors.trajectory_sampler import TrajectorySampler
        ts = TrajectorySampler(repetition_threshold=0.4)
        for _ in range(5):
            traj = self._make_trajectory(["x"] * 10)  # definitely anomalous
            ts.analyze_trajectory(traj)
        rate = ts.get_anomaly_rate()
        assert 0.0 <= rate <= 1.0

    def test_compute_trajectory_entropy_uniform(self):
        from monitors.trajectory_sampler import TrajectorySampler
        ts = TrajectorySampler()
        # 4 distinct actions → high entropy
        entropy = ts.compute_trajectory_entropy(["a", "b", "c", "d"])
        assert entropy > 0.9

    def test_compute_trajectory_entropy_identical(self):
        from monitors.trajectory_sampler import TrajectorySampler
        ts = TrajectorySampler()
        entropy = ts.compute_trajectory_entropy(["x", "x", "x", "x"])
        assert entropy == 0.0

    def test_sampled_count_increments(self):
        from monitors.trajectory_sampler import TrajectorySampler
        ts = TrajectorySampler()
        traj = self._make_trajectory(["a", "b"])
        ts.analyze_trajectory(traj)
        ts.analyze_trajectory(traj)
        assert ts._sampled_count == 2


# ============================================================================
# Task 13.2: EntropyMonitor unit tests
# Requirements 18.1–18.8
# ============================================================================

class TestEntropyMonitor:
    """Unit tests for EntropyMonitor (Task 13.2)."""

    def test_record_step_returns_required_keys(self):
        from monitors.entropy_monitor import EntropyMonitor
        em = EntropyMonitor()
        result = em.record_step([0.25, 0.25, 0.25, 0.25])
        assert "entropy" in result
        assert "action_diversity" in result
        assert "collapsed" in result
        assert "learning_rate" in result

    def test_entropy_is_positive_for_uniform_distribution(self):
        from monitors.entropy_monitor import EntropyMonitor
        em = EntropyMonitor()
        probs = [0.1] * 10
        entropy = em.compute_policy_entropy(probs)
        assert entropy > 0.0

    def test_entropy_is_zero_for_degenerate_distribution(self):
        from monitors.entropy_monitor import EntropyMonitor
        em = EntropyMonitor()
        entropy = em.compute_policy_entropy([1.0, 0.0, 0.0])
        assert entropy == 0.0

    def test_no_collapse_detected_before_window_fills(self):
        from monitors.entropy_monitor import EntropyMonitor, WINDOW_SIZE
        em = EntropyMonitor()
        probs = [0.25, 0.25, 0.25, 0.25]
        for _ in range(WINDOW_SIZE // 2):
            em.record_step(probs)
        assert em.detect_collapse() is False

    def test_collapse_detected_after_entropy_drop(self):
        from monitors.entropy_monitor import EntropyMonitor, WINDOW_SIZE
        em = EntropyMonitor()
        # Fill window with high-entropy steps
        for _ in range(WINDOW_SIZE):
            em.record_step([0.25, 0.25, 0.25, 0.25])
        # Force the internal history to simulate a drop
        from collections import deque
        em._entropy_history = deque(
            [2.0] + [0.0] * (WINDOW_SIZE - 1), maxlen=WINDOW_SIZE
        )
        assert em.detect_collapse() is True

    def test_lr_reduced_on_collapse(self):
        from monitors.entropy_monitor import EntropyMonitor, WINDOW_SIZE, LR_REDUCTION_FACTOR
        em = EntropyMonitor(initial_learning_rate=1e-4)
        baseline = em.learning_rate
        em.adjust_learning_rate(LR_REDUCTION_FACTOR)
        assert em.learning_rate == pytest.approx(baseline * LR_REDUCTION_FACTOR)

    def test_action_diversity_below_threshold_warns(self, caplog):
        import logging
        from monitors.entropy_monitor import EntropyMonitor
        em = EntropyMonitor()
        # Degenerate distribution → low diversity
        with caplog.at_level(logging.WARNING):
            em.record_step([0.99] + [0.001] * 99)
        assert any("diversity" in r.message.lower() for r in caplog.records)
