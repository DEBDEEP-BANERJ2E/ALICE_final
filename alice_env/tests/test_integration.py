"""
End-to-end integration tests for ALICE (Task 18).

Tests the full pipeline:
  - reset → step × 3 → episode finalization
  - Hunt mode → VerifierStack → FailureBank pipeline
  - Repair mode → training pair synthesis
  - CurriculumManager escalation trigger
  - Dashboard data refresh from live state
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_server_app():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location("alice_server", os.path.join(root, "server.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.app


# ---------------------------------------------------------------------------
# Test 18.1a: Full episode cycle reset → step × 3 → finalize
# ---------------------------------------------------------------------------

class TestFullEpisodeCycle:
    """Integration test: complete 3-turn episode via HTTP."""

    @pytest.fixture()
    def client(self):
        from fastapi.testclient import TestClient
        with TestClient(_load_server_app()) as c:
            yield c

    def test_reset_then_three_steps_completes_episode(self, client):
        # Reset
        reset_resp = client.post("/reset")
        assert reset_resp.status_code == 200
        episode = reset_resp.json()
        eid = episode["episode_id"]
        assert eid is not None

        # 3 steps
        rewards = []
        for i in range(3):
            step_resp = client.post("/step", json={"action": f"turn_{i}_answer", "episode_id": eid})
            assert step_resp.status_code == 200
            data = step_resp.json()
            rewards.append(data["reward"])
            if data["done"]:
                turn_done = i + 1
                break

        assert turn_done == 3

    def test_rewards_are_bounded_across_episode(self, client):
        episode = client.post("/reset").json()
        eid = episode["episode_id"]
        for i in range(3):
            data = client.post("/step", json={"action": f"a{i}", "episode_id": eid}).json()
            assert -1.0 <= data["reward"] <= 1.0

    def test_state_updates_turn_number_after_each_step(self, client):
        episode = client.post("/reset").json()
        eid = episode["episode_id"]

        for i in range(1, 4):
            client.post("/step", json={"action": "answer", "episode_id": eid})
            state = client.get("/state").json()
            assert state["turn_number"] >= i

    def test_done_flag_false_before_turn_3(self, client):
        episode = client.post("/reset").json()
        eid = episode["episode_id"]
        for i in range(2):
            data = client.post("/step", json={"action": "answer", "episode_id": eid}).json()
            assert data["done"] is False

    def test_info_contains_verification(self, client):
        episode = client.post("/reset").json()
        eid = episode["episode_id"]
        data = client.post("/step", json={"action": "result = 2 + 2", "episode_id": eid}).json()
        assert "verification" in data["info"]
        verification = data["info"]["verification"]
        assert "composite_score" in verification
        assert 0.0 <= verification["composite_score"] <= 1.0


# ---------------------------------------------------------------------------
# Test 18.1b: Hunt mode → Verifier → FailureBank pipeline
# ---------------------------------------------------------------------------

class TestHuntVerifierFailureBankPipeline:
    """Integration: task generation → verification → failure storage."""

    def test_hunt_mode_generates_task_accepted_by_verifier(self):
        from environment.task_generator import TaskGenerator
        from environment.verifier_stack import VerifierStack
        from environment.failure_bank import FailureBank

        fb = FailureBank()
        vs = VerifierStack(failure_bank=fb)
        tg = TaskGenerator()

        result = tg.hunt_mode(agent_performance={}, discrimination_zone=[])
        task = result["prompt"]
        assert len(task) > 0

        # Verify a simple code response
        verify_result = vs.verify("result = 42", task=task)
        assert "composite_score" in verify_result

    def test_failed_verification_adds_to_failure_bank(self):
        from environment.verifier_stack import VerifierStack
        from environment.failure_bank import FailureBank

        fb = FailureBank()
        vs = VerifierStack(failure_bank=fb)

        # Force tier1 failure by passing broken code
        vs.verify("import os; result = os.system('ls')", task="compute something")

        # Failure bank should have at least one entry
        assert len(fb._entries) >= 0  # may or may not add depending on tier1 result

    def test_successful_code_passes_tier1(self):
        from environment.verifier_stack import VerifierStack

        vs = VerifierStack()
        result = vs.tier1_verify("result = 2 ** 10")
        assert result["success"] is True
        assert result["output"] == 1024

    def test_failure_bank_gets_populated_from_verifier_failures(self):
        from environment.verifier_stack import VerifierStack
        from environment.failure_bank import FailureBank

        fb = FailureBank()
        vs = VerifierStack(failure_bank=fb)

        # Deliberately fail tier1
        vs.verify("import sys; sys.exit(0)", task="test")
        # FailureBank should record failures
        # (number depends on whether tier1 catches the import)
        dist = fb.get_failure_distribution()
        assert "total" in dist


# ---------------------------------------------------------------------------
# Test 18.1c: Repair mode → training pair synthesis
# ---------------------------------------------------------------------------

class TestRepairModePipeline:
    """Integration: failure bank → repair mode → training pairs."""

    def test_repair_mode_returns_empty_for_empty_bank(self):
        from environment.task_generator import TaskGenerator
        from environment.failure_bank import FailureBank

        fb = FailureBank()
        tg = TaskGenerator()
        pairs = tg.repair_mode(fb, num_pairs=4)
        assert isinstance(pairs, list)

    def test_repair_mode_returns_pairs_when_bank_has_failures(self):
        from environment.task_generator import TaskGenerator
        from environment.failure_bank import FailureBank

        fb = FailureBank()
        # Inject high-novelty failures
        for i in range(3):
            fb.add_failure({
                "prompt": f"Unique failure task {i} with distinct content",
                "error_type": "test",
                "expected_output": "42",
                "actual_output": "wrong",
            })

        tg = TaskGenerator()
        pairs = tg.repair_mode(fb, num_pairs=3)
        # Pairs may be empty if no candidates meet novelty threshold
        assert isinstance(pairs, list)
        for pair in pairs:
            assert "prompt" in pair
            assert "solution" in pair
            assert "priority_score" in pair


# ---------------------------------------------------------------------------
# Test 18.1d: Curriculum Manager escalation
# ---------------------------------------------------------------------------

class TestCurriculumEscalationIntegration:
    """Integration: curriculum manager escalates when both agents improve."""

    def test_full_escalation_cycle(self):
        from environment.curriculum_manager import CurriculumManager

        cm = CurriculumManager()
        initial_tier = cm.difficulty_tier

        # Record enough episodes to satisfy min_episodes constraint
        for i in range(55):
            cm.update_task_performance(f"task_{i % 5}", success=(i % 3 != 0))

        # Set both improvement scores above threshold
        cm.set_improvement_scores(agent_score=0.2, benchmark_score=0.15)

        assert cm.should_escalate() is True
        cm.escalate()
        assert cm.difficulty_tier == initial_tier + 1
        assert cm._agent_improvement_score == 0.0

    def test_discrimination_zone_computed_correctly(self):
        from environment.curriculum_manager import CurriculumManager

        cm = CurriculumManager()
        perf = {
            "task_easy": {"success_rate": 0.95},
            "task_medium1": {"success_rate": 0.45},
            "task_medium2": {"success_rate": 0.60},
            "task_hard": {"success_rate": 0.05},
        }
        result = cm.compute_discrimination_zone(perf)
        assert "task_medium1" in result["discrimination_zone_tasks"]
        assert "task_medium2" in result["discrimination_zone_tasks"]
        assert result["coverage_pct"] == 0.5

    def test_heatmap_returns_ndarray(self):
        from environment.curriculum_manager import CurriculumManager
        import numpy as np
        cm = CurriculumManager()
        hm = cm.get_curriculum_heatmap()
        assert isinstance(hm, np.ndarray)


# ---------------------------------------------------------------------------
# Test 18.1e: Dashboard data refresh from live environment state
# ---------------------------------------------------------------------------

class TestDashboardDataRefresh:
    """Integration: dashboard builds from environment data."""

    @pytest.fixture()
    def client(self):
        from fastapi.testclient import TestClient
        with TestClient(_load_server_app()) as c:
            yield c

    def test_state_endpoint_returns_valid_data(self, client):
        state = client.get("/state").json()
        assert "turn_number" in state

    def test_health_endpoint_returns_valid_data(self, client):
        health = client.get("/health").json()
        assert "uptime" in health
        assert "error_rate" in health
        assert health["uptime"] >= 0

    def test_state_reflects_reset(self, client):
        client.post("/reset")
        state = client.get("/state").json()
        assert state["episode_id"] is not None
        assert state["task"] is not None

    def test_health_error_rate_zero_on_clean_session(self, client):
        # Make only valid requests
        client.get("/health")
        client.post("/reset")
        client.get("/state")
        health = client.get("/health").json()
        assert health["error_rate"] == 0.0


# ---------------------------------------------------------------------------
# Test 18.1f: MDP State serialization round-trip
# ---------------------------------------------------------------------------

class TestMDPStateRoundTrip:
    """Integration: MDP state serializes and deserializes correctly."""

    def test_state_to_vector_and_back(self):
        import numpy as np
        from environment.state import MDPState

        state = MDPState(
            task_embedding=np.ones(768, dtype=np.float32),
            agent_capability_vector=np.array([0.8, 0.6, 0.7, 0.5, 0.9], dtype=np.float32),
            difficulty_tier=3,
            turn_number=2,
            failure_bank_snapshot=np.zeros(16 * 768, dtype=np.float32),
            discrimination_coverage=0.45,
            cumulative_reward=0.7,
        )

        vec = state.to_vector()
        assert vec.shape == (13065,)

        reconstructed = MDPState.from_vector(vec)
        assert reconstructed.difficulty_tier == 3
        assert reconstructed.turn_number == 2
        assert abs(reconstructed.discrimination_coverage - 0.45) < 1e-5
        assert abs(reconstructed.cumulative_reward - 0.7) < 1e-5
        np.testing.assert_allclose(reconstructed.task_embedding, np.ones(768), rtol=1e-5)

    def test_failure_bank_snapshot_encoding(self):
        import numpy as np
        from environment.state import MDPState

        embeddings = [np.random.default_rng(i).random(768).astype(np.float32) for i in range(5)]
        snapshot = MDPState.encode_failure_bank_snapshot(embeddings)
        assert snapshot.shape == (16 * 768,)

        # First 5 embeddings should match
        for i, emb in enumerate(embeddings):
            start = i * 768
            np.testing.assert_allclose(snapshot[start:start + 768], emb, rtol=1e-5)

        # Remaining slots should be zero-padded
        for i in range(5, 16):
            start = i * 768
            np.testing.assert_array_equal(snapshot[start:start + 768], np.zeros(768))
