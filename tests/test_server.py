"""Unit tests for ALICE OpenEnv server endpoints.

Tests: Task 2.2 — reset/step/state/health happy paths, invalid action handling.
"""

from __future__ import annotations

import os
import sys

import importlib.util

import pytest

# Load server.py directly (server/ directory shadows it in normal imports)
_ALICE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ALICE_ROOT)

_spec = importlib.util.spec_from_file_location("alice_server", os.path.join(_ALICE_ROOT, "server.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
app = _mod.app

from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_schema(self, client):
        data = client.get("/health").json()
        assert "uptime" in data
        assert "error_rate" in data
        assert "latency_p95" in data
        assert "memory_usage" in data

    def test_uptime_positive(self, client):
        data = client.get("/health").json()
        assert data["uptime"] >= 0.0

    def test_error_rate_bounded(self, client):
        data = client.get("/health").json()
        assert 0.0 <= data["error_rate"] <= 1.0


# ---------------------------------------------------------------------------
# /reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_returns_200(self, client):
        resp = client.post("/reset")
        assert resp.status_code == 200

    def test_reset_returns_episode_id(self, client):
        data = client.post("/reset").json()
        assert "episode_id" in data
        assert isinstance(data["episode_id"], str)
        assert len(data["episode_id"]) > 0

    def test_reset_returns_task(self, client):
        data = client.post("/reset").json()
        assert "task" in data
        assert isinstance(data["task"], str)

    def test_reset_returns_timestamp(self, client):
        data = client.post("/reset").json()
        assert "timestamp" in data

    def test_reset_returns_agent_version(self, client):
        data = client.post("/reset").json()
        assert "agent_version" in data

    def test_consecutive_resets_give_different_episode_ids(self, client):
        id1 = client.post("/reset").json()["episode_id"]
        id2 = client.post("/reset").json()["episode_id"]
        assert id1 != id2


# ---------------------------------------------------------------------------
# /state
# ---------------------------------------------------------------------------

class TestState:
    def test_state_returns_200(self, client):
        resp = client.get("/state")
        assert resp.status_code == 200

    def test_state_schema(self, client):
        data = client.get("/state").json()
        assert "turn_number" in data

    def test_state_has_episode_id_after_reset(self, client):
        client.post("/reset")
        data = client.get("/state").json()
        assert data["episode_id"] is not None

    def test_state_is_idempotent(self, client):
        client.post("/reset")
        s1 = client.get("/state").json()
        s2 = client.get("/state").json()
        assert s1["episode_id"] == s2["episode_id"]
        assert s1["turn_number"] == s2["turn_number"]


# ---------------------------------------------------------------------------
# /step
# ---------------------------------------------------------------------------

class TestStep:
    def _reset(self, client):
        return client.post("/reset").json()

    def test_step_returns_200_on_valid_action(self, client):
        episode = self._reset(client)
        resp = client.post("/step", json={"action": "test answer", "episode_id": episode["episode_id"]})
        assert resp.status_code == 200

    def test_step_returns_state_reward_done_info(self, client):
        episode = self._reset(client)
        data = client.post("/step", json={"action": "my answer", "episode_id": episode["episode_id"]}).json()
        assert "state" in data
        assert "reward" in data
        assert "done" in data
        assert "info" in data

    def test_step_reward_is_float(self, client):
        episode = self._reset(client)
        data = client.post("/step", json={"action": "my answer", "episode_id": episode["episode_id"]}).json()
        assert isinstance(data["reward"], (int, float))

    def test_step_done_is_bool(self, client):
        episode = self._reset(client)
        data = client.post("/step", json={"action": "my answer", "episode_id": episode["episode_id"]}).json()
        assert isinstance(data["done"], bool)

    def test_step_with_invalid_episode_id_returns_400(self, client):
        self._reset(client)
        resp = client.post("/step", json={"action": "my answer", "episode_id": "invalid-id-xyz"})
        assert resp.status_code == 400

    def test_step_with_empty_action_returns_422(self, client):
        episode = self._reset(client)
        resp = client.post("/step", json={"action": "", "episode_id": episode["episode_id"]})
        assert resp.status_code == 422

    def test_three_turns_complete_episode(self, client):
        episode = self._reset(client)
        eid = episode["episode_id"]
        done = False
        for i in range(3):
            r = client.post("/step", json={"action": f"answer {i}", "episode_id": eid}).json()
            done = r["done"]
        assert done is True

    def test_state_maintains_valid_state_after_invalid_step(self, client):
        episode = self._reset(client)
        eid = episode["episode_id"]
        # Invalid episode_id step
        client.post("/step", json={"action": "answer", "episode_id": "bad-id"})
        # State should still be valid
        state = client.get("/state").json()
        assert state["episode_id"] == eid
