"""Unit tests for Pseudo-Brain Interactive Neural Console Backend."""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from irene_brain.console.server import create_console_app


@pytest.fixture(scope="module")
def client():
    app = create_console_app()
    return TestClient(app)


def test_console_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["working_memory_bytes"] == 4096
    assert data["working_slots"] == 16
    assert data["slot_width"] == 64


def test_console_slots(client):
    res = client.get("/api/slots")
    assert res.status_code == 200
    data = res.json()
    assert len(data["slots"]) == 16


def test_console_chat_reflex(client):
    res = client.post("/api/chat", json={
        "prompt": "Hello!",
        "mode": "reflex",
        "max_tokens": 10,
        "temperature": 0.1,
    })
    assert res.status_code == 200
    data = res.json()
    assert "response" in data
    assert "latency_ms" in data
    assert len(data["slot_energies"]) == 16
    assert data["working_memory_bytes"] == 4096


def test_console_lookahead_tree(client):
    res = client.post("/api/lookahead_tree", json={
        "prompt": "Solve for x: 2*x = 10",
        "branch_factor": 2,
        "horizon": 1,
    })
    assert res.status_code == 200
    data = res.json()
    assert "branches" in data
    assert len(data["branches"]) == 2
    assert "entropy" in data


def test_console_index_html(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "Pseudo-Brain" in res.text
