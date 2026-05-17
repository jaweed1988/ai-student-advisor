"""
Tests for the FastAPI endpoints.
OpenAI calls are mocked so tests run without an API key.
"""

import json
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from student_advisor.api import app
from student_advisor.data_generator import generate, save

client = TestClient(app)

# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True, scope="module")
def seed_data(tmp_path_factory):
    """Write a small test dataset before any test in this module."""
    from student_advisor import config
    tmp = tmp_path_factory.mktemp("data")
    test_file = tmp / "students.csv"
    config.DATA_FILE = test_file
    df = generate(n=20, seed=42)
    df.to_csv(test_file, index=False)
    return test_file


MOCK_GPT_RESPONSE = {
    "risk_level": "Medium",
    "risk_summary": "Student shows moderate risk with attendance concerns.",
    "key_concerns": ["Attendance below 70%"],
    "interventions": [
        {"action": "Advisor outreach", "detail": "Schedule a meeting.", "priority": "High"}
    ],
    "urgency": "Within 1 week",
    "positive_signals": ["GPA above 3.0"],
}


def _mock_openai(monkeypatch):
    """Patch OpenAI client to return a fixed response."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = json.dumps(MOCK_GPT_RESPONSE)
    mock_response.usage.total_tokens = 120

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    monkeypatch.setattr("student_advisor.advisor._client", lambda: mock_client)
    return mock_client


# ── Health ─────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_returns_200(self):
        assert client.get("/health").status_code == 200

    def test_response_shape(self):
        data = client.get("/health").json()
        assert "status" in data
        assert data["status"] == "healthy"


# ── Student list ───────────────────────────────────────────────────────────────

class TestStudents:
    def test_returns_200(self):
        assert client.get("/students").status_code == 200

    def test_has_students_key(self):
        data = client.get("/students").json()
        assert "students" in data
        assert isinstance(data["students"], list)

    def test_limit_param(self):
        data = client.get("/students?limit=5").json()
        assert data["returned"] <= 5


# ── AI advisor ─────────────────────────────────────────────────────────────────

class TestAdvise:
    def test_valid_student_200(self, monkeypatch):
        _mock_openai(monkeypatch)
        df = generate(n=20, seed=42)
        sid = df["student_id"].iloc[0]
        response = client.get(f"/students/{sid}/advise")
        assert response.status_code == 200

    def test_response_has_required_fields(self, monkeypatch):
        _mock_openai(monkeypatch)
        df = generate(n=20, seed=42)
        sid = df["student_id"].iloc[0]
        data = client.get(f"/students/{sid}/advise").json()
        for field in ["student_id", "risk_level", "risk_summary", "interventions", "urgency"]:
            assert field in data, f"Missing field: {field}"

    def test_risk_level_is_valid(self, monkeypatch):
        _mock_openai(monkeypatch)
        df = generate(n=20, seed=42)
        sid = df["student_id"].iloc[0]
        data = client.get(f"/students/{sid}/advise").json()
        assert data["risk_level"] in {"Low", "Medium", "High"}

    def test_invalid_student_404(self):
        response = client.get("/students/INVALID_99999/advise")
        assert response.status_code == 404
