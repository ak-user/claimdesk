from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from claimdesk.api import create_app
from claimdesk.store import SAMPLES_DIR


@pytest.fixture(scope="module")
def client(settings):
    app = create_app(settings.model_copy(update={"claimdesk_static_dir": Path("/nonexistent")}))
    return TestClient(app)


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["llm_provider"] == "offline"
    assert body["retrieval"] == "bm25"
    assert body["sections"] == 23
    assert body["as_of"] == "2026-10-26"


def test_ask(client):
    response = client.post(
        "/api/ask", json={"question": "How much physio do I have left?", "member_id": "M-1001"}
    )
    assert response.status_code == 200
    body = response.json()
    assert "$446.00" in body["answer"]
    assert body["citations"][0]["section"] == "4.2"
    assert body["trace"][0]["tool"] == "get_member"


def test_ask_validates_input(client):
    assert client.post("/api/ask", json={"question": ""}).status_code == 422
    assert client.post("/api/ask", json={"question": "x" * 1001}).status_code == 422


def test_assess_text(client):
    text = (SAMPLES_DIR / "vision_late_M-1003.txt").read_text()
    body = client.post("/api/claims/assess", json={"text": text}).json()
    assert body["status"] == "not_payable"


def test_upload_pdf(client):
    pdf = (SAMPLES_DIR / "physio_M-1001.pdf").read_bytes()
    response = client.post(
        "/api/claims/upload", files={"file": ("claim.pdf", pdf, "application/pdf")}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "ready_for_review"


def test_upload_rejects_other_file_types(client):
    response = client.post(
        "/api/claims/upload", files={"file": ("claim.exe", b"MZ", "application/octet-stream")}
    )
    assert response.status_code == 400
    assert "Upload a .txt, .md or .pdf file." in response.json()["detail"]


def test_samples(client):
    names = [s["name"] for s in client.get("/api/samples").json()]
    assert names == [
        "dental_crown_M-1002",
        "massage_M-1003_injection",
        "physio_M-1001",
        "vision_late_M-1003",
    ]
    assert "Member ID: M-1001" in client.get("/api/samples/physio_M-1001").json()["text"]
    assert client.get("/api/samples/nope").status_code == 404
    assert client.get("/api/samples/..%2Fplan_rules").status_code == 404


def test_ask_with_history(client):
    response = client.post(
        "/api/ask",
        json={
            "question": "What about a $120 session?",
            "member_id": "M-1001",
            "history": [{"role": "user", "content": "How much physio do I have left?"}],
        },
    )
    assert response.status_code == 200
    assert "$96.00" in response.json()["answer"]
    too_long = [{"role": "user", "content": "hi"}] * 21
    assert client.post("/api/ask", json={"question": "x", "history": too_long}).status_code == 422
