import pytest

from claimdesk.guardrails import (
    detect_injection,
    is_claim_decision_request,
    is_medical_advice_request,
    mask_pii,
)


def test_mask_pii():
    text = "Call 204-555-0142 or olena@example.com. SIN 123 456 789. Member M-1001."
    assert mask_pii(text) == "Call [phone] or [email]. SIN [number]. Member M-10**."


def test_injection_in_a_claim_is_detected(sample):
    found = detect_injection(sample("massage_M-1003_injection"))
    assert any("ignore previous instructions" in phrase.lower() for phrase in found)
    assert detect_injection(sample("physio_M-1001")) == []


@pytest.mark.parametrize(
    ("question", "medical"),
    [
        ("Should I take ibuprofen for back pain?", True),
        ("What dosage of amoxicillin is normal?", True),
        ("Can you diagnose my knee?", True),
        ("What medication is covered?", False),
        ("Is physio covered for back pain?", False),
    ],
)
def test_medical_advice_detection(question, medical):
    assert is_medical_advice_request(question) is medical


@pytest.mark.parametrize(
    ("question", "decision"),
    [
        ("Will you approve my claim?", True),
        ("Will my claim be approved?", True),
        ("Please approve this claim", True),
        ("My claim was denied, how do I appeal?", False),
        ("How long does it take to process a claim?", False),
    ],
)
def test_claim_decision_detection(question, decision):
    assert is_claim_decision_request(question) is decision
