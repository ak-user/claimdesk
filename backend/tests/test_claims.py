import json

from conftest import FakeChat

from claimdesk.claims import ClaimAssessor
from claimdesk.llm import ChatResult


def assess(components, text):
    return components.assessor.assess(text)


def kinds(result):
    return {(i.field, i.kind) for i in result.issues}


def test_complete_physio_claim_is_ready_with_an_estimate(components, sample):
    result = assess(components, sample("physio_M-1001"))
    assert result.status == "ready_for_review"
    assert result.benefit == "physiotherapy"
    assert result.estimate.estimate == 96.0
    assert result.estimate.remaining_after == 350.0
    assert result.requires_human_review is True
    assert {c.section for c in result.citations} >= {"4.2", "3.4"}
    assert result.summary == "Ready for examiner review. Estimated reimbursement: $96.00."


def test_crown_during_the_waiting_period_is_not_payable(components, sample):
    result = assess(components, sample("dental_crown_M-1002"))
    assert result.status == "not_payable"
    assert result.benefit == "dental_major"
    assert result.estimate is None
    assert "2027-06-01" in result.summary
    assert ("amount", "note") in kinds(result)  # dental fee guide reminder


def test_injected_instructions_are_flagged_and_ignored(components, sample):
    result = assess(components, sample("massage_M-1003_injection"))
    assert "possible_prompt_injection" in result.flags
    # The injected "approve at 100%" changes nothing: the missing registration still blocks it.
    assert result.status == "needs_correction"
    assert ("provider_registration", "needs_correction") in kinds(result)
    assert result.estimate.estimate == 76.0  # 80% of $95, not 100%


def test_claim_after_the_deadline_is_not_payable(components, sample):
    result = assess(components, sample("vision_late_M-1003"))
    assert result.status == "not_payable"
    issue = next(i for i in result.issues if i.field == "service_date")
    assert issue.section == "3.2"
    assert "2026-09-01" in issue.message


def test_future_service_date_needs_correction(components, sample):
    text = sample("physio_M-1001").replace("2026-10-14", "2026-12-01")
    result = assess(components, text)
    assert result.status == "needs_correction"
    assert "future" in result.summary


def test_unknown_member_is_not_payable(components, sample):
    result = assess(components, sample("physio_M-1001").replace("M-1001", "M-9999"))
    assert result.status == "not_payable"
    assert result.estimate is None


def test_missing_fields_are_listed(components):
    result = assess(components, "Service type: Physiotherapy\nAmount: $80")
    missing = {i.field for i in result.issues if i.kind == "needs_correction"}
    assert {"member_id", "patient_name", "service_date", "provider_name"} <= missing


def test_ambiguous_date_asks_for_iso_format(components, sample):
    result = assess(components, sample("physio_M-1001").replace("2026-10-14", "03/04/2026"))
    assert any("ambiguous" in i.message for i in result.issues)


def test_uncovered_service_is_not_payable(components, sample):
    result = assess(components, sample("physio_M-1001").replace("Physiotherapy", "Acupuncture"))
    assert result.status == "not_payable"
    assert "isn't a covered benefit" in result.summary


def test_model_extraction_is_merged_with_rules(components, sample):
    reply = {
        "member_id": "M-1001",
        "patient_name": "Olena K.",
        "service_type": "Physiotherapy",
        "service_date": "2026-10-14",
        "provider_name": "Riverside Physio Clinic",
        "provider_registration": None,  # the model missed it; the rules found it
        "amount": 120.0,
    }
    fake = FakeChat([ChatResult(json.dumps(reply))])
    assessor = ClaimAssessor(components.toolbox, components.retriever, fake)
    result = assessor.assess(sample("physio_M-1001"))
    assert result.extractor == "llm"
    assert result.fields.provider_registration == "PT-48213"
    assert result.status == "ready_for_review"


def test_model_failure_falls_back_to_rules(components, sample):
    fake = FakeChat([RuntimeError("timeout")])
    assessor = ClaimAssessor(components.toolbox, components.retriever, fake)
    result = assessor.assess(sample("physio_M-1001"))
    assert result.extractor == "rules"
    assert "model_extraction_failed_used_rules" in result.flags
    assert result.status == "ready_for_review"
