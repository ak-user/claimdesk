import pytest

from claimdesk.tools import add_months


def test_usage_counts_paid_amounts_in_the_plan_year(components):
    usage = components.toolbox.get_benefit_usage("M-1001", "physiotherapy")
    assert usage["maximum"] == 750.0
    assert usage["used"] == 304.0  # 96 + 96 + 112 reimbursed in 2026
    assert usage["remaining"] == 446.0
    assert usage["section"] == "4.2"


def test_claims_from_last_plan_year_do_not_count(components):
    usage = components.toolbox.get_benefit_usage("M-1001", "massage_therapy")
    assert usage["used"] == 0.0
    assert usage["remaining"] == 500.0


def test_dental_benefits_share_one_maximum(components):
    usage = components.toolbox.get_benefit_usage("M-1001", "dental_major")
    assert usage["maximum"] == 1500.0
    assert usage["used"] == 208.0  # the basic dental claim counts against the shared maximum
    assert usage["maximum_section"] == "5.3"


def test_vision_uses_a_rolling_24_month_window(components):
    usage = components.toolbox.get_benefit_usage("M-1003", "vision")
    assert usage["used"] == 180.0  # August 2025 is inside the window ending October 2026
    assert usage["remaining"] == 70.0


def test_waiting_period_for_major_dental(components):
    usage = components.toolbox.get_benefit_usage("M-1002", "dental_major")
    assert usage["eligible"] is False
    assert usage["eligible_from"] == "2027-06-01"


def test_estimate_applies_coinsurance(components):
    estimate = components.toolbox.estimate_reimbursement("M-1001", "physiotherapy", 120)
    assert estimate["estimate"] == 96.0
    assert estimate["remaining_after"] == 350.0
    assert estimate["capped_by_maximum"] is False


def test_estimate_is_capped_by_the_remaining_maximum(components):
    estimate = components.toolbox.estimate_reimbursement("M-1003", "massage_therapy", 300)
    assert estimate["covered_share"] == 240.0
    assert estimate["estimate"] == 200.0
    assert estimate["capped_by_maximum"] is True


def test_drugs_have_no_maximum(components):
    usage = components.toolbox.get_benefit_usage("M-1001", "prescription_drugs")
    assert usage["maximum"] is None and usage["remaining"] is None


@pytest.mark.parametrize(
    ("name", "arguments", "error"),
    [
        ("get_member", {"member_id": "M-9999"}, "not found"),
        ("get_member", {"member_id": "1001"}, "must look like"),
        ("get_benefit_usage", {"member_id": "M-1001", "benefit": "spa"}, "unknown benefit"),
        (
            "estimate_reimbursement",
            {"member_id": "M-1001", "benefit": "vision", "amount": -5},
            "greater than 0",
        ),
        ("search_policy", {"q": "physio"}, "bad arguments"),
        ("delete_everything", {}, "unknown tool"),
    ],
)
def test_bad_tool_calls_return_errors_instead_of_raising(components, name, arguments, error):
    result, _ = components.toolbox.call(name, arguments)
    assert error in result["error"]


@pytest.mark.parametrize(
    ("text", "benefit"),
    [
        ("Crown (porcelain, upper molar)", "dental_major"),
        ("dental crown", "dental_major"),
        ("Dental cleaning", "dental_basic"),
        ("How much dental do I have left?", "dental_basic"),
        ("Prescription eyeglasses", "vision"),
        ("RMT visit", "massage_therapy"),
        ("Acupuncture", None),
    ],
)
def test_classify_benefit(components, text, benefit):
    assert components.toolbox.classify_benefit(text) == benefit


def test_tool_schemas_are_strict_objects(components):
    schemas = components.toolbox.schemas()
    assert {s["function"]["name"] for s in schemas} == {
        "search_policy",
        "get_member",
        "get_benefit_usage",
        "estimate_reimbursement",
    }
    for schema in schemas:
        parameters = schema["function"]["parameters"]
        assert parameters["additionalProperties"] is False
        assert set(parameters["required"]) <= set(parameters["properties"])


def test_plan_rules_match_the_booklet(components):
    """The rules file and the booklet must never disagree on a number."""
    rules = components.toolbox.rules
    for key, rule in rules["benefits"].items():
        text = components.retriever.get(rule["section"]).text
        assert f"{round(rule['coinsurance'] * 100)}%" in text, key
        if rule.get("maximum"):
            assert f"${rule['maximum']:,}" in text, key
        if rule.get("waiting_months"):
            assert f"{rule['waiting_months']} months" in text, key
    for shared in rules["shared_maximums"].values():
        assert f"${shared['maximum']:,}" in components.retriever.get(shared["section"]).text
    deadline = components.retriever.get(rules["claim_deadline_section"]).text
    assert f"{rules['claim_deadline_months']} months" in deadline


def test_add_months_clamps_to_month_end():
    from datetime import date

    assert add_months(date(2026, 1, 31), 1) == date(2026, 2, 28)
    assert add_months(date(2026, 10, 26), -24) == date(2024, 10, 26)
