import json

from conftest import FakeChat

from claimdesk.agent import Agent
from claimdesk.llm import ChatResult, ToolCallRequest


def tools_used(answer):
    return [step.tool for step in answer.trace]


# ----- offline planner -----------------------------------------------------------------------


def test_balance_question_uses_tools_and_cites_the_section(components):
    answer = components.agent.answer("How much physio do I have left?", member_id="M-1001")
    assert "$446.00" in answer.answer
    assert tools_used(answer) == ["get_member", "get_benefit_usage", "search_policy"]
    assert [c.section for c in answer.citations] == ["4.2"]
    assert answer.flags == []


def test_member_id_can_come_from_the_question(components):
    answer = components.agent.answer("What would I get back for a $120 physio session? M-1001")
    assert "$96.00" in answer.answer
    assert "estimate_reimbursement" in tools_used(answer)
    assert {c.section for c in answer.citations} == {"4.2", "3.4"}


def test_waiting_period_is_explained(components):
    answer = components.agent.answer("Can M-1002 get a crown covered now?")
    assert "2027-06-01" in answer.answer
    assert [c.section for c in answer.citations] == ["5.2"]


def test_shared_dental_maximum(components):
    answer = components.agent.answer("How much dental do I have left? M-1001")
    assert "$1,292.00" in answer.answer
    assert {c.section for c in answer.citations} == {"5.3", "5.1"}


def test_policy_question_is_answered_from_the_booklet(components):
    answer = components.agent.answer("How long do I have to submit a claim?")
    assert "within 12 months of the date of service" in answer.answer
    assert "3.2" in [c.section for c in answer.citations]


def test_balance_question_without_member_asks_for_it(components):
    answer = components.agent.answer("How much physio do I have left?")
    assert answer.answer.startswith("To check your balance I need your member ID")


def test_unknown_member(components):
    answer = components.agent.answer("How much physio do I have left?", member_id="M-9999")
    assert "couldn't find member M-9999" in answer.answer


def test_medical_advice_is_refused(components):
    answer = components.agent.answer("Should I take ibuprofen for my back pain?")
    assert answer.refused is True
    assert answer.trace == []


def test_claim_decisions_are_left_to_examiners(components):
    answer = components.agent.answer("Will you approve my claim?")
    assert answer.refused is True
    assert "can't approve or deny" in answer.answer
    assert [c.section for c in answer.citations] == ["3.4"]


def test_member_ids_are_masked_in_the_trace(components):
    answer = components.agent.answer("How much physio do I have left?", member_id="M-1001")
    assert answer.trace[0].arguments == {"member_id": "M-10**"}


# ----- model path ----------------------------------------------------------------------------


def call(name, arguments, call_id="call_1"):
    return ToolCallRequest(call_id, name, json.dumps(arguments))


def test_model_loop_runs_tools_and_keeps_only_grounded_citations(components):
    fake = FakeChat(
        [
            ChatResult(
                None,
                [call("get_benefit_usage", {"member_id": "M-1001", "benefit": "physiotherapy"})],
            ),
            ChatResult("Olena has $446.00 left [4.2]. Appeals take 90 days [8]. See [9.9]."),
        ]
    )
    answer = Agent(components.toolbox, components.retriever, fake).answer(
        "How much physio do I have left?", member_id="M-1001"
    )
    # [8] is a real section, but no tool returned it, so the answer can't cite it.
    assert answer.answer == "Olena has $446.00 left [4.2]. Appeals take 90 days. See."
    assert answer.flags == ["removed_unsupported_citation:8", "removed_unsupported_citation:9.9"]
    assert [c.section for c in answer.citations] == ["4.2"]
    assert answer.provider == "fake"

    first, second = fake.calls
    assert "(Member ID: M-1001)" in first["messages"][1]["content"]
    assert {t["function"]["name"] for t in first["tools"]} >= {"search_policy"}
    tool_message = second["messages"][-1]
    assert tool_message["role"] == "tool" and tool_message["tool_call_id"] == "call_1"
    assert json.loads(tool_message["content"])["remaining"] == 446.0


def test_invalid_tool_arguments_go_back_to_the_model(components):
    fake = FakeChat(
        [
            ChatResult(None, [ToolCallRequest("c1", "search_policy", "{not json")]),
            ChatResult("Submit within 12 months [3.2]."),
        ]
    )
    answer = Agent(components.toolbox, components.retriever, fake).answer("Claim deadline?")
    assert answer.trace[0].summary.startswith("error")
    assert "removed_unsupported_citation:3.2" in answer.flags  # the search never succeeded


def test_step_limit_stops_a_looping_model(components):
    replies = [
        ChatResult(None, [call("search_policy", {"query": "physio"}, f"c{i}")]) for i in range(5)
    ]
    agent = Agent(components.toolbox, components.retriever, FakeChat(replies), max_steps=3)
    answer = agent.answer("physio?")
    assert "step_limit_reached" in answer.flags
    assert len(answer.trace) == 3


def test_provider_failure_falls_back_to_the_offline_planner(components):
    agent = Agent(components.toolbox, components.retriever, FakeChat([RuntimeError("503")]))
    answer = agent.answer("How long do I have to submit a claim?")
    assert "model_unavailable_answered_offline" in answer.flags
    assert "12 months" in answer.answer
