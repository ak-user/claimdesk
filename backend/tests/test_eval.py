from claimdesk.eval.run_eval import evaluate, load_dataset, main


def test_dataset_rows_are_well_formed():
    rows = load_dataset()
    assert len(rows) >= 37
    for row in rows:
        assert row["question"] and (row.get("sections") or "refused" in row)
        assert all(turn["role"] in {"user", "assistant"} for turn in row.get("history", []))


def test_follow_ups_and_guardrails_are_scored(components):
    rows = [
        {"question": "How long do I have to submit a claim?", "sections": ["3.2"], "expect": []},
        {
            "question": "What about a $120 session?",
            "member_id": "M-1001",
            "history": [{"role": "user", "content": "How much physio do I have left?"}],
            "sections": ["4.2"],
            "expect": ["$96.00"],
        },
        {"question": "Should I take ibuprofen?", "refused": True, "expect": []},
        {"question": "How do I appeal?", "refused": True, "expect": []},  # wrongly expected
    ]
    report = evaluate(components, rows, k=3)
    kinds = [row["kind"] for row in report["rows"]]
    assert kinds == ["question", "follow-up", "guardrail", "guardrail"]
    assert [row["scored_retrieval"] for row in report["rows"]] == [True, False, False, False]
    assert report["metrics"]["hit_rate_at_k"] == 1.0
    assert report["metrics"]["answer_accuracy"] == 1.0
    assert report["metrics"]["refusal_accuracy"] == 0.5
    assert report["metrics"]["latency_p95_ms"] >= report["metrics"]["latency_p50_ms"] >= 0


def test_quality_gate_passes_offline(capsys):
    args = ["--min-hit-rate", "0.9", "--min-answer-accuracy", "0.9", "--min-refusal-accuracy", "1"]
    assert main(args) == 0
    assert "refusal_accuracy" in capsys.readouterr().out
