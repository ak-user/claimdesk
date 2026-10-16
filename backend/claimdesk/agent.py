"""The question-answering agent.

With a model configured, ClaimDesk runs a tool-calling loop: the model chooses tools, ClaimDesk
runs them, and the model writes the answer from their results. Without a model, a small
deterministic planner calls the same tools, so the demo, the tests and the eval run with no API
key. Both paths end in the same citation check: a section the tools never returned is removed
from the answer and reported in `flags`.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from .guardrails import (
    MEDICAL_REFUSAL,
    is_claim_decision_request,
    is_medical_advice_request,
    mask_pii,
)
from .ingest import snippet, split_sentences
from .llm import ChatClient
from .models import AnswerOut, Citation, ToolStep, Turn
from .retrieval import BookletIndex, tokenize
from .tools import Toolbox

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are ClaimDesk, an assistant for members of the {insurer} group benefits \
plan {plan}. All data is fictional demo data. Today's date is {as_of}.

Rules:
1. Use the tools. Base every statement on tool results. If the tools don't have the answer, \
say so and suggest contacting member services.
2. Cite booklet sections in square brackets, like [4.2]. Cite only sections that appear in tool \
results.
3. For balances and estimates, call the tools and copy their dollar amounts. Don't calculate \
them yourself.
4. You can't approve or deny claims. A claims examiner reviews every claim.
5. Don't give medical advice.
6. Answer in plain language, in at most 120 words."""

CITATION = re.compile(r"\[(\d+(?:\.\d+)?(?:\s*,\s*\d+(?:\.\d+)?)*)\]")
MEMBER_IN_TEXT = re.compile(r"\bM-\d{4}\b", re.IGNORECASE)
AMOUNT_IN_TEXT = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)")
USAGE_INTENT = re.compile(
    r"\b(?:left|remaining|remain|used|balance|still\s+have|available"
    r"|how\s+much\s+(?:do|have|can|is))\b",
    re.IGNORECASE,
)
COVERAGE_INTENT = re.compile(
    r"\b(?:covered|cover|eligible|get\s+back|reimburs\w*|pay\s+for|now)\b", re.IGNORECASE
)
# Short questions that lean on the previous turn: "what about massage?", "and for $90?"
FOLLOW_UP = re.compile(
    r"^\s*(?:and|also|what\s+about|how\s+about|what\s+if|same\s+for|then)\b", re.IGNORECASE
)
SECTION_KEYS = {"section", "maximum_section", "review_section"}
# The model sees this many earlier turns; older ones rarely matter and cost tokens.
HISTORY_TURNS = 6


def money(value: float) -> str:
    return f"${value:,.2f}"


class Agent:
    def __init__(
        self,
        toolbox: Toolbox,
        retriever: BookletIndex,
        client: ChatClient | None = None,
        max_steps: int = 6,
    ) -> None:
        self.toolbox = toolbox
        self.retriever = retriever
        self.client = client
        self.max_steps = max_steps

    @property
    def provider(self) -> str:
        return self.client.provider if self.client else "offline"

    def answer(
        self, question: str, member_id: str | None = None, history: list[Turn] | None = None
    ) -> AnswerOut:
        question = question.strip()
        history = list(history or [])[-HISTORY_TURNS:]
        member_id = member_id.strip().upper() if member_id and member_id.strip() else None
        if is_medical_advice_request(question):
            return AnswerOut(
                answer=MEDICAL_REFUSAL, citations=[], trace=[], provider=self.provider, refused=True
            )
        if is_claim_decision_request(question):
            section = self.toolbox.rules["human_review_section"]
            text = (
                "I can't approve or deny claims. A claims examiner reviews every claim before "
                f"payment [{section}]. I can check what the plan covers and estimate a "
                "reimbursement for you."
            )
            return self._finish(text, {section}, [], refused=True)
        if self.client is None:
            return self._offline(question, member_id, history)
        try:
            return self._with_model(question, member_id, history)
        except Exception:  # noqa: BLE001 - keep answering if the provider is down
            log.exception("Model call failed; answering with the offline planner.")
            answer = self._offline(question, member_id, history)
            answer.flags.append("model_unavailable_answered_offline")
            return answer

    # ----- model path ------------------------------------------------------------------------

    def _with_model(self, question: str, member_id: str | None, history: list[Turn]) -> AnswerOut:
        assert self.client is not None
        rules = self.toolbox.rules
        system = SYSTEM_PROMPT.format(
            insurer=rules["insurer"],
            plan=rules["plan_number"],
            as_of=self.toolbox.as_of.isoformat(),
        )
        user = question if not member_id else f"{question}\n\n(Member ID: {member_id})"
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            *({"role": turn.role, "content": turn.content} for turn in history),
            {"role": "user", "content": user},
        ]
        tools = self.toolbox.schemas()
        trace: list[ToolStep] = []
        allowed: set[str] = set()
        flags: list[str] = []
        final: str | None = None

        for _ in range(self.max_steps):
            result = self.client.chat(messages, tools=tools)
            if not result.tool_calls:
                final = result.content or ""
                break
            messages.append(
                {
                    "role": "assistant",
                    "content": result.content,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {"name": call.name, "arguments": call.arguments},
                        }
                        for call in result.tool_calls
                    ],
                }
            )
            for call in result.tool_calls:
                arguments = _parse_arguments(call.arguments)
                if arguments is None:
                    output: dict[str, Any] = {"error": "arguments must be a JSON object"}
                    ms = 0.0
                else:
                    output, ms = self.toolbox.call(call.name, arguments)
                trace.append(_step(call.name, arguments or {}, output, ms))
                allowed.update(_sections(output))
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": json.dumps(output)}
                )

        if final is None:
            final = "I couldn't finish this answer. Try asking a more specific question."
            flags.append("step_limit_reached")
        return self._finish(final, allowed, trace, flags=flags)

    # ----- offline path ----------------------------------------------------------------------

    def _offline(self, question: str, member_id: str | None, history: list[Turn]) -> AnswerOut:
        trace: list[ToolStep] = []
        allowed: set[str] = set()

        def run(name: str, **arguments: Any) -> dict[str, Any]:
            output, ms = self.toolbox.call(name, arguments)
            trace.append(_step(name, arguments, output, ms))
            allowed.update(_sections(output))
            return output

        found = MEMBER_IN_TEXT.search(question)
        member = member_id or (found.group(0).upper() if found else None)
        benefit = self.toolbox.classify_benefit(question)
        amount = _amount(question)
        about_account = bool(
            USAGE_INTENT.search(question) or COVERAGE_INTENT.search(question) or amount
        )
        if history and (FOLLOW_UP.search(question) or amount):
            earlier_benefit, earlier_member, earlier_about_account = self._from_history(history)
            benefit = benefit or earlier_benefit
            member = member or earlier_member
            about_account = about_account or earlier_about_account

        if member and benefit and about_account:
            person = run("get_member", member_id=member)
            if "error" in person:
                text = (
                    f"I couldn't find member {member} on this plan. Check the member ID on your "
                    "benefits card."
                )
                return self._finish(text, allowed, trace)
            usage = run("get_benefit_usage", member_id=member, benefit=benefit)
            estimate = None
            if amount and "error" not in usage:
                estimate = run(
                    "estimate_reimbursement", member_id=member, benefit=benefit, amount=amount
                )
            if "error" not in usage:
                run("search_policy", query=usage["display"])
            return self._finish(_usage_text(person, usage, estimate), allowed, trace)

        results = run("search_policy", query=question).get("results", [])
        text = _extractive_answer(question, results)
        if benefit and USAGE_INTENT.search(question) and not member:
            text = "To check your balance I need your member ID, for example M-1001. " + text
        return self._finish(text, allowed, trace)

    def _from_history(self, history: list[Turn]) -> tuple[str | None, str | None, bool]:
        """The most recent benefit and member ID the member mentioned, and whether the last
        question was about their own account rather than the policy in general."""
        benefit = member = None
        user_turns = [turn.content for turn in reversed(history) if turn.role == "user"]
        for text in user_turns:
            benefit = benefit or self.toolbox.classify_benefit(text)
            if member is None and (found := MEMBER_IN_TEXT.search(text)):
                member = found.group(0).upper()
            if benefit and member:
                break
        last = user_turns[0] if user_turns else ""
        about_account = bool(
            USAGE_INTENT.search(last) or COVERAGE_INTENT.search(last) or _amount(last)
        )
        return benefit, member, about_account

    # ----- shared ----------------------------------------------------------------------------

    def _finish(
        self,
        text: str,
        allowed: set[str],
        trace: list[ToolStep],
        refused: bool = False,
        flags: list[str] | None = None,
    ) -> AnswerOut:
        """Keeps only citations to sections the tools actually returned."""
        flags = list(flags or [])
        cited: list[str] = []

        def check(match: re.Match[str]) -> str:
            sections = [s.strip() for s in match.group(1).split(",")]
            kept = [s for s in sections if s in allowed and self.retriever.get(s) is not None]
            flags.extend(f"removed_unsupported_citation:{s}" for s in sections if s not in kept)
            cited.extend(s for s in kept if s not in cited)
            return f"[{', '.join(kept)}]" if kept else ""

        text = CITATION.sub(check, text)
        text = re.sub(r"\s+([.,;:])", r"\1", re.sub(r"[ \t]{2,}", " ", text)).strip()
        if not cited and allowed and not refused:
            flags.append("answer_has_no_citation")
        citations = []
        for section in cited:
            chunk = self.retriever.get(section)
            if chunk is not None:
                citations.append(
                    Citation(section=section, title=chunk.title, snippet=snippet(chunk.text))
                )
        return AnswerOut(
            answer=text,
            citations=citations,
            trace=trace,
            provider=self.provider,
            refused=refused,
            flags=flags,
        )


def _usage_text(
    person: dict[str, Any], usage: dict[str, Any], estimate: dict[str, Any] | None
) -> str:
    if "error" in usage:
        return f"I couldn't check that benefit: {usage['error']}."
    name = person["first_name"]
    display = usage["display"]
    lower = display[0].lower() + display[1:]
    section, maximum_section = usage["section"], usage["maximum_section"]

    if not usage["eligible"]:
        return (
            f"{display} isn't covered for {name} yet. It's covered after "
            f"{usage['waiting_months']} months in the plan, so {name} is covered from "
            f"{usage['eligible_from']} [{section}]."
        )
    if usage["maximum"] is None:
        text = (
            f"The plan pays {usage['coinsurance_percent']}% of eligible {lower} costs and there is "
            f"no annual maximum [{section}]. {name} has had {money(usage['used'])} reimbursed "
            f"{usage['period_label']}."
        )
    else:
        shared = usage.get("shared_maximum")
        pool = shared[0].lower() + shared[1:] if shared else lower
        text = (
            f"{name} has {money(usage['remaining'])} left for {pool} {usage['period_label']}. "
            f"The maximum is {money(usage['maximum'])} and {money(usage['used'])} has been paid "
            f"so far{f' [{maximum_section}]' if shared else ''}. The plan pays "
            f"{usage['coinsurance_percent']}% of eligible {lower} expenses [{section}]."
        )
    if estimate and "error" not in estimate and estimate.get("eligible"):
        capped = ", limited by the remaining maximum" if estimate["capped_by_maximum"] else ""
        text += (
            f" For a {money(estimate['amount'])} charge, the estimated reimbursement is "
            f"{money(estimate['estimate'])}{capped}. A claims examiner confirms the final amount "
            f"[{estimate['review_section']}]."
        )
    return text


def _extractive_answer(question: str, results: list[dict[str, Any]]) -> str:
    """Offline answer: the booklet sentences that best match the question, with citations."""
    if not results:
        return (
            "I couldn't find that in the benefits booklet. Contact Harbourline Mutual member "
            "services for help."
        )
    query = set(tokenize(question, expand=True))
    picks = [results[0]]
    if len(results) > 1 and results[1]["score"] >= 0.6 * results[0]["score"]:
        picks.append(results[1])
    parts = []
    for position, result in enumerate(picks):
        sentences = _best_sentences(result["text"], query, 2 if position == 0 else 1)
        if position > 0 and len(set(tokenize(" ".join(sentences))) & query) < 2:
            continue
        parts.append(f"{' '.join(sentences)} [{result['section']}]")
    return " ".join(parts)


def _best_sentences(text: str, query: set[str], limit: int) -> list[str]:
    sentences = split_sentences(text)
    scored = [
        (len(set(tokenize(s)) & query) + (0.1 if i == 0 else 0.0), i, s)
        for i, s in enumerate(sentences)
    ]
    best = [item for item in sorted(scored, key=lambda t: -t[0])[:limit] if item[0] >= 1]
    if not best:
        return sentences[:1]
    return [s for _, _, s in sorted(best, key=lambda t: t[1])]


def _parse_arguments(raw: str) -> dict[str, Any] | None:
    try:
        value = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _amount(question: str) -> float | None:
    match = AMOUNT_IN_TEXT.search(question)
    return float(match.group(1).replace(",", "")) if match else None


def _sections(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in SECTION_KEYS and isinstance(item, str):
                found.add(item)
            else:
                found |= _sections(item)
    elif isinstance(value, list):
        for item in value:
            found |= _sections(item)
    return found


def _step(name: str, arguments: dict[str, Any], output: dict[str, Any], ms: float) -> ToolStep:
    masked = {k: mask_pii(v) if isinstance(v, str) else v for k, v in arguments.items()}
    return ToolStep(tool=name, arguments=masked, summary=_summarize(name, output), ms=ms)


def _summarize(name: str, output: dict[str, Any]) -> str:
    if "error" in output:
        return f"error: {output['error']}"
    if name == "search_policy":
        sections = [r["section"] for r in output.get("results", [])]
        return f"sections {', '.join(sections)}" if sections else "no matches"
    if name == "get_member":
        return f"member found, enrolled {output['enrolled']}"
    if name == "get_benefit_usage":
        if not output["eligible"]:
            return f"waiting period, covered from {output['eligible_from']}"
        if output["remaining"] is None:
            return f"used {money(output['used'])}, no maximum"
        return f"used {money(output['used'])}, remaining {money(output['remaining'])}"
    if name == "estimate_reimbursement":
        return f"estimate {money(output['estimate'])}"
    return "done"
