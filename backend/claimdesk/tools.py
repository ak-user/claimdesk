"""The tools the agent can call. Every number in an answer comes from here, not from the model."""

from __future__ import annotations

import inspect
import re
import time
from calendar import monthrange
from datetime import date, timedelta
from typing import Any

from .retrieval import Retriever

MEMBER_ID = re.compile(r"^M-\d{4}$")


def add_months(day: date, months: int) -> date:
    month_index = day.month - 1 + months
    year, month = day.year + month_index // 12, month_index % 12 + 1
    return date(year, month, min(day.day, monthrange(year, month)[1]))


class ToolError(ValueError):
    pass


class Toolbox:
    def __init__(
        self,
        retriever: Retriever,
        rules: dict[str, Any],
        members: dict[str, dict[str, Any]],
        as_of: date,
        top_k: int = 4,
    ) -> None:
        self.retriever = retriever
        self.rules = rules
        self.members = members
        self.as_of = as_of
        self.top_k = top_k
        self.benefits: dict[str, dict[str, Any]] = rules["benefits"]
        self._keywords = sorted(
            ((kw, key) for key, rule in self.benefits.items() for kw in rule["keywords"]),
            key=lambda pair: len(pair[0]),
            reverse=True,
        )

    # ----- helpers ---------------------------------------------------------------------------

    def classify_benefit(self, text: str) -> str | None:
        """Maps free text ("Crown (porcelain)", "RMT visit") to a benefit key.

        Specific keywords win, longest first; generic words like "dental" only count when nothing
        more specific matched, so "dental crown" is major dental, not basic.
        """
        lowered = text.lower()
        fallback = self.rules.get("fallback_keywords", {})
        for keyword, key in [*self._keywords, *fallback.items()]:
            if re.search(rf"\b{re.escape(keyword)}\b", lowered):
                return key
        return None

    def section_of(self, benefit: str) -> str:
        return self.benefits[benefit]["section"]

    def _member(self, member_id: str) -> dict[str, Any]:
        member_id = member_id.strip().upper()
        if not MEMBER_ID.match(member_id):
            raise ToolError("member_id must look like M-1234")
        member = self.members.get(member_id)
        if member is None:
            raise ToolError(f"member {member_id} not found on plan {self.rules['plan_number']}")
        return member

    def _benefit(self, benefit: str) -> dict[str, Any]:
        if benefit not in self.benefits:
            raise ToolError(f"unknown benefit {benefit!r}; use one of {sorted(self.benefits)}")
        return self.benefits[benefit]

    # ----- tools -----------------------------------------------------------------------------

    def search_policy(self, query: str) -> dict[str, Any]:
        query = query.strip()
        if not query:
            raise ToolError("query must not be empty")
        hits = self.retriever.search(query, self.top_k)
        return {
            "results": [
                {
                    "section": h.chunk.section,
                    "title": h.chunk.title,
                    "text": h.chunk.text,
                    "score": h.score,
                }
                for h in hits
            ]
        }

    def get_member(self, member_id: str) -> dict[str, Any]:
        member = self._member(member_id)
        return {
            "member_id": member["member_id"],
            "first_name": member["first_name"],
            "plan_number": member["plan_number"],
            "enrolled": member["enrolled"].isoformat(),
        }

    def get_benefit_usage(self, member_id: str, benefit: str) -> dict[str, Any]:
        member = self._member(member_id)
        rule = self._benefit(benefit)
        shared_key = rule.get("shared_maximum")
        group = {benefit}
        maximum = rule.get("maximum")
        maximum_section = rule["section"]
        shared_display = None
        if shared_key:
            shared = self.rules["shared_maximums"][shared_key]
            group = {k for k, r in self.benefits.items() if r.get("shared_maximum") == shared_key}
            maximum, maximum_section = shared["maximum"], shared["section"]
            shared_display = shared["display"]

        if rule["period"] == "rolling_24_months":
            start = add_months(self.as_of, -24) + timedelta(days=1)
            end = self.as_of
            period_label = "in the last 24 months"
        else:
            start, end = date(self.as_of.year, 1, 1), date(self.as_of.year, 12, 31)
            period_label = f"in the {self.as_of.year} plan year"

        used = round(
            sum(
                c["paid"]
                for c in member["claims"]
                if c["benefit"] in group and start <= c["service_date"] <= end
            ),
            2,
        )
        eligible_from = None
        if rule.get("waiting_months"):
            eligible_from = add_months(member["enrolled"], rule["waiting_months"])

        return {
            "member_id": member["member_id"],
            "benefit": benefit,
            "display": rule["display"],
            "section": rule["section"],
            "coinsurance_percent": round(rule["coinsurance"] * 100),
            "maximum": float(maximum) if maximum is not None else None,
            "maximum_section": maximum_section,
            "shared_maximum": shared_display,
            "period_label": period_label,
            "window_start": start.isoformat(),
            "used": used,
            "remaining": round(max(maximum - used, 0.0), 2) if maximum is not None else None,
            "eligible": eligible_from is None or self.as_of >= eligible_from,
            "eligible_from": eligible_from.isoformat() if eligible_from else None,
            "waiting_months": rule.get("waiting_months"),
        }

    def estimate_reimbursement(
        self, member_id: str, benefit: str, amount: float, service_date: str | None = None
    ) -> dict[str, Any]:
        try:
            amount = float(amount)
        except (TypeError, ValueError) as error:
            raise ToolError("amount must be a number") from error
        if amount <= 0:
            raise ToolError("amount must be greater than 0")
        usage = self.get_benefit_usage(member_id, benefit)
        on_date = date.fromisoformat(service_date) if service_date else self.as_of
        base = {
            "member_id": usage["member_id"],
            "benefit": benefit,
            "display": usage["display"],
            "section": usage["section"],
            "review_section": self.rules["human_review_section"],
            "amount": round(amount, 2),
            "coinsurance_percent": usage["coinsurance_percent"],
        }
        if usage["eligible_from"] and on_date < date.fromisoformat(usage["eligible_from"]):
            return {
                **base,
                "eligible": False,
                "estimate": 0.0,
                "reason": f"{usage['display']} is covered from {usage['eligible_from']}",
            }
        covered = round(amount * usage["coinsurance_percent"] / 100, 2)
        remaining = usage["remaining"]
        estimate = min(covered, remaining) if remaining is not None else covered
        return {
            **base,
            "eligible": True,
            "covered_share": covered,
            "estimate": round(estimate, 2),
            "remaining_before": remaining,
            "remaining_after": round(remaining - estimate, 2) if remaining is not None else None,
            "capped_by_maximum": estimate < covered,
        }

    # ----- dispatch --------------------------------------------------------------------------

    def call(self, name: str, arguments: dict[str, Any]) -> tuple[dict[str, Any], float]:
        """Runs a tool by name. Bad arguments come back as {"error": ...} so the model can retry."""
        handlers = {
            "search_policy": self.search_policy,
            "get_member": self.get_member,
            "get_benefit_usage": self.get_benefit_usage,
            "estimate_reimbursement": self.estimate_reimbursement,
        }
        started = time.perf_counter()
        handler = handlers.get(name)
        try:
            if handler is None:
                raise ToolError(f"unknown tool {name!r}")
            try:
                inspect.signature(handler).bind(**arguments)
            except TypeError as error:
                raise ToolError(f"bad arguments for {name}: {error}") from error
            result = handler(**arguments)
        except ValueError as error:  # ToolError, or a malformed date or number
            result = {"error": str(error)}
        return result, round((time.perf_counter() - started) * 1000, 2)

    def schemas(self) -> list[dict[str, Any]]:
        benefit = {"type": "string", "enum": sorted(self.benefits), "description": "Benefit key"}
        member = {"type": "string", "description": "Member ID such as M-1001"}

        def tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
            return {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                        "additionalProperties": False,
                    },
                },
            }

        return [
            tool(
                "search_policy",
                "Search the benefits booklet. Returns numbered sections to quote and cite.",
                {"query": {"type": "string", "description": "What to look up"}},
                ["query"],
            ),
            tool(
                "get_member",
                "Look up a member's first name, plan and enrolment date.",
                {"member_id": member},
                ["member_id"],
            ),
            tool(
                "get_benefit_usage",
                "Amount used and remaining for one benefit, plus any waiting period.",
                {"member_id": member, "benefit": benefit},
                ["member_id", "benefit"],
            ),
            tool(
                "estimate_reimbursement",
                "Estimate what the plan would pay for a charge, after coinsurance and maximums.",
                {
                    "member_id": member,
                    "benefit": benefit,
                    "amount": {"type": "number", "description": "Amount charged, CAD"},
                },
                ["member_id", "benefit", "amount"],
            ),
        ]
