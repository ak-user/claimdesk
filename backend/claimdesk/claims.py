"""Claim intake: extract fields, check them against the plan rules, estimate the payment.

ClaimDesk never approves or denies a claim. It prepares the file for a claims examiner: what was
found, what is missing, which rules apply, and what the plan would likely pay.
"""

from __future__ import annotations

import logging
from datetime import date

from .agent import money
from .extraction import extract_with_model, extract_with_rules, merge
from .guardrails import detect_injection
from .ingest import snippet
from .llm import ChatClient
from .models import Citation, ClaimAssessment, ClaimFields, Estimate, Issue
from .retrieval import Retriever
from .tools import Toolbox, add_months

log = logging.getLogger(__name__)

FIELD_NAMES = {
    "member_id": "Member ID",
    "patient_name": "Patient name",
    "service_type": "Service type",
    "service_date": "Service date",
    "provider_name": "Provider",
    "amount": "Amount charged",
}


class ClaimAssessor:
    def __init__(
        self, toolbox: Toolbox, retriever: Retriever, client: ChatClient | None = None
    ) -> None:
        self.toolbox = toolbox
        self.retriever = retriever
        self.client = client

    def assess(self, text: str) -> ClaimAssessment:
        flags: list[str] = []
        fields, notes = extract_with_rules(text)
        extractor = "rules"
        if self.client is not None:
            try:
                fields = merge(extract_with_model(text, self.client), fields)
                extractor = "llm"
            except Exception:  # noqa: BLE001
                log.warning("Model extraction failed; using the rule-based fields.", exc_info=True)
                flags.append("model_extraction_failed_used_rules")

        issues = self._check(fields, notes)
        benefit = (
            self.toolbox.classify_benefit(fields.service_type) if fields.service_type else None
        )
        rule = self.toolbox.benefits.get(benefit) if benefit else None

        injected = detect_injection(text)
        if injected:
            flags.append("possible_prompt_injection")
            issues.append(
                Issue(
                    field=None,
                    severity="warning",
                    kind="note",
                    message=(
                        "The document contains instructions aimed at an AI system "
                        f'("{injected[0]}"). They were ignored and the claim is flagged for the '
                        "examiner."
                    ),
                )
            )
        if benefit and benefit.startswith("dental"):
            issues.append(
                Issue(
                    field="amount",
                    severity="info",
                    kind="note",
                    message="Dental amounts are paid up to the provincial dental fee guide.",
                    section="5.4",
                )
            )

        if any(i.kind == "not_payable" for i in issues):
            status = "not_payable"
        elif any(i.kind == "needs_correction" for i in issues):
            status = "needs_correction"
        else:
            status = "ready_for_review"

        estimate = None
        if status != "not_payable" and benefit and fields.member_id and (fields.amount or 0) > 0:
            estimate = self._estimate(fields, benefit)

        review_section = self.toolbox.rules["human_review_section"]
        sections = [i.section for i in issues if i.section]
        sections += [rule["section"]] if rule else []
        sections.append(review_section)
        citations = []
        for section in dict.fromkeys(sections):
            chunk = self.retriever.get(section)
            if chunk:
                citations.append(
                    Citation(section=section, title=chunk.title, snippet=snippet(chunk.text))
                )

        return ClaimAssessment(
            status=status,
            summary=_summary(status, issues, estimate),
            fields=fields,
            extractor=extractor,
            benefit=benefit,
            benefit_display=rule["display"] if rule else None,
            issues=issues,
            estimate=estimate,
            citations=citations,
            flags=flags,
        )

    def _check(self, fields: ClaimFields, notes: list[str]) -> list[Issue]:
        rules = self.toolbox.rules
        required_section = rules["required_info_section"]
        as_of = self.toolbox.as_of
        issues: list[Issue] = []

        for name, label in FIELD_NAMES.items():
            if getattr(fields, name) in (None, ""):
                message = f"{label} is missing."
                if name == "service_date" and "ambiguous_date" in notes:
                    message = "The service date is ambiguous. Write it as YYYY-MM-DD."
                issues.append(
                    Issue(
                        field=name,
                        severity="error",
                        kind="needs_correction",
                        message=message,
                        section=required_section,
                    )
                )

        member = None
        if fields.member_id:
            result, _ = self.toolbox.call("get_member", {"member_id": fields.member_id})
            if "error" in result:
                issues.append(
                    Issue(
                        field="member_id",
                        severity="error",
                        kind="not_payable",
                        message=(
                            f"Member ID {fields.member_id} isn't on plan {rules['plan_number']}."
                        ),
                    )
                )
            else:
                member = result

        benefit = (
            self.toolbox.classify_benefit(fields.service_type) if fields.service_type else None
        )
        rule = self.toolbox.benefits.get(benefit) if benefit else None
        if fields.service_type and rule is None:
            issues.append(
                Issue(
                    field="service_type",
                    severity="error",
                    kind="not_payable",
                    message=f'"{fields.service_type}" isn\'t a covered benefit in this plan.',
                )
            )
        if rule and rule["requires_registration"] and not fields.provider_registration:
            issues.append(
                Issue(
                    field="provider_registration",
                    severity="error",
                    kind="needs_correction",
                    message="Paramedical claims need the practitioner's registration number.",
                    section=required_section,
                )
            )

        if fields.service_date:
            if fields.service_date > as_of:
                issues.append(
                    Issue(
                        field="service_date",
                        severity="error",
                        kind="needs_correction",
                        message="The service date is in the future.",
                    )
                )
            else:
                deadline = add_months(fields.service_date, rules["claim_deadline_months"])
                if deadline < as_of:
                    issues.append(
                        Issue(
                            field="service_date",
                            severity="error",
                            kind="not_payable",
                            message=(
                                f"Claims must be submitted within {rules['claim_deadline_months']} "
                                f"months of the service date. The deadline for this claim was "
                                f"{deadline.isoformat()}."
                            ),
                            section=rules["claim_deadline_section"],
                        )
                    )

        if fields.amount is not None and fields.amount <= 0:
            issues.append(
                Issue(
                    field="amount",
                    severity="error",
                    kind="needs_correction",
                    message="The amount must be greater than $0.",
                )
            )

        if member and rule and rule.get("waiting_months") and fields.service_date:
            covered_from = add_months(
                date.fromisoformat(member["enrolled"]), rule["waiting_months"]
            )
            if fields.service_date < covered_from:
                issues.append(
                    Issue(
                        field="service_type",
                        severity="error",
                        kind="not_payable",
                        message=(
                            f"{rule['display']} is covered after {rule['waiting_months']} months "
                            f"in the plan. This member is covered from {covered_from.isoformat()}."
                        ),
                        section=rule["section"],
                    )
                )
        return issues

    def _estimate(self, fields: ClaimFields, benefit: str) -> Estimate | None:
        result, _ = self.toolbox.call(
            "estimate_reimbursement",
            {"member_id": fields.member_id, "benefit": benefit, "amount": fields.amount},
        )
        if "error" in result or not result.get("eligible"):
            return None
        return Estimate(
            benefit=benefit,
            benefit_display=result["display"],
            coinsurance_percent=result["coinsurance_percent"],
            covered_share=result["covered_share"],
            estimate=result["estimate"],
            remaining_before=result["remaining_before"],
            remaining_after=result["remaining_after"],
            capped_by_maximum=result["capped_by_maximum"],
        )


def _summary(status: str, issues: list[Issue], estimate: Estimate | None) -> str:
    if status == "not_payable":
        reason = next(i.message for i in issues if i.kind == "not_payable")
        return f"Not payable. {reason}"
    if status == "needs_correction":
        reason = next(i.message for i in issues if i.kind == "needs_correction")
        tail = (
            f" Estimated reimbursement once corrected: {money(estimate.estimate)}."
            if estimate
            else ""
        )
        return f"Needs correction before review. {reason}{tail}"
    tail = f" Estimated reimbursement: {money(estimate.estimate)}." if estimate else ""
    return f"Ready for examiner review.{tail}"
