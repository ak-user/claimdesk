"""Checks that run before and around the model: PII masking, prompt injection, scope."""

from __future__ import annotations

import re

_INJECTION = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(?:ignore|forget)\s+(?:all\s+|any\s+)?(?:previous|prior|above|earlier)\s+instructions\b",
        r"\bdisregard\s+(?:all\s+|any\s+)?(?:previous|prior|the|your)\s+(?:instructions|rules)\b",
        r"\bsystem\s+prompt\b",
        r"\byou\s+are\s+now\b",
        r"\bapprove\s+(?:this|the)\s+claim\b",
        r"\bpay\s+(?:this|the)\s+claim\s+in\s+full\b",
        r"\bnote\s+to\s+(?:the\s+)?ai\b",
    )
]

# Narrow on purpose: "What medication is covered?" is a coverage question, not medical advice.
_MEDICAL = re.compile(
    r"\b(?:should\s+i\s+take|is\s+it\s+safe\s+to\s+take|what\s+(?:dose|dosage)|dosage"
    r"|how\s+many\s+(?:pills|tablets|mg)|diagnos\w*|side\s+effects?"
    r"|what(?:'s|\s+is)\s+wrong\s+with\s+my)\b",
    re.IGNORECASE,
)

# Requests for a decision. "My claim was denied, how do I appeal?" must not match.
_DECISION = re.compile(
    r"\b(?:can|could|will|would|please)\s+(?:you\s+)?(?:approve|deny|reject|guarantee|pay)\b"
    r".{0,30}\bclaims?\b"
    r"|\bwill\s+(?:my|this|the)\s+claim\s+be\s+(?:approved|paid|denied|accepted)\b"
    r"|\bapprove\s+(?:my|this|the)\s+claim\b",
    re.IGNORECASE,
)

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE = re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)")
_SIN = re.compile(r"(?<!\d)\d{3}[\s-]?\d{3}[\s-]?\d{3}(?!\d)")
_MEMBER = re.compile(r"\bM-(\d{2})\d{2}\b", re.IGNORECASE)

MEDICAL_REFUSAL = (
    "I can't give medical advice. For questions about symptoms, medication or treatment, talk to "
    "your doctor or pharmacist. I can tell you what the plan covers, for example prescription "
    "drugs or physiotherapy."
)


def detect_injection(text: str) -> list[str]:
    """Returns the suspicious phrases found in untrusted text, such as an uploaded claim."""
    return [m.group(0) for pattern in _INJECTION for m in pattern.finditer(text)]


def is_medical_advice_request(question: str) -> bool:
    return bool(_MEDICAL.search(question))


def is_claim_decision_request(question: str) -> bool:
    return bool(_DECISION.search(question))


def mask_pii(text: str) -> str:
    """Masks contact details, SIN-like numbers and most of a member ID before logging."""
    text = _EMAIL.sub("[email]", text)
    text = _PHONE.sub("[phone]", text)
    text = _SIN.sub("[number]", text)
    return _MEMBER.sub(lambda m: f"M-{m.group(1)}**", text)
