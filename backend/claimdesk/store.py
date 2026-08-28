"""Loads the demo data: booklet, plan rules, members and sample claims."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from .config import DATA_DIR

SAMPLES_DIR = DATA_DIR / "sample_claims"


def load_booklet(path: Path = DATA_DIR / "policy" / "benefits_booklet.md") -> str:
    return path.read_text(encoding="utf-8")


def load_rules(path: Path = DATA_DIR / "plan_rules.json") -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_members(path: Path = DATA_DIR / "members.json") -> dict[str, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    members: dict[str, dict[str, Any]] = {}
    for member in raw["members"]:
        member = dict(member)
        member["enrolled"] = date.fromisoformat(member["enrolled"])
        member["claims"] = [
            {**claim, "service_date": date.fromisoformat(claim["service_date"])}
            for claim in member["claims"]
        ]
        members[member["member_id"]] = member
    return members


def list_samples() -> list[dict[str, str]]:
    samples = []
    for path in sorted(SAMPLES_DIR.glob("*.txt")):
        first_line = path.read_text(encoding="utf-8").splitlines()[0]
        samples.append({"name": path.stem, "title": first_line.split(" - ")[-1].strip()})
    return samples


def read_sample(name: str) -> str | None:
    if not name.replace("_", "").replace("-", "").isalnum():
        return None
    path = SAMPLES_DIR / f"{name}.txt"
    return path.read_text(encoding="utf-8") if path.is_file() else None
