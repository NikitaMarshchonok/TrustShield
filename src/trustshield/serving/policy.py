from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from trustshield.features import extract_reason_flags


def load_policy(config_path: str = "configs/policy.yaml") -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def decide(score: float, payload: dict[str, Any], policy: dict[str, Any]) -> tuple[str, list[str]]:
    reasons = extract_reason_flags(payload)
    thresholds = policy["score_thresholds"]
    hard_rules = policy["hard_rules"]

    if int(payload.get("payment_attempts", 0)) >= int(hard_rules["max_payment_attempts"]):
        reasons.append("hard_rule:max_payment_attempts")
        return "block", sorted(set(reasons))

    if int(payload.get("account_age_days", 0)) <= int(hard_rules["min_account_age_days"]):
        reasons.append("hard_rule:min_account_age_days")
        return "review", sorted(set(reasons))

    if score >= float(thresholds["block"]):
        return "block", sorted(set(reasons))
    if score >= float(thresholds["review"]):
        return "review", sorted(set(reasons))
    return "allow", sorted(set(reasons))
