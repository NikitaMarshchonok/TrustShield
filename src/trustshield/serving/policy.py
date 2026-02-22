from __future__ import annotations

import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import yaml

from trustshield.features import extract_reason_flags


def load_policy(config_path: str = "configs/policy.yaml") -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class PolicyState:
    def __init__(self) -> None:
        self.user_events: dict[str, deque[float]] = defaultdict(deque)
        self.device_events: dict[str, deque[float]] = defaultdict(deque)
        self.ip_events: dict[str, deque[float]] = defaultdict(deque)


def init_policy_state() -> PolicyState:
    return PolicyState()


def reset_policy_state(state: PolicyState) -> None:
    state.user_events.clear()
    state.device_events.clear()
    state.ip_events.clear()


def _push_event(queue: deque[float], event_ts: float, window_seconds: int) -> int:
    queue.append(event_ts)
    min_allowed = event_ts - float(window_seconds)
    while queue and queue[0] < min_allowed:
        queue.popleft()
    return len(queue)


def _rate_limit_triggers(payload: dict[str, Any], policy: dict[str, Any], state: PolicyState) -> list[str]:
    cfg = policy.get("rate_limits", {})
    if not cfg:
        return []

    raw_event_ts = payload.get("event_ts")
    event_ts = float(raw_event_ts) if raw_event_ts is not None else float(time.time())
    window_seconds = int(cfg["window_seconds"])

    user_id = str(payload.get("user_id", "unknown_user"))
    device_id = str(payload.get("device_id", "unknown_device"))
    ip_id = str(payload.get("ip_id", "unknown_ip"))

    user_count = _push_event(state.user_events[user_id], event_ts, window_seconds)
    device_count = _push_event(state.device_events[device_id], event_ts, window_seconds)
    ip_count = _push_event(state.ip_events[ip_id], event_ts, window_seconds)

    triggers: list[str] = []
    if user_count >= int(cfg["user_block_events"]):
        triggers.append("rate_limit:user:block")
    elif user_count >= int(cfg["user_review_events"]):
        triggers.append("rate_limit:user:review")

    if device_count >= int(cfg["device_block_events"]):
        triggers.append("rate_limit:device:block")
    elif device_count >= int(cfg["device_review_events"]):
        triggers.append("rate_limit:device:review")

    if ip_count >= int(cfg["ip_block_events"]):
        triggers.append("rate_limit:ip:block")
    elif ip_count >= int(cfg["ip_review_events"]):
        triggers.append("rate_limit:ip:review")

    return sorted(set(triggers))


def decide(
    score: float, payload: dict[str, Any], policy: dict[str, Any], state: PolicyState | None = None
) -> tuple[str, list[str], list[str]]:
    reasons = extract_reason_flags(payload)
    thresholds = policy["score_thresholds"]
    hard_rules = policy["hard_rules"]
    policy_triggers: list[str] = []
    if state is not None:
        policy_triggers.extend(_rate_limit_triggers(payload, policy, state))

    if int(payload.get("payment_attempts", 0)) >= int(hard_rules["max_payment_attempts"]):
        reasons.append("hard_rule:max_payment_attempts")
        policy_triggers.append("hard_rule:max_payment_attempts")
        return "block", sorted(set(reasons)), sorted(set(policy_triggers))

    if int(payload.get("account_age_days", 0)) <= int(hard_rules["min_account_age_days"]):
        reasons.append("hard_rule:min_account_age_days")
        policy_triggers.append("hard_rule:min_account_age_days")
        if any("block" in trigger for trigger in policy_triggers):
            return "block", sorted(set(reasons)), sorted(set(policy_triggers))
        return "review", sorted(set(reasons)), sorted(set(policy_triggers))

    if any("block" in trigger for trigger in policy_triggers):
        return "block", sorted(set(reasons)), sorted(set(policy_triggers))
    if any("review" in trigger for trigger in policy_triggers):
        return "review", sorted(set(reasons)), sorted(set(policy_triggers))

    if score >= float(thresholds["block"]):
        policy_triggers.append("score:block_threshold")
        return "block", sorted(set(reasons)), sorted(set(policy_triggers))
    if score >= float(thresholds["review"]):
        policy_triggers.append("score:review_threshold")
        return "review", sorted(set(reasons)), sorted(set(policy_triggers))
    return "allow", sorted(set(reasons)), sorted(set(policy_triggers))
