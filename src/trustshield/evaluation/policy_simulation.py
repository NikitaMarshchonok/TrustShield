from __future__ import annotations

import json
from pathlib import Path

from trustshield.ingestion import generate_synthetic_events
from trustshield.preprocessing import normalize_text
from trustshield.serving.policy import decide, init_policy_state, load_policy


def _heuristic_score(payload: dict) -> float:
    score = 0.05
    text = str(payload.get("message_text", "")).lower()
    if any(token in text for token in ("urgent", "otp", "click", "outside platform", "transfer")):
        score += 0.35
    score += min(float(payload.get("payment_attempts", 0)) * 0.06, 0.25)
    score += 0.2 if float(payload.get("account_age_days", 0)) < 7 else 0.0
    score += 0.18 if int(payload.get("chargeback_history", 0)) == 1 else 0.0
    score += 0.14 if float(payload.get("device_reuse_count", 0)) >= 4 else 0.0
    return float(min(score, 0.99))


def run_policy_simulation(n_events: int = 1200) -> dict:
    policy = load_policy()
    state = init_policy_state()
    df = generate_synthetic_events(n_samples=n_events, random_state=77)
    df["message_text"] = df["message_text"].map(normalize_text)

    decisions = {"allow": 0, "review": 0, "block": 0}
    trigger_counts: dict[str, int] = {}
    review_true_fraud = 0
    block_true_fraud = 0

    ts = 1_700_000_000
    for row in df.to_dict(orient="records"):
        row["event_ts"] = ts
        ts += 5
        score = _heuristic_score(row)
        decision, _, triggers = decide(score, row, policy, state=state)
        decisions[decision] += 1
        for trigger in triggers:
            trigger_counts[trigger] = trigger_counts.get(trigger, 0) + 1

        if int(row["is_fraud"]) == 1 and decision == "review":
            review_true_fraud += 1
        if int(row["is_fraud"]) == 1 and decision == "block":
            block_true_fraud += 1

    report = {
        "n_events": n_events,
        "decisions": decisions,
        "review_precision_proxy": round(review_true_fraud / max(decisions["review"], 1), 4),
        "block_precision_proxy": round(block_true_fraud / max(decisions["block"], 1), 4),
        "top_policy_triggers": sorted(trigger_counts.items(), key=lambda x: x[1], reverse=True)[:10],
    }

    out_path = Path("reports/policy_simulation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Policy simulation report saved to {out_path}")
    return report


if __name__ == "__main__":
    run_policy_simulation()
