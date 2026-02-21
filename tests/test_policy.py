from trustshield.serving.policy import decide, load_policy


def test_policy_blocks_high_score() -> None:
    policy = load_policy()
    payload = {
        "message_text": "normal text",
        "country": "US",
        "payment_attempts": 1,
        "account_age_days": 20,
        "device_reuse_count": 1,
        "chargeback_history": 0,
    }
    decision, _ = decide(0.9, payload, policy)
    assert decision == "block"


def test_policy_hard_rule_payment_attempts() -> None:
    policy = load_policy()
    payload = {
        "message_text": "normal text",
        "country": "US",
        "payment_attempts": 9,
        "account_age_days": 20,
        "device_reuse_count": 1,
        "chargeback_history": 0,
    }
    decision, reasons = decide(0.1, payload, policy)
    assert decision == "block"
    assert "hard_rule:max_payment_attempts" in reasons
