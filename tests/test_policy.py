from trustshield.serving.policy import decide, init_policy_state, load_policy


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
    decision, _, triggers = decide(0.9, payload, policy)
    assert decision == "block"
    assert "score:block_threshold" in triggers


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
    decision, reasons, triggers = decide(0.1, payload, policy)
    assert decision == "block"
    assert "hard_rule:max_payment_attempts" in reasons
    assert "hard_rule:max_payment_attempts" in triggers


def test_policy_rate_limit_review_then_block() -> None:
    policy = load_policy()
    state = init_policy_state()
    payload = {
        "message_text": "normal text",
        "country": "US",
        "user_id": "u-rate",
        "device_id": "d-rate",
        "ip_id": "ip-rate",
        "payment_attempts": 1,
        "account_age_days": 30,
        "device_reuse_count": 1,
        "chargeback_history": 0,
    }

    review_seen = False
    block_seen = False
    for i in range(8):
        payload["event_ts"] = 1_700_000_000 + i
        decision, _, triggers = decide(0.05, payload, policy, state=state)
        if "rate_limit:user:review" in triggers:
            review_seen = True
        if "rate_limit:user:block" in triggers:
            block_seen = True
            assert decision == "block"

    assert review_seen
    assert block_seen
