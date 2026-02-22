import pandas as pd
import pytest

from trustshield.preprocessing.validation import validate_events


def test_validate_events_passes_for_valid_frame() -> None:
    df = pd.DataFrame(
        [
            {
                "event_id": 1,
                "message_text": "hello",
                "country": "US",
                "user_id": "u1",
                "device_id": "d1",
                "ip_id": "i1",
                "card_id": "c1",
                "merchant_id": "m1",
                "payment_attempts": 1,
                "account_age_days": 10,
                "device_reuse_count": 1,
                "chargeback_history": 0,
                "is_fraud": 0,
            }
        ]
    )
    out = validate_events(df)
    assert len(out) == 1


def test_validate_events_raises_on_invalid_binary_values() -> None:
    df = pd.DataFrame(
        [
            {
                "event_id": 1,
                "message_text": "hello",
                "country": "US",
                "user_id": "u1",
                "device_id": "d1",
                "ip_id": "i1",
                "card_id": "c1",
                "merchant_id": "m1",
                "payment_attempts": 1,
                "account_age_days": 10,
                "device_reuse_count": 1,
                "chargeback_history": 2,
                "is_fraud": 0,
            }
        ]
    )
    with pytest.raises(Exception):
        validate_events(df)
