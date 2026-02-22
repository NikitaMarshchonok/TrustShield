from __future__ import annotations

import pandas as pd

try:
    import pandera as pa
    from pandera import Check
except Exception:
    pa = None
    Check = None


REQUIRED_COLUMNS = [
    "event_id",
    "message_text",
    "country",
    "user_id",
    "device_id",
    "ip_id",
    "card_id",
    "merchant_id",
    "payment_attempts",
    "account_age_days",
    "device_reuse_count",
    "chargeback_history",
    "is_fraud",
]


def _fallback_validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if (df["payment_attempts"] < 0).any() or (df["account_age_days"] < 0).any():
        raise ValueError("Negative values found in non-negative feature columns.")
    if not set(df["chargeback_history"].unique()).issubset({0, 1}):
        raise ValueError("chargeback_history must be binary.")
    if not set(df["is_fraud"].unique()).issubset({0, 1}):
        raise ValueError("is_fraud must be binary.")
    return df


if pa is not None:
    EVENT_SCHEMA = pa.DataFrameSchema(
        {
            "event_id": pa.Column(int, Check.ge(0)),
            "message_text": pa.Column(str, Check.str_length(min_value=1)),
            "country": pa.Column(str, Check.str_length(min_value=2, max_value=2)),
            "user_id": pa.Column(str, Check.str_length(min_value=1)),
            "device_id": pa.Column(str, Check.str_length(min_value=1)),
            "ip_id": pa.Column(str, Check.str_length(min_value=1)),
            "card_id": pa.Column(str, Check.str_length(min_value=1)),
            "merchant_id": pa.Column(str, Check.str_length(min_value=1)),
            "payment_attempts": pa.Column(int, Check.ge(0)),
            "account_age_days": pa.Column(int, Check.ge(0)),
            "device_reuse_count": pa.Column(int, Check.ge(0)),
            "chargeback_history": pa.Column(int, Check.isin([0, 1])),
            "is_fraud": pa.Column(int, Check.isin([0, 1])),
        },
        strict=True,
    )
else:
    EVENT_SCHEMA = None


def validate_events(df: pd.DataFrame) -> pd.DataFrame:
    if EVENT_SCHEMA is None:
        return _fallback_validate(df)
    return EVENT_SCHEMA.validate(df, lazy=True)
