from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd


SAFE_MESSAGES = [
    "Hi, I want to buy this item.",
    "Can you confirm delivery date?",
    "Thanks for the quick response.",
    "Please share pickup details.",
    "Payment completed through marketplace checkout.",
]

RISK_MESSAGES = [
    "Pay me directly and I will ship now.",
    "Urgent transfer needed, send card details.",
    "Click this link to verify your account.",
    "Your account is blocked, share OTP immediately.",
    "I can offer discount outside platform.",
]

COUNTRIES = ["US", "DE", "PL", "UA", "GB", "NG", "RU", "BR"]
HIGH_RISK_COUNTRIES = {"NG", "RU"}


@dataclass
class SyntheticConfig:
    n_samples: int = 3000
    random_state: int = 42


def _sample_message(risky: bool, rnd: random.Random) -> str:
    pool = RISK_MESSAGES if risky else SAFE_MESSAGES
    return rnd.choice(pool)


def generate_synthetic_events(n_samples: int = 3000, random_state: int = 42) -> pd.DataFrame:
    rnd = random.Random(random_state)
    np.random.seed(random_state)

    rows = []
    for idx in range(n_samples):
        latent_risk = np.clip(np.random.beta(2.0, 5.0) + np.random.normal(0.0, 0.08), 0, 1)
        risky_message = bool(latent_risk > 0.5 or np.random.rand() < 0.12)
        country = rnd.choice(COUNTRIES)
        payment_attempts = int(np.random.poisson(2.0 + latent_risk * 3.0))
        account_age_days = int(max(1, np.random.gamma(4.5, 25.0) * (1.1 - latent_risk)))
        device_reuse_count = int(np.random.poisson(1.5 + latent_risk * 4.0))
        chargeback_history = int(np.random.rand() < (0.06 + 0.4 * latent_risk))
        message = _sample_message(risky_message, rnd)

        risk_boost = 0.0
        risk_boost += 0.16 if country in HIGH_RISK_COUNTRIES else 0.0
        risk_boost += 0.08 if payment_attempts >= 4 else 0.0
        risk_boost += 0.12 if account_age_days < 7 else 0.0
        risk_boost += 0.1 if device_reuse_count >= 4 else 0.0
        risk_boost += 0.18 if risky_message else 0.0
        risk_boost += 0.22 if chargeback_history else 0.0

        fraud_prob = float(np.clip(latent_risk * 0.6 + risk_boost, 0, 1))
        is_fraud = int(np.random.rand() < fraud_prob)

        rows.append(
            {
                "event_id": idx,
                "message_text": message,
                "country": country,
                "payment_attempts": payment_attempts,
                "account_age_days": account_age_days,
                "device_reuse_count": device_reuse_count,
                "chargeback_history": chargeback_history,
                "is_fraud": is_fraud,
            }
        )

    return pd.DataFrame(rows)
