from __future__ import annotations

from trustshield.ingestion import generate_synthetic_events
from trustshield.preprocessing import normalize_text, validate_events


def main() -> None:
    df = generate_synthetic_events(n_samples=1000, random_state=42)
    df["message_text"] = df["message_text"].map(normalize_text)
    validate_events(df)
    print("Validation passed: synthetic training frame is schema-compliant.")


if __name__ == "__main__":
    main()
