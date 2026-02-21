from __future__ import annotations

import numpy as np


def cost_saved_metric(
    y_true: np.ndarray,
    y_pred_block: np.ndarray,
    fraud_cost: float = 100.0,
    review_cost: float = 2.0,
) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred_block = np.asarray(y_pred_block).astype(int)
    prevented_fraud = ((y_true == 1) & (y_pred_block == 1)).sum() * fraud_cost
    false_positive_ops = ((y_true == 0) & (y_pred_block == 1)).sum() * review_cost
    return float(prevented_fraud - false_positive_ops)
