"""Validación y evaluación reproducible de escenarios BMR hipotéticos."""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_SCORE_COLUMNS = {
    "TransactionID",
    "TransactionAmt",
    "isFraud",
    "probability_calibrated",
    "action_fixed_0_5",
}


def validate_validation_scores(scores: pd.DataFrame) -> tuple[np.ndarray, ...]:
    """Valida scores de validation y devuelve amount, y, p, acción y mediana."""
    missing = REQUIRED_SCORE_COLUMNS.difference(scores.columns)
    if missing:
        raise ValueError(f"Faltan columnas en validation_scores: {sorted(missing)}")
    if scores.empty:
        raise ValueError("validation_scores no puede estar vacío")
    if scores[list(REQUIRED_SCORE_COLUMNS)].isna().any().any():
        raise ValueError("Hay nulos en las columnas requeridas")
    if scores["TransactionID"].duplicated().any():
        raise ValueError("TransactionID debe ser único en validation_scores")

    amount = scores["TransactionAmt"].to_numpy(dtype=float)
    y_raw = scores["isFraud"].to_numpy()
    p = scores["probability_calibrated"].to_numpy(dtype=float)
    fixed = scores["action_fixed_0_5"].astype(str).to_numpy()
    if not np.isfinite(amount).all() or (amount < 0).any():
        raise ValueError("TransactionAmt debe ser finito y no negativo")
    if not np.isin(y_raw, (0, 1)).all():
        raise ValueError("isFraud debe ser binario")
    y = y_raw.astype(int)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Las probabilidades deben ser finitas y estar entre 0 y 1")
    if not np.isin(fixed, ("approve", "block")).all():
        raise ValueError("action_fixed_0_5 debe contener solo approve/block")
    expected_fixed = np.where(p >= 0.5, "block", "approve")
    if not np.array_equal(fixed, expected_fixed):
        raise ValueError("action_fixed_0_5 no coincide con probability_calibrated >= 0.5")
    if not np.any(y == 1) or not np.any(y == 0):
        raise ValueError("validation_scores debe contener ambas clases")

    reference_amount = float(np.median(amount))
    if not np.isfinite(reference_amount) or reference_amount <= 0:
        raise ValueError("La mediana de TransactionAmt debe ser finita y positiva")
    return amount, y, p, fixed, reference_amount


def evaluate_hypothetical_scenarios(
    scores: pd.DataFrame,
    *,
    lambda_grid: tuple[float, ...] = (0.25, 0.50, 1.00),
    b_ratio_grid: tuple[float, ...] = (0.25, 0.50, 1.00, 2.00),
    fraud_block_residual_cost: float = 0.0,
) -> pd.DataFrame:
    """Calcula sensibilidad BMR binaria; los parámetros no son costos aprobados.

    ``B`` se referencia a la mediana de TransactionAmt en el propio bloque
    validation_policy. Por tanto, el resultado es descriptivo y retrospectivo.
    En empates de riesgo aprueba, de acuerdo con el protocolo del proyecto.
    """
    amount, y, p, fixed, reference_amount = validate_validation_scores(scores)
    if not lambda_grid or any(
        not np.isfinite(value) or not 0 <= value <= 1 for value in lambda_grid
    ):
        raise ValueError("lambda_grid debe contener valores finitos entre 0 y 1")
    if not b_ratio_grid or any(
        not np.isfinite(value) or value <= 0 for value in b_ratio_grid
    ):
        raise ValueError("b_ratio_grid debe contener valores finitos positivos")
    if (
        not np.isfinite(fraud_block_residual_cost)
        or fraud_block_residual_cost < 0
    ):
        raise ValueError("fraud_block_residual_cost debe ser finito y no negativo")

    fraud = y == 1
    legitimate = ~fraud
    rows: list[dict[str, object]] = []
    for loss_fraction in lambda_grid:
        for b_ratio in b_ratio_grid:
            legitimate_block_cost = float(b_ratio * reference_amount)
            risk_approve = p * loss_fraction * amount
            risk_block = (
                (1 - p) * legitimate_block_cost
                + p * fraud_block_residual_cost
            )
            bmr = np.where(risk_approve <= risk_block, "approve", "block")
            fixed_cost = 0.0
            scenario_costs = []
            for strategy, action in (("fixed_0_5", fixed), ("bmr_binary", bmr)):
                block = action == "block"
                cost = np.zeros(len(y), dtype=float)
                cost[fraud & ~block] = loss_fraction * amount[fraud & ~block]
                cost[fraud & block] = fraud_block_residual_cost
                cost[legitimate & block] = legitimate_block_cost
                scenario_costs.append((strategy, action, cost))
                if strategy == "fixed_0_5":
                    fixed_cost = float(cost.sum())

            for strategy, action, cost in scenario_costs:
                block = action == "block"
                savings = fixed_cost - float(cost.sum())
                rows.append({
                    "scenario_status": "HIPOTETICO_NO_APROBADO",
                    "strategy": strategy,
                    "rows": len(y),
                    "frauds": int(fraud.sum()),
                    "loss_fraction_lambda": float(loss_fraction),
                    "legitimate_block_cost_ratio_of_validation_median_amount": float(b_ratio),
                    "validation_median_amount_reference": reference_amount,
                    "legitimate_block_cost_B_proxy": legitimate_block_cost,
                    "fraud_block_residual_cost_C_proxy": float(fraud_block_residual_cost),
                    "review_enabled": False,
                    "cost_total_proxy": float(cost.sum()),
                    "cost_per_transaction_proxy": float(cost.mean()),
                    "fixed_baseline_cost_proxy": fixed_cost,
                    "savings_vs_fixed_proxy": savings,
                    "savings_fraction_vs_fixed_proxy": savings / fixed_cost if fixed_cost > 0 else np.nan,
                    "frauds_blocked": int(np.sum(fraud & block)),
                    "frauds_missed": int(np.sum(fraud & ~block)),
                    "fraud_recall": float(np.sum(fraud & block) / fraud.sum()),
                    "legitimate_block_count": int(np.sum(legitimate & block)),
                    "legitimate_block_rate": float(np.mean(block[legitimate])),
                    "policy_block_count": int(block.sum()),
                })
    return pd.DataFrame(rows)
