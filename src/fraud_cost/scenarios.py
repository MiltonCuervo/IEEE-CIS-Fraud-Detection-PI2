"""Sensibilidad de Ca sobre probabilidades guardadas de validación."""
import numpy as np
import pandas as pd
from fraud_cost.costs import bmr_actions, evaluate_policies, validate_labels, validate_ca

HYPOTHETICAL_CA_GRID = (1.0, 5.0, 10.0, 20.0, 50.0, 100.0)
REQUIRED_SCORE_COLUMNS = {"TransactionID", "TransactionAmt", "isFraud", "probability_calibrated", "action_fixed_0_5"}


def validate_validation_scores(scores):
    missing = REQUIRED_SCORE_COLUMNS.difference(scores.columns)
    if missing or scores.empty:
        raise ValueError(f"Scores vacíos o columnas faltantes: {sorted(missing)}")
    if scores[list(REQUIRED_SCORE_COLUMNS)].isna().any().any() or scores.TransactionID.duplicated().any():
        raise ValueError("Los scores deben tener IDs únicos y campos requeridos sin nulos")
    y, amount = validate_labels(scores.isFraud, scores.TransactionAmt)
    p = scores.probability_calibrated.to_numpy(dtype=float)
    # Valida rango y dimensiones utilizando la misma regla económica.
    bmr_actions(p, amount, 0)
    fixed = scores.action_fixed_0_5.to_numpy()
    if not np.array_equal(fixed, np.where(p >= 0.5, "block", "approve")):
        raise ValueError("La política fija debe corresponder al corte 0,5")
    if np.unique(y).size != 2:
        raise ValueError("La validación debe contener ambas clases")
    return amount, y, p, fixed


def evaluate_hypothetical_scenarios(scores, *, ca_grid=HYPOTHETICAL_CA_GRID):
    """Ca absoluto en unidades del dataset; grilla ilustrativa, no observada."""
    amount, y, p, fixed = validate_validation_scores(scores)
    values = tuple(validate_ca(c) for c in ca_grid)
    if not values or len(set(values)) != len(values):
        raise ValueError("La grilla de Ca debe ser no vacía y sin duplicados")
    results = []
    for ca in values:
        policies = {"fixed_0_5": fixed, "bmr": bmr_actions(p, amount, ca)}
        table = evaluate_policies(y, amount, policies, ca)
        table["scenario_status"] = "HIPOTETICO_CA_NO_ESTIMACION_OPERATIVA"
        table["evaluation_block"] = "validation_policy"
        results.append(table)
    return pd.concat(results, ignore_index=True)
