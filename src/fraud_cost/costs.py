"""Matriz del anteproyecto: intervenir cuesta Ca en ambas clases."""
from __future__ import annotations

import numpy as np
import pandas as pd


def validate_ca(ca):
    value = float(ca)
    if not np.isfinite(value) or value < 0:
        raise ValueError("Ca debe ser finito y no negativo")
    return value


def validate_amount(amount):
    a = np.asarray(amount, dtype=float)
    if a.ndim != 1 or not np.isfinite(a).all() or (a < 0).any():
        raise ValueError("Los montos deben formar un vector finito y no negativo")
    return a


def validate_labels(y, amount):
    a = validate_amount(amount)
    labels = np.asarray(y)
    if labels.ndim != 1 or len(labels) != len(a) or not np.isin(labels, (0, 1)).all():
        raise ValueError("Las etiquetas deben ser 0/1 y coincidir con los montos")
    return labels.astype(int), a


def bmr_actions(probability, amount, ca):
    """Compara riesgos sin dividir: aprueba en empate y con monto cero."""
    a = validate_amount(amount)
    p = np.asarray(probability, dtype=float)
    if p.ndim != 1 or len(p) != len(a) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Las probabilidades deben estar en [0,1] y coincidir con los montos")
    return np.where(p * a > validate_ca(ca), "block", "approve")


def realized_cost(y, amount, actions, ca):
    """Costo por fila: Ca si intervenimos; monto si aprobamos fraude."""
    labels, a = validate_labels(y, amount)
    action = np.asarray(actions)
    if action.ndim != 1 or len(action) != len(a) or not np.isin(action, ("approve", "block")).all():
        raise ValueError("Las acciones deben ser approve/block y coincidir con los montos")
    return np.where(action == "block", validate_ca(ca), labels * a)


def cost_sensitive_targets_weights(y, amount, ca):
    """Acción de menor costo con etiqueta conocida y arrepentimiento de errar.

    Fraude: diferencia A-Ca; si A<=Ca, aprobar es óptimo. Legítima:
    diferencia Ca, aprobar es óptimo. Esto reproduce el objetivo empírico
    como error ponderado más una constante, aunque el árbol es heurístico.
    """
    labels, a = validate_labels(y, amount)
    c = validate_ca(ca)
    approve_cost = labels * a
    targets = (approve_cost > c).astype(int)
    weights = np.abs(approve_cost - c)
    return targets, weights


def evaluate_policies(y, amount, policies, ca):
    """Compara políticas bajo la misma matriz y dos referencias explícitas."""
    labels, a = validate_labels(y, amount)
    c = validate_ca(ca)
    if not len(a) or "fixed_0_5" not in policies:
        raise ValueError("Se necesitan filas y la política fixed_0_5")
    fixed_cost = float(realized_cost(labels, a, policies["fixed_0_5"], c).sum())
    approve_all = float(a[labels == 1].sum())
    intervene_all = len(a) * c
    reference = min(approve_all, intervene_all)
    fraud = labels == 1
    legitimate = ~fraud
    rows = []
    for name, action in policies.items():
        action = np.asarray(action)
        cost = realized_cost(labels, a, action, c)
        block = action == "block"
        total = float(cost.sum())
        admin = float(block.sum() * c)
        missed = float(a[fraud & ~block].sum())
        rows.append({
            "strategy": name, "administrative_cost_ca": c,
            "rows": len(a), "frauds": int(fraud.sum()),
            "cost_total_proxy": total, "administrative_cost_total": admin,
            "missed_fraud_amount": missed, "cost_per_transaction_proxy": total / len(a),
            "approve_all_cost": approve_all, "intervene_all_cost": intervene_all,
            "reference_cost": reference,
            "savings_vs_reference": reference - total,
            "savings_fraction_vs_reference": 1 - total / reference if reference > 0 else np.nan,
            "fixed_baseline_cost_proxy": fixed_cost,
            "savings_vs_fixed_proxy": fixed_cost - total,
            "savings_fraction_vs_fixed_proxy": 1 - total / fixed_cost if fixed_cost > 0 else np.nan,
            "frauds_blocked": int((fraud & block).sum()),
            "frauds_missed": int((fraud & ~block).sum()),
            "fraud_recall": float(block[fraud].mean()) if fraud.any() else np.nan,
            "legitimate_block_count": int((legitimate & block).sum()),
            "legitimate_block_rate": float(block[legitimate].mean()) if legitimate.any() else np.nan,
            "policy_block_count": int(block.sum()),
            "amount_weighted_recall": float(a[fraud & block].sum() / approve_all) if approve_all > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def fit_cost_tree(X, y, amount, ca, *, random_state=42):
    """Árbol ponderado por arrepentimiento; representa acciones, no etiquetas."""
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.dummy import DummyClassifier
    target, weight = cost_sensitive_targets_weights(y, amount, ca)
    if X.shape[0] != len(target):
        raise ValueError("X y las etiquetas deben tener igual número de filas")
    if not np.any(weight > 0):
        return DummyClassifier(strategy="constant", constant=0).fit(X, target)
    tree = DecisionTreeClassifier(max_depth=12, min_samples_leaf=250, random_state=random_state)
    tree.fit(X, target, sample_weight=weight / weight.mean())
    return tree


def require_approved_costs(config):
    """Valida escenarios acordados antes de una comparación final."""
    if not config.get("approved") or not str(config.get("approval_reference") or "").strip():
        raise ValueError("Registre el acuerdo y su referencia antes de evaluar costos finales")
    unit = config.get("amount_unit")
    if not isinstance(unit, str) or not unit.strip() or unit != config.get("cost_unit"):
        raise ValueError("Ca y TransactionAmt deben tener la misma unidad documentada")
    values = config.get("administrative_cost_grid")
    if values is None or len(values) == 0:
        raise ValueError("Defina los escenarios de Ca")
    validated = tuple(validate_ca(c) for c in values)
    if len(set(validated)) != len(validated):
        raise ValueError("Los escenarios de Ca no deben repetirse")
    return validated
