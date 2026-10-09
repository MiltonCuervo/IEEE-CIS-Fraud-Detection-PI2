"""Comparación económica exploratoria de tres políticas, sin test."""
from __future__ import annotations

from datetime import datetime, timezone
import gc
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time

import joblib
import numpy as np
import pandas as pd

from fraud_cost.costs import bmr_actions, evaluate_policies, fit_cost_tree, realized_cost, validate_ca
from fraud_cost.features import PastOnlyFeatures, transaction_features
from fraud_cost.refit import (calibrated_probabilities, check_blocks, load_development,
                             source_sha256, split_validation)
from fraud_cost.scenarios import HYPOTHETICAL_CA_GRID, validate_validation_scores


def checked_ca_grid(values):
    values = tuple(validate_ca(c) for c in values)
    if not values or any(c <= 0 for c in values) or any(a >= b for a, b in zip(values, values[1:])):
        raise ValueError("Esta exploración requiere Ca positivos, únicos y crecientes")
    return values


def verified_refit(root):
    """Verifica hashes antes de cargar el artefacto local confiable del 06."""
    root = Path(root)
    config = json.loads((root / "results/tables/reajuste_config.json").read_text(encoding="utf-8"))
    if config.get("test_evaluated") is not False:
        raise ValueError("El reajuste debe mantener test sin puntuar")
    for group in ("source_sha256", "artifact_sha256"):
        if not config.get(group):
            raise ValueError("Falta trazabilidad del reajuste")
        for relative, expected in config[group].items():
            path = root / relative.replace("\\", "/")
            if source_sha256(path, code_only=path.suffix == ".ipynb") != expected:
                raise ValueError(f"Cambió una entrada del 06: {relative}. Revise su procedencia.")
    predictor = joblib.load(root / "results/models/selected_xgboost_platt.joblib")
    scores = pd.read_parquet(root / "data/interim/validation_scores_selected.parquet")
    validate_validation_scores(scores)
    np.testing.assert_allclose(calibrated_probabilities(predictor.calibrator, scores.probability_raw),
                               scores.probability_calibrated, rtol=0, atol=1e-12)
    if predictor.params != config["predictor_params"]:
        raise ValueError("El predictor no coincide con su configuración")
    return predictor, scores, config


def load_economic_inputs(root, scores, config):
    train, validation = load_development(root)
    early, policy = split_validation(validation, calibration_fraction=config["validation_calibration_fraction_requested"])
    windows = check_blocks(train, early, policy)
    if windows.to_dict(orient="records") != config["windows"]:
        raise ValueError("Las ventanas no coinciden con el reajuste")
    aligned = scores.sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)
    for column in ("TransactionID", "TransactionDT", "TransactionAmt", "isFraud"):
        if not np.array_equal(policy[column].to_numpy(), aligned[column].to_numpy()):
            raise ValueError(f"Las probabilidades no corresponden a la población actual: {column}")
    del validation, early
    gc.collect()
    return train, policy, aligned


def prepare_cost_matrices(predictor, train, policy, scores):
    """Reconstruye atributos; NO reajusta vocabulario, XGBoost o calibración."""
    if not train.split.eq("train").all() or not policy.split.eq("validation").all():
        raise ValueError("Solo se admiten train y validation posterior")
    if train.TransactionDT.max() != predictor.train_end or policy.TransactionDT.min() <= predictor.train_end:
        raise ValueError("El pasado de entrenamiento no coincide con el predictor")
    if set(train.TransactionID).intersection(policy.TransactionID):
        raise ValueError("Entrenamiento y evaluación comparten identificadores")
    if not np.array_equal(policy.TransactionID.to_numpy(), scores.TransactionID.to_numpy()):
        raise ValueError("Las filas de atributos y probabilidades están desalineadas")
    builder = PastOnlyFeatures(entities=predictor.builder.entities) if predictor.builder else None
    X_train = builder.fit_transform(train) if builder else transaction_features(train)
    if set(X_train.columns) != set(predictor.feature_columns):
        raise ValueError("Cambió el esquema de entrenamiento")
    Xt = predictor.preprocessor.transform(X_train)
    del X_train, builder
    gc.collect()
    X_policy = predictor.builder.transform(policy) if predictor.builder else transaction_features(policy)
    if set(X_policy.columns) != set(predictor.feature_columns):
        raise ValueError("Cambió el esquema de evaluación")
    Xv = predictor.preprocessor.transform(X_policy)
    del X_policy
    gc.collect()
    # Los árboles reciben CSC una sola vez, sin conversiones repetidas por Ca.
    if hasattr(Xt, "tocsc"):
        Xt = Xt.tocsc()
        Xt.sort_indices()
    np.testing.assert_allclose(predictor.model.predict_proba(Xv)[:, 1], scores.probability_raw,
                               rtol=0, atol=1e-12)
    return Xt, Xv


def calibration_action_diagnostic(scores, ca_grid=HYPOTHETICAL_CA_GRID):
    """Diagnóstico BMR raw/calibrado; no selecciona el calibrador ni Ca."""
    amount, y, p, _ = validate_validation_scores(scores)
    raw = scores.probability_raw.to_numpy(dtype=float)
    records = []
    for ca in checked_ca_grid(ca_grid):
        before, after = bmr_actions(raw, amount, ca), bmr_actions(p, amount, ca)
        old_cost, new_cost = realized_cost(y, amount, before, ca), realized_cost(y, amount, after, ca)
        old_block, new_block = before == "block", after == "block"
        records.append({"administrative_cost_ca": ca, "rows": len(scores),
                        "actions_changed": int((before != after).sum()),
                        "additional_interventions": int((~old_block & new_block).sum()),
                        "fewer_interventions": int((old_block & ~new_block).sum()),
                        "raw_bmr_cost": float(old_cost.sum()), "calibrated_bmr_cost": float(new_cost.sum()),
                        "calibrated_minus_raw_cost": float((new_cost - old_cost).sum()),
                        "administrative_cost_delta": float(ca * (int(new_block.sum()) - int(old_block.sum()))),
                        "missed_fraud_amount_delta": float(amount[(y == 1) & ~new_block].sum() - amount[(y == 1) & ~old_block].sum()),
                        "scenario_status": "HIPOTETICO_NO_SELECCION_DE_CALIBRADOR"})
    return pd.DataFrame(records)


def compare_economic_policies(train, scores, Xt, Xv, *, ca_grid=HYPOTHETICAL_CA_GRID, random_state=42, progress=print):
    if not train.split.eq("train").all() or Xt.shape[0] != len(train) or Xv.shape[0] != len(scores):
        raise ValueError("Las matrices deben corresponder a train y validación posterior")
    if set(train.TransactionID).intersection(scores.TransactionID):
        raise ValueError("La evaluación comparte identificadores con train")
    amount, y, p, fixed = validate_validation_scores(scores)
    trees, records, diagnostics = {}, [], []
    actions = scores[["TransactionID"]].copy()
    actions["fixed_0_5"] = fixed == "block"
    for ca in checked_ca_grid(ca_grid):
        progress(f"Ca={ca:g}: ajuste del árbol ponderado con {len(train):,} transacciones de train.", flush=True)
        start = time.perf_counter()
        tree = fit_cost_tree(Xt, train.isFraud, train.TransactionAmt, ca, random_state=random_state)
        tree_actions = np.where(tree.predict(Xv) == 1, "block", "approve")
        policies = {"fixed_0_5": fixed, "bmr": bmr_actions(p, amount, ca), "cost_sensitive_tree": tree_actions}
        table = evaluate_policies(y, amount, policies, ca)
        expected = {name: float(np.where(action == "block", ca, p * amount).sum()) for name, action in policies.items()}
        table["expected_cost_common_probabilities"] = table.strategy.map(expected)
        table["realized_minus_expected_cost"] = table.cost_total_proxy - table.expected_cost_common_probabilities
        table["evaluation_block"] = "validation_policy"
        table["scenario_status"] = "HIPOTETICO_CA_NO_APROBADO"
        records.append(table)
        key = f"ca_{ca:g}"
        actions[f"bmr_{key}"] = policies["bmr"] == "block"
        actions[f"tree_{key}"] = tree_actions == "block"
        actions[f"raw_bmr_{key}"] = bmr_actions(scores.probability_raw, amount, ca) == "block"
        trees[ca] = tree
        diagnostics.append({"administrative_cost_ca": ca, "train_rows": len(train), "features": Xt.shape[1],
                            "depth": tree.get_depth() if hasattr(tree, "get_depth") else 0,
                            "leaves": tree.get_n_leaves() if hasattr(tree, "get_n_leaves") else 1,
                            "seconds": time.perf_counter() - start,
                            "max_depth": 12, "min_samples_leaf": 250, "random_state": random_state})
        progress(f"  Costo BMR={table.loc[table.strategy.eq('bmr'), 'cost_total_proxy'].iloc[0]:,.2f}; árbol={table.loc[table.strategy.eq('cost_sensitive_tree'), 'cost_total_proxy'].iloc[0]:,.2f}.", flush=True)
    return pd.concat(records, ignore_index=True), calibration_action_diagnostic(scores, ca_grid), pd.DataFrame(diagnostics), trees, actions


def save_economic_comparison(root, comparison, calibration, tree_diagnostics, trees, actions, *, ca_grid=HYPOTHETICAL_CA_GRID):
    root = Path(root)
    tables = root / "results/tables"
    comparison.to_csv(tables / "comparacion_economica_validation.csv", index=False)
    calibration.to_csv(tables / "diagnostico_bmr_calibracion.csv", index=False)
    tree_diagnostics.to_csv(tables / "arboles_costos_diagnostico.csv", index=False)
    tree_path = root / "results/models/cost_trees_hypothetical.joblib"
    action_path = root / "data/interim/decisiones_economicas_validation.parquet"
    joblib.dump(trees, tree_path, compress=3)
    actions.to_parquet(action_path, index=False)
    sources = [root / f"src/fraud_cost/{name}.py" for name in ("economics", "costs", "scenarios", "refit", "features")]
    sources += [root / "notebooks/07_comparacion_economica_validation.ipynb", root / "results/tables/reajuste_config.json",
                root / "results/models/selected_xgboost_platt.joblib", root / "data/interim/validation_scores_selected.parquet"]
    record = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "ca_grid": checked_ca_grid(ca_grid),
              "scenario_status": "HIPOTETICO_CA_NO_APROBADO", "cost_units": "unidades_del_dataset_moneda_no_verificada",
              "matrix": {"TP": "Ca", "FP": "Ca", "FN": "TransactionAmt", "TN": 0},
              "strategies": ["fixed_0_5", "bmr", "cost_sensitive_tree"], "classification_threshold": .5,
              "test_evaluated": False, "predictor_or_calibrator_refitted": False,
              "tree_parameters": {str(c): tree.get_params() for c, tree in trees.items()},
              "expected_risk_note": "Mismas probabilidades del XGBoost para las tres acciones; ventaja BMR esperada es por construcción, no una prueba empírica.",
              "raw_bmr_role": "diagnóstico de sensibilidad, no estrategia principal ni elección posterior de calibrador",
              "source_sha256": {p.relative_to(root).as_posix(): source_sha256(p, code_only=p.suffix == ".ipynb") for p in sources},
              "artifact_sha256": {p.relative_to(root).as_posix(): source_sha256(p) for p in (tree_path, action_path)},
              "notebook_hash_method": "JSON de tipo y fuente de celdas de código; excluye Markdown, salidas y conteos",
              "versions": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scikit-learn", "scipy", "joblib")},
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              "git_uncommitted": subprocess.check_output(["git", "status", "--short"], cwd=root, text=True).splitlines(),
              "note": "Exploración retrospectiva en validation posterior. Árbol ponderado es aproximación; Ca no se optimiza a partir del ahorro."}
    (tables / "comparacion_economica_config.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record
