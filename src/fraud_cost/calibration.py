"""Platt/isotónica sobre el mismo predictor fijo; experimento de desarrollo."""
from __future__ import annotations

from datetime import datetime, timezone
import gc
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss

from fraud_cost.costs import bmr_actions, evaluate_policies, realized_cost, validate_labels
from fraud_cost.diagnostics import (AMOUNT_EDGES, RISK_EDGES, band_labels, bands,
                                    checked_scores, probability_summary)
from fraud_cost.economics import checked_ca_grid, verified_refit
from fraud_cost.refit import (calibrated_probabilities, check_blocks, load_development,
                             predictive_metrics, probabilities, source_sha256,
                             split_validation, verified_selection)

LITERATURE_INSPIRED_CA = (1., 2.5, 5., 10.)
EXTENDED_CA = (20., 50., 100.)
DIAGNOSTIC_CA = LITERATURE_INSPIRED_CA + EXTENDED_CA
METHOD_COLUMNS = {"raw": "probability_raw", "platt": "probability_calibrated", "isotonic": "probability_isotonic"}


def aligned_policy(policy, saved):
    saved = checked_scores(saved)
    if len(policy) != len(saved):
        raise ValueError("La población posterior no coincide con los scores")
    for column in ("TransactionID", "TransactionDT", "TransactionAmt", "isFraud"):
        if not np.array_equal(policy[column].to_numpy(), saved[column].to_numpy()):
            raise ValueError(f"La población posterior cambió o está desalineada: {column}")
    return saved


def load_calibration_inputs(root, *, progress=print):
    """Reutiliza predictor/preprocesamiento; descarta test/gaps antes de inferir."""
    verified_selection(root)
    predictor, saved, config = verified_refit(root)
    progress("Fuentes verificadas. Cargando únicamente train y validación para reconstruir las ventanas.", flush=True)
    train, validation = load_development(root)
    early, policy = split_validation(validation, calibration_fraction=config["validation_calibration_fraction_requested"])
    windows = check_blocks(train, early, policy)
    if windows.to_dict(orient="records") != config["windows"] or train.TransactionDT.max() != predictor.train_end:
        raise ValueError("Las ventanas no corresponden al reajuste del 06")
    saved = aligned_policy(policy, saved)
    del train, validation
    gc.collect()
    return predictor, early, policy, saved, config


def fit_isotonic(p, y):
    p = probabilities(p)
    labels, _ = validate_labels(y, np.zeros(len(p)))
    if np.unique(labels).size != 2:
        raise ValueError("La calibración requiere ambas clases")
    # Sin pesos ni remuestreo: conservamos la prevalencia temprana observada.
    return IsotonicRegression(increasing=True, y_min=0., y_max=1., out_of_bounds="clip").fit(p, labels)


def isotonic_probabilities(calibrator, raw):
    return probabilities(calibrator.predict(probabilities(raw)))


def group_diagnostics(scores):
    """Grupos comunes del 08 (riesgo Platt), nunca definidos con etiquetas."""
    scores = checked_scores(scores)
    probabilities(scores.probability_isotonic)
    amount_groups = bands(scores.TransactionAmt, AMOUNT_EDGES)
    risk_groups = bands(scores.probability_calibrated, RISK_EDGES)
    times = scores.TransactionDT.to_numpy(float)
    if times.min() == times.max():
        raise ValueError("Se necesita una ventana de duración positiva")
    periods = bands(times, np.linspace(times.min(), times.max(), 5))
    groups = [("all", 0, "all", np.ones(len(scores), dtype=bool))]
    groups += [("amount", i, label, amount_groups == i) for i, label in enumerate(band_labels(AMOUNT_EDGES))]
    groups += [("risk_platt", i, label, risk_groups == i) for i, label in enumerate(band_labels(RISK_EDGES))]
    groups += [("period", i+1, str(i+1), periods == i) for i in range(4)]
    records = []
    for grouping, index, label, mask in groups:
        part = scores.loc[mask]
        for method, column in METHOD_COLUMNS.items():
            records.append({"grouping": grouping, "group_index": index, "group_label": label,
                            "method": method, **probability_summary(part, column),
                            "log_loss": float(log_loss(part.isFraud, part[column].to_numpy(float), labels=[0,1])) if len(part) else np.nan})
    return pd.DataFrame(records)


def bmr_calibrator_diagnostic(scores, *, ca_grid=DIAGNOSTIC_CA):
    """La política fija Platt permanece como ancla común, no se retoca por método."""
    scores = checked_scores(scores)
    probabilities(scores.probability_isotonic)
    amount = scores.TransactionAmt.to_numpy(float)
    y = scores.isFraud.to_numpy()
    comparisons, changes = [], []
    for ca in checked_ca_grid(ca_grid):
        policies = {"fixed_0_5": scores.action_fixed_0_5.to_numpy()}
        policies.update({f"bmr_{method}": bmr_actions(scores[column], amount, ca) for method, column in METHOD_COLUMNS.items()})
        table = evaluate_policies(y, amount, policies, ca)
        table["role"] = "DIAGNOSTICO_CALIBRADORES_NO_COMPARACION_FINAL_TRES_POLITICAS"
        table["scenario_group"] = "inspirado_en_literatura" if ca in LITERATURE_INSPIRED_CA else "sensibilidad_ampliada"
        comparisons.append(table)
        before, after = policies["bmr_platt"], policies["bmr_isotonic"]
        old_block, new_block = before == "block", after == "block"
        old_cost, new_cost = realized_cost(y, amount, before, ca), realized_cost(y, amount, after, ca)
        admin_delta = ca*(int(new_block.sum())-int(old_block.sum()))
        missed_delta = amount[(y == 1) & ~new_block].sum()-amount[(y == 1) & ~old_block].sum()
        changes.append({"administrative_cost_ca": ca, "rows": len(scores),
                        "actions_changed": int((before != after).sum()),
                        "additional_interventions": int((~old_block & new_block).sum()),
                        "fewer_interventions": int((old_block & ~new_block).sum()),
                        "platt_bmr_cost": float(old_cost.sum()), "isotonic_bmr_cost": float(new_cost.sum()),
                        "isotonic_minus_platt_cost": float((new_cost-old_cost).sum()),
                        "administrative_cost_delta": float(admin_delta), "missed_fraud_amount_delta": float(missed_delta)})
    return pd.concat(comparisons, ignore_index=True), pd.DataFrame(changes)


def compare_calibrators(predictor, early, policy, saved, *, ca_grid=DIAGNOSTIC_CA, progress=print):
    """Ajusta SOLO isotónica en early. Las etiquetas policy solo evalúan."""
    for block in (early, policy):
        if block.empty or not block.split.eq("validation").all() or block.TransactionID.duplicated().any():
            raise ValueError("Solo se admiten ventanas de validación con IDs únicos")
        validate_labels(block.isFraud, block.TransactionAmt)
        if not np.isfinite(block.TransactionDT.to_numpy(float)).all():
            raise ValueError("Tiempo no finito")
    if (set(early.TransactionID).intersection(policy.TransactionID) or
            predictor.train_end+7*86400 >= early.TransactionDT.min() or
            early.TransactionDT.max() >= policy.TransactionDT.min()):
        raise ValueError("Las ventanas deben ser disjuntas, posteriores a train y cronológicas")
    saved = aligned_policy(policy, saved)
    before = joblib.hash(predictor)
    raw_policy = predictor.predict_raw(policy)
    np.testing.assert_allclose(raw_policy, saved.probability_raw, rtol=0, atol=1e-12)
    np.testing.assert_allclose(calibrated_probabilities(predictor.calibrator, raw_policy),
                               saved.probability_calibrated, rtol=0, atol=1e-12)
    progress(f"Reconstruidas puntuaciones posteriores. Inferencia temprana para {len(early):,} operaciones.", flush=True)
    raw_early = predictor.predict_raw(early)
    calibrator = fit_isotonic(raw_early, early.isFraud)
    result = saved.copy()
    result["probability_isotonic"] = isotonic_probabilities(calibrator, raw_policy)
    if joblib.hash(predictor) != before:
        raise ValueError("La comparación modificó el predictor congelado")
    records = []
    for method, column in METHOD_COLUMNS.items():
        p = result[column].to_numpy(float)
        unique, counts = np.unique(p, return_counts=True)
        records.append({"method": method, **predictive_metrics(result.isFraud, p, name=method),
                        "unique_probabilities": len(unique), "rows_in_tied_probabilities": int(counts[counts > 1].sum()),
                        "probability_zero_count": int((p == 0).sum()), "probability_one_count": int((p == 1).sum()),
                        "frauds_with_probability_zero": int(((p == 0) & result.isFraud.eq(1)).sum()),
                        "legitimate_with_probability_one": int(((p == 1) & result.isFraud.eq(0)).sum())})
    metrics = pd.DataFrame(records)
    comparisons, changes = bmr_calibrator_diagnostic(result, ca_grid=ca_grid)
    diagnostics = {"calibration_rows": len(early), "calibration_frauds": int(early.isFraud.sum()),
                   "isotonic_knots": len(calibrator.X_thresholds_),
                   "calibration_raw_min": float(np.min(raw_early)), "calibration_raw_max": float(np.max(raw_early)),
                   "policy_below_calibration_raw_min": int((raw_policy < np.min(raw_early)).sum()),
                   "policy_above_calibration_raw_max": int((raw_policy > np.max(raw_early)).sum()),
                   "predictor_unchanged_joblib_hash": before}
    progress("Isotónica ajustada solo en validación temprana; predictor y Platt intactos.", flush=True)
    return calibrator, result, metrics, group_diagnostics(result), comparisons, changes, diagnostics


def calibration_figures(predictor, calibrator, groups, comparison):
    import matplotlib.pyplot as plt

    styles = (("raw", "Original", "#777777"), ("platt", "Platt", "#007f86"), ("isotonic", "Isotónica", "#c16a1b"))
    fig1, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.linspace(0.,1.,1001)
    mapped = {"raw": x, "platt": calibrated_probabilities(predictor.calibrator, x), "isotonic": isotonic_probabilities(calibrator, x)}
    for method, label, color in styles:
        axes[0].plot(x, mapped[method], label=label, color=color)
        part = groups.loc[groups.grouping.eq("risk_platt") & groups.method.eq(method) & groups.rows.gt(0)]
        axes[1].plot(part.mean_probability, part.observed_rate, "o-", color=color, label=label)
    axes[1].plot([0,1],[0,1],"k--",linewidth=1,label="Coincidencia ideal")
    axes[0].set(title="Transformación aprendida en validación temprana", xlabel="Probabilidad original", ylabel="Probabilidad transformada")
    axes[1].set(title="Confiabilidad posterior — grupos comunes Platt", xlabel="Probabilidad media", ylabel="Frecuencia de fraude observada")
    for axis in axes:
        axis.set_xlim(0,1)
        axis.set_ylim(0,1)
        axis.legend(fontsize=8)
        axis.grid(alpha=.2)
    fig1.tight_layout()
    fig2, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for method, label, color in styles:
        part = comparison.loc[comparison.strategy.eq(f"bmr_{method}")]
        axes[0].plot(part.administrative_cost_ca,part.cost_total_proxy,"o-",label=f"BMR {label}",color=color)
        amount_part = groups.loc[groups.grouping.eq("amount") & groups.method.eq(method)]
        positions = amount_part.group_index.to_numpy()+(list(METHOD_COLUMNS).index(method)-1)*.25
        axes[1].bar(positions,amount_part.observed_minus_expected_fraud_amount,width=.25,color=color,label=label)
    fixed = comparison.loc[comparison.strategy.eq("fixed_0_5")]
    axes[0].plot(fixed.administrative_cost_ca,fixed.cost_total_proxy,"k--",label="Fija Platt 0,5 (referencia)")
    axes[0].set_xscale("log")
    axes[0].set(title="Sensibilidad BMR — escenarios hipotéticos",xlabel="Ca en unidades del dataset (escala log)",ylabel="Costo realizado")
    axes[1].set_xticks(range(6),band_labels(AMOUNT_EDGES),rotation=30,ha="right",fontsize=8)
    axes[1].axhline(0,color="black",linewidth=1)
    axes[1].set(title="Residuo de monto: observado − esperado",ylabel="Unidades del dataset")
    for axis in axes:
        axis.legend(fontsize=8)
        axis.grid(alpha=.2,axis="y")
    fig2.tight_layout()
    return {"comparacion_calibradores_validation.png":fig1, "calibradores_bmr_y_montos.png":fig2}


def save_calibration_experiment(root, calibrator, metrics, groups, comparison, changes, diagnostics, figures, *, ca_grid=DIAGNOSTIC_CA):
    root = Path(root)
    outputs = []
    for filename, frame in (("calibradores_metricas_validation.csv", metrics), ("calibradores_grupos_validation.csv", groups),
                            ("calibradores_bmr_sensibilidad.csv", comparison), ("calibradores_cambios_bmr.csv", changes)):
        path = root/"results/tables"/filename
        frame.to_csv(path,index=False)
        outputs.append(path)
    base = root/"results/models/selected_xgboost_platt.joblib"
    candidate_path = root/"results/models/candidate_isotonic_calibrator.joblib"
    joblib.dump({"calibrator":calibrator, "input":"raw_xgboost_probability_not_logit",
                 "base_predictor_sha256":source_sha256(base), "status":"candidate_not_approved"},candidate_path,compress=3)
    outputs.append(candidate_path)
    for filename, figure in figures.items():
        path = root/"results/figures"/filename
        figure.savefig(path,dpi=160,bbox_inches="tight")
        outputs.append(path)
    sources = [root/f"src/fraud_cost/{name}.py" for name in ("calibration","diagnostics","economics","refit","costs","features")]
    sources += [root/"notebooks/09_comparacion_calibradores.ipynb", root/"results/tables/reajuste_config.json",
                root/"results/tables/seleccion_temporal_config.json", base, root/"data/interim/validation_scores_selected.parquet"]
    record = {"created_at_utc":datetime.now(timezone.utc).isoformat(), "test_evaluated":False,
              "base_predictor_refitted":False, "platt_refitted":False, "selected_calibrator":None,
              "calibrator_approved":False, "costs_approved":False, "experiment":"Platt_vs_isotonic_development",
              "isotonic_params":calibrator.get_params(), "diagnostics":diagnostics,
              "comparison_protocol":"log-loss global como lectura principal, Brier y grupos como complementos; sin selección automática ni por Ca",
              "log_loss_probability_dtype":"float64; clipping interno sklearn con epsilon de máquina, sin suavizado adicional",
              "group_protocol":"mismas bandas de monto/riesgo Platt y cuatro subperiodos que el 08",
              "ca_grid":checked_ca_grid(ca_grid), "literature_inspired_ca":LITERATURE_INSPIRED_CA, "extended_ca":EXTENDED_CA,
              "scenario_status":"HIPOTETICO_NO_COSTO_OBSERVADO_IEEE_CIS", "cost_units":"unidades_del_dataset_moneda_no_verificada",
              "literature_sources":[{"doi":"10.1109/ICMLA.2013.68", "original_ca_eur":[1,2.5,5]},
                                    {"doi":"10.1137/1.9781611973440.78", "original_ca_eur":[10]}],
              "note":"Desarrollo adicional motivado por 06/08; validación posterior reutilizada, no confirmación independiente. No cambia la comparación principal del 07.",
              "source_sha256":{p.relative_to(root).as_posix():source_sha256(p,code_only=p.suffix==".ipynb") for p in sources},
              "artifact_sha256":{p.relative_to(root).as_posix():source_sha256(p) for p in outputs},
              "versions":{p:importlib.metadata.version(p) for p in ("numpy","pandas","scikit-learn","xgboost","joblib","matplotlib")},
              "python_version":sys.version.split()[0], "python_executable":sys.executable,
              "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip(),
              "git_uncommitted":subprocess.check_output(["git","status","--short"],cwd=root,text=True).splitlines()}
    (root/"results/tables/comparacion_calibradores_config.json").write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    return record
