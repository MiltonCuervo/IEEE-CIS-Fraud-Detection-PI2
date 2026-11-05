"""Protocolo técnico previo a test: fija decisiones sin generar predicciones."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fraud_cost.artifacts import (
    notebook_code_sha256, read_record, sha256_file, verify_hashes, verify_reference_artifacts,
)
from fraud_cost.split import validate_temporal_split
from fraud_cost.stability import load_validation_evidence

PROTOCOL = "results/tables/protocolo_final.json"
RECORD = "results/tables/protocolo_final_config.json"
AUDIT = "results/tables/protocolo_final_comprobaciones.csv"
DECISIONS = "results/tables/protocolo_final_decisiones.csv"
TEST_RECORD = "results/tables/evaluacion_test_config.json"
NOTEBOOK = "notebooks/07_protocolo_final.ipynb"
CA_GRID = [1, 5, 10, 20, 50, 100]
TYPE_DISCREPANCIES = ["id_13", "id_14", "id_17", "id_18", "id_19", "id_20", "id_21",
                      "id_22", "id_24", "id_25", "id_26", "id_32"]
REFERENCES = [
    {"id": "bahnsen2013", "title": "Cost Sensitive Credit Card Fraud Detection using Bayes Minimum Risk",
     "year": 2013, "doi": "10.1109/ICMLA.2013.68",
     "url": "https://albahnsen.github.io/files/Cost%20Sensitive%20Credit%20Card%20Fraud%20Detection%20using%20Bayes%20Minimum%20Risk%20-%20Publish.pdf",
     "location": "Sección V.E y tabla VI, páginas 5–6 del PDF",
     "scope": "Escenarios 1, 2.5 y 5 euros en datos de otra procesadora; no estima costos IEEE-CIS."},
    {"id": "bahnsen2014", "title": "Improving Credit Card Fraud Detection with Calibrated Probabilities",
     "year": 2014,
     "url": "https://albahnsen.github.io/files/%20Improving%20Credit%20Card%20Fraud%20Detection%20by%20using%20Calibrated%20Probabilities%20-%20Publish.pdf",
     "location": "Sección 4, página 4 del PDF",
     "scope": "Ca=10 euros estimado con el equipo de riesgo de otra empresa; no trasladamos moneda ni costo operativo."},
]


def validate_decisions(protocol):
    """Evita congelar una formulación distinta de la desarrollada en 03–06."""
    if protocol["model"]["name"] != "lightgbm" or protocol["calibration"]["method"] != "platt_on_clipped_logit":
        raise ValueError("El predictor/calibrador difiere del protocolo desarrollado")
    if protocol["representation"] != {"columns": 432, "categorical": 37, "numeric": 395,
                                     "encoding": "ordinal", "unknown_category": -1, "dtype": "float32"}:
        raise ValueError("Cambió la representación ordinal evaluada")
    costs = protocol["costs"]
    if costs["ca_grid"] != CA_GRID or costs["unit"] != "unidades de TransactionAmt; moneda no verificada" or costs["operational_costs_estimated"]:
        raise ValueError("Los escenarios deben ser la sensibilidad hipotética fijada")
    if costs["matrix"] != {"approve_fraud": "TransactionAmt", "approve_legitimate": 0,
                            "intervene_fraud": "Ca", "intervene_legitimate": "Ca"}:
        raise ValueError("La matriz debe cobrar Ca en ambas clases intervenidas")
    if protocol["policies"] != {"fixed_0_5": "p_platt >= 0.5", "bmr": "p_platt * TransactionAmt > Ca; ties approve",
                                "cost_sensitive_tree": "reuse_stage03_tree_per_Ca; action_1_intervenes"}:
        raise ValueError("Cambió una regla de decisión")
    if not protocol["limitations"]["historical_test_label_aggregates_consulted"] or not protocol["limitations"]["development_validation_consulted"]:
        raise ValueError("Deben declararse las consultas históricas")
    if protocol["test_evaluated"] is not False or protocol["new_model_fitted"] is not False or protocol["new_calibrator_fitted"] is not False:
        raise ValueError("Congelar no implica ajustar ni evaluar test")


def gather_evidence(root):
    """Lee artefactos y manifiesto; no lee etiquetas test ni llama predict/fit."""
    root = Path(root)
    verify_reference_artifacts(root)
    scores, aligned, adoption = load_validation_evidence(root)
    stability = read_record(root, "results/tables/estabilidad_config.json")
    if stability["status"] != "complete" or stability["test_evaluated"] or stability["new_model_fitted"] or stability["new_calibrator_fitted"]:
        raise ValueError("La etapa 06 no es el diagnóstico completo esperado")
    if not stability["period_cost_components_reconciled_with_05"] or not stability["baseline_perturbation_reconciled_with_05"]:
        raise ValueError("Los costos del diagnóstico no se reconciliaron")
    verify_hashes(root, {**stability["source_sha256"], **stability["input_sha256"], **stability["output_sha256"],
                         "results/tables/estabilidad_protocolo.json": stability["protocol_sha256"]})
    if notebook_code_sha256(root / "notebooks/06_estabilidad_y_errores_economicos.ipynb") != stability["notebook_code_sha256"]:
        raise ValueError("Cambió el código del diagnóstico 06")
    chosen = read_record(root, "results/tables/seleccion_ordinal_config.json")["selected_candidate"]
    if chosen != adoption["selected_candidate"] or adoption["ca_grid"] != CA_GRID:
        raise ValueError("Selección y adopción no corresponden al protocolo esperado")
    model_path = "results/models/lightgbm_seleccionado_platt.joblib"
    # Su hash fue verificado por load_validation_evidence antes de deserializar.
    bundle = joblib.load(root / model_path)
    if bundle["model_name"] != "lightgbm" or bundle["calibration_method"] != "platt_on_clipped_logit" or bundle["logit_clip"] != 1e-6:
        raise ValueError("El paquete seleccionado declara otro predictor o calibrador")
    if any(bundle["model"].get_params()[k] != v for k, v in chosen["params"].items()):
        raise ValueError("El artefacto seleccionado tiene otros parámetros")
    pre = bundle["preprocessor"]
    branches = {name: len(columns) for name, _, columns in pre.transformers_ if name in ("categorical", "numeric")}
    if branches != {"categorical": 37, "numeric": 395} or bundle["model"].n_features_in_ != 432:
        raise ValueError("El modelo no recibe las 432 entradas verificadas")
    calibrator = bundle["calibrator"]
    if list(calibrator.classes_) != [0, 1] or calibrator.coef_.shape != (1, 1):
        raise ValueError("El Platt no tiene el contrato binario esperado")
    trees = joblib.load(root / "results/models/cost_sensitive_trees.joblib")
    if set(trees) != set(CA_GRID) or any(
        tree.get_params()[key] != expected
        for tree in trees.values()
        for key, expected in {"max_depth": 12, "min_samples_leaf": 250, "random_state": 42}.items()
    ):
        raise ValueError("Los árboles no tienen los parámetros desarrollados")
    manifest = pd.read_parquet(root / "data/interim/split_manifest.parquet", columns=["TransactionID", "TransactionDT", "split"])
    if manifest.TransactionID.isna().any() or manifest.TransactionID.duplicated().any():
        raise ValueError("IDs del manifiesto nulos o duplicados")
    validate_temporal_split(manifest, manifest.split)
    test = manifest.loc[manifest.split.eq("test")].sort_values(["TransactionDT", "TransactionID"])
    if test.empty or test.TransactionDT.min() <= scores.TransactionDT.max():
        raise ValueError("Test debe ser posterior a la validación")
    counts = manifest.split.value_counts().to_dict()
    if counts != {"train": 380815, "gap_1": 20995, "validation": 84093, "gap_2": 20404, "test": 84233}:
        raise ValueError("Cambió la población temporal congelada")
    identity = test[["TransactionID", "TransactionDT"]].to_numpy(dtype="<i8")
    test_metadata = {"rows": len(test), "time_start": int(test.TransactionDT.min()),
                     "time_end": int(test.TransactionDT.max()),
                     "id_time_sha256": hashlib.sha256(identity.tobytes()).hexdigest(),
                     "identity_order": "TransactionDT_then_TransactionID; little_endian_int64_row_major",
                     "labels_read": False}
    records = ["results/tables/comparacion_modelos_config.json", "results/tables/seleccion_ordinal_protocolo.json",
               "results/tables/seleccion_ordinal_config.json", "results/tables/lightgbm_adopcion_config.json",
               "results/tables/estabilidad_protocolo.json", "results/tables/estabilidad_config.json"]
    if adoption["source_artifact_status"] == "verified_reconstruction": records.append("results/tables/recuperacion_03_config.json")
    dependencies = {**adoption["input_sha256"], **adoption["source_sha256"], **adoption["output_sha256"],
                    **stability["input_sha256"], **stability["source_sha256"], **stability["output_sha256"]}
    paths = records + [p.relative_to(root).as_posix() for p in (root / "src/fraud_cost").glob("*.py")]
    paths += ["requirements.txt", "requirements-reproduction.txt"]
    dependencies.update({p: sha256_file(root / p) for p in paths})
    verify_hashes(root, dependencies)
    checks = [
        ("cadena_03_04_05_06", "Fuentes, registros, salidas y artefactos verificados"),
        ("seleccion_y_adopcion", "LightGBM referencia coincide en parámetros y 432 entradas"),
        ("platt_propio", "Calibrador del modelo adoptado; origen en validación temprana"),
        ("poblacion_validation", f"{len(scores)} operaciones posteriores y {int(scores.isFraud.sum())} fraudes"),
        ("acciones_alineadas", f"{sum(len(a) for a in aligned.values())} filas operación/Ca verificadas"),
        ("costos_reconciliados", "Subperiodos y perturbación base reproducen los costos del 05"),
        ("test_sin_etiquetas", f"Solo ID/tiempo/split: {len(test)} operaciones futuras"),
        ("escenarios", "Seis Ca hipotéticos; ninguna estimación de gasto operativo"),
    ]
    audit = pd.DataFrame([{"check": name, "status": "passed", "detail": detail} for name, detail in checks])
    return adoption, chosen, calibrator, test_metadata, dependencies, audit


def build_protocol(adoption, chosen, calibrator, test, dependencies, notebook_hash):
    protocol = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_for_academic_evaluation", "scope": "academic_hypothetical_cost_sensitivity",
        "model": {"name": "lightgbm", "candidate": chosen["candidate"], "params": chosen["params"],
                  "artifact": "results/models/lightgbm_seleccionado_platt.joblib", "fit_origin": "stage03_train"},
        "representation": {"columns": 432, "categorical": 37, "numeric": 395, "encoding": "ordinal",
                           "unknown_category": -1, "dtype": "float32"},
        "preparation": {"numeric_missing": "train_median; all_empty_kept_zero", "categorical_missing": "__MISSING__",
                        "additional_scaling": False, "new_engineered_features": False, "resampling": False,
                        "amount_for_costs": "original_float64", "excluded": ["TransactionID", "TransactionDT", "isFraud", "split"]},
        "calibration": {"method": "platt_on_clipped_logit", "clip_epsilon": 1e-6,
                        "coefficient_a": float(calibrator.coef_[0, 0]), "intercept_b": float(calibrator.intercept_[0]),
                        "fit_origin": "stage03_validation_early", "fit_rows": adoption["validation_windows"]["calibration_rows"],
                        "final_probability": "probability_calibrated", "raw_probability_role": "probabilistic_diagnostic_only"},
        "policies": {"fixed_0_5": "p_platt >= 0.5", "bmr": "p_platt * TransactionAmt > Ca; ties approve",
                     "cost_sensitive_tree": "reuse_stage03_tree_per_Ca; action_1_intervenes"},
        "trees": {"artifact": "results/models/cost_sensitive_trees.joblib", "target": "1[y * TransactionAmt > Ca]",
                  "weight": "abs(y * TransactionAmt - Ca); normalized_by_training_mean", "max_depth": 12,
                  "min_samples_leaf": 250, "random_state": 42, "global_cost_optimum_guaranteed": False},
        "costs": {"ca_grid": list(CA_GRID), "illustrative_ca": 10,
                  "unit": "unidades de TransactionAmt; moneda no verificada", "operational_costs_estimated": False,
                  "matrix": {"approve_fraud": "TransactionAmt", "approve_legitimate": 0,
                             "intervene_fraud": "Ca", "intervene_legitimate": "Ca"},
                  "grid_rationale": {"1_and_5": "numeric_scenarios_inspired_by_bahnsen2013",
                                     "10": "numeric_scenario_inspired_by_bahnsen2014",
                                     "20_50_100": "expanded_sensitivity_retained_from_development"},
                  "selection_by_observed_savings": False, "aggregate_across_ca": False,
                  "perfect_loss_prevention_assumed": True, "capacity_limit_modelled": False},
        "metrics": {"primary": ["cost_total_proxy", "administrative_cost_total", "missed_fraud_amount",
                                "cost_per_transaction_proxy", "savings_fraction_vs_reference", "savings_fraction_vs_fixed_proxy"],
                    "probabilistic_raw_and_platt": ["average_precision", "brier_score", "log_loss", "recall_at_0_5",
                                                    "precision_at_0_5", "legitimate_block_rate"],
                    "policy_diagnostics": ["TP", "FP", "FN", "TN", "fraud_recall", "legitimate_block_rate",
                                           "policy_block_count", "amount_weighted_recall"],
                    "average_precision_definition": "sklearn.average_precision_score; not_trapezoidal_PR_AUC",
                    "reference": "min(sum(y * original_amount), N * Ca)",
                    "savings_reference": "1 - cost/reference; NaN_if_reference_zero",
                    "improvement_fixed": "1 - cost/fixed_cost; NaN_if_fixed_cost_zero",
                    "zero_precision": "0_if_no_interventions", "zero_class_denominator": "NaN_if_class_absent"},
        "test": test, "validation_windows": adoption["validation_windows"],
        "reporting": {"per_ca_all_three_policies": True, "time_subperiods": 3,
                      "amount_edges": ["0", "50", "100", "250", "500", "1000", "inf"],
                      "interpretation": "descriptive_observed_costs; no_claim_of_statistical_superiority",
                      "change_model_or_threshold_after_test": False,
                      "test_run_record": TEST_RECORD, "refit_with_validation": False},
        "limitations": {"historical_test_label_aggregates_consulted": True, "development_validation_consulted": True,
                        "numeric_identity_codes": TYPE_DISCREPANCIES, "ordinal_artificial_order": True,
                        "catboost_native_categories_evaluated": False, "amount_is_net_banking_loss": False,
                        "calibration_remaining_underestimation_500_1000": True,
                        "review_imperfection_recovery_customer_harm_modelled": False},
        "review": {"professor_approval_of_final_protocol_recorded": False,
                   "note": "Congelación técnica del experimento académico; no acredita aval docente ni costos de negocio."},
        "references": REFERENCES, "dependency_sha256": dependencies, "notebook_code_sha256": notebook_hash,
        "test_evaluated": False, "new_model_fitted": False, "new_calibrator_fitted": False,
    }
    validate_decisions(protocol)
    return protocol


def verify_frozen_protocol(root):
    root = Path(root)
    record = read_record(root, RECORD)
    if record.get("status") != "complete" or not record.get("technical_protocol_frozen") or record.get("test_evaluated") is not False:
        raise ValueError("No hay una congelación completa previa a test")
    verify_hashes(root, record["output_sha256"])
    protocol = read_record(root, PROTOCOL)
    validate_decisions(protocol)
    verify_hashes(root, protocol["dependency_sha256"])
    if notebook_code_sha256(root / NOTEBOOK) != protocol["notebook_code_sha256"]:
        raise ValueError("Cambió el código del notebook de congelación")
    return protocol


def verify_before_test(root):
    root = Path(root)
    protocol = verify_frozen_protocol(root)
    if (root / TEST_RECORD).exists():
        raise FileExistsError("Ya existe un registro de ejecución test; revisar esa corrida antes de repetirla")
    return protocol


def freeze_protocol(root):
    root = Path(root)
    if (root / PROTOCOL).exists() or (root / RECORD).exists():
        return verify_frozen_protocol(root)
    adoption, chosen, calibrator, test, dependencies, audit = gather_evidence(root)
    protocol = build_protocol(adoption, chosen, calibrator, test, dependencies, notebook_code_sha256(root / NOTEBOOK))
    decisions = pd.DataFrame([
        {"decision": "predictor", "value": chosen["candidate"], "basis": "AP media interna; 04"},
        {"decision": "preparacion", "value": "432 ordinal/imputada", "basis": "Mismo ajuste train; 03/05"},
        {"decision": "calibrador", "value": "Platt propio", "basis": "Mejora probabilística; límites 05/06 declarados"},
        {"decision": "politicas", "value": "Fija 0.5; BMR; árbol por Ca", "basis": "Matriz original común"},
        {"decision": "Ca", "value": "1,5,10,20,50,100", "basis": "Escenarios hipotéticos; bibliografía y sensibilidad ampliada"},
        {"decision": "evaluacion", "value": "Test temporal completo; sin reajuste", "basis": "Identidad/tiempo congelados; etiquetas agregadas históricamente consultadas"},
    ])
    # No sobreescribir una congelación parcial ni una previa; requiere revisión explícita.
    for relative in (PROTOCOL, RECORD, AUDIT, DECISIONS):
        if (root / relative).exists(): raise FileExistsError(f"Salida de congelación ya existente: {relative}")
    audit.to_csv(root / AUDIT, index=False, lineterminator="\n")
    decisions.to_csv(root / DECISIONS, index=False, lineterminator="\n")
    with (root / PROTOCOL).open("x", encoding="utf-8", newline="\n") as target:
        json.dump(protocol, target, ensure_ascii=False, indent=2)
        target.write("\n")
    record = {"status": "complete", "technical_protocol_frozen": True,
              "created_utc": protocol["created_utc"], "test_evaluated": False,
              "new_model_fitted": False, "new_calibrator_fitted": False,
              "test_labels_read": False, "test_rows": test["rows"],
              "output_sha256": {p: sha256_file(root / p) for p in (PROTOCOL, AUDIT, DECISIONS)}}
    with (root / RECORD).open("x", encoding="utf-8", newline="\n") as target:
        json.dump(record, target, ensure_ascii=False, indent=2)
        target.write("\n")
    return verify_frozen_protocol(root)
