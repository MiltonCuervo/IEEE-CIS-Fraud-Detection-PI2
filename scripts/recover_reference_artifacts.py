"""Reconstruye el 03 en una copia aislada y publica solo artefactos verificados.

No sobrescribe evidencia histórica versionada, no vuelve a seleccionar y no
puntúa test. Ejecutar con el entorno del proyecto; --verify-only no entrena.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fraud_cost.artifacts import (
    ARTIFACTS, RECOVERY_RECORD, REFERENCE_RECORD, SELECTION_PROTOCOL, SELECTION_RECORD,
    notebook_code_sha256, project_path, read_record, sha256_file,
    verify_experimental_chain, verify_hashes, verify_reference_artifacts,
)

# Fijadas antes de ejecutar. Permiten redondeo de serialización, no diferencias
# materiales de métricas, acciones, vocabulario o tratamiento de variables.
ATOL = 1e-8
RTOL = 1e-8
COMPARISON_TABLES = (
    "comparacion_modelos_validation.csv", "modelos_calibracion_validation.csv",
    "costos_validation_sensibilidad.csv", "modelo_variables_ordinal.csv",
    "modelo_vocabulario_ordinal.csv",
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def compare_tables(original, reconstructed):
    import pandas as pd
    reports = []
    for filename in COMPARISON_TABLES:
        before = pd.read_csv(original / filename)
        after = pd.read_csv(reconstructed / filename)
        pd.testing.assert_frame_equal(before, after, check_exact=False, atol=ATOL, rtol=RTOL)
        errors = []
        for column in before.select_dtypes(include="number"):
            delta = (before[column].astype(float) - after[column].astype(float)).abs()
            if delta.notna().any():
                errors.append(float(delta.max()))
        reports.append({"table": filename, "rows": len(before), "passed": True,
                        "max_absolute_numeric_difference": max(errors, default=0.0)})
    return {"passed": True, "atol": ATOL, "rtol": RTOL, "tables": reports}


def verify_inference(stage):
    """Solo deserializa archivos recién generados en esta ejecución propia."""
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
    from fraud_cost.reference import read_model_frame, make_features
    manifest = pd.read_parquet(stage / "data/interim/split_manifest.parquet", columns=["TransactionID", "split"])
    frame = read_model_frame(stage / "data/raw", manifest)
    validation = frame.loc[frame.split.eq("validation")].reset_index(drop=True)
    end = int(round(len(validation) * 0.5))
    while end < len(validation) and validation.iloc[end - 1].TransactionDT == validation.iloc[end].TransactionDT:
        end += 1
    policy = validation.iloc[end:].reset_index(drop=True)
    scores = pd.read_parquet(stage / ARTIFACTS[0])
    pd.testing.assert_frame_equal(scores[list(policy.columns.intersection(scores.columns))],
                                  policy[list(policy.columns.intersection(scores.columns))], check_dtype=True)
    bundle = joblib.load(stage / "results/models/comparacion_modelos_ordinal.joblib")
    baseline = joblib.load(stage / "results/models/baseline_xgboost_platt.joblib")
    X = bundle["preprocessor"].transform(make_features(policy))
    assert X.shape == (42047, 432) and np.isfinite(X).all()
    metrics = pd.read_csv(stage / "results/tables/modelos_calibracion_validation.csv")
    y = policy.isFraud.to_numpy()
    for name, model in bundle["models"].items():
        inputs = bundle["logistic_scaler"].transform(X) if name == "logistica" else X
        raw = model.predict_proba(inputs)[:, 1]
        clipped = np.clip(np.asarray(raw, dtype=float), 1e-6, 1 - 1e-6)
        calibrated = bundle["calibrators"][name].predict_proba(np.log(clipped / (1 - clipped)).reshape(-1, 1))[:, 1]
        for kind, p in (("raw", raw), ("platt", calibrated)):
            row = metrics.loc[metrics.modelo.eq(name) & metrics.probabilidad.eq(kind)].iloc[0]
            actual = [average_precision_score(y, p), brier_score_loss(y, p), log_loss(y, p, labels=[0, 1])]
            np.testing.assert_allclose(actual, row[["average_precision", "brier_score", "log_loss"]].to_numpy(dtype=float), atol=ATOL, rtol=RTOL)
        if name == "xgboost":
            np.testing.assert_allclose(raw, scores.probability_raw, atol=ATOL, rtol=RTOL)
            np.testing.assert_allclose(calibrated, scores.probability_calibrated, atol=ATOL, rtol=RTOL)
            np.testing.assert_array_equal(np.where(calibrated >= 0.5, "block", "approve"), scores.action_fixed_0_5)
            np.testing.assert_allclose(baseline["model"].predict_proba(baseline["preprocessor"].transform(make_features(policy)))[:, 1], raw, atol=ATOL, rtol=RTOL)
    return {"passed": True, "policy_rows": len(policy), "matrix_columns": X.shape[1],
            "models_checked": list(bundle["models"]), "scores_match_xgboost": True,
            "test_predicted": False}


def publish(stage, backup, receipt):
    """Respalda todos los destinos y revierte si la publicación falla."""
    destinations = (*ARTIFACTS, RECOVERY_RECORD)
    existed = {}
    for relative in destinations:
        target = project_path(ROOT, relative)
        existed[relative] = target.exists()
        if target.exists():
            saved = backup / relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, saved)
    try:
        for relative in ARTIFACTS:
            target = project_path(ROOT, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            pending = target.with_name(target.name + ".recovering")
            shutil.copy2(stage / relative, pending)
            pending.replace(target)
        pending = project_path(ROOT, RECOVERY_RECORD).with_suffix(".json.recovering")
        write_json(pending, receipt)
        pending.replace(project_path(ROOT, RECOVERY_RECORD))
        verify_reference_artifacts(ROOT)
    except BaseException:
        for relative in destinations:
            target = project_path(ROOT, relative)
            if existed[relative]:
                shutil.copy2(backup / relative, target)
            elif target.exists():
                target.unlink()
        raise


def reconstruct():
    import nbformat
    from nbclient import NotebookClient
    reference, selection = verify_experimental_chain(ROOT)
    for package, expected in reference["versions"].items():
        if version(package) != expected:
            raise RuntimeError(f"Versión distinta: {package}; requiere {expected}. Use requirements-reproduction.txt")
    if sys.version_info[:2] not in ((3, 11), (3, 12)):
        raise RuntimeError("Use Python 3.11/3.12")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    run = project_path(ROOT, "data/interim/reference_recovery/" + run_id)
    stage = run / "workspace"
    for name in ("data/raw", "data/interim", "notebooks", "results/tables", "results/figures", "results/models", "src/fraud_cost"):
        (stage / name).mkdir(parents=True, exist_ok=True)
    # Las copias aisladas impiden que el notebook reemplace evidencia activa.
    for relative in reference["input_sha256"]:
        source, destination = project_path(ROOT, relative), stage / relative
        if relative.startswith("data/raw/"):
            os.link(source, destination)  # CSV de solo lectura lógica; el notebook no los modifica.
        else:
            shutil.copy2(source, destination)
    for source in (ROOT / "src/fraud_cost").glob("*.py"):
        shutil.copy2(source, stage / "src/fraud_cost" / source.name)
    shutil.copy2(ROOT / "requirements.txt", stage / "requirements.txt")
    shutil.copy2(ROOT / "results/tables/eda_diccionario_variables.csv", stage / "results/tables/eda_diccionario_variables.csv")
    notebook_path = stage / "notebooks/03_modelado_y_decisiones_costos.ipynb"
    shutil.copy2(ROOT / "notebooks/03_modelado_y_decisiones_costos.ipynb", notebook_path)
    plan = {"status": "planned_before_execution", "run_id": run_id, "atol": ATOL, "rtol": RTOL,
            "test_evaluated": False, "historical_records_sha256": {p: sha256_file(ROOT / p) for p in (REFERENCE_RECORD, SELECTION_PROTOCOL, SELECTION_RECORD)},
            "python": platform.python_version(), "executable": sys.executable}
    write_json(run / "plan.json", plan)
    (run / "environment.txt").write_text(subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
    notebook = nbformat.read(notebook_path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    def progress(cell, cell_index, **kwargs):
        if cell.cell_type == "code":
            print(f"Celda {cell_index} ejecutada", flush=True)
            for output in cell.get("outputs", []):
                if output.get("output_type") == "stream":
                    print(output.text[-1200:], flush=True)
    print(f"Reconstrucción aislada: {run_id}", flush=True)
    client = NotebookClient(notebook, timeout=3600, kernel_name="ieee-cis-pi2",
                            resources={"metadata": {"path": str(stage / "notebooks")}},
                            on_cell_executed=progress)
    try:
        client.execute()
    finally:
        nbformat.write(notebook, notebook_path)
    numerical = compare_tables(ROOT / "results/tables", stage / "results/tables")
    inference = verify_inference(stage)
    rebuilt = read_record(stage, REFERENCE_RECORD)
    for key in ("params", "rows", "representation", "categorical", "numeric", "ca_grid", "test_evaluated"):
        if reference[key] != rebuilt[key]:
            raise ValueError(f"Cambió el contrato de reconstrucción: {key}")
    verify_hashes(stage, rebuilt["output_sha256"])
    verify_experimental_chain(ROOT)  # Fuentes/entradas no cambiaron durante el ajuste.
    evidence = [run / "plan.json", run / "environment.txt", notebook_path, stage / REFERENCE_RECORD]
    evidence += [stage / "results/tables" / name for name in COMPARISON_TABLES]
    receipt = {"status": "verified_reconstruction", "created_utc": datetime.now(timezone.utc).isoformat(),
               "run_id": run_id, "historical_records_sha256": plan["historical_records_sha256"],
               "python_original": reference["python"], "python_reconstruction": platform.python_version(),
               "versions": rebuilt["versions"], "numerical_comparison": numerical, "inference_verification": inference,
               "selected_candidate": selection["selected_candidate"], "test_evaluated": False,
               "artifact_sha256": {p: sha256_file(stage / p) for p in ARTIFACTS},
               "original_artifact_sha256": {p: reference["output_sha256"][p] for p in ARTIFACTS},
               "reconstruction_evidence_sha256": {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in evidence},
               "recovery_script_sha256": sha256_file(Path(__file__)),
               "note": "Nueva corrida con equivalencia numérica verificada; no identidad binaria ni nueva selección. Los registros originales 03/04 permanecen intactos."}
    publish(stage, run / "previous_artifacts", receipt)
    print("Cuatro artefactos publicados y verificados; originales anteriores respaldados.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Audita sin entrenar ni deserializar modelos")
    args = parser.parse_args()
    if args.verify_only:
        receipt = verify_reference_artifacts(ROOT)
        print(json.dumps({"status": receipt["status"], "selected_candidate": receipt["selected_candidate"],
                          "test_evaluated": False}, ensure_ascii=False, indent=2))
    else:
        reconstruct()


if __name__ == "__main__":
    main()
