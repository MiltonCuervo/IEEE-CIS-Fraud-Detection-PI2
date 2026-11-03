"""Auditoría sin deserializar modelos: evidencia histórica y reconstrucción local.

Una reconstrucción tiene sus propios hashes. No sustituye los hashes originales
de 03/04 ni convierte equivalencia numérica en identidad binaria.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


REFERENCE_RECORD = "results/tables/comparacion_modelos_config.json"
SELECTION_PROTOCOL = "results/tables/seleccion_ordinal_protocolo.json"
SELECTION_RECORD = "results/tables/seleccion_ordinal_config.json"
RECOVERY_RECORD = "results/tables/recuperacion_03_config.json"
ARTIFACTS = (
    "data/interim/validation_scores.parquet",
    "results/models/baseline_xgboost_platt.joblib",
    "results/models/comparacion_modelos_ordinal.joblib",
    "results/models/cost_sensitive_trees.joblib",
)


def project_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Ruta fuera del proyecto: {relative}")
    return path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_record(root, relative):
    return json.loads(project_path(root, relative).read_text(encoding="utf-8"))


def notebook_code_sha256(path):
    notebook = json.loads(Path(path).read_text(encoding="utf-8"))
    code = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    return hashlib.sha256(json.dumps(code, ensure_ascii=False).encode()).hexdigest()


def verify_hashes(root, expected):
    for relative, digest in expected.items():
        path = project_path(root, relative)
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Archivo ausente o hash distinto: {relative}")


def verify_experimental_chain(root):
    """Comprueba la cadena 03→04 original, sin alterar sus registros ni ajustar."""
    reference = read_record(root, REFERENCE_RECORD)
    protocol = read_record(root, SELECTION_PROTOCOL)
    selection = read_record(root, SELECTION_RECORD)
    verify_hashes(root, {**reference["source_sha256"], **reference["input_sha256"]})
    # Las salidas versionadas también deben pertenecer a esa corrida del 03.
    verify_hashes(root, {p: h for p, h in reference["output_sha256"].items() if p not in ARTIFACTS})
    if notebook_code_sha256(project_path(root, "notebooks/03_modelado_y_decisiones_costos.ipynb")) != reference["notebook_executed_code_sha256"]:
        raise ValueError("Cambió el código ejecutable de la etapa 03")
    verify_hashes(root, {REFERENCE_RECORD: protocol["reference_record_sha256"],
                         SELECTION_PROTOCOL: selection["protocol_sha256"],
                         "results/tables/seleccion_ordinal_ventanas.csv": protocol["windows_sha256"]})
    verify_hashes(root, {**protocol["source_sha256"], **protocol["input_sha256"], **selection["output_sha256"]})
    code_hash = notebook_code_sha256(project_path(root, "notebooks/04_seleccion_temporal_modelos.ipynb"))
    if code_hash != protocol["notebook_code_sha256"] or code_hash != selection["notebook_code_sha256"]:
        raise ValueError("Cambió el código ejecutable de la etapa 04")
    if selection["status"] != "complete" or selection["fit_count"] != protocol["expected_fits"]:
        raise ValueError("La selección temporal no está completa")
    if reference["test_evaluated"] or selection["test_evaluated"] or protocol["test_evaluated"]:
        raise ValueError("La cadena esperada mantiene test sin puntuar")
    return reference, selection


def verify_reference_artifacts(root):
    """Valida antes de cargar joblib; acepta originales o reconstrucción auditada.

    El recibo de reconstrucción es obligatorio si cambian los bytes originales.
    Los hashes y tablas comparativas se verifican sin ejecutar inferencia.
    """
    reference, selection = verify_experimental_chain(root)
    originals = all(project_path(root, p).is_file() and sha256_file(project_path(root, p)) == reference["output_sha256"][p]
                    for p in ARTIFACTS)
    if originals:
        return {"status": "verified_original_artifacts", "selected_candidate": selection["selected_candidate"],
                "artifact_sha256": {p: reference["output_sha256"][p] for p in ARTIFACTS}}
    receipt = read_record(root, RECOVERY_RECORD)
    if receipt.get("status") != "verified_reconstruction" or receipt.get("test_evaluated") is not False:
        raise ValueError("La reconstrucción no está verificada")
    verify_hashes(root, receipt["historical_records_sha256"])
    if receipt["historical_records_sha256"] != {p: sha256_file(project_path(root, p)) for p in (REFERENCE_RECORD, SELECTION_PROTOCOL, SELECTION_RECORD)}:
        raise ValueError("La reconstrucción corresponde a otra cadena experimental")
    if set(receipt["artifact_sha256"]) != set(ARTIFACTS):
        raise ValueError("El recibo no contiene los cuatro artefactos esperados")
    if not receipt["numerical_comparison"]["passed"] or not receipt["inference_verification"]["passed"]:
        raise ValueError("La equivalencia numérica o de inferencia no fue verificada")
    verify_hashes(root, receipt["artifact_sha256"])
    verify_hashes(root, receipt["reconstruction_evidence_sha256"])
    return receipt
