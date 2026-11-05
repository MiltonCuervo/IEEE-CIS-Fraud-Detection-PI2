"""Revalida un 03 reconstruido después de una nueva selección 04, sin fit.

El 03 no depende del ranking posterior. Reutilizar sus binarios exige que
su evidencia completa siga íntegra y que la selección coincida con un ajuste
de referencia. Se conserva el recibo anterior y se repite la verificación.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fraud_cost.artifacts import (
    ARTIFACTS, RECOVERY_RECORD, REFERENCE_RECORD, SELECTION_PROTOCOL, SELECTION_RECORD,
    notebook_code_sha256, project_path, read_record, sha256_file, verify_experimental_chain,
    verify_hashes, verify_reference_artifacts,
)
from recover_reference_artifacts import compare_tables, verify_inference, write_json


def validate_binding(reference, selection, previous, reference_hash):
    if previous.get("status") != "verified_reconstruction" or previous.get("test_evaluated") is not False:
        raise ValueError("El recibo previo no corresponde a una reconstrucción verificada")
    if previous["historical_records_sha256"][REFERENCE_RECORD] != reference_hash:
        raise ValueError("Cambió la corrida 03: no es una actualización del vínculo con 04")
    if set(previous["artifact_sha256"]) != set(ARTIFACTS):
        raise ValueError("Faltan artefactos de la reconstrucción previa")
    if previous["original_artifact_sha256"] != {p: reference["output_sha256"][p] for p in ARTIFACTS}:
        raise ValueError("Los originales de referencia cambiaron")
    chosen = selection["selected_candidate"]
    if reference["params"].get(chosen["model"]) != chosen["params"]:
        raise ValueError("La configuración elegida requiere un nuevo ajuste")
    if not previous["numerical_comparison"]["passed"] or not previous["inference_verification"]["passed"]:
        raise ValueError("Falta evidencia de equivalencia del ajuste anterior")


def revalidate(root):
    root = Path(root)
    if (root / "results/tables/protocolo_final.json").exists():
        raise FileExistsError("Existe un protocolo congelado; una reproducción nueva requiere otra área y versión de experimento")
    reference, selection = verify_experimental_chain(root)
    previous = read_record(root, RECOVERY_RECORD)
    validate_binding(reference, selection, previous, sha256_file(root / REFERENCE_RECORD))
    verify_hashes(root, previous["artifact_sha256"])
    verify_hashes(root, previous["reconstruction_evidence_sha256"])
    records = [p for p in previous["reconstruction_evidence_sha256"]
               if p.endswith("/workspace/" + REFERENCE_RECORD)]
    if len(records) != 1:
        raise ValueError("No hay una evidencia única de la corrida 03 reconstruida")
    stage = project_path(root, records[0]).parents[2]
    rebuilt = read_record(stage, REFERENCE_RECORD)
    for key in ("params", "rows", "representation", "categorical", "numeric", "ca_grid", "test_evaluated", "source_sha256", "input_sha256"):
        if rebuilt[key] != reference[key]:
            raise ValueError(f"El ajuste reconstruido tiene otro contrato: {key}")
    if notebook_code_sha256(stage / "notebooks/03_modelado_y_decisiones_costos.ipynb") != reference["notebook_executed_code_sha256"]:
        raise ValueError("El código reconstruido del 03 difiere")
    verify_hashes(stage, rebuilt["output_sha256"])
    numerical = compare_tables(root / "results/tables", stage / "results/tables")
    inference = verify_inference(stage)
    timestamp = datetime.now(timezone.utc)
    run = project_path(root, "data/interim/reference_recovery/revalidation_" + timestamp.strftime("%Y%m%dT%H%M%S%fZ"))
    run.mkdir(parents=True)
    original_bytes = (root / RECOVERY_RECORD).read_bytes()
    backup = run / "previous_receipt.json"
    backup.write_bytes(original_bytes)
    note = run / "revalidation.json"
    current = {p: sha256_file(root / p) for p in (REFERENCE_RECORD, SELECTION_PROTOCOL, SELECTION_RECORD)}
    write_json(note, {"created_utc": timestamp.isoformat(), "previous_receipt_sha256": sha256_file(backup),
                     "previous_historical_records_sha256": previous["historical_records_sha256"],
                     "current_historical_records_sha256": current, "new_model_fitted": False,
                     "new_calibrator_fitted": False, "test_evaluated": False,
                     "reason": "El 04 se volvió a ejecutar para ampliar diagnósticos; 03 y la configuración adoptable permanecen verificados."})
    receipt = deepcopy(previous)
    receipt.update(historical_records_sha256=current, selected_candidate=selection["selected_candidate"],
                   numerical_comparison=numerical, inference_verification=inference)
    receipt["revalidation"] = {"created_utc": timestamp.isoformat(), "script_sha256": sha256_file(Path(__file__)),
                               "previous_receipt_path": backup.relative_to(root).as_posix(),
                               "new_model_fitted": False, "new_calibrator_fitted": False, "test_evaluated": False}
    for path in (backup, note):
        receipt["reconstruction_evidence_sha256"][path.relative_to(root).as_posix()] = sha256_file(path)
    pending = (root / RECOVERY_RECORD).with_suffix(".json.revalidating")
    try:
        write_json(pending, receipt)
        pending.replace(root / RECOVERY_RECORD)
        verify_reference_artifacts(root)
    except BaseException:
        (root / RECOVERY_RECORD).write_bytes(original_bytes)
        raise
    print("Reconstrucción previa revalidada contra 03/04 vigentes; sin nuevos ajustes.")


if __name__ == "__main__":
    revalidate(ROOT)
