"""Reproduce 05/06 con artefactos verificados y contrasta las tablas anteriores.

No reentrena. Conserva respaldo y revierte las salidas si falla la reproducción.
La tolerancia se fija antes de ejecutar y las etapas 03/04 quedan intactas.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fraud_cost.artifacts import (
    notebook_code_sha256, project_path, read_record, sha256_file, verify_hashes, verify_reference_artifacts,
)

STAGES = (
    ("05_lightgbm_calibracion_y_costos", "lightgbm_adopcion_config.json"),
    ("06_estabilidad_y_errores_economicos", "estabilidad_config.json"),
)
ATOL = RTOL = 1e-8


def main():
    if (ROOT / "results/tables/protocolo_final.json").exists():
        raise FileExistsError("Existe un protocolo congelado; no regenerar sus dependencias sobre las mismas rutas")
    import nbformat
    import pandas as pd
    from nbclient import NotebookClient
    verify_reference_artifacts(ROOT)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run = project_path(ROOT, "data/interim/validation_reproduction/" + run_id)
    backup = run / "previous"
    previous = {}
    expected_codes = {}
    records = {}
    for notebook, config in STAGES:
        record = read_record(ROOT, "results/tables/" + config)
        records[config] = record
        expected_codes[notebook] = record["notebook_code_sha256"]
        verify_hashes(ROOT, record["source_sha256"])
        if notebook_code_sha256(ROOT / "notebooks" / (notebook + ".ipynb")) != record["notebook_code_sha256"]:
            raise ValueError(f"Cambió el código de {notebook}")
        for path in ["notebooks/" + notebook + ".ipynb", "results/tables/" + config, *record["output_sha256"]]:
            previous[path] = project_path(ROOT, path).is_file()
    previous["results/tables/estabilidad_protocolo.json"] = True
    for path, existed in previous.items():
        if existed:
            target = backup / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(project_path(ROOT, path), target)
    plan = {"status": "planned_before_execution", "atol": ATOL, "rtol": RTOL,
            "notebook_code_sha256": expected_codes, "new_model_fitted": False,
            "new_calibrator_fitted": False, "test_evaluated": False}
    (run / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    comparisons = []
    def progress(cell, cell_index, **kwargs):
        if cell.cell_type == "code":
            print(f"Celda {cell_index} ejecutada", flush=True)
            for output in cell.get("outputs", []):
                if output.get("output_type") == "stream": print(output.text[-1000:], flush=True)
    try:
        for notebook, config in STAGES:
            path = ROOT / "notebooks" / (notebook + ".ipynb")
            print("Ejecutando", notebook, flush=True)
            content = nbformat.read(path, as_version=4)
            for cell in content.cells:
                if cell.cell_type == "code": cell.outputs = []; cell.execution_count = None
            client = NotebookClient(content, timeout=1200, kernel_name="ieee-cis-pi2",
                                    resources={"metadata": {"path": str(ROOT / "notebooks")}},
                                    on_cell_executed=progress)
            client.execute()
            nbformat.write(content, path)
            if notebook_code_sha256(path) != expected_codes[notebook]:
                raise ValueError("Cambió una fuente ejecutable durante la reproducción")
            for relative in records[config]["output_sha256"]:
                if relative.endswith(".csv"):
                    a, b = pd.read_csv(backup / relative), pd.read_csv(ROOT / relative)
                    pd.testing.assert_frame_equal(a, b, check_exact=False, atol=ATOL, rtol=RTOL)
                    comparisons.append({"table": relative, "rows": len(b), "passed": True})
            current = read_record(ROOT, "results/tables/" + config)
            if current["test_evaluated"] or current["new_model_fitted"] or current["new_calibrator_fitted"]:
                raise ValueError("La reproducción excedió el alcance diagnóstico")
            verify_hashes(ROOT, current["output_sha256"])
    except BaseException:
        for relative, existed in previous.items():
            target = project_path(ROOT, relative)
            if existed: shutil.copy2(backup / relative, target)
            elif target.exists(): target.unlink()
        raise
    result = {**plan, "status": "complete", "run_id": run_id, "comparisons": comparisons,
              "historical_table_values_preserved": True,
              "new_records_sha256": {config: sha256_file(ROOT / "results/tables" / config) for _, config in STAGES}}
    (run / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("05/06 reproducidos; todas las tablas coinciden con la evidencia previa.", flush=True)


if __name__ == "__main__": main()
