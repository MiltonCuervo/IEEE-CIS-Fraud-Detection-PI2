"""Reproducción aislada previa a test; conserva la congelación del otro equipo."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd

from fraud_cost.artifacts import (ARTIFACTS, notebook_code_sha256, project_path, read_record,
                                 sha256_file, verify_hashes, verify_reference_artifacts)
from fraud_cost.final_protocol import PROTOCOL, RECORD, AUDIT, DECISIONS, verify_frozen_protocol

BINDING = 'results/tables/protocolo_ejecucion_local_v2.json'
TEST_RECORD = 'results/tables/evaluacion_test_config.json'
NOTEBOOK = 'notebooks/08_evaluacion_temporal_final.ipynb'
ATOL = RTOL = 1e-8


def semantic_equivalence(parent, local):
    """Compara decisiones completas; permite únicamente procedencia y ruido numérico."""
    a, b = deepcopy(parent), deepcopy(local)
    for value in (a, b):
        for key in ('created_utc', 'dependency_sha256', 'notebook_code_sha256'):
            value.pop(key, None)
    coefficients = []
    for key in ('coefficient_a', 'intercept_b'):
        x, y = a['calibration'].pop(key), b['calibration'].pop(key)
        np.testing.assert_allclose(x, y, atol=ATOL, rtol=RTOL)
        coefficients.append(abs(x-y))
    if a != b:
        raise ValueError('La reproducción modifica decisiones congeladas, no solo procedencia')
    return {'all_decisions_equal': True, 'calibration_max_absolute_difference': max(coefficients),
            'atol': ATOL, 'rtol': RTOL}


def verify_workspace(root):
    root = Path(root)
    binding = read_record(root, BINDING)
    if binding.get('status') != 'verified_pretest_reproduction':
        raise ValueError('Falta reproducción local verificada')
    verify_hashes(root, {PROTOCOL: binding['parent_protocol_sha256']})
    stage = project_path(root, binding['workspace'])
    verify_hashes(stage, {PROTOCOL: binding['local_protocol_sha256']})
    protocol = verify_frozen_protocol(stage)
    semantic_equivalence(read_record(root, PROTOCOL), protocol)
    verify_hashes(root, {p: protocol['dependency_sha256'][p] for p in (
        'src/fraud_cost/execution_workspace.py', 'src/fraud_cost/final_evaluation.py')})
    if notebook_code_sha256(root / NOTEBOOK) != binding['evaluator_notebook_code_sha256']:
        raise ValueError('Cambió el evaluador posterior a la reproducción')
    if not binding['tables_equivalent'] or len(binding['comparisons']) != 12:
        raise ValueError('No se contrastaron las doce tablas de validación')
    return stage, protocol, binding


def prepare_workspace(root):
    """Ejecuta 05/06/07 en copia aislada antes de leer/puntuar test."""
    root = Path(root).resolve()
    if (root / BINDING).exists():
        return verify_workspace(root)
    if (root / TEST_RECORD).exists():
        raise FileExistsError('Ya existe ejecución final; no preparar otra versión automáticamente')
    original = verify_reference_artifacts(root)
    parent = read_record(root, PROTOCOL)
    parent_record = read_record(root, RECORD)
    verify_hashes(root, parent_record['output_sha256'])
    if parent_record['test_evaluated'] or parent['test_evaluated']:
        raise ValueError('El protocolo no es previo a test')
    # Todo cambio de fuentes debe ser exclusivamente de fin de línea.
    differences = []
    for relative, expected in parent['dependency_sha256'].items():
        path = project_path(root, relative)
        if not path.is_file() or sha256_file(path) != expected:
            reason = 'local_artifact_original_vs_reconstructed'
            if relative.startswith('src/'):
                import hashlib
                raw = path.read_bytes().replace(b'\r\n', b'\n')
                variants = (raw, raw.replace(b'\n', b'\r\n'))
                if expected not in [hashlib.sha256(v).hexdigest() for v in variants]:
                    raise ValueError(f'Fuente distinta de fin de línea: {relative}')
                reason = 'line_endings_only'
            elif relative not in {*ARTIFACTS, 'results/models/lightgbm_seleccionado_platt.joblib',
                                  'data/interim/lightgbm_validation_scores.parquet',
                                  'data/interim/lightgbm_validation_actions.parquet'}:
                raise ValueError(f'Dependencia no reproducible sin revisar: {relative}')
            differences.append({'path': relative, 'reason': reason, 'parent_sha256': expected,
                                'local_sha256': sha256_file(path)})
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    stage = project_path(root, f'data/interim/final_evaluation_v2/{run_id}/workspace')
    stage.mkdir(parents=True, exist_ok=False)
    protected = {p: sha256_file(root/p) for p in parent['dependency_sha256']}
    protected.update({p: sha256_file(root/p) for p in (PROTOCOL, RECORD, AUDIT, DECISIONS)})
    planned = {'status': 'planned_before_reproduction', 'atol': ATOL, 'rtol': RTOL,
               'parent_protocol_sha256': sha256_file(root/PROTOCOL), 'differences': differences,
               'test_evaluated': False, 'new_model_fitted': False, 'new_calibrator_fitted': False}
    (stage.parent/'plan.json').write_text(json.dumps(planned, ensure_ascii=False, indent=2), encoding='utf-8')
    files = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
    excluded = {PROTOCOL, RECORD, AUDIT, DECISIONS, BINDING, TEST_RECORD}
    files = {p for p in files if p and p not in excluded}
    files.update(p.relative_to(root).as_posix() for p in (root/'src/fraud_cost').glob('*.py'))
    files.add(NOTEBOOK)
    for relative in files:
        source = project_path(root, relative)
        if source.is_file():
            destination = project_path(stage, relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    # Copias de los cuatro originales verificados; los CSV originales se enlazan para lectura.
    local_files = [*ARTIFACTS, 'data/interim/split_manifest.parquet',
                   'results/models/lightgbm_seleccionado_platt.joblib',
                   'data/interim/lightgbm_validation_scores.parquet', 'data/interim/lightgbm_validation_actions.parquet']
    for relative in local_files:
        destination = project_path(stage, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root/relative, destination)
    for name in ('train_transaction.csv', 'train_identity.csv'):
        destination = stage/'data/raw'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(root/'data/raw'/name, destination)
    import nbformat
    from nbclient import NotebookClient
    comparisons = []
    for number, name, config in [
        (5, '05_lightgbm_calibracion_y_costos', 'lightgbm_adopcion_config.json'),
        (6, '06_estabilidad_y_errores_economicos', 'estabilidad_config.json'),
        (7, '07_protocolo_final', None)]:
        previous = read_record(root, 'results/tables/'+config) if config else None
        print(f'Reproducción aislada: notebook {number}', flush=True)
        path = stage/'notebooks'/(name+'.ipynb')
        nb = nbformat.read(path, as_version=4)
        NotebookClient(nb, timeout=1800, kernel_name='ieee-cis-pi2',
                       resources={'metadata': {'path': str(stage/'notebooks')}}).execute()
        nbformat.write(nb, path)
        if notebook_code_sha256(path) != notebook_code_sha256(root/'notebooks'/(name+'.ipynb')):
            raise ValueError('Cambió código durante la reproducción')
        if previous:
            for relative in previous['output_sha256']:
                if relative.endswith('.csv'):
                    left, right = pd.read_csv(root/relative), pd.read_csv(stage/relative)
                    pd.testing.assert_frame_equal(left, right, check_exact=False, atol=ATOL, rtol=RTOL)
                    comparisons.append({'table': relative, 'rows': len(right), 'passed': True})
    local = verify_frozen_protocol(stage)
    semantic = semantic_equivalence(parent, local)
    verify_hashes(root, protected)
    binding = {'status': 'verified_pretest_reproduction', 'version': 'local_v2', 'created_utc': datetime.now(timezone.utc).isoformat(),
               'workspace': stage.relative_to(root).as_posix(), 'parent_protocol_sha256': sha256_file(root/PROTOCOL),
               'local_protocol_sha256': sha256_file(stage/PROTOCOL),
               'local_protocol': local, 'semantic_equivalence': semantic, 'differences': differences,
               'comparisons': comparisons, 'tables_equivalent': len(comparisons)==12,
               'reference_artifacts': original['status'], 'new_model_fitted': False, 'new_calibrator_fitted': False,
               'test_evaluated': False, 'evaluator_notebook_code_sha256': notebook_code_sha256(root/NOTEBOOK)}
    with (root/BINDING).open('x', encoding='utf-8', newline='\n') as output:
        json.dump(binding, output, ensure_ascii=False, indent=2); output.write('\n')
    print('Doce tablas equivalentes y decisiones iguales; versión local congelada antes de test.', flush=True)
    return verify_workspace(root)
