"""Diagnóstico de validación guardada: tiempo, pérdidas y frontera económica.

No ajusta estimadores, no genera predicciones nuevas y no carga modelos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

from fraud_cost.artifacts import read_record, verify_hashes, notebook_code_sha256, verify_experimental_chain
from fraud_cost.costs import bmr_actions, evaluate_policies

POLICIES = ('fixed_0_5', 'bmr', 'cost_sensitive_tree')
SCORE_COLUMNS = ('TransactionID', 'TransactionDT', 'TransactionAmt', 'isFraud',
                 'probability_raw', 'probability_calibrated', 'action_fixed_0_5')


def validate_evidence(scores, actions, ca_grid):
    """Valida poblaciones y reglas; alinea acciones por ID, nunca por posición."""
    grid = tuple(float(c) for c in ca_grid)
    if not grid or len(set(grid)) != len(grid) or not all(np.isfinite(c) and c > 0 for c in grid):
        raise ValueError('Los escenarios deben ser positivos, finitos y únicos')
    if not set(SCORE_COLUMNS).issubset(scores) or scores.empty:
        raise ValueError('Faltan scores de validación')
    if scores.TransactionID.isna().any() or scores.TransactionID.duplicated().any():
        raise ValueError('IDs de scores nulos o duplicados')
    values = scores[['TransactionDT', 'TransactionAmt', 'probability_raw', 'probability_calibrated']].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (scores.TransactionAmt < 0).any():
        raise ValueError('Tiempo, monto o probabilidades inválidos')
    if not np.isin(scores.isFraud, [0, 1]).all() or ((values[:, 2:] < 0) | (values[:, 2:] > 1)).any():
        raise ValueError('Etiquetas o probabilidades fuera del contrato')
    fixed = np.where(scores.probability_calibrated >= .5, 'block', 'approve')
    if not np.array_equal(scores.action_fixed_0_5, fixed):
        raise ValueError('La regla fija no corresponde a Platt')
    required = {'TransactionID', 'administrative_cost_ca', *POLICIES}
    if not required.issubset(actions) or actions[list(required)].isna().any().any():
        raise ValueError('Faltan acciones o claves')
    if actions[['TransactionID', 'administrative_cost_ca']].duplicated().any():
        raise ValueError('Claves operación/Ca duplicadas')
    if set(actions.administrative_cost_ca) != set(grid) or len(actions) != len(scores) * len(grid):
        raise ValueError('Grid o cobertura de acciones incompatibles')
    aligned = {}
    for ca in grid:
        block = actions.loc[actions.administrative_cost_ca.eq(ca)]
        if len(block) != len(scores) or set(block.TransactionID) != set(scores.TransactionID):
            raise ValueError('IDs económicos distintos de los scores')
        block = block.set_index('TransactionID').loc[scores.TransactionID].reset_index()
        if not np.isin(block[list(POLICIES)].to_numpy(), ['approve', 'block']).all():
            raise ValueError('Acciones distintas de approve/block')
        if not np.array_equal(block.fixed_0_5, fixed):
            raise ValueError('Acción fija guardada incompatible')
        expected = bmr_actions(scores.probability_calibrated, scores.TransactionAmt, ca)
        if not np.array_equal(block.bmr, expected):
            raise ValueError('BMR guardada incompatible con pA>Ca')
        aligned[ca] = block
    return aligned


def load_validation_evidence(root):
    """Verifica la cadena 03→04→05 antes de leer scores, sin deserializar modelos."""
    root = __import__('pathlib').Path(root)
    verify_experimental_chain(root)
    record = read_record(root, 'results/tables/lightgbm_adopcion_config.json')
    if record.get('status') != 'complete' or record.get('model') != 'lightgbm' or record.get('test_evaluated') is not False:
        raise ValueError('La etapa 05 no está completa o corresponde a otro alcance')
    for key in ('input_sha256', 'source_sha256', 'output_sha256', 'historical_record_sha256'):
        verify_hashes(root, record[key])
    if notebook_code_sha256(root / 'notebooks/05_lightgbm_calibracion_y_costos.ipynb') != record['notebook_code_sha256']:
        raise ValueError('Cambió el código ejecutado en 05')
    scores = pd.read_parquet(root / 'data/interim/lightgbm_validation_scores.parquet')
    scores = scores.sort_values(['TransactionDT', 'TransactionID']).reset_index(drop=True)
    actions = pd.read_parquet(root / 'data/interim/lightgbm_validation_actions.parquet')
    aligned = validate_evidence(scores, actions, record['ca_grid'])
    window = record['validation_windows']
    if len(scores) != window['policy_rows'] or scores.TransactionDT.min() != window['policy_time_start'] or scores.TransactionDT.max() != window['policy_time_end']:
        raise ValueError('La población difiere de validación posterior del 05')
    manifest = pd.read_parquet(root / 'data/interim/split_manifest.parquet', columns=['TransactionID', 'split'])
    membership = scores[['TransactionID']].merge(manifest, on='TransactionID', how='left', validate='one_to_one')
    if not membership.split.eq('validation').all() or scores.TransactionAmt.dtype != np.dtype('float64'):
        raise ValueError('Filas externas o pérdida de precisión del monto original')
    return scores, aligned, record


def temporal_periods(scores, count=3):
    """Igual duración sobre tiempo observado; límites completos, sin usar etiquetas."""
    if not isinstance(count, int) or count < 2 or scores.empty:
        raise ValueError('Se requieren al menos dos subperiodos')
    time = scores.TransactionDT.to_numpy(dtype=float)
    if not np.isfinite(time).all() or time.max() <= time.min():
        raise ValueError('Tiempo sin duración positiva')
    edges = np.linspace(time.min(), time.max(), count + 1)
    periods = np.searchsorted(edges[1:-1], time, side='right') + 1
    if set(periods) != set(range(1, count + 1)):
        raise ValueError('Algún subperiodo quedó vacío')
    data = scores.reset_index(drop=True).assign(period=periods)
    rows = []
    for period, group in data.groupby('period', sort=True):
        i = int(period) - 1
        rows.append({'period': int(period), 'boundary_start': edges[i], 'boundary_end': edges[i+1],
                     'end_inclusive': period == count, 'time_start': int(group.TransactionDT.min()),
                     'time_end': int(group.TransactionDT.max()), 'rows': len(group),
                     'frauds': int(group.isFraud.sum()), 'fraud_rate': float(group.isFraud.mean()),
                     'fraud_amount': float((group.isFraud * group.TransactionAmt).sum())})
    return data, pd.DataFrame(rows)


def period_probabilities(data):
    rows = []
    for period, group in data.groupby('period', sort=True):
        y = group.isFraud.to_numpy()
        for stage, column in [('raw', 'probability_raw'), ('platt', 'probability_calibrated')]:
            p = group[column].to_numpy()
            blocked = p >= .5
            fraud, legit = y == 1, y == 0
            tp, fp = int((fraud & blocked).sum()), int((legit & blocked).sum())
            rows.append({'period': int(period), 'probability': stage, 'rows': len(group),
                         'frauds': int(fraud.sum()), 'fraud_rate': float(y.mean()), 'mean_probability': float(p.mean()),
                         'average_precision': float(average_precision_score(y, p)) if np.unique(y).size == 2 else np.nan,
                         'brier_score': float(brier_score_loss(y, p)), 'log_loss': float(log_loss(y, p, labels=[0, 1])),
                         'recall_at_0_5': tp / fraud.sum() if fraud.any() else np.nan,
                         'precision_at_0_5': tp / (tp + fp) if tp + fp else 0.,
                         'legitimate_block_rate': fp / legit.sum() if legit.any() else np.nan,
                         'block_count': int(blocked.sum())})
    return pd.DataFrame(rows)


def period_costs(data, aligned):
    rows = []
    for ca, actions in aligned.items():
        for period, group in data.groupby('period', sort=True):
            # group.index identifica posiciones del dataframe de scores validado.
            policies = {p: actions.iloc[group.index][p].to_numpy() for p in POLICIES}
            table = evaluate_policies(group.isFraud, group.TransactionAmt, policies, ca)
            table['period'] = int(period)
            rows.append(table)
    return pd.concat(rows, ignore_index=True)


def period_amount_calibration(data, edges):
    bands = pd.cut(data.TransactionAmt, edges, right=False, include_lowest=True)
    if bands.isna().any():
        raise ValueError('Las bandas deben cubrir los montos')
    rows = []
    for (period, band), group in data.assign(amount_band=bands).groupby(['period', 'amount_band'], observed=True):
        rows.append({'period': int(period), 'amount_band': str(band), 'rows': len(group),
                     'frauds': int(group.isFraud.sum()), 'observed_rate': float(group.isFraud.mean()),
                     'mean_raw': float(group.probability_raw.mean()), 'mean_platt': float(group.probability_calibrated.mean()),
                     'observed_fraud_amount': float((group.isFraud * group.TransactionAmt).sum()),
                     'expected_fraud_amount_raw': float((group.probability_raw * group.TransactionAmt).sum()),
                     'expected_fraud_amount_platt': float((group.probability_calibrated * group.TransactionAmt).sum())})
    return pd.DataFrame(rows)


def missed_loss_concentration(scores, aligned, top_counts=(1, 5, 10)):
    if not top_counts or any(not isinstance(k, int) or k < 1 for k in top_counts) or len(set(top_counts)) != len(top_counts):
        raise ValueError('Los tamaños superiores deben ser enteros positivos únicos')
    rows = []
    for ca, actions in aligned.items():
        for policy in POLICIES:
            mask = scores.isFraud.eq(1).to_numpy() & actions[policy].eq('approve').to_numpy()
            missed = scores.loc[mask].sort_values(['TransactionAmt', 'TransactionID'], ascending=[False, True])
            total = float(missed.TransactionAmt.sum())
            for k in top_counts:
                amount = float(missed.head(k).TransactionAmt.sum())
                rows.append({'administrative_cost_ca': ca, 'strategy': policy, 'missed_frauds': len(missed),
                             'missed_fraud_amount': total, 'top_k_requested': k, 'top_k_available': min(k, len(missed)),
                             'top_k_missed_amount': amount, 'fraction_of_missed_amount': amount / total if total > 0 else np.nan})
    return pd.DataFrame(rows)


def boundary_sensitivity(scores, aligned, relative_band=.10, factors=(.9, 1., 1.1)):
    """Perturbación relativa de p, recortada a [0,1]; no es otro calibrador."""
    factors = tuple(float(f) for f in factors)
    if not 0 < relative_band < 1 or len(set(factors)) != len(factors) or 1. not in factors or any(not np.isfinite(f) or f <= 0 for f in factors):
        raise ValueError('Banda o factores inválidos; incluir el escenario base 1')
    y, amount = scores.isFraud.to_numpy(), scores.TransactionAmt.to_numpy()
    p = scores.probability_calibrated.to_numpy()
    boundary_rows, sensitivity = [], []
    for ca, actions in aligned.items():
        q = p * amount / ca
        near = (q >= 1 - relative_band) & (q <= 1 + relative_band)
        base = actions.bmr.eq('block').to_numpy()
        raw = bmr_actions(scores.probability_raw, amount, ca) == 'block'
        changed = raw != base
        boundary_rows.append({'administrative_cost_ca': ca, 'rows': len(scores), 'relative_band': relative_band,
                              'near_rows': int(near.sum()), 'near_fraction': float(near.mean()),
                              'near_frauds': int(((y == 1) & near).sum()), 'near_legitimate': int(((y == 0) & near).sum()),
                              'exact_ties': int((q == 1).sum()), 'raw_platt_changed_actions': int(changed.sum()),
                              'raw_platt_new_blocks': int((~raw & base).sum()), 'raw_platt_new_approvals': int((raw & ~base).sum())})
        baseline_cost = None
        pending = []
        for factor in factors:
            perturbed = np.clip(p * factor, 0, 1)
            bmr = bmr_actions(perturbed, amount, ca)
            result = evaluate_policies(y, amount, {'fixed_0_5': actions.fixed_0_5, 'bmr': bmr}, ca)
            row = result.loc[result.strategy.eq('bmr')].iloc[0].to_dict()
            block = bmr == 'block'
            row.update({'probability_factor': factor, 'changed_actions': int((block != base).sum()),
                        'new_blocks': int((block & ~base).sum()), 'new_approvals': int((~block & base).sum())})
            if factor == 1.:
                baseline_cost = row['cost_total_proxy']
            pending.append(row)
        for row in pending:
            row['cost_difference_vs_unperturbed'] = row['cost_total_proxy'] - baseline_cost
        sensitivity.extend(pending)
    return pd.DataFrame(boundary_rows), pd.DataFrame(sensitivity)
