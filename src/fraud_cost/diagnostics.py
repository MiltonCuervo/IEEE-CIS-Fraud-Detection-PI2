"""Diagnósticos descriptivos sobre scores y acciones congelados; no ajusta modelos."""
from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from fraud_cost.costs import bmr_actions, evaluate_policies, realized_cost
from fraud_cost.economics import checked_ca_grid
from fraud_cost.refit import probabilities, source_sha256
from fraud_cost.scenarios import validate_validation_scores

# Cortes descriptivos fijados antes de inspeccionar las tablas de esta etapa.
AMOUNT_EDGES = (0., 50., 100., 250., 500., 1000., np.inf)
RISK_EDGES = (0., .01, .05, .10, .25, .50, 1.)
MARGIN_EDGES = (0., .5, .9, 1.1, 2., np.inf)
N_PERIODS = 4


def band_labels(edges):
    return [f"[{a:g}, {b:g}{']' if np.isfinite(b) and i == len(edges)-2 else ')'}"
            for i, (a, b) in enumerate(zip(edges, edges[1:]))]


def bands(values, edges):
    """Intervalos cerrados por izquierda; último límite finito incluido."""
    x = np.asarray(values, dtype=float)
    e = np.asarray(edges, dtype=float)
    if (x.ndim != 1 or not np.isfinite(x).all() or len(e) < 2 or
            np.isnan(e).any() or not np.all(np.diff(e) > 0) or
            (x < e[0]).any() or (x > e[-1]).any()):
        raise ValueError("Valores o límites de bandas inválidos")
    return np.searchsorted(e[1:-1], x, side="right")


def checked_scores(scores):
    validate_validation_scores(scores)
    if not {"TransactionDT", "probability_raw"}.issubset(scores.columns):
        raise ValueError("Faltan tiempos o probabilidades originales")
    times = scores.TransactionDT.to_numpy(dtype=float)
    if not np.isfinite(times).all() or (times < 0).any():
        raise ValueError("TransactionDT debe ser finito y no negativo")
    probabilities(scores.probability_raw)
    return scores.reset_index(drop=True).copy()


def checked_actions(scores, actions, ca_grid):
    if (len(actions) != len(scores) or "TransactionID" not in actions or
            not np.array_equal(actions.TransactionID, scores.TransactionID)):
        raise ValueError("Las acciones deben estar alineadas por ID con los scores")
    required = ["fixed_0_5"]
    for ca in ca_grid:
        required += [f"{name}_ca_{ca:g}" for name in ("bmr", "tree", "raw_bmr")]
    for column in required:
        if column not in actions or not pd.api.types.is_bool_dtype(actions[column]) or actions[column].isna().any():
            raise ValueError(f"Falta una acción booleana válida: {column}")
    if not np.array_equal(actions.fixed_0_5, scores.probability_calibrated >= .5):
        raise ValueError("La política fija no coincide con sus probabilidades")
    for ca in ca_grid:
        for prefix, probability in (("bmr", scores.probability_calibrated), ("raw_bmr", scores.probability_raw)):
            expected = bmr_actions(probability, scores.TransactionAmt, ca) == "block"
            if not np.array_equal(actions[f"{prefix}_ca_{ca:g}"], expected):
                raise ValueError("Una acción BMR no coincide con la regla congelada")


def verified_diagnostic_inputs(root):
    """Comprueba procedencia 06/07 sin cargar modelos ni puntuar otras filas."""
    root = Path(root)
    records = []
    for name in ("reajuste_config.json", "comparacion_economica_config.json"):
        record = json.loads((root / "results/tables" / name).read_text(encoding="utf-8"))
        if record.get("test_evaluated") is not False:
            raise ValueError("Estas fuentes deben conservar test sin puntuar")
        for group in ("source_sha256", "artifact_sha256"):
            if not record.get(group):
                raise ValueError("Falta trazabilidad de las fuentes")
            for relative, expected in record[group].items():
                path = root / relative
                if source_sha256(path, code_only=path.suffix == ".ipynb") != expected:
                    raise ValueError(f"Cambió una fuente previa: {relative}")
        records.append(record)
    refit, economic = records
    if economic.get("predictor_or_calibrator_refitted") is not False:
        raise ValueError("La etapa 07 debe usar el predictor congelado")
    grid = checked_ca_grid(economic["ca_grid"])
    scores = checked_scores(pd.read_parquet(root / "data/interim/validation_scores_selected.parquet"))
    actions = pd.read_parquet(root / "data/interim/decisiones_economicas_validation.parquet")
    window = next(w for w in refit["windows"] if w["block"] == "validation_policy")
    if (len(scores) != window["rows"] or scores.TransactionDT.min() != window["time_start"] or
            scores.TransactionDT.max() != window["time_end"] or int(scores.isFraud.sum()) != window["frauds"]):
        raise ValueError("Los scores no corresponden a validación posterior")
    checked_actions(scores, actions, grid)
    return scores, actions, economic


def probability_summary(frame, probability):
    """La suma p*A y la suma y*A incluyen todas las filas, no pérdidas de una política."""
    y = frame.isFraud.to_numpy(dtype=float)
    a = frame.TransactionAmt.to_numpy(dtype=float)
    p = frame[probability].to_numpy(dtype=float)
    n = len(frame)
    observed = float(y.mean()) if n else np.nan
    mean = float(p.mean()) if n else np.nan
    return {"rows": n, "frauds": int(y.sum()), "observed_rate": observed,
            "mean_probability": mean, "observed_minus_predicted_rate": observed - mean,
            "brier_score": float(np.mean((p-y)**2)) if n else np.nan,
            "amount_total": float(a.sum()), "fraud_amount_observed": float((y*a).sum()),
            "fraud_amount_expected": float((p*a).sum()),
            "observed_minus_expected_fraud_amount": float(((y-p)*a).sum())}


def conditional_calibration(scores):
    """Mismos grupos para raw/calibrado, incluidos grupos vacíos; sin reajuste."""
    scores = checked_scores(scores)
    amounts = bands(scores.TransactionAmt, AMOUNT_EDGES)
    risks = bands(scores.probability_calibrated, RISK_EDGES)
    alabels, rlabels = band_labels(AMOUNT_EDGES), band_labels(RISK_EDGES)
    groups = [("amount", i, -1, amounts == i) for i in range(len(alabels))]
    groups += [("risk", -1, j, risks == j) for j in range(len(rlabels))]
    groups += [("amount_risk", i, j, (amounts == i) & (risks == j))
               for i in range(len(alabels)) for j in range(len(rlabels))]
    records = []
    for grouping, i, j, mask in groups:
        part = scores.loc[mask]
        for name in ("probability_raw", "probability_calibrated"):
            records.append({"grouping": grouping, "amount_band_index": i, "risk_band_index": j,
                            "amount_band": alabels[i] if i >= 0 else "all",
                            "risk_band": rlabels[j] if j >= 0 else "all", "probabilities": name,
                            **probability_summary(part, name)})
    return pd.DataFrame(records)


def boundary_diagnostic(scores, *, ca_grid):
    """Bandas de q=p_cal*A/Ca; [0.9,1.1) solo designa proximidad descriptiva."""
    scores = checked_scores(scores)
    a, y = scores.TransactionAmt.to_numpy(float), scores.isFraud.to_numpy()
    p, raw = scores.probability_calibrated.to_numpy(float), scores.probability_raw.to_numpy(float)
    records = []
    for ca in checked_ca_grid(ca_grid):
        q = p*a/ca
        membership = bands(q, MARGIN_EDGES)
        before = bmr_actions(raw, a, ca)
        after = bmr_actions(p, a, ca)
        old_cost, new_cost = realized_cost(y, a, before, ca), realized_cost(y, a, after, ca)
        for i, label in enumerate(band_labels(MARGIN_EDGES)):
            mask = membership == i
            old_block, new_block = before[mask] == "block", after[mask] == "block"
            n = int(mask.sum())
            records.append({"administrative_cost_ca": ca, "margin_band_index": i, "margin_band": label,
                            "near_boundary": i == 2, **probability_summary(scores.loc[mask], "probability_calibrated"),
                            "population_fraction": n/len(scores),
                            "amount_le_ca_count": int((a[mask] <= ca).sum()),
                            "calibrated_block_count": int(new_block.sum()),
                            "actions_changed": int((before[mask] != after[mask]).sum()),
                            "additional_interventions": int((~old_block & new_block).sum()),
                            "fewer_interventions": int((old_block & ~new_block).sum()),
                            "raw_bmr_cost": float(old_cost[mask].sum()), "calibrated_bmr_cost": float(new_cost[mask].sum()),
                            "calibrated_minus_raw_cost": float((new_cost[mask]-old_cost[mask]).sum()),
                            "administrative_cost_delta": float(ca*(new_block.sum()-old_block.sum())),
                            "missed_fraud_amount_delta": float((new_cost[mask]-old_cost[mask]).sum()-ca*(new_block.sum()-old_block.sum()))})
    return pd.DataFrame(records)


def temporal_diagnostic(scores, actions, *, ca_grid, n_periods=N_PERIODS):
    """Subperiodos de igual duración relativa; no rompe lotes simultáneos."""
    scores = checked_scores(scores)
    grid = checked_ca_grid(ca_grid)
    checked_actions(scores, actions, grid)
    if not isinstance(n_periods, int) or n_periods < 1:
        raise ValueError("Número de subperiodos inválido")
    times = scores.TransactionDT.to_numpy(float)
    start, stop = float(times.min()), float(times.max())
    if start == stop:
        raise ValueError("Se necesita una ventana con duración positiva")
    edges = np.linspace(start, stop, n_periods+1)
    membership = bands(times, edges)
    populations, comparisons = [], []
    for period in range(n_periods):
        mask = membership == period
        part = scores.loc[mask]
        metadata = {"period": period+1, "interval_start_dt": edges[period], "interval_end_dt": edges[period+1],
                    "relative_day_start": (edges[period]-start)/86400,
                    "relative_day_end": (edges[period+1]-start)/86400,
                    "time_start_observed": float(part.TransactionDT.min()) if len(part) else np.nan,
                    "time_end_observed": float(part.TransactionDT.max()) if len(part) else np.nan}
        for probability in ("probability_raw", "probability_calibrated"):
            populations.append({**metadata, "probabilities": probability, **probability_summary(part, probability)})
        if not len(part):
            continue
        for ca in grid:
            policies = {name: np.where(actions.loc[mask, column], "block", "approve") for name, column in
                        (("fixed_0_5", "fixed_0_5"), ("bmr", f"bmr_ca_{ca:g}"), ("cost_sensitive_tree", f"tree_ca_{ca:g}"))}
            table = evaluate_policies(part.isFraud, part.TransactionAmt, policies, ca)
            p, a = part.probability_calibrated.to_numpy(float), part.TransactionAmt.to_numpy(float)
            expected = {name: float(np.where(action == "block", ca, p*a).sum()) for name, action in policies.items()}
            table["expected_cost_common_probabilities"] = table.strategy.map(expected)
            table["realized_minus_expected_cost"] = table.cost_total_proxy-table.expected_cost_common_probabilities
            for key, value in metadata.items():
                table[key] = value
            comparisons.append(table)
    return pd.DataFrame(populations), pd.concat(comparisons, ignore_index=True)


def diagnostic_figures(calibration, temporal, *, example_ca=10.):
    """Dos figuras científicas; no exporta registros por transacción."""
    import matplotlib.pyplot as plt

    fig1, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    styles = (("probability_raw", "Original", "#7c6aa6"),
              ("probability_calibrated", "Calibrada", "#007f86"))
    for axis, grouping, label_column, title in ((axes[0], "amount", "amount_band", "Probabilidad por monto"),
                                               (axes[1], "risk", "risk_band", "Probabilidad por banda de riesgo")):
        part = calibration.loc[calibration.grouping.eq(grouping)]
        observed = part.loc[part.probabilities.eq("probability_calibrated")]
        x = np.arange(len(observed))
        axis.plot(x, observed.observed_rate, "o-", color="#222222", label="Fraude observado")
        for name, label, color in styles:
            axis.plot(x, part.loc[part.probabilities.eq(name), "mean_probability"], "o--", color=color, label=label)
        axis.set_xticks(x, observed[label_column], rotation=35, ha="right", fontsize=8)
        axis.set(title=title, ylabel="Tasa / probabilidad media")
        axis.grid(alpha=.2)
    part = calibration.loc[calibration.grouping.eq("amount")]
    observed = part.loc[part.probabilities.eq("probability_calibrated")]
    x = np.arange(len(observed))
    axes[2].plot(x, observed.fraud_amount_observed, "o-", color="#222222", label="Monto fraudulento observado")
    for name, label, color in styles:
        axes[2].plot(x, part.loc[part.probabilities.eq(name), "fraud_amount_expected"], "o--", color=color, label=label)
    axes[2].set_xticks(x, observed.amount_band, rotation=35, ha="right", fontsize=8)
    axes[2].set(title="Monto fraudulento: observado y esperado", ylabel="Unidades del dataset")
    axes[2].grid(alpha=.2)
    for axis in axes:
        axis.legend(fontsize=8)
    fig1.suptitle("Validación posterior: grupos descriptivos comunes, sin reajuste")
    fig1.tight_layout()

    fig2, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for strategy, label, color in (("fixed_0_5", "Fija 0,5", "#555555"), ("bmr", "BMR", "#007f86"),
                                    ("cost_sensitive_tree", "Árbol ponderado", "#c16a1b")):
        part = temporal.loc[temporal.administrative_cost_ca.eq(example_ca) & temporal.strategy.eq(strategy)]
        axes[0].plot(part.period, part.cost_per_transaction_proxy, "o-", color=color, label=label)
    axes[0].set(title=f"Costo por transacción — Ca={example_ca:g} hipotético", xlabel="Subperiodo relativo", ylabel="Unidades del dataset")
    axes[0].set_xticks(sorted(temporal.period.unique()))
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=.2)
    bmr = temporal.loc[temporal.strategy.eq("bmr")].copy()
    bmr["delta_per_transaction"] = -bmr.savings_vs_fixed_proxy/bmr.rows
    matrix = bmr.pivot(index="administrative_cost_ca", columns="period", values="delta_per_transaction")
    extent = max(float(np.abs(matrix.to_numpy()).max()), 1e-12)
    im = axes[1].imshow(matrix, cmap="RdBu_r", vmin=-extent, vmax=extent, aspect="auto")
    axes[1].set_xticks(range(len(matrix.columns)), matrix.columns)
    axes[1].set_yticks(range(len(matrix.index)), [f"{c:g}" for c in matrix.index])
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            value = matrix.iloc[i, j]
            axes[1].text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=9,
                         color="white" if abs(value) > extent*.55 else "black")
    axes[1].set(title="BMR − fija: costo por transacción", xlabel="Subperiodo relativo", ylabel="Ca hipotético")
    fig2.colorbar(im, ax=axes[1], label="Negativo: menor costo BMR")
    fig2.suptitle("Estabilidad dentro de validación: mismas políticas, sin nuevos ajustes")
    fig2.tight_layout()
    return {"diagnostico_probabilistico_validation.png": fig1, "estabilidad_economica_validation.png": fig2}


def save_diagnostics(root, calibration, boundary, populations, temporal, figures, *, ca_grid):
    root = Path(root)
    outputs = []
    for filename, frame in (("diagnostico_calibracion_condicional.csv", calibration),
                            ("diagnostico_frontera_bmr.csv", boundary),
                            ("diagnostico_subperiodos_validation.csv", populations),
                            ("diagnostico_costos_subperiodos.csv", temporal)):
        path = root / "results/tables" / filename
        frame.to_csv(path, index=False)
        outputs.append(path)
    for filename, figure in figures.items():
        path = root / "results/figures" / filename
        figure.savefig(path, dpi=160, bbox_inches="tight")
        outputs.append(path)
    sources = [root / "src/fraud_cost/diagnostics.py", root / "notebooks/08_diagnostico_y_estabilidad.ipynb",
               root / "results/tables/reajuste_config.json", root / "results/tables/comparacion_economica_config.json",
               root / "data/interim/validation_scores_selected.parquet", root / "data/interim/decisiones_economicas_validation.parquet"]
    record = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "evaluation_block": "validation_policy",
              "purpose": "diagnostico_descriptivo_no_seleccion", "test_evaluated": False,
              "models_refitted": False, "calibrator_selected": False, "costs_approved": False,
              "cost_units": "unidades_del_dataset_moneda_no_verificada", "ca_grid": checked_ca_grid(ca_grid),
              "amount_edges": ["inf" if np.isinf(v) else v for v in AMOUNT_EDGES], "risk_edges": RISK_EDGES,
              "margin_edges": ["inf" if np.isinf(v) else v for v in MARGIN_EDGES],
              "near_boundary_definition": "0.9 <= p_calibrada*TransactionAmt/Ca < 1.1; banda descriptiva, no error garantizado",
              "group_protocol": "cortes fijos; riesgo definido con p calibrada, mismos grupos para raw/cal",
              "temporal_protocol": "4 intervalos de igual duración entre mínimo y máximo DT posterior; último extremo incluido; sin dividir simultáneas",
              "uncertainty_note": "No hay intervalos ni pruebas inferenciales; filas temporalmente dependientes y validación reutilizada.",
              "reference_note": "La referencia trivial se recalcula en cada subperiodo; no sumar sus mínimos como referencia global.",
              "source_sha256": {p.relative_to(root).as_posix(): source_sha256(p, code_only=p.suffix == ".ipynb") for p in sources},
              "artifact_sha256": {p.relative_to(root).as_posix(): source_sha256(p) for p in outputs},
              "python_version": sys.version.split()[0], "python_executable": sys.executable,
              "versions": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "matplotlib")},
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              "git_uncommitted": subprocess.check_output(["git", "status", "--short"], cwd=root, text=True).splitlines()}
    path = root / "results/tables/diagnostico_estabilidad_config.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return record
