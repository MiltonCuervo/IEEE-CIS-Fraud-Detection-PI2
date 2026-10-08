"""Selección temporal dentro de train, con preparación ajustada por fold."""
from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from xgboost import XGBClassifier

from fraud_cost.features import PastOnlyFeatures, make_preprocessor, transaction_features

BASE_PARAMS = {"n_estimators": 250, "max_depth": 6, "learning_rate": .05,
               "subsample": .8, "colsample_bytree": .8, "min_child_weight": 10,
               "tree_method": "hist", "objective": "binary:logistic", "eval_metric": "aucpr",
               "n_jobs": 4, "random_state": 42, "verbosity": 0}
DEFAULT_CANDIDATES = (
    {"candidate": "reference", "features": "base", "params": {}},
    {"candidate": "history", "features": "history", "params": {}},
    {"candidate": "history_regularized", "features": "history", "params": {"max_depth": 4, "min_child_weight": 20, "reg_lambda": 5}},
    {"candidate": "history_deeper", "features": "history", "params": {"max_depth": 8, "min_child_weight": 10, "reg_lambda": 5}},
)


def load_train_only(root, *, chunk_size=100000):
    """Filtra filas train antes de unir identidad; no produce resúmenes externos."""
    root = Path(root)
    manifest = pd.read_parquet(root / "data/interim/split_manifest.parquet", columns=["TransactionID", "split"])
    if manifest.TransactionID.duplicated().any() or manifest.split.isna().any():
        raise ValueError("Manifiesto incompleto o con identificadores duplicados")
    allowed = set(manifest.loc[manifest.split.eq("train"), "TransactionID"])
    if not allowed:
        raise ValueError("No hay transacciones de train")
    pieces = []
    for chunk in pd.read_csv(root / "data/raw/train_transaction.csv", chunksize=chunk_size):
        chunk = chunk.loc[chunk.TransactionID.isin(allowed)].copy()
        if not chunk.empty:
            floats = [c for c in chunk.select_dtypes(include="floating") if c != "TransactionAmt"]
            chunk[floats] = chunk[floats].astype("float32")
            pieces.append(chunk)
    transactions = pd.concat(pieces, ignore_index=True)
    del pieces
    if transactions.TransactionID.duplicated().any() or set(transactions.TransactionID) != allowed:
        raise ValueError("Los CSV no cubren exactamente el train del manifiesto")
    identity = pd.read_csv(root / "data/raw/train_identity.csv")
    identity = identity.loc[identity.TransactionID.isin(allowed)].copy()
    if identity.TransactionID.duplicated().any():
        raise ValueError("Identidad duplicada")
    floats = identity.select_dtypes(include="floating").columns
    identity[floats] = identity[floats].astype("float32")
    frame = transactions.merge(identity, on="TransactionID", how="left", validate="one_to_one", indicator=True)
    frame["has_identity"] = frame.pop("_merge").eq("both").astype("int8")
    frame["split"] = "train"
    return frame.sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)


def temporal_folds(frame, *, train_end_fractions=(.60, .80), gap_days=7):
    """Ventanas expansivas sobre duración, no sobre cantidad de filas."""
    if frame.empty or not frame.split.eq("train").all():
        raise ValueError("La selección solo recibe train")
    t = frame.TransactionDT.to_numpy(dtype=float)
    if not np.isfinite(t).all() or np.ptp(t) <= 0 or not np.isfinite(gap_days) or gap_days < 0:
        raise ValueError("Tiempo o separación inválidos")
    fractions = tuple(train_end_fractions)
    if not fractions or any(not 0 < f < 1 for f in fractions) or any(a >= b for a,b in zip(fractions, fractions[1:])):
        raise ValueError("Las fracciones deben ser crecientes y estar entre cero y uno")
    ends = (*fractions[1:], 1.0)
    folds = []
    for fold_id, (fraction, end) in enumerate(zip(fractions, ends), 1):
        cutoff = t.min() + fraction * np.ptp(t)
        eval_end = t.min() + end * np.ptp(t)
        train_idx = np.flatnonzero(t <= cutoff)
        eval_idx = np.flatnonzero((t > cutoff + gap_days * 86400) & (t <= eval_end))
        if not len(train_idx) or not len(eval_idx):
            raise ValueError("Un gap consume su ventana de evaluación")
        for idx in (train_idx, eval_idx):
            if frame.iloc[idx].isFraud.nunique() != 2:
                raise ValueError("Cada ventana debe contener ambas clases")
        folds.append({"fold": fold_id, "train_idx": train_idx, "eval_idx": eval_idx,
                      "nominal_cutoff": cutoff, "gap_days": gap_days})
    return folds


def fold_summary(frame, folds):
    records = []
    for fold in folds:
        past, future = frame.iloc[fold["train_idx"]], frame.iloc[fold["eval_idx"]]
        records.append({"fold": fold["fold"], "train_rows": len(past), "eval_rows": len(future),
                        "train_frauds": int(past.isFraud.sum()), "eval_frauds": int(future.isFraud.sum()),
                        "train_end": int(past.TransactionDT.max()), "eval_start": int(future.TransactionDT.min()),
                        "eval_end": int(future.TransactionDT.max()), "gap_days": fold["gap_days"]})
    return pd.DataFrame(records)


def select_summary(metrics):
    result = metrics.groupby(["candidate", "features"], sort=False).agg(
        mean_ap=("average_precision", "mean"), std_ap=("average_precision", "std"),
        mean_brier=("brier_score", "mean"), mean_log_loss=("log_loss", "mean"),
        total_seconds=("seconds", "sum"), folds=("fold", "nunique"),
    ).reset_index()
    # Orden predeclarado de la grilla resuelve empates exactos.
    return result.sort_values("mean_ap", ascending=False, kind="stable").reset_index(drop=True)


def run_temporal_search(frame, folds, candidates=DEFAULT_CANDIDATES, *, progress=print):
    if not frame.split.eq("train").all():
        raise ValueError("La búsqueda no admite filas ajenas a train")
    if not candidates or len({c["candidate"] for c in candidates}) != len(candidates):
        raise ValueError("Defina candidatos únicos")
    for c in candidates:
        if c["features"] not in {"base", "history"}:
            raise ValueError("Conjunto de atributos desconocido")
    records, diagnostics = [], []
    for fold in folds:
        past = frame.iloc[fold["train_idx"]].copy()
        future = frame.iloc[fold["eval_idx"]].copy()
        if past.TransactionDT.max() + fold["gap_days"]*86400 >= future.TransactionDT.min():
            raise ValueError("El fold no respeta la separación temporal")
        for mode in dict.fromkeys(c["features"] for c in candidates):
            builder = PastOnlyFeatures() if mode == "history" else None
            X_past = builder.fit_transform(past) if builder else transaction_features(past)
            X_future = builder.transform(future) if builder else transaction_features(future)
            preprocessor = make_preprocessor(X_past)
            Xt = preprocessor.fit_transform(X_past)
            Xv = preprocessor.transform(X_future)
            diagnostic = {"fold": fold["fold"], "features": mode,
                          "input_columns": X_past.shape[1], "transformed_columns": Xt.shape[1],
                          "matrix_dtype": str(Xt.dtype),
                          "missing_columns": int(X_past.isna().any().sum()),
                          "empty_columns": int(X_past.isna().all().sum())}
            if builder:
                for entity in builder.entities:
                    diagnostic[f"{entity}_zero_history_rate"] = float(X_future[f"{entity}_past_count"].eq(0).mean())
            diagnostics.append(diagnostic)
            for candidate in (c for c in candidates if c["features"] == mode):
                progress(f"Fold {fold['fold']} / {candidate['candidate']}: {len(past):,} -> {len(future):,}", flush=True)
                params = BASE_PARAMS | candidate["params"]
                start = time.perf_counter()
                model = XGBClassifier(**params)
                # Sin early stopping en la ventana que se usa para puntuar.
                model.fit(Xt, past.isFraud, verbose=False)
                p = model.predict_proba(Xv)[:, 1]
                row = {"candidate": candidate["candidate"], "features": mode, "fold": fold["fold"],
                       "average_precision": float(average_precision_score(future.isFraud, p)),
                       "brier_score": float(brier_score_loss(future.isFraud, p)),
                       "log_loss": float(log_loss(future.isFraud, p, labels=[0,1])),
                       "seconds": time.perf_counter() - start,
                       "eval_rows": len(future), "eval_frauds": int(future.isFraud.sum())}
                records.append(row)
                progress(f"  AP={row['average_precision']:.5f}; {row['seconds']:.1f}s", flush=True)
                del model
            del Xt, Xv, X_past, X_future, preprocessor, builder
            gc.collect()
        del past, future
        gc.collect()
    metrics = pd.DataFrame(records)
    return metrics, select_summary(metrics), pd.DataFrame(diagnostics)


def save_selection(root, frame, folds, candidates, metrics, summary, diagnostics):
    root = Path(root)
    tables = root / "results/tables"
    tables.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(tables / "seleccion_temporal_folds.csv", index=False)
    summary.to_csv(tables / "seleccion_temporal_resumen.csv", index=False)
    diagnostics.to_csv(tables / "preparacion_diagnostico.csv", index=False)
    fold_summary(frame, folds).to_csv(tables / "seleccion_temporal_ventanas.csv", index=False)
    selected = next(c for c in candidates if c["candidate"] == summary.iloc[0].candidate)
    sources = [root/"src/fraud_cost"/n for n in ("features.py", "selection.py")]
    sources += [root/"notebooks/05_preparacion_y_seleccion_temporal.ipynb", root/"data/interim/split_manifest.parquet",
                root/"data/raw/train_transaction.csv", root/"data/raw/train_identity.csv"]
    def source_hash(path):
        digest = hashlib.sha256()
        if path.suffix == ".ipynb":
            nb = json.loads(path.read_text(encoding="utf-8"))
            payload = [(c["cell_type"], c["source"]) for c in nb["cells"]]
            digest.update(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        else:
            with path.open("rb") as file:
                for chunk in iter(lambda: file.read(8*1024*1024), b""):
                    digest.update(chunk)
        return digest.hexdigest()
    record = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
              "criterion": "mean_average_precision_internal_train_folds",
              "selected_candidate": selected, "selected_params": BASE_PARAMS | selected["params"],
              "candidates": candidates, "test_evaluated": False, "external_validation_evaluated": False,
              "train_rows": len(frame), "train_end": int(frame.TransactionDT.max()),
              "folds": fold_summary(frame, folds).to_dict(orient="records"),
              "versions": {p:importlib.metadata.version(p) for p in ("numpy", "pandas", "scikit-learn", "xgboost")},
              "source_sha256": {str(p.relative_to(root)):source_hash(p) for p in sources},
              "notebook_hash_method": "JSON de tipo y fuente por celda; excluye salidas y conteos de ejecución",
              "git_uncommitted": subprocess.check_output(["git", "status", "--short"], cwd=root, text=True).splitlines(),
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              "note": "Selección dentro de train; falta refit y nueva calibración. Commit puede tener cambios locales; hashes identifican fuentes."}
    (tables / "seleccion_temporal_config.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record
