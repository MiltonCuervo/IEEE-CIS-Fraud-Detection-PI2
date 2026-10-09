"""Reajuste del predictor seleccionado y calibración temporal, sin test."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from xgboost import XGBClassifier

from fraud_cost.costs import validate_labels
from fraud_cost.features import PastOnlyFeatures, make_preprocessor, transaction_features


def source_sha256(path, *, code_only=False):
    path = Path(path)
    digest = hashlib.sha256()
    if path.suffix == ".ipynb":
        notebook = json.loads(path.read_text(encoding="utf-8"))
        cells = notebook["cells"]
        if code_only:
            cells = [cell for cell in cells if cell["cell_type"] == "code"]
        payload = [(cell["cell_type"], cell["source"]) for cell in cells]
        digest.update(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    else:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def verified_selection(root):
    """La elección del 05 debe corresponder a sus fuentes y datos actuales."""
    root = Path(root)
    record = json.loads((root / "results/tables/seleccion_temporal_config.json").read_text(encoding="utf-8"))
    if record.get("test_evaluated") or record.get("external_validation_evaluated"):
        raise ValueError("La selección debe haber usado solo ventanas internas de train")
    if record["selected_candidate"]["features"] not in {"base", "history"}:
        raise ValueError("Esquema de variables desconocido")
    for relative, expected in record["source_sha256"].items():
        path = root / relative.replace("\\", "/")
        if source_sha256(path) != expected:
            raise ValueError(f"Cambió una fuente de selección: {relative}. Revise o regenere el 05.")
    return record


def load_development(root, *, chunk_size=100000):
    """Carga train/validation; descarta test y gaps antes de unir o modelar."""
    root = Path(root)
    manifest = pd.read_parquet(root / "data/interim/split_manifest.parquet", columns=["TransactionID", "split"])
    if manifest.TransactionID.duplicated().any() or manifest.split.isna().any():
        raise ValueError("Manifiesto incompleto o con identificadores duplicados")
    selected = manifest.loc[manifest.split.isin(["train", "validation"])].copy()
    if set(selected.split) != {"train", "validation"}:
        raise ValueError("Se necesitan train y validation")
    allowed = set(selected.TransactionID)
    pieces = []
    for chunk in pd.read_csv(root / "data/raw/train_transaction.csv", chunksize=chunk_size):
        part = chunk.loc[chunk.TransactionID.isin(allowed)].copy()
        if not part.empty:
            floats = [c for c in part.select_dtypes(include="floating") if c != "TransactionAmt"]
            part[floats] = part[floats].astype("float32")
            pieces.append(part)
    if not pieces:
        raise ValueError("Los CSV no contienen los bloques de desarrollo")
    transactions = pd.concat(pieces, ignore_index=True)
    del pieces
    if transactions.TransactionID.duplicated().any() or set(transactions.TransactionID) != allowed:
        raise ValueError("Las transacciones no cubren exactamente los bloques de desarrollo")
    identity = pd.read_csv(root / "data/raw/train_identity.csv")
    identity = identity.loc[identity.TransactionID.isin(allowed)].copy()
    if identity.TransactionID.duplicated().any():
        raise ValueError("Identidad duplicada")
    floats = identity.select_dtypes(include="floating").columns
    identity[floats] = identity[floats].astype("float32")
    frame = transactions.merge(identity, on="TransactionID", how="left", validate="one_to_one", indicator=True)
    frame["has_identity"] = frame.pop("_merge").eq("both").astype("int8")
    frame = frame.merge(selected, on="TransactionID", validate="one_to_one")
    train = frame.loc[frame.split.eq("train")].sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)
    validation = frame.loc[frame.split.eq("validation")].sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)
    return train, validation


def split_validation(validation, *, calibration_fraction=.5):
    """Mitad temprana por filas; un lote simultáneo nunca cruza el límite."""
    if not 0 < calibration_fraction < 1 or validation.empty or not validation.split.eq("validation").all():
        raise ValueError("Defina una fracción válida sobre validation")
    ordered = validation.sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)
    nominal_rows = int(round(len(ordered) * calibration_fraction))
    if not 0 < nominal_rows < len(ordered):
        raise ValueError("La fracción no deja dos ventanas")
    cutoff = ordered.TransactionDT.iloc[nominal_rows - 1]
    early = ordered.loc[ordered.TransactionDT <= cutoff].copy()
    late = ordered.loc[ordered.TransactionDT > cutoff].copy()
    if early.empty or late.empty:
        raise ValueError("Los tiempos no permiten separar calibración y diagnóstico")
    return early, late


def check_blocks(train, calibration, policy, *, gap_days=7):
    if not np.isfinite(gap_days) or gap_days < 0:
        raise ValueError("Gap inválido")
    windows, identifiers = [], set()
    for name, expected, block in (("train", "train", train), ("validation_cal", "validation", calibration),
                                  ("validation_policy", "validation", policy)):
        if block.empty or not block.split.eq(expected).all() or block.TransactionID.duplicated().any():
            raise ValueError(f"Bloque inválido: {name}")
        labels, _ = validate_labels(block.isFraud, block.TransactionAmt)
        if len(np.unique(labels)) != 2 or not np.isfinite(block.TransactionDT.to_numpy(dtype=float)).all():
            raise ValueError(f"Se requieren tiempo finito y ambas clases: {name}")
        current = set(block.TransactionID)
        if identifiers.intersection(current):
            raise ValueError("Las ventanas comparten identificadores")
        identifiers.update(current)
        windows.append({"block": name, "rows": len(block), "frauds": int(labels.sum()),
                        "fraud_rate": float(labels.mean()), "time_start": int(block.TransactionDT.min()),
                        "time_end": int(block.TransactionDT.max())})
    if train.TransactionDT.max() + gap_days * 86400 >= calibration.TransactionDT.min():
        raise ValueError("No se respeta el gap entre train y calibración")
    if calibration.TransactionDT.max() >= policy.TransactionDT.min():
        raise ValueError("Calibración debe preceder estrictamente al diagnóstico")
    return pd.DataFrame(windows)


def probabilities(p):
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Se requieren probabilidades finitas en [0,1]")
    return p


def clipped_logit(p):
    p = np.clip(probabilities(p), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def fit_sigmoid(p, y):
    p = probabilities(p)
    labels, _ = validate_labels(y, np.zeros(len(p)))
    if len(np.unique(labels)) != 2:
        raise ValueError("Calibración requiere ambas clases")
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    calibrator.fit(clipped_logit(p), labels)
    return calibrator


def calibrated_probabilities(calibrator, p):
    return calibrator.predict_proba(clipped_logit(p))[:, 1]


@dataclass
class FittedPredictor:
    feature_mode: str
    builder: object
    preprocessor: object
    model: object
    calibrator: object
    feature_columns: list
    train_end: float
    params: dict

    def predict_raw(self, frame):
        if frame.empty or frame.TransactionDT.min() <= self.train_end:
            raise ValueError("La inferencia requiere operaciones posteriores a train")
        X = self.builder.transform(frame) if self.builder else transaction_features(frame)
        if set(X.columns) != set(self.feature_columns):
            raise ValueError("La inferencia debe conservar el esquema de variables entrenado")
        return self.model.predict_proba(self.preprocessor.transform(X))[:, 1]

    def predict_calibrated(self, frame):
        return calibrated_probabilities(self.calibrator, self.predict_raw(frame))


def fit_selected(train, calibration, policy, selection, *, min_frequency=50, progress=print):
    """Un predictor fijo; el calibrador solo recibe puntuaciones tempranas."""
    windows = check_blocks(train, calibration, policy)
    if len(train) != selection["train_rows"] or train.TransactionDT.max() != selection["train_end"]:
        raise ValueError("Train no coincide con la población de selección")
    mode = selection["selected_candidate"]["features"]
    if mode not in {"base", "history"}:
        raise ValueError("Esquema de variables desconocido")
    params = dict(selection["selected_params"])
    start = time.perf_counter()
    builder = PastOnlyFeatures() if mode == "history" else None
    progress(f"Preparación de {len(train):,} transacciones de train; esquema {mode}.", flush=True)
    X = builder.fit_transform(train) if builder else transaction_features(train)
    preprocessor = make_preprocessor(X, min_frequency=min_frequency)
    Xt = preprocessor.fit_transform(X)
    feature_columns = X.columns.tolist()
    transformed_columns = Xt.shape[1]
    del X
    gc.collect()
    progress(f"Ajuste de XGBoost: {Xt.shape}; {Xt.dtype}.", flush=True)
    model = XGBClassifier(**params)
    model.fit(Xt, train.isFraud, verbose=False)
    del Xt
    gc.collect()
    predictor = FittedPredictor(mode, builder, preprocessor, model, None, feature_columns,
                                float(train.TransactionDT.max()), params)
    p_cal = predictor.predict_raw(calibration)
    predictor.calibrator = fit_sigmoid(p_cal, calibration.isFraud)
    progress("Calibrador nuevo ajustado solo con validation temprana.", flush=True)
    raw = predictor.predict_raw(policy)
    calibrated = calibrated_probabilities(predictor.calibrator, raw)
    slope = float(predictor.calibrator.coef_[0, 0])
    diagnostics = {"feature_mode": mode, "input_columns": len(feature_columns),
                   "transformed_columns": transformed_columns, "matrix_dtype": "float32",
                   "calibrator": "sigmoid_on_clipped_probability_logit", "calibrator_C": 1e6,
                   "logit_clip_epsilon": 1e-6, "slope": slope,
                   "intercept": float(predictor.calibrator.intercept_[0]),
                   "positive_slope": slope > 0, "min_category_frequency": min_frequency,
                   "elapsed_seconds": time.perf_counter() - start}
    return predictor, raw, calibrated, windows, diagnostics


def predictive_metrics(y, p, *, name):
    p = probabilities(p)
    labels, _ = validate_labels(y, np.zeros(len(p)))
    if len(np.unique(labels)) != 2:
        raise ValueError("El diagnóstico requiere ambas clases")
    predicted = p >= .5
    fraud = labels == 1
    tp = int((predicted & fraud).sum())
    fp = int((predicted & ~fraud).sum())
    return {"probabilities": name, "block": "validation_policy", "rows": len(p),
            "frauds": int(fraud.sum()), "fraud_rate": float(fraud.mean()),
            "average_precision": float(average_precision_score(labels, p)),
            "brier_score": float(brier_score_loss(labels, p)),
            "log_loss": float(log_loss(labels, p, labels=[0, 1])),
            "mean_probability": float(p.mean()), "tp": tp, "fp": fp,
            "fn": int((~predicted & fraud).sum()), "tn": int((~predicted & ~fraud).sum()),
            "recall_at_fixed_0_5": tp / fraud.sum(),
            "precision_at_fixed_0_5": tp / (tp + fp) if tp + fp else 0.,
            "legitimate_block_rate_at_fixed_0_5": fp / (~fraud).sum()}


def reliability_table(y, p, *, name, n_bins=10):
    p = probabilities(p)
    labels, _ = validate_labels(y, np.zeros(len(p)))
    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("Número de bins inválido")
    bins = pd.qcut(p, q=n_bins, labels=False, duplicates="drop") if np.unique(p).size > 1 else np.zeros(len(p))
    frame = pd.DataFrame({"bin": bins, "p": p, "y": labels})
    table = frame.groupby("bin", observed=True).agg(
        rows=("y", "size"), frauds=("y", "sum"), observed_rate=("y", "mean"),
        mean_probability=("p", "mean"), min_probability=("p", "min"), max_probability=("p", "max"),
    ).reset_index()
    table.insert(0, "probabilities", name)
    return table


def save_refit(root, predictor, policy, raw, calibrated, windows, diagnostics, metrics, reliability, *, calibration_fraction=.5):
    root = Path(root)
    tables, models = root / "results/tables", root / "results/models"
    for directory in (tables, models, root / "data/interim"):
        directory.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(tables / "reajuste_metricas_validation.csv", index=False)
    reliability.to_csv(tables / "reajuste_calibracion_bins.csv", index=False)
    windows.to_csv(tables / "reajuste_ventanas.csv", index=False)
    model_path = models / "selected_xgboost_platt.joblib"
    joblib.dump(predictor, model_path, compress=3)
    scores = policy[["TransactionID", "TransactionDT", "TransactionAmt", "isFraud"]].copy()
    scores["probability_raw"] = raw
    scores["probability_calibrated"] = calibrated
    scores["action_fixed_0_5"] = np.where(calibrated >= .5, "block", "approve")
    score_path = root / "data/interim/validation_scores_selected.parquet"
    scores.to_parquet(score_path, index=False)
    notebook = root / "notebooks/06_reajuste_y_calibracion.ipynb"
    sources = [root / "src/fraud_cost/refit.py", root / "src/fraud_cost/features.py",
               root / "src/fraud_cost/costs.py", root / "requirements.txt", notebook,
               root / "results/tables/seleccion_temporal_config.json", root / "data/interim/split_manifest.parquet",
               root / "data/raw/train_transaction.csv", root / "data/raw/train_identity.csv"]
    record = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
              "selected_candidate": json.loads((tables / "seleccion_temporal_config.json").read_text(encoding="utf-8"))["selected_candidate"],
              "predictor_params": predictor.params, "diagnostics": diagnostics,
              "validation_calibration_fraction_requested": calibration_fraction,
              "validation_boundary_rule": "mitad por filas; lote del tiempo de corte completo en calibración",
              "history_protocol": "strict_past_train_frozen_state_for_validation", "classification_threshold": .5,
              "test_evaluated": False, "economic_policies_evaluated": False,
              "windows": windows.to_dict(orient="records"),
              "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "xgboost", "joblib")},
              "source_sha256": {path.relative_to(root).as_posix(): source_sha256(path, code_only=path == notebook) for path in sources},
              "notebook_hash_method": "JSON de tipo y fuente de celdas de código; excluye Markdown, salidas y conteos",
              "artifact_sha256": {path.relative_to(root).as_posix(): source_sha256(path) for path in (model_path, score_path)},
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              "git_uncommitted": subprocess.check_output(["git", "status", "--short"], cwd=root, text=True).splitlines(),
              "note": "Predictor fijo después de calibrar. Validación posterior es desarrollo; Ca sigue pendiente. No cargar joblib de fuentes no confiables."}
    (tables / "reajuste_config.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record
