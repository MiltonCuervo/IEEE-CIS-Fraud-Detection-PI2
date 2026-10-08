"""Variables de transacción e historial estricto, sin utilizar etiquetas."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

EXCLUDED = {"TransactionID", "TransactionDT", "isFraud", "split"}
CATEGORICAL = {
    "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo",
    *(f"M{i}" for i in range(1, 10)),
}
ENTITY_COLUMNS = {
    "card_proxy": ("card1", "card2", "card3", "card5"),
    "device_proxy": ("DeviceInfo",),
    "email_domain": ("P_emaildomain",),
}


def _validated(frame):
    if not {"TransactionID", "TransactionDT", "TransactionAmt"}.issubset(frame):
        raise ValueError("Faltan identificador, tiempo o monto")
    if frame.empty or not frame.index.is_unique or frame.TransactionID.duplicated().any():
        raise ValueError("Se necesitan filas e índices/identificadores únicos")
    if not np.isfinite(frame.TransactionDT.to_numpy(dtype=float)).all():
        raise ValueError("El tiempo debe ser finito")
    amount = frame.TransactionAmt.to_numpy(dtype=float)
    if not np.isfinite(amount).all() or (amount < 0).any():
        raise ValueError("El monto debe ser finito y no negativo")


def entity_key(frame, columns):
    """Clave operativa; componentes incompletos no identifican una entidad."""
    if not set(columns).issubset(frame):
        return pd.Series(pd.NA, index=frame.index, dtype="string")
    parts = frame[list(columns)].astype("string")
    # Longitud prefijada evita colisiones si un texto incluye un separador.
    encoded = parts.apply(lambda col: col.str.len().astype("string") + ":" + col)
    key = encoded.iloc[:, 0]
    for column in encoded.columns[1:]:
        key = key + "|" + encoded[column]
    return key.mask(parts.isna().any(axis=1))


def transaction_features(frame, *, engineered=False):
    _validated(frame)
    X = frame.drop(columns=list(EXCLUDED.intersection(frame))).copy()
    numeric = X.select_dtypes(include="number").columns
    X[numeric] = X[numeric].replace([np.inf, -np.inf], np.nan).astype("float32")
    # Consolidar bloques tras convertir columnas evita fragmentar el historial.
    X = X.copy()
    if engineered:
        X["amount_log1p"] = np.log1p(frame.TransactionAmt).astype("float32")
        X["missing_count"] = X.isna().sum(axis=1).astype("float32")
        # TransactionDT no tiene origen calendario: son fases relativas.
        for name, period in (("day", 86400), ("week", 7 * 86400)):
            angle = 2 * np.pi * frame.TransactionDT / period
            X[f"relative_{name}_sin"] = np.sin(angle).astype("float32")
            X[f"relative_{name}_cos"] = np.cos(angle).astype("float32")
    categorical = set(X.select_dtypes(include=["object", "string", "category"]).columns) | CATEGORICAL.intersection(X)
    for column in categorical:
        X[column] = X[column].astype("string").fillna("__MISSING__").astype(object)
    return X


class PastOnlyFeatures:
    """Train: historial estrictamente anterior. Futuro: estado de train congelado.

    No añade operaciones de validación al historial ni usa los gaps.
    Dos operaciones simultáneas comparten exactamente el mismo pasado.
    """
    def __init__(self, entities=None):
        self.entities = ENTITY_COLUMNS if entities is None else entities

    def fit_transform(self, frame):
        _validated(frame)
        self.cutoff_ = float(frame.TransactionDT.max())
        self.states_ = {}
        output = transaction_features(frame, engineered=True)
        for name, columns in self.entities.items():
            key = entity_key(frame, columns)
            work = pd.DataFrame({"key": key.to_numpy(), "time": frame.TransactionDT.to_numpy(),
                                 "amount": frame.TransactionAmt.to_numpy(), "position": np.arange(len(frame))})
            valid = work.dropna(subset=["key"])
            batches = valid.groupby(["key", "time"], sort=True).agg(n=("amount", "size"), total=("amount", "sum")).reset_index()
            groups = batches.groupby("key", sort=False)
            batches["prior_n"] = groups.n.cumsum() - batches.n
            batches["prior_total"] = groups.total.cumsum() - batches.total
            batches["prior_time"] = groups.time.shift()
            rows = work.merge(batches[["key", "time", "prior_n", "prior_total", "prior_time"]], on=["key", "time"], how="left", validate="many_to_one").sort_values("position")
            self.states_[name] = valid.groupby("key", sort=False).agg(n=("amount", "size"), total=("amount", "sum"), last_time=("time", "max"))
            self._attach(output, name, rows, frame.TransactionDT.to_numpy())
        return output

    @staticmethod
    def _attach(output, name, rows, time):
        n = rows.prior_n.fillna(0).to_numpy(dtype=float)
        total = rows.prior_total.to_numpy(dtype=float)
        mean = np.divide(total, n, out=np.full(len(n), np.nan), where=n > 0)
        output[f"{name}_past_count"] = n.astype("float32")
        output[f"{name}_past_mean_amount"] = mean.astype("float32")
        output[f"{name}_seconds_since_last"] = (time - rows.prior_time.to_numpy(dtype=float)).astype("float32")

    def transform(self, frame):
        _validated(frame)
        if not hasattr(self, "states_"):
            raise ValueError("Ajuste el historial antes de transformar")
        if frame.TransactionDT.min() <= self.cutoff_:
            raise ValueError("La transformación debe ocurrir estrictamente después de train")
        output = transaction_features(frame, engineered=True)
        for name, columns in self.entities.items():
            key = entity_key(frame, columns)
            state = self.states_[name]
            rows = pd.DataFrame({"prior_n": key.map(state.n).to_numpy(),
                                 "prior_total": key.map(state.total).to_numpy(),
                                 "prior_time": key.map(state.last_time).to_numpy()})
            self._attach(output, name, rows, frame.TransactionDT.to_numpy())
        return output


def _float32(matrix):
    return matrix.astype(np.float32)


def make_preprocessor(X, *, min_frequency=50):
    """Ajustar en cada fold; vocabulario/medianas nunca se comparten entre folds."""
    categorical = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric = [c for c in X if c not in categorical]
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("float32", FunctionTransformer(_float32)),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__", keep_empty_features=True)),
        ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=min_frequency, sparse_output=True, dtype=np.float32)),
    ])
    return ColumnTransformer([("numeric", numeric_pipeline, numeric), ("categorical", categorical_pipeline, categorical)], sparse_threshold=1.0)
