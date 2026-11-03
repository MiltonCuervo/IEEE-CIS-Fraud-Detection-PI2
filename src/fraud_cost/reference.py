"""Preparación ordinal de la comparación exploratoria acordada en el notebook 03."""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.validation import check_is_fitted

EXCLUDED = {"TransactionID", "TransactionDT", "isFraud", "split"}
CATEGORICAL = {
    "ProductCD", *(f"card{i}" for i in range(1, 7)), "addr1", "addr2",
    "P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo",
    *(f"M{i}" for i in range(1, 10)),
}


def make_features(data):
    """Excluye controles y homogeneiza categorías; no aprende estadísticas."""
    X = data.drop(columns=list(EXCLUDED.intersection(data.columns))).copy()
    categorical = set(X.select_dtypes(include=["object", "string", "category"]).columns) | CATEGORICAL.intersection(X.columns)
    for column in categorical:
        X[column] = X[column].astype("string").fillna("__MISSING__").astype(object)
    numeric = [c for c in X if c not in categorical]
    X[numeric] = X[numeric].replace([np.inf, -np.inf], np.nan)
    return X


class ColumnMedianImputer(TransformerMixin, BaseEstimator):
    """Aprende una mediana por vector, evitando ordenar la matriz completa.

    Conserva columnas vacías con cero y produce float32. No añade indicadores.
    Las estadísticas se guardan para aplicar exactamente el mismo ajuste después.
    """
    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        self.statistics_ = []
        self.empty_columns_ = []
        for column in self.feature_names_in_:
            values = X[column].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            if not len(finite):
                self.empty_columns_.append(column)
            self.statistics_.append(float(np.median(finite)) if len(finite) else 0.0)
        self.statistics_ = np.asarray(self.statistics_)
        if self.empty_columns_:
            warnings.warn(f"Columnas vacías en train, conservadas con cero: {self.empty_columns_}")
        return self

    def transform(self, X):
        check_is_fitted(self, 'statistics_')
        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError('Cambió el esquema numérico aprendido en train')
        output = np.empty((len(X), self.n_features_in_), dtype=np.float32)
        for i, column in enumerate(self.feature_names_in_):
            values = X[column].to_numpy(dtype=float, copy=True)
            values[~np.isfinite(values)] = self.statistics_[i]
            output[:, i] = values
        return output

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'statistics_')
        return self.feature_names_in_


def build_preprocessor(X):
    categorical = X.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
    numeric = [c for c in X if c not in categorical]
    preprocessor = ColumnTransformer([
        ('categorical', OrdinalEncoder(handle_unknown='use_encoded_value',
                                      unknown_value=-1, dtype=np.float32), categorical),
        ('numeric', ColumnMedianImputer(), numeric),
    ], sparse_threshold=0)
    return preprocessor, categorical, numeric


def read_model_frame(raw_dir, manifest):
    """Reduce precisión predictiva; conserva el monto económico en float64.

    Una muestra con filas permite inferir dtypes; un encabezado vacío no lo hace.
    La correspondencia de identity se determina con el merge, no con un atributo.
    """
    raw_dir = __import__('pathlib').Path(raw_dir)
    tables = []
    for filename in ('train_transaction.csv', 'train_identity.csv'):
        path = raw_dir / filename
        sample = pd.read_csv(path, nrows=1000)
        dtype = {c: 'float32' for c in sample.select_dtypes('number').columns
                 if c not in {'TransactionID', 'TransactionDT', 'TransactionAmt', 'isFraud'}}
        table = pd.read_csv(path, dtype=dtype)
        if table.TransactionID.isna().any() or table.TransactionID.duplicated().any():
            raise ValueError(f'Claves nulas o duplicadas en {filename}')
        tables.append(table)
    transactions, identity = tables
    required = {'TransactionID', 'TransactionDT', 'TransactionAmt', 'isFraud'}
    if not required.issubset(transactions) or not identity.TransactionID.isin(transactions.TransactionID).all():
        raise ValueError('Esquema o correspondencia de identidad inválidos')
    frame = transactions.merge(identity, on='TransactionID', how='left', validate='one_to_one', indicator=True)
    frame['has_identity'] = frame.pop('_merge').eq('both').astype('int8')
    frame = frame.merge(manifest, on='TransactionID', how='left', validate='one_to_one')
    if len(frame) != len(transactions) or frame['split'].isna().any():
        raise ValueError('El manifiesto no cubre exactamente las transacciones')
    return frame.sort_values(['TransactionDT', 'TransactionID']).reset_index(drop=True)
