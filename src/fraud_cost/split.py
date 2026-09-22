"""Partición temporal reproducible para las transacciones etiquetadas."""

from __future__ import annotations

import pandas as pd


VALID_SPLITS = ("train", "gap_1", "validation", "gap_2", "test")


def create_temporal_split(
    frame: pd.DataFrame,
    *,
    train_fraction: float = 0.60,
    validation_end_fraction: float = 0.80,
    gap_days: int = 7,
    time_column: str = "TransactionDT",
) -> pd.Series:
    """Asigna bloques cronológicos usando fracciones del rango temporal.

    Las fracciones definen el final de entrenamiento y validación sobre la
    duración total. Los gaps comienzan después de esos límites y consumen
    parte del intervalo siguiente. La salida conserva el índice de ``frame``.
    """
    if not 0 < train_fraction < validation_end_fraction < 1:
        raise ValueError("Se requiere 0 < train_fraction < validation_end_fraction < 1")
    if gap_days < 0:
        raise ValueError("gap_days no puede ser negativo")
    if time_column not in frame:
        raise KeyError(f"No existe la columna temporal {time_column}")
    if frame[time_column].isna().any():
        raise ValueError("La columna temporal no puede contener valores ausentes")

    time = frame[time_column]
    start = time.min()
    span = time.max() - start
    if span <= 0:
        raise ValueError("El rango temporal debe ser positivo")

    train_end = start + train_fraction * span
    validation_end = start + validation_end_fraction * span
    gap_seconds = gap_days * 86_400
    gap_1_end = train_end + gap_seconds
    gap_2_end = validation_end + gap_seconds

    if gap_1_end >= validation_end:
        raise ValueError("El primer gap consume todo el bloque de validación")
    if gap_2_end >= time.max():
        raise ValueError("El segundo gap consume todo el bloque de prueba")

    split = pd.Series(index=frame.index, dtype="string", name="split")
    split.loc[time <= train_end] = "train"
    split.loc[(time > train_end) & (time <= gap_1_end)] = "gap_1"
    split.loc[(time > gap_1_end) & (time <= validation_end)] = "validation"
    split.loc[(time > validation_end) & (time <= gap_2_end)] = "gap_2"
    split.loc[time > gap_2_end] = "test"
    return split


def summarize_temporal_split(
    frame: pd.DataFrame,
    split: pd.Series,
    *,
    start_time: float | None = None,
) -> pd.DataFrame:
    """Resume tamaño, fraude, monto y límites temporales de cada bloque."""
    required = {"TransactionDT", "TransactionAmt", "isFraud"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Faltan columnas para resumir la partición: {sorted(missing)}")
    if not split.index.equals(frame.index):
        raise ValueError("Los índices de frame y split deben coincidir")

    base = frame.assign(split=split)
    origin = frame["TransactionDT"].min() if start_time is None else start_time
    summary = base.groupby("split", observed=True).agg(
        transacciones=("isFraud", "size"),
        fraudes=("isFraud", "sum"),
        tasa_fraude=("isFraud", "mean"),
        monto_mediano=("TransactionAmt", "median"),
        monto_total=("TransactionAmt", "sum"),
        tiempo_inicio=("TransactionDT", "min"),
        tiempo_fin=("TransactionDT", "max"),
    )
    summary["dia_inicio_relativo"] = (summary["tiempo_inicio"] - origin) / 86_400
    summary["dia_fin_relativo"] = (summary["tiempo_fin"] - origin) / 86_400
    return summary.reindex([label for label in VALID_SPLITS if label in summary.index])


def validate_temporal_split(frame: pd.DataFrame, split: pd.Series) -> None:
    """Comprueba cobertura, exclusividad y orden estricto de los bloques."""
    if len(frame) != len(split):
        raise AssertionError("Cada transacción debe tener exactamente una asignación")
    if split.isna().any():
        raise AssertionError("Existen transacciones sin bloque")
    if not set(split.unique()).issubset(VALID_SPLITS):
        raise AssertionError("La partición contiene etiquetas desconocidas")

    time = frame["TransactionDT"]
    ordered = [label for label in VALID_SPLITS if split.eq(label).any()]
    for previous, current in zip(ordered, ordered[1:]):
        if time[split.eq(previous)].max() >= time[split.eq(current)].min():
            raise AssertionError(f"El orden temporal falla entre {previous} y {current}")

