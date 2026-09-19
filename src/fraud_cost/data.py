"""Carga y validaciones mínimas para los datos IEEE-CIS."""

from pathlib import Path

import pandas as pd


REQUIRED_TRANSACTION_COLUMNS = {
    "TransactionID",
    "TransactionDT",
    "TransactionAmt",
    "isFraud",
}


def load_ieee_cis(raw_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga las tablas originales y realiza un left join validado.

    No descarga datos ni modifica los archivos de origen. Esto hace explícito
    que la adquisición de datos y el análisis son responsabilidades distintas.
    """
    raw_dir = Path(raw_dir)
    transaction_path = raw_dir / "train_transaction.csv"
    identity_path = raw_dir / "train_identity.csv"

    missing_files = [
        str(path) for path in (transaction_path, identity_path) if not path.exists()
    ]
    if missing_files:
        raise FileNotFoundError(
            "Faltan archivos IEEE-CIS en data/raw: " + ", ".join(missing_files)
        )

    transactions = pd.read_csv(transaction_path)
    identity = pd.read_csv(identity_path)
    missing_columns = REQUIRED_TRANSACTION_COLUMNS.difference(transactions.columns)
    if missing_columns:
        raise ValueError(f"Faltan columnas obligatorias: {sorted(missing_columns)}")
    if transactions["TransactionID"].duplicated().any():
        raise ValueError("TransactionID debe ser único en train_transaction")
    if identity["TransactionID"].duplicated().any():
        raise ValueError("TransactionID debe ser único en train_identity")

    merged = transactions.merge(
        identity,
        on="TransactionID",
        how="left",
        validate="one_to_one",
    )
    if len(merged) != len(transactions):
        raise AssertionError("El left join cambió el número de transacciones")
    return transactions, identity, merged


def missingness_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Resume ausencia por columna, ordenada de mayor a menor."""
    result = pd.DataFrame(
        {
            "n_missing": frame.isna().sum(),
            "missing_rate": frame.isna().mean(),
            "dtype": frame.dtypes.astype(str),
        }
    )
    return result.loc[result["n_missing"] > 0].sort_values(
        "missing_rate", ascending=False
    )

