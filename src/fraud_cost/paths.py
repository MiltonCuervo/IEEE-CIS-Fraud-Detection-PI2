"""Rutas del proyecto independientes del directorio de ejecución."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
RESULTS = PROJECT_ROOT / "results"

