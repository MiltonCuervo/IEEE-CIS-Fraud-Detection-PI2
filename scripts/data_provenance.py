"""Registra procedencia local de los CSV fuente sin copiarlos al repositorio."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_FILES = (
    Path("data/raw/train_transaction.csv"),
    Path("data/raw/train_identity.csv"),
)
PACKAGE_NAMES = (
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "scikit-learn",
    "xgboost",
    "pyarrow",
    "jupyterlab",
    "ipykernel",
    "kaggle",
    "pytest",
)
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "interim" / "data_provenance.json"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in PACKAGE_NAMES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def build_manifest() -> dict[str, object]:
    files = []
    for relative_path in RAW_FILES:
        path = PROJECT_ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"No existe el archivo fuente esperado: {path}")
        stat = path.stat()
        files.append({
            "path": relative_path.as_posix(),
            "size_bytes": stat.st_size,
            "sha256": sha256_file(path),
        })
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(),
        "source_files": files,
        "note": "Local provenance record; raw data and this manifest are not versioned by Git.",
    }


def main() -> None:
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    DEFAULT_OUTPUT.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Manifiesto de procedencia guardado: {DEFAULT_OUTPUT}")
    for source in manifest["source_files"]:
        print(f"{source['path']}: {source['size_bytes']:,} bytes; sha256={source['sha256']}")
    print(f"Commit: {manifest['git_commit'] or 'no disponible'}")
    print(f"Python: {manifest['python_version']}")


if __name__ == "__main__":
    main()
