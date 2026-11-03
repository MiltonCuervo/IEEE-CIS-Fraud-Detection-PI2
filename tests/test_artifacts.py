"""Evita reutilizar artefactos de otra corrida o publicar una recuperación parcial."""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from fraud_cost import artifacts


@pytest.fixture
def recovery_script():
    path = Path(__file__).resolve().parents[1] / "scripts/recover_reference_artifacts.py"
    spec = importlib.util.spec_from_file_location("recover_reference_artifacts", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def original_files(tmp_path, monkeypatch):
    expected = {}
    for relative in artifacts.ARTIFACTS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"verified original")
        expected[relative] = artifacts.sha256_file(path)
    monkeypatch.setattr(artifacts, "verify_experimental_chain",
                        lambda root: ({"output_sha256": expected}, {"selected_candidate": {"candidate": "selected"}}))
    return tmp_path


def test_original_hashes_are_required_before_reuse(original_files):
    assert artifacts.verify_reference_artifacts(original_files)["status"] == "verified_original_artifacts"
    (original_files / artifacts.ARTIFACTS[0]).write_bytes(b"stale scores")
    with pytest.raises(FileNotFoundError):
        artifacts.verify_reference_artifacts(original_files)


def test_unverified_reconstruction_is_rejected(original_files):
    (original_files / artifacts.ARTIFACTS[0]).write_bytes(b"stale scores")
    (original_files / artifacts.RECOVERY_RECORD).parent.mkdir(parents=True, exist_ok=True)
    (original_files / artifacts.RECOVERY_RECORD).write_text(
        json.dumps({"status": "started", "test_evaluated": False}), encoding="utf-8")
    with pytest.raises(ValueError, match="no está verificada"):
        artifacts.verify_reference_artifacts(original_files)


def test_reconstruction_is_bound_to_its_receipt_and_evidence(original_files):
    root = original_files
    for relative in (artifacts.REFERENCE_RECORD, artifacts.SELECTION_PROTOCOL, artifacts.SELECTION_RECORD):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    (root / artifacts.ARTIFACTS[0]).write_bytes(b"reconstructed scores")
    evidence = root / "data/interim/evidence.json"
    evidence.write_text("{}", encoding="utf-8")
    receipt = {
        "status": "verified_reconstruction", "test_evaluated": False,
        "historical_records_sha256": {p: artifacts.sha256_file(root / p) for p in
                                      (artifacts.REFERENCE_RECORD, artifacts.SELECTION_PROTOCOL, artifacts.SELECTION_RECORD)},
        "artifact_sha256": {p: artifacts.sha256_file(root / p) for p in artifacts.ARTIFACTS},
        "numerical_comparison": {"passed": True}, "inference_verification": {"passed": True},
        "reconstruction_evidence_sha256": {"data/interim/evidence.json": artifacts.sha256_file(evidence)},
    }
    (root / artifacts.RECOVERY_RECORD).write_text(json.dumps(receipt), encoding="utf-8")
    assert artifacts.verify_reference_artifacts(root)["status"] == "verified_reconstruction"
    evidence.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="hash distinto"):
        artifacts.verify_reference_artifacts(root)


def test_hash_check_does_not_silently_normalize_line_endings(tmp_path):
    path = tmp_path / "source.py"
    path.write_bytes(b"x = 1\n")
    expected = artifacts.sha256_file(path)
    path.write_bytes(b"x = 1\r\n")
    with pytest.raises(ValueError, match="hash distinto"):
        artifacts.verify_hashes(tmp_path, {"source.py": expected})


def test_recovery_paths_must_stay_in_project(tmp_path):
    with pytest.raises(ValueError, match="fuera del proyecto"):
        artifacts.project_path(tmp_path, "../outside.joblib")


def test_table_comparison_rejects_changed_metrics_and_actions(tmp_path, recovery_script, monkeypatch):
    monkeypatch.setattr(recovery_script, "COMPARISON_TABLES", ("metrics.csv",))
    before, after = tmp_path / "before", tmp_path / "after"
    before.mkdir(); after.mkdir()
    pd.DataFrame({"model": ["m"], "ap": [0.5], "actions": [3]}).to_csv(before / "metrics.csv", index=False)
    pd.DataFrame({"model": ["m"], "ap": [0.5 + 1e-12], "actions": [3]}).to_csv(after / "metrics.csv", index=False)
    assert recovery_script.compare_tables(before, after)["passed"]
    pd.DataFrame({"model": ["m"], "ap": [0.51], "actions": [3]}).to_csv(after / "metrics.csv", index=False)
    with pytest.raises(AssertionError):
        recovery_script.compare_tables(before, after)
    pd.DataFrame({"model": ["m"], "ap": [0.5], "actions": [4]}).to_csv(after / "metrics.csv", index=False)
    with pytest.raises(AssertionError):
        recovery_script.compare_tables(before, after)


def test_publication_restores_previous_artifacts_on_failure(tmp_path, recovery_script, monkeypatch):
    monkeypatch.setattr(recovery_script, "ROOT", tmp_path)
    stage, backup = tmp_path / "stage", tmp_path / "backup"
    for relative in artifacts.ARTIFACTS:
        old, new = tmp_path / relative, stage / relative
        old.parent.mkdir(parents=True, exist_ok=True)
        new.parent.mkdir(parents=True, exist_ok=True)
        old.write_bytes(b"previous")
        new.write_bytes(b"replacement")
    def fail_audit(root):
        raise ValueError("failed publication audit")
    monkeypatch.setattr(recovery_script, "verify_reference_artifacts", fail_audit)
    with pytest.raises(ValueError, match="failed publication"):
        recovery_script.publish(stage, backup, {"status": "verified_reconstruction"})
    for relative in artifacts.ARTIFACTS:
        assert (tmp_path / relative).read_bytes() == b"previous"
        assert (backup / relative).read_bytes() == b"previous"
    assert not (tmp_path / artifacts.RECOVERY_RECORD).exists()
