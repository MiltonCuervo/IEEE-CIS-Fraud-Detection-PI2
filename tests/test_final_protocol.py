"""Protege reglas, datos congelados e imposibilidad de repetir test sin revisar."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fraud_cost import final_protocol as final
from fraud_cost.artifacts import notebook_code_sha256, sha256_file


def protocol():
    adoption = {"validation_windows": {"calibration_rows": 42046}}
    chosen = {"candidate": "lightgbm_referencia", "params": {"num_leaves": 63}}
    calibrator = SimpleNamespace(coef_=np.array([[1.1]]), intercept_=np.array([.1]))
    return final.build_protocol(adoption, chosen, calibrator, {"rows": 84233, "labels_read": False}, {}, "code")


def test_protocol_separates_technical_freeze_from_operational_costs_and_review():
    p = protocol()
    assert p["status"] == "frozen_for_academic_evaluation"
    assert not p["review"]["professor_approval_of_final_protocol_recorded"]
    assert not p["costs"]["operational_costs_estimated"]
    assert p["costs"]["ca_grid"] == [1, 5, 10, 20, 50, 100]
    assert p["calibration"]["fit_rows"] == 42046
    assert not p["reporting"]["refit_with_validation"]


@pytest.mark.parametrize("change", ["free_true_positive", "new_ca", "new_threshold", "hide_test_exposure",
                                   "new_encoding", "new_calibrator", "test_already_evaluated"])
def test_protocol_rejects_changed_rules_and_undeclared_exposure(change):
    p = deepcopy(protocol())
    if change == "free_true_positive": p["costs"]["matrix"]["intervene_fraud"] = 0
    elif change == "new_ca": p["costs"]["ca_grid"].append(2.5)
    elif change == "new_threshold": p["policies"]["fixed_0_5"] = "p >= 0.2"
    elif change == "hide_test_exposure": p["limitations"]["historical_test_label_aggregates_consulted"] = False
    elif change == "new_encoding": p["representation"]["encoding"] = "onehot"
    elif change == "new_calibrator": p["calibration"]["method"] = "isotonic"
    else: p["test_evaluated"] = True
    with pytest.raises(ValueError): final.validate_decisions(p)


@pytest.fixture
def frozen(tmp_path):
    (tmp_path / "results/tables").mkdir(parents=True)
    notebook = tmp_path / final.NOTEBOOK
    notebook.parent.mkdir(parents=True)
    notebook.write_text(json.dumps({"cells": [{"cell_type": "code", "source": ["pass\n"]}]}), encoding="utf-8")
    source = tmp_path / "source.py"
    source.write_text("version = 1\n", encoding="utf-8")
    p = protocol()
    p["notebook_code_sha256"] = notebook_code_sha256(notebook)
    p["dependency_sha256"] = {"source.py": sha256_file(source)}
    (tmp_path / final.PROTOCOL).write_text(json.dumps(p), encoding="utf-8")
    record = {"status": "complete", "technical_protocol_frozen": True, "test_evaluated": False,
              "output_sha256": {final.PROTOCOL: sha256_file(tmp_path / final.PROTOCOL)}}
    (tmp_path / final.RECORD).write_text(json.dumps(record), encoding="utf-8")
    return tmp_path


def test_frozen_protocol_is_idempotent_and_does_not_repeat_preparation(frozen, monkeypatch):
    before = (frozen / final.PROTOCOL).read_bytes()
    def forbidden(*args): raise AssertionError("No repetir la preparación de un protocolo congelado")
    monkeypatch.setattr(final, "gather_evidence", forbidden)
    assert final.freeze_protocol(frozen)["test"]["rows"] == 84233
    assert (frozen / final.PROTOCOL).read_bytes() == before


def test_changed_source_or_frozen_file_stops_evaluation(frozen):
    source = frozen / "source.py"
    source.write_text("version = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash distinto"): final.verify_before_test(frozen)


def test_test_run_record_prevents_silent_second_evaluation(frozen):
    assert final.verify_before_test(frozen)["test"]["labels_read"] is False
    (frozen / final.TEST_RECORD).write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Ya existe"): final.verify_before_test(frozen)
