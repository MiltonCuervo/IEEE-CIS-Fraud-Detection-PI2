"""Una nueva selección nunca debe legitimar artefactos de otro ajuste."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import pytest
from fraud_cost.artifacts import ARTIFACTS, REFERENCE_RECORD

scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location("revalidate_reference_recovery", scripts / "revalidate_reference_recovery.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def evidence():
    original = {p: "original" for p in ARTIFACTS}
    reference = {"params": {"lightgbm": {"num_leaves": 63}}, "output_sha256": original}
    selection = {"selected_candidate": {"model": "lightgbm", "params": {"num_leaves": 63}}}
    previous = {"status": "verified_reconstruction", "test_evaluated": False,
                "historical_records_sha256": {REFERENCE_RECORD: "same03"},
                "artifact_sha256": {p: "reconstructed" for p in ARTIFACTS},
                "original_artifact_sha256": original,
                "numerical_comparison": {"passed": True}, "inference_verification": {"passed": True}}
    return reference, selection, previous


def test_rebinding_accepts_same03_and_matching_configuration():
    module.validate_binding(*evidence(), "same03")


@pytest.mark.parametrize("change", ["changed03", "changed_parameters", "missing_artifact", "unverified_inference"])
def test_rebinding_rejects_a_different_or_incomplete_fit(change):
    reference, selection, previous = deepcopy(evidence())
    if change == "changed03": previous["historical_records_sha256"][REFERENCE_RECORD] = "other03"
    elif change == "changed_parameters": selection["selected_candidate"]["params"]["num_leaves"] = 31
    elif change == "missing_artifact": previous["artifact_sha256"].pop(ARTIFACTS[0])
    else: previous["inference_verification"]["passed"] = False
    with pytest.raises(ValueError): module.validate_binding(reference, selection, previous, "same03")


def test_rebinding_cannot_change_dependencies_after_freeze(tmp_path):
    frozen = tmp_path / "results/tables/protocolo_final.json"
    frozen.parent.mkdir(parents=True)
    frozen.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="protocolo congelado"):
        module.revalidate(tmp_path)


def test_reproduction_cannot_overwrite_frozen_dependencies(tmp_path, monkeypatch):
    import reproduce_validation
    frozen = tmp_path / "results/tables/protocolo_final.json"
    frozen.parent.mkdir(parents=True)
    frozen.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(reproduce_validation, "ROOT", tmp_path)
    with pytest.raises(FileExistsError, match="protocolo congelado"):
        reproduce_validation.main()
