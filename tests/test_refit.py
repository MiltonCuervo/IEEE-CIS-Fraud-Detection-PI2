import json

import joblib
import numpy as np
import pandas as pd
import pytest

from fraud_cost.refit import (
    calibrated_probabilities, check_blocks, clipped_logit, fit_selected,
    fit_sigmoid, load_development, predictive_metrics, reliability_table,
    source_sha256, split_validation, verified_selection,
)


def blocks():
    n = 200
    frame = pd.DataFrame({"TransactionID": np.arange(n), "TransactionDT": np.arange(n) * 86400,
                          "TransactionAmt": np.where(np.arange(n) % 2, 100., 10.),
                          "isFraud": np.arange(n) % 2, "split": "train",
                          "DeviceInfo": np.where(np.arange(n) % 2, "a", "b")})
    train = frame.iloc[:120].copy()
    validation = frame.iloc[140:].copy()
    validation["split"] = "validation"
    early, late = split_validation(validation)
    return train, early, late


def test_split_keeps_simultaneous_batch_in_calibration():
    validation = pd.DataFrame({"TransactionID": range(6), "TransactionDT": [1,2,3,3,4,5], "split": "validation"})
    early, late = split_validation(validation)
    assert early.TransactionDT.tolist() == [1,2,3,3]
    assert late.TransactionDT.tolist() == [4,5]
    with pytest.raises(ValueError):
        split_validation(validation.assign(TransactionDT=1))


def test_block_checks_reject_test_overlap_and_wrong_time():
    train, early, late = blocks()
    assert check_blocks(train, early, late).rows.tolist() == [120,30,30]
    with pytest.raises(ValueError):
        check_blocks(train, early, late.assign(split="test"))
    invalid = late.copy()
    invalid.loc[invalid.index[0], "TransactionID"] = train.TransactionID.iloc[0]
    with pytest.raises(ValueError):
        check_blocks(train, early, invalid)
    with pytest.raises(ValueError):
        check_blocks(train, late, early)


def test_logits_and_sigmoid_validate_inputs_and_preserve_order():
    p = np.linspace(.01, .99, 80)
    y = (p > .5).astype(int)
    calibrator = fit_sigmoid(p, y)
    mapped = calibrated_probabilities(calibrator, p)
    assert np.isfinite(clipped_logit([0,1])).all()
    assert calibrator.coef_[0,0] > 0
    assert (np.diff(mapped) >= 0).all()
    for invalid in ([np.nan], [-.1], [1.1], [[.5]]):
        with pytest.raises(ValueError):
            clipped_logit(invalid)
    with pytest.raises(ValueError):
        fit_sigmoid(p, np.zeros(len(p)))


def test_metrics_match_confusion_matrix_at_fixed_threshold():
    metrics = predictive_metrics([0,0,1,1], [.1,.5,.4,.9], name="calibrated")
    assert [metrics[name] for name in ("tn", "fp", "fn", "tp")] == [1,1,1,1]
    assert metrics["recall_at_fixed_0_5"] == .5
    assert metrics["legitimate_block_rate_at_fixed_0_5"] == .5
    with pytest.raises(ValueError):
        predictive_metrics([0,1], [.1], name="raw")


def test_reliability_covers_constant_and_tied_probabilities():
    for p in ([.2]*4, [.1,.1,.8,.8]):
        table = reliability_table([0,0,1,1], p, name="raw")
        assert table.rows.sum() == 4 and table.frauds.sum() == 2
        assert table.mean_probability.between(0,1).all()


def test_loader_excludes_test_and_gaps_before_join(tmp_path):
    (tmp_path / "data/raw").mkdir(parents=True)
    (tmp_path / "data/interim").mkdir(parents=True)
    frame = pd.DataFrame({"TransactionID": [1,2,3,4], "TransactionDT": [1,2,3,4],
                          "TransactionAmt": [1.,2.,3.,4.], "isFraud": [0,1,999,999]})
    frame.to_csv(tmp_path / "data/raw/train_transaction.csv", index=False)
    pd.DataFrame({"TransactionID": [1,3,4], "DeviceInfo": ["train","gap","test"]}).to_csv(tmp_path / "data/raw/train_identity.csv", index=False)
    pd.DataFrame({"TransactionID": [1,2,3,4], "split": ["train","validation","gap_1","test"]}).to_parquet(tmp_path / "data/interim/split_manifest.parquet", index=False)
    train, validation = load_development(tmp_path, chunk_size=2)
    assert train.TransactionID.tolist() == [1] and validation.TransactionID.tolist() == [2]
    assert train.has_identity.iloc[0] == 1 and validation.has_identity.iloc[0] == 0
    assert not train.isFraud.isin([999]).any() and not validation.isFraud.isin([999]).any()


def test_selection_verifies_historical_source_hash(tmp_path):
    (tmp_path / "results/tables").mkdir(parents=True)
    path = tmp_path / "source.txt"
    path.write_text("original", encoding="utf-8")
    record = {"selected_candidate": {"features": "history"}, "source_sha256": {"source.txt": source_sha256(path)}}
    (tmp_path / "results/tables/seleccion_temporal_config.json").write_text(json.dumps(record), encoding="utf-8")
    assert verified_selection(tmp_path)["selected_candidate"]["features"] == "history"
    path.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError):
        verified_selection(tmp_path)


def selection_for(train):
    return {"train_rows": len(train), "train_end": int(train.TransactionDT.max()),
            "selected_candidate": {"features":"history"},
            "selected_params": {"n_estimators": 5, "max_depth": 2, "min_child_weight": 1,
                                "n_jobs": 1, "random_state": 42, "tree_method":"hist"}}


def test_fit_does_not_learn_labels_or_history_from_policy(tmp_path):
    train, early, late = blocks()
    selection = selection_for(train)
    a, raw_a, calibrated_a, _, _ = fit_selected(train, early, late, selection, progress=lambda *a, **k: None)
    changed = late.copy()
    changed["isFraud"] = 1 - changed.isFraud
    b, raw_b, calibrated_b, _, _ = fit_selected(train, early, changed, selection, progress=lambda *a, **k: None)
    np.testing.assert_allclose(raw_a, raw_b)
    np.testing.assert_allclose(calibrated_a, calibrated_b)
    np.testing.assert_allclose(a.calibrator.coef_, b.calibrator.coef_)
    state = a.builder.states_["device_proxy"].copy()
    a.predict_calibrated(late)
    pd.testing.assert_frame_equal(state, a.builder.states_["device_proxy"])
    model_path = tmp_path / "trusted.joblib"
    joblib.dump(a, model_path)
    restored = joblib.load(model_path)
    np.testing.assert_allclose(restored.predict_calibrated(late), calibrated_a)
    with pytest.raises(ValueError):
        restored.predict_raw(train)
    with pytest.raises(ValueError):
        restored.predict_raw(late.assign(unexpected_column=1))
