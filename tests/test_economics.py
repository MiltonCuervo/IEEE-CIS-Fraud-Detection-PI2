import json

import numpy as np
import pandas as pd
import pytest

from fraud_cost.costs import realized_cost
from fraud_cost.economics import (calibration_action_diagnostic, checked_ca_grid,
                                  compare_economic_policies, prepare_cost_matrices, verified_refit)
from fraud_cost.refit import fit_selected, source_sha256, split_validation


def example_scores():
    p = np.array([.01,.10,.40,.70,.90,.05,.30,.80])
    return pd.DataFrame({"TransactionID": np.arange(1001,1009), "TransactionDT": np.arange(2001,2009),
                         "TransactionAmt": [20.,100.,200.,50.,10.,500.,40.,150.],
                         "isFraud": [0,1,1,0,1,0,0,1], "probability_raw": p,
                         "probability_calibrated": p * .8, "action_fixed_0_5": np.where(p*.8 >= .5, "block", "approve")})


def example_train():
    n = 600
    amount = np.linspace(1,300,n)
    train = pd.DataFrame({"TransactionID": np.arange(n), "TransactionDT": np.arange(n),
                          "TransactionAmt": amount, "isFraud": (amount > 150).astype(int), "split":"train"})
    return train, amount.reshape(-1,1)


def test_grid_is_positive_unique_and_ordered():
    assert checked_ca_grid([1,5,10]) == (1.,5.,10.)
    for invalid in ([], [0], [-1], [1,1], [5,1], [float("nan")]):
        with pytest.raises(ValueError):
            checked_ca_grid(invalid)


def test_three_policies_share_population_matrix_and_expected_risk():
    train, Xt = example_train()
    scores = example_scores()
    comparison, diagnostic, _, _, actions = compare_economic_policies(
        train, scores, Xt, scores.TransactionAmt.to_numpy().reshape(-1,1), ca_grid=(1,10), progress=lambda *a, **k: None)
    assert len(comparison) == 6 and len(diagnostic) == 2
    for ca in (1,10):
        block = comparison.loc[comparison.administrative_cost_ca.eq(ca)]
        assert set(block.strategy) == {"fixed_0_5","bmr","cost_sensitive_tree"}
        assert block.reference_cost.nunique() == 1 and block.rows.eq(len(scores)).all()
        bmr_risk = block.loc[block.strategy.eq("bmr"), "expected_cost_common_probabilities"].iloc[0]
        assert (block.expected_cost_common_probabilities >= bmr_risk - 1e-12).all()
        for strategy, column in (("fixed_0_5","fixed_0_5"), ("bmr",f"bmr_ca_{ca}"), ("cost_sensitive_tree",f"tree_ca_{ca}")):
            total = realized_cost(scores.isFraud, scores.TransactionAmt, np.where(actions[column], "block", "approve"), ca).sum()
            assert total == block.loc[block.strategy.eq(strategy), "cost_total_proxy"].iloc[0]


def test_policy_labels_never_change_actions_or_tree_training():
    train, Xt = example_train()
    scores = example_scores()
    changed = scores.copy()
    changed["isFraud"] = 1 - changed.isFraud
    arguments = dict(ca_grid=(1,10), progress=lambda *a, **k: None)
    a = compare_economic_policies(train, scores, Xt, scores.TransactionAmt.to_numpy().reshape(-1,1), **arguments)
    b = compare_economic_policies(train, changed, Xt, scores.TransactionAmt.to_numpy().reshape(-1,1), **arguments)
    pd.testing.assert_frame_equal(a[-1], b[-1])
    for ca in a[-2]:
        np.testing.assert_array_equal(a[-2][ca].tree_.threshold, b[-2][ca].tree_.threshold)


def test_calibration_cost_delta_is_exactly_decomposed():
    table = calibration_action_diagnostic(example_scores(), ca_grid=(1,10,50))
    np.testing.assert_allclose(table.calibrated_minus_raw_cost, table.administrative_cost_delta + table.missed_fraud_amount_delta)
    assert (table.actions_changed == table.additional_interventions + table.fewer_interventions).all()


def test_comparison_rejects_external_rows_and_overlap():
    train, Xt = example_train()
    scores = example_scores()
    Xv = scores.TransactionAmt.to_numpy().reshape(-1,1)
    with pytest.raises(ValueError):
        compare_economic_policies(train.assign(split="test"), scores, Xt, Xv)
    overlap = scores.copy()
    overlap.loc[0,"TransactionID"] = 0
    with pytest.raises(ValueError):
        compare_economic_policies(train, overlap, Xt, Xv)


def test_changed_source_is_rejected_before_joblib_load(tmp_path, monkeypatch):
    (tmp_path / "results/tables").mkdir(parents=True)
    source = tmp_path / "source.txt"
    source.write_text("original", encoding="utf-8")
    record = {"test_evaluated": False, "source_sha256": {"source.txt": source_sha256(source)},
              "artifact_sha256": {"source.txt": source_sha256(source)}}
    (tmp_path / "results/tables/reajuste_config.json").write_text(json.dumps(record), encoding="utf-8")
    source.write_text("changed", encoding="utf-8")
    def forbidden_load(*args, **kwargs):
        pytest.fail("No debe cargar un artefacto antes de comprobar sus hashes")
    monkeypatch.setattr("fraud_cost.economics.joblib.load", forbidden_load)
    with pytest.raises(ValueError):
        verified_refit(tmp_path)


def test_reconstructed_matrices_use_frozen_preprocessor_and_history():
    n = 200
    frame = pd.DataFrame({"TransactionID": np.arange(n), "TransactionDT": np.arange(n)*86400,
                          "TransactionAmt": np.where(np.arange(n)%2,100.,10.), "isFraud": np.arange(n)%2,
                          "DeviceInfo": np.where(np.arange(n)%2,"a","b"), "split":"train"})
    train = frame.iloc[:120].copy()
    validation = frame.iloc[140:].copy().assign(split="validation")
    early, policy = split_validation(validation)
    selection = {"train_rows":len(train), "train_end":int(train.TransactionDT.max()),
                 "selected_candidate":{"features":"history"},
                 "selected_params":{"n_estimators":3,"max_depth":2,"min_child_weight":1,"n_jobs":1,"random_state":42}}
    predictor, raw, calibrated, _, _ = fit_selected(train, early, policy, selection, progress=lambda *a, **k: None)
    scores = policy[["TransactionID","TransactionDT","TransactionAmt","isFraud"]].copy()
    scores["probability_raw"], scores["probability_calibrated"] = raw, calibrated
    scores["action_fixed_0_5"] = np.where(calibrated >= .5, "block", "approve")
    state = predictor.builder.states_["device_proxy"].copy()
    Xt, Xv = prepare_cost_matrices(predictor, train, policy, scores)
    assert Xt.shape[0] == len(train) and Xv.shape[0] == len(policy)
    np.testing.assert_allclose(predictor.model.predict_proba(Xv)[:,1], raw)
    pd.testing.assert_frame_equal(state, predictor.builder.states_["device_proxy"])
    with pytest.raises(ValueError):
        prepare_cost_matrices(predictor, train, policy.assign(split="test"), scores)
