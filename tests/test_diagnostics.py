import json

import numpy as np
import pandas as pd
import pytest

from fraud_cost.costs import evaluate_policies
from fraud_cost.diagnostics import (AMOUNT_EDGES, RISK_EDGES, bands, boundary_diagnostic,
                                    conditional_calibration, temporal_diagnostic, verified_diagnostic_inputs)
from fraud_cost.economics import calibration_action_diagnostic
from fraud_cost.refit import source_sha256


def scores_and_actions():
    p = np.array([0., .1, .2, .5, .6, 1., .01, .8])
    raw = np.array([0., .15, .25, .4, .65, 1., .02, .75])
    scores = pd.DataFrame({"TransactionID": np.arange(8), "TransactionDT": [10,10,20,30,40,50,60,70],
                          "TransactionAmt": [0.,50.,100.,250.,500.,1000.,20.,200.],
                          "isFraud": [0,1,0,1,0,1,0,1], "probability_raw": raw,
                          "probability_calibrated": p, "action_fixed_0_5": np.where(p >= .5,"block","approve")})
    actions = scores[["TransactionID"]].copy()
    actions["fixed_0_5"] = p >= .5
    for ca in (1.,10.):
        actions[f"bmr_ca_{ca:g}"] = p*scores.TransactionAmt > ca
        actions[f"raw_bmr_ca_{ca:g}"] = raw*scores.TransactionAmt > ca
        actions[f"tree_ca_{ca:g}"] = scores.TransactionAmt > 100
    return scores, actions


def test_band_limits_zero_one_and_exact_edges():
    np.testing.assert_array_equal(bands([0,50,100,250,500,1000], AMOUNT_EDGES), np.arange(6))
    np.testing.assert_array_equal(bands([0,.01,.05,.1,.25,.5,1], RISK_EDGES), [0,1,2,3,4,5,5])
    for invalid in ([-1], [np.nan], [np.inf]):
        with pytest.raises(ValueError):
            bands(invalid, AMOUNT_EDGES)
    with pytest.raises(ValueError):
        bands([1.01], RISK_EDGES)


def test_groups_cover_population_and_preserve_empty_cells():
    scores, _ = scores_and_actions()
    table = conditional_calibration(scores)
    assert len(table) == (6+6+36)*2
    for grouping in ("amount", "risk", "amount_risk"):
        for name in ("probability_raw", "probability_calibrated"):
            part = table.loc[table.grouping.eq(grouping) & table.probabilities.eq(name)]
            assert part.rows.sum() == len(scores)
            assert part.frauds.sum() == scores.isFraud.sum()
            assert part.fraud_amount_observed.sum() == (scores.isFraud*scores.TransactionAmt).sum()
            assert part.fraud_amount_expected.sum() == pytest.approx((scores[name]*scores.TransactionAmt).sum())
    assert table.loc[table.rows.eq(0), "observed_rate"].isna().all()
    assert table.loc[table.rows.eq(0), "brier_score"].isna().all()
    assert table.loc[table.rows.eq(0), "fraud_amount_observed"].eq(0).all()


def test_raw_and_calibrated_use_same_groups_not_their_own_bins():
    scores, _ = scores_and_actions()
    table = conditional_calibration(scores)
    a = table.loc[table.probabilities.eq("probability_raw")].reset_index(drop=True)
    b = table.loc[table.probabilities.eq("probability_calibrated")].reset_index(drop=True)
    pd.testing.assert_frame_equal(a[["grouping","amount_band","risk_band","rows","frauds"]],
                                  b[["grouping","amount_band","risk_band","rows","frauds"]])


def test_changing_labels_does_not_change_groups_or_actions():
    scores, actions = scores_and_actions()
    changed = scores.assign(isFraud=1-scores.isFraud)
    a, b = conditional_calibration(scores), conditional_calibration(changed)
    pd.testing.assert_frame_equal(a[["grouping","amount_band","risk_band","rows","mean_probability"]],
                                  b[["grouping","amount_band","risk_band","rows","mean_probability"]])
    a = temporal_diagnostic(scores, actions, ca_grid=(1,10))[1]
    b = temporal_diagnostic(changed, actions, ca_grid=(1,10))[1]
    pd.testing.assert_frame_equal(a[["period","strategy","administrative_cost_ca","policy_block_count"]],
                                  b[["period","strategy","administrative_cost_ca","policy_block_count"]])


def test_boundary_totals_match_previous_calibration_diagnostic():
    scores, _ = scores_and_actions()
    table = boundary_diagnostic(scores, ca_grid=(1,10))
    baseline = calibration_action_diagnostic(scores, ca_grid=(1,10)).set_index("administrative_cost_ca")
    fields = ["actions_changed","additional_interventions","fewer_interventions","raw_bmr_cost",
              "calibrated_bmr_cost","calibrated_minus_raw_cost","administrative_cost_delta","missed_fraud_amount_delta"]
    for ca, part in table.groupby("administrative_cost_ca"):
        assert part.rows.sum() == len(scores)
        for field in fields:
            assert part[field].sum() == pytest.approx(baseline.loc[ca, field])
        np.testing.assert_allclose(part.calibrated_minus_raw_cost,
                                   part.administrative_cost_delta+part.missed_fraud_amount_delta)


def test_margin_is_descriptive_and_bmr_approves_ties_and_infeasible_amounts():
    scores, _ = scores_and_actions()
    scores["TransactionAmt"] = [0., 10., 45., 20., 15., 10., 100., 100.]
    # q=0.9 entra en cercanía; q=1.1 sale; q=1 aprueba incluso con p=1,A=Ca.
    table = boundary_diagnostic(scores, ca_grid=(10,))
    assert table.rows.sum() == len(scores)
    near = table.loc[table.near_boundary].iloc[0]
    assert near.rows == 4 and near.calibrated_block_count == 0
    assert near.amount_le_ca_count == 1


def test_temporal_costs_add_up_but_reference_minimum_is_local():
    scores, actions = scores_and_actions()
    population, table = temporal_diagnostic(scores, actions, ca_grid=(1,10))
    for name in ("probability_raw", "probability_calibrated"):
        assert population.loc[population.probabilities.eq(name), "rows"].sum() == len(scores)
    for ca in (1,10):
        policies = {"fixed_0_5": np.where(actions.fixed_0_5,"block","approve"),
                    "bmr": np.where(actions[f"bmr_ca_{ca}"],"block","approve"),
                    "cost_sensitive_tree": np.where(actions[f"tree_ca_{ca}"],"block","approve")}
        global_table = evaluate_policies(scores.isFraud,scores.TransactionAmt,policies,ca).set_index("strategy")
        for strategy, part in table.loc[table.administrative_cost_ca.eq(ca)].groupby("strategy"):
            for field in ("cost_total_proxy","administrative_cost_total","missed_fraud_amount","policy_block_count"):
                assert part[field].sum() == pytest.approx(global_table.loc[strategy,field])
            assert part.reference_cost.sum() <= global_table.loc[strategy,"reference_cost"]+1e-12
    # Empates de tiempo permanecen en el mismo intervalo, incluido el máximo.
    assert population.loc[population.probabilities.eq("probability_raw") & population.period.eq(1),"rows"].iloc[0] == 3
    assert population.loc[population.probabilities.eq("probability_raw") & population.period.eq(4),"rows"].iloc[0] == 2


def test_temporal_handles_empty_or_single_class_periods():
    scores, actions = scores_and_actions()
    scores["TransactionDT"] = [0,0,0,0,100,100,100,100]
    population, table = temporal_diagnostic(scores, actions, ca_grid=(1,10))
    assert set(table.period) == {1,4}
    assert population.loc[population.period.isin([2,3]),"rows"].eq(0).all()
    assert population.loc[population.period.isin([2,3]),"observed_rate"].isna().all()
    with pytest.raises(ValueError):
        temporal_diagnostic(scores.assign(TransactionDT=0), actions, ca_grid=(1,10))


def test_invalid_scores_and_misaligned_actions_are_rejected():
    scores, actions = scores_and_actions()
    for invalid in (scores.assign(probability_raw=np.nan), scores.assign(TransactionDT=np.inf),
                    scores.assign(TransactionID=0), scores.assign(action_fixed_0_5="approve")):
        with pytest.raises(ValueError):
            conditional_calibration(invalid)
    with pytest.raises(ValueError):
        temporal_diagnostic(scores, actions.iloc[::-1], ca_grid=(1,10))
    with pytest.raises(ValueError):
        temporal_diagnostic(scores, actions.assign(bmr_ca_10=False), ca_grid=(1,10))
    with pytest.raises(ValueError):
        temporal_diagnostic(scores, actions.assign(tree_ca_10="approve"), ca_grid=(1,10))


def test_changed_source_is_rejected_before_loading_scores(tmp_path, monkeypatch):
    (tmp_path/"results/tables").mkdir(parents=True)
    source = tmp_path/"input.txt"
    source.write_text("original",encoding="utf-8")
    record = {"test_evaluated":False, "source_sha256":{"input.txt":source_sha256(source)},
              "artifact_sha256":{"input.txt":source_sha256(source)}}
    (tmp_path/"results/tables/reajuste_config.json").write_text(json.dumps(record),encoding="utf-8")
    source.write_text("changed",encoding="utf-8")
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: pytest.fail("No debe cargar scores con fuentes distintas"))
    with pytest.raises(ValueError):
        verified_diagnostic_inputs(tmp_path)
