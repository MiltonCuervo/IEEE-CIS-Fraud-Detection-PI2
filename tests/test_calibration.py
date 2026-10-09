import joblib
import numpy as np
import pandas as pd
import pytest

from fraud_cost.calibration import (DIAGNOSTIC_CA, aligned_policy, bmr_calibrator_diagnostic,
                                    compare_calibrators, fit_isotonic, group_diagnostics,
                                    isotonic_probabilities)
from fraud_cost.costs import bmr_actions, realized_cost
from fraud_cost.refit import calibrated_probabilities, fit_sigmoid


class NoRefit:
    def fit(self, *args, **kwargs):
        pytest.fail("No se puede reajustar el predictor o el preprocesamiento")


class FrozenStub:
    def __init__(self, early):
        self.train_end = 0
        self.calibrator = fit_sigmoid(early.stub_raw,early.isFraud)
        self.model = NoRefit()
        self.preprocessor = NoRefit()

    def predict_raw(self, frame):
        return frame.stub_raw.to_numpy(float)


def experiment():
    early = pd.DataFrame({"TransactionID":range(12), "TransactionDT":8*86400+np.arange(12),
                          "TransactionAmt":np.linspace(10,200,12), "isFraud":[0,0,1,0,0,1,0,1,0,1,1,1],
                          "stub_raw":np.linspace(.01,.99,12), "split":"validation"})
    policy = pd.DataFrame({"TransactionID":range(100,108), "TransactionDT":9*86400+np.arange(8),
                           "TransactionAmt":[0.,50.,100.,250.,500.,1000.,20.,200.],
                           "isFraud":[0,1,0,1,0,1,0,1], "stub_raw":[0,.05,.2,.5,.6,1.,.01,.8],"split":"validation"})
    predictor = FrozenStub(early)
    saved = policy[["TransactionID","TransactionDT","TransactionAmt","isFraud"]].copy()
    saved["probability_raw"] = policy.stub_raw
    saved["probability_calibrated"] = calibrated_probabilities(predictor.calibrator,policy.stub_raw)
    saved["action_fixed_0_5"] = np.where(saved.probability_calibrated >= .5,"block","approve")
    return predictor, early, policy, saved


def run(predictor,early,policy,saved):
    return compare_calibrators(predictor,early,policy,saved,ca_grid=(1,2.5,10),progress=lambda *a,**k:None)


def test_isotonic_monotonic_bounded_and_clips_unseen_endpoints():
    model = fit_isotonic([.1,.2,.3,.4,.5,.6],[0,1,0,1,0,1])
    values = isotonic_probabilities(model,np.linspace(0,1,100))
    assert np.isfinite(values).all() and (np.diff(values) >= 0).all()
    assert values.min() >= 0 and values.max() <= 1
    assert values[0] == model.predict([.1])[0] and values[-1] == model.predict([.6])[0]
    assert len(np.unique(values)) < len(values)


def test_isotonic_rejects_invalid_labels_probabilities_and_length():
    for p,y in (([np.nan,.5],[0,1]),([-.1,.5],[0,1]),([.1,1.1],[0,1]),
                ([.1,.5],[0,2]),([.1,.5],[0,0]),([.1,.5],[0])):
        with pytest.raises(ValueError):
            fit_isotonic(p,y)
    assert np.allclose(isotonic_probabilities(fit_isotonic([.5,.5],[0,1]),[0,.5,1]),.5)


def test_comparison_does_not_refit_base_predictor_or_existing_platt():
    predictor,early,policy,saved = experiment()
    before = joblib.hash(predictor)
    result = run(predictor,early,policy,saved)
    assert joblib.hash(predictor) == before
    np.testing.assert_array_equal(result[1].probability_raw,saved.probability_raw)
    np.testing.assert_array_equal(result[1].probability_calibrated,saved.probability_calibrated)
    assert set(result[2].method)=={"raw","platt","isotonic"}
    assert len(result[3])==51 and len(result[4])==12 and len(result[5])==3


def test_posterior_labels_do_not_change_calibrator_or_predictions():
    predictor,early,policy,saved = experiment()
    first = run(predictor,early,policy,saved)
    changed = policy.assign(isFraud=1-policy.isFraud)
    second = run(predictor,early,changed,saved.assign(isFraud=1-saved.isFraud))
    np.testing.assert_array_equal(first[0].X_thresholds_,second[0].X_thresholds_)
    np.testing.assert_array_equal(first[0].y_thresholds_,second[0].y_thresholds_)
    for column in ("probability_raw","probability_calibrated","probability_isotonic"):
        np.testing.assert_array_equal(first[1][column],second[1][column])
    pd.testing.assert_frame_equal(first[4][["strategy","administrative_cost_ca","policy_block_count"]],
                                  second[4][["strategy","administrative_cost_ca","policy_block_count"]])


def test_comparison_rejects_test_simultaneous_overlap_and_wrong_gap():
    predictor,early,policy,saved = experiment()
    for invalid in (early.assign(split="test"),early.assign(TransactionDT=policy.TransactionDT.min()),
                    early.assign(TransactionDT=100),early.assign(TransactionID=100)):
        with pytest.raises(ValueError):
            run(predictor,invalid,policy,saved)
    with pytest.raises(ValueError):
        run(predictor,early,policy.assign(split="test"),saved)


def test_changed_saved_population_or_probabilities_are_rejected():
    predictor,early,policy,saved = experiment()
    with pytest.raises(ValueError):
        aligned_policy(policy,saved.iloc[::-1])
    with pytest.raises(ValueError):
        aligned_policy(policy,saved.assign(TransactionAmt=10))
    with pytest.raises(AssertionError):
        run(predictor,early,policy,saved.assign(probability_raw=.2))
    with pytest.raises(ValueError):
        aligned_policy(policy,saved.assign(TransactionID=0))


def test_group_population_is_common_for_all_methods_and_empty_groups_are_missing_rates():
    result = run(*experiment())
    scores, groups = result[1],result[3]
    for grouping in ("all","amount","risk_platt","period"):
        population = []
        for method in ("raw","platt","isotonic"):
            part = groups.loc[groups.grouping.eq(grouping)&groups.method.eq(method)]
            assert part.rows.sum()==len(scores)
            assert part.frauds.sum()==scores.isFraud.sum()
            population.append(part[["group_index","rows","frauds"]].reset_index(drop=True))
        pd.testing.assert_frame_equal(population[0],population[1])
        pd.testing.assert_frame_equal(population[1],population[2])
    assert groups.loc[groups.rows.eq(0),"log_loss"].isna().all()
    # Grupos con una sola clase pueden evaluarse sin redefinir las etiquetas.
    assert np.isfinite(groups.loc[groups.rows.gt(0),"log_loss"]).all()


def test_extreme_predictions_and_log_loss_are_reported_not_silently_smoothed():
    _,scores,metrics,*_ = run(*experiment())
    iso = metrics.loc[metrics.method.eq("isotonic")].iloc[0]
    assert iso.probability_zero_count > 0 and iso.probability_one_count > 0
    assert iso.unique_probabilities < len(scores) and np.isfinite(iso.log_loss)
    assert iso.probability_zero_count == scores.probability_isotonic.eq(0).sum()
    assert iso.probability_one_count == scores.probability_isotonic.eq(1).sum()


def test_bmr_matrix_fixed_reference_and_delta_components_are_consistent():
    _,scores,_,_,comparison,changes,_ = run(*experiment())
    for ca,part in comparison.groupby("administrative_cost_ca"):
        fixed = realized_cost(scores.isFraud,scores.TransactionAmt,scores.action_fixed_0_5,ca).sum()
        assert part.fixed_baseline_cost_proxy.eq(fixed).all() and part.reference_cost.nunique()==1
        for method,column in (("raw","probability_raw"),("platt","probability_calibrated"),("isotonic","probability_isotonic")):
            actual = realized_cost(scores.isFraud,scores.TransactionAmt,bmr_actions(scores[column],scores.TransactionAmt,ca),ca).sum()
            assert part.loc[part.strategy.eq(f"bmr_{method}"),"cost_total_proxy"].iloc[0] == pytest.approx(actual)
    np.testing.assert_allclose(changes.isotonic_minus_platt_cost,
                               changes.administrative_cost_delta+changes.missed_fraud_amount_delta)
    assert (changes.actions_changed==changes.additional_interventions+changes.fewer_interventions).all()


def test_diagnostic_grid_preserves_history_and_distinguishes_new_2_5():
    assert DIAGNOSTIC_CA==(1.,2.5,5.,10.,20.,50.,100.)
    _,scores,*_ = run(*experiment())
    table,_ = bmr_calibrator_diagnostic(scores)
    assert len(table)==28
    assert table.loc[table.administrative_cost_ca.eq(2.5),"scenario_group"].eq("inspirado_en_literatura").all()
    assert table.loc[table.administrative_cost_ca.eq(100),"scenario_group"].eq("sensibilidad_ampliada").all()
