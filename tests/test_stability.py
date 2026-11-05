import numpy as np
import pandas as pd
import pytest
from fraud_cost.costs import bmr_actions, evaluate_policies
from fraud_cost.stability import (POLICIES, validate_evidence, temporal_periods, period_probabilities,
    period_costs, period_amount_calibration, missed_loss_concentration, boundary_sensitivity)


def fixture():
    p=np.array([.1,.6,.2,.05,.9,.4,.01,1.,.8,.1,.01,.2])
    scores=pd.DataFrame({'TransactionID':np.arange(12),'TransactionDT':np.repeat([0,1,2,3,4,6],2),
        'TransactionAmt':[100.,20.,200.,30.,10.,400.,150.,5.,0.,200.,1000.,50.],
        'isFraud':np.tile([1,0],6),'probability_raw':p*.8,'probability_calibrated':p,
        'action_fixed_0_5':np.where(p>=.5,'block','approve')})
    actions=[]
    for ca in [10,20]:
        actions.append(pd.DataFrame({'TransactionID':scores.TransactionID,'administrative_cost_ca':ca,
            'fixed_0_5':scores.action_fixed_0_5,'bmr':bmr_actions(p,scores.TransactionAmt,ca),
            'cost_sensitive_tree':np.where(scores.TransactionID%3==0,'block','approve')}))
    return scores,pd.concat(actions,ignore_index=True)


def test_time_periods_keep_equal_instants_and_exact_boundaries():
    scores,_=fixture()
    data,windows=temporal_periods(scores)
    assert windows.rows.tolist()==[4,4,4]
    assert data.loc[data.TransactionDT.eq(2),'period'].eq(2).all()
    assert data.loc[data.TransactionDT.eq(6),'period'].eq(3).all()
    assert data.groupby('TransactionDT').period.nunique().eq(1).all()
    with pytest.raises(ValueError):temporal_periods(scores.assign(TransactionDT=1))


def test_actions_are_aligned_by_id_and_reject_changed_or_missing_operations():
    scores,actions=fixture()
    expected=validate_evidence(scores,actions,[10,20])
    shuffled=validate_evidence(scores,actions.sample(frac=1,random_state=4),[10,20])
    pd.testing.assert_frame_equal(shuffled[10],expected[10])
    with pytest.raises(ValueError):validate_evidence(scores,actions.iloc[1:],[10,20])
    invalid=actions.copy();invalid.loc[0,'bmr']='block' # pA=Ca: aprobar.
    with pytest.raises(ValueError,match='BMR'):validate_evidence(scores,invalid,[10,20])
    with pytest.raises(ValueError):validate_evidence(scores,pd.concat([actions,actions.iloc[:1]]),[10,20])


def test_period_costs_and_amount_bands_conserve_population_and_loss():
    scores,actions=fixture();aligned=validate_evidence(scores,actions,[10,20])
    data,windows=temporal_periods(scores)
    costs=period_costs(data,aligned)
    for ca,block in aligned.items():
        whole=evaluate_policies(scores.isFraud,scores.TransactionAmt,{p:block[p] for p in POLICIES},ca)
        for p in POLICIES:
            subset=costs.loc[costs.administrative_cost_ca.eq(ca)&costs.strategy.eq(p)]
            row=whole.loc[whole.strategy.eq(p)].iloc[0]
            assert subset.rows.sum()==len(scores)
            assert subset.cost_total_proxy.sum()==pytest.approx(row.cost_total_proxy)
            assert subset.missed_fraud_amount.sum()==pytest.approx(row.missed_fraud_amount)
    bands=period_amount_calibration(data,[0,50,100,250,500,1000,np.inf])
    assert bands.rows.sum()==len(scores)
    assert bands.observed_fraud_amount.sum()==pytest.approx((scores.isFraud*scores.TransactionAmt).sum())
    metrics=period_probabilities(data)
    assert len(metrics)==6 and metrics.groupby('probability').rows.sum().eq(len(scores)).all()
    one_class=period_probabilities(data.assign(isFraud=0))
    assert one_class.average_precision.isna().all() and one_class.recall_at_0_5.isna().all()


def test_loss_concentration_and_empty_missed_loss_have_explicit_denominators():
    scores,actions=fixture();aligned=validate_evidence(scores,actions,[10,20])
    table=missed_loss_concentration(scores,aligned)
    r=table.loc[table.administrative_cost_ca.eq(10)&table.strategy.eq('bmr')&table.top_k_requested.eq(1)].iloc[0]
    # En empate se aprueban montos 100 y 1000; otros fraudes aprobados: 10,150,0.
    assert r.missed_fraud_amount==1260 and r.top_k_missed_amount==1000
    assert r.fraction_of_missed_amount==pytest.approx(1000/1260)
    everything={ca:b.assign(bmr='block') for ca,b in aligned.items()}
    empty=missed_loss_concentration(scores,everything)
    assert empty.loc[empty.strategy.eq('bmr'),'fraction_of_missed_amount'].isna().all()


def test_probability_perturbation_keeps_ties_clipping_and_base_policy():
    scores,actions=fixture();aligned=validate_evidence(scores,actions,[10,20])
    frontier,sensitivity=boundary_sensitivity(scores,aligned)
    base=sensitivity.loc[sensitivity.probability_factor.eq(1)]
    assert base.changed_actions.eq(0).all() and base.cost_difference_vs_unperturbed.eq(0).all()
    assert frontier.loc[frontier.administrative_cost_ca.eq(10),'exact_ties'].iloc[0]==3
    higher=sensitivity.loc[sensitivity.administrative_cost_ca.eq(10)&sensitivity.probability_factor.eq(1.1)].iloc[0]
    assert higher.new_blocks==3 and higher.new_approvals==0
    with pytest.raises(ValueError):boundary_sensitivity(scores,aligned,factors=(.9,1.1))
