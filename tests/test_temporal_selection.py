import numpy as np
import pandas as pd
import pytest
from fraud_cost.temporal_selection import expanding_folds,describe_folds,rank_candidates,threshold_diagnostics


def train_fixture():
    return pd.DataFrame({'TransactionID':np.arange(220),'TransactionDT':np.repeat(np.arange(110)*86400,2),
                         'isFraud':np.tile([0,1],110),'split':'train'})


def test_expanding_windows_respect_gap_equal_times_and_outer_scope():
    train=train_fixture()
    folds=expanding_folds(train)
    for f in folds:
        fit=train.iloc[f['fit']];future=train.iloc[f['evaluate']]
        assert set(fit.TransactionID).isdisjoint(future.TransactionID)
        assert future.TransactionDT.min()-fit.TransactionDT.max()>7*86400
        for t in future.TransactionDT.unique():
            assert (future.TransactionDT.eq(t).sum()==train.TransactionDT.eq(t).sum())
    assert set(folds[0]['fit']).issubset(folds[1]['fit'])
    assert set(folds[0]['evaluate']).isdisjoint(folds[1]['evaluate'])
    assert len(describe_folds(train,folds))==4
    invalid=train.copy();invalid.loc[0,'split']='validation'
    with pytest.raises(ValueError,match='Solo'):expanding_folds(invalid)
    with pytest.raises(ValueError):expanding_folds(train,gap_days=100)
    with pytest.raises(ValueError):expanding_folds(train,fractions=(.75,.5))
    with pytest.raises(ValueError):expanding_folds(train.iloc[::-1])


def test_selection_requires_complete_grid_and_uses_mean_not_best_fold():
    candidates=[{'candidate':'a'},{'candidate':'b'}]
    results=pd.DataFrame({'candidate':['a','a','b','b'],'fold':[1,2,1,2],
        'model':['m']*4,'variant':['v']*4,'average_precision':[.9,.1,.6,.6],
        'recall_raw_at_0_5':[.1]*4,'precision_raw_at_0_5':[.1]*4,
        'legitimate_block_rate_raw_at_0_5':[.1]*4,'brier_raw':[.1]*4,'log_loss_raw':[.2]*4,'fit_seconds':[1.]*4,'convergence_warning':[False]*4})
    assert rank_candidates(results,candidates,2).iloc[0].candidate=='b'
    with pytest.raises(ValueError):rank_candidates(results.iloc[:3],candidates,2)
    with pytest.raises(ValueError):rank_candidates(pd.concat([results,results.iloc[:1]]),candidates,2)


def test_threshold_diagnostics_uses_fixed_boundary_and_correct_denominators():
    # Dos TP, un FN, un FP y dos TN. Exactamente 0,5 se marca positivo.
    y=np.array([1,1,1,0,0,0])
    p=np.array([0.5,0.9,0.1,0.7,0.2,0.01])
    result=threshold_diagnostics(y,p)
    assert result['recall_raw_at_0_5']==pytest.approx(2/3)
    assert result['precision_raw_at_0_5']==pytest.approx(2/3)
    assert result['legitimate_block_rate_raw_at_0_5']==pytest.approx(1/3)
    assert [result[k+'_raw_at_0_5'] for k in ('tp','fp','fn','tn')]==[2,1,1,2]
    assert result['block_count_raw_at_0_5']==3
    empty_positive=threshold_diagnostics(y,np.zeros(6))
    assert empty_positive['precision_raw_at_0_5']==0
    assert empty_positive['recall_raw_at_0_5']==0
    assert np.isnan(threshold_diagnostics([1],[0.9])['legitimate_block_rate_raw_at_0_5'])
    with pytest.raises(ValueError): threshold_diagnostics(y,p[:-1])
    with pytest.raises(ValueError): threshold_diagnostics(y,[np.nan]*6)
