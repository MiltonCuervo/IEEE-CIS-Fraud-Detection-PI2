import numpy as np
import pandas as pd
import pytest
from fraud_cost.adoption import apply_platt,validation_policy,economic_validation,amount_diagnostics


def test_validation_boundary_keeps_simultaneous_rows_and_excludes_other_blocks():
    frame=pd.DataFrame({'TransactionID':range(10),'TransactionDT':[0,1,2,3,3,3,4,5,6,99],
                        'isFraud':[0,0,1,0,1,0,1,0,1,1],
                        'split':['train']+['validation']*8+['test']})
    late,meta=validation_policy(frame)
    assert meta['calibration_rows']==5 and meta['policy_rows']==3
    assert late.TransactionID.tolist()==[6,7,8]
    assert meta['calibration_time_end']<meta['policy_time_start']


def test_platt_clips_endpoints_without_modifying_input():
    class Calibrator:
        def predict_proba(self,X):
            self.seen=X.copy()
            p=1/(1+np.exp(-X[:,0]))
            return np.column_stack([1-p,p])
    calibrator=Calibrator();raw=np.array([0.,.5,1.])
    result=apply_platt(raw,calibrator)
    np.testing.assert_allclose(result,[1e-6,.5,1-1e-6])
    np.testing.assert_array_equal(raw,[0.,.5,1.])
    assert calibrator.seen.shape==(3,1)
    with pytest.raises(ValueError):apply_platt([np.nan],calibrator)


def test_policy_costs_use_original_amount_and_distinct_actions():
    class Tree:
        def predict(self,X):return np.array([0,1,0])
    scores=pd.DataFrame({'TransactionID':[1,2,3],'TransactionAmt':[100.,100.,0.],
        'isFraud':[1,0,1],'probability_raw':[.3,.1,.9],
        'probability_calibrated':[.2,.05,.8],'action_fixed_0_5':['approve','approve','block']})
    costs,actions,diagnostic=economic_validation(scores,np.zeros((3,2)),{10:Tree()},[10])
    assert costs.set_index('strategy').cost_total_proxy.to_dict()=={'fixed_0_5':110.,'bmr':10.,'cost_sensitive_tree':110.}
    assert actions.bmr.tolist()==['block','approve','approve']
    assert len(diagnostic)==2
    invalid=scores.copy();invalid.loc[1,'TransactionID']=1
    with pytest.raises(ValueError):economic_validation(invalid,np.zeros((3,2)),{10:Tree()},[10])


def test_amount_bands_preserve_total_observed_and_expected_exposure():
    scores=pd.DataFrame({'TransactionAmt':[0.,50.,100.,1000.], 'isFraud':[0,1,0,1],
                        'probability_raw':[.1,.2,.3,.4],'probability_calibrated':[.2,.3,.4,.5]})
    table=amount_diagnostics(scores)
    assert table.rows.sum()==4 and table.observed_fraud_amount.sum()==1050
    assert table.expected_fraud_amount_platt.sum()==555
