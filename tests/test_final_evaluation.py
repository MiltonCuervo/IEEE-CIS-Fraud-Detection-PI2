from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fraud_cost.execution_workspace import semantic_equivalence
from fraud_cost.final_evaluation import (claim_run, update_run, verify_test_identity,
                                        actions_and_costs, diagnostic_tables)
from fraud_cost.final_protocol import build_protocol, TEST_RECORD


def protocol_fixture():
    return build_protocol({'validation_windows':{'calibration_rows':42046}},
        {'candidate':'lightgbm_referencia','params':{'num_leaves':63}},
        SimpleNamespace(coef_=np.array([[1.1]]),intercept_=np.array([.1])),
        {'rows':12,'labels_read':False},{},'code')


def test_local_version_allows_only_provenance_and_numerical_roundoff():
    parent=protocol_fixture();local=deepcopy(parent)
    local['created_utc']='new';local['dependency_sha256']={'local':'bytes'}
    local['calibration']['coefficient_a']+=1e-13
    assert semantic_equivalence(parent,local)['all_decisions_equal']
    local['costs']['ca_grid'].append(2.5)
    with pytest.raises(ValueError):semantic_equivalence(parent,local)


def test_exclusive_claim_precedes_test_and_preserves_failure(tmp_path):
    (tmp_path/'results/tables').mkdir(parents=True)
    record=claim_run(tmp_path,'protocol','binding')
    assert not record['test_labels_read'] and not record['predictions_started']
    with pytest.raises(FileExistsError):claim_run(tmp_path,'protocol','binding')
    record.update(status='failed',error_message='interrupted before scoring')
    update_run(tmp_path,record)
    assert json.loads((tmp_path/TEST_RECORD).read_text())['status']=='failed'
    with pytest.raises(FileExistsError):claim_run(tmp_path,'protocol','binding')
    wrong=deepcopy(record);wrong['run_id']='other'
    with pytest.raises(ValueError):update_run(tmp_path,wrong)


def test_frozen_test_identity_uses_ids_times_and_complete_membership():
    frame=pd.DataFrame({'TransactionID':[3,1,2],'TransactionDT':[20,10,10],'split':'test'})
    ordered=frame.sort_values(['TransactionDT','TransactionID'])
    expected={'rows':3,'id_time_sha256':hashlib.sha256(ordered[['TransactionID','TransactionDT']].to_numpy(dtype='<i8').tobytes()).hexdigest()}
    assert verify_test_identity(frame,expected).TransactionID.tolist()==[1,2,3]
    with pytest.raises(ValueError):verify_test_identity(frame.assign(split='train'),expected)
    bad=frame.copy();bad.loc[0,'TransactionDT']=21
    with pytest.raises(ValueError):verify_test_identity(bad,expected)


def test_final_tables_use_saved_policies_and_reconcile_time_components():
    class Tree:
        def predict(self,X):return (X[:,0]%3==0).astype(int)
    scores=pd.DataFrame({'TransactionID':np.arange(12),'TransactionDT':np.repeat([0,1,2,3,4,6],2),
        'TransactionAmt':[100.,20.,200.,30.,10.,400.,150.,5.,0.,200.,1000.,50.],
        'isFraud':np.tile([1,0],6),'probability_raw':[.1,.6]*6,
        'probability_calibrated':[.1,.6]*6,'action_fixed_0_5':['approve','block']*6})
    matrix=np.arange(12).reshape(-1,1)
    protocol=protocol_fixture();protocol['costs']['ca_grid']=[10,20]
    tables,actions=diagnostic_tables(scores,matrix,{10:Tree(),20:Tree()},protocol)
    assert len(tables['metricas_probabilidades'])==2 and len(tables['costos'])==6
    assert len(actions)==24 and len(tables['costos_periodo'])==18
    for r in tables['costos'].itertuples():
        assert r.TP+r.FP+r.FN+r.TN==12
        assert r.TP+r.FP==r.policy_block_count
        assert r.precision==pytest.approx(r.TP/(r.TP+r.FP)) if r.TP+r.FP else r.precision==0
        assert r.cost_total_proxy==pytest.approx(r.administrative_cost_total+r.missed_fraud_amount)
    # Fraude pA=Ca se aprueba, aunque el árbol pueda elegir otra acción.
    assert actions.loc[actions.TransactionID.eq(0)&actions.administrative_cost_ca.eq(10),'bmr'].iloc[0]=='approve'


def test_completed_run_loads_tables_without_reading_or_predicting(tmp_path,monkeypatch):
    from fraud_cost import final_evaluation as final
    from fraud_cost.artifacts import notebook_code_sha256,sha256_file
    (tmp_path/'results/tables').mkdir(parents=True)
    (tmp_path/'notebooks').mkdir()
    (tmp_path/final.NOTEBOOK).write_text(json.dumps({'cells':[{'cell_type':'code','source':['pass']}]}))
    (tmp_path/final.BINDING).write_text('{}')
    outputs={}
    for name in final.TABLE_NAMES:
        relative=f'results/tables/test_{name}.csv'
        (tmp_path/relative).write_text('value\n1\n')
        outputs[relative]=sha256_file(tmp_path/relative)
    record={'status':'complete','binding_sha256':sha256_file(tmp_path/final.BINDING),
        'notebook_code_sha256':notebook_code_sha256(tmp_path/final.NOTEBOOK),'output_sha256':outputs}
    (tmp_path/final.TEST_RECORD).write_text(json.dumps(record))
    monkeypatch.setattr(final,'verify_workspace',lambda root:(None,{},{}))
    def forbidden(*args,**kwargs):raise AssertionError('No repetir lectura o inferencia de test')
    monkeypatch.setattr(final,'read_model_frame',forbidden)
    monkeypatch.setattr(final.joblib,'load',forbidden)
    tables,returned=final.evaluate_or_load(tmp_path)
    assert len(tables)==7 and returned==record


def test_incomplete_run_stops_without_reading_test(tmp_path,monkeypatch):
    from fraud_cost import final_evaluation as final
    (tmp_path/'results/tables').mkdir(parents=True)
    (tmp_path/final.BINDING).write_text('{}')
    (tmp_path/final.TEST_RECORD).write_text(json.dumps({'status':'failed'}))
    monkeypatch.setattr(final,'verify_workspace',lambda root:(None,{},{}))
    def forbidden(*args,**kwargs):raise AssertionError('No repetir lectura de test')
    monkeypatch.setattr(final,'read_model_frame',forbidden)
    with pytest.raises(RuntimeError,match='incompleta'):final.evaluate_or_load(tmp_path)
