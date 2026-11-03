"""Contratos del nuevo procesamiento y de la matriz económica."""
import joblib
import numpy as np
import pandas as pd
import pytest
from fraud_cost.reference import make_features, build_preprocessor
from fraud_cost.costs import bmr_actions, realized_cost, cost_sensitive_targets_weights


def test_preparation_reuses_train_statistics_and_schema(tmp_path):
    train = pd.DataFrame({'card1': [10, 20, None], 'id_30': ['a', None, 'b'],
                          'TransactionAmt': [1., 3., 5.], 'V1': [1., np.nan, 9.],
                          'empty': [np.nan]*3, 'TransactionID': [1,2,3], 'isFraud': [0,1,0]})
    X = make_features(train)
    assert 'isFraud' not in X and 'TransactionID' not in X
    pre, cats, nums = build_preprocessor(X)
    with pytest.warns(UserWarning, match='vacías'):
        pre.fit(X)
    future = pd.DataFrame({'card1': [999, None], 'id_30': ['new', None],
                           'TransactionAmt': [100., 200.], 'V1': [np.nan, np.inf],
                           'empty': [np.nan, 2.]})
    result = pre.transform(make_features(future))
    assert result.dtype == np.float32 and result.shape == (2, 5)
    assert result[0, cats.index('card1')] == -1
    assert result[1, cats.index('card1')] != -1  # Ausencia vista en train.
    assert result[0, cats.index('id_30')] == -1
    assert np.all(result[:, len(cats)+nums.index('V1')] == 5)
    assert result[0, len(cats)+nums.index('empty')] == 0
    path = tmp_path/'pre.joblib'
    joblib.dump(pre,path)
    np.testing.assert_array_equal(joblib.load(path).transform(make_features(future)), result)
    assert future.TransactionAmt.tolist() == [100., 200.]


def test_expected_risk_ties_and_realized_matrix():
    assert bmr_actions([1., .5, .9, 1.], [0., 20., 100., 10.], 10).tolist() == ['approve','approve','block','approve']
    np.testing.assert_array_equal(realized_cost([1,0,1,0],[50,50,50,50],['block','block','approve','approve'],10),[10,10,50,0])


def test_cost_tree_target_weights_reproduce_cost_for_each_action():
    y, amount, ca = np.array([0,1,1,1]), np.array([50.,5.,10.,100.]), 10
    target, weight = cost_sensitive_targets_weights(y,amount,ca)
    for action in (0,1):
        exact = realized_cost(y,amount,np.where(np.full(4,action),'block','approve'),ca)
        np.testing.assert_array_equal(exact, np.minimum(y*amount,ca)+weight*(target!=action))


def test_reader_preserves_amount_and_uses_join_for_identity(tmp_path):
    from fraud_cost.reference import read_model_frame
    pd.DataFrame({'TransactionID': [1, 2], 'TransactionDT': [10, 20],
                  'TransactionAmt': [123.123456789, 987.987654321],
                  'isFraud': [0, 1], 'V1': [1., 2.]}).to_csv(tmp_path/'train_transaction.csv', index=False)
    # Incluso si el atributo id_01 es ausente, sí existe un registro de identidad.
    pd.DataFrame({'TransactionID': [1], 'id_01': [np.nan]}).to_csv(tmp_path/'train_identity.csv', index=False)
    manifest = pd.DataFrame({'TransactionID': [1, 2], 'split': ['train', 'test']})
    frame = read_model_frame(tmp_path, manifest)
    assert frame.has_identity.tolist() == [1, 0]
    assert frame.TransactionAmt.dtype == np.float64
    np.testing.assert_array_equal(frame.TransactionAmt, [123.123456789, 987.987654321])
    with pytest.raises(ValueError, match='cubre exactamente'):
        read_model_frame(tmp_path, manifest.iloc[:1])

