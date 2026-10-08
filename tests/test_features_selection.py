import warnings

import numpy as np
import pandas as pd
import pytest
from fraud_cost.features import PastOnlyFeatures, transaction_features, make_preprocessor
from fraud_cost.selection import temporal_folds, run_temporal_search


def example():
    return pd.DataFrame({"TransactionID": [1,2,3,4], "TransactionDT": [10,20,20,30],
                         "TransactionAmt": [10.,20.,30.,40.], "entity": ["a"]*4,
                         "isFraud": [0,1,0,1], "split": ["train"]*4})


def test_simultaneous_operations_share_strict_past():
    frame = example()
    X = PastOnlyFeatures({"entity": ("entity",)}).fit_transform(frame)
    assert X.entity_past_count.tolist() == [0,1,1,3]
    assert X.entity_past_mean_amount.iloc[1:3].tolist() == [10,10]
    assert X.entity_past_mean_amount.iloc[3] == 20
    assert X.entity_seconds_since_last.iloc[1:].tolist() == [10,10,10]


def test_future_and_labels_do_not_change_past_features():
    before = example()
    after = before.copy()
    after.loc[3,"TransactionAmt"] = 99999
    after["isFraud"] = 1 - after.isFraud
    a = PastOnlyFeatures({"entity": ("entity",)}).fit_transform(before)
    b = PastOnlyFeatures({"entity": ("entity",)}).fit_transform(after)
    columns = [c for c in a if c.startswith("entity_")]
    pd.testing.assert_frame_equal(a[columns], b[columns])
    assert not {"TransactionID", "TransactionDT", "isFraud", "split"}.intersection(a.columns)


def test_order_and_index_are_preserved():
    frame = example().iloc[[3,1,0,2]]
    X = PastOnlyFeatures({"entity": ("entity",)}).fit_transform(frame)
    assert X.index.equals(frame.index)
    assert X.entity_past_count.tolist() == [3,1,0,1]


def test_wide_frames_do_not_fragment_when_adding_history():
    frame = pd.concat([example(), pd.DataFrame({f"v_{i}": [0.,1.,2.,3.] for i in range(120)})], axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.PerformanceWarning)
        X = PastOnlyFeatures({"entity": ("entity",)}).fit_transform(frame)
    assert X.entity_past_count.tolist() == [0,1,1,3]
    assert X.v_0.dtype == np.float32


def test_future_history_is_frozen_unknown_and_incomplete_keys():
    builder = PastOnlyFeatures({"entity": ("entity",)})
    builder.fit_transform(example())
    future = pd.DataFrame({"TransactionID": [5,6,7,8], "TransactionDT": [40,50,60,70],
                           "TransactionAmt": [200.,300.,400.,500.], "entity": ["a","a","b",None]})
    X = builder.transform(future)
    assert X.entity_past_count.tolist() == [4,4,0,0]
    assert X.entity_past_mean_amount.iloc[:2].tolist() == [25,25]
    assert X.entity_past_mean_amount.iloc[2:].isna().all()
    assert X.entity_seconds_since_last.iloc[:2].tolist() == [10,20]
    with pytest.raises(ValueError):
        builder.transform(example())


def test_encoder_does_not_learn_future_categories():
    past = transaction_features(example())
    preprocessor = make_preprocessor(past, min_frequency=1)
    Xt = preprocessor.fit_transform(past)
    future = past.iloc[:1].copy()
    future["entity"] = "never_seen"
    Xv = preprocessor.transform(future)
    assert Xt.dtype == np.float32 and Xv.shape[1] == Xt.shape[1]
    assert "never_seen" not in preprocessor.named_transformers_["categorical"].named_steps["encode"].categories_[0]


def long_example():
    n = 401
    return pd.DataFrame({"TransactionID": np.arange(n), "TransactionDT": np.arange(n)*86400,
                         "TransactionAmt": np.ones(n)*10, "isFraud": np.arange(n)%2,
                         "split": ["train"]*n})


def test_folds_are_disjoint_in_time_and_respect_gap():
    frame = long_example()
    folds = temporal_folds(frame)
    assert len(folds) == 2
    for fold in folds:
        past, future = frame.iloc[fold["train_idx"]], frame.iloc[fold["eval_idx"]]
        assert past.TransactionDT.max()+7*86400 < future.TransactionDT.min()
        assert not set(past.TransactionID).intersection(future.TransactionID)
    frame.loc[0,"split"] = "test"
    with pytest.raises(ValueError):
        temporal_folds(frame)


def test_gap_cannot_consume_entire_window():
    with pytest.raises(ValueError):
        temporal_folds(long_example(), gap_days=1000)


def test_small_end_to_end_search_uses_all_predeclared_candidates():
    frame = long_example()
    candidates = ({"candidate":"base", "features":"base", "params":{"n_estimators":2,"max_depth":2,"n_jobs":1}},
                  {"candidate":"history", "features":"history", "params":{"n_estimators":2,"max_depth":2,"n_jobs":1}})
    metrics, summary, diagnostics = run_temporal_search(frame, temporal_folds(frame), candidates, progress=lambda *a, **k: None)
    assert len(metrics) == 4 and len(diagnostics) == 4
    assert summary.folds.tolist() == [2,2]
    assert metrics.average_precision.between(0,1).all()
