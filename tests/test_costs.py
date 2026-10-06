import numpy as np
import pandas as pd
import pytest
from fraud_cost.costs import bmr_actions, realized_cost, evaluate_policies, cost_sensitive_targets_weights
from fraud_cost.scenarios import evaluate_hypothetical_scenarios


def test_four_cells_of_matrix():
    assert realized_cost([1, 0, 1, 0], [200]*4, ["block", "block", "approve", "approve"], 10).tolist() == [10, 10, 200, 0]


def test_bmr_ties_zero_and_small_amounts():
    assert bmr_actions([.08, .05, 1, 1, 1], [200, 200, 5, 10, 0], 10).tolist() == ["block", "approve", "approve", "approve", "approve"]


def test_two_savings_references_and_components():
    y, a = [1, 0, 1, 0], [200, 100, 50, 100]
    p = {"fixed_0_5": ["approve"]*4, "bmr": ["block", "approve", "block", "approve"]}
    result = evaluate_policies(y, a, p, 10).set_index("strategy")
    assert result.loc["bmr", "cost_total_proxy"] == 20
    assert result.loc["bmr", "reference_cost"] == 40
    assert result.loc["bmr", "savings_fraction_vs_reference"] == .5
    assert result.loc["bmr", "savings_vs_fixed_proxy"] == 230
    assert np.allclose(result.cost_total_proxy, result.administrative_cost_total + result.missed_fraud_amount)


def test_weighted_action_errors_equal_cost_minus_constant():
    y, a, ca = np.array([1, 1, 0, 1]), np.array([200, 5, 100, 10]), 10
    target, weight = cost_sensitive_targets_weights(y, a, ca)
    assert target.tolist() == [1, 0, 0, 0]
    assert weight.tolist() == [190, 5, 10, 0]
    constant = np.minimum(y*a, ca).sum()
    for bits in range(16):
        action = np.array([(bits >> i) & 1 for i in range(4)])
        assert realized_cost(y, a, np.where(action, "block", "approve"), ca).sum() == constant + weight[action != target].sum()


def test_zero_reference_is_undefined_not_infinite():
    table = evaluate_policies([1, 0], [0, 0], {"fixed_0_5": ["approve"]*2}, 0)
    assert np.isnan(table.savings_fraction_vs_reference.iloc[0])


@pytest.mark.parametrize("p,a,ca", [([1.1], [10], 1), ([.5], [-1], 1), ([.5], [10], -1), ([np.nan], [10], 1), ([.5], [10], np.inf)])
def test_reject_invalid_inputs(p, a, ca):
    with pytest.raises(ValueError):
        bmr_actions(p, a, ca)


def test_scenarios_use_absolute_ca_and_same_matrix():
    scores = pd.DataFrame({"TransactionID": [1, 2], "TransactionAmt": [200, 100], "isFraud": [1, 0], "probability_calibrated": [.08, .01], "action_fixed_0_5": ["approve", "approve"]})
    table = evaluate_hypothetical_scenarios(scores, ca_grid=(10, 20))
    assert table.query("strategy == 'bmr'").cost_total_proxy.tolist() == [10, 200]
    with pytest.raises(ValueError):
        evaluate_hypothetical_scenarios(scores, ca_grid=(10, 10))


def test_tree_learns_actions_and_handles_zero_regret():
    from fraud_cost.costs import fit_cost_tree
    X = np.repeat([[0], [1]], 300, axis=0)
    y = np.ones(600)
    amounts = np.repeat([5, 200], 300)
    tree = fit_cost_tree(X, y, amounts, 10)
    assert tree.predict([[0], [1]]).tolist() == [0, 1]
    no_preference = fit_cost_tree(X, np.zeros(600), amounts, 0)
    assert no_preference.predict([[0], [1]]).tolist() == [0, 0]


def test_final_configuration_requires_matching_units_and_agreement():
    from fraud_cost.costs import require_approved_costs
    config = {"approved": True, "approval_reference": "acuerdo de prueba", "amount_unit": "dataset", "cost_unit": "dataset", "administrative_cost_grid": [10, 20]}
    assert require_approved_costs(config) == (10, 20)
    for change in ({"approved": False}, {"cost_unit": "otra escala"}, {"approval_reference": None}):
        with pytest.raises(ValueError):
            require_approved_costs(config | change)
