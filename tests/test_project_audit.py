import importlib.util
from pathlib import Path
import pandas as pd
import pytest

path=Path(__file__).resolve().parents[1]/'scripts/audit_project.py'
spec=importlib.util.spec_from_file_location('audit_project',path)
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def fixture():
    return pd.DataFrame([
        {'strategy':'fixed_0_5','TP':1,'FP':1,'FN':1,'TN':1,'rows':4,'frauds':2,
         'policy_block_count':2,'administrative_cost_ca':10,'administrative_cost_total':20,
         'missed_fraud_amount':100,'cost_total_proxy':120,'cost_per_transaction_proxy':30.0,
         'approve_all_cost':150,'reference_cost':40,'fixed_baseline_cost_proxy':120,
         'fraud_recall':.5,'legitimate_block_rate':.5,'precision':.5,'amount_weighted_recall':1/3,
         'savings_fraction_vs_reference':-2,'savings_fraction_vs_fixed_proxy':0},
        {'strategy':'bmr','TP':1,'FP':0,'FN':1,'TN':2,'rows':4,'frauds':2,
         'policy_block_count':1,'administrative_cost_ca':10,'administrative_cost_total':10,
         'missed_fraud_amount':50,'cost_total_proxy':60,'cost_per_transaction_proxy':15.0,
         'approve_all_cost':150,'reference_cost':40,'fixed_baseline_cost_proxy':120,
         'fraud_recall':.5,'legitimate_block_rate':0,'precision':1,'amount_weighted_recall':2/3,
         'savings_fraction_vs_reference':-.5,'savings_fraction_vs_fixed_proxy':.5}])


def test_audit_reconstructs_both_savings_references_and_zero_alerts():
    table=fixture();audit.validate_costs(table)
    changes={'TP':0,'FP':0,'FN':2,'TN':2,'policy_block_count':0,'administrative_cost_total':0,
             'missed_fraud_amount':150,'cost_total_proxy':150,'cost_per_transaction_proxy':37.5,
             'fraud_recall':0,'legitimate_block_rate':0,'precision':0,'amount_weighted_recall':0,
             'savings_fraction_vs_reference':-2.75,'savings_fraction_vs_fixed_proxy':-.25}
    for key,value in changes.items():table.loc[1,key]=value
    audit.validate_costs(table)


def test_audit_rejects_wrong_cost_denominator_confusion_or_rate():
    for column in ['cost_total_proxy','reference_cost','FP','precision','savings_fraction_vs_fixed_proxy']:
        table=fixture();table.loc[1,column]+=1
        with pytest.raises((AssertionError,ValueError)):audit.validate_costs(table)
