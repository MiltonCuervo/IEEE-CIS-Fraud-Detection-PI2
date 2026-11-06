"""Auditoría de resultados publicados y, opcionalmente, dependencias locales.

No ajusta estimadores, no deserializa modelos y no genera predicciones.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fraud_cost.artifacts import notebook_code_sha256, read_record, sha256_file, verify_hashes


def ratio(a, b, zero=np.nan):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    return np.divide(a,b,out=np.full(np.broadcast_shapes(a.shape,b.shape),zero,dtype=float),where=b!=0)


def validate_costs(table):
    """Reconstruye conteos, componentes y ambos denominadores desde la tabla."""
    check=lambda a,b:np.testing.assert_allclose(a,b,atol=1e-8,rtol=1e-10,equal_nan=True)
    for col in ['TP','FP','FN','TN']:
        if (table[col]<0).any() or not np.equal(table[col],np.floor(table[col])).all():
            raise ValueError('Conteos negativos o fraccionarios')
    check(table.TP+table.FP+table.FN+table.TN,table.rows)
    check(table.TP+table.FN,table.frauds)
    check(table.TP+table.FP,table.policy_block_count)
    check(table.administrative_cost_total,table.policy_block_count*table.administrative_cost_ca)
    check(table.cost_total_proxy,table.administrative_cost_total+table.missed_fraud_amount)
    check(table.cost_per_transaction_proxy,table.cost_total_proxy/table.rows)
    check(table.fraud_recall,ratio(table.TP,table.TP+table.FN))
    check(table.legitimate_block_rate,ratio(table.FP,table.FP+table.TN))
    check(table.precision,ratio(table.TP,table.TP+table.FP,zero=0))
    check(table.amount_weighted_recall,ratio(table.approve_all_cost-table.missed_fraud_amount,table.approve_all_cost))
    check(table.reference_cost,np.minimum(table.approve_all_cost,table.rows*table.administrative_cost_ca))
    check(table.savings_fraction_vs_reference,1-ratio(table.cost_total_proxy,table.reference_cost))
    groups=['period','administrative_cost_ca'] if 'period' in table else ['administrative_cost_ca']
    for _,group in table.groupby(groups):
        fixed=group.loc[group.strategy.eq('fixed_0_5'),'cost_total_proxy']
        if len(fixed)!=1:raise ValueError('Falta una única referencia fija')
        for col in ['rows','frauds','approve_all_cost']:
            if group[col].nunique()!=1:raise ValueError('Políticas sobre poblaciones distintas')
        check(group.fixed_baseline_cost_proxy,fixed.iloc[0])
        check(group.savings_fraction_vs_fixed_proxy,1-ratio(group.cost_total_proxy,fixed.iloc[0]))


def audit(root,local=False):
    root=Path(root).resolve()
    checks=[]
    def passed(name,detail):checks.append({'check':name,'status':'passed','detail':detail})
    notebooks=sorted((root/'notebooks').glob('*.ipynb'))
    if len(notebooks)!=8 or [p.name[:2] for p in notebooks]!=[f'{i:02}' for i in range(1,9)]:
        raise ValueError('La secuencia debe contener únicamente 01–08')
    texts=[]
    import nbformat
    for path in notebooks:
        nb=nbformat.read(path,as_version=4);nbformat.validate(nb)
        for cell in nb.cells:
            if cell.cell_type=='code':
                ast.parse(cell.source)
                if cell.execution_count is None or any(o.output_type=='error' for o in cell.get('outputs',[])):
                    raise ValueError(f'Notebook incompleto o con errores: {path.name}')
            else:texts.append((path,cell.source))
    for p in [*(root/'src/fraud_cost').glob('*.py'),*(root/'scripts').glob('*.py')]:ast.parse(p.read_text(encoding='utf-8'))
    passed('secuencia_y_sintaxis','Ocho notebooks ejecutados sin errores; código parseable')
    records={3:'comparacion_modelos_config.json',4:'seleccion_ordinal_config.json',5:'lightgbm_adopcion_config.json',
             6:'estabilidad_config.json',7:'protocolo_final.json',8:'evaluacion_test_config.json'}
    for number,name in records.items():
        rec=read_record(root,'results/tables/'+name)
        key='notebook_executed_code_sha256' if number==3 else 'notebook_code_sha256'
        if notebook_code_sha256(notebooks[number-1])!=rec[key]:raise ValueError(f'Código distinto en {number}')
    passed('codigo_evaluado','Las celdas ejecutables 03–08 coinciden con sus registros')
    for p in [root/'README.md',*(root/'docs').glob('*.md')]:texts.append((p,p.read_text(encoding='utf-8')))
    links=0
    for path,text in texts:
        for match in re.finditer(r'\[[^\]]*\]\(([^)]+)\)',text):
            target=match.group(1).strip('<>').split('#')[0]
            if not target or re.match(r'^[a-z]+://',target):continue
            if not (path.parent/unquote(target)).exists():raise ValueError(f'Enlace roto en {path.name}: {target}')
            links+=1
    passed('enlaces_documentacion',f'{links} enlaces locales válidos')
    record=read_record(root,'results/tables/evaluacion_test_config.json')
    binding=read_record(root,'results/tables/protocolo_ejecucion_local_v2.json')
    if record['status']!='complete' or not record['test_evaluated'] or record['rows']!=84233:
        raise ValueError('La evaluación final no está completa')
    if any(record[k] for k in ['new_model_fitted','new_calibrator_fitted','cost_trees_refitted']):
        raise ValueError('El cierre declara reajustes no previstos')
    if sha256_file(root/'results/tables/protocolo_ejecucion_local_v2.json')!=record['binding_sha256']:
        raise ValueError('Otra versión de ejecución')
    if sha256_file(root/'results/tables/protocolo_final.json')!=binding['parent_protocol_sha256']:
        raise ValueError('Cambió el protocolo original')
    verify_hashes(root,{p:h for p,h in record['output_sha256'].items() if p.startswith('results/')})
    passed('congelacion_y_salidas','Protocolo original conservado; versión explícita; siete tablas y dos figuras íntegras')
    catalogue=pd.read_csv(root/'results/tables/eda_diccionario_variables.csv')
    inventory=pd.read_csv(root/'results/tables/modelo_variables_ordinal.csv')
    if len(catalogue)!=434 or not catalogue.variable.is_unique or len(inventory)!=432 or not inventory.variable.is_unique:
        raise ValueError('Inventario de atributos incompleto')
    if inventory.rama.value_counts().to_dict()!={'numeric':395,'categorical':37}:
        raise ValueError('Ramas distintas de las evaluadas')
    expected=set(catalogue.variable)-{'TransactionID','TransactionDT','isFraud'}|{'has_identity'}
    if set(inventory.variable)!=expected or inventory.discrepancia_tipo.sum()!=12:
        raise ValueError('Linaje/discrepancias distintos del predictor documentado')
    passed('linaje_variables','434 originales → 431 predictores + has_identity → 432 entradas; 12 discrepancias declaradas')
    costs=pd.read_csv(root/'results/tables/test_costos.csv')
    temporal=pd.read_csv(root/'results/tables/test_costos_periodo.csv')
    if len(costs)!=18 or len(temporal)!=54:raise ValueError('Comparación económica incompleta')
    for frame in (costs,temporal):validate_costs(frame)
    keys=['administrative_cost_ca','strategy'];cols=['cost_total_proxy','administrative_cost_total','missed_fraud_amount']
    np.testing.assert_allclose(temporal.groupby(keys)[cols].sum().sort_index(),costs.set_index(keys)[cols].sort_index(),atol=1e-8,rtol=1e-10)
    passed('matriz_y_metricas','18 comparaciones y 54 costos temporales: conteos, componentes, tasas y referencias reconciliados')
    metrics=pd.read_csv(root/'results/tables/test_metricas_probabilidades.csv').set_index('probability')
    if set(metrics.index)!={'raw','platt'}:raise ValueError('Falta comparación probabilística')
    platt_worse=bool(metrics.loc['platt','brier_score']>metrics.loc['raw','brier_score'] and metrics.loc['platt','log_loss']>metrics.loc['raw','log_loss'])
    passed('interpretacion_probabilidades',f'Raw y Platt conservados; deterioro global de Platt en test documentado: {platt_worse}')
    if local:
        from fraud_cost.execution_workspace import verify_workspace
        stage,protocol,_=verify_workspace(root)
        verify_hashes(root,record['output_sha256'])
        for comparison in binding['comparisons']:
            p=comparison['table']
            pd.testing.assert_frame_equal(pd.read_csv(root/p),pd.read_csv(stage/p),check_exact=False,atol=1e-8,rtol=1e-8)
        raw=pd.read_csv(root/'data/raw/train_transaction.csv',usecols=['TransactionID','TransactionDT','TransactionAmt','isFraud'])
        identity=pd.read_csv(root/'data/raw/train_identity.csv',usecols=['TransactionID'])
        manifest=pd.read_parquet(root/'data/interim/split_manifest.parquet')
        if len(raw)!=590540 or raw.TransactionID.duplicated().any() or identity.TransactionID.duplicated().any() or not identity.TransactionID.isin(raw.TransactionID).all():
            raise ValueError('Claves fuente inválidas')
        from fraud_cost.split import create_temporal_split,validate_temporal_split
        expected=raw.sort_values(['TransactionDT','TransactionID']).reset_index(drop=True)
        expected['split']=create_temporal_split(expected)
        pd.testing.assert_frame_equal(expected[['TransactionID','TransactionDT','split']],manifest.sort_values(['TransactionDT','TransactionID']).reset_index(drop=True),check_dtype=False)
        validate_temporal_split(expected,expected.split)
        for label,path in [('test','lightgbm_test_scores.parquet'),('validation','lightgbm_validation_scores.parquet')]:
            # Los scores de validación corresponden únicamente a su mitad posterior.
            scores=pd.read_parquet(root/'data/interim'/path)
            actual=scores[['TransactionID','TransactionDT','TransactionAmt','isFraud']]
            source=actual[['TransactionID']].merge(raw,on='TransactionID',validate='one_to_one')
            pd.testing.assert_frame_equal(actual,source[actual.columns],check_dtype=False,check_exact=True)
            ids=set(expected.loc[expected.split.eq(label),'TransactionID'])
            if not scores.TransactionID.isin(ids).all():raise ValueError('Scores de otra población')
        scores=pd.read_parquet(root/'data/interim/lightgbm_test_scores.parquet')
        actions=pd.read_parquet(root/'data/interim/lightgbm_test_actions.parquet')
        from fraud_cost.stability import validate_evidence
        aligned=validate_evidence(scores,actions,protocol['costs']['ca_grid'])
        for ca,block in aligned.items():
            for policy in ('fixed_0_5','bmr','cost_sensitive_tree'):
                flagged=block[policy].eq('block').to_numpy()
                total=flagged.sum()*ca+scores.loc[scores.isFraud.eq(1)&~flagged,'TransactionAmt'].sum()
                saved=costs.loc[costs.administrative_cost_ca.eq(ca)&costs.strategy.eq(policy)].iloc[0]
                np.testing.assert_allclose(saved.cost_total_proxy,total,atol=1e-8,rtol=1e-10)
        passed('datos_y_protocolo_local','CSV originales, unión, manifiesto, scores y acciones consistentes; versión activa verificada sin inferencia')
    return {'status':'passed','scope':'local_full' if local else 'published_results',
        'created_utc':datetime.now(timezone.utc).isoformat(),'test_evaluated_previously':True,
        'new_predictions_generated':False,'new_models_fitted':False,'checks':checks,
        'local_dependencies_checked':local,'final_record_sha256':sha256_file(root/'results/tables/evaluacion_test_config.json')}


def main():
    if hasattr(sys.stdout, 'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local',action='store_true',help='Incluye fuentes y artefactos locales; no genera predicciones')
    parser.add_argument('--output',type=Path,help='Guardar únicamente en results/tables/auditoria_proyecto.json')
    args=parser.parse_args()
    report=audit(ROOT,local=args.local)
    if args.output:
        target=args.output.resolve()
        if target!=(ROOT/'results/tables/auditoria_proyecto.json').resolve():
            raise ValueError('El informe de auditoría no puede sobrescribir evidencia experimental')
        target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    for check in report['checks']:print(f"OK {check['check']}: {check['detail']}")
    print('Auditoría completa; no se reentrenó ni se repitieron predicciones de test.')


if __name__=='__main__':main()
