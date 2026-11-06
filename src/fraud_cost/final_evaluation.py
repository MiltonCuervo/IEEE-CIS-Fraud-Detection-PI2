"""Evaluación final única, sin ajustes y con registro exclusivo antes de puntuar."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

import joblib
import numpy as np
import pandas as pd

from fraud_cost.artifacts import notebook_code_sha256, sha256_file, verify_hashes
from fraud_cost.adoption import selected_scores
from fraud_cost.costs import bmr_actions, evaluate_policies
from fraud_cost.execution_workspace import BINDING, NOTEBOOK, TEST_RECORD, verify_workspace
from fraud_cost.final_protocol import verify_before_test
from fraud_cost.reference import read_model_frame
from fraud_cost.stability import (temporal_periods, period_probabilities, period_costs,
                                  period_amount_calibration, validate_evidence)

TABLE_NAMES = ('metricas_probabilidades', 'costos', 'periodos', 'metricas_periodo',
               'costos_periodo', 'calibracion_montos', 'calibracion_periodo_monto')


def claim_run(root, protocol_hash, binding_hash):
    """open(x) es el bloqueo exclusivo; no basta comprobar exists antes."""
    root = Path(root)
    record = {'status': 'started', 'run_id': uuid.uuid4().hex, 'started_utc': datetime.now(timezone.utc).isoformat(),
              'protocol_sha256': protocol_hash, 'binding_sha256': binding_hash,
              'test_labels_read': False, 'predictions_started': False, 'test_evaluated': False,
              'new_model_fitted': False, 'new_calibrator_fitted': False, 'cost_trees_refitted': False}
    with (root/TEST_RECORD).open('x', encoding='utf-8', newline='\n') as output:
        json.dump(record, output, ensure_ascii=False, indent=2); output.write('\n')
    return record


def update_run(root, record):
    path = Path(root)/TEST_RECORD
    current = json.loads(path.read_text(encoding='utf-8'))
    if current['run_id'] != record['run_id']:
        raise ValueError('Otro registro ocupa el destino de evaluación')
    pending = path.with_suffix('.json.pending')
    pending.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    pending.replace(path)


def update_both(root, stage, record):
    update_run(root, record)
    update_run(stage, record)


def verify_test_identity(frame, expected):
    ordered = frame.sort_values(['TransactionDT','TransactionID']).reset_index(drop=True)
    if ordered.TransactionID.isna().any() or ordered.TransactionID.duplicated().any() or len(ordered)!=expected['rows']:
        raise ValueError('Población final distinta del protocolo')
    identity = ordered[['TransactionID','TransactionDT']].to_numpy(dtype='<i8')
    if hashlib.sha256(identity.tobytes()).hexdigest()!=expected['id_time_sha256']:
        raise ValueError('IDs/tiempos distintos del test congelado')
    if not ordered['split'].eq('test').all():
        raise ValueError('La evaluación contiene filas externas')
    return ordered


def actions_and_costs(scores, matrix, trees, grid):
    if matrix.shape[0]!=len(scores) or set(trees)!=set(grid):
        raise ValueError('Matriz o árboles incompatibles')
    tables, actions = [], []
    for ca in grid:
        predicted = np.asarray(trees[ca].predict(matrix))
        if not np.isin(predicted,[0,1]).all():
            raise ValueError('El árbol no devuelve acciones binarias')
        policies={'fixed_0_5':scores.action_fixed_0_5.to_numpy(),
                  'bmr':bmr_actions(scores.probability_calibrated,scores.TransactionAmt,ca),
                  'cost_sensitive_tree':np.where(predicted==1,'block','approve')}
        tables.append(evaluate_policies(scores.isFraud,scores.TransactionAmt,policies,ca))
        actions.append(pd.DataFrame({'TransactionID':scores.TransactionID,'administrative_cost_ca':ca,**policies}))
    return policy_counts(pd.concat(tables,ignore_index=True)), pd.concat(actions,ignore_index=True)


def policy_counts(table):
    table=table.copy()
    table['TP']=table.frauds_blocked
    table['FP']=table.legitimate_block_count
    table['FN']=table.frauds_missed
    table['TN']=table.rows-table.frauds-table.FP
    table['precision']=np.divide(table.TP,table.TP+table.FP,out=np.zeros(len(table)),where=(table.TP+table.FP).to_numpy()>0)
    return table


def diagnostic_tables(scores, matrix, trees, protocol):
    costs, actions=actions_and_costs(scores,matrix,trees,protocol['costs']['ca_grid'])
    aligned=validate_evidence(scores,actions,protocol['costs']['ca_grid'])
    data, periods=temporal_periods(scores,protocol['reporting']['time_subperiods'])
    metrics=period_probabilities(scores.assign(period=0)).drop(columns='period')
    metrics['model']='lightgbm'
    temporal=period_probabilities(data)
    time_costs=policy_counts(period_costs(data,aligned))
    edges=[float(x) for x in protocol['reporting']['amount_edges']]
    amounts=period_amount_calibration(scores.assign(period=0),edges).drop(columns='period')
    time_amounts=period_amount_calibration(data,edges)
    keys=['administrative_cost_ca','strategy']
    columns=['cost_total_proxy','administrative_cost_total','missed_fraud_amount']
    np.testing.assert_allclose(time_costs.groupby(keys)[columns].sum().sort_index(),
        costs.set_index(keys)[columns].sort_index(),atol=1e-8,rtol=1e-8)
    return dict(zip(TABLE_NAMES,[metrics,costs,periods,temporal,time_costs,amounts,time_amounts])),actions


def create_figures(tables, root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels={'fixed_0_5':'Regla fija 0,5','bmr':'BMR','cost_sensitive_tree':'Árbol ponderado'}
    costs=tables['costos']
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for policy,group in costs.groupby('strategy'):
        axes[0].plot(group.administrative_cost_ca,group.cost_total_proxy,'o-',label=labels[policy])
        axes[1].plot(group.legitimate_block_rate,group.fraud_recall,'o-',label=labels[policy])
    axes[0].set(title='Costo observado en test',xlabel='Ca hipotético',ylabel='Unidades del dataset')
    axes[1].set(title='Compromiso de intervención',xlabel='Fracción de legítimas intervenidas',ylabel='Recall de fraude')
    for ax in axes:ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(root/'results/figures/test_costos_y_operacion.png',dpi=160,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    periods=tables['periodos'];metrics=tables['metricas_periodo']
    axes[0,0].plot(periods.period,periods.fraud_rate,'o--',label='Fraude observado')
    for stage,group in metrics.groupby('probability'):
        axes[0,0].plot(group.period,group.mean_probability,'o-',label=stage)
        axes[0,1].plot(group.period,group.average_precision,'o-',label=stage)
    for policy,group in tables['costos_periodo'].loc[tables['costos_periodo'].administrative_cost_ca.eq(10)].groupby('strategy'):
        axes[1,0].plot(group.period,group.cost_per_transaction_proxy,'o-',label=labels[policy])
    for ax in (axes[0,0],axes[0,1],axes[1,0]):ax.set(xlabel='Subperiodo test',xticks=periods.period)
    axes[0,0].set(title='Probabilidad y frecuencia',ylabel='Proporción')
    axes[0,1].set(title='AP temporal',ylabel='Average precision')
    axes[1,0].set(title='Costo por operación con Ca=10',ylabel='Unidades del dataset')
    amounts=tables['calibracion_montos'];x=np.arange(len(amounts))
    axes[1,1].bar(x-.18,amounts.observed_fraud_amount,width=.36,label='Fraude observado')
    axes[1,1].bar(x+.18,amounts.expected_fraud_amount_platt,width=.36,label='Exposición esperada Platt')
    axes[1,1].set(xticks=x,xticklabels=amounts.amount_band,title='Exposición por monto',ylabel='Unidades del dataset')
    axes[1,1].tick_params(axis='x',rotation=30)
    for ax in axes.flat:ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(root/'results/figures/test_estabilidad_y_calibracion.png',dpi=160,bbox_inches='tight');plt.close(fig)
    return ['results/figures/test_costos_y_operacion.png','results/figures/test_estabilidad_y_calibracion.png']


def evaluate_or_load(root):
    root=Path(root).resolve()
    stage,protocol,binding=verify_workspace(root)
    binding_hash=sha256_file(root/BINDING)
    if (root/TEST_RECORD).exists():
        record=json.loads((root/TEST_RECORD).read_text(encoding='utf-8'))
        if record.get('status')!='complete':
            raise RuntimeError('La corrida final existe y está incompleta: auditar esa corrida, sin volver a puntuar')
        if record['binding_sha256']!=binding_hash or record['notebook_code_sha256']!=notebook_code_sha256(root/NOTEBOOK):
            raise ValueError('Resultados de otro protocolo/evaluador')
        verify_hashes(root,record['output_sha256'])
        print('Resultados finales verificados y recuperados; no se repite inferencia.',flush=True)
        return {name:pd.read_csv(root/f'results/tables/test_{name}.csv') for name in TABLE_NAMES},record
    verify_before_test(stage)
    output_paths=[f'results/tables/test_{name}.csv' for name in TABLE_NAMES]
    output_paths+=['data/interim/lightgbm_test_scores.parquet','data/interim/lightgbm_test_actions.parquet',
                   'results/figures/test_costos_y_operacion.png','results/figures/test_estabilidad_y_calibracion.png']
    if any((root/p).exists() for p in output_paths):
        raise FileExistsError('Existen salidas test sin registro; revisarlas antes de continuar')
    record=claim_run(root,binding['local_protocol_sha256'],binding_hash)
    record['notebook_code_sha256']=notebook_code_sha256(root/NOTEBOOK)
    try:
        with (stage/TEST_RECORD).open('x',encoding='utf-8') as output:
            json.dump(record,output,ensure_ascii=False,indent=2)
        record.update(test_labels_read=True,last_step='reading_frozen_test');update_both(root,stage,record)
        manifest=pd.read_parquet(stage/'data/interim/split_manifest.parquet',columns=['TransactionID','split'])
        frame=read_model_frame(stage/'data/raw',manifest)
        test=verify_test_identity(frame.loc[frame.split.eq('test')].copy(),protocol['test'])
        del frame,manifest
        predictor=joblib.load(stage/protocol['model']['artifact'])
        trees=joblib.load(stage/protocol['trees']['artifact'])
        record.update(predictions_started=True,last_step='predicting_once');update_both(root,stage,record)
        scores,matrix=selected_scores(predictor,test)
        tables,actions=diagnostic_tables(scores,matrix,trees,protocol)
        print(f'Test puntuado: {len(scores)} operaciones; {int(scores.isFraud.sum())} fraudes.',flush=True)
        scores.to_parquet(root/'data/interim/lightgbm_test_scores.parquet',index=False)
        actions.to_parquet(root/'data/interim/lightgbm_test_actions.parquet',index=False)
        for name,table in tables.items():table.to_csv(root/f'results/tables/test_{name}.csv',index=False)
        create_figures(tables,root)
        verify_frozen=__import__('fraud_cost.final_protocol',fromlist=['verify_frozen_protocol']).verify_frozen_protocol
        verify_frozen(stage)
        verify_hashes(root,{BINDING:binding_hash})
        if notebook_code_sha256(root/NOTEBOOK)!=record['notebook_code_sha256']:
            raise ValueError('El notebook cambió durante la evaluación')
        record.update(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),test_evaluated=True,
            rows=len(scores),frauds=int(scores.isFraud.sum()),model='lightgbm',
            original_protocol_sha256=binding['parent_protocol_sha256'],
            test_id_time_sha256=protocol['test']['id_time_sha256'],costs_hypothetical=True,
            historical_test_label_aggregates_consulted=True,validation_previously_consulted=True,
            input_sha256={p:h for p,h in protocol['dependency_sha256'].items()},
            output_sha256={p:sha256_file(root/p) for p in output_paths},last_step='complete')
        update_both(root,stage,record)
        return tables,record
    except BaseException as error:
        record.update(status='failed',failed_utc=datetime.now(timezone.utc).isoformat(),
                      error_type=type(error).__name__,error_message=str(error))
        update_run(root,record)
        if (stage/TEST_RECORD).exists() and json.loads((stage/TEST_RECORD).read_text(encoding='utf-8')).get('run_id')==record['run_id']:
            update_run(stage,record)
        raise
