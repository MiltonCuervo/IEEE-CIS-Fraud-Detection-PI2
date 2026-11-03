"""Adopción del LightGBM seleccionado y evaluación sin nuevos ajustes."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, precision_score, recall_score

from fraud_cost.artifacts import read_record, verify_reference_artifacts
from fraud_cost.reference import make_features
from fraud_cost.costs import bmr_actions, evaluate_policies

LOGIT_CLIP=1e-6
ATOL=RTOL=1e-8


def load_selected_reference(root):
    """Verifica bytes antes de deserializar un paquete propio, original o recuperado."""
    root=__import__('pathlib').Path(root)
    evidence=verify_reference_artifacts(root)
    selection=read_record(root,'results/tables/seleccion_ordinal_config.json')['selected_candidate']
    reference=read_record(root,'results/tables/comparacion_modelos_config.json')
    if selection['model']!='lightgbm' or selection['params']!=reference['params']['lightgbm']:
        raise ValueError('La selección no coincide con el LightGBM del 03; requiere otro protocolo de ajuste')
    bundle=joblib.load(root/'results/models/comparacion_modelos_ordinal.joblib')
    model=bundle['models']['lightgbm']
    calibrator=bundle['calibrators']['lightgbm']
    preprocessor=bundle['preprocessor']
    if any(model.get_params()[k]!=v for k,v in selection['params'].items()):
        raise ValueError('Los parámetros del modelo no corresponden a la selección')
    if preprocessor.n_features_in_!=432 or model.n_features_in_!=432:
        raise ValueError('No corresponde a la representación ordinal de 432 entradas')
    if list(model.classes_)!=[0,1] or list(calibrator.classes_)!=[0,1] or calibrator.coef_.shape!=(1,1):
        raise ValueError('El predictor o el Platt no tienen el contrato binario esperado')
    trees=joblib.load(root/'results/models/cost_sensitive_trees.joblib')
    if set(trees)!=set(reference['ca_grid']):
        raise ValueError('No hay un árbol verificado por cada Ca de referencia')
    for tree in trees.values():
        if tree.n_features_in_!=432:
            raise ValueError('Árbol de otra representación')
    return {'preprocessor':preprocessor,'model':model,'calibrator':calibrator,
            'model_name':'lightgbm','calibration_method':'platt_on_clipped_logit',
            'logit_clip':LOGIT_CLIP,'selected_candidate':selection,
            'adoption_method':'verified_reuse_stage03_no_refit'},trees,evidence,reference


def validation_policy(frame,fraction=0.5):
    """Recupera las ventanas del 03; nunca transforma ni resume etiquetas de test."""
    if not 0<fraction<1:
        raise ValueError('Fracción de calibración inválida')
    validation=frame.loc[frame['split'].eq('validation')].sort_values(['TransactionDT','TransactionID']).reset_index(drop=True)
    end=int(round(len(validation)*fraction))
    if end==0 or end>=len(validation):
        raise ValueError('Validación insuficiente')
    while end<len(validation) and validation.iloc[end-1].TransactionDT==validation.iloc[end].TransactionDT:
        end+=1
    early=validation.iloc[:end]
    late=validation.iloc[end:].reset_index(drop=True)
    if late.empty or early.isFraud.nunique()!=2 or late.isFraud.nunique()!=2:
        raise ValueError('Cada ventana de validación debe contener ambas clases')
    if late.TransactionID.isna().any() or late.TransactionID.duplicated().any():
        raise ValueError('IDs de evaluación inválidos')
    metadata={'calibration_rows':len(early),'policy_rows':len(late),
              'calibration_time_start':int(early.TransactionDT.min()),'calibration_time_end':int(early.TransactionDT.max()),
              'policy_time_start':int(late.TransactionDT.min()),'policy_time_end':int(late.TransactionDT.max())}
    return late,metadata


def apply_platt(raw,calibrator,eps=LOGIT_CLIP):
    raw=np.asarray(raw,dtype=float)
    if raw.ndim!=1 or not np.isfinite(raw).all() or ((raw<0)|(raw>1)).any() or not 0<eps<0.5:
        raise ValueError('Probabilidades o clipping inválidos')
    clipped=np.clip(raw,eps,1-eps)
    return calibrator.predict_proba(np.log(clipped/(1-clipped)).reshape(-1,1))[:,1]


def selected_scores(predictor,policy):
    features=make_features(policy)
    expected=list(predictor['preprocessor'].feature_names_in_)
    if list(features.columns)!=expected:
        raise ValueError('El orden/esquema de atributos difiere del ajuste verificado')
    matrix=predictor['preprocessor'].transform(features)
    if matrix.shape!=(len(policy),432) or not np.isfinite(matrix).all():
        raise ValueError('Matriz incompatible con el modelo seleccionado')
    raw=predictor['model'].predict_proba(matrix)[:,1]
    calibrated=apply_platt(raw,predictor['calibrator'],predictor['logit_clip'])
    scores=policy[['TransactionID','TransactionDT','TransactionAmt','isFraud']].copy()
    scores['probability_raw']=raw
    scores['probability_calibrated']=calibrated
    scores['action_fixed_0_5']=np.where(calibrated>=0.5,'block','approve')
    return scores,matrix


def probability_metrics(scores):
    y=scores.isFraud.to_numpy()
    rows=[]
    for kind,column in [('raw','probability_raw'),('platt','probability_calibrated')]:
        p=scores[column].to_numpy()
        blocked=p>=0.5
        rows.append({'model':'lightgbm','probability':kind,'rows':len(y),'frauds':int(y.sum()),
            'fraud_rate':float(y.mean()),'mean_probability':float(p.mean()),
            'average_precision':float(average_precision_score(y,p)),
            'brier_score':float(brier_score_loss(y,p)), 'log_loss':float(log_loss(y,p,labels=[0,1])),
            'recall_at_0_5':float(recall_score(y,blocked)),
            'precision_at_0_5':float(precision_score(y,blocked,zero_division=0)),
            'legitimate_block_rate':float(blocked[y==0].mean()),'block_count':int(blocked.sum())})
    return pd.DataFrame(rows)


def calibration_groups(scores,bins=10):
    """Grupos por cuantiles de Platt, compartidos para comparar raw y calibrada."""
    data=scores.copy()
    data['group']=pd.qcut(data.probability_calibrated,q=bins,duplicates='drop')
    rows=[]
    for group,block in data.groupby('group',observed=True):
        rows.append({'group':str(group),'rows':len(block),'frauds':int(block.isFraud.sum()),
            'observed_rate':float(block.isFraud.mean()),'mean_raw':float(block.probability_raw.mean()),
            'mean_platt':float(block.probability_calibrated.mean()),
            'min_platt':float(block.probability_calibrated.min()),'max_platt':float(block.probability_calibrated.max())})
    return pd.DataFrame(rows)


def amount_diagnostics(scores,edges=(0,50,100,250,500,1000,np.inf)):
    data=scores.copy()
    data['band']=pd.cut(data.TransactionAmt,bins=edges,right=False,include_lowest=True)
    if data.band.isna().any():
        raise ValueError('Las bandas deben cubrir todos los montos')
    rows=[]
    for band,block in data.groupby('band',observed=True):
        rows.append({'amount_band':str(band),'rows':len(block),'frauds':int(block.isFraud.sum()),
            'observed_rate':float(block.isFraud.mean()),'mean_probability':float(block.probability_calibrated.mean()),
            'observed_fraud_amount':float((block.isFraud*block.TransactionAmt).sum()),
            'expected_fraud_amount_raw':float((block.probability_raw*block.TransactionAmt).sum()),
            'expected_fraud_amount_platt':float((block.probability_calibrated*block.TransactionAmt).sum())})
    return pd.DataFrame(rows)


def economic_validation(scores,matrix,trees,ca_grid):
    """Tres políticas con el mismo Ca y filas; árboles existentes, sin fit."""
    if scores.empty or scores.TransactionID.isna().any() or scores.TransactionID.duplicated().any() or matrix.shape[0]!=len(scores):
        raise ValueError('Filas económicas desalineadas o vacías')
    values=tuple(float(c) for c in ca_grid)
    if not values or len(set(values))!=len(values) or set(trees)!=set(values):
        raise ValueError('Grid y árboles incompatibles')
    p=scores.probability_calibrated.to_numpy()
    fixed=scores.action_fixed_0_5.to_numpy()
    if not np.array_equal(fixed,np.where(p>=0.5,'block','approve')):
        raise ValueError('La política fija no corresponde a la probabilidad calibrada')
    costs,actions,sensitivity=[],[],[]
    for ca in values:
        prediction=np.asarray(trees[ca].predict(matrix))
        if not np.isin(prediction,[0,1]).all():
            raise ValueError('El árbol debe devolver acciones binarias')
        policies={'fixed_0_5':fixed,'bmr':bmr_actions(p,scores.TransactionAmt,ca),
                  'cost_sensitive_tree':np.where(prediction==1,'block','approve')}
        table=evaluate_policies(scores.isFraud,scores.TransactionAmt,policies,ca)
        table['model']='lightgbm';table['evaluation_block']='validation_policy'
        costs.append(table)
        action_table=pd.DataFrame({'TransactionID':scores.TransactionID,'administrative_cost_ca':ca,**policies})
        actions.append(action_table)
        for kind,column in [('raw','probability_raw'),('platt','probability_calibrated')]:
            diagnostic=evaluate_policies(scores.isFraud,scores.TransactionAmt,
                {'fixed_0_5':fixed,'bmr':bmr_actions(scores[column],scores.TransactionAmt,ca)},ca)
            row=diagnostic.loc[diagnostic.strategy.eq('bmr')].copy()
            row['probability']=kind
            sensitivity.append(row)
    return pd.concat(costs,ignore_index=True),pd.concat(actions,ignore_index=True),pd.concat(sensitivity,ignore_index=True)
