"""Selección acotada de cuatro predictores con pasado expansivo dentro de train."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from fraud_cost.reference import build_preprocessor, make_features


def sha256_file(path):
    digest=hashlib.sha256()
    with open(path,'rb') as source:
        for chunk in iter(lambda:source.read(8*1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_grid(reference_params):
    """Una referencia del 03 y una alternativa regularizada por algoritmo.

    Es un contraste de configuraciones completas, no una ablación de parámetros.
    No cambia la representación ni activa el manejo nativo de categorías.
    """
    changes={
        'xgboost': {'max_depth':4,'min_child_weight':20,'reg_lambda':10},
        'lightgbm': {'num_leaves':31,'min_child_samples':100,'reg_lambda':5},
        'catboost': {'depth':4,'l2_leaf_reg':10},
        'logistica': {'C':0.01},
    }
    if set(reference_params)!=set(changes):
        raise ValueError('Se requieren los cuatro algoritmos de la etapa 03')
    candidates=[]
    for model,alternative in changes.items():
        for variant in ('referencia','regularizada'):
            params=deepcopy(reference_params[model])
            if variant=='regularizada':
                params.update(alternative)
            candidates.append({'candidate':f'{model}_{variant}','model':model,
                               'variant':variant,'params':params})
    return candidates


def expanding_folds(train, *, fractions=(0.5,0.75), gap_days=7):
    """Cortes sobre duración; timestamps iguales permanecen en un solo bloque.

    La segunda ventana puede entrenar con el pasado evaluado en la primera:
    cada ajuste es nuevo y solo usa información anterior a su propio futuro.
    """
    required={'TransactionID','TransactionDT','isFraud','split'}
    if not required.issubset(train) or train.empty:
        raise ValueError('Falta el esquema temporal de entrenamiento')
    if not train['split'].eq('train').all() or train.TransactionID.isna().any() or train.TransactionID.duplicated().any():
        raise ValueError('Solo se permiten filas únicas del bloque train')
    time=train.TransactionDT.to_numpy(dtype=float)
    if not np.isfinite(time).all() or np.any(np.diff(time)<0):
        raise ValueError('Train debe estar ordenado por tiempo finito')
    if not np.isin(train.isFraud,[0,1]).all():
        raise ValueError('Las etiquetas deben ser binarias')
    fractions=tuple(float(f) for f in fractions)
    if not fractions or any(not 0<f<1 for f in fractions) or any(a>=b for a,b in zip(fractions,fractions[1:])):
        raise ValueError('Los cortes deben ser crecientes y estar entre cero y uno')
    if not np.isfinite(gap_days) or gap_days<0 or time[-1]<=time[0]:
        raise ValueError('Gap o duración temporal inválidos')
    start,span=time[0],time[-1]-time[0]
    folds=[]
    for i,(fraction,end_fraction) in enumerate(zip(fractions,(*fractions[1:],1.0)),1):
        cutoff=start+span*fraction
        future_end=start+span*end_fraction
        future_start=cutoff+gap_days*86400
        fit=np.flatnonzero(time<=cutoff)
        evaluate=np.flatnonzero((time>future_start)&(time<=future_end))
        if not len(fit) or not len(evaluate):
            raise ValueError('Un gap consume la ventana o faltan filas')
        for indices in (fit,evaluate):
            if train.iloc[indices].isFraud.nunique()!=2:
                raise ValueError('Cada bloque interno debe contener ambas clases')
        if time[evaluate[0]]-time[fit[-1]]<=gap_days*86400:
            raise AssertionError('La separación real debe superar el gap')
        folds.append({'fold':i,'fit':fit,'evaluate':evaluate,
                      'cutoff':float(cutoff),'future_start_exclusive':float(future_start),
                      'future_end_inclusive':float(future_end),'gap_days':float(gap_days)})
    return folds


def describe_folds(train,folds):
    rows=[]
    for fold in folds:
        for role in ('fit','evaluate'):
            block=train.iloc[fold[role]]
            ids=block.TransactionID.to_numpy(dtype='<i8')
            rows.append({'fold':fold['fold'],'role':role,'rows':len(block),
                         'frauds':int(block.isFraud.sum()),'fraud_rate':float(block.isFraud.mean()),
                         'time_start':int(block.TransactionDT.min()),'time_end':int(block.TransactionDT.max()),
                         'gap_days':fold['gap_days'],'ids_sha256':hashlib.sha256(ids.tobytes()).hexdigest()})
    return pd.DataFrame(rows)


def prepare_fold(train,fold):
    """Ajusta vocabularios y medianas solo con el pasado de esta ventana."""
    fit=train.iloc[fold['fit']]
    evaluate=train.iloc[fold['evaluate']]
    X_fit,X_evaluate=make_features(fit),make_features(evaluate)
    pre,categorical,numeric=build_preprocessor(X_fit)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        A=pre.fit_transform(X_fit)
        B=pre.transform(X_evaluate)
    if not np.isfinite(A).all() or not np.isfinite(B).all():
        raise ValueError('La preparación produjo valores no finitos')
    encoder=pre.named_transformers_['categorical']
    novelty=[]
    for column,known in zip(categorical,encoder.categories_):
        unknown=~X_evaluate[column].isin(known)
        novelty.append({'fold':fold['fold'],'variable':column,'train_categories':len(known),
                        'unknown_rows':int(unknown.sum()),'evaluation_rows':len(evaluate),
                        'unknown_fraction':float(unknown.mean())})
    details={'categorical_count':len(categorical),'numeric_count':len(numeric),'n_features':A.shape[1],
             'empty_numeric_columns':list(pre.named_transformers_['numeric'].empty_columns_),
             'preparation_warnings':[str(w.message) for w in caught]}
    return A,B,fit.isFraud.to_numpy(dtype=np.int8),evaluate.isFraud.to_numpy(dtype=np.int8),pd.DataFrame(novelty),details


def build_candidate(candidate):
    factories={'xgboost':XGBClassifier,'lightgbm':LGBMClassifier,
               'catboost':CatBoostClassifier,'logistica':LogisticRegression}
    model=factories[candidate['model']](**candidate['params'])
    if candidate['model']=='logistica':
        return make_pipeline(StandardScaler(with_mean=False),model)
    return model


def evaluate_candidate(candidate,A,B,y_fit,y_evaluate):
    """Entrena desde cero, sin eval_set, calibrador ni early stopping externo."""
    model=build_candidate(candidate)
    started=perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model.fit(A,y_fit)
        fit_seconds=perf_counter()-started
        p=np.asarray(model.predict_proba(B)[:,1],dtype=float)
    if len(p)!=len(y_evaluate) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Probabilidades inválidas en la evaluación interna')
    return {'candidate':candidate['candidate'],'model':candidate['model'],'variant':candidate['variant'],
            'average_precision':float(average_precision_score(y_evaluate,p)),
            'brier_raw':float(brier_score_loss(y_evaluate,p)),
            'log_loss_raw':float(log_loss(y_evaluate,p,labels=[0,1])),
            'mean_probability_raw':float(p.mean()),'fraud_rate':float(np.mean(y_evaluate)),
            'fit_seconds':float(fit_seconds),'evaluation_rows':len(y_evaluate),
            'fit_warnings':json.dumps([{'category':w.category.__name__,'message':str(w.message)} for w in caught],ensure_ascii=False),
            'convergence_warning':any(issubclass(w.category, ConvergenceWarning) for w in caught)}


def rank_candidates(results,candidates,fold_count):
    expected={(c['candidate'],f) for c in candidates for f in range(1,fold_count+1)}
    actual=list(zip(results.candidate,results.fold))
    if len(actual)!=len(expected) or set(actual)!=expected:
        raise ValueError('Se necesitan exactamente todos los candidatos y folds, sin duplicados')
    if not np.isfinite(results.average_precision).all():
        raise ValueError('No se puede seleccionar con AP no finita')
    summary=results.groupby(['candidate','model','variant'],sort=False).agg(
        mean_ap=('average_precision','mean'),std_ap=('average_precision','std'),
        min_ap=('average_precision','min'),max_ap=('average_precision','max'),
        mean_brier_raw=('brier_raw','mean'),mean_log_loss_raw=('log_loss_raw','mean'),
        fit_seconds_total=('fit_seconds','sum'),convergence_warnings=('convergence_warning','sum'),
    ).reset_index()
    # Igual peso por ventana. Ante igualdad exacta, identificador lexicográfico.
    # La dispersión no es un intervalo de confianza ni se usa para cambiar la regla.
    return summary.sort_values(['mean_ap','candidate'],ascending=[False,True],kind='stable').reset_index(drop=True)
