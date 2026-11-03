"""Diccionario de columnas y linaje del predictor existente, sin ajuste."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd

from fraud_cost.reference import CATEGORICAL, EXCLUDED

SOURCE_URL = "https://www.kaggle.com/competitions/ieee-fraud-detection/data"
SEMANTIC_CATEGORICAL = CATEGORICAL | {f"id_{i:02d}" for i in range(12, 39)}


def column_definition(name):
    """Describe solo lo verificable; no atribuye significados a códigos ocultos."""
    special = {
        "TransactionID": ("identificador", "identificador", "Clave de unión y alineación, no predictor"),
        "isFraud": ("objetivo", "etiqueta_binaria", "Etiqueta observada: 1 fraude; 0 legítima"),
        "TransactionDT": ("tiempo", "tiempo_relativo", "Segundos desde una referencia, no fecha calendario"),
        "TransactionAmt": ("monto", "numerica", "Monto original; moneda no verificada en este proyecto"),
        "ProductCD": ("producto", "categorica", "Código de producto; significado de cada nivel no verificado"),
        "P_emaildomain": ("correo", "categorica", "Dominio del correo del comprador; no identidad personal"),
        "R_emaildomain": ("correo", "categorica", "Dominio del correo del destinatario; no identidad personal"),
        "DeviceType": ("dispositivo", "categorica", "Categoría del tipo de dispositivo"),
        "DeviceInfo": ("dispositivo", "categorica", "Descriptor del dispositivo, no identificador físico verificado"),
    }
    if name in special:
        return special[name]
    family = next((f for f in ("card", "addr", "dist", "id_", "C", "D", "M", "V")
                   if re.fullmatch(re.escape(f)+r"\d+", name)), "otra")
    descriptions = {"card": "Código de información de tarjeta, sin identidad física verificada",
                    "addr": "Código de información de dirección, no dirección personal reconstruida",
                    "dist": "Descriptor numérico de distancia; escala específica no verificada",
                    "id_": "Descriptor anonimizado de identidad; significado individual no verificado",
                    "C": "Descriptor anonimizado de la familia C; significado individual no verificado",
                    "D": "Descriptor anonimizado de la familia D; significado individual no verificado",
                    "M": "Código anonimizado de la familia M; significado individual no verificado",
                    "V": "Atributo anonimizado del proveedor; no fue creado por nuestro equipo"}
    return family, "categorica" if name in SEMANTIC_CATEGORICAL else "numerica", descriptions.get(family, "Significado no documentado")


def engineering_definitions():
    return {
        "has_identity": ("TransactionID", "1 si existe fila de identity; 0 si no", "Cobertura estructural de identidad"),
        "split": ("TransactionID; TransactionDT", "Bloque del manifiesto temporal", "Control; no predictor"),
    }


def source_catalogue(transactions, identity, profile):
    """434 filas originales; perfil EDA descriptivo, nunca ajusta transformaciones.

    profile contiene dtype, n_unicos y ausencia de la unión inicial. La regla
    de tipos describe el código vigente, no una selección aprendida del EDA.
    """
    columns = list(dict.fromkeys([*transactions.columns, *identity.columns]))
    if not profile.index.is_unique or set(columns) != set(profile.index):
        raise ValueError("El perfil debe cubrir exactamente las columnas originales")
    rows = []
    for name in columns:
        family, semantic, description = column_definition(name)
        dtype = str(profile.loc[name, "dtype"])
        inferred = "categorica" if name in CATEGORICAL or dtype in {"object", "string", "category"} or dtype.startswith("string") else "numerica"
        direct = name not in EXCLUDED
        origins = [label for label, frame in (("transaction", transactions), ("identity", identity)) if name in frame.columns]
        derived = [feature for feature, (parents, _, _) in engineering_definitions().items() if name in parents.split("; ")]
        rows.append({"variable": name, "origen": "+".join(origins), "familia": family,
                     "significado": description, "tipo_semantico": semantic, "dtype_union_eda": dtype,
                     "tipo_regla_vigente": inferred if direct else "no_entrada_directa",
                     "entra_modelo": direct,
                     "uso_auxiliar": {"TransactionID": "Unión uno a uno; manifiesto; alineación de scores/acciones",
                                       "TransactionDT": "Orden y partición temporal; no entrada directa",
                                       "TransactionAmt": "Monto original de pérdida proxy y evaluación económica",
                                       "isFraud": "Objetivo en ajuste; etiqueta real solo al evaluar acciones"}.get(name, ""),
                     "motivo_uso": "Conservar atributo original; sin selección por ausencia/correlación/importancia" if direct else description,
                     "ingenieria_asociada": "; ".join(dict.fromkeys(derived)),
                     "ausencia_eda": float(profile.loc[name, "ausencia"]),
                     "cardinalidad_eda": int(profile.loc[name, "n_unicos"]),
                     "discrepancia_tipo": bool(direct and semantic == "categorica" and inferred != "categorica")})
    return pd.DataFrame(rows)



def export_ordinal_inventory(preprocessor, output_dir):
    """Registra columnas, medianas y vocabularios reales de la nueva corrida."""
    output_dir = Path(output_dir)
    original = pd.read_csv(output_dir / 'eda_diccionario_variables.csv').set_index('variable')
    rows, categories = [], []
    for branch, transformer, columns in preprocessor.transformers_:
        if branch not in {'categorical', 'numeric'}:
            continue
        for j, column in enumerate(columns):
            semantic = column_definition(column)[1] if column != 'has_identity' else 'binaria'
            rows.append({
                'posicion_matriz': len(rows), 'variable': column,
                'origen': original.loc[column, 'origen'] if column in original.index else 'ingenieria',
                'tipo_semantico': semantic, 'rama': branch,
                'tratamiento': 'ordinal; desconocido=-1' if branch == 'categorical' else 'mediana de train; sin indicador',
                'mediana_train': transformer.statistics_[j] if branch == 'numeric' else np.nan,
                'categorias_train': len(transformer.categories_[j]) if branch == 'categorical' else np.nan,
                'discrepancia_tipo': semantic == 'categorica' and branch != 'categorical',
            })
            if branch == 'categorical':
                categories.extend({'variable': column, 'codigo': k, 'categoria': str(value)}
                                  for k, value in enumerate(transformer.categories_[j]))
    inventory = pd.DataFrame(rows)
    inventory.to_csv(output_dir / 'modelo_variables_ordinal.csv', index=False)
    pd.DataFrame(categories).to_csv(output_dir / 'modelo_vocabulario_ordinal.csv', index=False)
    return inventory
