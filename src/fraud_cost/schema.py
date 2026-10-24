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

from fraud_cost.features import CATEGORICAL, ENTITY_COLUMNS, EXCLUDED
from fraud_cost.refit import source_sha256, verified_selection

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
    definitions = {
        "has_identity": ("TransactionID", "1 si existe fila de identity; 0 si no", "Conservar ausencia estructural sin eliminar operaciones"),
        "split": ("TransactionID; TransactionDT; manifiesto", "Etiqueta de bloque temporal, no atributo predictivo", "Controlar usos permitidos y excluir gaps/test del ajuste"),
        "amount_log1p": ("TransactionAmt", "log(1+A)", "Comprimir escala predictiva sin cambiar el monto económico"),
        "missing_count": ("Atributos base y amount_log1p antes de rellenar categorías", "Número de ausencias por fila antes de imputar", "Resumir disponibilidad de información, sin consultar etiquetas"),
    }
    for cycle, period in (("day", 86400), ("week", 604800)):
        for function in ("sin", "cos"):
            definitions[f"relative_{cycle}_{function}"] = ("TransactionDT", f"{function}(2*pi*TransactionDT/{period})", "Fase relativa; no hora local ni día calendario verificado")
    for entity, parents in ENTITY_COLUMNS.items():
        key = "; ".join(parents)
        definitions[f"{entity}_past_count"] = (key+"; TransactionDT", "Conteo con tiempo estrictamente anterior en train", "Frecuencia histórica del proxy, no tasa de fraude")
        definitions[f"{entity}_past_mean_amount"] = (key+"; TransactionDT; TransactionAmt", "Suma de montos anteriores / conteo anterior", "Monto habitual del proxy; ausente si no hay historia")
        definitions[f"{entity}_seconds_since_last"] = (key+"; TransactionDT", "TransactionDT actual - tiempo previo del proxy", "Recencia; ausente si no hay historia")
    return definitions


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
        if direct:
            derived.append("missing_count")
        rows.append({"variable": name, "origen": "+".join(origins), "familia": family,
                     "significado": description, "tipo_semantico": semantic, "dtype_union_eda": dtype,
                     "tipo_regla_vigente": inferred if direct else "no_entrada_directa",
                     "entra_modelo": direct,
                     "uso_auxiliar": {"TransactionID": "Unión uno a uno; manifiesto; alineación de scores/acciones",
                                       "TransactionDT": "Partición; pasado estricto; fases y recencia",
                                       "TransactionAmt": "Monto original de pérdida proxy y evaluación económica",
                                       "isFraud": "Objetivo en ajuste; etiqueta real solo al evaluar acciones"}.get(name, ""),
                     "motivo_uso": "Conservar atributo original; sin selección por ausencia/correlación/importancia" if direct else description,
                     "ingenieria_asociada": "; ".join(dict.fromkeys(derived)),
                     "ausencia_eda": float(profile.loc[name, "ausencia"]),
                     "cardinalidad_eda": int(profile.loc[name, "n_unicos"]),
                     "discrepancia_tipo": bool(direct and semantic == "categorica" and inferred != "categorica")})
    return pd.DataFrame(rows)


def matrix_lineage(predictor):
    """Una fila por posición real de la matriz, con su columna de origen."""
    pre = predictor.preprocessor
    if [name for name, _, _ in pre.transformers_ if name in {"numeric", "categorical"}] != ["numeric", "categorical"]:
        raise ValueError("Cambió el orden de ramas; revisar el linaje antes de exportar")
    groups = {name: (transformer, list(columns)) for name, transformer, columns in pre.transformers_ if name in {"numeric", "categorical"}}
    if set(groups) != {"numeric", "categorical"}:
        raise ValueError("Se requieren las dos ramas del preprocesador registrado")
    numeric_pipeline, numeric = groups["numeric"]
    categorical_pipeline, categorical = groups["categorical"]
    imputer, encoder = numeric_pipeline.named_steps["impute"], categorical_pipeline.named_steps["encode"]
    if encoder.drop is not None:
        raise ValueError("Esta auditoría debe adaptarse si cambia la política de drop del encoder")
    rows = [{"variable": name, "nombre_matriz": "numeric__"+name, "operacion": "mediana_train_y_float32"} for name in numeric]
    rows += [{"variable": numeric[int(index)], "nombre_matriz": "numeric__missingindicator_"+numeric[int(index)],
              "operacion": "indicador_ausencia_detectada_en_train"} for index in imputer.indicator_.features_]
    names = encoder.get_feature_names_out(categorical)
    offset = 0
    for index, name in enumerate(categorical):
        rare = encoder.infrequent_categories_[index] if encoder.min_frequency is not None or encoder.max_categories is not None else None
        width = len(encoder.categories_[index]) - (len(rare)-1 if rare is not None else 0)
        rows += [{"variable": name, "nombre_matriz": "categorical__"+out, "operacion": "one_hot_vocabulario_train"} for out in names[offset:offset+width]]
        offset += width
    if offset != len(names) or len(rows) != predictor.model.n_features_in_:
        raise ValueError("El linaje no coincide con la anchura del modelo entrenado")
    result = pd.DataFrame(rows)
    result.insert(0, "posicion_cero_based", np.arange(len(result)))
    if result.nombre_matriz.duplicated().any() or set(numeric) & set(categorical):
        raise ValueError("El esquema tiene nombres o ramas ambiguos")
    return result


def fitted_catalogue(predictor, original):
    """Amplía el inventario con creadas y tratamiento efectivamente aprendido."""
    if original.variable.duplicated().any():
        raise ValueError("El diccionario original debe tener una fila por variable")
    result = original.copy()
    extras = []
    for name, (parents, formula, reason) in engineering_definitions().items():
        extras.append({"variable": name, "origen": "equipo", "familia": "control" if name == "split" else "ingenieria",
                       "significado": reason, "tipo_semantico": "bloque" if name == "split" else "binaria" if name == "has_identity" else "numerica",
                       "dtype_union_eda": "no_existia_en_union_original", "motivo_uso": reason,
                       "formula_o_procedencia": formula, "variables_padre": parents})
    result = pd.concat([result, pd.DataFrame(extras)], ignore_index=True)
    lineage = matrix_lineage(predictor)
    branches = {name: (pipeline, list(columns)) for name, pipeline, columns in predictor.preprocessor.transformers_ if name in {"numeric", "categorical"}}
    numeric, categorical = branches["numeric"][1], branches["categorical"][1]
    if set(predictor.feature_columns) != set(numeric+categorical) or len(predictor.feature_columns) != len(numeric+categorical):
        raise ValueError("Las ramas no corresponden a las entradas del predictor")
    if not set(predictor.feature_columns).issubset(set(result.variable)):
        raise ValueError("Hay entradas entrenadas sin documentación")
    mapping = {name: "numerica" for name in numeric} | {name: "categorica" for name in categorical}
    result["entra_modelo"] = result.variable.isin(predictor.feature_columns)
    result["tipo_pipeline_verificado"] = result.variable.map(mapping).fillna("no_entrada_directa")
    result["discrepancia_tipo"] = result.entra_modelo & result.tipo_semantico.eq("categorica") & ~result.tipo_pipeline_verificado.eq("categorica")
    result["columnas_matriz"] = result.variable.map(Counter(lineage.variable)).fillna(0).astype(int)
    result["tratamiento"] = np.select([result.tipo_pipeline_verificado.eq("numerica"), result.tipo_pipeline_verificado.eq("categorica")],
                                       ["Inf a NaN; mediana train; indicador si faltó en train; float32; sin escalado",
                                        "Texto y token __MISSING__; one-hot; mínimo 50; desconocidas ignoradas"], default="Solo control, etiqueta o construcción auxiliar; no predictor directo")
    imp = branches["numeric"][0].named_steps["impute"]
    result["mediana_ajustada_train"] = result.variable.map(dict(zip(numeric, imp.statistics_)))
    indicator_names = {numeric[int(i)] for i in imp.indicator_.features_}
    result["indicador_numerico_ausencia"] = result.variable.isin(indicator_names)
    return result.fillna({"ingenieria_asociada": "", "variables_padre": "", "formula_o_procedencia": ""}), lineage


def export_model_inventory(root):
    """Audita artefactos propios verificados; no fit, predicción o carga de test."""
    root = Path(root)
    dictionary = root/"results/tables/eda_diccionario_variables.csv"
    if not dictionary.exists():
        raise FileNotFoundError("Ejecute el perfil y diccionario del notebook 01 antes de auditar el modelo")
    verified_selection(root)
    config = json.loads((root/"results/tables/reajuste_config.json").read_text(encoding="utf-8"))
    if config.get("test_evaluated") is not False:
        raise ValueError("La auditoría de desarrollo requiere conservar test sin evaluar")
    for group in ("source_sha256", "artifact_sha256"):
        for relative, expected in config[group].items():
            path = root/relative.replace("\\", "/")
            if source_sha256(path, code_only=path.suffix == ".ipynb") != expected:
                raise ValueError(f"Cambió una fuente o artefacto del 06: {relative}")
    model_path = root/"results/models/selected_xgboost_platt.joblib"
    predictor = joblib.load(model_path)
    if predictor.params != config["predictor_params"]:
        raise ValueError("El predictor no corresponde a los parámetros registrados")
    before = joblib.hash(predictor)
    original = pd.read_csv(dictionary, keep_default_na=False, float_precision="round_trip")
    catalogue, lineage = fitted_catalogue(predictor, original)
    if joblib.hash(predictor) != before:
        raise ValueError("La auditoría modificó el predictor")
    paths = [root/"results/tables/modelo_diccionario_variables.csv", root/"results/tables/modelo_matriz_variables.csv"]
    for path, frame in zip(paths, (catalogue, lineage)):
        frame.to_csv(path, index=False)
    record = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "model_refitted": False, "test_evaluated": False,
              "audit_scope": "esquema y transformaciones del predictor existente; sin nuevas predicciones ni estadísticas de etiquetas",
              "source": SOURCE_URL, "original_columns": len(original), "catalogue_rows": len(catalogue),
              "original_direct_inputs": int(original.entra_modelo.sum()),
              "model_inputs": int(catalogue.entra_modelo.sum()),
              "numeric_inputs": int(catalogue.tipo_pipeline_verificado.eq("numerica").sum()),
              "categorical_inputs": int(catalogue.tipo_pipeline_verificado.eq("categorica").sum()),
              "matrix_columns": len(lineage), "matrix_operations": dict(Counter(lineage.operacion)),
              "categorical_discrepancies": catalogue.loc[catalogue.discrepancia_tipo, "variable"].tolist(),
              "model_sha256": source_sha256(model_path),
              "source_sha256": {str(p.relative_to(root)).replace("\\", "/"): source_sha256(p) for p in
                                (root/"src/fraud_cost/schema.py", root/"src/fraud_cost/features.py", dictionary, root/"results/tables/reajuste_config.json")},
              "artifact_sha256": {str(p.relative_to(root)).replace("\\", "/"): source_sha256(p) for p in paths},
              "profile_scope": "ausencia/cardinalidad originales son EDA global histórico, no estadísticas de ajuste; medianas e indicadores son de train"}
    (root/"results/tables/inventario_variables_config.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return catalogue, lineage, record
