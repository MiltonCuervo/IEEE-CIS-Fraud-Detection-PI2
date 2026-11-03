# Diccionario de variables y recorrido hasta la decisión

**Base vigente: notebooks 01–03 de la profesora, consolidada el 2 de noviembre de 2026.** Esta versión reemplaza la descripción del predictor anterior de 447 entradas y one-hot. Conservamos las definiciones verificables del EDA y describimos el tratamiento ordinal realmente implementado.

## 1. Inventario y procedencia

`train_transaction` tiene 394 columnas y `train_identity`, 41. Comparten TransactionID, por lo que la unión izquierda uno a uno tiene 434 columnas y conserva 590.540 transacciones. Retiramos ID, tiempo crudo y etiqueta de X: quedan 431 atributos originales. Añadimos `has_identity`, que indica correspondencia en la unión y no una identidad personal verificada. `split` controla los bloques y tampoco entra en X.

| Archivo | Qué documenta |
|---|---|
| eda_diccionario_variables.csv | Las 434 originales, origen, tipo semántico, cardinalidad y ausencia descriptivas |
| modelo_variables_ordinal.csv | Las 432 posiciones reales de X, rama, mediana de train y discrepancias de tipo |
| modelo_vocabulario_ordinal.csv | Correspondencia categoría–código aprendida solo con train |
| comparacion_modelos_config.json | Fuentes, versiones, parámetros, entradas, salidas y alcance de la corrida |

Los perfiles globales del EDA describen el dataset; no se utilizan para aprender medianas o vocabularios. El inventario del modelo se genera desde el preprocesador ajustado, no desde una lista escrita a mano.

## 2. Familias originales

| Familia | Variables | Interpretación verificable |
|---|---|---|
| Control | TransactionID, TransactionDT, isFraud | Identificador, segundos relativos y etiqueta de fraude |
| Monto | TransactionAmt | Magnitud original, sin moneda verificada |
| Producto | ProductCD | Código de producto |
| Tarjeta | card1–card6 | Códigos de información de tarjeta, sin reconstrucción de titular |
| Dirección y distancia | addr1–addr2, dist1–dist2 | Códigos de dirección y descriptores numéricos de distancia |
| Correo | P_emaildomain, R_emaildomain | Dominios, no direcciones completas ni personas |
| C | C1–C14 | Descriptores anonimizados del proveedor |
| D | D1–D15 | Descriptores anonimizados; no atribuimos un retraso particular a cada columna |
| M | M1–M9 | Códigos categóricos anonimizados |
| V | V1–V339 | Atributos del proveedor, no ingeniería propia |
| Identidad | id_01–id_38 | Descriptores anonimizados de identidad |
| Dispositivo | DeviceType, DeviceInfo | Tipo y descripción; no identifican un equipo físico de manera garantizada |

La tabla completa del EDA identifica cada columna. No inventamos interpretaciones individuales para los descriptores anonimizados ni eliminamos columnas automáticamente por ausencia, correlación o importancia.

## 3. Tipos y preparación efectiva

Las 37 categóricas son ProductCD; card1–card6; addr1–addr2; P_emaildomain; R_emaildomain; M1–M9; DeviceType; DeviceInfo; id_12, id_15, id_16, id_23, id_27, id_28, id_29, id_30, id_31, id_33, id_34, id_35, id_36, id_37 e id_38. Combinamos una lista explícita con detección de columnas de texto. Un dtype numérico no garantiza significado cuantitativo.

Normalizamos su ausencia con `__MISSING__` antes de codificar. OrdinalEncoder aprende niveles exclusivamente en train y genera una columna float32 por variable. Una categoría no vista, incluido un token de ausencia no observado al ajustar, recibe −1. No usamos frecuencia mínima ni agrupación de categorías raras. Los códigos no expresan distancia, jerarquía ni tasa de fraude: un corte del árbol separa grupos según ese orden artificial.

Las otras 395 entradas son numéricas. Sustituimos infinitos por ausencias y aprendemos medianas por columna, únicamente con train. El imputador se conserva dentro del preprocesador y se reutiliza sin fit en validación y futuras predicciones. Una columna completamente vacía en train se conserva con cero y se informa mediante advertencia. No añadimos indicadores de ausencia.

El orden de matriz es primero categóricas y después numéricas; el inventario registra sus posiciones. No añadimos los quince atributos de la línea experimental retirada: no hay logaritmo de monto, ciclos, conteos, medias históricas ni recencia. La representación de los cuatro modelos tiene 432 columnas.

### Discrepancia de identidad pendiente

La [descripción de Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/data) clasifica id_12–id_38 como categóricas. En la implementación conservada de la profesora, id_13, id_14, id_17, id_18, id_19, id_20, id_21, id_22, id_24, id_25, id_26 e id_32 siguen numéricas. Registramos estas doce discrepancias y su efecto potencial: imponer un orden cuantitativo sin justificación semántica. No las recodificamos silenciosamente; hacerlo constituiría otra configuración que exige evaluación.

## 4. Qué recibe cada modelo

XGBoost, LightGBM y CatBoost reciben la misma matriz ordinal e imputada. CatBoost no recibe `cat_features`: esta comparación no evalúa su tratamiento nativo de categorías. La regresión logística recibe además StandardScaler ajustado exclusivamente en train; el escalado no elimina el orden artificial de las categorías.

El monto predictivo forma parte de X en float32. El monto económico se conserva por separado en float64 desde la lectura del CSV, antes de transformar X. Nunca sustituimos A por un monto estandarizado, imputado o logarítmico al evaluar costos.

## 5. Salidas y responsabilidades

| Etapa | Entrada | Salida y restricción |
|---|---|---|
| Preparación | 431 originales permitidas y has_identity | 432 columnas; ajuste de estadísticas solo en train |
| Predictor | Matriz, escalada solo para logística | predict_proba; columna de fraude como p_raw |
| Platt propio de cada predictor | logit de p_raw recortada a [10⁻⁶, 1−10⁻⁶] | p calibrada; ajuste solo en validación temprana |
| Regla fija | p calibrada | Intervenir si p≥0,5 |
| BMR | p calibrada, A original y Ca | Intervenir si pA>Ca; aprobar en empate |
| Árbol por Ca | Matriz; en train objetivo 1[yA>Ca], peso abs(yA−Ca) | Acción aprendida, no probabilidad calibrada de fraude |
| Evaluación | Acciones, etiqueta, monto original y Ca | Costo, ahorro, recall y bloqueo de legítimas |

La etiqueta se usa para ajustar y medir en bloques permitidos; nunca es una entrada de inferencia. TransactionID alinea las operaciones y TransactionDT determina el tiempo. Ca es un escenario exógeno, no una variable original ni parte del objetivo logístico de los predictores.

El archivo local `validation_scores.parquet` identifica operaciones y conserva monto, etiqueta, probabilidades raw/Platt y acción fija de XGBoost. Su etiqueta sirve para evaluar, no es una predicción. El paquete `comparacion_modelos_ordinal.joblib` conserva los cuatro predictores, sus calibradores, el preprocesador y el escalador de logística. Ninguno equivale a seleccionar automáticamente el modelo final.
