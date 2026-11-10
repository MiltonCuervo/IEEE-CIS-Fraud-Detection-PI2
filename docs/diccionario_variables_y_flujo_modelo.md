# Diccionario de variables y recorrido hasta la decisión

**Base vigente: metodología de los notebooks 01–03, consolidada el 2 de noviembre de 2026.** Esta versión reemplaza la descripción del predictor anterior de 447 entradas y one-hot. Conservamos las definiciones verificables del EDA y describimos el tratamiento ordinal realmente implementado.

> Actualización del 10 de noviembre de 2026: la [aclaración oficial de Vesta](https://www.kaggle.com/competitions/ieee-fraud-detection/discussion/100304) confirma que TransactionAmt fue convertido a USD. Esta identificación documental de la unidad no altera los montos, los escenarios hipotéticos ni los registros históricos congelados.

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
| Monto | TransactionAmt | Monto original en USD, según la aclaración oficial de Vesta |
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

### Ejemplo real del vocabulario ordinal

En `modelo_vocabulario_ordinal.csv`, ProductCD tiene C→0, H→1, R→2, S→3 y W→4.
Son los códigos de esta corrida, aprendidos en train, no una escala de riesgo.
Un corte numérico puede separar grupos contiguos de esos códigos; la logística
les asigna un efecto lineal sobre su logit y el escalado no elimina ese supuesto.
Una categoría nueva recibe −1. En otra ventana el vocabulario se aprende desde
su propio pasado: no suponemos que los códigos coincidan entre ajustes.

### Discrepancia de identidad pendiente

La [descripción de Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/data) clasifica id_12–id_38 como categóricas. En la implementación vigente, id_13, id_14, id_17, id_18, id_19, id_20, id_21, id_22, id_24, id_25, id_26 e id_32 siguen numéricas. Registramos estas doce discrepancias y su efecto potencial: imponer un orden cuantitativo sin justificación semántica. No las recodificamos silenciosamente; hacerlo constituiría otra configuración que exige evaluación.

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

## 6. Selección actual y recuperación local

El notebook 04 conserva exactamente esta representación y seleccionó `lightgbm_referencia` entre ocho configuraciones mediante AP media de dos ventanas internas de train (0,588117). La selección no cambia las 432 entradas ni corrige las doce discrepancias. Sus parámetros coinciden con el LightGBM del 03; la etapa 05 verificó la procedencia y reutilizó ese ajuste y su propio Platt.

Un paquete reconstruido tiene hashes nuevos y un recibo separado en `recuperacion_03_config.json`. Usamos `verify_reference_artifacts` antes de cargarlo. El recibo prueba integridad y equivalencia numérica dentro de tolerancias declaradas, sin reemplazar las huellas históricas del 03/04. `validation_scores.parquet` sigue identificando scores XGBoost: no se convierte en LightGBM por haber seleccionado ese algoritmo. Véase [recuperación y trazabilidad](recuperacion_local_y_trazabilidad.md).


## 7. Adopción del predictor seleccionado

La etapa 05 reutiliza el LightGBM seleccionado, el preprocesador completo y su propio Platt del 03 tras verificar originales o reconstrucción auditada. Conserva 432 entradas y las mismas posiciones, medianas y vocabularios de modelo_variables_ordinal.csv y modelo_vocabulario_ordinal.csv. No aplica otra ingeniería ni corrige silenciosamente tipos.

El paquete local lightgbm_seleccionado_platt.joblib contiene preprocessor, model y calibrator; el Platt recibe el logit recortado de p_raw, no X directamente. lightgbm_validation_scores.parquet contiene ID, tiempo, monto original, etiqueta observada, probabilidad raw, probabilidad calibrada y acción fija a 0,5. La etiqueta allí sirve para medir; no entra al predictor. lightgbm_validation_actions.parquet identifica cada operación/Ca y las acciones fija, BMR y árbol.

validation_scores.parquet continúa siendo XGBoost. Los árboles reutilizados predicen acciones con la misma matriz transformada; no consumen p de LightGBM. Los costos se evalúan con TransactionAmt original, no con su representación predictiva float32. Los archivos por operación y el modelo permanecen locales e ignorados por Git.

## 8. Cierre del recorrido en test

El notebook 08 aplicó esa misma representación a 84.233 operaciones de test. No añadió ni seleccionó variables después de observar resultados. El Platt recibe una columna de logits y las políticas usan probabilidades y montos originales; los árboles conservan sus entradas numéricas de 432 columnas. Los scores guardan siete columnas de control/salida y las acciones registran una fila por operación/Ca. Los conteos y costos se agregan en las tablas test; el informe 08 describe sus resultados y limitaciones.


## 9. Definiciones para leer todos los notebooks e informes

| Término | Definición y uso en este proyecto |
|---|---|
| EDA | Análisis exploratorio de datos; describe estructura y asociaciones antes del modelado. Aquí recorrió todo el conjunto etiquetado y por eso declaramos la consulta histórica de datos futuros. |
| X / y | X es la matriz de atributos que recibe el predictor; y es `isFraud`, la etiqueta conocida solo para entrenar o evaluar. |
| Train / validación / test | Pasado para ajustar; periodo posterior dividido entre calibración y diagnóstico; periodo final para medir el protocolo fijado. El test propio proviene de train etiquetado, no del test sin etiquetas de Kaggle. |
| Gap | Intervalo excluido entre bloques. Son siete días en tiempo relativo; no es una medición del retraso real de etiquetas ni garantiza independencia. |
| Fit / transform / predict | Aprender parámetros; aplicar una transformación ya aprendida; producir una salida con un modelo fijo. Transformar validación no autoriza ajustar sus medianas o vocabularios. |
| Ordinal | Una columna numérica por categoría original, con códigos aprendidos en train. Los códigos no son rangos económicos ni tasas de fraude. El orden artificial puede afectar los cuatro modelos. |
| One-hot | Una columna indicadora por nivel; pertenece a la línea retirada. La reducción de columnas no demuestra por sí sola una mejora predictiva ni constituye una ablación controlada del codificador. |
| Raw / Platt | Probabilidad original del clasificador / probabilidad transformada por su calibrador propio. Llamar a la segunda “calibrada” identifica el procedimiento, no garantiza calibración perfecta en datos futuros. |
| p_i | Estimación de P(fraude dado X_i). Es un riesgo estimado, no una etiqueta conocida ni una probabilidad de intervención. |
| A_i / Ca | Monto original de la operación / costo administrativo por intervención. Ca se cobra tanto al intervenir fraude como al intervenir legítima, bajo nuestra matriz hipotética. |
| p* | Frontera económica Ca/A_i para A_i>0. BMR interviene si p_i>p*, equivalente a p_i A_i>Ca. Si A_i=0 aprobamos; en empate también. |
| BMR | Bayes Minimum Risk: elegir la acción con menor riesgo esperado usando la probabilidad suministrada y la matriz asumida. No garantiza el menor costo realizado en toda muestra. |
| TP / FP / FN / TN | Fraude intervenido / legítima intervenida / fraude aprobado / legítima aprobada. Son conteos de acciones cuando evaluamos políticas. |
| Precision | TP/(TP+FP): proporción de fraudes entre las intervenciones. No es recall ni exactitud global. |
| Recall | TP/(TP+FN): proporción de fraudes intervenidos por cantidad. |
| FPR | False Positive Rate, FP/(FP+TN): fracción de todas las legítimas intervenidas; difiere de la proporción de legítimas entre alertas, 1−precision. |
| Recall por monto | Suma de A_i en fraudes intervenidos dividida por suma de A_i en todos los fraudes del bloque. No incluye el gasto administrativo ni equivale al ahorro. |
| AP | Average Precision: suma de precision ponderada por incrementos de recall al recorrer umbrales; se calcula con `average_precision_score`. Mayor es mejor. Cambiar prevalencia o población limita comparaciones entre etapas. |
| AUC-PR | Área bajo precision–recall. Puede calcularse con integración trapezoidal; por eso no la usamos como sinónimo numéricamente exacto de AP. |
| Brier | Promedio de (p_i−y_i)²; menor es mejor. Evalúa calidad probabilística y no aísla únicamente calibración. |
| Log-loss | Promedio de −[y_i log(p_i)+(1−y_i) log(1−p_i)]; menor es mejor. Penaliza con fuerza errores muy confiados. |
| Discriminación / calibración | Ordenar casos por riesgo / concordancia entre probabilidades anunciadas y frecuencias observadas. Una no garantiza la otra. |
| MCAR / MAR / MNAR | Ausencia completamente al azar / al azar condicionada a información observada / dependiente de información no observada. No diagnosticamos estos mecanismos por porcentajes de ausencia ni por asociación con fraude. |
| SMOTE | Synthetic Minority Over-sampling Technique: genera ejemplos sintéticos minoritarios. No se aplicó; cambiar el balance requiere evaluar el efecto sobre probabilidades y costos. |

Cuando no hay intervenciones, la implementación reporta precision=0 por
convención; eso no equivale a estimar la calidad de alertas inexistentes. Las
fracciones con denominador monetario nulo no se interpretan como ahorros reales.

### Cómo funciona Platt aquí y qué no volvimos a ajustar

Para cada uno de los cuatro predictores del 03 obtenemos su p_raw sobre validación
temprana. Recortamos p_raw a [10⁻⁶,1−10⁻⁶] y calculamos el logit
z=log(p_raw/(1−p_raw)). Ajustamos una regresión logística de una sola entrada:
p_calibrada=1/(1+exp(−(a z+b))), usando las etiquetas de esa ventana. Aprendemos
a y b; no volvemos a entrenar los árboles de boosting ni usamos las 432 entradas
como atributos adicionales del calibrador. Conservamos el mismo recorte al
aplicarlo en validación posterior y test.

El 04 no calibra sus 16 ajustes internos: selecciona por AP raw. La configuración
LightGBM ganadora coincide con la ya entrenada en el 03; el 05 verifica y reutiliza
ese modelo y su propio Platt. No ensayamos isotónica como alternativa vigente.
En esta corrida la transformación preservó el orden y AP; sí cambió probabilidades,
etiquetas al corte 0,5 y decisiones BMR. El corte 0,5 sigue fijo: bajo probabilidades
verdaderas y costos simétricos de clasificación es la frontera de mínimo error;
con probabilidades estimadas no garantiza el mínimo empírico de cada muestra.

### Qué hacemos con el desbalance y con los costos

No aplicamos SMOTE, submuestreo ni pesos de clase a los cuatro clasificadores.
Aprenden con las frecuencias de su pasado autorizado; evaluamos AP, recall,
precision y FPR para que la clase mayoritaria no oculte el fraude. Esto no
significa que el desbalance esté “resuelto”. Platt se aprende con validación
temporal sin remuestrear y se comprueba en periodos posteriores.

Los costos se incorporan en BMR y en los árboles de acción. Para cada Ca, el árbol
aprende t_i=1[y_i A_i>Ca] y pondera por |y_i A_i−Ca| (normalizado por su media,
sin alterar los pesos relativos). Esos pesos representan diferencia de costos,
no un balanceo de clases. El árbol aproxima mediante impureza una decisión con
costos dependientes del ejemplo; no es otro clasificador probabilístico calibrado.

El costo realizado es Ca×(TP+FP)+suma de A_i en FN. La mejora frente a fija usa
el costo fijo como denominador; el ahorro trivial usa min(suma de todo fraude,
N×Ca). Un costo menor puede acompañarse de más legítimas intervenidas. No
atribuimos causalmente el ahorro final a Platt ni afirmamos haber seleccionado
por costo al mejor de los cuatro clasificadores.
